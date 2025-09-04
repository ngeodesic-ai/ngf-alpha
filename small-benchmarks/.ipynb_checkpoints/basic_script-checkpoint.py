# basic_script_grok_v3.py
# Minimal HellaSwag benchmark for GPT-2 family (e.g., gpt2-medium).
# Scores P(ending | context) only, with length-normalized log-likelihood.
#
# Usage:
#   python3 basic_script_grok_v3.py --model gpt2-medium --split validation --n 100 --device auto --debug 5

import argparse, json, time
from typing import Tuple
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=100, help="How many examples to evaluate")
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    ap.add_argument("--max_length", type=int, default=512, help="truncate from the left to keep endings")
    ap.add_argument("--debug", type=int, default=0, help="print first K per-item decisions")
    return ap.parse_args()

def ensure_pad_token(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

def build_four_choices(tokenizer, ctx_text, endings, max_len):
    # tokenize ctx once
    ctx_ids = tokenizer.encode(ctx_text.strip(), add_special_tokens=False)

    rows, emasks = [], []
    for end in endings:
        if not end.startswith((" ", "\n")):
            end = " " + end
        end_ids = tokenizer.encode(end, add_special_tokens=False)

        row = torch.tensor(ctx_ids + end_ids, dtype=torch.long)
        em  = torch.tensor([0]*len(ctx_ids) + [1]*len(end_ids), dtype=torch.long)
        rows.append(row); emasks.append(em)

    # left-truncate to keep endings
    T = min(max(len(r) for r in rows), max_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    tokens   = torch.full((4, T), pad_id, dtype=torch.long)
    end_mask = torch.zeros((4, T), dtype=torch.long)
    attn     = torch.zeros((4, T), dtype=torch.long)

    for i, (row, em) in enumerate(zip(rows, emasks)):
        if len(row) > T:
            row = row[-T:]
            em  = em[-T:]
        # ensure ≥1 ending token survives
        if em.sum().item() == 0 and len(row) > 0:
            em[-1] = 1
        tokens[i, :len(row)]   = row
        end_mask[i, :len(em)]  = em
        attn[i, :len(row)]     = 1

    return tokens, end_mask, attn

@torch.no_grad()
def score_choices(model, tokenizer, tokens, end_mask, attn, length_normalize=True):
    import torch.nn.functional as F
    tokens   = tokens.to(model.device)
    end_mask = end_mask.to(model.device)
    attn     = attn.to(model.device)

    labels = tokens.clone()
    labels[attn == 0]     = -100  # ignore pads
    labels[end_mask == 0] = -100  # ignore context

    out = model(input_ids=tokens, attention_mask=attn, labels=labels)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_tok_loss = F.cross_entropy(
        shift_logits.transpose(1, 2),  # (4, V, T-1)
        shift_labels,                  # (4, T-1)
        reduction="none",
        ignore_index=-100,
    )
    valid = (shift_labels != -100)
    tok_counts = valid.sum(dim=1).clamp_min(1)
    nll = (per_tok_loss * valid).sum(dim=1)

    scores = -nll
    if length_normalize:
        scores = scores / tok_counts

    # NaN guard (happens if a row still had no valid tokens somehow)
    scores = torch.where(torch.isnan(scores), torch.full_like(scores, -1e30), scores)
    return scores.cpu()

def main():
    args = parse_args()

    use_cuda = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    # lock loss type (optional; silences the warning)
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.loss_type = "ForCausalLMLoss"

    tok = AutoTokenizer.from_pretrained(args.model)
    ensure_pad_token(tok)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=cfg).to(device)
    model.eval()

    ds = load_dataset("hellaswag", split=args.split)
    if args.n and args.n < len(ds):
        ds = ds.select(range(args.n))

    correct = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        # Standard HellaSwag prefix
        ctx = (ex["ctx"] + " " + ex["ctx_a"]).strip()
        endings = list(ex["endings"])     # preserve order
        label = int(ex["label"])          # 0..3

        tokens, end_mask, attn = build_four_choices(tok, ctx, endings, args.max_length)
        scores = score_choices(model, tok, tokens, end_mask, attn, length_normalize=True)

        pred = int(torch.argmax(scores).item())
        correct += int(pred == label)

        # lightweight progress like your v2 script
        n_done = i + 1
        if n_done % 10 == 0 or n_done == args.n:
            print(f"{n_done} acc_norm: {correct}/{n_done}={correct/n_done:.4f}")

        if args.debug and i < args.debug:
            dbg = {
                "i": i,
                "label": label,
                "pred": pred,
                "scores": [float(s) for s in scores.tolist()],
                "ending_gold_head": endings[label][:60].replace("\n", " "),
            }
            print("[DEBUG]", json.dumps(dbg))

    total = min(args.n, len(ds))
    acc = correct / total if total else 0.0
    elapsed = time.time() - t0
    print(json.dumps({
        "model": args.model,
        "split": args.split,
        "n": total,
        "accuracy": acc,
        "correct": correct,
        "elapsed_sec": elapsed,
        "device": str(device),
    }, indent=2))

if __name__ == "__main__":
    from datasets import load_dataset  # placed here for a cleaner top section
    main()
