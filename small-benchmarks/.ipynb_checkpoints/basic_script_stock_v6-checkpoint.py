
# basic_script_stock_v6.py
# HellaSwag benchmark for GPT-2 family using CE over ending tokens only.
# Robust to truncation, avoids NaNs, and fixes possible ctx/ctx_a duplication.
#
# Usage:
#   python3 basic_script_stock_v6.py --model gpt2-medium --split validation --n 100 --max_length 768 --device auto --debug 5

import argparse, json, time
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--debug", type=int, default=0)
    return ap.parse_args()

def ensure_pad_token(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

def safe_prefix(ctx: str, ctx_a: str) -> str:
    ctx = (ctx or "").strip()
    ctx_a = (ctx_a or "").strip()
    if not ctx_a:
        return ctx
    # If ctx already ends with ctx_a (common duplication in some dumps), don't re-append
    if ctx.endswith(ctx_a):
        return ctx
    # If ctx_a already contains ctx (rare), just return ctx_a
    if ctx and ctx in ctx_a:
        return ctx_a
    return (ctx + " " + ctx_a).strip()

def build_choice_batch(tokenizer, prefix: str, endings: list, max_len: int):
    # Tokenize prefix and each ending; build labels that score only ending tokens
    pre_ids = tokenizer.encode(prefix, add_special_tokens=False)
    rows = []
    labels = []
    for end in endings:
        end_str = end if end.startswith((" ", "\n")) else " " + end
        end_ids = tokenizer.encode(end_str, add_special_tokens=False)

        ids = pre_ids + end_ids
        # left-truncate to keep tail
        if len(ids) > max_len:
            overflow = len(ids) - max_len
            ids = ids[overflow:]
            pre_len = max(0, len(pre_ids) - overflow)  # recompute prefix length post-trunc
        else:
            pre_len = len(pre_ids)

        # labels: -100 for pads & prefix positions, token id for ending positions
        lab = [-100] * pre_len + ids[pre_len:]

        # ---- CRITICAL GUARD ----
        # If every label is -100 (i.e., all ending tokens got chopped), force a position
        # that will SURVIVE the causal shift: choose the second-to-last if possible.
        if all(x == -100 for x in lab) and len(ids) >= 2:
            lab[-2] = ids[-2]         # survives shift_labels[:, 1:]
        elif all(x == -100 for x in lab) and len(ids) == 1:
            # worst-case fallback (almost never happens on HellaSwag)
            lab[-1] = ids[-1]

        rows.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))

    T = min(max(len(r) for r in rows), max_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokens = torch.full((4, T), pad_id, dtype=torch.long)
    labs   = torch.full((4, T), -100, dtype=torch.long)
    attn   = torch.zeros((4, T), dtype=torch.long)

    for i, (r, l) in enumerate(zip(rows, labels)):
        if len(r) > T:
            r = r[-T:]
            l = l[-T:]
        tokens[i, :len(r)] = r
        labs[i,   :len(l)] = l
        attn[i,   :len(r)] = 1

    return tokens, labs, attn

@torch.no_grad()
def score_choices(model, tokens, labs, attn, length_normalize=True):
    tokens = tokens.to(model.device)
    labs   = labs.to(model.device)
    attn   = attn.to(model.device)

    out = model(input_ids=tokens, attention_mask=attn, labels=labs)
    # Shift for CLM
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labs[:, 1:].contiguous()

    per_tok_loss = F.cross_entropy(
        shift_logits.transpose(1, 2),  # (B, V, T-1)
        shift_labels,                  # (B, T-1)
        reduction="none",
        ignore_index=-100,
    )
    valid = (shift_labels != -100)
    tok_counts = valid.sum(dim=1).clamp_min(1)
    nll = (per_tok_loss * valid).sum(dim=1)
    scores = -nll
    if length_normalize:
        scores = scores / tok_counts

    # NaN guard
    scores = torch.where(torch.isnan(scores), torch.full_like(scores, -1e30), scores)
    return scores.detach().cpu()

def main():
    args = parse_args()
    use_cuda = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

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
        prefix = safe_prefix(ex.get("ctx", ""), ex.get("ctx_a", ""))
        endings = list(ex["endings"])
        label = int(ex["label"])

        tokens, labs, attn = build_choice_batch(tok, prefix, endings, args.max_length)
        scores = score_choices(model, tokens, labs, attn, length_normalize=True)

        pred = int(torch.argmax(scores).item())
        correct += int(pred == label)

        if args.debug and i < args.debug:
            dbg = {
                "i": i, "label": label, "pred": pred,
                "scores": [float(s) for s in scores.tolist()],
                "prefix_head": prefix[:70].replace("\n", " "),
                "gold_head": endings[label][:60].replace("\n", " "),
            }
            print("[DEBUG]", json.dumps(dbg))

        n_done = i + 1
        if n_done % 10 == 0 or n_done == args.n:
            print(f"{n_done} acc_norm: {correct}/{n_done}={correct/n_done:.4f}")

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
    main()
