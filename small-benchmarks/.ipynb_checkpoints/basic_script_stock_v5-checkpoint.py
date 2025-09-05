
# basic_script_stock_v5.py
# Robust HellaSwag benchmark for GPT-2 family (e.g., gpt2-medium).
# Scores only P(ending | context) with length-normalized log-likelihood.
# Avoids NaNs via explicit log-softmax + gather and a shifted end-mask.
#
# Usage:
#   python3 basic_script_stock_v5.py --model gpt2-medium --split validation --n 100 --device auto --debug 5 --max_length 768
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

def build_four_choices(tokenizer, ctx_text: str, endings: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Tokenize context once
    ctx_ids = tokenizer.encode(ctx_text.strip(), add_special_tokens=False)
    rows, end_masks = [], []
    for end in endings:
        if not end.startswith((" ", "\n")):
            end = " " + end
        end_ids = tokenizer.encode(end, add_special_tokens=False)
        row = torch.tensor(ctx_ids + end_ids, dtype=torch.long)
        em  = torch.tensor([0]*len(ctx_ids) + [1]*len(end_ids), dtype=torch.long)
        rows.append(row); end_masks.append(em)

    # Left-truncate to keep endings
    T = min(max(len(r) for r in rows), max_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokens   = torch.full((4, T), pad_id, dtype=torch.long)
    end_mask = torch.zeros((4, T), dtype=torch.long)
    attn     = torch.zeros((4, T), dtype=torch.long)

    for i, (row, em) in enumerate(zip(rows, end_masks)):
        if len(row) > T:
            row = row[-T:]; em = em[-T:]
        # Ensure >= 2 tokens exist so at least 1 predictive position is in ending after shift
        # If all ending tokens were chopped, force the last two positions to be ending (when possible)
        if em.sum().item() == 0 and len(row) >= 2:
            em[-1] = 1; em[-2] = 1
        elif em.sum().item() == 0 and len(row) == 1:
            em[-1] = 1  # as best effort

        tokens[i, :len(row)]  = row
        end_mask[i, :len(em)] = em
        attn[i, :len(row)]    = 1

    return tokens, end_mask, attn

@torch.no_grad()
def score_choices(model, tokenizer, tokens, end_mask, attn, length_normalize=True):
    """Compute sum/mean log P(ending tokens | prefix) per choice using explicit log-softmax.
    We score positions t in [0..T-2] whose *next* token (labels[:, t+1]) is an ending token.
    """
    tokens   = tokens.to(model.device)
    end_mask = end_mask.to(model.device)
    attn     = attn.to(model.device)

    out = model(input_ids=tokens, attention_mask=attn, use_cache=False)
    logits = out.logits  # (4, T, V)

    # Shift for CLM: positions 0..T-2 predict labels at 1..T-1
    shift_logits = logits[:, :-1, :]     # (4, T-1, V)
    shift_tokens = tokens[:, 1:]         # (4, T-1)
    shift_attn   = attn[:, 1:]           # (4, T-1)

    # Build score mask: we score positions where the *label* (next token) is part of the ending
    shift_end_mask = end_mask[:, 1:]     # (4, T-1)
    score_mask = (shift_end_mask == 1) & (shift_attn == 1)

    # Guard: if a row has no True in score_mask, force the last valid position to be scored
    for i in range(score_mask.shape[0]):
        if score_mask[i].sum().item() == 0:
            valid_positions = torch.nonzero(shift_attn[i] == 1, as_tuple=False).flatten()
            if len(valid_positions) > 0:
                score_mask[i, valid_positions[-1]] = True

    logprobs = torch.log_softmax(shift_logits, dim=-1)  # (4, T-1, V)
    gathered = torch.gather(logprobs, dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)  # (4, T-1)

    # Zero out unscored positions
    gathered = gathered * score_mask

    # Sum / mean over scored positions
    token_counts = score_mask.sum(dim=1).clamp_min(1)  # (4,)
    seq_scores = gathered.sum(dim=1)                   # (4,)
    if length_normalize:
        seq_scores = seq_scores / token_counts

    # Final NaN guard
    seq_scores = torch.where(torch.isnan(seq_scores), torch.full_like(seq_scores, -1e30), seq_scores)
    return seq_scores.detach().cpu()

def main():
    args = parse_args()
    use_cuda = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    cfg = AutoConfig.from_pretrained(args.model)
    cfg.loss_type = "ForCausalLMLoss"  # quiet the warning

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
        ctx = (ex["ctx"] + " " + ex["ctx_a"]).strip()
        endings = list(ex["endings"])
        label = int(ex["label"])

        tokens, end_mask, attn = build_four_choices(tok, ctx, endings, args.max_length)
        scores = score_choices(model, tok, tokens, end_mask, attn, length_normalize=True)

        pred = int(torch.argmax(scores).item())
        correct += int(pred == label)

        if args.debug and i < args.debug:
            dbg = {
                "i": i, "label": label, "pred": pred,
                "scores": [float(s) for s in scores.tolist()],
                "ctx_head": ctx[:60].replace("\n", " "),
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
