
# basic_script_ngf_v8.py
# HellaSwag benchmark for GPT-2 family with optional NGF tap-9 warp.
# - Uses correct P(ending | context) scoring via CE over ending tokens only.
# - Robust truncation + attention + duplication-safe prefix.
# - Adds --mode {stock,ngf}. In ngf mode, attempts to attach NGF hooks to the model.
#
# Usage examples:
#   python3 basic_script_ngf_v8.py --mode stock --model gpt2-medium --split validation --n 200 --max_length 768 --device auto
#   python3 basic_script_ngf_v8.py --mode ngf   --model gpt2-medium --split validation --n 200 --max_length 768 --device auto
#
# If NGF code is available in your PYTHONPATH (e.g., text_arc_unified_base.py with an attach function),
# this script will try to import it automatically. Otherwise it falls back to stock.
import argparse, json, time, importlib, sys
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import numpy as np  # ← NEW


"""

MODEL=gpt2              
SPLIT=validation
N=1000
MAXLEN=768
DEVICE=auto
OUTDIR=results/${MODEL}_n${N}
mkdir -p "$OUTDIR"

python3 ngf_benchmark.py \
  --mode stock \
  --model $MODEL --split $SPLIT --n $N --max_length $MAXLEN --device $DEVICE \
  --out_json  $OUTDIR/stock.json \
  --save_jsonl $OUTDIR/stock.jsonl

python3 ngf_benchmark.py \
  --mode ngf --ngf_import ngf_hooks:attach_ngf_hooks \
  --model $MODEL --split $SPLIT --n $N --max_length $MAXLEN --device $DEVICE \
  --tap -9 \
  --alpha0 0.05 --alpha_min 0.006 --trend_tau 0.32 --k_tr 12 --ema_center_beta 0.05 \
  --out_json  $OUTDIR/geo.json \
  --save_jsonl $OUTDIR/geo.jsonl

python3 ngf_benchmark.py \
  --mode ngf --ngf_import ngf_hooks:attach_ngf_hooks \
  --model $MODEL --split $SPLIT --n $N --max_length $MAXLEN --device $DEVICE \
  --tap -9 \
  --alpha0 0.05 --alpha_min 0.006 --trend_tau 0.32 --k_tr 12 --ema_center_beta 0.05 \
  --use_detect 1 --detect_width 20 --null_K 32 --null_q 0.92 --k_det 8 \
  --out_json  $OUTDIR/geo_detect.json \
  --save_jsonl $OUTDIR/geo_detect.jsonl

python3 ngf_benchmark.py \
  --mode ngf --ngf_import ngf_hooks:attach_ngf_hooks \
  --model $MODEL --split $SPLIT --n $N --max_length $MAXLEN --device $DEVICE \
  --tap -9 \
  --alpha0 0.05 --alpha_min 0.006 --trend_tau 0.32 --k_tr 12 --ema_center_beta 0.05 \
  --use_detect 1 --detect_width 20 --null_K 32 --null_q 0.92 --k_det 8 \
  --out_json  $OUTDIR/geo_detect_denoise.json \
  --save_jsonl $OUTDIR/geo_detect_denoise.jsonl


"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="stock", choices=["stock", "ngf"], help="stock baseline or NGF-warped model")
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--debug", type=int, default=0, help="print first K debug decisions")
    ap.add_argument("--debug_counts", action="store_true", help="print number of scored tokens per choice")
    # NGF baseline flags (v4b T4/L4 geo) — tweak if needed
    ap.add_argument("--tap", type=int, default=-9, help="NGF tap index (e.g., -9)")
    ap.add_argument("--alpha0", type=float, default=0.05)
    ap.add_argument("--alpha_min", type=float, default=0.006)
    ap.add_argument("--trend_tau", type=float, default=0.35)
    ap.add_argument("--k_tr", type=int, default=12)
    ap.add_argument("--use_detect", type=int, default=1)
    ap.add_argument("--detect_width", type=int, default=24)
    ap.add_argument("--detect_sigma", type=float, default=5.0)
    ap.add_argument("--null_K", type=int, default=32)
    ap.add_argument("--null_q", type=float, default=0.92)
    ap.add_argument("--k_det", type=int, default=7)
    ap.add_argument("--s_latch", type=float, default=0.30)
    ap.add_argument("--linger", type=int, default=2)
    ap.add_argument("--ema_center_beta", type=float, default=0.05)
    ap.add_argument("--gen_mode", type=str, default="geo")
    # Optional import hook path, e.g., text_arc_unified_base:attach_ngf_hooks
    ap.add_argument("--ngf_import", type=str, default="", help="MODULE:FUNC path to attach NGF hooks (overrides auto-detect)")
    ap.add_argument("--out_json", type=str, default="",
                help="Optional path to write the final results JSON")
    return ap.parse_args()

def ensure_pad_token(tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

def safe_prefix(ctx: str, ctx_a: str) -> str:
    ctx = (ctx or "").strip()
    ctx_a = (ctx_a or "").strip()
    if not ctx_a:
        return ctx
    if ctx.endswith(ctx_a):
        return ctx
    if ctx and ctx in ctx_a:
        return ctx_a
    return (ctx + " " + ctx_a).strip()

def build_choice_batch(tokenizer, prefix: str, endings: List[str], max_len: int):
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
            pre_len = max(0, len(pre_ids) - overflow)  # recompute prefix length
        else:
            pre_len = len(pre_ids)

        # labels: -100 for prefix positions, token id for ending positions
        lab = [-100] * pre_len + ids[pre_len:]

        # Guards: ensure ≥1 (preferably ≥2) scoreable positions survive the causal shift
        if all(x == -100 for x in lab):
            if len(ids) >= 3:
                lab[-3] = ids[-3]
                lab[-2] = ids[-2]
            elif len(ids) == 2:
                lab[-2] = ids[-2]
            elif len(ids) == 1:
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
def score_choices(model, tokens, labs, attn, length_normalize=True, return_counts=False):
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
    if return_counts:
        return scores.detach().cpu(), tok_counts.detach().cpu()
    return scores.detach().cpu()


# === Metrics helpers (MC classification on 4 choices) ===
def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _confusion_matrix(y_true, y_pred, C=4):
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def _prec_rec_f1_from_cm(cm):
    eps = 1e-12
    C = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    f1   = (2*prec*rec) / (prec + rec + eps)
    macro = dict(
        precision=float(np.mean(prec)),
        recall=float(np.mean(rec)),
        f1=float(np.mean(f1)),
    )
    micro_tp = tp.sum()
    micro = dict(
        precision=float(micro_tp / (micro_tp + fp.sum() + eps)),
        recall=float(micro_tp / (micro_tp + fn.sum() + eps)),
        f1=float((2*micro_tp) / (2*micro_tp + fp.sum() + fn.sum() + eps)),
    )
    per_class = [dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i])) for i in range(C)]
    return macro, micro, per_class

def _expected_calibration_error(probs, y_true, n_bins=10):
    # probs: [N, C], y_true: [N]
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)



def try_attach_ngf(model, tokenizer, device, args) -> Optional[str]:
    """Attempt to attach NGF hooks to `model`. Returns a status string or None on stock."""
    if args.mode != "ngf":
        return None
    cfg = {
        "tap": args.tap,
        "alpha0": args.alpha0,
        "alpha_min": args.alpha_min,
        "trend_tau": args.trend_tau,
        "k_tr": args.k_tr,
        "use_detect": args.use_detect,
        "detect_width": args.detect_width,
        "detect_sigma": args.detect_sigma,
        "null_K": args.null_K,
        "null_q": args.null_q,
        "k_det": args.k_det,
        "s_latch": args.s_latch,
        "linger": args.linger,
        "ema_center_beta": args.ema_center_beta,
        "gen_mode": args.gen_mode,
    }
    # 1) explicit path via --ngf_import MODULE:FUNC
    if args.ngf_import:
        try:
            mod_name, func_name = args.ngf_import.split(":", 1)
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, func_name)
            fn(model=model, tokenizer=tokenizer, device=device, **cfg)
            return f"NGF attached via {args.ngf_import} with cfg={cfg}"
        except Exception as e:
            return f"NGF attach failed via {args.ngf_import}: {e}"
    # 2) common auto-detects
    candidates = [
        ("text_arc_unified_base", "attach_ngf_hooks"),
        ("ngf_hooks", "attach_ngf_hooks"),
        ("ngf_alpha.llm_hooks", "attach_ngf_hooks"),
    ]
    for mod_name, func_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, func_name):
                getattr(mod, func_name)(model=model, tokenizer=tokenizer, device=device, **cfg)
                return f"NGF attached via {mod_name}:{func_name} with cfg={cfg}"
        except Exception as e:
            # keep trying others
            continue
    return "NGF mode requested, but no attach function was found (stock behavior). Provide --ngf_import MODULE:FUNC."

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

    ngf_status = try_attach_ngf(model, tok, device, args)
    if ngf_status:
        print(f"[NGF] {ngf_status}")

    ds = load_dataset("hellaswag", split=args.split)
    if args.n and args.n < len(ds):
        ds = ds.select(range(args.n))

    correct = 0
    t0 = time.time()

    # NEW: collect per-item data for richer metrics
    all_scores = []  # list of [4] arrays (length-normalized log-prob scores)
    all_labels = []  # list of ints

    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        prefix = safe_prefix(ex.get("ctx", ""), ex.get("ctx_a", ""))
        endings = list(ex["endings"])
        label = int(ex["label"])

        tokens, labs, attn = build_choice_batch(tok, prefix, endings, args.max_length)
        scores, counts = score_choices(model, tokens, labs, attn, length_normalize=True, return_counts=True)

        all_scores.append(np.asarray(scores.tolist(), dtype=np.float64))  # shape (4,)
        all_labels.append(label)
        
        pred = int(torch.argmax(scores).item())
        correct += int(pred == label)

        if args.debug and i < args.debug:
            dbg = {
                "i": i, "label": label, "pred": pred,
                "scores": [float(s) for s in scores.tolist()],
                "counts": [int(c) for c in counts.tolist()],
                "prefix_head": prefix[:70].replace("\n", " "),
                "gold_head": endings[label][:60].replace("\n", " "),
            }
            print("[DEBUG]", json.dumps(dbg))

        n_done = i + 1
        if n_done % 10 == 0 or n_done == args.n:
            print(f"{n_done} acc_norm: {correct}/{n_done}={correct/n_done:.4f}")

    # after the for-loop ends:
    total = min(args.n, len(ds))
    acc = correct / total if total else 0.0
    elapsed = time.time() - t0
    
    # === metrics from collected scores/labels (unchanged logic) ===
    metrics = {}
    if total > 0 and len(all_scores) == total:
        scores_np = np.stack(all_scores, axis=0)                 # [N, 4]
        probs = _softmax(scores_np, axis=1)
        y_true = np.asarray(all_labels, dtype=np.int64)
        pred = probs.argmax(axis=1)
    
        cm = _confusion_matrix(y_true, pred, C=scores_np.shape[1])
        macro, micro, per_class = _prec_rec_f1_from_cm(cm)
    
        ece_10 = _expected_calibration_error(probs, y_true, n_bins=10)
        onehot = np.eye(probs.shape[1])[y_true]
        brier  = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
        sorted_p = np.sort(probs, axis=1)
        margin = float(np.mean(sorted_p[:, -1] - sorted_p[:, -2]))
        entropy = float(np.mean(-np.sum(np.where(probs > 0, probs*np.log(probs + 1e-12), 0.0), axis=1)))
    
        maxp = probs.max(axis=1)
        wrong = (pred != y_true)
        overconf_90 = float(np.mean(wrong & (maxp >= 0.90)))
        overconf_70 = float(np.mean(wrong & (maxp >= 0.70)))
    
        top2 = np.argsort(-probs, axis=1)[:, :2]
        top2_acc = float(np.mean((top2[:, 0] == y_true) | (top2[:, 1] == y_true)))
    
        metrics = {
            "accuracy_top1": float(acc),
            "accuracy_top2": top2_acc,
            "macro": macro,
            "micro": micro,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "calibration": {
                "ece_10": ece_10,
                "brier": brier,
                "mean_margin": margin,
                "entropy": entropy,
            },
            "overconfidence_rate": {
                "wrong_p>=0.90": overconf_90,
                "wrong_p>=0.70": overconf_70,
            },
        }
    
    # assemble result once
    result = {
        "mode": args.mode,
        "model": args.model,
        "split": args.split,
        "n": total,
        "accuracy": acc,
        "correct": correct,
        "elapsed_sec": elapsed,
        "device": str(device),
        "ngf_status": ngf_status or "stock",
        "metrics": metrics,
    }
    
    # write to --out_json if provided
    if args.out_json:
        import os, io, json
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with io.open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[WRITE] Results → {args.out_json}")
    
    # still print to stdout
    #print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
