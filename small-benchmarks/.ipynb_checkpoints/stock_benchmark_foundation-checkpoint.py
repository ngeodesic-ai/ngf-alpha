#!/usr/bin/env python3
"""
stock_benchmark_foundation.py — robust single-pass STOCK benchmark (JSON outputs)
---------------------------------------------------------------------------------
Purpose:
  Establish a reliable STOCK baseline (30–40% target on gpt2-medium) with one run.
  Generates a simple integer-only task set, runs the unified runner once, then scores.

Key features:
  • Single-pass by default (no calibration loops) with a recommended easy mix.
  • Tiny smoke test (n=6) before the full run to catch runner/format issues early.
  • Auto-detects the prediction field among: generation, prediction, output, text, decoded, answer, completion.
  • Resilient scoring: falls back to order-based alignment if IDs are missing/misaligned.
  • Emits both JSONL (streamable) and JSON array (clickable), plus per-item CSV and metrics JSON/CSV.

Example (CUDA + gpt2-medium):
  python3 /mnt/data/stock_benchmark_foundation.py \
    --runner /mnt/data/text_arc_unified_base.py --device cuda --model gpt2-medium \
    --n 120 --k_shot 8 \
    --outdir ./layer9_wiring_plan/stock_foundation_run

Example (CPU):
  python3 /mnt/data/stock_benchmark_foundation.py \
    --runner /mnt/data/text_arc_unified_base.py --device cpu --model gpt2-medium \
    --n 120 --k_shot 8 \
    --outdir ./layer9_wiring_plan/stock_foundation_cpu

python3 stock_benchmark_foundation.py \
  --runnertext_arc_unified_base.py \
  --device cuda --model gpt2-medium --tap -9 \
  --n 120 --k_shot 8 \
  --outdir benchmark_results/stock_foundation_run
    
"""

import os, re, csv, json, time, argparse, random, subprocess, sys
from pathlib import Path
from collections import Counter

DEFAULT_RUNNER = "text_arc_unified_base.py"

# ---------------- Few-shot exemplars ----------------
BASE_FEWSHOT = [
    ("Return only the integer 7.", "7"),
    ("Which is larger, 4 or 9? Return only the integer.", "9"),
    ("Return the smallest integer: 5, 2, 8.", "2"),
    ("In 'IDs: 19, 7, 42', return the last integer.", "42"),
    ("Compute 4+7. Return only the integer.", "11"),
]

def make_fewshot(k_shot: int):
    few = []; i = 0
    while len(few) < k_shot:
        q,a = BASE_FEWSHOT[i % len(BASE_FEWSHOT)]
        few.append((q,a)); i += 1
    return few

# ---------------- Utils ----------------
_int_pat = re.compile(r"[+-]?\d+")
def numbers_only(s: str) -> str:
    if not s: return ""
    m = _int_pat.search(s)
    return m.group(0) if m else ""

def prompt_wrap(q: str, fewshot_pairs):
    few = "\n".join([f"Q: {q0}\nA: {a0}" for q0,a0 in fewshot_pairs])
    return f"{few}\n\nQ: {q}\nA:"

# ---------------- Task families ----------------
def gen_echo_integer(rng, few):
    x = rng.randint(2, 99)
    q = f"Return only the integer {x}."
    return prompt_wrap(q, few), str(x), "echo_integer"

def gen_extract_last(rng, few):
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 99)
    q = f"In 'IDs: {a}, {b}, {c}', return the last integer."
    return prompt_wrap(q, few), str(c), "extract_last"

def gen_pick_larger(rng, few, lo=2, hi=99):
    a = rng.randint(lo, hi); b = rng.randint(lo, hi)
    while b == a: b = rng.randint(lo, hi)
    larger = a if a > b else b
    q = f"Which is larger, {a} or {b}? Return only the integer."
    return prompt_wrap(q, few), str(larger), "pick_larger"

def gen_min_of_three(rng, few, lo=2, hi=99):
    a, b, c = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)
    smallest = min(a,b,c)
    q = f"Return the smallest integer: {a}, {b}, {c}."
    return prompt_wrap(q, few), str(smallest), "min_of_three"

FAMILIES = [gen_echo_integer, gen_extract_last, gen_pick_larger, gen_min_of_three]

def sample_tasks(rng, n, few, mix):
    # mix is dict of family_name -> weight
    fam_map = {
        "echo_integer": gen_echo_integer,
        "extract_last": gen_extract_last,
        "pick_larger": gen_pick_larger,
        "min_of_three": gen_min_of_three,
    }
    fams = list(mix.items()); weights = [w for _,w in fams]
    tot = sum(weights) or 1.0; weights = [w/tot for w in weights]
    tasks = []
    for i in range(1, n+1):
        fam_name = random.choices([k for k,_ in fams], weights=weights, k=1)[0]
        fn = fam_map[fam_name]
        p, t, fam = fn(rng, few); tasks.append((i, p, t, fam))
    return tasks

def write_prompts_truths(outdir: Path, tasks, stem: str):
    prompts_path = str(outdir / f"{stem}_prompts.txt")
    truths_csv   = str(outdir / f"{stem}_truths.csv")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for _, p, _, _ in tasks: f.write(p.strip()+"\n")
    with open(truths_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id","truth","family"])
        for i, _, t, fam in tasks: w.writerow([i, t, fam])
    return prompts_path, truths_csv

# ---------------- Runner ----------------
def run_stock(runner, prompts_path, out_jsonl, tap, device=None, model=None, extra_args=None, max_new_tokens="8"):
    cmd = [sys.executable, runner,
           "--gen_mode", "stock",
           "--prompts", prompts_path,
           "--out", out_jsonl,
           "--tap", str(tap),
           "--temperature", "0.0",
           "--top_p", "1.0",
           "--max_new_tokens", str(max_new_tokens)]
    if device: cmd += ["--device", device]
    if model:  cmd += ["--model", model]
    if extra_args: cmd += extra_args
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")
    return True

# ---------------- IO helpers ----------------
def read_jsonl(path: str):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: arr.append(json.loads(ln))
            except Exception: pass
    return arr

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------------- Auto field detection ----------------
FIELD_CANDIDATES = ["generation","prediction","output","text","decoded","answer","completion"]

def pick_pred_field(items):
    # Returns the first field from FIELD_CANDIDATES that appears in >=50% items and is stringy
    if not items: return None
    counts = Counter()
    for it in items:
        for k in FIELD_CANDIDATES:
            if k in it and isinstance(it[k], (str, int, float)):
                counts[k] += 1
    if not counts: return None
    field, c = max(counts.items(), key=lambda kv: kv[1])
    if c >= max(1, int(0.5*len(items))):
        return field
    # fallback: pick the most frequent anyway
    return field

def extract_id(it, default_id=None):
    for k in ["id","idx","index"]:
        if k in it:
            try: return int(it[k])
            except Exception: pass
    return default_id

# ---------------- Scoring with alignment fallback ----------------
def score(items, truths_csv, pred_field=None):
    # truths
    truths = {}; fams = {}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try: i = int(row["id"])
            except: continue
            truths[i] = str(row["truth"]); fams[i] = row.get("family","")

    n_items = len(items)
    if not pred_field:
        pred_field = pick_pred_field(items) or "generation"

    # Build maps by ID if possible
    pred_by_id = {}
    order_preds = []
    id_errors = 0
    for idx, it in enumerate(items, start=1):
        pid = extract_id(it, default_id=None)
        pred_raw = it.get(pred_field, "")
        pred_str = str(pred_raw)
        if pid is None:
            order_preds.append((idx, pred_str))
        else:
            pred_by_id[pid] = pred_str

    # Check alignment quality
    overlap = len(set(pred_by_id.keys()) & set(truths.keys()))
    align_ratio = overlap / max(1, len(truths))

    use_order_align = False
    if align_ratio < 0.6:  # too little overlap → fallback to order
        use_order_align = True

    # Build aligned list
    aligned = []
    if not use_order_align:
        for tid in sorted(truths.keys()):
            aligned.append((tid, pred_by_id.get(tid, "")))
    else:
        # order-based: pair sequentially by sorted truth ids
        tid_list = sorted(truths.keys())
        for j, tid in enumerate(tid_list, start=1):
            pred = ""
            if j-1 < len(items):
                it = items[j-1]
                pred = str(it.get(pred_field, ""))
            aligned.append((tid, pred))

    # Score
    tp=fp=fn=0; n_scored=0
    fam_counts=Counter(); fam_tp=Counter()
    per_item=[]
    for tid, pred in aligned:
        truth = truths.get(tid, None)
        fam = fams.get(tid, "")
        if truth is None: continue
        n_scored += 1; 
        if fam: fam_counts[fam]+=1
        pred_n = numbers_only(pred); truth_n = numbers_only(truth)
        if pred_n == "":
            fn += 1; ok=0
        elif pred_n == truth_n and truth_n != "":
            tp += 1; ok=1; fam_tp[fam]+=1
        else:
            fp += 1; ok=0
        per_item.append({"id": tid, "prediction": pred_n or pred.strip(), "truth": truth_n, "family": fam, "correct": ok})

    n = max(1, n_scored)
    accuracy = tp/n; precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision>0 and recall>0) else 0.0
    fam_acc = {k: round(fam_tp[k]/v, 3) for k,v in fam_counts.items() if v>0}

    metrics = {
        "n_scored": n_scored, "tp": tp, "fp": fp, "fn": fn,
        "accuracy": round(accuracy,6), "precision": round(precision,6),
        "recall": round(recall,6), "f1": round(f1,6),
        "hallucination_rate": round(fp/n,6), "omission_rate": round(fn/n,6),
        "family_accuracy": fam_acc, "match_mode": "numbers_only",
        "pred_field_used": pred_field, "alignment": "by_id" if not use_order_align else "by_order",
    }
    return per_item, metrics

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Robust single-pass STOCK baseline (JSON outputs)")
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu")
    ap.add_argument("--model", type=str, default="gpt2-medium")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_shot", type=int, default=8)
    ap.add_argument("--fixed_mix", type=str, default='{"echo_integer":0.60,"extract_last":0.20,"pick_larger":0.20}')
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="stock_foundation_run")
    args, extra = ap.parse_known_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    few = make_fewshot(max(1, min(12, args.k_shot)))
    try:
        mix = json.loads(args.fixed_mix)
    except Exception as e:
        raise SystemExit(f"--fixed_mix must be JSON: {e}")

    # 1) Tiny smoke test (n=6) — catches format/device issues early
    smoke_tasks = sample_tasks(rng, 6, few, mix)
    smoke_prompts, smoke_truths = write_prompts_truths(outdir, smoke_tasks, stem="smoke")
    smoke_jsonl = str(outdir / "generations_smoke.jsonl")
    run_stock(args.runner, smoke_prompts, smoke_jsonl, args.tap, device=args.device, model=args.model,
              max_new_tokens=str(args.max_new_tokens))
    smoke_items = read_jsonl(smoke_jsonl)
    smoke_field = pick_pred_field(smoke_items) or "generation"
    _, smoke_metrics = score(smoke_items, smoke_truths, pred_field=smoke_field)

    # If smoke yields zero scored or all empty, bail early with a hint
    if smoke_metrics["n_scored"] == 0 or (smoke_metrics["tp"]==0 and smoke_metrics["fp"]==0 and smoke_metrics["fn"]>0):
        hint = ("No usable predictions detected. Ensure your runner writes a string field like "
                f"'generation' (detected={smoke_field}) and includes digit-only answers. "
                "Also verify --device and --model are honored.")
        print(json.dumps({"SMOKE_SUMMARY": smoke_metrics, "hint": hint}, indent=2))
        # Still proceed to full run, but warn
        print("[WARN] Smoke test suggests empty/misaligned outputs; proceeding anyway.")

    # 2) Full run
    tasks = sample_tasks(rng, args.n, few, mix)
    prompts_path, truths_csv = write_prompts_truths(outdir, tasks, stem="final")
    gen_jsonl = str(outdir / "generations_stock.jsonl")
    run_stock(args.runner, prompts_path, gen_jsonl, args.tap, device=args.device, model=args.model,
              max_new_tokens=str(args.max_new_tokens))
    items = read_jsonl(gen_jsonl)

    # 3) Score (auto-detected pred field + alignment fallback)
    pred_field = pick_pred_field(items) or "generation"
    per_item, metrics = score(items, truths_csv, pred_field=pred_field)

    # 4) Save artifacts
    items_csv = str(outdir / "items_stock.csv")
    with open(items_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prediction","truth","family","correct"])
        w.writeheader()
        for r in per_item: w.writerow(r)
    print(f"[CSV]  {items_csv}")

    metrics_json = str(outdir / "metrics_stock.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[JSON] {metrics_json}")

    metrics_csv = str(outdir / "metrics_stock.csv")
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode","pred_field_used","alignment"])
        w.writeheader()
        row = {k: metrics.get(k, "") for k in ["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode","pred_field_used","alignment"]}
        w.writerow(row)
    print(f"[CSV]  {metrics_csv}")

    # Clickable JSON array of generations
    out_json = str(outdir / "generations_stock.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"[JSON] {out_json}")

    print(json.dumps({"SUMMARY": {"StockFoundation": metrics}}, indent=2))
    print(f"[DONE] elapsed_sec={time.time()-t0:.3f}")

if __name__ == "__main__":
    main()
