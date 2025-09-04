#!/usr/bin/env python3
"""
arc_easy_benchmark_stock_adaptive.py — STOCK-only adaptive "easy tasks" benchmark
----------------------------------------------------------------------------------
Goal: build a STOCK baseline around ~30–40% accuracy so we have more positives.
Strategy:
  • Provide several easy-ish task families that return a single integer:
      - echo_integer:      "Q: Return only the integer 7. A:"
      - pick_larger:       "Q: Which is larger, 17 or 23? Return only the integer. A:"
      - min_of_three:      "Q: Return the smallest integer: 5, 2, 8. A:"
      - extract_last:      "Q: In 'IDs: 19, 7, 42', return the last integer. A:"
      - add_single_digit:  "Q: Compute 4+7. Return only the integer. A:"
      - add_two_digit:     "Q: Compute 13+29. Return only the integer. A:"   (harder)
  • Use QA + inline few-shot exemplars to reduce format drift.
  • Adaptive calibration: run a small batch (M) and adjust the task mix until
    stock accuracy is in [target_min, target_max]. Then generate the full N.
  • Greedy-ish decode and numbers_only scoring.

Example:
  python3 arc_easy_benchmark_stock_adaptive.py \
    --n 120 --calib_n 24 --tap -9 --device cuda \
    --runner text_arc_unified_base.py \
    --outdir benchmark_results/llm_benchmark_stock_easy_adaptive

  python3 arc_matrix_benchmark_stock.py \
    --n 60 --tap -9 --device cuda \
    --runner text_arc_unified_base.py \
    --outdir benchmark_results/llm_benchmark_stock_matrix
"""

import os, re, csv, json, time, argparse, random, subprocess, sys
from pathlib import Path
from collections import Counter

DEFAULT_RUNNER = "/mnt/data/text_arc_unified_base.py"

# ---------------- Few-shot exemplars ----------------
FEWSHOT = [
    ("Return only the integer 7.", "7"),
    ("Which is larger, 4 or 9? Return only the integer.", "9"),
    ("Return the smallest integer: 5, 2, 8.", "2"),
    ("In 'IDs: 19, 7, 42', return the last integer.", "42"),
    ("Compute 4+7. Return only the integer.", "11"),
]

# ---------------- Utilities ----------------
def ensure_dir(p):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def alnum_lower(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", " ", s or "")
    return " ".join(s.lower().strip().split())

_int_pat = re.compile(r"[+-]?\d+")
def numbers_only(s: str) -> str:
    if not s:
        return ""
    m = _int_pat.search(s)
    return m.group(0) if m else ""

def normalize(s: str, mode: str) -> str:
    if mode == "numbers_only":
        return numbers_only(s)
    return (s or "").strip()

def read_generations_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                items.append(json.loads(ln))
            except Exception:
                pass
    return items

# ---------------- Task families ----------------
def prompt_wrap(q: str) -> str:
    few = "\n".join([f"Q: {q0}\nA: {a0}" for q0,a0 in FEWSHOT])
    return f"{few}\n\nQ: {q}\nA:"

def gen_echo_integer(rng):
    x = rng.randint(2, 99)
    q = f"Return only the integer {x}."
    return prompt_wrap(q), str(x)

def gen_pick_larger(rng, lo=2, hi=99):
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    while b == a:
        b = rng.randint(lo, hi)
    larger = a if a > b else b
    q = f"Which is larger, {a} or {b}? Return only the integer."
    return prompt_wrap(q), str(larger)

def gen_min_of_three(rng, lo=2, hi=99):
    a, b, c = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)
    smallest = min(a,b,c)
    q = f"Return the smallest integer: {a}, {b}, {c}."
    return prompt_wrap(q), str(smallest)

def gen_extract_last(rng):
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 99)
    q = f"In 'IDs: {a}, {b}, {c}', return the last integer."
    return prompt_wrap(q), str(c)

def gen_add_single_digit(rng):
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    q = f"Compute {a}+{b}. Return only the integer."
    return prompt_wrap(q), str(a+b)

def gen_add_two_digit(rng):
    a, b = rng.randint(10, 29), rng.randint(10, 29)
    q = f"Compute {a}+{b}. Return only the integer."
    return prompt_wrap(q), str(a+b)

FAMILIES = {
    "echo_integer": gen_echo_integer,
    "pick_larger": gen_pick_larger,
    "min_of_three": gen_min_of_three,
    "extract_last": gen_extract_last,
    "add_single_digit": gen_add_single_digit,
    "add_two_digit": gen_add_two_digit,
}

# ---------------- Sampling mixes ----------------
def sample_tasks(rng, n, mix):
    fams = list(mix.items())
    weights = [w for _,w in fams]
    tot = sum(weights) or 1.0
    weights = [w/tot for w in weights]
    tasks = []
    for i in range(1, n+1):
        fam = rng.choices(fams, weights=weights, k=1)[0][0]
        p, t = FAMILIES[fam](rng)
        tasks.append((i, p, t, fam))
    return tasks

def write_prompts_truths(outdir: Path, tasks, stem: str):
    prompts_path = str(outdir / f"{stem}_prompts.txt")
    truths_csv   = str(outdir / f"{stem}_truths.csv")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for _, p, _, _ in tasks:
            f.write(p.strip()+"\n")
    with open(truths_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","truth","family"])
        for i, _, t, fam in tasks:
            w.writerow([i, t, fam])
    return prompts_path, truths_csv

# ---------------- Scoring ----------------
def score(items, truths_csv, match_mode="numbers_only"):
    truths = {}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try: i = int(row["id"])
            except: continue
            truths[i] = (row["truth"], row.get("family",""))
    per_item = []
    tp=fp=fn=0
    n_scored=0
    for obj in items:
        pid = int(obj.get("id", 0) or 0)
        pred = str(obj.get("generation",""))
        truth_tuple = truths.get(pid, None)
        if truth_tuple is None:
            continue
        truth, fam = truth_tuple
        n_scored += 1
        pred_n  = normalize(pred, match_mode)
        truth_n = normalize(truth, match_mode)
        if pred_n == "":
            fn += 1; ok=0
        elif pred_n == truth_n and truth_n != "":
            tp += 1; ok=1
        else:
            fp += 1; ok=0
        per_item.append({"id": pid, "prediction": pred_n or pred.strip(), "truth": truth_n, "family": fam, "correct": ok})
    n = n_scored or 1
    accuracy = tp/n
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision>0 and recall>0) else 0.0
    return per_item, {
        "n_scored": n_scored, "tp": tp, "fp": fp, "fn": fn,
        "accuracy": round(accuracy,6),
        "precision": round(precision,6),
        "recall": round(recall,6),
        "f1": round(f1,6),
        "hallucination_rate": round(fp/n,6),
        "omission_rate": round(fn/n,6),
        "match_mode": match_mode,
    }

# ---------------- Runner ----------------
def run_stock(runner, prompts_path, out_jsonl, tap, device=None, extra_args=None):
    cmd = [sys.executable, runner,
           "--gen_mode", "stock",
           "--prompts", prompts_path,
           "--out", out_jsonl,
           "--tap", str(tap),
           "--temperature", "0.0",
           "--top_p", "1.0",
           "--max_new_tokens", "8"]
    if device: cmd += ["--device", device]
    if extra_args: cmd += extra_args
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")
    return True

# ---------------- Adaptive calibration ----------------
def calibrate_mix(args, base_mix, rng, outdir: Path):
    """
    Try a sequence of mixes from easier → harder until accuracy lands in target window.
    """
    mixes = []
    # Start with moderately easy mix
    mixes.append(base_mix.copy())
    # If too low, tilt easier (more echo/extract)
    mixes.append({"echo_integer":0.45, "extract_last":0.25, "pick_larger":0.20, "min_of_three":0.05, "add_single_digit":0.05})
    # If too high, tilt harder (more add_two_digit / pick_larger)
    mixes.append({"echo_integer":0.10, "extract_last":0.15, "pick_larger":0.40, "min_of_three":0.15, "add_single_digit":0.10, "add_two_digit":0.10})
    mixes.append({"echo_integer":0.05, "extract_last":0.10, "pick_larger":0.45, "min_of_three":0.20, "add_single_digit":0.10, "add_two_digit":0.10})

    for idx, mix in enumerate(mixes, 1):
        tasks = sample_tasks(rng, args.calib_n, mix)
        p_path, t_csv = write_prompts_truths(outdir, tasks, stem=f"calib_{idx}")
        gen_path = str(outdir / f"generations_calib_{idx}.jsonl")
        run_stock(args.runner, p_path, gen_path, args.tap, device=args.device)
        items = read_generations_jsonl(gen_path)
        _, m = score(items, t_csv, match_mode="numbers_only")
        print(f"[CALIB {idx}] mix={mix} → acc={m['accuracy']:.3f}")
        if args.target_min <= m["accuracy"] <= args.target_max:
            return mix, m
    # Fall back to last tried mix
    return mixes[-1], m

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="STOCK-only adaptive easy benchmark (targets ~30–40% accuracy)")
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--n", type=int, default=120, help="Final benchmark size")
    ap.add_argument("--calib_n", type=int, default=24, help="Calibration batch size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--target_min", type=float, default=0.30)
    ap.add_argument("--target_max", type=float, default=0.40)
    ap.add_argument("--outdir", type=str, default="/mnt/data/arc_bench_stock_easy_adaptive")
    args, extra = ap.parse_known_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Base mix (moderately easy)
    base_mix = {"echo_integer":0.15, "extract_last":0.20, "pick_larger":0.35, "min_of_three":0.15, "add_single_digit":0.15}

    # 1) Calibrate on a small batch to find a mix near the target window
    chosen_mix, calib_metrics = calibrate_mix(args, base_mix, rng, outdir)

    # 2) Generate final tasks with the chosen mix
    tasks = sample_tasks(rng, args.n, chosen_mix)
    p_path, t_csv = write_prompts_truths(outdir, tasks, stem="final")
    print(f"[WRITE] Prompts → {p_path}")
    print(f"[WRITE] Truths  → {t_csv}")
    # Persist chosen mix
    with open(str(outdir / "chosen_mix.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen_mix": chosen_mix, "calib_metrics": calib_metrics}, f, indent=2)

    # 3) Run stock and score
    gen_path = str(outdir / "generations_stock.jsonl")
    run_stock(args.runner, p_path, gen_path, args.tap, device=args.device)
    items = read_generations_jsonl(gen_path)
    per_item, metrics = score(items, t_csv, match_mode="numbers_only")

    # 4) Save outputs
    items_csv = str(outdir / "items_stock.csv")
    with open(items_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prediction","truth","family","correct"])
        w.writeheader()
        for r in per_item:
            w.writerow(r)
    print(f"[CSV]  {items_csv}")

    metrics_json = str(outdir / "metrics_stock.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[JSON] {metrics_json}")

    metrics_csv = str(outdir / "metrics_stock.csv")
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode"])
        w.writeheader()
        w.writerow(metrics)
    print(f"[CSV]  {metrics_csv}")

    print(json.dumps({"SUMMARY": {"StockEasyAdaptive": metrics, "ChosenMix": chosen_mix, "Calib": calib_metrics}}, indent=2))
    print(f"[DONE] elapsed_sec={time.time()-t0:.3f}")

if __name__ == "__main__":
    main()
