#!/usr/bin/env python3
"""
arc_matrix_benchmark_stock.py — STOCK-only benchmark for matrix rotations
-------------------------------------------------------------------------
This mirrors the "matrix rotate" family you used at the embedding level, but for text.
It:
  • Generates small 2x2 and 3x3 matrices with integer entries
  • Asks for 90° rotations (clockwise / counterclockwise) and 180°
  • Uses QA-style prompts + few-shot exemplars
  • Forces greedy-ish decoding for short, bracketed answers
  • Parses JSON-like matrices from generations and compares to ground truth

Example:
  python3 arc_matrix_benchmark_stock.py \
    --n 60 --tap -9 --device cuda \
    --runner /mnt/data/text_arc_unified_base.py \
    --outdir ./layer9_wiring_plan/llm_benchmark_stock_matrix

  python3 arc_matrix_benchmark_stock.py \
    --n 60 --tap -9 --device cuda \
    --runner text_arc_unified_base.py \
    --outdir benchmark_results/llm_benchmark_stock_matrix
"""

import os, re, csv, json, time, argparse, random, subprocess, sys
from pathlib import Path

DEFAULT_RUNNER = "/mnt/data/text_arc_unified_base.py"

FEWSHOT = [
    # 2x2 clockwise
    ("Rotate the matrix [[2,3],[4,5]] by 90° clockwise. Return only the matrix in bracketed form.",
     "[[4,2],[5,3]]"),
    # 2x2 counterclockwise
    ("Rotate the matrix [[2,3],[4,5]] by 90° counterclockwise. Return only the matrix in bracketed form.",
     "[[3,5],[2,4]]"),
    # 3x3 180
    ("Rotate the matrix [[1,2,3],[4,5,6],[7,8,9]] by 180°. Return only the matrix in bracketed form.",
     "[[9,8,7],[6,5,4],[3,2,1]]"),
]

def ensure_dir(p):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def rot90_cw(mat):
    # list of lists
    return [list(row) for row in zip(*mat[::-1])]

def rot90_ccw(mat):
    return [list(row) for row in zip(*mat)][::-1]

def rot180(mat):
    return [row[::-1] for row in mat[::-1]]

def gen_matrix(rng, size, lo=-9, hi=9):
    return [[rng.randint(lo, hi) for _ in range(size)] for _ in range(size)]

def mat_to_str(mat):
    return json.dumps(mat)  # canonical bracketed format

def gen_task(rng):
    size = rng.choice([2,3])
    mat = gen_matrix(rng, size)
    op = rng.choice(["cw","ccw","180"])
    if op == "cw":
        q = f"Rotate the matrix {mat_to_str(mat)} by 90° clockwise. Return only the matrix in bracketed form."
        t = rot90_cw(mat)
    elif op == "ccw":
        q = f"Rotate the matrix {mat_to_str(mat)} by 90° counterclockwise. Return only the matrix in bracketed form."
        t = rot90_ccw(mat)
    else:
        q = f"Rotate the matrix {mat_to_str(mat)} by 180°. Return only the matrix in bracketed form."
        t = rot180(mat)
    # Inline few-shot exemplars for this prompt
    few = "\n".join([f"Q: {q0}\nA: {a0}" for q0,a0 in FEWSHOT])
    prompt = f"{few}\n\nQ: {q}\nA:"
    truth = mat_to_str(t)
    return prompt, truth

def generate_tasks(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(1, n+1):
        p, t = gen_task(rng)
        tasks.append((i, p, t))
    return tasks

def write_prompts_and_truths(outdir: Path, tasks):
    prompts_path = str(outdir / "arc_prompts.txt")
    truths_csv   = str(outdir / "arc_truths.csv")
    ensure_dir(prompts_path)
    with open(prompts_path, "w", encoding="utf-8") as f:
        for _, p, _ in tasks:
            f.write(p.strip()+"\n")
    with open(truths_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","truth"])
        for i, _, t in tasks:
            w.writerow([i, t])
    return prompts_path, truths_csv

# -------- Parsing matrices from model output --------
MATRIX_RE = re.compile(r"\[\s*(\[[^\[\]]*\]\s*(,\s*\[[^\[\]]*\]\s*)*)\s*\]")

def parse_first_matrix(s):
    """
    Extract the first top-level [...] block that looks like a matrix and parse JSON.
    Returns a list of lists of ints if successful; else None.
    """
    if not s:
        return None
    m = MATRIX_RE.search(s)
    if not m:
        return None
    block = m.group(0)
    try:
        arr = json.loads(block)
        if isinstance(arr, list) and all(isinstance(r, list) for r in arr) and all(all(isinstance(x, (int,float)) for x in r) for r in arr):
            # coerce floats that are actually ints
            arr2 = [[int(x) if int(x) == x else x for x in r] for r in arr]
            return arr2
    except Exception:
        return None
    return None

def normalize_mat_str(s):
    # Keep canonical json format spacing
    try:
        arr = parse_first_matrix(s)
        if arr is None:
            return ""
        return json.dumps(arr)
    except Exception:
        return ""

# -------- Scoring --------
def score(items, truths_csv):
    truths = {}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                i = int(row["id"])
            except Exception:
                continue
            truths[i] = row["truth"]

    per_item = []
    tp = fp = fn = 0
    n_scored = 0
    for obj in items:
        pid = int(obj.get("id", 0) or 0)
        pred_raw = str(obj.get("generation",""))
        truth = truths.get(pid)

        pred_norm = normalize_mat_str(pred_raw)
        truth_norm = truth  # already canonical json

        correct = None
        if truth_norm is not None:
            n_scored += 1
            if pred_norm == "":
                fn += 1
                correct = 0
            else:
                if pred_norm == truth_norm:
                    tp += 1
                    correct = 1
                else:
                    fp += 1
                    correct = 0

        per_item.append({
            "id": pid,
            "prediction": pred_norm if pred_norm else pred_raw.strip(),
            "truth": truth_norm,
            "correct": correct,
        })

    n = n_scored
    accuracy = (tp / n) if n else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision>0 and recall>0) else 0.0
    halluc_rate = (fp / n) if n else 0.0
    omission_rate = (fn / n) if n else 0.0

    metrics = {
        "n_scored": n, "tp": tp, "fp": fp, "fn": fn,
        "accuracy": round(accuracy,6), "precision": round(precision,6),
        "recall": round(recall,6), "f1": round(f1,6),
        "hallucination_rate": round(halluc_rate,6), "omission_rate": round(omission_rate,6),
        "match_mode": "matrix_json",
    }
    return per_item, metrics

def read_generations_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception:
                pass
    return items

def run_stock(runner, prompts_path, out_jsonl, tap, device=None, extra_args=None):
    cmd = [sys.executable, runner,
           "--gen_mode", "stock",
           "--prompts", prompts_path,
           "--out", out_jsonl,
           "--tap", str(tap),
           "--temperature", "0.0",
           "--top_p", "1.0",
           "--max_new_tokens", "32"]
    if device:
        cmd += ["--device", device]
    if extra_args:
        cmd += extra_args
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")
    return True

def main():
    ap = argparse.ArgumentParser(description="STOCK-only: matrix rotation benchmark")
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="/mnt/data/arc_bench_stock_matrix")
    args, extra = ap.parse_known_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks(args.n, args.seed)
    prompts_path, truths_csv = write_prompts_and_truths(outdir, tasks)
    print(f"[WRITE] Prompts → {prompts_path}")
    print(f"[WRITE] Truths  → {truths_csv}")

    generations_jsonl = str(outdir / "generations_stock.jsonl")
    run_stock(args.runner, prompts_path, generations_jsonl, args.tap, device=args.device, extra_args=extra)

    items = read_generations_jsonl(generations_jsonl)
    per_item, metrics = score(items, truths_csv)

    # Save
    items_csv = str(outdir / "items_stock_matrix.csv")
    with open(items_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prediction","truth","correct"])
        w.writeheader()
        for r in per_item:
            w.writerow(r)
    print(f"[CSV]  {items_csv}")

    metrics_json = str(outdir / "metrics_stock_matrix.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[JSON] {metrics_json}")

    metrics_csv = str(outdir / "metrics_stock_matrix.csv")
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode"])
        w.writeheader()
        w.writerow(metrics)
    print(f"[CSV]  {metrics_csv}")

    print(json.dumps({"SUMMARY": {"StockMatrix": metrics}}, indent=2))
    print(f"[DONE] elapsed_sec={time.time()-t0:.3f}")

if __name__ == "__main__":
    main()
