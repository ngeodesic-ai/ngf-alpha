#!/usr/bin/env python3
"""
arc_task_benchmark_stock_smoke.py — STOCK-only smoke test with controllable difficulty
--------------------------------------------------------------------------------------
Goals:
  • Keep everything legacy/stock (no geo/detect/denoise)
  • Make it easy to confirm nonzero TP by dialing task difficulty
  • Use strict QA format + explicit instruction + greedy decoding

New flags:
  --op {mix,add,sub,mul}       # which operation(s) to generate (default: mix)
  --a_min --a_max --b_min --b_max  # operand ranges (defaults: add/sub 2..20, mul 2..9)
  --n, --seed, --tap, --device, --match_mode, --outdir  # as usual

Examples:
  # Easiest: single-digit addition only
  python3 arc_task_benchmark_stock_smoke.py --op add --a_min 2 --a_max 9 --b_min 2 --b_max 9 \
    --n 40 --runner text_arc_unified_base.py --device cuda

python3 arc_task_benchmark_stock_smoke.py \
  --op add --a_min 2 --a_max 9 --b_min 2 --b_max 9 \
  --n 60 --tap -9 --device cuda \
  --runner text_arc_unified_base.py \
  --outdir benchmark_results/llm_benchmark_stock_add_smalldigits

python3 arc_task_benchmark_stock_smoke.py \
  --op add --a_min 2 --a_max 9 --b_min 2 --b_max 9 \
  --n 60 --tap -9 --device cuda \
  --runner text_arc_unified_base.py \
  --outdir benchmark_results/llm_benchmark_stock_add_smalldigits


  # Medium: add+sub within two digits
  python3 arc_task_benchmark_stock_smoke.py --op mix --a_min 2 --a_max 20 --b_min 2 --b_max 20 \
    --n 60 --runner /mnt/data/text_arc_unified_base.py --device cuda
"""

import os, re, csv, json, time, argparse, random, subprocess, sys
from pathlib import Path
from collections import Counter

DEFAULT_RUNNER = "text_arc_unified_base.py"

FEWSHOT = [
    ("Compute 3+4. Return only the integer.", "7"),
    ("Compute 9-2. Return only the integer.", "7"),
    ("Compute 6*5. Return only the integer.", "30"),
]

def ensure_dir(p):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def gen_task(rng, op, a_min, a_max, b_min, b_max):
    if op == "add":
        a, b = rng.randint(a_min, a_max), rng.randint(b_min, b_max)
        core = f"Compute {a}+{b}. Return only the integer."
        truth = a + b
    elif op == "sub":
        a, b = rng.randint(a_min, a_max), rng.randint(b_min, b_max)
        core = f"Compute {a}-{b}. Return only the integer."
        truth = a - b
    elif op == "mul":
        a, b = rng.randint(a_min, a_max), rng.randint(b_min, b_max)
        core = f"Compute {a}*{b}. Return only the integer."
        truth = a * b
    else:
        # mix: randomly pick an op each time
        op2 = random.choice(["add","sub","mul"])
        return gen_task(rng, op2, a_min, a_max, b_min, b_max)
    few = "\n".join([f"Q: {q}\nA: {a}" for q,a in FEWSHOT])
    prompt = f"{few}\n\nQ: {core}\nA:"
    return prompt, truth

def generate_tasks(n, seed, op, a_min, a_max, b_min, b_max):
    rng = random.Random(seed)
    tasks = []
    for i in range(1, n+1):
        p, t = gen_task(rng, op, a_min, a_max, b_min, b_max)
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
            w.writerow([i, str(t)])
    return prompts_path, truths_csv

# --- normalization & reading ---
def alnum_lower(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", " ", s or "")
    return " ".join(s.lower().strip().split())

_int_pat = re.compile(r"[+-]?\d+")
def numbers_only(s: str) -> str:
    if not s:
        return ""
    m = _int_pat.search(s)
    return m.group(0) if m else ""

def exact_strip(s: str) -> str:
    return (s or "").strip()

def normalize(s: str, mode: str) -> str:
    if mode == "numbers_only":
        return numbers_only(s)
    if mode == "alnum_lower":
        return alnum_lower(s)
    return exact_strip(s)

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

# --- scoring ---
def score(items, truths_csv, match_mode="numbers_only"):
    truths = {}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                i = int(row["id"])
            except Exception:
                continue
            truths[i] = str(row["truth"])

    per_item, tp, fp, fn, n_scored = [], 0,0,0,0
    for obj in items:
        pid = int(obj.get("id", 0) or 0)
        prompt = str(obj.get("prompt",""))
        pred   = str(obj.get("generation",""))
        truth  = truths.get(pid, None)

        pred_is_empty = (pred.strip() == "")
        pred_n  = normalize(pred, match_mode)
        truth_n = normalize(truth, match_mode) if (truth is not None) else None

        correct = None
        if truth is not None:
            n_scored += 1
            if pred_is_empty:
                fn += 1
                correct = 0
            else:
                if pred_n == truth_n and truth_n != "":
                    tp += 1
                    correct = 1
                else:
                    fp += 1
                    correct = 0

        toks = pred.strip().split()
        n_words = len(toks)
        adj_dup = 0.0
        tri_rep = 0.0
        uniq_ratio = 1.0
        loopish = 0
        if n_words > 1:
            adj_dup = sum(1 for i in range(1, n_words) if toks[i]==toks[i-1]) / (n_words-1)
            tri = [" ".join(toks[i:i+3]) for i in range(n_words-2)]
            if tri:
                counts = Counter(tri)
                tri_rep = sum(max(0,c-1) for c in counts.values()) / max(1,len(tri))
            uniq_ratio = len(set(toks)) / n_words
            loopish = int(any(c>=6 for c in Counter(toks).values()))

        per_item.append({
            "id": pid, "prompt": prompt, "prediction": pred, "truth": truth, "correct": correct,
            "len_words": n_words, "adj_dup": round(adj_dup,4), "tri_rep": round(tri_rep,4),
            "uniq_ratio": round(uniq_ratio,4), "loopish": loopish,
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
        "match_mode": match_mode,
    }
    return per_item, metrics

def run_stock(runner, prompts_path, out_jsonl, tap, device=None, extra_args=None):
    cmd = [sys.executable, runner,
           "--gen_mode", "stock",
           "--prompts", prompts_path,
           "--out", out_jsonl,
           "--tap", str(tap),
           "--temperature", "0.0",
           "--top_p", "1.0",
           "--max_new_tokens", "8"]
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
    ap = argparse.ArgumentParser(description="STOCK-only smoke test with controllable difficulty")
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--op", type=str, default="mix", choices=["mix","add","sub","mul"])
    ap.add_argument("--a_min", type=int, default=2)
    ap.add_argument("--a_max", type=int, default=20)
    ap.add_argument("--b_min", type=int, default=2)
    ap.add_argument("--b_max", type=int, default=20)
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--match_mode", type=str, default="numbers_only", choices=["numbers_only","alnum_lower","exact"])
    ap.add_argument("--outdir", type=str, default="arc_bench_stock_smoke")
    args, extra = ap.parse_known_args()

    # Default narrower ranges for mul to keep it easy unless user overrides
    if args.op == "mul" and args.a_max == 20 and args.b_max == 20:
        args.a_max = min(args.a_max, 9)
        args.b_max = min(args.b_max, 9)

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    tasks = generate_tasks(args.n, args.seed, args.op, args.a_min, args.a_max, args.b_min, args.b_max)
    prompts_path, truths_csv = write_prompts_and_truths(outdir, tasks)
    print(f"[WRITE] Prompts → {prompts_path}")
    print(f"[WRITE] Truths  → {truths_csv}")

    generations_jsonl = str(outdir / "generations_stock.jsonl")
    run_stock(args.runner, prompts_path, generations_jsonl, args.tap, device=args.device, extra_args=extra)

    items = read_generations_jsonl(generations_jsonl)
    per_item, metrics = score(items, truths_csv, match_mode=args.match_mode)

    items_csv = str(outdir / "items_stock.csv")
    ensure_dir(items_csv)
    with open(items_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prompt","prediction","truth","correct","len_words","adj_dup","tri_rep","uniq_ratio","loopish"])
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

    print(json.dumps({"SUMMARY": {"Stock": metrics}}, indent=2))
    print(f"[DONE] elapsed_sec={time.time()-t0:.3f}")

if __name__ == "__main__":
    main()
