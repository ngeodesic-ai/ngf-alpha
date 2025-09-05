#!/usr/bin/env python3
import os, re, csv, json, time, argparse, random, subprocess, sys
from pathlib import Path
from collections import Counter


"""
python3 arc_easy_benchmark_stock_adaptive_v2_json.py \
  --runner text_arc_unified_base.py \
  --device cuda --tap -9 --model gpt2-medium \
  --n 120 --calib_n 24 --k_shot 5 \
  --outdir stock_easy_adaptive_v2_cuda

python3 stock_benchmark_foundation.py \
  --runner text_arc_unified_base.py \
  --device cuda --model gpt2-medium --tap -9 \
  --n 120 --k_shot 8 --max_new_tokens 8 \
  --fixed_mix '{"echo_integer":0.60,"extract_last":0.20,"pick_larger":0.20}' \
  --outdir stock_foundation_run
  
"""

DEFAULT_RUNNER = "text_arc_unified_base.py"

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

_int_pat = re.compile(r"[+-]?\d+")
def numbers_only(s: str) -> str:
    if not s: return ""
    m = _int_pat.search(s); return m.group(0) if m else ""

def read_generations_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: items.append(json.loads(ln))
            except Exception: pass
    return items

def prompt_wrap(q: str, fewshot_pairs):
    few = "\n".join([f"Q: {q0}\nA: {a0}" for q0,a0 in fewshot_pairs])
    return f"{few}\n\nQ: {q}\nA:"

def gen_echo_integer(rng, fewshot_pairs):
    x = rng.randint(2, 99)
    q = f"Return only the integer {x}."
    return prompt_wrap(q, fewshot_pairs), str(x)

def gen_pick_larger(rng, fewshot_pairs, lo=2, hi=99):
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    while b == a: b = rng.randint(lo, hi)
    larger = a if a > b else b
    q = f"Which is larger, {a} or {b}? Return only the integer."
    return prompt_wrap(q, fewshot_pairs), str(larger)

def gen_min_of_three(rng, fewshot_pairs, lo=2, hi=99):
    a, b, c = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)
    smallest = min(a,b,c)
    q = f"Return the smallest integer: {a}, {b}, {c}."
    return prompt_wrap(q, fewshot_pairs), str(smallest)

def gen_extract_last(rng, fewshot_pairs):
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 99)
    q = f"In 'IDs: {a}, {b}, {c}', return the last integer."
    return prompt_wrap(q, fewshot_pairs), str(c)

def gen_add_single_digit(rng, fewshot_pairs):
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    q = f"Compute {a}+{b}. Return only the integer."
    return prompt_wrap(q, fewshot_pairs), str(a+b)

def gen_add_two_digit(rng, fewshot_pairs):
    a, b = rng.randint(10, 29), rng.randint(10, 29)
    q = f"Compute {a}+{b}. Return only the integer."
    return prompt_wrap(q, fewshot_pairs), str(a+b)

FAMILIES = {
    "echo_integer": gen_echo_integer,
    "pick_larger": gen_pick_larger,
    "min_of_three": gen_min_of_three,
    "extract_last": gen_extract_last,
    "add_single_digit": gen_add_single_digit,
    "add_two_digit": gen_add_two_digit,
}

def sample_tasks(rng, n, mix, fewshot_pairs):
    fams = list(mix.items()); weights = [w for _,w in fams]
    tot = sum(weights) or 1.0; weights = [w/tot for w in weights]
    tasks = []
    for i in range(1, n+1):
        fam = rng.choices(fams, weights=weights, k=1)[0][0]
        p, t = FAMILIES[fam](rng, fewshot_pairs)
        tasks.append((i, p, t, fam))
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

def score(items, truths_csv):
    truths = {}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try: i = int(row["id"])
            except: continue
            truths[i] = (row["truth"], row.get("family",""))
    per_item=[]; tp=fp=fn=0; n_scored=0; fam_counts=Counter(); fam_tp=Counter()
    for obj in items:
        pid = int(obj.get("id", 0) or 0)
        pred = str(obj.get("generation",""))
        t = truths.get(pid, None)
        if t is None: continue
        truth, fam = t; n_scored += 1; fam_counts[fam] += 1
        pred_n = numbers_only(pred); truth_n = numbers_only(truth)
        if pred_n == "": fn += 1; ok=0
        elif pred_n == truth_n and truth_n != "": tp += 1; ok=1; fam_tp[fam]+=1
        else: fp += 1; ok=0
        per_item.append({"id": pid, "prediction": pred_n or pred.strip(), "truth": truth_n, "family": fam, "correct": ok})
    n = n_scored or 1
    accuracy = tp/n; precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision>0 and recall>0) else 0.0
    fam_acc = {k: round(fam_tp[k]/v, 3) for k,v in fam_counts.items() if v>0}
    return per_item, {
        "n_scored": n_scored, "tp": tp, "fp": fp, "fn": fn,
        "accuracy": round(accuracy,6), "precision": round(precision,6),
        "recall": round(recall,6), "f1": round(f1,6),
        "hallucination_rate": round(fp/n,6), "omission_rate": round(fn/n,6),
        "family_accuracy": fam_acc, "match_mode": "numbers_only",
    }

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
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")
    return True

def calibrate_mix(args, rng, outdir: Path, fewshot_pairs):
    mixes = [
        {"echo_integer":0.70, "extract_last":0.20, "pick_larger":0.10},
        {"echo_integer":0.60, "extract_last":0.20, "pick_larger":0.20},
        {"echo_integer":0.50, "extract_last":0.25, "pick_larger":0.25},
        {"echo_integer":0.45, "extract_last":0.25, "pick_larger":0.25, "min_of_three":0.05},
        {"echo_integer":0.35, "extract_last":0.25, "pick_larger":0.30, "min_of_three":0.10},
        {"echo_integer":0.25, "extract_last":0.25, "pick_larger":0.35, "min_of_three":0.15},
        {"echo_integer":0.20, "extract_last":0.20, "pick_larger":0.40, "min_of_three":0.15, "add_single_digit":0.05},
        {"echo_integer":0.15, "extract_last":0.15, "pick_larger":0.45, "min_of_three":0.15, "add_single_digit":0.10},
    ]
    last_m=None
    for idx, mix in enumerate(mixes, 1):
        tasks = sample_tasks(rng, args.calib_n, mix, fewshot_pairs)
        p_path, t_csv = write_prompts_truths(outdir, tasks, stem=f"calib_{idx}")
        gen_path = str(outdir / f"generations_calib_{idx}.jsonl")
        run_stock(args.runner, p_path, gen_path, args.tap, device=args.device)
        items = read_generations_jsonl(gen_path)
        _, m = score(items, t_csv); last_m=m
        print(f"[CALIB {idx}] acc={m['accuracy']:.3f} mix={mix}")
        if args.target_min <= m["accuracy"] <= args.target_max: return mix, m
    return mixes[-1], last_m

def main():
    ap = argparse.ArgumentParser(description="STOCK-only adaptive easy benchmark (v2 JSON, targets ~30–40% accuracy)")
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--calib_n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--target_min", type=float, default=0.30)
    ap.add_argument("--target_max", type=float, default=0.40)
    ap.add_argument("--k_shot", type=int, default=7)
    ap.add_argument("--fixed_mix", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="arc_bench_stock_easy_adaptive_v2")
    args, extra = ap.parse_known_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    fewshot_pairs = make_fewshot(max(1, min(10, args.k_shot)))

    if args.fixed_mix:
        try: chosen_mix = json.loads(args.fixed_mix)
        except Exception as e: raise SystemExit(f"--fixed_mix must be JSON: {e}")
        calib_metrics = {"note": "fixed_mix used; no calibration"}
    else:
        chosen_mix, calib_metrics = calibrate_mix(args, rng, outdir, fewshot_pairs)

    tasks = sample_tasks(rng, args.n, chosen_mix, fewshot_pairs)
    p_path, t_csv = write_prompts_truths(outdir, tasks, stem="final")
    print(f"[WRITE] Prompts → {p_path}")
    print(f"[WRITE] Truths  → {t_csv}")
    with open(str(outdir / "chosen_mix.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen_mix": chosen_mix, "calib_metrics": calib_metrics, "k_shot": args.k_shot}, f, indent=2)

    gen_path = str(outdir / "generations_stock.jsonl")
    run_stock(args.runner, p_path, gen_path, args.tap, device=args.device)
    items = read_generations_jsonl(gen_path)
    per_item, metrics = score(items, t_csv)

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

    # Emit clickable JSON array for generations
    try:
        out_json = str(outdir / "generations_stock.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"[JSON] {out_json}")
    except Exception as e:
        print(f"[WARN] failed to write generations_stock.json: {e}")

    print(json.dumps({"SUMMARY": {"StockEasyAdaptiveV2": metrics, "ChosenMix": chosen_mix, "k_shot": args.k_shot}}, indent=2))
    print(f"[DONE] elapsed_sec={time.time()-t0:.3f}")

if __name__ == "__main__":
    main()
