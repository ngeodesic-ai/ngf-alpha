#!/usr/bin/env python3
# mcq_benchmark.py — unified MCQ (stock + geo only), letter-only, oneline prompts, CPU fallback
import os, csv, json, time, argparse, random, subprocess, sys, re
from pathlib import Path
from collections import Counter

"""
# Stock (generate prompts) — GPT-2 medium, 50 Qs
python3 mcq_benchmark.py \
  --runner text_arc_unified_base.py \
  --device cuda --model gpt2-medium --tap -9 \
  --mode stock \
  --n 50 --k_shot 5 --max_new_tokens 1 \
  --outdir benchmark_results/mcq_stock_medium_k5

# GEO (reuse exact same prompts/truths)
python3 mcq_benchmark.py \
  --runner text_arc_unified_base.py \
  --device cuda --model gpt2-medium --tap -9 \
  --mode geo \
  --prompts_in benchmark_results/mcq_stock_medium_k5/final_prompts.txt \
  --truths_in  benchmark_results/mcq_stock_medium_k5/final_truths.csv \
  --outdir benchmark_results/mcq_geo_medium_k5

"""

DEFAULT_RUNNER = "/mnt/data/text_arc_unified_base.py"
FIELD_CANDIDATES = ["generation","prediction","output","text","decoded","answer","completion"]
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

def pick_pred_field(items):
    if not items: return "generation"
    counts = Counter()
    for it in items:
        for k in FIELD_CANDIDATES:
            if k in it and isinstance(it[k], (str, int, float, bool)):
                counts[k] += 1
    return max(counts, key=counts.get) if counts else "generation"

def extract_letter(s: str) -> str:
    if not s: return ""
    m = LETTER_RE.search(s)
    return m.group(1).upper() if m else ""

# ---------- Few-shot (inline, one-line) ----------
def make_oneline_examples():
    return [
        "Q: Which is larger? A) 7 B) 12 C) 5 D) 3 A: B",
        "Q: Return the smallest number. A) 9 B) 2 C) 4 D) 8 A: B",
        "Q: Compute 4+7. A) 9 B) 10 C) 11 D) 12 A: C",
        "Q: Is 13 even or odd? A) Even B) Odd C) Prime D) Composite A: B",
    ]

def build_prefix(k_shot: int):
    ex = make_oneline_examples()
    k = max(0, min(k_shot, len(ex)))
    return (" | ".join(ex[:k]) + " | ") if k > 0 else ""

# ---------- MCQ generators (one-line prompts, letter-only) ----------
def mcq_pick_larger(rng):
    a = rng.randint(2, 99); b = rng.randint(2, 99)
    while b == a: b = rng.randint(2, 99)
    correct = max(a, b)
    pool = set([a, b, abs(a-b), a+1, b+1, max(0, correct-1), correct+2, max(0, correct-2)])
    pool.discard(correct)
    while len(pool) < 3:
        x = rng.randint(0, 120)
        if x != correct:
            pool.add(x)
    distractors = random.sample(list(pool), 3)
    choices = distractors + [correct]; random.shuffle(choices)
    letters = ["A","B","C","D"]
    assert correct in choices
    truth_letter = letters[choices.index(correct)]
    prompt = f"Q: Which is larger? A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]} Respond with exactly one capital letter (A, B, C, or D). A:"
    return prompt, truth_letter, "pick_larger", [str(x) for x in choices]

def mcq_min_of_three(rng):
    a,b,c = rng.randint(2,99), rng.randint(2,99), rng.randint(2,99)
    correct = min(a,b,c)
    pool = [a,b,c, correct+1, max(0, correct-1), a+b-c]
    pool = [x for x in pool if x != correct and x >= 0]
    while len(pool) < 3:
        x = rng.randint(0, 120)
        if x != correct and x not in pool:
            pool.append(x)
    distractors = random.sample(pool, k=3)
    choices = distractors + [correct]; random.shuffle(choices)
    letters = ["A","B","C","D"]
    assert correct in choices
    truth_letter = letters[choices.index(correct)]
    prompt = f"Q: Return the smallest number. A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]} Respond with exactly one capital letter (A, B, C, or D). A:"
    return prompt, truth_letter, "min_of_three", [str(x) for x in choices]

def mcq_add_single(rng):
    a,b = rng.randint(2,9), rng.randint(2,9)
    correct = a+b
    pool = [correct-1, correct+1, abs(a-b), max(0, correct-2), correct+2]
    pool = [x for x in pool if x != correct]
    while len(pool) < 3:
        x = rng.randint(max(0, correct-5), correct+5)
        if x != correct and x not in pool:
            pool.append(x)
    distractors = random.sample(pool, k=3)
    choices = distractors + [correct]; random.shuffle(choices)
    letters = ["A","B","C","D"]
    assert correct in choices
    truth_letter = letters[choices.index(correct)]
    prompt = f"Q: Compute {a}+{b}. A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]} Respond with exactly one capital letter (A, B, C, or D). A:"
    return prompt, truth_letter, "add_single_digit", [str(x) for x in choices]

def mcq_parity(rng):
    x = rng.randint(2, 99)
    opts = ["Even", "Odd", "Prime", "Composite"]
    correct = "Even" if x % 2 == 0 else "Odd"
    random.shuffle(opts)
    letters = ["A","B","C","D"]
    truth_letter = letters[opts.index(correct)]
    prompt = f"Q: Is {x} even or odd? A) {opts[0]} B) {opts[1]} C) {opts[2]} D) {opts[3]} Respond with exactly one capital letter (A, B, C, or D). A:"
    return prompt, truth_letter, "parity", opts

FAMS = {
    "pick_larger": mcq_pick_larger,
    "min_of_three": mcq_min_of_three,
    "add_single_digit": mcq_add_single,
    "parity": mcq_parity,
}

def sample_mcqs(rng, n):
    mix = {"pick_larger":0.35, "min_of_three":0.25, "add_single_digit":0.25, "parity":0.15}
    fams = list(mix.items()); weights = [w for _,w in fams]
    tot = sum(weights) or 1.0; weights = [w/tot for w in weights]
    items = []
    for i in range(1, n+1):
        fam_name = random.choices([k for k,_ in fams], weights=weights, k=1)[0]
        stem, letter, fam, choices = FAMS[fam_name](rng)
        items.append((i, stem, letter, fam, choices))
    return items

def write_prompts_truths(outdir: Path, items, stem="final", prefix=""):
    prompts_path = str(outdir / f"{stem}_prompts.txt")
    truths_csv   = str(outdir / f"{stem}_truths.csv")
    questions_csv = str(outdir / f"{stem}_questions.csv")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for _, p, _, _, _ in items:
            f.write((prefix + p).strip()+"\n")   # ONE LINE PER PROMPT
    with open(truths_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id","truth","family"])
        for i, _, letter, fam, _ in items:
            w.writerow([i, letter, fam])
    with open(questions_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id","family","A","B","C","D","truth_letter"])
        for i, _, letter, fam, choices in items:
            w.writerow([i, fam] + (choices if isinstance(choices[0], str) else list(map(str, choices))) + [letter])
    return prompts_path, truths_csv, questions_csv

# ---------- Runner (stock + geo only) ----------
def run_mode(runner, gen_mode, prompts_path, out_jsonl, tap, device=None, model=None, max_new_tokens="1", auto_cpu_fallback=True):
    cmd = [sys.executable, runner,
           "--gen_mode", gen_mode,
           "--prompts", prompts_path,
           "--out", out_jsonl,
           "--tap", str(tap),
           "--temperature", "0.0",
           "--top_p", "1.0",
           "--max_new_tokens", str(max_new_tokens)]
    if device: cmd += ["--device", device]
    if model:  cmd += ["--model", model]

    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 and auto_cpu_fallback and ("CUDA error" in (p.stderr or "") or "device-side assert" in (p.stderr or "")):
        print("[WARN] CUDA error detected; retrying on CPU...", file=sys.stderr)
        cmd_cpu = [x for x in cmd if x != "cuda"]
        if "--device" in cmd_cpu:
            i = cmd_cpu.index("--device")
            cmd_cpu[i+1] = "cpu"
        else:
            cmd_cpu += ["--device", "cpu"]
        print("[RUN]", " ".join(cmd_cpu))
        p = subprocess.run(cmd_cpu, capture_output=True, text=True)

    if p.returncode != 0:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")
    return True

# ---------- IO & scoring ----------
def read_jsonl(path: str):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: arr.append(json.loads(ln))
            except Exception: pass
    return arr

def score(items, truths_csv):
    truths = {}; fams={}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try: i = int(row["id"])
            except: continue
            truths[i] = row["truth"].strip().upper()
            fams[i] = row.get("family","")
    pred_field = pick_pred_field(items)
    per_item=[]; tp=fp=fn=0; n_scored=0; fam_counts=Counter(); fam_tp=Counter()
    for idx, it in enumerate(items, start=1):
        pid = int(it.get("id", idx) or idx)  # fallback by order
        pred_raw = str(it.get(pred_field, ""))
        letter = extract_letter(pred_raw)
        truth = truths.get(pid, "")
        fam = fams.get(pid, "")
        n_scored += 1
        if fam: fam_counts[fam]+=1
        if letter == "":
            fn += 1; ok=0
        elif letter == truth:
            tp += 1; ok=1; fam_tp[fam]+=1
        else:
            fp += 1; ok=0
        per_item.append({"id": pid, "prediction": letter or pred_raw.strip(), "truth": truth, "family": fam, "correct": ok})
    n = max(1, n_scored)
    accuracy = tp/n; precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision>0 and recall>0) else 0.0
    metrics = {
        "n_scored": n_scored, "tp": tp, "fp": fp, "fn": fn,
        "accuracy": round(accuracy,6), "precision": round(precision,6),
        "recall": round(recall,6), "f1": round(f1,6),
        "hallucination_rate": round(fp/n,6), "omission_rate": round(fn/n,6),
        "match_mode": "letter_exact", "pred_field_used": pred_field
    }
    return per_item, metrics

def main():
    ap = argparse.ArgumentParser(description="Unified MCQ benchmark — stock + geo only (letter-only, one-line prompts)")
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--mode", type=str, default="stock", choices=["stock","geo"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="/mnt/data/mcq_bench_stock_geo")
    # Reuse existing prompts/truths (for GEO after STOCK)
    ap.add_argument("--prompts_in", type=str, default=None)
    ap.add_argument("--truths_in", type=str, default=None)
    ap.add_argument("--no_cpu_fallback", action="store_true", help="disable auto CPU retry on CUDA errors")
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 1) Prepare prompts/truths
    if args.prompts_in and args.truths_in:
        prompts_path = args.prompts_in
        truths_csv = args.truths_in
        print(f"[REUSE] Prompts → {prompts_path}")
        print(f"[REUSE] Truths  → {truths_csv}")
    else:
        prefix = build_prefix(args.k_shot)
        tasks = sample_mcqs(rng, args.n)
        prompts_path, truths_csv, questions_csv = write_prompts_truths(outdir, tasks, stem="final", prefix=prefix)
        print(f"[WRITE] Prompts   → {prompts_path}")
        print(f"[WRITE] Truths    → {truths_csv}")
        print(f"[WRITE] Questions → {questions_csv}")

    # 2) Run generation
    gen_mode = "stock" if args.mode=="stock" else "geo"
    gens_jsonl = str(outdir / f"generations_{args.mode}.jsonl")
    run_mode(args.runner, gen_mode, prompts_path, gens_jsonl, args.tap,
             device=args.device, model=args.model, max_new_tokens=str(args.max_new_tokens),
             auto_cpu_fallback=(not args.no_cpu_fallback))

    # 3) Score
    items = read_jsonl(gens_jsonl)
    per_item, metrics = score(items, truths_csv)

    # 4) Save artifacts
    items_csv = str(outdir / f"items_{args.mode}.csv")
    with open(items_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prediction","truth","family","correct"])
        w.writeheader()
        for r in per_item: w.writerow(r)
    print(f"[CSV]  {items_csv}")

    metrics_json = str(outdir / f"metrics_{args.mode}.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[JSON] {metrics_json}")

    metrics_csv = str(outdir / f"metrics_{args.mode}.csv")
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode","pred_field_used"])
        w.writeheader()
        row = {k: metrics.get(k, "") for k in ["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode","pred_field_used"]}
        w.writerow(row)
    print(f"[CSV]  {metrics_csv}")

    out_json = str(outdir / f"generations_{args.mode}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"[JSON] {out_json}")

    print(json.dumps({"SUMMARY": {f"{args.mode.upper()}": metrics}}, indent=2))
    print(f"[DONE] elapsed_sec={time.time()-t0:.3f}")

if __name__ == "__main__":
    main()
