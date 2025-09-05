#!/usr/bin/env python3
# Likelihood-ranking harness for MCQ benchmarks using text_arc_unified_base.py
# Runs: stock, geo, geo+detect, geo+detect+denoise (tap -9)
# Datasets: HellaSwag, TruthfulQA (mc2)
# Requires: pip install datasets

import argparse, os, json, csv, subprocess, sys, random
from pathlib import Path
from collections import Counter

# -----------------------------
# Config
# -----------------------------
DEFAULT_RUNNER = "text_arc_unified_base.py"
LETTERS = ["A","B","C","D"]

# v4b baseline (NO detect_sigma)
V4B_BASE = dict(
    alpha0="0.05", alpha_min="0.006",
    trend_tau="0.35", k_tr="12",
    detect_width="24",
    null_K="32", null_q="0.92", k_det="7",
    s_latch="0.30", linger="2", ema_center_beta="0.05",
)
PROFILE_JSON = "calib/profile_v4b_tap9_text.json"  # optional; used if present

# -----------------------------
# Dataset loaders (HuggingFace)
# -----------------------------
def load_hellaswag(n, seed, split="validation"):
    """Return list of (id, prompt, truth_letter, family, options[4])"""
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("hellaswag", split=split)
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    idxs = idxs[:n]
    items=[]
    for i, idx in enumerate(idxs, start=1):
        row = ds[int(idx)]
        ctx = row.get("ctx", row.get("context",""))
        opts = list(row["endings"])
        if len(opts) < 4: opts = (opts + [opts[-1]]*4)[:4]
        else:             opts = opts[:4]
        label = int(row["label"])
        truth_letter = LETTERS[label]
        prompt = f"Choose the most plausible ending to complete the sentence:\n\"{ctx.strip()} ...\""
        items.append((i, prompt, truth_letter, "hellaswag", opts))
    return items

def load_truthfulqa_mc2(n, seed, split="validation"):
    """Return list of (id, prompt, truth_letter, family, options[4])"""
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("truthful_qa", "mc2", split=split)
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    idxs = idxs[:n]
    items=[]
    for i, idx in enumerate(idxs, start=1):
        row = ds[int(idx)]
        q = row["question"].strip()
        opts = list(row["mc2_targets"])
        if len(opts) < 4: opts = (opts + [opts[-1]]*4)[:4]
        else:             opts = opts[:4]
        correct_idx = int(row.get("mc2_idx_correct", 0))
        if correct_idx > 3: correct_idx = 0
        truth_letter = LETTERS[correct_idx]
        prompt = f"{q}\nChoose the MOST TRUTHFUL answer."
        items.append((i, prompt, truth_letter, "truthfulqa_mc2", opts))
    return items

def make_dataset(task, n, seed, split):
    if task == "hellaswag":
        return load_hellaswag(n, seed, split)
    elif task == "truthfulqa":
        return load_truthfulqa_mc2(n, seed, split)
    else:
        raise ValueError("Unsupported task. Use: hellaswag or truthfulqa")

# -----------------------------
# I/O helpers
# -----------------------------
def write_truths(outdir: Path, items):
    p = outdir / "truths.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id","truth","family"])
        for i,_,letter,fam,_ in items: w.writerow([i,letter,fam])
    return str(p)

def write_choices_jsonl(outdir: Path, items):
    p = outdir / "choices.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, prompt, _, _, options in items:
            f.write(json.dumps({"id": i, "prompt": prompt, "options": options}) + "\n")
    return str(p)

def read_jsonl(path: str):
    arr=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                arr.append(json.loads(ln))
    return arr

# -----------------------------
# Runner invocation
# -----------------------------
def add_v4b_flags(cmd, use_detect=False, use_denoise=False):
    if os.path.exists(PROFILE_JSON):
        cmd += ["--config", PROFILE_JSON]
        cmd += ["--use_detect", "1" if use_detect else "0"]
        cmd += ["--use_denoise", "1" if use_denoise else "0"]
        return cmd
    for k, v in V4B_BASE.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--use_detect", "1" if use_detect else "0"]
    cmd += ["--use_denoise", "1" if use_denoise else "0"]
    return cmd

def run_rank_profile(runner, profile, choices_path, out_jsonl, tap, device, model, outdir="."):
    # some runners still require --prompts/--out; give harmless dummies
    dummy_prompts = os.path.join(outdir, "dummy_prompts.txt")
    if not os.path.exists(dummy_prompts):
        with open(dummy_prompts, "w", encoding="utf-8") as f:
            f.write("Q: dummy\n")
    dummy_out = os.path.join(outdir, "dummy_generations.jsonl")

    cmd = [sys.executable, runner,
           "--mcq_rank", "1",
           "--choices_jsonl", choices_path,
           "--rank_out", out_jsonl,
           "--tap", str(tap),
           "--model", model,
           "--device", device,
           "--gen_mode", ("stock" if profile == "stock" else "geo"),
           "--prompts", dummy_prompts,
           "--out", dummy_out]

    if profile == "stock":
        pass
    elif profile == "geo":
        cmd = add_v4b_flags(cmd, use_detect=False, use_denoise=False)
    elif profile == "geo_detect":
        cmd = add_v4b_flags(cmd, use_detect=True,  use_denoise=False)
    elif profile == "geo_detect_denoise":
        cmd = add_v4b_flags(cmd, use_detect=True,  use_denoise=True)
    else:
        raise ValueError("unknown profile")

    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")

# -----------------------------
# Scoring (reads 'prediction' from runner output)
# -----------------------------
def score_ranked(gens_jsonl, truths_csv):
    truths, fams = {}, {}
    with open(truths_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["id"])
            truths[i] = row["truth"].strip().upper()
            fams[i]   = row.get("family","")
    items = read_jsonl(gens_jsonl)
    tp=fp=fn=0; n_scored=0; fam_counts=Counter(); fam_tp=Counter()
    for idx, it in enumerate(items, start=1):
        pid = int(it.get("id", idx))
        pred = str(it.get("prediction","")).strip().upper()
        truth = truths.get(pid,"")
        fam = fams.get(pid,"")
        if fam: fam_counts[fam]+=1
        n_scored += 1
        if pred == "":
            fn += 1; ok = 0
        elif pred == truth:
            tp += 1; ok = 1; fam_tp[fam]+=1
        else:
            fp += 1; ok = 0
    n = max(1, n_scored)
    acc = tp/n
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec and rec) else 0.0
    fam_acc = {fam: (fam_tp[fam]/cnt if cnt else 0.0) for fam,cnt in fam_counts.items()}
    return {
        "n_scored":n_scored,"tp":tp,"fp":fp,"fn":fn,
        "accuracy":round(acc,6),"precision":round(prec,6),
        "recall":round(rec,6),"f1":round(f1,6),
        "hallucination_rate":round(fp/n,6),"omission_rate":round(fn/n,6),
        "family_accuracy":{k:round(v,6) for k,v in fam_acc.items()}
    }

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="MCQ likelihood ranking via NGF runner")
    ap.add_argument("--task", choices=["hellaswag","truthfulqa"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model",  type=str, default="gpt2-large")
    ap.add_argument("--tap",    type=int, default=-9)
    ap.add_argument("--outdir", type=str, default="runs_mcq_rank")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Data → truths + choices.jsonl
    items = make_dataset(args.task, args.n, args.seed, args.split)
    truths_csv   = write_truths(outdir, items)
    choices_path = write_choices_jsonl(outdir, items)

    # 2) Run four profiles
    profiles = ["stock","geo","geo_detect","geo_detect_denoise"]
    metrics_all = {}
    for prof in profiles:
        out_rank = str(outdir / f"rank_{args.task}_{prof}.jsonl")
        run_rank_profile(args.runner, prof, choices_path, out_rank, args.tap, args.device, args.model, outdir=str(outdir))
        metrics = score_ranked(out_rank, truths_csv)
        metrics_all[prof] = metrics
        # persist metrics
        with open(str(outdir / f"metrics_{args.task}_{prof}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(str(outdir / f"metrics_{args.task}_{prof}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate"])
            w.writeheader(); w.writerow({k:metrics.get(k,"") for k in w.fieldnames})
        print(f"[DONE] {prof} :: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} fp={metrics['hallucination_rate']:.3f} fn={metrics['omission_rate']:.3f}")

    # 3) Scoreboard
    print("\n=== SCOREBOARD ===")
    for prof in profiles:
        m = metrics_all[prof]
        print(f"{prof:>20s} | acc={m['accuracy']:.3f} | f1={m['f1']:.3f} | fp={m['hallucination_rate']:.3f} | fn={m['omission_rate']:.3f}")

if __name__ == "__main__":
    main()
