#!/usr/bin/env python3
# bench_mcq_hf.py — Real MCQ benchmarks (HellaSwag, TruthfulQA-MC2) via text_arc_unified_base.py
import argparse, os, json, csv, time, re, subprocess, sys, random
from pathlib import Path
from collections import Counter


"""
python3 bench_mcq_hf.py \
  --task hellaswag --split validation --n 50 --seed 42 --k_shot 4 \
  --runner text_arc_unified_base.py --device cuda --model gpt2-medium \
  --tap -9 --outdir runs_hellaswag_fix \
  --max_new_tokens 4 --temperature 0.3 --top_p 0.8
"""

# -----------------------------
# Config & constants
# -----------------------------
DEFAULT_RUNNER = "text_arc_unified_base.py"  # path or name on PYTHONPATH
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

# v4b baseline flags (no detect_sigma; detect is gain-only; denoise only in final stage)
V4B_BASE = dict(
    alpha0="0.05", alpha_min="0.006",
    trend_tau="0.35", k_tr="12",
    detect_width="24",
    null_K="32", null_q="0.92", k_det="7",
    s_latch="0.30", linger="2", ema_center_beta="0.05",
)
PROFILE_JSON = "calib/profile_v4b_tap9_text.json"  # optional; used if found

# -----------------------------
# Prompt utils
# -----------------------------
def extract_letter(s: str) -> str:
    if not isinstance(s, str): return ""
    m = LETTER_RE.search(s)
    return m.group(1).upper() if m else ""

def build_prefix(k_shot: int):
    # A few very short universal MCQ exemplars
    examples = [
        "Q: Choose the correct answer. A) 3 B) 5 C) 7 D) 9 A: B",
        "Q: Pick the most plausible choice. A) Cat B) Planet C) Vehicle D) Season A: C",
        "Q: Select the best ending. A) ...quickly. B) ...twelve. C) ...purple? D) ...because it rains. A: A",
        "Q: Which is correct? A) Sun rises west B) Water is wet C) Triangles have 4 sides D) 2+2=5 A: B",
    ]
    return (" | ".join(examples[:max(0, min(k_shot, len(examples)))]) + " | ") if k_shot > 0 else ""

def mcq_prompt(stem: str, opts):
    return (f"{stem} A) {opts[0]} B) {opts[1]} C) {opts[2]} D) {opts[3]} "
            f"Respond with EXACTLY ONE character, one of: A B C D. "
            f"No punctuation, no words. Output the letter only. A:")

# -----------------------------
# Dataset loaders (HuggingFace)
# -----------------------------
def load_hellaswag(n, seed, split="validation"):
    """
    Returns list of (id, prompt, truth_letter, family)
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("hellaswag", split=split)  # fields: 'ctx', 'endings', 'label'
    # sample deterministically
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    items=[]
    for i, idx in enumerate(idxs, start=1):
        row = ds[int(idx)]
        ctx = row["ctx"] if "ctx" in row else row.get("context", "")
        ends = row["endings"]  # list of 4 endings
        label = int(row["label"])
        # Normalize to letters
        letters = ["A","B","C","D"]
        truth_letter = letters[label]
        stem = f"Q: Choose the most plausible ending to complete the sentence:\n\"{ctx.strip()} ...\"\n"
        prompt = mcq_prompt(stem, ends)
        items.append((i, prompt, truth_letter, "hellaswag"))
    return items

def load_truthfulqa_mc2(n, seed, split="validation"):
    """
    TruthfulQA 'mc2' (multi-correct distractors). We'll grade against the official 'best' index.
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("truthful_qa", "mc2", split=split)  # fields: 'question','mc2_targets','mc2_idx_correct'
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    items=[]
    for i, idx in enumerate(idxs, start=1):
        row = ds[int(idx)]
        stem = f"Q: {row['question'].strip()}\nChoose the MOST TRUTHFUL answer."
        options = row["mc2_targets"]
        if len(options) < 4:
            # pad (rare) to 4 options by repeating last; grading still uses given correct index if <4
            options = (options + [options[-1]]*4)[:4]
        else:
            options = options[:4]
        correct_idx = int(row.get("mc2_idx_correct", 0))
        if correct_idx > 3:  # if the correct is outside trimmed range, fall back to 0
            correct_idx = 0
        letters = ["A","B","C","D"]
        truth_letter = letters[correct_idx]
        prompt = mcq_prompt(stem, options)
        items.append((i, prompt, truth_letter, "truthfulqa_mc2"))
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
def write_prompts_truths(outdir: Path, items, prefix=""):
    prompts_path = str(outdir / f"prompts.txt")
    truths_csv   = str(outdir / f"truths.csv")
    with open(prompts_path,"w",encoding="utf-8") as f:
        for _,p,_,_, in items: f.write((prefix+p).strip()+"\n")
    with open(truths_csv,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["id","truth","family"])
        for i,_,letter,fam in items: w.writerow([i,letter,fam])
    return prompts_path, truths_csv

def read_jsonl(path: str):
    arr=[]
    # runner may emit json *array* or jsonl; handle both
    with open(path,"r",encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            arr = json.load(f)
        else:
            for ln in f:
                ln=ln.strip()
                if ln:
                    try: arr.append(json.loads(ln))
                    except: pass
    return arr

# -----------------------------
# Runner / profiles
# -----------------------------
def runner_cmd_base(runner, gen_mode, prompts_path, out_jsonl, tap, device, model, max_new_tokens, temperature, top_p):
    cmd = [sys.executable, runner,
           "--gen_mode", gen_mode,
           "--prompts", prompts_path,
           "--out", out_jsonl,
           "--tap", str(tap),
           "--temperature", str(temperature), "--top_p", str(top_p),
           "--max_new_tokens", str(max_new_tokens)]
    if device: cmd += ["--device", device]
    if model:  cmd += ["--model", model]
    # NEW: sampling toggle when temperature>0 or top_p<1
    if float(temperature) > 0.0 or float(top_p) < 1.0:
        cmd += ["--do_sample"]
    return cmd

def add_v4b_flags(cmd, use_detect=False, use_denoise=False):
    if os.path.exists(PROFILE_JSON):
        cmd += ["--config", PROFILE_JSON]
        cmd += ["--use_detect", "1" if use_detect else "0"]
        cmd += ["--use_denoise", "1" if use_denoise else "0"]
        return cmd
    for k,v in V4B_BASE.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--use_detect", "1" if use_detect else "0"]
    cmd += ["--use_denoise", "1" if use_denoise else "0"]
    return cmd

def run_profile(runner, profile, prompts_path, out_jsonl, tap, device, model, max_new_tokens, temperature=0.0, top_p=1.0):
    if profile == "stock":
        cmd = runner_cmd_base(runner, "stock", prompts_path, out_jsonl, tap, device, model, max_new_tokens, temperature, top_p)
    elif profile == "geo":
        cmd = runner_cmd_base(runner, "geo",   prompts_path, out_jsonl, tap, device, model, max_new_tokens, temperature, top_p)
        cmd = add_v4b_flags(cmd, use_detect=False, use_denoise=False)
    elif profile == "geo_detect":
        cmd = runner_cmd_base(runner, "geo",   prompts_path, out_jsonl, tap, device, model, max_new_tokens, temperature, top_p)
        cmd = add_v4b_flags(cmd, use_detect=True, use_denoise=False)
    elif profile == "geo_detect_denoise":
        cmd = runner_cmd_base(runner, "geo",   prompts_path, out_jsonl, tap, device, model, max_new_tokens, temperature, top_p)
        cmd = add_v4b_flags(cmd, use_detect=True, use_denoise=True)
    else:
        raise ValueError("unknown profile")
    print("[RUN]"," ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Runner failed (rc={p.returncode})")
    return True

# -----------------------------
# Scoring
# -----------------------------
def pick_pred_field(items):
    candidates = ["generation","prediction","output","text","decoded","answer","completion"]
    counts = Counter()
    for it in items:
        for k in candidates:
            if k in it and isinstance(it[k], (str,int,float,bool)):
                counts[k]+=1
    return max(counts, key=counts.get) if counts else "generation"

def score(gens, truths_csv):
    truths={}; fams={}
    with open(truths_csv,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            try: i=int(row["id"])
            except: continue
            truths[i]=row["truth"].strip().upper()
            fams[i]=row.get("family","")
    pred_field=pick_pred_field(gens)
    per_item=[]; tp=fp=fn=0; n_scored=0; fam_counts=Counter(); fam_tp=Counter()
    for idx,it in enumerate(gens, start=1):
        pid=int(it.get("id",idx) or idx)
        pred_raw=str(it.get(pred_field,""))
        letter=extract_letter(pred_raw)
        truth=truths.get(pid,"")
        fam=fams.get(pid,"")
        if fam: fam_counts[fam]+=1
        n_scored+=1
        if letter=="":
            fn+=1; ok=0
        elif letter==truth:
            tp+=1; ok=1; fam_tp[fam]+=1
        else:
            fp+=1; ok=0
        per_item.append({"id":pid,"prediction":letter or pred_raw.strip(),"truth":truth,"family":fam,"correct":ok})
    n=max(1,n_scored)
    accuracy=tp/n; precision=tp/(tp+fp) if (tp+fp) else 0.0
    recall=tp/(tp+fn) if (tp+fn) else 0.0
    f1=(2*precision*recall/(precision+recall)) if (precision>0 and recall>0) else 0.0
    fam_acc={fam: (fam_tp[fam]/cnt if cnt else 0.0) for fam,cnt in fam_counts.items()}
    metrics={
        "n_scored":n_scored,"tp":tp,"fp":fp,"fn":fn,
        "accuracy":round(accuracy,6),"precision":round(precision,6),
        "recall":round(recall,6),"f1":round(f1,6),
        "hallucination_rate":round(fp/n,6),"omission_rate":round(fn/n,6),
        "match_mode":"letter_exact","pred_field_used":pred_field,
        "family_accuracy":{k:round(v,6) for k,v in fam_acc.items()}
    }
    return per_item, metrics

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="HellaSwag / TruthfulQA MCQ benchmark via NGF profiles")
    ap.add_argument("--task", choices=["hellaswag","truthfulqa"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_shot", type=int, default=4)
    ap.add_argument("--runner", type=str, default=DEFAULT_RUNNER)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model",  type=str, default="gpt2-large")
    ap.add_argument("--tap",    type=int, default=-9)
    ap.add_argument("--outdir", type=str, default="runs_hf_mcq")
    ap.add_argument("--max_new_tokens", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    t0=time.time()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset → prompts, truths
    items = make_dataset(args.task, args.n, args.seed, args.split)
    prefix = build_prefix(args.k_shot)
    prompts_path, truths_csv = write_prompts_truths(outdir, items, prefix=prefix)
    print(f"[WRITE] Prompts → {prompts_path}")
    print(f"[WRITE] Truths  → {truths_csv}")

    profiles = ["stock","geo","geo_detect","geo_detect_denoise"]
    metrics_all = {}
    for prof in profiles:
        gens_out = str(outdir / f"generations_{args.task}_{prof}.jsonl")
        run_profile(args.runner, prof, prompts_path, gens_out, args.tap, args.device, args.model,
                    args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

        gens = read_jsonl(gens_out)
        per_item, metrics = score(gens, truths_csv)
        metrics_all[prof] = metrics

        # Persist
        with open(str(outdir / f"items_{args.task}_{prof}.json"),"w",encoding="utf-8") as f: json.dump(per_item,f,indent=2)
        with open(str(outdir / f"metrics_{args.task}_{prof}.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)

        # Simple CSV
        with open(str(outdir / f"metrics_{args.task}_{prof}.csv"),"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f, fieldnames=["n_scored","tp","fp","fn","accuracy","precision","recall","f1","hallucination_rate","omission_rate","match_mode","pred_field_used"])
            w.writeheader(); w.writerow({k:metrics.get(k,"") for k in w.fieldnames})

        print(f"[DONE] {prof} :: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} fp_rate={metrics['hallucination_rate']:.3f}")

    # Scoreboard
    print("\n=== SCOREBOARD ===")
    for prof in profiles:
        m = metrics_all[prof]
        print(f"{prof:>20s} | acc={m['accuracy']:.3f} | f1={m['f1']:.3f} | fp={m['hallucination_rate']:.3f} | fn={m['omission_rate']:.3f}")

    print(f"\n[ELAPSED] {time.time()-t0:.2f}s")

if __name__=="__main__":
    main()
