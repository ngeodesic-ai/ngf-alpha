#!/usr/bin/env python3
import json, csv, os, re, time, argparse, subprocess, sys
from collections import Counter
from pathlib import Path

RUNNER = "/mnt/data/text_arc_unified.py"

def norm(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z]+', ' ', s or '')
    return ' '.join(s.lower().strip().split())

def read_truths(path: str):
    if not path or not os.path.exists(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    idcol = cols.get('id') or list(df.columns)[0]
    truthcol = cols.get('truth') or cols.get('answer') or cols.get('gold') or list(df.columns)[1]
    truths = {}
    for _, row in df.iterrows():
        try:
            i = int(row[idcol])
        except Exception:
            continue
        truths[i] = str(row[truthcol])
    return truths

def read_generations_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                items.append(obj)
            except Exception:
                pass
    return items

def text_stats(text: str):
    text = (text or "").strip()
    toks = text.split()
    n = len(toks)
    if n <= 1:
        return {"len_words": n, "adj_dup": 0.0, "tri_rep": 0.0, "uniq_ratio": 1.0, "loopish": 0}
    dup = sum(1 for i in range(1,n) if toks[i]==toks[i-1]) / (n-1)
    trigrams = [" ".join(toks[i:i+3]) for i in range(n-2)]
    tri_counts = Counter(trigrams)
    tri_rep = (sum(c-1 for c in tri_counts.values() if c>1) / max(1,len(trigrams))) if trigrams else 0.0
    uniq_ratio = len(set(toks)) / n
    tok_counts = Counter(toks)
    loopish = int(any(c>=6 for c in tok_counts.values()))
    return {"len_words": n, "adj_dup": round(dup,4), "tri_rep": round(tri_rep,4), "uniq_ratio": round(uniq_ratio,4), "loopish": loopish}

def main():
    ap = argparse.ArgumentParser(description="ARC task benchmark harness (stage11-benchmark-like IO)")
    ap.add_argument("--mode", type=str, default="stock", choices=["stock","geo"], help="Decode mode")
    ap.add_argument("--tap", type=int, default=-9, help="Tap layer metadata")
    ap.add_argument("--prompts", type=str, required=True, help="Path to prompts (one per line)")
    ap.add_argument("--profile", type=str, default=None, help="Geo profile (JSON)")
    ap.add_argument("--truths", type=str, default=None, help="Optional truths CSV (id,truth) for accuracy")
    ap.add_argument("--runner_metrics_json", type=str, default=None, help="Optional metrics JSON from the runner")
    ap.add_argument("--out_json", type=str, required=True, help="Summary JSON output (stage11-benchmark-like schema)")
    ap.add_argument("--out_csv",  type=str, default=None, help="Optional per-item CSV output")
    ap.add_argument("--gen_jsonl", type=str, default=None, help="(Advanced) Provide an existing JSONL to score instead of running")
    ap.add_argument("--label", type=str, default=None, help="Optional run label")
    args = ap.parse_args()

    t0 = time.time()
    tmp_gen = args.gen_jsonl or (str(Path(args.out_json).with_suffix(".jsonl")))

    # Run the generator if not provided
    if args.gen_jsonl is None:
        cmd = [sys.executable, RUNNER, "--prompts", args.prompts, "--out", tmp_gen, "--tap", str(args.tap)]
        if args.mode == "geo":
            if args.profile:
                cmd.extend(["--config", args.profile])
        else:
            cmd.extend(["--gen_mode", "stock"])
        # capture a metrics json if requested
        if args.runner_metrics_json:
            cmd.extend(["--metrics_json", args.runner_metrics_json])
        print("[RUN]", " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr, file=sys.stderr)
            sys.exit(p.returncode)

    # Load generations
    items = read_generations_jsonl(tmp_gen)

    # Load truths for accuracy (optional)
    truths = read_truths(args.truths) if args.truths else {}

    # Compose per-item records
    per_item = []
    correct = 0
    for obj in items:
        pid = int(obj.get("id", 0) or 0)
        prompt = str(obj.get("prompt",""))
        gen = str(obj.get("generation","")).strip()
        rec = {
            "id": pid,
            "prompt": prompt,
            "prediction": gen,
            "mode": args.mode,
            "tap": args.tap,
        }
        # quality proxies
        rec.update(text_stats(gen))
        # accuracy if truths present
        if pid in truths:
            ok = int(norm(truths[pid]) == norm(gen) and truths[pid].strip() != "")
            rec["truth"] = truths[pid]
            rec["correct"] = ok
            correct += ok
        per_item.append(rec)

    # Aggregate summary
    elapsed = time.time() - t0
    total = len(per_item)
    summary = {
        "label": args.label or f"{args.mode}_tap{args.tap}",
        "mode": args.mode,
        "tap": args.tap,
        "prompts_path": args.prompts,
        "total": total,
        "correct": int(correct) if truths else None,
        "accuracy": (correct / total) if truths and total else None,
        "elapsed_sec": elapsed,
        "runner_metrics_json": args.runner_metrics_json,
        "items": per_item[:10]  # preview in the JSON (full list can be large)
    }

    # Write JSON summary (benchmark-like)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {args.out_json}")

    # Optional per-item CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        keys = ["id","prompt","prediction","mode","tap","len_words","adj_dup","tri_rep","uniq_ratio","loopish"]
        if truths:
            keys += ["truth","correct"]
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_item:
                w.writerow({k: r.get(k) for k in keys})
        print(f"[CSV]  {args.out_csv}")

if __name__ == "__main__":
    main()
