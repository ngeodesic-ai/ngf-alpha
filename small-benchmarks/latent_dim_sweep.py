# Create a sweep helper script that runs stage11_benchmark over multiple latent_dim values,
# collects metrics into a CSV, and produces a simple plot.
import os, json, subprocess, shlex, sys, textwrap
from pathlib import Path

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latent_dim_sweep.py
Run stage11 benchmark at multiple latent dimensions, collect metrics, and plot curves.
Example:
  python3 latent_dim_sweep.py \
    --samples 100 --seed 42 --latent_arc_noise 0.05 \
    --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
    --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
    --seed_jitter 2 --log INFO \
    --dims 16,32,64,128,256,512,768 \
    --stage11_path ./stage11_benchmark_latest.py \
    --out_dir results_sweep
"""
import argparse, os, sys, json, subprocess
from pathlib import Path

def find_stage11(path_hint: str) -> str:
    if path_hint and Path(path_hint).exists():
        return path_hint
    # fallbacks commonly used in this repo
    cands = [
        "stage11_benchmark_latest.py",
        "stage11-benchmark-latest.py",
        "./stage11_benchmark_latest.py",
        "./stage11-benchmark-latest.py",
    ]
    for c in cands:
        if Path(c).exists():
            return c
    # final fallback: search in working dir
    for p in Path(".").rglob("stage11*_benchmark*latest.py"):
        return str(p)
    raise FileNotFoundError("Could not locate stage11 benchmark script; pass --stage11_path path/to/script.py")

def run_one(stage11, base_args, dim, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"latent_arc_denoise_dim{dim}.json"
    cpath = out_dir / f"latent_arc_denoise_dim{dim}.csv"

    args = [
        "python3", "-u", stage11,
        "--samples", str(base_args.samples),
        "--seed", str(base_args.seed),
        "--latent_arc",
        "--latent_dim", str(dim),
        "--latent_arc_noise", str(base_args.latent_arc_noise),
        "--denoise_mode", base_args.denoise_mode,
        "--ema_decay", str(base_args.ema_decay),
        "--median_k", str(base_args.median_k),
        "--probe_k", str(base_args.probe_k),
        "--probe_eps", str(base_args.probe_eps),
        "--conf_gate", str(base_args.conf_gate),
        "--noise_floor", str(base_args.noise_floor),
        "--seed_jitter", str(base_args.seed_jitter),
        "--log", base_args.log,
        "--out_json", str(jpath),
        "--out_csv", str(cpath),
    ]
    print(f"[sweep] running dim={dim} → {jpath.name}")
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"stage11 run failed for dim={dim} (rc={proc.returncode})")

    # Parse summary from JSON (if present)
    metrics = {
        "dim": dim,
        "accuracy_exact": None,
        "f1": None,
        "jaccard": None,
        "hallucination_rate": None,
        "precision": None,
        "recall": None,
        "omission_rate": None,
    }
    try:
        with open(jpath, "r") as f:
            data = json.load(f)
        # Try common keys
        summary = data.get("summary") or data.get("SUMMARY") or data
        # direct fields
        for k in ["accuracy_exact","f1","jaccard","hallucination_rate","precision","recall","omission_rate"]:
            if k in summary:
                metrics[k] = summary[k]
        # or nested (e.g., "Geodesic (denoise path)")
        if metrics["f1"] is None:
            for key, val in summary.items():
                if isinstance(val, dict) and "f1" in val:
                    metrics["f1"] = val.get("f1")
                    metrics["accuracy_exact"] = val.get("accuracy_exact", metrics["accuracy_exact"])
                    metrics["jaccard"] = val.get("jaccard", metrics["jaccard"])
                    metrics["hallucination_rate"] = val.get("hallucination_rate", metrics["hallucination_rate"])
                    metrics["precision"] = val.get("precision", metrics["precision"])
                    metrics["recall"] = val.get("recall", metrics["recall"])
                    metrics["omission_rate"] = val.get("omission_rate", metrics["omission_rate"])
                    break
    except Exception as e:
        print(f"[sweep] warning: could not parse JSON metrics for dim={dim}: {e}")

    return metrics, jpath, cpath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent_arc_noise", type=float, default=0.05)
    ap.add_argument("--denoise_mode", type=str, default="hybrid")
    ap.add_argument("--ema_decay", type=float, default=0.85)
    ap.add_argument("--median_k", type=int, default=3)
    ap.add_argument("--probe_k", type=int, default=5)
    ap.add_argument("--probe_eps", type=float, default=0.02)
    ap.add_argument("--conf_gate", type=float, default=0.65)
    ap.add_argument("--noise_floor", type=float, default=0.03)
    ap.add_argument("--seed_jitter", type=int, default=2)
    ap.add_argument("--log", type=str, default="INFO")
    ap.add_argument("--dims", type=str, default="16,32,64,128,256,512,768")
    ap.add_argument("--stage11_path", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="sweep_results")
    ap.add_argument("--plot", action="store_true", help="Produce a matplotlib plot of F1 and hallucination vs dim")
    args = ap.parse_args()

    stage11 = find_stage11(args.stage11_path)
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for d in dims:
        row, j, c = run_one(stage11, args, d, out_dir)
        all_rows.append(row)

    # write combined CSV
    csv_path = out_dir / "sweep_summary.csv"
    headers = ["dim","f1","jaccard","accuracy_exact","precision","recall","hallucination_rate","omission_rate"]
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in all_rows:
            f.write(",".join(str(r.get(h,"")) for h in headers) + "\n")
    print(f"[sweep] wrote summary CSV → {csv_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            dims_sorted = sorted(all_rows, key=lambda r: r["dim"])
            xs = [r["dim"] for r in dims_sorted]
            f1s = [r["f1"] for r in dims_sorted]
            halls = [r["hallucination_rate"] for r in dims_sorted]

            plt.figure()
            plt.plot(xs, f1s, marker="o", label="F1")
            plt.xlabel("latent_dim")
            plt.ylabel("F1")
            plt.title("F1 vs latent_dim")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "f1_vs_dim.png", dpi=160)

            plt.figure()
            plt.plot(xs, halls, marker="o", label="hallucination_rate")
            plt.xlabel("latent_dim")
            plt.ylabel("hallucination_rate")
            plt.title("Hallucination rate vs latent_dim")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "hallucination_vs_dim.png", dpi=160)

            print(f"[sweep] wrote plots → {out_dir/'f1_vs_dim.png'} and {out_dir/'hallucination_vs_dim.png'}")
        except Exception as e:
            print(f"[sweep] plotting skipped: {e}")

if __name__ == "__main__":
    main()

