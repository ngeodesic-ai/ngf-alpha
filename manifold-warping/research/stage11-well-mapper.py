#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 11 - Well Mapper
======================
Driver/aggregator for the NGeodesic Stage 11 "well mapping" plan.

Features
- Runs one or more benchmark variants (v1…v5) OR just aggregates existing outputs.
- Harvests *_summary.json and/or CSVs (v1…v5), consolidates key metrics.
- Computes a Phantom Index (PI) proxy and guardrail checks (Recall>=0.90, Halluc<=0.26).
- Produces comparison plots (metrics bars; recall vs hallucination; margins if available).
- Optional manifold dump visual (if *.npz provided by underlying scripts).

Inputs (expected defaults; override via --search_dirs)
  /mnt/data/stage11_v1_summary.json  (and/or /mnt/data/stage11_v1.csv)
  /mnt/data/stage11_v2_summary.json  (and/or /mnt/data/stage11_v2.csv)
  /mnt/data/stage11_v3_summary.json  (and/or /mnt/data/stage11_v3.csv)
  /mnt/data/stage11_v4_summary.json  (and/or /mnt/data/stage11_v4.csv)
  /mnt/data/stage11_v5_summary.json  (and/or /mnt/data/stage11_v5.csv)

Underlying benchmark scripts (if --run is used; override via --v*_path)
  /mnt/data/stage11-well-benchmark-v1.py
  /mnt/data/stage11-well-benchmark-v2.py
  /mnt/data/stage11-well-benchmark-v3.py
  /mnt/data/stage11-well-benchmark-v4.py
  /mnt/data/stage11-well-benchmark-v5.py

Usage examples
  # Aggregate only (no running); write report + plots
  python3 stage11-well-mapper.py --mode all --outdir benchmark_data/well_map

  # Run recommended recipe for "inhibit" (v4), then aggregate everything found
  python3 stage11-well-mapper.py --mode inhibit --run --samples 50 --outdir benchmark_data/well_map

  # Run v5 mini sweep, lock an operating point, then aggregate
  python3 stage11-well-mapper.py --mode sweep --run --samples 50 --outdir benchmark_data/well_map
"""

import argparse, json, sys, subprocess, shutil, math, time, textwrap
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Config -----------------------------

@dataclass
class VersionConfig:
    key: str
    script: str
    args: List[str]

# Recommended default flags per version (conservative, recall-friendly, yet precise)
def default_version_args(samples: int) -> Dict[str, VersionConfig]:
    return {
        "v1": VersionConfig("v1", "/mnt/data/stage11-well-benchmark-v1.py",
            ["--samples", str(samples), "--pi"]),
        "v2": VersionConfig("v2", "/mnt/data/stage11-well-benchmark-v2.py",
            ["--samples", str(samples), "--recall_bias", "0.80", "--pi"]),
        "v3": VersionConfig("v3", "/mnt/data/stage11-well-benchmark-v3.py",
            ["--samples", str(samples), "--rank_r", "1", "--res_drop_frac", "0.03", "--pi"]),
        "v4": VersionConfig("v4", "/mnt/data/stage11-well-benchmark-v4.py",
            ["--samples", str(samples), "--rank_r", "1",
             "--cand_floor", "0.40", "--inhib_sigma", "1.6", "--inhib_lambda", "0.35",
             "--T0", "1.6", "--Tmin", "0.7", "--anneal_steps", "2", "--p_floor", "0.18", "--pi"]),
        "v5": VersionConfig("v5", "/mnt/data/stage11-well-benchmark-v5.py",
            ["--samples", str(samples), "--sweep", "quick", "--pi"]),
    }

# ----------------------- Utilities -----------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def which_file(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None

def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def infer_version_key_from_path(p: Path) -> Optional[str]:
    name = p.name.lower()
    for v in ("v1","v2","v3","v4","v5"):
        if v in name:
            return v
    return None

# ----------------------- Data Harvest -----------------------

def find_summaries(search_dirs: List[Path]) -> Dict[str, Dict[str, Path]]:
    """Return mapping: version -> {'json': path or None, 'csv': path or None}"""
    out = {f"v{i}": {"json": None, "csv": None} for i in range(1,6)}
    for d in search_dirs:
        if not d.exists(): continue
        for p in d.glob("stage11_v*_summary.json"):
            v = infer_version_key_from_path(p)
            if v and out[v]["json"] is None: out[v]["json"] = p
        for p in d.glob("stage11_v*.csv"):
            v = infer_version_key_from_path(p)
            if v and out[v]["csv"] is None: out[v]["csv"] = p
    return out

def metrics_from_csv(path: Path) -> Optional[dict]:
    """
    Try to infer key metrics from a CSV like the Stage-10/11 formats.
    We look for geodesic_* columns and aggregate means.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Candidate columns (will be averaged if present)
    col_map = {
        "accuracy_exact": ["geodesic_ok"],  # treat mean(geodesic_ok) as accuracy
        "grid_similarity": ["geodesic_grid"],
        "precision": ["geodesic_precision"],
        "recall": ["geodesic_recall"],
        "f1": ["geodesic_f1"],
        "jaccard": ["geodesic_jaccard"],
        "hallucination_rate": ["geodesic_hallucination"],
        "omission_rate": ["geodesic_omission"],
        # Optional margins if present
        "margin_mean": ["margin_mean", "geodesic_margin_mean"],
        "margin_min": ["margin_min", "geodesic_margin_min"],
        # Optional speed
        "runtime_mean": ["geodesic_time", "runtime"],
    }

    result = {}
    for k, cands in col_map.items():
        for c in cands:
            if c in df.columns:
                val = float(pd.to_numeric(df[c], errors="coerce").mean())
                result[k] = val
                break

    # Normalize accuracy if it was mean of {0,1}
    if "accuracy_exact" in result:
        acc = result["accuracy_exact"]
        if not (0.0 <= acc <= 1.0):
            ok_series = df.get("geodesic_ok", None)
            if ok_series is not None:
                ok = pd.to_numeric(ok_series, errors="coerce").fillna(0.0)
                if len(ok) > 0:
                    result["accuracy_exact"] = float((ok > 0.5).mean())

    return result if result else None

def harvest_version_metrics(vkey: str, paths: Dict[str, Optional[Path]]) -> Optional[dict]:
    """Prefer JSON summary; fallback to CSV aggregation."""
    data = None
    if paths.get("json"):
        data = load_json(paths["json"])
    if data is None and paths.get("csv"):
        data = metrics_from_csv(paths["csv"])
    if data is None:
        return None

    # Normalize keys to a common schema
    norm = {}
    key_aliases = {
        "accuracy": "accuracy_exact",
        "accuracy_exact": "accuracy_exact",
        "grid_similarity": "grid_similarity",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "jaccard": "jaccard",
        "hallucination": "hallucination_rate",
        "hallucination_rate": "hallucination_rate",
        "omission": "omission_rate",
        "omission_rate": "omission_rate",
        "margin_mean": "margin_mean",
        "margin_min": "margin_min",
        "phantom_index": "phantom_index",
        "pi": "phantom_index",
        "runtime_mean": "runtime_mean",
    }
    for k,v in data.items():
        k2 = key_aliases.get(k, k)
        try:
            norm[k2] = float(v)
        except Exception:
            pass

    # Derived PI proxy if missing
    if "phantom_index" not in norm:
        h = norm.get("hallucination_rate", np.nan)
        r = norm.get("recall", np.nan)
        if not math.isnan(h) and not math.isnan(r):
            norm["phantom_index"] = float(0.7*h + 0.3*(1.0 - r))
        elif not math.isnan(h):
            norm["phantom_index"] = float(h)

    norm["version"] = vkey
    return norm

# ----------------------- Runner -----------------------

def run_version(vconf: VersionConfig, extra_env=None) -> int:
    """Execute the given version script with its args. Returns subprocess return code."""
    cmd = [sys.executable, vconf.script] + vconf.args
    try:
        print(f"[RUN] {' '.join(cmd)}")
        rc = subprocess.run(cmd, check=False).returncode
        print(f"[RUN] {vconf.key} -> rc={rc}")
        return rc
    except FileNotFoundError:
        print(f"[WARN] Script not found for {vconf.key}: {vconf.script}")
        return 127

# ----------------------- Plotting -----------------------

def plot_metrics_bar(df: pd.DataFrame, out_png: Path):
    cols = ["accuracy_exact", "precision", "recall", "f1", "jaccard", "hallucination_rate"]
    present = [c for c in cols if c in df.columns]
    if not present: return
    x = np.arange(len(df))
    fig = plt.figure(figsize=(10, 5))
    width = 0.12
    for i, c in enumerate(present):
        y = df[c].values
        plt.bar(x + (i - len(present)/2)*width + width/2, y, width, label=c)
    plt.xticks(x, df["version"].tolist())
    plt.ylabel("value")
    plt.title("Stage 11 - metrics by version")
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

def plot_recall_vs_halluc(df: pd.DataFrame, out_png: Path):
    if "recall" not in df.columns or "hallucination_rate" not in df.columns: return
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(df["recall"].values, df["hallucination_rate"].values)
    for _, row in df.iterrows():
        plt.annotate(row["version"], (row["recall"], row["hallucination_rate"]))
    plt.xlabel("Recall")
    plt.ylabel("Hallucination rate")
    plt.title("Recall vs Hallucination (lower is better on y)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

def plot_margins(df: pd.DataFrame, out_png: Path):
    if "margin_mean" not in df.columns and "margin_min" not in df.columns: return
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    if "margin_mean" in df.columns:
        plt.bar(x - width/2, df["margin_mean"].values, width, label="margin_mean")
    if "margin_min" in df.columns:
        plt.bar(x + width/2, df["margin_min"].values, width, label="margin_min")
    plt.xticks(x, df["version"].tolist())
    plt.ylabel("margin")
    plt.title("Well depth margins (higher is deeper/safer)")
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Stage 11 - Well Mapper (runner + aggregator + plots)"
    )
    ap.add_argument("--mode", type=str, default="all",
                    choices=["baseline","recall","ortho","inhibit","sweep","all"],
                    help=textwrap.dedent("""
                        baseline -> v1
                        recall   -> v2
                        ortho    -> v3
                        inhibit  -> v4
                        sweep    -> v5
                        all      -> v1..v5
                    """))
    ap.add_argument("--run", action="store_true", help="Run underlying benchmark script(s) for the chosen mode(s)")
    ap.add_argument("--samples", type=int, default=50, help="Samples to pass to underlying scripts when --run")
    ap.add_argument("--outdir", type=str, default="./wellmap_out", help="Output dir for consolidated report/plots")
    ap.add_argument("--search_dirs", type=str, default="/mnt/data,.",
                    help="Comma-separated dirs to search for summaries/CSVs (in addition to outdir)")
    # Allow overriding script paths
    ap.add_argument("--v1_path", type=str, default="/mnt/data/stage11-well-benchmark-v1.py")
    ap.add_argument("--v2_path", type=str, default="/mnt/data/stage11-well-benchmark-v2.py")
    ap.add_argument("--v3_path", type=str, default="/mnt/data/stage11-well-benchmark-v3.py")
    ap.add_argument("--v4_path", type=str, default="/mnt/data/stage11-well-benchmark-v4.py")
    ap.add_argument("--v5_path", type=str, default="/mnt/data/stage11-well-benchmark-v5.py")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Map mode -> versions to process
    mode2vers = {
        "baseline": ["v1"],
        "recall":   ["v2"],
        "ortho":    ["v3"],
        "inhibit":  ["v4"],
        "sweep":    ["v5"],
        "all":      ["v1","v2","v3","v4","v5"],
    }
    todo = mode2vers[args.mode]

    # Build default version configs and apply path overrides
    vcfg = default_version_args(args.samples)
    vcfg["v1"].script = args.v1_path
    vcfg["v2"].script = args.v2_path
    vcfg["v3"].script = args.v3_path
    vcfg["v4"].script = args.v4_path
    vcfg["v5"].script = args.v5_path

    # Optional run
    if args.run:
        for v in todo:
            run_version(vcfg[v])

    # Harvest
    search_dirs = [outdir] + [Path(s.strip()) for s in args.search_dirs.split(",") if s.strip()]
    summ_map = find_summaries(search_dirs)

    rows = []
    for v in todo:
        paths = summ_map.get(v, {})
        m = harvest_version_metrics(v, paths)
        if m:
            rows.append(m)
        else:
            # Try default locations if not found via search map
            guess_json = Path(f"/mnt/data/stage11_{v}_summary.json")
            guess_csv  = Path(f"/mnt/data/stage11_{v}.csv")
            m2 = harvest_version_metrics(v, {"json": guess_json if guess_json.exists() else None,
                                             "csv":  guess_csv if guess_csv.exists() else None})
            if m2:
                rows.append(m2)

    if not rows:
        print("[ERROR] No summaries/CSVs found. Provide *_summary.json or CSV for the selected mode(s).", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows).sort_values("version")
    # Guardrail checks (where present)
    if "recall" in df.columns:
        df["guard_recall_OK"] = df["recall"] >= 0.90
    else:
        df["guard_recall_OK"] = np.nan
    if "hallucination_rate" in df.columns:
        df["guard_halluc_OK"] = df["hallucination_rate"] <= 0.26
    else:
        df["guard_halluc_OK"] = np.nan

    # Save consolidated
    out_csv = outdir / "wellmap_report.csv"
    out_json = outdir / "wellmap_report.json"
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)
    print(f"[WROTE] {out_csv}")
    print(f"[WROTE] {out_json}")

    # Plots
    plot_metrics_bar(df, outdir / "fig_metrics.png")
    plot_recall_vs_halluc(df, outdir / "fig_recall_vs_halluc.png")
    plot_margins(df, outdir / "fig_margins.png")
    print(f"[PLOTS] fig_metrics.png | fig_recall_vs_halluc.png | fig_margins.png")

if __name__ == "__main__":
    main()
