# Re-generate the mapper script and save it where you can download it.
from pathlib import Path

# code = r"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 11 — Well Mapper (v2)
- Robust file discovery for v1–v5 (handles custom names like *best*).
- Accepts additional metric key aliases (acc,P,R,F1,J,H,O,grid,margin_mu,margin_min,PI).
- Better logging (prints which files were found / parsed per version).
- Optional --verbose to echo subprocess return codes and discovery details.

python3 stage11-well-mapper-v2.py --mode all --run --samples 100 \
  --outdir /benchmark_data/well_map --verbose


python3 stage11-well-mapper-v2.py --mode sweep --run --samples 100 \
  --v5_path ./stage11-well-benchmark-v5.py \
  --outdir ./benchmark_data/well_map_v5 \
  --search_dirs "./,./benchmark_data/well_map_v5" \
  --verbose

"""

import argparse, json, sys, subprocess, math
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class VersionConfig:
    key: str
    script: str
    args: List[str]

def default_version_args(samples: int) -> Dict[str, VersionConfig]:
    return {
        "v1": VersionConfig("v1", "/mnt/data/stage11-well-benchmark-v1.py", ["--samples", str(samples), "--pi"]),
        "v2": VersionConfig("v2", "/mnt/data/stage11-well-benchmark-v2.py", ["--samples", str(samples), "--recall_bias", "0.80", "--pi"]),
        "v3": VersionConfig("v3", "/mnt/data/stage11-well-benchmark-v3.py", ["--samples", str(samples), "--rank_r", "1", "--res_drop_frac", "0.03", "--pi"]),
        "v4": VersionConfig("v4", "/mnt/data/stage11-well-benchmark-v4.py", ["--samples", str(samples), "--rank_r", "1",
                                                                              "--cand_floor", "0.40", "--inhib_sigma", "1.6", "--inhib_lambda", "0.35",
                                                                              "--T0", "1.6", "--Tmin", "0.7", "--anneal_steps", "2", "--p_floor", "0.18", "--pi"]),
        "v5": VersionConfig("v5", "/mnt/data/stage11-well-benchmark-v5.py", ["--samples", str(samples), "--sweep", "quick", "--pi"]),
    }

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_version_key_from_name(name: str) -> Optional[str]:
    name = name.lower()
    for v in ("v1","v2","v3","v4","v5"):
        if v in name:
            return v
    return None

def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def discover_files(search_dirs: List[Path], verbose=False) -> Dict[str, Dict[str, Path]]:
    """Return a map per version with discovered JSON/CSV summary files."""
    out = {f"v{i}": {"json": None, "csv": None} for i in range(1,6)}
    patterns_json = [
        "stage11_v*_summary.json", "*v*_summary*.json", "*stage11*_*v*.json", "*stage11*v*summary*.json"
    ]
    patterns_csv = [
        "stage11_v*.csv", "*v*.csv"
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns_json:
            for p in d.glob(pat):
                v = infer_version_key_from_name(p.name)
                if v and out[v]["json"] is None:
                    out[v]["json"] = p
        for pat in patterns_csv:
            for p in d.glob(pat):
                v = infer_version_key_from_name(p.name)
                if v and out[v]["csv"] is None:
                    out[v]["csv"] = p
    if verbose:
        for v, m in out.items():
            print(f"[DISCOVER] {v}: json={m['json']} csv={m['csv']}")
    return out

ALIASES = {
    "accuracy": "accuracy_exact",
    "acc": "accuracy_exact",
    "accuracy_exact": "accuracy_exact",
    "grid": "grid_similarity",
    "grid_similarity": "grid_similarity",
    "precision": "precision",
    "P": "precision",
    "recall": "recall",
    "R": "recall",
    "f1": "f1",
    "F1": "f1",
    "jaccard": "jaccard",
    "J": "jaccard",
    "hallucination": "hallucination_rate",
    "H": "hallucination_rate",
    "hallucination_rate": "hallucination_rate",
    "omission": "omission_rate",
    "O": "omission_rate",
    "omission_rate": "omission_rate",
    "margin_mean": "margin_mean",
    "margin_mu": "margin_mean",
    "margin_min": "margin_min",
    "phantom_index": "phantom_index",
    "PI": "phantom_index",
    "pi": "phantom_index",
    "runtime_mean": "runtime_mean"
}

def metrics_from_csv(path: Path) -> Optional[dict]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    candidates = {
        "accuracy_exact": ["geodesic_ok","accuracy","acc"],
        "grid_similarity": ["geodesic_grid","grid"],
        "precision": ["geodesic_precision","precision","P"],
        "recall": ["geodesic_recall","recall","R"],
        "f1": ["geodesic_f1","f1","F1"],
        "jaccard": ["geodesic_jaccard","jaccard","J"],
        "hallucination_rate": ["geodesic_hallucination","hallucination_rate","hallucination","H"],
        "omission_rate": ["geodesic_omission","omission_rate","omission","O"],
        "margin_mean": ["margin_mean","margin_mu"],
        "margin_min": ["margin_min"],
        "runtime_mean": ["geodesic_time","runtime"]
    }
    res = {}
    for k, cols in candidates.items():
        for c in cols:
            if c in df.columns:
                val = pd.to_numeric(df[c], errors="coerce")
                res[k] = float(val.mean())
                break
    if "accuracy_exact" in res and ("geodesic_ok" in df.columns):
        ok = pd.to_numeric(df["geodesic_ok"], errors="coerce").fillna(0.0)
        res["accuracy_exact"] = float((ok > 0.5).mean())
    return res if res else None

def normalize_metrics(d: dict) -> dict:
    norm = {}
    for k, v in d.items():
        k2 = ALIASES.get(k, k)
        try:
            norm[k2] = float(v)
        except Exception:
            pass
    if "phantom_index" not in norm:
        h = norm.get("hallucination_rate", math.nan)
        r = norm.get("recall", math.nan)
        if not math.isnan(h) and not math.isnan(r):
            norm["phantom_index"] = 0.7*h + 0.3*(1.0 - r)
        elif not math.isnan(h):
            norm["phantom_index"] = h
    return norm

def harvest_version_metrics(vkey: str, paths: Dict[str, Optional[Path]], verbose=False) -> Optional[dict]:
    data = None
    if paths.get("json"):
        data = load_json(paths["json"])
        if verbose: print(f"[HARVEST:{vkey}] JSON={paths['json']} -> {'ok' if data else 'fail'}")
    if data is None and paths.get("csv"):
        data = metrics_from_csv(paths["csv"])
        if verbose: print(f"[HARVEST:{vkey}] CSV={paths['csv']} -> {'ok' if data else 'fail'}")
    if data is None:
        return None
    norm = normalize_metrics(data)
    norm["version"] = vkey
    return norm

def run_version(vconf: VersionConfig, verbose=False) -> int:
    cmd = [sys.executable, vconf.script] + vconf.args
    if verbose: print("[RUN]", " ".join(cmd))
    try:
        rc = subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        if verbose: print(f"[RUN:{vconf.key}] script not found: {vconf.script}")
        return 127
    if verbose: print(f"[RUN:{vconf.key}] rc={rc}")
    return rc

def plot_metrics_bar(df: pd.DataFrame, out_png: Path):
    cols = ["accuracy_exact","precision","recall","f1","jaccard","hallucination_rate"]
    present = [c for c in cols if c in df.columns]
    if not present: return
    x = np.arange(len(df))
    fig = plt.figure(figsize=(10,5))
    width = 0.12
    for i,c in enumerate(present):
        plt.bar(x + (i - len(present)/2)*width + width/2, df[c].values, width, label=c)
    plt.xticks(x, df["version"].tolist())
    plt.ylabel("value"); plt.title("Stage 11 — metrics by version"); plt.legend(loc="best"); plt.tight_layout()
    fig.savefig(out_png, dpi=120); plt.close(fig)

def plot_recall_vs_halluc(df: pd.DataFrame, out_png: Path):
    if "recall" not in df.columns or "hallucination_rate" not in df.columns: return
    fig = plt.figure(figsize=(6,5))
    plt.scatter(df["recall"], df["hallucination_rate"])
    for _, row in df.iterrows():
        plt.annotate(row["version"], (row["recall"], row["hallucination_rate"]))
    plt.xlabel("Recall"); plt.ylabel("Hallucination rate"); plt.title("Recall vs Hallucination"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def plot_margins(df: pd.DataFrame, out_png: Path):
    if "margin_mean" not in df.columns and "margin_min" not in df.columns: return
    x = np.arange(len(df)); width = 0.35
    fig = plt.figure(figsize=(8,5))
    if "margin_mean" in df.columns: plt.bar(x - width/2, df["margin_mean"], width, label="margin_mean")
    if "margin_min" in df.columns:  plt.bar(x + width/2, df["margin_min"],  width, label="margin_min")
    plt.xticks(x, df["version"].tolist()); plt.ylabel("margin"); plt.title("Well depth margins (higher is deeper/safer)")
    plt.legend(loc="best"); plt.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description="Stage 11 — Well Mapper v2 (robust harvesting + logging)")
    ap.add_argument("--mode", type=str, default="all", choices=["baseline","recall","ortho","inhibit","sweep","all"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="./wellmap_out")
    ap.add_argument("--search_dirs", type=str, default="/mnt/data,.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--v1_path", type=str, default="/mnt/data/stage11-well-benchmark-v1.py")
    ap.add_argument("--v2_path", type=str, default="/mnt/data/stage11-well-benchmark-v2.py")
    ap.add_argument("--v3_path", type=str, default="/mnt/data/stage11-well-benchmark-v3.py")
    ap.add_argument("--v4_path", type=str, default="/mnt/data/stage11-well-benchmark-v4.py")
    ap.add_argument("--v5_path", type=str, default="/mnt/data/stage11-well-benchmark-v5.py")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    mode2vers = {"baseline":["v1"],"recall":["v2"],"ortho":["v3"],"inhibit":["v4"],"sweep":["v5"],"all":["v1","v2","v3","v4","v5"]}
    todo = mode2vers[args.mode]

    vcfg = default_version_args(args.samples)
    vcfg["v1"].script=args.v1_path; vcfg["v2"].script=args.v2_path; vcfg["v3"].script=args.v3_path; vcfg["v4"].script=args.v4_path; vcfg["v5"].script=args.v5_path

    if args.run:
        for v in todo:
            rc = run_version(vcfg[v], verbose=args.verbose)
            if rc != 0 and args.verbose:
                print(f"[WARN] Underlying script for {v} returned rc={rc}")

    search_dirs = [outdir] + [Path(s.strip()) for s in args.search_dirs.split(",") if s.strip()]
    summ_map = discover_files(search_dirs, verbose=args.verbose)

    rows = []
    for v in todo:
        m = harvest_version_metrics(v, summ_map.get(v, {}), verbose=args.verbose)
        if not m:
            guess_json = Path(f"/mnt/data/stage11_{v}_summary.json")
            guess_csv  = Path(f"/mnt/data/stage11_{v}.csv")
            m = harvest_version_metrics(v, {"json": guess_json if guess_json.exists() else None,
                                            "csv":  guess_csv if guess_csv.exists() else None},
                                            verbose=args.verbose)
        if m: rows.append(m)
        else: rows.append({"version": v})

    df = pd.DataFrame(rows).sort_values("version")
    if "recall" in df.columns: df["guard_recall_OK"] = df["recall"] >= 0.90
    if "hallucination_rate" in df.columns: df["guard_halluc_OK"] = df["hallucination_rate"] <= 0.26

    out_csv = outdir / "wellmap_report.csv"
    out_json = outdir / "wellmap_report.json"
    df.to_csv(out_csv, index=False); df.to_json(out_json, orient="records", indent=2)
    print(f"[WROTE] {out_csv}"); print(f"[WROTE] {out_json}")

    plot_metrics_bar(df, outdir / "fig_metrics.png")
    plot_recall_vs_halluc(df, outdir / "fig_recall_vs_halluc.png")
    plot_margins(df, outdir / "fig_margins.png")
    print("[PLOTS] fig_metrics.png | fig_recall_vs_halluc.png | fig_margins.png")

if __name__ == "__main__":
    main()
# """

# path = Path("/mnt/data/stage11-well-mapper_v2.py")
# path.write_text(code)

# print(str(path))
