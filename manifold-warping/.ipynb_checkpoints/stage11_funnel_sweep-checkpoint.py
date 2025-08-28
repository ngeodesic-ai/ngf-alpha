#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Stage 11 — Funnel Prior Sweep
# Grids key prior parameters for `stage11-well-benchmark-latest-funnel.py`,
# runs each combo, collects metrics from the JSON summary, and writes a
# compact CSV. Prints the 'best' configuration according to:
#   (1) minimize hallucination_rate
#   (2) subject to recall >= recall_floor (default 0.98)
#   (3) break ties by maximizing F1
#
# Usage (example):
#   python3 stage11_funnel_sweep.py #     --bench ./stage11-well-benchmark-latest-funnel.py #     --samples 200 --seed 42 #     --grid alpha=0.03,0.05,0.08 #            beta_s=0.15,0.25,0.35 #            q_s=2,3 #            tau_rel=0.55,0.60,0.65 #            tau_abs_q=0.90,0.93,0.95 #            null_K=40,80 #     --fixed T=720 sigma=9 proto_width=160 cm_amp=0.02 overlap=0.5 amp_jitter=0.4 distractor_prob=0.4 #     --out sweep_summary.csv
#
# Notes:
# - You can add any number of key=value1,value2,... pairs to --grid.
# - Anything passed via --fixed (or --samples/--seed) is plumbed straight to the benchmark.
# - The sweep runs with --use_funnel_prior 1 automatically.

import argparse, json, csv, itertools, subprocess, shlex, sys
from pathlib import Path

def parse_kv_list(pairs):
    cfg = {}
    for item in pairs:
        if '=' not in item:
            raise ValueError(f'Bad --grid/--fixed item: {item}')
        k, vs = item.split('=', 1)
        vals = [v.strip() for v in vs.split(',') if v.strip()]
        # grid: keep list; fixed: collapse singletons
        cfg[k.strip()] = vals if len(vals) > 1 else vals[0]
    return cfg

def product_dicts(d):
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in itertools.product(*vals):
        out = dict(zip(keys, combo))
        yield out

def run_once(bench, args_fixed, args_grid, run_idx):
    # build output names
    out_csv = f'_tmp_sweep_{run_idx:04d}.csv'
    out_json = f'_tmp_sweep_{run_idx:04d}.json'
    # assemble command
    cmd_parts = [sys.executable, str(bench)]
    # defaults
    cmd_parts += ['--samples', str(args_fixed.get('samples', 200))]
    cmd_parts += ['--seed', str(args_fixed.get('seed', 42))]
    # required fixed toggles
    cmd_parts += ['--use_funnel_prior', '1']
    # fixed passthroughs
    for k, v in args_fixed.items():
        if k in ('samples','seed'):
            continue
        cmd_parts += [f'--{k}', str(v)]
    # grid params (strings are fine; benchmark parses types)
    for k, v in args_grid.items():
        cmd_parts += [f'--{k}', str(v)]
    # outputs
    cmd_parts += ['--out_csv', out_csv, '--out_json', out_json]
    # run
    print('[RUN]', ' '.join(shlex.quote(c) for c in cmd_parts))
    proc = subprocess.run(cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    # parse json
    J = json.load(open(out_json, 'r'))
    # extract metrics
    G = J.get('geodesic', {})
    S = J.get('stock', {})
    row = dict(
        **{f'g_{k}': G.get(k) for k in ('accuracy_exact','precision','recall','f1','jaccard','hallucination_rate','omission_rate')},
        **{f's_{k}': S.get(k) for k in ('accuracy_exact','precision','recall','f1','jaccard','hallucination_rate','omission_rate')},
        phantom_index=J.get('phantom_index'),
        margin=J.get('margin'),
    )
    # also record params we used
    for k, v in args_grid.items():
        row[k] = v
    return row, out_csv, out_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', type=str, required=True, help='Path to stage11-well-benchmark-latest-funnel.py')
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grid', nargs='+', required=True, help='Param grid entries like alpha=0.03,0.05 q_s=2,3 ...')
    ap.add_argument('--fixed', nargs='*', default=[], help='Fixed key=value entries to pass through to the benchmark')
    ap.add_argument('--out', type=str, default='sweep_summary.csv')
    ap.add_argument('--recall_floor', type=float, default=0.98)
    ap.add_argument('--keep_tmp', action='store_true')
    args = ap.parse_args()

    bench = Path(args.bench)
    if not bench.exists():
        raise SystemExit(f'Benchmark script not found: {bench}')

    grid_cfg = parse_kv_list(args.grid)
    fixed_cfg = parse_kv_list(args.fixed)
    # ensure samples/seed flow through
    fixed_cfg['samples'] = args.samples
    fixed_cfg['seed'] = args.seed

    rows = []
    artifacts = []
    for i, gparams in enumerate(product_dicts(grid_cfg), start=1):
        row, csv_path, json_path = run_once(bench, fixed_cfg, gparams, i)
        rows.append(row)
        artifacts.append(csv_path); artifacts.append(json_path)

    # write summary CSV
    field_order = (
        ['alpha','beta_s','q_s','tau_rel','tau_abs_q','null_K'] +  # typical keys
        [k for k in grid_cfg.keys() if k not in {'alpha','beta_s','q_s','tau_rel','tau_abs_q','null_K'}] +
        ['g_accuracy_exact','g_precision','g_recall','g_f1','g_jaccard','g_hallucination_rate','g_omission_rate',
         's_accuracy_exact','s_precision','s_recall','s_f1','s_jaccard','s_hallucination_rate','s_omission_rate',
         'phantom_index','margin']
    )
    # augment rows with missing keys to keep header stable
    for r in rows:
        for k in field_order:
            r.setdefault(k, '')
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # choose 'best' row by objective
    def score_row(r):
        # infeasible if recall below floor
        recall = r.get('g_recall', 0.0) or 0.0
        hallu = r.get('g_hallucination_rate', 1.0) or 1.0
        f1    = r.get('g_f1', 0.0) or 0.0
        feasible = recall >= args.recall_floor
        return (feasible, -hallu, f1)  # maximize feasible, minimize hallucinations, maximize F1

    best = max(rows, key=score_row) if rows else None

    print('\n[SWEEP] Wrote summary:', args.out)
    if best:
        print('[SWEEP] Best config (recall_floor=%.3f):' % args.recall_floor)
        keys = [k for k in field_order if k in grid_cfg]
        print('  params:', {k: best[k] for k in keys})
        print('  geodesic:', {k: round(best[f'g_{k}'], 3) for k in ('precision','recall','f1','jaccard','hallucination_rate','omission_rate','accuracy_exact')})
        print('  phantom:', {'phantom_index': round(best['phantom_index'], 4) if best['phantom_index'] is not None else None,
                              'margin': round(best['margin'], 4) if best['margin'] is not None else None})
    else:
        print('[SWEEP] No rows produced.')

    # cleanup tmp unless asked to keep
    if not args.keep_tmp:
        for p in artifacts:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

if __name__ == '__main__':
    main()
