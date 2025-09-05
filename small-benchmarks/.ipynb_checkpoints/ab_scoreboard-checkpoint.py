#!/usr/bin/env python3

'''
python3 ab_scoreboard.py \
  --stock  generations_stock.v4b.tap9.jsonl \
  --geo    generations_geo_steps.v4b.tap9.jsonl \
  --truths truths.csv \
  --out    ab_scoreboard.v4b.tap9.json
'''

import argparse, json, subprocess, sys, os, shutil

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        sys.exit(p.returncode)
    return p.stdout

def main():
    ap = argparse.ArgumentParser(description="A/B scoreboard (stock vs geo)")
    ap.add_argument('--stock', required=True, help='Path to STOCK generations JSONL')
    ap.add_argument('--geo', required=True, help='Path to GEO generations JSONL')
    ap.add_argument('--truths', required=True, help='Path to truths CSV')
    ap.add_argument('--out', default='ab_scoreboard.json', help='Write combined metrics JSON')
    args = ap.parse_args()

    mydir = os.path.dirname(os.path.abspath(__file__))
    scorer = os.path.join(mydir, 'arc_scorer.py')

    s_out = os.path.join(mydir, 'metrics_stock.json')
    g_out = os.path.join(mydir, 'metrics_geo.json')

    run([sys.executable, scorer, '--generations', args.stock, '--truths', args.truths, '--out', s_out])
    run([sys.executable, scorer, '--generations', args.geo,   '--truths', args.truths, '--out', g_out])

    with open(s_out, 'r', encoding='utf-8') as f: s = json.load(f)
    with open(g_out, 'r', encoding='utf-8') as f: g = json.load(f)

    table = {
        'stock_total': s['total'], 'geo_total': g['total'],
        'stock_correct': s['correct'], 'geo_correct': g['correct'],
        'stock_accuracy': s['accuracy'], 'geo_accuracy': g['accuracy'],
        'delta_accuracy': g['accuracy'] - s['accuracy']
    }
    print(json.dumps(table, indent=2))

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'stock': s, 'geo': g, 'summary': table}, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
