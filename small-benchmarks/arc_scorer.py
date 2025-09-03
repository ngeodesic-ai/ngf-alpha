#!/usr/bin/env python3
import argparse, csv, json, re
from collections import defaultdict
import pandas as pd

'''
python3 text_arc_unified.py \
  --gen_mode stock \
  --prompts calib/ngf_eval_prompts_60.txt \
  --out generations_stock.v4b.tap9.jsonl
'''


def norm(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z]+', ' ', s or '')
    return ' '.join(s.lower().strip().split())

def read_truths(path: str):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    idcol = cols.get('id') or list(df.columns)[0]
    # heuristics for truth column
    truthcol = cols.get('truth') or cols.get('answer') or cols.get('gold') or list(df.columns)[1]
    truths = {}
    for _, row in df.iterrows():
        try:
            i = int(row[idcol])
        except Exception:
            continue
        truths[i] = str(row[truthcol])
    return truths

def read_generations(path: str):
    gens = {}
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                i = int(obj.get('id'))
                gens[i] = str(obj.get('generation',''))
            except Exception:
                pass
    return gens

def score(generations_path: str, truths_path: str):
    truths = read_truths(truths_path)
    gens   = read_generations(generations_path)
    ids = sorted(set(truths.keys()) & set(gens.keys()))
    total = len(ids)
    correct = 0
    rows = []
    for i in ids:
        y = norm(truths[i]); yhat = norm(gens[i])
        ok = int(y == yhat and y != '')
        correct += ok
        rows.append({'id': i, 'truth': truths[i], 'generation': gens[i], 'match': ok})
    acc = (correct / total) if total else 0.0
    return {'total': total, 'correct': correct, 'accuracy': acc}, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generations', required=True)
    ap.add_argument('--truths', required=True)
    ap.add_argument('--out', default=None, help='Write metrics JSON here')
    ap.add_argument('--matches_csv', default=None, help='Optional per-item CSV with match flags')
    args = ap.parse_args()

    metrics, rows = score(args.generations, args.truths)
    metrics['generations'] = args.generations
    metrics['truths'] = args.truths
    print(json.dumps(metrics, indent=2))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.matches_csv:
        import csv
        with open(args.matches_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id','truth','generation','match'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

if __name__ == '__main__':
    main()
