import re, json, argparse
import pandas as pd


"""

python3 score_latent_arc.py \
  --truths_csv synth_arc_truth.csv \
  --generations_jsonl logs/prewarp_generations.jsonl \
  --out_csv synth_arc_scored.csv \
  --out_json synth_arc_summary.json

python3 score_latent_arc_v2.py \
  --truths_csv synth_arc_truth.csv \
  --generations_jsonl logs/prewarp_generations.jsonl \
  --normalize_prompts \
  --geo_field geodesic_prewarp \
  --out_csv synth_arc_scored.csv \
  --out_json synth_arc_summary.json

python3 debug_prompt_alignment.py \
  --truths_csv synth_arc_truth.csv \
  --generations_jsonl logs/prewarp_generations.jsonl \
  --out_normalized_jsonl normalized_generations.jsonl

"""

def normalize_text(s: str):
    return re.sub(r"\s+", " ", s.strip()).strip()

def tokenize_for_metrics(s: str):
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())

def metrics_against_expected(pred: str, expected: str):
    pred, exp = normalize_text(pred or ""), normalize_text(expected or "")
    acc_exact = int(pred == exp)
    ptoks, etoks = tokenize_for_metrics(pred), tokenize_for_metrics(exp)
    pset, eset = set(ptoks), set(etoks)
    inter = pset & eset
    tp, fp, fn = len(inter), len(pset - inter), len(eset - inter)
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    jaccard = len(inter)/len(pset|eset) if (pset|eset) else 0.0
    hallu, omission = fp/max(1,len(ptoks)), fn/max(1,len(etoks))
    return dict(accuracy_exact=acc_exact, precision=precision, recall=recall,
                f1=f1, jaccard=jaccard, hallucination_rate=hallu, omission_rate=omission)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truths_csv', required=True)
    ap.add_argument('--generations_jsonl', required=True)
    ap.add_argument('--out_csv', default='latent_arc_scored.csv')
    ap.add_argument('--out_json', default='latent_arc_summary.json')
    args = ap.parse_args()

    truths_df = pd.read_csv(args.truths_csv)
    tmap = dict(zip(truths_df["prompt"].astype(str), truths_df["true"].astype(str)))

    rows = []
    with open(args.generations_jsonl, encoding='utf-8') as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            pr = str(r.get("prompt","")).strip()
            if pr not in tmap: continue
            stock, geo, exp = r.get("stock",""), r.get("geodesic_prewarp","") or r.get("geodesic",""), tmap[pr]
            ms, mg = metrics_against_expected(stock, exp), metrics_against_expected(geo, exp)
            rec = {"prompt": pr, "expected": exp, "stock": stock, "geo": geo}
            for k,v in ms.items(): rec[f"stock_{k}"]=v
            for k,v in mg.items(): rec[f"geo_{k}"]=v
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    def agg(prefix): return {k: float(df[f"{prefix}_{k}"].mean()) for k in ["accuracy_exact","precision","recall","f1","jaccard","hallucination_rate","omission_rate"]}
    summary = {"n_scored": int(len(df)), "stock": agg("stock"), "geodesic_prewarp": agg("geo")}
    json.dump(summary, open(args.out_json,"w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
