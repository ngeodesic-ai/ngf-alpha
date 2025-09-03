

"""
python3 stage11_scorer_wrapper.py \
  --generations layer9_wiring_plan/simple_benchmark/ab_stock_patterned.json \
  --truths      layer9_wiring_plan/simple_benchmark/patterned_truths.csv \
  --out         layer9_wiring_plan/simple_benchmark/ab_stock_patterned_metrics.json \
  --match_mode numbers_only --method_label "Stock"

python3 stage11_scorer_wrapper.py \
  --generations layer9_wiring_plan/simple_benchmark/ab_geo_patterned.json \
  --truths layer9_wiring_plan/simple_benchmark/patterned_truths.csv \
  --out ab_geo_patterned_metrics.json \
  --match_mode alnum_lower \
  --method_label "Geo"

python3 stage11_scorer_wrapper.py \
  --generations layer9_wiring_plan/simple_benchmark/ab_geo_detect_patterned.json \
  --truths layer9_wiring_plan/simple_benchmark/patterned_truths.csv \
  --out ab_geo_detect_patterned_metrics.json \
  --match_mode alnum_lower \
  --method_label "Geo Detect"

python3 stage11_scorer_wrapper.py \
  --generations layer9_wiring_plan/simple_benchmark/ab_geo_detect_denoise_patterned.json \
  --truths layer9_wiring_plan/simple_benchmark/patterned_truths.csv \
  --out ab_geo_detect_denoise_patterned_metrics.json \
  --match_mode numbers_only --method_label "Geo Detect Denoise"


python3 stage11_scorer_wrapper.py \
  --generations layer9_wiring_plan/simple_benchmark/ab_stock_patterned.json \
  --truths      layer9_wiring_plan/simple_benchmark/patterned_truths.csv \
  --out         ab_stock_v3fewshot_metrics.json \
  --match_mode alnum_lower \
  --regex '(?s).*?(-?\\d+)(?!.*-?\\d+)' \
  --method_label "Stock (v3 few-shot)"

python3 stage11_ab_eval_base_denoise_v2.py \
  --model gpt2 --layer -9 \
  --prompts patterned_prompts_v1.txt \
  --max_new_tokens 8 \
  --gen_mode stock --device cuda \
  --out_json ab_stock_patterned.json

"""
import json, csv, re, argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def read_json_any(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    if text[0] == "{":
        return json.loads(text)
    if text[0] == "[":
        return json.loads(text)
    # fallback as JSONL
    return read_jsonl(path)

def read_generations(path: Path) -> List[dict]:
    data = read_json_any(path)
    # Accept list or dict with "items"/"generations"
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("items") or data.get("generations") or data.get("rows") or []
    else:
        rows = []
    # Normalize keys
    out = []
    for i, r in enumerate(rows):
        rid = r.get("id", r.get("example_id", i))
        gen = r.get("generation") or r.get("output") or r.get("text") or r.get("answer") or ""
        method = r.get("method") or r.get("label") or r.get("config", {}).get("method")
        prompt = r.get("prompt") or r.get("input") or ""
        out.append({"id": str(rid), "prompt": prompt, "generation": str(gen), "method": method})
    return out

def read_truths(path: Path) -> Dict[str, List[str]]:
    # CSV: expects columns [id, truth] possibly repeated for multiple truths per id
    if path.suffix.lower() == ".csv":
        truths = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Try to find id and truth columns
            cols = reader.fieldnames or []
            id_col = next((c for c in cols if c.lower() in ("id","example_id","qid")), None)
            truth_col = next((c for c in cols if "truth" in c.lower() or c.lower()=="answer"), None)
            if id_col is None or truth_col is None:
                raise ValueError("CSV must contain id and truth/answer columns")
            for row in reader:
                k = str(row[id_col])
                v = str(row[truth_col])
                truths.setdefault(k, []).append(v)
        return truths
    # JSON/JSONL: support {"id": "...", "truth": ...} rows or dict id->truths
    data = read_json_any(path)
    truths = {}
    if isinstance(data, dict):
        # dict id -> truth(s)
        for k, v in data.items():
            if isinstance(v, list):
                truths[str(k)] = [str(x) for x in v]
            else:
                truths[str(k)] = [str(v)]
        return truths
    if isinstance(data, list):
        for i, r in enumerate(data):
            rid = str(r.get("id", r.get("example_id", i)))
            tv = r.get("truth")
            if tv is None:
                tv = r.get("answer")
            if tv is None:
                continue
            if isinstance(tv, list):
                truths.setdefault(rid, []).extend([str(x) for x in tv])
            else:
                truths.setdefault(rid, []).append(str(tv))
        return truths
    return truths

def normalize_text(s: str, mode: str, regex: Optional[str]) -> str:
    s = s.strip()
    if regex:
        m = re.search(regex, s, flags=re.DOTALL)
        if m:
            # Prefer first capturing group if present
            if m.groups():
                s = m.group(1)
            else:
                s = m.group(0)
    if mode == "exact":
        return s
    if mode == "alnum_lower":
        return re.sub(r"[^0-9a-z]+", "", s.lower())
    if mode == "numbers_only":
        # Capture first signed integer or float
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        return m.group(0) if m else ""
    if mode == "strip_nonascii":
        return re.sub(r"[^\x20-\x7E]+", "", s).strip()
    return s

def score_items(gens: List[dict], truths: Dict[str, List[str]], mode: str, regex: Optional[str]) -> Tuple[List[dict], Dict[str, float]]:
    tp = fp = fn = 0
    n = len(gens)
    n_halluc = 0
    n_omit = 0
    per = []
    for row in gens:
        rid = str(row["id"])
        gen_raw = row["generation"]
        gen_norm = normalize_text(gen_raw, mode, regex)
        truth_list = truths.get(rid, [])
        truth_norms = [normalize_text(t, mode, regex) for t in truth_list]

        # Define omission: empty after normalization
        omitted = (gen_norm == "")
        # Define hallucination: non-empty but not matching any truth
        matched = gen_norm in truth_norms if truth_norms else False
        halluc = (not omitted) and (not matched)

        if matched:
            tp += 1
        else:
            if truth_list:
                # Missed a known truth
                fn += 1
            if not omitted:
                fp += 1

        if halluc:
            n_halluc += 1
        if omitted:
            n_omit += 1

        per.append({
            "id": rid,
            "generation": gen_raw,
            "gen_norm": gen_norm,
            "truths": truth_list,
            "truths_norm": truth_norms,
            "matched": matched,
            "omitted": omitted,
            "hallucinated": halluc,
            "method": row.get("method"),
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    accuracy  = tp / n if n > 0 else 0.0
    halluc_rate = n_halluc / n if n > 0 else 0.0
    omission_rate = n_omit / n if n > 0 else 0.0

    agg = {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": halluc_rate,
        "omission_rate": omission_rate,
    }
    return per, agg

def main():
    ap = argparse.ArgumentParser(description="Stage-11 scorer wrapper: compute F1/hallucination from generations + truths")
    ap.add_argument("--generations", required=True, help="Path to generations JSON/JSONL")
    ap.add_argument("--truths", required=True, help="Path to truths CSV/JSON/JSONL")
    ap.add_argument("--out", required=True, help="Output metrics JSON path")
    ap.add_argument("--match_mode", default="alnum_lower", choices=["exact","alnum_lower","numbers_only","strip_nonascii"], help="Normalization/matching mode")
    ap.add_argument("--regex", default=None, help="Optional regex with one capturing group to extract the canonical answer")
    ap.add_argument("--method_label", default=None, help="Override method label to stamp on results")
    args = ap.parse_args()

    gen_path = Path(args.generations)
    truths_path = Path(args.truths)
    out_path = Path(args.out)

    gens = read_generations(gen_path)
    truths = read_truths(truths_path)

    per, agg = score_items(gens, truths, args.match_mode, args.regex)

    # If method_label provided, stamp onto each row
    if args.method_label:
        for r in per:
            r["method"] = args.method_label

    out = {
        "config": {
            "match_mode": args.match_mode,
            "regex": args.regex,
            "method_label": args.method_label,
            "generations_file": str(gen_path),
            "truths_file": str(truths_path),
        },
        "aggregate": agg,
        "items": per,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[WRITE] Metrics → {out_path}")
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
