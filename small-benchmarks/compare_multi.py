import json, sys
from math import comb
from collections import OrderedDict

def read_preds(path):
    gold, pred = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            gold.append(int(r["label"]))
            pred.append(int(r["pred"]))
    return gold, pred

def mcnemar(gold, pA, pB):
    b = sum(1 for y,a,b_ in zip(gold,pA,pB) if a!=y and b_==y)
    c = sum(1 for y,a,b_ in zip(gold,pA,pB) if a==y and b_!=y)
    n = b + c
    if n == 0: return 1.0, b, c
    p = sum(comb(n, k) for k in range(0, min(b,c)+1)) / (2**n) * 2
    return min(1.0, p), b, c

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python compare_multi.py run1.jsonl run2.jsonl [run3.jsonl ...]")

    files = sys.argv[1:]
    names = [f.split("/")[-1].replace(".jsonl","") for f in files]

    gold, preds = None, []
    for f in files:
        g, p = read_preds(f)
        if gold is None:
            gold = g
        else:
            assert g == gold, "Mismatched gold labels across runs"
        preds.append(p)

    n = len(gold)
    accs = [sum(int(p==y) for p,y in zip(pr,gold)) / n for pr in preds]

    # Build summary
    summary = {
        "n": n,
        "runs": [
            OrderedDict([("name", nme), ("acc", accs[i])])
            for i,nme in enumerate(names)
        ],
        "comparisons": []
    }

    for i in range(len(files)):
        for j in range(i+1, len(files)):
            pval, b, c = mcnemar(gold, preds[i], preds[j])
            summary["comparisons"].append(OrderedDict([
                ("runA", names[i]), ("accA", accs[i]),
                ("runB", names[j]), ("accB", accs[j]),
                ("delta_acc", accs[j]-accs[i]),
                ("mcnemar_p", pval),
                ("b_Awrong_Bright", b),
                ("c_Aright_Bwrong", c)
            ]))

    print(json.dumps(summary, indent=2))
