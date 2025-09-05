import json, sys
from collections import OrderedDict
from math import comb

"""
python3 compare_runs.py results/stock_gpt2_n1000.jsonl results/ngf_geo_gpt2_n1000.jsonl
"""


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
    # two-sided exact binomial
    p = sum(comb(n, k) for k in range(0, min(b,c)+1)) / (2**n) * 2
    return min(1.0, p), b, c

if __name__ == "__main__":
    j1, j2 = sys.argv[1], sys.argv[2]
    gold, pred1 = read_preds(j1)
    _, pred2 = read_preds(j2)
    assert len(gold)==len(pred1)==len(pred2)
    acc1 = sum(int(a==y) for a,y in zip(pred1,gold)) / len(gold)
    acc2 = sum(int(b==y) for b,y in zip(pred2,gold)) / len(gold)
    pval, b, c = mcnemar(gold, pred1, pred2)
    out = OrderedDict([
        ("n", len(gold)),
        ("acc_run1", acc1),
        ("acc_run2", acc2),
        ("delta_acc", acc2 - acc1),
        ("mcnemar_p", pval),
        ("b_Awrong_Bright", b),
        ("c_Aright_Bwrong", c),
    ])
    print(json.dumps(out, indent=2))
