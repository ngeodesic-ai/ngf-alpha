import argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def per_item_top1(scores, y, items):
    # pick argmax among 4 endings within each item
    correct = 0
    total = 0
    for uid in np.unique(items):
        mask = (items == uid)
        s = scores[mask]
        yy = y[mask]
        # guard: sometimes fewer than 4 if you changed probe_items
        if s.size == 0: 
            continue
        pick = np.argmax(s)
        correct += int(yy[pick] == 1)
        total += 1
    return correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--C", type=float, default=1.0)  # inverse regularization
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]            # [N, D]
    y = data["y"].astype(int)
    items = data["item"]

    # standardize features (important for linear probe)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=args.folds)
    aucs, accs_item, accs_point = [], [], []

    for tr, te in gkf.split(X, y, groups=items):
        clf = LogisticRegression(
            penalty="l2", C=args.C, solver="lbfgs", max_iter=200, n_jobs=None
        )
        clf.fit(X[tr], y[tr])

        proba = clf.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)

        # binary point-wise metrics (gold vs distractor)
        aucs.append(roc_auc_score(y[te], proba))
        accs_point.append(accuracy_score(y[te], pred))

        # per-item top-1 metric (choose among endings)
        accs_item.append(per_item_top1(proba, y[te], items[te]))

    print(f"[Probe] point-wise AUC:   mean={np.mean(aucs):.3f}  std={np.std(aucs):.3f}")
    print(f"[Probe] point-wise Acc:   mean={np.mean(accs_point):.3f}")
    print(f"[Probe] per-item Top-1:   mean={np.mean(accs_item):.3f}")
    print("Baseline (random choice among 4) ≈ 0.25 per-item")

if __name__ == "__main__":
    main()
