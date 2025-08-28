"""NGF evaluation metrics for ARC-style geodesic parser/executor.

Provides:
- seq_metrics(gt, pred): sequence accuracy, semantic similarity, hallucination (tokens)
- grid_metrics(y_true, y_pred): execution accuracy, IoUs, hallucination (pixels)
- aggregate(list_of_dicts, prefix="avg_")
"""

from typing import List, Dict
import numpy as np
from collections import Counter


def levenshtein(a: List[str], b: List[str]) -> int:
    """Edit distance between two sequences of primitives."""
    n, m = len(a), len(b)
    dp = np.zeros((n+1, m+1), dtype=int)
    dp[0, :] = np.arange(m+1)
    dp[:, 0] = np.arange(n+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,
                dp[i, j-1] + 1,
                dp[i-1, j-1] + cost
            )
    return int(dp[n, m])


def seq_metrics(gt: List[str], pred: List[str]) -> Dict[str, float]:
    """Sequence-level metrics for predicted primitive order."""
    exact = float(gt == pred)
    maxlen = max(1, len(gt), len(pred))
    ed = levenshtein(gt, pred)
    edit_norm = ed / maxlen
    semantic_sim = 1.0 - edit_norm

    c_gt, c_pred = Counter(gt), Counter(pred)
    tp = sum((c_gt & c_pred).values())
    fp = sum((c_pred - c_gt).values())
    fn = sum((c_gt - c_pred).values())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    hallucination_rate = fp / max(1, len(pred))
    miss_rate = fn / max(1, len(gt))

    return {
        "seq_exact": exact,
        "seq_edit_norm": edit_norm,
        "seq_semantic_sim": semantic_sim,
        "seq_precision": precision,
        "seq_recall": recall,
        "seq_f1": f1,
        "seq_hallucination_rate": hallucination_rate,
        "seq_miss_rate": miss_rate,
        "seq_len_gt": float(len(gt)),
        "seq_len_pred": float(len(pred)),
        "seq_edit_distance": float(ed),
        "seq_tp": float(tp),
        "seq_fp": float(fp),
        "seq_fn": float(fn),
    }


def _iou_per_label(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, float]:
    labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
    ious = {}
    for lbl in labels:
        t = (y_true == lbl)
        p = (y_pred == lbl)
        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        ious[int(lbl)] = inter / union if union > 0 else 1.0
    return ious


def grid_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Grid-level metrics for ARC-style integer arrays."""
    if y_true.shape != y_pred.shape:
        return {
            "exec_exact": 0.0,
            "cell_acc": 0.0,
            "iou_macro": 0.0,
            "iou_micro": 0.0,
            "hallucination_px_rate": 1.0,
            "n_rows": float(y_true.shape[0]),
            "n_cols": float(y_true.shape[1]),
        }

    exec_exact = float(np.array_equal(y_true, y_pred))
    cell_acc = float((y_true == y_pred).mean())

    ious = _iou_per_label(y_true, y_pred)
    iou_macro = float(np.mean(list(ious.values()))) if ious else 1.0

    inter_total, union_total = 0, 0
    labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
    for lbl in labels:
        t = (y_true == lbl)
        p = (y_pred == lbl)
        inter_total += np.logical_and(t, p).sum()
        union_total += np.logical_or(t, p).sum()
    iou_micro = inter_total / union_total if union_total > 0 else 1.0

    gt_labels = set(np.unique(y_true))
    pred_labels = set(np.unique(y_pred))
    halluc_labels = pred_labels - gt_labels
    if len(halluc_labels) == 0:
        halluc_px_rate = 0.0
    else:
        mask_halluc = np.isin(y_pred, list(halluc_labels))
        halluc_px_rate = float(mask_halluc.mean())

    return {
        "exec_exact": exec_exact,
        "cell_acc": cell_acc,
        "iou_macro": iou_macro,
        "iou_micro": float(iou_micro),
        "hallucination_px_rate": halluc_px_rate,
        "n_rows": float(y_true.shape[0]),
        "n_cols": float(y_true.shape[1]),
    }


def aggregate(sample_metrics: List[Dict[str, float]], prefix: str = "avg_") -> Dict[str, float]:
    """Average numeric keys across samples and include n."""
    if not sample_metrics:
        return {}
    sums = {}
    for d in sample_metrics:
        for k, v in d.items():
            try:
                sums[k] = sums.get(k, 0.0) + float(v)
            except Exception:
                pass
    n = len(sample_metrics)
    avg = {f"{prefix}{k}": v / n for k, v in sums.items()}
    avg[f"{prefix}n_samples"] = float(n)
    return avg
