"""Circularity audit — §16.1 blocking."""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import balanced_accuracy_score

def sum_threshold_predictions(X: np.ndarray, threshold: int) -> np.ndarray:
    """Sum of item responses >= threshold => 1 else 0. NaN treated as 0 for this oracle."""
    s = np.nansum(X, axis=1)
    return (s >= threshold).astype(int)

def audit_circularity(records: List[Dict[str, Any]], thresholds: range | None = None) -> Dict[str, Any]:
    """Audit whether labels are deterministically derived from item responses.
    Returns dict with best_threshold, exact_match_rate, UAR, sensitivity, specificity, classification.
    """
    X = np.stack([r["item_responses"] for r in records])
    y = np.array([r["label"] for r in records], dtype=int)
    n = X.shape[1]
    if thresholds is None:
        thresholds = range(0, n + 1)

    best = None
    best_acc = -1
    for t in thresholds:
        pred = sum_threshold_predictions(X, t)
        acc = np.mean(pred == y)
        if acc > best_acc:
            best_acc = acc
            best = t

    pred_best = sum_threshold_predictions(X, best)
    exact_match = float(np.mean(pred_best == y))
    # UAR, sens, spec
    # handle edge case where one class missing
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp_ = 0, 0, 0, 0
    if len(np.unique(y)) == 2:
        cm = confusion_matrix(y, pred_best, labels=[0, 1])
        tn, fp, fn, tp_ = cm.ravel()
        sens = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        uar = (sens + spec) / 2
    else:
        sens = spec = uar = float("nan")

    if exact_match == 1.0:
        classification = "Deterministic"
    elif exact_match >= 0.99:
        classification = "Near-deterministic"
    else:
        classification = "Not circular"

    return {
        "best_threshold": best,
        "exact_match_rate": exact_match,
        "uar": float(uar) if not np.isnan(uar) else None,
        "sensitivity": float(sens) if not np.isnan(sens) else None,
        "specificity": float(spec) if not np.isnan(spec) else None,
        "classification": classification,
        "n": len(records),
    }

def is_circular(report: Dict[str, Any]) -> bool:
    return report["classification"] in ("Deterministic", "Near-deterministic")
