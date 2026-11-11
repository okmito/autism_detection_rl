"""Metrics — §19.1"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, tau: float = 0.5) -> dict:
    y_pred = (y_prob >= tau).astype(int)
    out = {}
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["terminal_utility"] = float(np.mean(1 - (y_prob - y_true)**2))
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["auprc"] = float("nan")
    out["uar"] = float(balanced_accuracy_score(y_true, y_pred))
    # sensitivity/specificity
    if len(np.unique(y_true))==2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        out["sensitivity"] = float(tp/(tp+fn) if (tp+fn)>0 else 0)
        out["specificity"] = float(tn/(tn+fp) if (tn+fp)>0 else 0)
    else:
        out["sensitivity"]=out["specificity"]=float("nan")
    # ECE 10 bins
    out["ece"] = float(_ece(y_true, y_prob))
    cal = calibration_slope_intercept(y_true, y_prob)
    out["calibration_slope"] = cal["slope"]
    out["calibration_intercept"] = cal["intercept"]
    return out


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calibration slope/intercept: logistic recalibration of y on logit(p).

    Returns {slope, intercept}. Uninformative (clipped) when predictions
    saturate; computed on the bounded raw probability via an L2 logistic
    fit so it cannot diverge on separable data (same rationale as
    PlattOnProbability in the v3 predictor).
    """
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    try:
        m = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000)
        m.fit(p.reshape(-1, 1), np.asarray(y_true, dtype=int))
        return {"slope": float(m.coef_[0][0]), "intercept": float(m.intercept_[0])}
    except Exception:
        return {"slope": float("nan"), "intercept": float("nan")}


def compute_metrics_with_ci(y_true: np.ndarray, y_prob: np.ndarray, tau: float = 0.5,
                            n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """compute_metrics + percentile bootstrap CIs for brier/auroc/uar.

    CIs are resampling (model-stability) statements over the evaluation
    sample; they are not clinical confidence intervals.
    """
    base = compute_metrics(y_true, y_prob, tau)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["brier", "auroc", "uar"]
    vals = {k: [] for k in keys}
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            m = compute_metrics(y_true[idx], y_prob[idx], tau)
            for k in keys:
                vals[k].append(m[k])
        except Exception:
            continue
    out = dict(base)
    for k in keys:
        arr = np.asarray(vals[k], dtype=float)
        if arr.size:
            out[f"{k}_ci95"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            out[f"{k}_bootstrap_sd"] = float(arr.std(ddof=1))
        else:
            out[f"{k}_ci95"] = [float("nan"), float("nan")]
            out[f"{k}_bootstrap_sd"] = float("nan")
    return out

def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    e=0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob>=lo) & (y_prob<hi if i<n_bins-1 else y_prob<=hi)
        if mask.sum()==0: continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        e+= abs(acc-conf)*mask.sum()/len(y_true)
    return e

def acquisition_burden(items_asked_list):
    lens = [len(x) for x in items_asked_list]
    return {"mean": float(np.mean(lens)), "median": float(np.median(lens)), "lens": lens}
