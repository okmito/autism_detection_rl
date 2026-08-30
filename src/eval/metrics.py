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
