"""Subgroup analysis — §19.4"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from .metrics import compute_metrics

def subgroup_report(records: List[Dict[str, Any]], y_prob: np.ndarray, tau: float=0.5, min_cell: int=20) -> dict:
    """UAR, acquisition burden by sex and age_band, interaction if cell sizes >=min_cell."""
    y_true = np.array([r["label"] for r in records])
    # by sex
    out = {}
    sexes = set(r["covariates"]["sex"] for r in records)
    for s in sexes:
        idx = [i for i,r in enumerate(records) if r["covariates"]["sex"]==s]
        if len(idx)<min_cell:
            out[f"sex_{s}"]={"n":len(idx), "note":"underpowered"}
            continue
        m = compute_metrics(y_true[idx], y_prob[idx], tau)
        out[f"sex_{s}"]={"n":len(idx), "uar":m["uar"], "brier":m["brier"]}
    # by age_band
    ages = set(r["covariates"]["age_band"] for r in records)
    for a in ages:
        idx = [i for i,r in enumerate(records) if r["covariates"]["age_band"]==a]
        if len(idx)<min_cell:
            out[f"age_{a}"]={"n":len(idx), "note":"underpowered"}
            continue
        m = compute_metrics(y_true[idx], y_prob[idx], tau)
        out[f"age_{a}"]={"n":len(idx), "uar":m["uar"]}
    # interaction
    for s in sexes:
        for a in ages:
            idx = [i for i,r in enumerate(records) if r["covariates"]["sex"]==s and r["covariates"]["age_band"]==a]
            key=f"sex_{s}_x_age_{a}"
            if len(idx)<min_cell:
                out[key]={"n":len(idx), "note":"underpowered — not interpreted"}
            else:
                m = compute_metrics(y_true[idx], y_prob[idx], tau)
                out[key]={"n":len(idx), "uar":m["uar"]}
    return out
