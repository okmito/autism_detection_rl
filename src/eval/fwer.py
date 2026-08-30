"""Holm-Bonferroni — §19.3"""
from __future__ import annotations
import numpy as np
from typing import List

def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni procedure. Returns rejected indices and adjusted."""
    m = len(pvals)
    order = np.argsort(pvals)
    rejected = [False]*m
    for k, idx in enumerate(order):
        thresh = alpha / (m - k)
        if pvals[idx] <= thresh:
            rejected[idx]=True
        else:
            break
    # adjusted p-values (Holm)
    adj = [0]*m
    sorted_p = [pvals[i] for i in order]
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, sorted_p[i] * (m - i))
    # ensure monotonic
    for i in range(1, m):
        oi = order[i]; pi = order[i-1]
        if adj[oi] < adj[pi]:
            adj[oi]=adj[pi]
    return {"rejected": rejected, "adjusted": adj, "alpha": alpha}
