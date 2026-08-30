"""Greedy information gain — §17 #7"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

STOP = -1

def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

class GreedyIGPolicy:
    """One-step lookahead IG(s,j)=H(Y|s)-Σ_v P(v|s,j)H(Y|s,j,v) using empirical training distribution."""
    def __init__(self, records: List[Dict[str, Any]], n_items: int):
        self.records = records
        self.n_items = n_items
        self.X = np.stack([r["item_responses"] for r in records])
        self.y = np.array([r["label"] for r in records], dtype=int)
        self.missing_mask = np.stack([r["missing_mask"] for r in records])

    def _support(self, mask: np.ndarray, value: np.ndarray) -> np.ndarray:
        consistent = np.ones(len(self.records), dtype=bool)
        for j in range(self.n_items):
            m = mask[j]
            if m == 1:
                v = value[j]
                consistent &= (~self.missing_mask[:, j]) & (self.X[:, j]==v)
            elif m == 2:
                consistent &= self.missing_mask[:, j]
        return np.where(consistent)[0]

    def __call__(self, state: Dict[str, Any], legal: List[int]) -> int:
        mask = state["mask"]; value = state["value"]
        # if STOP in legal, we still choose greedy among items; stopping decision handled externally
        # For this policy, we never choose STOP unless forced (handled by env). If STOP in legal and we are called with b>0, we choose best item.
        items_only = [a for a in legal if a != STOP]
        if not items_only:
            return STOP
        idx = self._support(mask, value)
        if len(idx)==0:
            return items_only[0]
        y_s = self.y[idx]
        p_s = float(y_s.mean()) if len(y_s)>0 else 0.5
        h_s = entropy(p_s)
        best_j = items_only[0]
        best_ig = float("-inf")
        for j in items_only:
            # distribution over v
            total = len(idx)
            # count values among support where not missing
            sub = idx[~self.missing_mask[idx, j]]
            if len(sub)==0:
                # missing branch only
                continue
            vals, counts = np.unique(self.X[sub, j], return_counts=True)
            n_miss = int(np.sum(self.missing_mask[idx, j]))
            exp_h = 0.0
            for v,c in zip(vals, counts):
                p = c/total
                # child support where X==v
                # compute child entropy
                child_idx = idx[(~self.missing_mask[idx, j]) & (self.X[idx,j]==v)]
                if len(child_idx)==0:
                    continue
                p_child = float(self.y[child_idx].mean())
                exp_h += p * entropy(p_child)
            if n_miss>0:
                p_m = n_miss/total
                child_idx = idx[self.missing_mask[idx, j]]
                p_child = float(self.y[child_idx].mean()) if len(child_idx)>0 else 0.5
                exp_h += p_m * entropy(p_child)
            ig = h_s - exp_h
            if ig > best_ig:
                best_ig = ig
                best_j = j
        return best_j
