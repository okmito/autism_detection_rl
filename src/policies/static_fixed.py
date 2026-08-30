"""Exact best fixed subset — §17 #4 (exhaustive search over subsets of size B)"""
from __future__ import annotations
import numpy as np
import itertools
from typing import List, Dict, Any

class ExactFixedSubsetPolicy:
    """Exhaustive subset search on training data under same empirical utility as §14."""
    def __init__(self, records: List[Dict[str, Any]], n_items: int, budget: int, lambda_cost: float=0.0):
        self.n_items=n_items; self.budget=budget; self.lambda_cost=lambda_cost
        X=np.stack([r["item_responses"] for r in records])
        y=np.array([r["label"] for r in records])
        best=None; best_subset=None
        for subset in itertools.combinations(range(n_items), budget):
            # compute empirical utility for this fixed subset (observe exactly these items)
            # grouping by pattern on subset
            patterns={}
            for i,row in enumerate(X):
                # handle missing: pattern includes NaN handling
                key=tuple("NA" if np.isnan(row[j]) else int(row[j]) for j in subset)
                patterns.setdefault(key, []).append(i)
            brier=0
            for idx in patterns.values():
                p=float(y[idx].mean()) if len(idx)>0 else 0.5
                for i in idx:
                    brier+=(p - y[i])**2
            brier/=len(y)
            util= 1 - brier - lambda_cost*budget
            if best is None or util>best:
                best=util; best_subset=list(subset)
        self.best_subset=best_subset if best_subset is not None else list(range(min(budget,n_items)))
        self.best_value=best

    def __call__(self, state, legal):
        items=[a for a in legal if a!=-1]
        if not items:
            return -1
        for j in self.best_subset:
            if j in items:
                return j
        return items[0]
