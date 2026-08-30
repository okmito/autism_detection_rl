"""Static RFE subset baseline — §17 #5"""
from __future__ import annotations
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Any

class StaticRFEPolicy:
    """Select fixed subset of size B via RFE on training data, then evaluate without further selection.
    At episode time, picks items from that fixed set in arbitrary order until budget.
    """
    def __init__(self, records: List[Dict[str, Any]], n_items: int, budget: int):
        self.n_items = n_items
        self.budget = budget
        X = np.stack([np.nan_to_num(r["item_responses"], nan=0.0) for r in records])
        y = np.array([r["label"] for r in records])
        # Handle case where budget > n or budget==n
        k = min(budget, n_items)
        if k <=0:
            self.selected = []
        else:
            # If n_items small and budget close to n, just select top by univariate
            try:
                est = LogisticRegression(max_iter=500)
                rfe = RFE(est, n_features_to_select=k)
                rfe.fit(X, y)
                self.selected = [i for i in range(n_items) if rfe.support_[i]]
            except Exception:
                # fallback: select first k
                self.selected = list(range(k))
        self._order = self.selected  # fixed order

    def __call__(self, state, legal):
        items_only = [a for a in legal if a != -1]
        if not items_only:
            return -1
        # pick first remaining item from fixed set that is still legal
        for j in self._order:
            if j in items_only:
                return j
        # if fixed set exhausted but still have budget, pick any remaining
        return items_only[0]
