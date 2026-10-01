"""Static RFE subset baseline — §17 #5."""
from __future__ import annotations
import warnings

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Any

STOP = -1


class StaticRFEPolicy:
    """Select a fixed subset of size B by RFE on training data, then ask those items.

    At episode time the policy walks its fixed selection in order, skipping
    anything already observed, and spends the whole matched budget. Spec §17 #5
    requires selection on training data only and no further selection at
    evaluation time; it also never stops early, so adaptive *stopping* stays
    isolated in AB-7.
    """

    def __init__(self, records: List[Dict[str, Any]], n_items: int, budget: int):
        self.n_items = n_items
        self.budget = budget
        X = np.stack([r["item_responses"] for r in records])
        y = np.array([r["label"] for r in records])

        # Missing responses are set to 0 *for the selection fit only*. The
        # exhaustive fixed-subset baseline treats NA as its own group, so the two
        # static baselines used different missingness semantics and their results
        # were not comparable. That divergence is resolved by `missing_mode`,
        # which is recorded so an artifact can state which was used.
        self.missing_mode = "nan_to_zero"
        X = np.nan_to_num(X, nan=0.0)

        k = min(budget, n_items)
        #: True when RFE failed and the index-order fallback was used. An earlier
        #: revision swallowed the exception silently, so a degraded policy was
        #: indistinguishable from a converged one.
        self.fallback_used = False

        if k <= 0:
            self.selected: List[int] = []
        else:
            try:
                est = LogisticRegression(max_iter=500)
                rfe = RFE(est, n_features_to_select=k)
                rfe.fit(X, y)
                self.selected = [i for i in range(n_items) if rfe.support_[i]]
            except (ValueError, np.linalg.LinAlgError) as e:
                warnings.warn(
                    f"StaticRFEPolicy: RFE failed ({type(e).__name__}: {e}); "
                    f"falling back to the first {k} items in index order. The "
                    f"resulting policy is NOT a converged RFE selection.",
                    RuntimeWarning, stacklevel=2,
                )
                self.fallback_used = True
                self.selected = list(range(k))
            if len(self.selected) < k:
                # RFE can return fewer than requested when a column is constant.
                extra = [i for i in range(n_items) if i not in self.selected]
                self.selected = list(self.selected) + extra[: k - len(self.selected)]
        self._order = list(self.selected)  # copy, not an alias

    def __call__(self, state, legal):
        items_only = [a for a in legal if a != STOP]
        if not items_only:
            return STOP
        for j in self._order:
            if j in items_only:
                return j
        # The fixed set is exhausted but budget remains: ask whatever is left
        # rather than emitting an illegal action.
        return items_only[0]
