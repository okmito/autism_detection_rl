"""Exact best fixed subset — §17 #4 (exhaustive search over subsets of size B)."""
from __future__ import annotations
import numpy as np
import itertools
import warnings
from typing import List, Dict, Any

STOP = -1


class ExactFixedSubsetPolicy:
    """Exhaustive subset search on training data under the same empirical utility as §14.

    This is the comparator H1 is written against: the adaptive policy must beat
    the best *fixed* choice of exactly ``B`` items, measured on identical data
    with the same objective.

    On ``lambda_cost``
    ------------------
    Every candidate subset has size exactly ``B``, so with uniform costs the
    acquisition term ``lambda * sum_j c_j`` is the *same constant* for all
    candidates and cannot change the argmax. It is retained in the signature
    because callers pass it, and it is applied to ``best_value``, but it is
    deliberately **not** used to rank subsets — an earlier revision added it to
    the comparison, where it silently did nothing while looking meaningful.
    Adaptive *cost* variation is the job of AB-7 / the λ sweep, not of a
    fixed-size subset search.
    """

    def __init__(self, records: List[Dict[str, Any]], n_items: int, budget: int,
                 lambda_cost: float = 0.0):
        self.n_items = n_items
        self.budget = budget
        self.lambda_cost = lambda_cost
        X = np.stack([r["item_responses"] for r in records])
        y = np.array([r["label"] for r in records])

        #: True when the requested budget exceeded the item count and the
        #: search had to fall back. Recorded rather than applied silently.
        self.fallback_used = False
        #: Missing responses form their own pattern group here, matching the
        #: empirical support semantics of `ExactDP._support_indices`. RFE uses a
        #: different convention (`nan_to_zero`), and the two `missing_mode`
        #: attributes record that so artifacts can state it.
        self.missing_mode = "na_is_own_group"

        k = min(budget, n_items)
        if budget > n_items:
            warnings.warn(
                f"ExactFixedSubsetPolicy: budget {budget} exceeds n_items "
                f"{n_items}; searching over all {n_items} items instead.",
                RuntimeWarning, stacklevel=2,
            )
            self.fallback_used = True

        best = None
        best_subset = None
        for subset in itertools.combinations(range(n_items), k):
            patterns: Dict[tuple, List[int]] = {}
            for i, row in enumerate(X):
                key = tuple("NA" if np.isnan(row[j]) else int(row[j]) for j in subset)
                patterns.setdefault(key, []).append(i)
            brier = 0.0
            for idx in patterns.values():
                p = float(y[idx].mean()) if idx else 0.5
                for i in idx:
                    brier += (p - y[i]) ** 2
            brier /= len(y)
            util = 1.0 - brier  # cost is constant across equal-size subsets
            if best is None or util > best:
                best = util
                best_subset = list(subset)

        self.best_subset = best_subset if best_subset is not None else list(range(k))
        self.best_value = None if best is None else float(best) - self.lambda_cost * k

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        if not items:
            return STOP
        for j in self.best_subset:
            if j in items:
                return j
        return items[0]
