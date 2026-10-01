"""Beta-prior adaptive policy with EVOI stopping — P1-f.

**This is a new arm. ``GreedyIGPolicy`` is deliberately left untouched** so its
near-optimality result (§17 #7, reproduced in ``POLICY_BENCHMARK_REPORT.md`` §A.3)
remains intact and comparable.

Why this policy exists
----------------------
``GreedyIGPolicy`` selects by information gain, ``IG = H(Y|s) - E_v[H(Y|s,j,v)]``,
on the **raw** empirical support mean. Under this project's circular label
(``label = 1[sum(A) >= 4]``, verified 506/506) the support becomes *pure* — every
remaining training record carries the same label — very quickly:

    61.3% of the states at which greedy actually makes a decision have a pure
    support (measured on the held-out test split at B=6).

On a pure support ``H(Y|s) = 0`` and every child support is pure too, so **every
legal item's IG is exactly 0**. The criterion is identically zero, ``argmax``
returns the first maximum, and ``greedy.py``'s ``best_j = items_only[0]``
initialisation silently takes over: the "policy" becomes "ask the lowest legal
index". This is not a tie-break, it is the criterion being vacuous.

Two changes fix it, both cheap and both interpretable:

1. **A Beta(1,1) posterior instead of the raw support mean.**
   ``p = (n_pos + 1) / (n + 2)``. On a pure support the posterior is interior, so
   entropy is non-zero and IG is graded again. It also does the statistically
   right thing for free: a *small* pure support is weak evidence and retains more
   entropy than a large one (n=1 → 0.918 bits, n=100 → 0.080 bits), so the
   criterion naturally stops trusting a 3-record agreement.

2. **EVOI stopping rather than entropy stopping.** Selection uses the *expected
   improvement in the actual terminal utility* rather than an entropy proxy:

       u(s)   = E_{y~S}[ 1 - (p_s - y)^2 ]              (Brier-based, per §10)
       u(s,j) = sum_v P(v|s,j) * E_{y~S_v}[ 1 - (p_sjv - y)^2 ]
       gain_j = u(s,j) - u(s)

   This is the decision-theoretic quantity the reward already measures, so the
   stopping rule is consistent with the objective rather than a heuristic
   threshold. ``STOP`` is returned when ``max_j gain_j < lambda * c_j`` — i.e.
   when the best remaining question cannot repay its acquisition cost. At
   ``lambda = 0`` that condition is never met, so the policy never stops early
   unless the item pool is exhausted — which makes AB-7 (fixed-length vs adaptive
   stopping) measurable rather than confounded with item selection.

Interpretability
----------------
``explain()`` returns the posterior, its credible interval, the per-item gain, the
entropy-based IG for comparison against ``GreedyIGPolicy``, and the stop margin.
That is the P3-1 explainability hook built in from the start rather than
retrofitted, and every number in it is one the policy actually used to decide.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

STOP = -1

#: Beta(a0, b0) prior pseudo-counts. Beta(1,1) is the uniform prior and is the
#: weakest assumption that still yields an interior posterior.
BETA_A0 = 1.0
BETA_B0 = 1.0


def entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def beta_interval(n_pos: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Normal-approximation credible interval for a Beta posterior.

    Adequate for the support sizes seen here (median 65) and, unlike an exact
    Beta quantile, costs nothing. Deliberately an *interval* rather than a point:
    the explainability goal is to make support collapse visible, and a smoothed
    point estimate hides it.
    """
    a = n_pos + BETA_A0
    b = (n - n_pos) + BETA_B0
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    half = 1.959963985 * math.sqrt(max(var, 0.0))
    return (max(0.0, mean - half), min(1.0, mean + half))


class BetaGreedyPolicy:
    """Adaptive item selection and stopping on a Beta-smoothed empirical posterior.

    Parameters
    ----------
    records:
        Training records. The support is conditioned on them; the policy holds no
        neural weights, so it is fully interpretable and cheap to evaluate.
    n_items:
        Instrument size.
    lambda_cost:
        Cost per question in utility units, matching §11.1's
        ``R = (1 - (p_hat - y)^2) - lambda * sum_j c_j``. At 0 the policy never
        stops early, which is what makes AB-7 separable.
    costs:
        Per-item costs. Defaults to uniform 1.
    select_by:
        ``"evoi"`` (default) selects on expected utility gain; ``"ig"`` selects on
        smoothed entropy information gain, which is the criterion
        ``GreedyIGPolicy`` uses but with a non-degenerate posterior.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        n_items: int,
        lambda_cost: float = 0.0,
        costs: Sequence[float] | None = None,
        select_by: str = "evoi",
    ):
        if select_by not in ("evoi", "ig"):
            raise ValueError("select_by must be 'evoi' or 'ig'")
        self.n_items = n_items
        self.lambda_cost = float(lambda_cost)
        self.costs = (np.ones(n_items) if costs is None
                      else np.asarray(costs, dtype=float))
        if self.costs.shape != (n_items,):
            raise ValueError("costs must have one entry per item")
        if np.any(self.costs < 0):
            raise ValueError("costs must be non-negative")
        self.select_by = select_by

        self.X = np.stack([r["item_responses"] for r in records])
        self.missing_mask = np.stack([r["missing_mask"] for r in records])
        self.y = np.array([r["label"] for r in records], dtype=float)
        self.n_train = len(records)

    # -- posterior helpers -------------------------------------------------
    def support(self, mask, value) -> np.ndarray:
        """Training records consistent with the observed state."""
        idx = np.ones(self.n_train, dtype=bool)
        for j in range(self.n_items):
            m = mask[j]
            if m == 1:
                idx &= (~self.missing_mask[:, j]) & (self.X[:, j] == value[j])
            elif m == 2:
                idx &= self.missing_mask[:, j]
            if not idx.any():
                break
        return np.where(idx)[0]

    def _posterior(self, idx: np.ndarray) -> Tuple[float, float, int]:
        """Return ``(posterior_mean, expected_utility, n_pos)`` for a support set."""
        n = len(idx)
        if n == 0:
            return 0.5, 1.0 - (0.5 - 0.5) ** 2, 0
        n_pos = int(self.y[idx].sum())
        p = (n_pos + BETA_A0) / (n + BETA_A0 + BETA_B0)
        ys = self.y[idx]
        # E_y[ 1 - (p - y)^2 ] averaged over the support's own label distribution.
        util = float(np.mean(1.0 - (p - ys) ** 2))
        return float(p), util, n_pos

    # -- candidate scoring -------------------------------------------------
    def _child_stats(self, idx: np.ndarray, j: int):
        """Split the support on item ``j`` into observed-value and missing branches.

        Returns ``[(weight, p_child, util_child, n_child, n_pos_child), ...]``.
        The missing branch is included because observing "this item was not
        answered" is itself evidence, matching ``ExactDP``'s treatment.
        """
        total = len(idx)
        out = []
        observed = idx[~self.missing_mask[idx, j]]
        if len(observed):
            for v in np.unique(self.X[observed, j]):
                child = observed[self.X[observed, j] == v]
                p_c, u_c, npos_c = self._posterior(child)
                out.append((len(child) / total, p_c, u_c, len(child), npos_c))
        missing = idx[self.missing_mask[idx, j]]
        if len(missing):
            p_c, u_c, npos_c = self._posterior(missing)
            out.append((len(missing) / total, p_c, u_c, len(missing), npos_c))
        return out

    def scores(self, state, legal) -> Dict[str, Dict[int, float]]:
        """Per-legal-item criterion values and the state posterior.

        Returns a dict with ``posterior``, ``utility``, ``support_size``,
        ``evoi`` and ``ig`` maps, so callers (benchmarks, explanation layers,
        tests) never have to recompute them inconsistently.
        """
        mask, value = state["mask"], state["value"]
        idx = self.support(mask, value)
        p_s, u_s, n_pos = self._posterior(idx)
        h_s = entropy(p_s)

        evoi: Dict[int, float] = {}
        ig: Dict[int, float] = {}
        for a in legal:
            if a == STOP:
                continue
            branches = self._child_stats(idx, a)
            if not branches:
                # Item unobserved for every consistent record: asking reveals
                # nothing about Y, so its gain is exactly zero.
                evoi[a] = 0.0
                ig[a] = 0.0
                continue
            u_after = sum(w * u_c for (w, p_c, u_c, _n, _np_) in branches)
            evoi[a] = u_after - u_s
            ig[a] = h_s - sum(w * entropy(p_c) for (w, p_c, _u, _n, _np_) in branches)

        return {
            "posterior": p_s,
            "utility": u_s,
            "support_size": len(idx),
            "n_pos": n_pos,
            "pure_support": len(idx) > 0 and n_pos in (0, len(idx)),
            "entropy": h_s,
            "evoi": evoi,
            "ig": ig,
        }

    def criterion(self, stats) -> Dict[int, float]:
        key = "evoi" if self.select_by == "evoi" else "ig"
        return stats[key]

    # -- policy protocol ---------------------------------------------------
    def __call__(self, state, legal):
        items_only = [a for a in legal if a != STOP]
        if not items_only:
            return STOP
        stats = self.scores(state, legal)
        crit = self.criterion(stats)

        best = max(items_only, key=lambda a: (crit[a], -a))
        if STOP in legal and self.lambda_cost > 0.0:
            # Stop when the best remaining question cannot repay its cost.
            if crit[best] < self.lambda_cost * float(self.costs[best]):
                return STOP
        return best

    # -- explanation -------------------------------------------------------
    def explain(self, state, legal) -> Dict[str, Any]:
        """Full rationale for the decision taken at ``state``.

        Every field is a number the policy actually used. This is the P3-1 hook,
        designed in rather than retrofitted.
        """
        stats = self.scores(state, legal)
        crit = self.criterion(stats)
        items_only = [a for a in legal if a != STOP]
        action = self(state, legal)
        lo, hi = beta_interval(stats["n_pos"], stats["support_size"])
        ranked = sorted(items_only, key=lambda a: (-crit[a], a))
        out = {
            "selected": action,
            "selected_item": f"A{action + 1}" if action != STOP else "STOP",
            "posterior": stats["posterior"],
            "posterior_ci95": [lo, hi],
            "support_size": stats["support_size"],
            "support_is_pure": stats["pure_support"],
            "entropy_bits": stats["entropy"],
            "utility": stats["utility"],
            "criterion": self.select_by,
            "lambda_cost": self.lambda_cost,
            "criterion_values": {f"A{a + 1}": crit[a] for a in items_only},
            "evoi_values": {f"A{a + 1}": stats["evoi"][a] for a in items_only},
            "ig_values": {f"A{a + 1}": stats["ig"][a] for a in items_only},
            "ranking": [f"A{a + 1}" for a in ranked],
        }
        if items_only:
            best = ranked[0]
            out["runner_up"] = f"A{ranked[1] + 1}" if len(ranked) > 1 else None
            out["margin_over_runner_up"] = (
                crit[best] - crit[ranked[1]] if len(ranked) > 1 else None
            )
            out["cost_of_selected"] = float(self.costs[best])
            out["worth_asking"] = bool(crit[best] >= self.lambda_cost * float(self.costs[best]))
        if action == STOP:
            out["stop_margin"] = max(crit.values()) if crit else 0.0
        return out
