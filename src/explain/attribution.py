"""Exact group-Shapley attribution over questionnaire items — spec §18.3/§18.4.

What this module explains
-------------------------
The **final screening outcome**: the calibrated probability ``p_hat`` the
predictor produces for the terminal state of an interview, and the referral
decision derived from it. It answers *which observed responses moved the
model's risk estimate, and by how much*.

Method
------
Shapley values (Shapley, 1953; Lundberg & Lee, 2017) are computed **exactly by
coalition enumeration** over the questionnaire items, under a value function
defined by the predictor itself:

    v(S) = predictor( state with items in S held at their observed responses,
                      all other attributed items set UNASKED )

"Absent" means UNASKED — not "zero", not "dropped". The predictor's own
partial-state semantics then define absent: for ``LogisticPredictor`` (v3) an
UNASKED item is marginalised exactly over its 2-item support under the
factorised Saudi-train prior; for ``MaskedPredictor`` (v2) it is whatever the
network was trained to do with an unobserved item. The attribution therefore
explains the movement *from the model's prior-only estimate* v(∅) *to the
final estimate* v(all observed), and inherits whatever assumption that
partial-state semantics makes — which is why the uncertainty module reports
prior sensitivity separately.

Why exact enumeration is tractable here
---------------------------------------
A Q-CHAT-10 interview observes at most ``B <= 6`` items (and the instrument
has 10), so enumerating all coalitions costs at most 2^10 = 1024 predictor
evaluations, each a table lookup (v3) or a 41-dim MLP forward pass (v2). No
sampling (LIME, KernelSHAP) is needed, the result is deterministic, and the
Shapley **efficiency** identity

    sum_j phi_j == v(full) - v(empty)

holds to floating-point and is asserted in the tests.

Interpretation caveat (stated, not hidden)
-------------------------------------------
* phi is on the **probability scale** of the screening estimate. ``phi > 0``
  means "given this response, the model's estimate is higher than it would be
  with the other observed responses held fixed and this one unobserved".
* Items are correlated by construction on these datasets (the Saudi label is
  a deterministic sum-threshold, so responses co-vary with the label). Shapley
  values **split credit among correlated items**; they are not causal effects
  of a response on a child's traits, and the output schema says so.
* This explains the *model*. It says nothing about clinical validity.

The deprecated ``shap_baseline`` module (single-flip deltas, non-additive) is
NOT used here; see its docstring and ``tests/test_shap_baseline_honesty.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.env.state import UNASKED, OBSERVED, MISSING

#: 2^12 = 4096 coalition evaluations is still sub-second for these predictors;
#: beyond this the exact method stops being cheap and a sampling method should
#: be used deliberately instead.
MAX_ATTRIBUTED_ITEMS = 12

_TOL = 1e-12


@dataclass
class ItemAttribution:
    """Shapley attribution of one observed questionnaire response."""
    item: str                 # human-facing code, e.g. "A3"
    item_idx: int             # 0-based index into the instrument
    response: int             # the value actually observed
    phi: float                # Shapley value on the probability scale
    rank: int                 # 1 = largest |phi|
    direction: str            # "raises" | "lowers" | "neutral"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "item_idx": self.item_idx,
            "response": self.response,
            "phi": float(self.phi),
            "direction": self.direction,
            "rank": int(self.rank),
        }


@dataclass
class AttributionResult:
    """Full result of an exact group-Shapley computation for one state."""
    items: List[ItemAttribution]
    baseline_p: float         # v(empty): prior-only estimate
    full_p: float             # v(all observed): the episode's p_hat
    n_attributed: int = 0
    n_evaluations: int = 0    # predictor evaluations actually performed
    value_function: str = "predictor_partial_state_value_function"

    @property
    def additivity_error(self) -> float:
        """|sum(phi) - (v(full) - v(empty))| — the efficiency identity."""
        return abs(sum(it.phi for it in self.items)
                   - (self.full_p - self.baseline_p))

    def ranked(self) -> List[ItemAttribution]:
        """Items sorted by descending |phi| (ties broken by item index)."""
        return sorted(self.items, key=lambda a: (-abs(a.phi), a.item_idx))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": "exact_group_shapley",
            "value_function": self.value_function,
            "baseline_p": float(self.baseline_p),
            "full_p": float(self.full_p),
            "n_attributed": int(self.n_attributed),
            "n_evaluations": int(self.n_evaluations),
            "additivity_error": float(self.additivity_error),
            "contributions": [a.as_dict() for a in self.ranked()],
        }


def _predict_fn(predictor):
    """Accept either a MaskedPredictor/LogisticPredictor or a bare callable."""
    fn = getattr(predictor, "predict_state", None)
    return fn if callable(fn) else predictor


def _make_value_fn(state: Dict[str, Any], predictor, items: Sequence[int]):
    """Build the coalition value function v(S) for the given observed items.

    The returned callable takes a frozenset of item indices that are held at
    their observed responses; every other attributed item is set UNASKED.
    MISSING items are never touched — they are structurally absent and are
    marginalised by the predictor's own semantics in both predictor versions.
    """
    predict = _predict_fn(predictor)
    n = int(state["n"])
    mask0 = np.asarray(state["mask"], dtype=int)
    value0 = np.asarray(state["value"], dtype=int)
    observed_value = {int(j): int(value0[j]) for j in items}
    questions_remaining = state.get("questions_remaining", 0)
    budget = state.get("budget", max(questions_remaining, 1))

    n_calls = 0

    def value(present: frozenset) -> float:
        nonlocal n_calls
        n_calls += 1
        mask = mask0.copy()
        value = value0.copy()
        for j in items:
            if j not in present:
                mask[j] = UNASKED
                value[j] = -1
        probe = {
            "mask": mask,
            "value": value,
            "n": n,
            "questions_remaining": questions_remaining,
            "budget": budget,
        }
        return float(predict(probe))

    value.n_calls = lambda: n_calls  # type: ignore[attr-defined]
    return value


def exact_shapley(
    state: Dict[str, Any],
    predictor,
    items: Optional[Sequence[int]] = None,
    max_items: int = MAX_ATTRIBUTED_ITEMS,
) -> AttributionResult:
    """Exact Shapley values for the observed items of ``state``.

    Parameters
    ----------
    state:
        The episode's terminal state (as returned by ``run_episode`` under
        ``final_state``). Read-only: never mutated.
    predictor:
        Any object exposing ``predict_state(state)`` (both predictor versions
        do) or a plain callable with that signature.
    items:
        Item indices to attribute. Default: every OBSERVED item in the state.
        MISSING items are always excluded.
    max_items:
        Hard guard on coalition count (2^max_items evaluations).

    Returns
    -------
    AttributionResult with per-item ``phi`` on the probability scale, ranked
    by magnitude, plus the efficiency identity's residual.
    """
    mask = np.asarray(state["mask"], dtype=int)
    if items is None:
        items = [j for j in range(int(state["n"])) if mask[j] == OBSERVED]
    items = [int(j) for j in items]
    if not items:
        predict = _predict_fn(predictor)
        p = float(predict(state))
        return AttributionResult(items=[], baseline_p=p, full_p=p,
                                 n_attributed=0, n_evaluations=1)
    if len(set(items)) != len(items):
        raise ValueError("items must be unique")
    for j in items:
        if j < 0 or j >= int(state["n"]):
            raise ValueError(f"item index {j} out of range")
        if mask[j] != OBSERVED:
            raise ValueError(
                f"item {j} is not OBSERVED (mask={mask[j]}); only observed "
                f"responses can be attributed — MISSING/UNASKED items are "
                f"marginalised by the predictor, not attributed")
    if len(items) > max_items:
        raise ValueError(
            f"{len(items)} items requested but exact enumeration caps at "
            f"{max_items} (2^{max_items} evaluations); choose a subset or use a "
            f"sampling-based method deliberately")

    value = _make_value_fn(state, predictor, items)
    k = len(items)

    # Enumerate v over all 2^k coalitions once, then combine.
    vals = np.empty(1 << k, dtype=float)
    for bits in range(1 << k):
        present = frozenset(items[i] for i in range(k) if (bits >> i) & 1)
        vals[bits] = value(present)

    def v_of(bit_indices: frozenset) -> float:
        bits = 0
        for i in bit_indices:
            bits |= 1 << i
        return vals[bits]

    # Closed-form Shapley over positional indices 0..k-1.
    phi = np.zeros(k, dtype=float)
    fact = math.factorial
    for i in range(k):
        others = [x for x in range(k) if x != i]
        total = 0.0
        for r in range(k):
            weight = fact(r) * fact(k - r - 1) / fact(k)
            for subset in combinations(others, r):
                s_without = frozenset(subset)
                s_with = frozenset(subset + (i,))
                total += weight * (v_of(s_with) - v_of(s_without))
        phi[i] = total

    attributions: List[ItemAttribution] = []
    for i, j in enumerate(items):
        attributions.append(ItemAttribution(
            item=f"A{j + 1}",
            item_idx=j,
            response=int(np.asarray(state["value"], dtype=int)[j]),
            phi=float(phi[i]),
            rank=0,
            direction=_direction(phi[i]),
        ))
    ranked = sorted(attributions, key=lambda a: (-abs(a.phi), a.item_idx))
    for rank, a in enumerate(ranked, start=1):
        a.rank = rank

    return AttributionResult(
        items=attributions,
        baseline_p=float(vals[0]),
        full_p=float(vals[(1 << k) - 1]),
        n_attributed=k,
        n_evaluations=value.n_calls(),
    )


def _direction(phi: float) -> str:
    if phi > _TOL:
        return "raises"
    if phi < -_TOL:
        return "lowers"
    return "neutral"
