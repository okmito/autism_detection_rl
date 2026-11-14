"""Question payloads and the selection diagnostic, shared by the web app and the demo server.

This code used to exist twice (``scripts/demo_app.py`` and ``app/services/screening.py``); the copies
had already drifted once (audit finding C4/T1). The demo's documented version is the canonical one.
"""
from __future__ import annotations

import numpy as np

from src.env.environment import STOP
from src.policies.greedy import entropy
from src.support.questionnaire import ITEMS as _QCHAT_ITEMS

# The wording is derived from the verified Q-CHAT-10 feature contract
# (src/data/qchat10_contract.py) rather than restated here; tests/test_support_questionnaire.py
# asserts this list equals the contract's items.
ITEMS = [item.question_text for item in _QCHAT_ITEMS]


def item_payload(j: int) -> dict:
    return {"index": int(j), "code": f"A{j + 1}", "text": ITEMS[int(j)]}


# --------------------------------------------------------------------------
# selection diagnostic
# --------------------------------------------------------------------------
# Why this exists
# ---------------
# The demo asks a viewer to watch which question the policy picks next. On this
# dataset it very often picks the lowest-numbered unasked item in order, and
# without an explanation that reads as a broken policy.
#
# It is not a bug, and this is the measured reason. ``GreedyIGPolicy`` selects on
# ``IG = H(Y|s) - E_v[H(Y|s,j,v)]`` over the *raw* empirical support mean. Once
# the support is **pure** — every training record consistent with the answers so
# far carries the same label — ``H(Y|s) = 0`` and every child support is pure too,
# so **every legal item's IG is exactly 0**. ``argmax`` then returns the first
# maximum, which is ``greedy.py``'s ``best_j = items_only[0]`` initialisation.
# The criterion is vacuous, not tied.
#
# Measured on the held-out test split at B=6: the support is already pure at
# 61.3% of decision states, and the criterion is identically zero at 195 of 195
# pure-support states (`POLICY_BENCHMARK_REPORT.md` §A.6, `diagnosisReady.md`
# §0A.1). A Beta(1,1)-smoothed posterior fixes the degeneracy
# (`src/policies/beta_greedy.py`) but the demo deliberately still runs the
# shipped greedy policy, so the UI reports the degeneracy rather than hiding it.
#
# ``GreedyIGPolicy`` is intentionally NOT modified, so this recomputes its
# formula rather than calling into it. ``tests/test_demo_behavior_audit.py::
# test_selection_diagnostic_agrees_with_policy_choice`` asserts the two agree,
# which is what keeps this from drifting away from the real policy.
DEGENERACY_TOL = 1e-12


def selection_diagnostic(policy, state, legal) -> dict:
    """Explain the item choice at ``state``: support size, purity, IG spread."""
    items_only = [a for a in legal if a != STOP]
    out = {
        "applicable": hasattr(policy, "_support"),
        "n_legal": len(items_only),
        "support_size": None, "n_pos": None, "posterior": None,
        "support_is_pure": None, "ig_spread": None, "criterion_vacuous": None,
        "tie_at_max": None, "top_items": [],
    }
    if not out["applicable"] or not items_only:
        out["criterion_vacuous"] = None
        return out

    idx = policy._support(state["mask"], state["value"])
    out["support_size"] = int(len(idx))
    if len(idx) == 0:
        out["criterion_vacuous"] = True
        return out

    ys = policy.y[idx]
    total = len(idx)
    n_pos = int(ys.sum())
    p_s = float(ys.mean())
    h_s = entropy(p_s)
    out["n_pos"] = n_pos
    out["posterior"] = p_s
    out["support_is_pure"] = bool(n_pos in (0, total))

    ig: dict[int, float] = {}
    for j in items_only:
        sub = idx[~policy.missing_mask[idx, j]]
        if len(sub) == 0:
            continue
        exp_h = 0.0
        vals, counts = np.unique(policy.X[sub, j], return_counts=True)
        for v, c in zip(vals, counts):
            child = idx[(~policy.missing_mask[idx, j]) & (policy.X[idx, j] == v)]
            if len(child) == 0:
                continue
            exp_h += (c / total) * entropy(float(policy.y[child].mean()))
        n_miss = int(np.sum(policy.missing_mask[idx, j]))
        if n_miss > 0:
            child = idx[policy.missing_mask[idx, j]]
            p_child = float(policy.y[child].mean()) if len(child) > 0 else 0.5
            exp_h += (n_miss / total) * entropy(p_child)
        ig[j] = h_s - exp_h

    if not ig:
        out["criterion_vacuous"] = True
        return out

    best = max(ig.values())
    worst = min(ig.values())
    out["ig_spread"] = best - worst
    # A spread of zero means every scored item is indistinguishable, so argmax
    # carries no information and the lowest legal index wins by default.
    out["criterion_vacuous"] = bool((best - worst) <= DEGENERACY_TOL)
    tied = sorted(j for j, v in ig.items() if best - v <= DEGENERACY_TOL)
    out["tie_at_max"] = len(tied)
    out["top_items"] = [{"code": f"A{j + 1}", "ig": ig[j]} for j in tied[:6]]
    return out


def selection_summary(trace: list) -> dict:
    """Aggregate the per-step diagnostics into one honest headline number."""
    steps = [t.get("selection") or {} for t in trace]
    scored = [s for s in steps if s.get("ig_spread") is not None]
    vacuous = [s for s in scored if s.get("criterion_vacuous")]
    pure = [s for s in steps if s.get("support_is_pure") is True]
    return {
        "steps_diagnosed": len(scored),
        "steps_criterion_vacuous": len(vacuous),
        "steps_support_pure": len(pure),
        "support_pure_at_stop": (pure[-1]["support_is_pure"] if pure else None),
        "support_size_at_stop": (steps[-1].get("support_size") if steps else None),
    }
