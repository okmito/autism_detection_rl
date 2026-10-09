"""Counterfactual — §18.2, outcome-level explainability (P3-outcome).

Two layers live here:

``find_counterfactual``
    The original §18.2 function, **unchanged**: minimum-cost single-response
    flip among *observed* items that crosses the operating threshold. Retained
    so existing callers and artifacts behave identically.

``find_counterfactual_rich``
    The outcome-explanation layer's counterfactual analysis. Same validity
    guarantee (every reported flip is re-applied and checked), plus:

    * **feasibility** — each flip scored by its plausibility under a factorised
      item prior, so a mathematically valid flip that no real respondent would
      give is visible as implausible rather than presented as an alternative
      answer;
    * **minimal set** — smallest set of simultaneous flips that changes the
      decision (fewest flips, then minimum cost);
    * **unask** — what the estimate would be if a question had not been asked
      (the item is marginalised out, not flipped);
    * **ask-more** — what each still-askable question would do to the estimate
      if answered atypically / typically.

    Wording contract (spec §25, terminology lock): a counterfactual describes a
    change to *the model's output given different answers*. It is not a
    statement about a person's underlying condition, and never clinical advice.
    Every consumer of this module must present it with that framing.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Sequence
import copy
import itertools

import numpy as np

from src.env.state import UNASKED, OBSERVED, MISSING


def find_counterfactual(state: Dict[str, Any], p_hat: float, predictor, tau: float = 0.5, costs=None) -> Optional[Dict[str, Any]]:
    """Search observed items for minimum single-response flip that crosses tau.
    Returns dict with item, original_value, flipped_value, new_p if found, else None (robust).
    Tie-break: minimum cost (if costs provided) else minimum index.
    """
    decision = 1 if p_hat >= tau else 0
    candidates = []
    for j in range(state["n"]):
        if state["mask"][j] != 1:
            continue
        orig = int(state["value"][j])
        # For binary primary, flip 0->1,1->0; for multi-category, try all other values (assume binary for now)
        for flipped in [0,1]:
            if flipped == orig:
                continue
            # create perturbed state
            s2 = {"mask": state["mask"].copy(), "value": state["value"].copy(), "n": state["n"],
                  "questions_remaining": state.get("questions_remaining",0), "budget": state.get("budget", state["n"])}
            s2["value"][j] = flipped
            p2 = float(predictor(s2))
            dec2 = 1 if p2 >= tau else 0
            if dec2 != decision:
                cost = costs[j] if costs is not None else 1
                candidates.append((cost, j, orig, flipped, p2))
    if not candidates:
        return None  # robust
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, j, orig, flipped, p2 = candidates[0]
    return {"item": f"A{j+1}", "item_idx": j, "original_value": orig, "flipped_value": flipped, "new_p": float(p2)}


# ---------------------------------------------------------------------------
# Outcome-explanation counterfactuals (P3-outcome)
# ---------------------------------------------------------------------------

def _decide(p: float, tau: float) -> int:
    return 1 if p >= tau else 0


def _perturbed(state: Dict[str, Any], changes: Dict[int, Optional[int]]) -> Dict[str, Any]:
    """Copy of ``state`` with the requested item changes applied.

    ``changes[idx] = v``  -> that item is OBSERVED with response v (legal for
    both an observed item and an ask-more candidate).
    ``changes[idx] = None`` -> that item is set UNASKED (marginalised out).
    MISSING items may never be assigned a response.
    The input state is never mutated.
    """
    mask = np.asarray(state["mask"], dtype=int).copy()
    value = np.asarray(state["value"], dtype=int).copy()
    for j, v in changes.items():
        if v is None:
            if mask[j] == OBSERVED:
                mask[j] = UNASKED
                value[j] = -1
        else:
            if mask[j] == MISSING:
                raise ValueError(
                    f"item {j} is MISSING (structurally absent); no response "
                    f"can be assigned to it")
            mask[j] = OBSERVED
            value[j] = int(v)
    return {"mask": mask, "value": value, "n": state["n"],
            "questions_remaining": state.get("questions_remaining", 0),
            "budget": state.get("budget", state["n"])}


def _prior_plausibility_ratio(prior, item_idx: int, orig: int, flipped: int) -> Optional[float]:
    """P(flipped)/P(original) for one item under a factorised Bernoulli prior.

    ``prior[j]`` is P(item j answered atypically) on the training data. Returns
    None when no prior is supplied — the caller then reports feasibility as
    unavailable rather than inventing one.
    """
    if prior is None:
        return None
    q = float(np.clip(prior[item_idx], 1e-6, 1 - 1e-6))

    def pmf(v: int) -> float:
        return q if v == 1 else 1.0 - q

    return float(pmf(flipped) / pmf(orig))


def _pattern_seen_in_reference(reference_rows, mask: np.ndarray,
                               value: np.ndarray) -> Optional[bool]:
    """Does any reference (training) response vector agree on the determined items?

    A flip that produces a combination absent from training is an
    extrapolation for the model — on these datasets the label is a
    deterministic sum-threshold, so such combinations exist. Reported, never
    filtered out.
    """
    if reference_rows is None:
        return None
    ref = np.asarray(reference_rows, dtype=float)
    if ref.ndim != 2:
        raise ValueError("reference_rows must be a 2-D (n_records, n_items) array")
    determined = [j for j in range(mask.size) if mask[j] == OBSERVED]
    if not determined:
        return True
    matches = np.all(ref[:, determined] == np.asarray(value, dtype=float)[determined],
                     axis=1)
    return bool(matches.any())


def find_counterfactual_rich(
    state: Dict[str, Any],
    predictor,
    tau: float = 0.5,
    costs=None,
    prior=None,
    max_flips: int = 2,
    item_value_choices: Optional[Dict[int, Sequence[int]]] = None,
    reference_rows=None,
    include_ask_more: bool = True,
) -> Dict[str, Any]:
    """Full counterfactual analysis of the outcome for one terminal state.

    Every probability reported is recomputed by the predictor; every flip is
    verified by construction (the perturbed state is what produced it). The
    result describes **model behaviour under different answers** — never a
    statement about the person, and never clinical advice.

    Parameters
    ----------
    state:
        Terminal state of the episode (``run_episode``'s ``final_state``).
    predictor:
        Object with ``predict_state`` (or a callable) — the same predictor that
        produced the outcome.
    tau:
        Operating threshold (frozen, spec §10).
    costs:
        Optional per-item costs for the minimal-set tie-break.
    prior:
        Optional per-item P(atypical) on training data; enables the
        plausibility ratio. Absent -> reported as None, not guessed.
    max_flips:
        Largest simultaneous-flip set searched for the minimal set.
    item_value_choices:
        Optional {item_idx: [value, ...]} for multi-category instruments;
        defaults to the binary {0, 1} flip.
    reference_rows:
        Optional (n_records, n_items) training matrix; enables the
        seen-in-training flag.
    include_ask_more:
        Include the ask-one-more-question analysis for UNASKED items.
    """
    predict = getattr(predictor, "predict_state", predictor)
    n = int(state["n"])
    mask0 = np.asarray(state["mask"], dtype=int)
    value0 = np.asarray(state["value"], dtype=int)
    observed = [j for j in range(n) if mask0[j] == OBSERVED]
    unasked = [j for j in range(n) if mask0[j] == UNASKED]
    cost_vec = np.ones(n) if costs is None else np.asarray(costs, dtype=float)

    def choices_for(j: int) -> List[int]:
        if item_value_choices and j in item_value_choices:
            return [int(v) for v in item_value_choices[j]]
        return [0, 1]

    p0 = float(predict(state))
    d0 = _decide(p0, tau)

    # ---- single flips -----------------------------------------------------
    single_flips: List[Dict[str, Any]] = []
    for j in observed:
        orig = int(value0[j])
        for fv in choices_for(j):
            if fv == orig:
                continue
            s2 = _perturbed(state, {j: fv})
            p2 = float(predict(s2))
            single_flips.append({
                "item": f"A{j + 1}", "item_idx": j,
                "original_value": orig, "flipped_value": int(fv),
                "new_p": p2, "new_decision": _decide(p2, tau),
                "flips_decision": _decide(p2, tau) != d0,
                "delta_p": p2 - p0,
                "prior_plausibility_ratio": _prior_plausibility_ratio(prior, j, orig, fv),
                "pattern_seen_in_reference": _pattern_seen_in_reference(
                    reference_rows, s2["mask"], s2["value"]),
                "cost": float(cost_vec[j]),
            })
    single_flips.sort(key=lambda e: (not e["flips_decision"], e["cost"], e["item_idx"]))
    robust = not any(f["flips_decision"] for f in single_flips)

    # ---- minimal set -------------------------------------------------------
    minimal: Optional[Dict[str, Any]] = None
    for r in range(1, max_flips + 1):
        best_key = None
        best: Optional[Dict[str, Any]] = None
        for combo in itertools.combinations(observed, r):
            alt_lists = [[v for v in choices_for(j) if v != int(value0[j])]
                         for j in combo]
            if any(not alts for alts in alt_lists):
                continue
            for values in itertools.product(*alt_lists):
                changes = {j: v for j, v in zip(combo, values)}
                s2 = _perturbed(state, changes)
                p2 = float(predict(s2))
                if _decide(p2, tau) != d0:
                    cost_sum = float(sum(cost_vec[j] for j in combo))
                    key = (r, cost_sum, combo)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = {
                            "flips": [{"item": f"A{j + 1}", "item_idx": j,
                                       "from": int(value0[j]), "to": int(v)}
                                      for j, v in changes.items()],
                            "new_p": p2, "new_decision": _decide(p2, tau),
                            "n_flips": r, "total_cost": cost_sum,
                        }
        if best is not None:
            minimal = best
            break

    # ---- unask -------------------------------------------------------------
    unask: List[Dict[str, Any]] = []
    for j in observed:
        s2 = _perturbed(state, {j: None})
        p2 = float(predict(s2))
        unask.append({"item": f"A{j + 1}", "item_idx": j,
                      "p_hat_without": p2, "decision_without": _decide(p2, tau),
                      "delta_p": p2 - p0,
                      "decision_change": _decide(p2, tau) != d0})
    unask.sort(key=lambda e: -abs(e["delta_p"]))

    # ---- ask one more ------------------------------------------------------
    ask_more: List[Dict[str, Any]] = []
    if include_ask_more:
        for j in unasked:
            options: Dict[str, float] = {}
            for v in choices_for(j):
                s2 = _perturbed(state, {j: v})
                options[str(v)] = float(predict(s2))
            ask_more.append({"item": f"A{j + 1}", "item_idx": j,
                             "options": options})

    return {
        "p_hat": p0, "decision": d0, "tau": float(tau),
        "single_flips": single_flips,
        "robust_to_all_single_flips": robust,
        "minimal_set": minimal,
        "unask": unask,
        "ask_more": ask_more,
        "feasibility_prior_available": prior is not None,
        "reference_rows_available": reference_rows is not None,
        "framing_note": (
            "Each entry describes how the model's screening estimate would "
            "change under different answers. It is not a statement about the "
            "person's underlying condition and not clinical advice."),
    }
