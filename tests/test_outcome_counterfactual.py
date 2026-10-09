"""Tests for the outcome-level counterfactual layer (P3-outcome).

Validates: single-flip validity (re-applied flips behave as reported),
robustness detection, minimal-set minimality, unask/ask-more semantics against
independent recomputation, prior-plausibility arithmetic, seen-in-training
flags, and the read-only contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.state import UNASKED, OBSERVED, MISSING, init_state, update_state
from src.explain.counterfactual import find_counterfactual_rich


def _terminal_state(observed: dict, n: int = 10, budget: int = 6):
    rec = {"item_responses": np.zeros(n, dtype=float), "label": 1,
           "label_source": "questionnaire",
           "covariates": {"age_band": "1-2", "sex": "M"},
           "missing_mask": np.zeros(n, dtype=bool)}
    s = init_state(rec)
    s["questions_remaining"] = budget - len(observed)
    s["budget"] = budget
    for k, (j, v) in enumerate(observed.items()):
        s = update_state(s, j, v, budget - k - 1)
        s["questions_remaining"] = budget - k - 1
        s["budget"] = budget
    return s


class _LinearStub:
    """p = clip(0.05 * sum(observed), 0, 1); absent items contribute nothing."""

    def predict_state(self, state):
        s = sum(int(state["value"][j]) for j in range(state["n"])
                if state["mask"][j] == OBSERVED)
        return min(0.05 * s, 1.0)


def _threshold_stub(state):
    """0.2 + 0.25 * (#atypical observed): crosses 0.5 at 2 atypical answers."""
    s = sum(int(state["value"][j]) for j in range(state["n"])
            if state["mask"][j] == OBSERVED)
    return 0.2 + 0.25 * s


def test_single_flip_validity_is_verified_by_recomputation():
    # 4 atypical -> p=1.0; any single flip to typical -> 0.75: still >= 0.5
    s = _terminal_state({0: 1, 1: 1, 2: 1, 3: 1})
    out = find_counterfactual_rich(s, _LinearStub(), tau=0.5)
    assert out["p_hat"] == pytest.approx(0.2)
    assert out["decision"] == 0
    # make p above threshold: 12 observed atypical would exceed n=10, so use a
    # stub where 4 atypical crosses: switch predictor
    out2 = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert out2["p_hat"] == pytest.approx(1.2)  # 0.2+0.25*4
    assert out2["decision"] == 1
    # flipping any one item to typical: 0.95 -> still REFER; robust
    assert all(f["flips_decision"] is False for f in out2["single_flips"])
    assert out2["robust_to_all_single_flips"] is True
    # but flipping one of them 0->? all are already 1; each flip is 1->0
    assert {f["original_value"] for f in out2["single_flips"]} == {1}
    # recompute one entry independently
    f0 = out2["single_flips"][0]
    s2 = _terminal_state({0: 0, 1: 1, 2: 1, 3: 1})
    assert f0["new_p"] == pytest.approx(_threshold_stub(s2))
    assert f0["new_decision"] == 1


def test_flip_that_crosses_is_reported_and_valid():
    # 2 atypical -> p = 0.70; flipping one to typical -> 0.45 crosses tau
    s = _terminal_state({0: 1, 1: 1})
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert out["p_hat"] == pytest.approx(0.70)
    flips = [f for f in out["single_flips"] if f["flips_decision"]]
    assert len(flips) == 2
    assert out["robust_to_all_single_flips"] is False
    for f in flips:
        obs = {0: f["flipped_value"], 1: 1} if f["item_idx"] == 0 else {0: 1, 1: f["flipped_value"]}
        assert f["new_p"] == pytest.approx(_threshold_stub(_terminal_state(obs)))
        assert f["new_decision"] == 0


def test_minimal_set_is_minimal():
    # single flip survives (0.45 stays 0.45... wait: with 1 atypical -> 0.45 < 0.5)
    # So minimal set has exactly one flip when a pair exists: use 3 atypical.
    s = _terminal_state({0: 1, 1: 1, 2: 1})
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    # p = 0.95 -> decision 1; single flip -> 0.70 -> still 1 => no size-1 set
    assert out["minimal_set"] is None or out["minimal_set"]["n_flips"] > 1
    # With 2 atypical, one flip suffices and minimal_set must have n_flips == 1
    s2 = _terminal_state({0: 1, 1: 1})
    out2 = find_counterfactual_rich(s2, _threshold_stub, tau=0.5, max_flips=2)
    assert out2["minimal_set"] is not None
    assert out2["minimal_set"]["n_flips"] == 1
    # verify the reported minimal set really flips the decision
    changes = {f["item_idx"]: f["to"] for f in out2["minimal_set"]["flips"]}
    obs = {0: changes.get(0, 1), 1: changes.get(1, 1)}
    assert _threshold_stub(_terminal_state(obs)) < 0.5


def test_unask_matches_independent_recomputation():
    s = _terminal_state({0: 1, 1: 0, 2: 1})
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert len(out["unask"]) == 3
    for e in out["unask"]:
        obs = {0: 1, 1: 0, 2: 1}
        obs[e["item_idx"]] = None
        obs = {j: v for j, v in obs.items() if v is not None}
        # p with that item UNASKED: rebuild and set it back to unasked
        s2 = _terminal_state({0: 1, 1: 0, 2: 1})
        s2["mask"][e["item_idx"]] = UNASKED
        s2["value"][e["item_idx"]] = -1
        assert e["p_hat_without"] == pytest.approx(_threshold_stub(s2))


def test_ask_more_covers_unasked_items():
    s = _terminal_state({0: 1, 1: 1})
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert len(out["ask_more"]) == 8  # 10 items - 2 observed
    entry = out["ask_more"][0]
    assert set(entry["options"]) == {"0", "1"}
    # each option equals the predictor on the state with that item observed
    for v, p in entry["options"].items():
        s2 = _terminal_state({0: 1, 1: 1})
        s2["mask"][entry["item_idx"]] = OBSERVED
        s2["value"][entry["item_idx"]] = int(v)
        assert p == pytest.approx(_threshold_stub(s2))


def test_prior_plausibility_ratio_arithmetic():
    s = _terminal_state({0: 1})
    prior = np.array([0.2] * 10)  # P(atypical)=0.2
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5, prior=prior)
    f = out["single_flips"][0]
    # flip 1 -> 0: ratio = (1-q)/q = 0.8/0.2 = 4.0
    assert f["prior_plausibility_ratio"] == pytest.approx(4.0)
    # without a prior the field is None, not guessed
    out2 = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert out2["single_flips"][0]["prior_plausibility_ratio"] is None
    assert out2["feasibility_prior_available"] is False


def test_pattern_seen_in_reference():
    s = _terminal_state({0: 1, 1: 1})
    ref = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5, reference_rows=ref)
    # original pattern (1,1) is in reference; flipped (0,1) and (1,0) are not
    seen = {f["item_idx"]: f["pattern_seen_in_reference"] for f in out["single_flips"]}
    assert seen[0] is False and seen[1] is False
    # a reference that also contains the flipped patterns -> True
    ref2 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    out2 = find_counterfactual_rich(s, _threshold_stub, tau=0.5, reference_rows=ref2)
    seen2 = {f["item_idx"]: f["pattern_seen_in_reference"] for f in out2["single_flips"]}
    assert seen2[0] is True and seen2[1] is True
    # no reference supplied -> flag unavailable, not guessed
    out3 = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert all(f["pattern_seen_in_reference"] is None for f in out3["single_flips"])


def test_state_is_not_mutated():
    s = _terminal_state({0: 1, 1: 1, 2: 0})
    mask_before = s["mask"].copy()
    value_before = s["value"].copy()
    find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert np.array_equal(s["mask"], mask_before)
    assert np.array_equal(s["value"], value_before)


def test_missing_item_cannot_be_assigned_a_response():
    rec = {"item_responses": np.zeros(10), "label": 1,
           "label_source": "questionnaire",
           "covariates": {"age_band": "1-2", "sex": "M"},
           "missing_mask": np.array([True] + [False] * 9)}
    s = init_state(rec)
    s["questions_remaining"] = 6
    s["budget"] = 6
    s = update_state(s, 1, 1, 5)
    out = find_counterfactual_rich(s, _threshold_stub, tau=0.5)
    assert all(e["item_idx"] != 0 for e in out["single_flips"])
    assert all(e["item_idx"] != 0 for e in out["ask_more"])


def test_legacy_single_flip_function_still_works():
    """The original §18.2 API is preserved for existing callers."""
    from src.explain.counterfactual import find_counterfactual
    s = _terminal_state({0: 1, 1: 1})
    cf = find_counterfactual(s, _threshold_stub(s), _threshold_stub, tau=0.5)
    assert cf is not None and cf["item"] in ("A1", "A2")
    assert cf["new_p"] < 0.5
