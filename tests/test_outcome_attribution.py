"""Tests for exact group-Shapley attribution — §18.3/§18.4, RQ4.

Four independent validation layers:

1. **Hand-computed toy** — on a perfectly additive value function the Shapley
   value of each item is analytically known; the module must reproduce it.
2. **Efficiency identity** — on the real frozen predictor (v3) over a real
   episode: sum(phi) == p_hat - prior-only baseline, and v(all observed) must
   equal the episode's reported p_hat bit-for-bit (same state, same call).
3. **Cross-method agreement** — against ``shap.explainers.Exact`` (the SHAP
   package, already a project dependency, used here for validation only) on a
   zero-baseline stub, agreement to 1e-6.
4. **Contract/robustness** — determinism, no state mutation, MISSING items are
   never attributed, guards on coalition size and non-observed items.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.environment import run_episode
from src.env.state import UNASKED, OBSERVED, MISSING, init_state, update_state
from src.explain.attribution import (
    MAX_ATTRIBUTED_ITEMS, AttributionResult, exact_shapley)
from src.models.logistic_predictor import LogisticPredictor


# --------------------------------------------------------------------------
# fixtures / stubs
# --------------------------------------------------------------------------

def _record(responses, label=1):
    n = len(responses)
    return {"item_responses": np.asarray(responses, dtype=float),
            "label": label, "label_source": "questionnaire",
            "covariates": {"age_band": "1-2", "sex": "M"},
            "missing_mask": np.zeros(n, dtype=bool)}


def _tiny_training_set():
    """Deterministic 20-record set so the predictor is reproducible."""
    rng = np.random.default_rng(7)
    recs = []
    for _ in range(20):
        r = rng.integers(0, 2, size=10).astype(float)
        recs.append(_record(r, label=int(r.sum() >= 5)))
    return recs


def _first_item_policy(state, legal):
    items = [a for a in legal if a != -1]
    return items[0] if items else -1


class _AdditiveStub:
    """v(S) = 0.1 + sum_{j in S} coef_j * response_j (absent contributes 0)."""

    def __init__(self, coefs):
        self.coefs = np.asarray(coefs, dtype=float)

    def predict_state(self, state):
        s = 0.1
        for j in range(state["n"]):
            if state["mask"][j] == OBSERVED:
                s += self.coefs[j] * float(state["value"][j])
        return float(s)


class _ZeroImputeStub:
    """Complete-vector logistic model; absent items imputed with 0."""

    def __init__(self, beta, intercept):
        self.beta = np.asarray(beta, dtype=float)
        self.intercept = float(intercept)

    def predict_state(self, state):
        x = np.zeros(self.beta.size)
        for j in range(self.beta.size):
            if state["mask"][j] == OBSERVED:
                x[j] = float(state["value"][j])
        return float(1.0 / (1.0 + np.exp(-(self.intercept + x @ self.beta))))


# --------------------------------------------------------------------------
# 1. hand-computed toy
# --------------------------------------------------------------------------

def test_additive_value_function_has_analytic_shapley():
    # On an additive value function phi_j == coef_j * response_j exactly.
    coefs = [0.05, -0.10, 0.20]
    resp = [1, 0, 1]
    rec = _record(resp + [0.0] * 2)
    s = init_state(rec)
    s["questions_remaining"] = 5
    s["budget"] = 5
    for j in range(3):
        s = update_state(s, j, resp[j], 5 - j - 1)
        s["questions_remaining"] = 5 - j - 1
        s["budget"] = 5

    res = exact_shapley(s, _AdditiveStub(coefs + [0.3, 0.4]))
    got = {a.item: a.phi for a in res.items}
    assert got["A1"] == pytest.approx(0.05, abs=1e-12)
    assert got["A2"] == pytest.approx(0.00, abs=1e-12)   # response 0, coef -0.10
    assert got["A3"] == pytest.approx(0.20, abs=1e-12)
    # efficiency: sum == v(full) - v(empty) == (0.1+0.25) - 0.1
    assert res.additivity_error < 1e-12
    assert res.full_p == pytest.approx(0.35, abs=1e-12)
    assert res.baseline_p == pytest.approx(0.10, abs=1e-12)


def test_direction_labels_follow_sign():
    coefs = [0.05, -0.10, 0.20]
    resp = [1, 1, 0]
    rec = _record(resp + [0.0] * 2)
    s = init_state(rec)
    s["questions_remaining"] = 5
    s["budget"] = 5
    for j in range(3):
        s = update_state(s, j, resp[j], 5 - j - 1)
        s["questions_remaining"] = 5 - j - 1
        s["budget"] = 5
    res = exact_shapley(s, _AdditiveStub(coefs + [0.3, 0.4]))
    d = {a.item: a.direction for a in res.items}
    assert d["A1"] == "raises"
    assert d["A2"] == "lowers"
    assert d["A3"] == "neutral"      # response 0 -> phi 0


# --------------------------------------------------------------------------
# 2. efficiency + fidelity on the real frozen predictor over a real episode
# --------------------------------------------------------------------------

def test_efficiency_and_fidelity_on_real_v3_predictor():
    train = _tiny_training_set()
    pred = LogisticPredictor.fit(train, train, [f"A{j+1}" for j in range(10)],
                                 seed=0)
    rec = _record([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    ep = run_episode(rec, question_budget=6, b_min=0,
                     policy=_first_item_policy, predictor=pred)

    state = ep["final_state"]
    mask_before = state["mask"].copy()
    value_before = state["value"].copy()

    res = exact_shapley(state, pred)

    # v(all observed) is the same call the episode made -> must be identical.
    assert res.full_p == pytest.approx(ep["p_hat"], abs=1e-12)
    # v(empty) is the predictor's prior-only estimate.
    empty = init_state(rec)
    empty["questions_remaining"] = 0
    empty["budget"] = 6
    assert res.baseline_p == pytest.approx(pred.predict_state(empty), abs=1e-12)
    # Shapley efficiency identity.
    assert res.additivity_error < 1e-9, res.additivity_error
    assert len(res.items) == len(ep["items_asked"])
    # every observed item attributed exactly once
    assert sorted(a.item_idx for a in res.items) == sorted(ep["items_asked"])
    # ranking is by descending |phi|
    ranked = res.ranked()
    assert [a.rank for a in ranked] == list(range(1, len(ranked) + 1))
    assert all(abs(ranked[i].phi) >= abs(ranked[i + 1].phi)
               for i in range(len(ranked) - 1))
    # read-only contract
    assert np.array_equal(state["mask"], mask_before)
    assert np.array_equal(state["value"], value_before)
    # determinism
    res2 = exact_shapley(state, pred)
    assert [a.phi for a in res.items] == [a.phi for a in res2.items]


# --------------------------------------------------------------------------
# 3. cross-method agreement with the SHAP package (validation only)
# --------------------------------------------------------------------------

def test_matches_shap_exact_explainer():
    shap = pytest.importorskip("shap")
    beta = [0.5, -0.8, 1.2, 0.3, -0.6, 0.9, 0.2, -0.4, 0.7, 0.1]
    intercept = -0.4
    stub = _ZeroImputeStub(beta, intercept)
    resp = [1, 0, 1, 1, 0, 1]
    rec = _record(resp + [0.0] * 4)
    s = init_state(rec)
    s["questions_remaining"] = 10
    s["budget"] = 10
    for j in range(6):
        s = update_state(s, j, resp[j], 10 - j - 1)
        s["questions_remaining"] = 10 - j - 1
        s["budget"] = 10

    res = exact_shapley(s, stub)

    # shap side: complete-vector model, zero background, absent == 0
    def f(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return 1.0 / (1.0 + np.exp(-(intercept + X @ np.asarray(beta))))

    x = np.zeros((1, 10))
    for j in range(6):
        x[0, j] = resp[j]
    masker = shap.maskers.Independent(np.zeros((1, 10)))
    explainer = shap.explainers.Exact(f, masker)
    ev = explainer(x)
    shap_vals = np.asarray(ev.values).ravel()

    mine = np.zeros(10)
    for a in res.items:
        mine[a.item_idx] = a.phi
    assert float(np.max(np.abs(mine - shap_vals))) < 1e-6


# --------------------------------------------------------------------------
# 4. contract and robustness
# --------------------------------------------------------------------------

def test_missing_items_are_never_attributed():
    rec = _record([1, np.nan, 1, 0, 0, 0, 0, 0, 0, 0])
    rec["missing_mask"][1] = True
    s = init_state(rec)
    s["questions_remaining"] = 10
    s["budget"] = 10
    for j in (0, 2):
        s = update_state(s, j, 1, 10 - j - 1)
    s["questions_remaining"] = 10 - 2
    s["budget"] = 10
    res = exact_shapley(s, _AdditiveStub([0.1] * 10))
    assert sorted(a.item_idx for a in res.items) == [0, 2]
    assert all(a.item != "A2" for a in res.items)


def test_guard_on_too_many_items():
    rec = _record([1] * 14)
    s = init_state(rec)
    s["questions_remaining"] = 14
    s["budget"] = 14
    for j in range(14):
        s = update_state(s, j, 1, 14 - j - 1)
    with pytest.raises(ValueError, match="exact enumeration caps"):
        exact_shapley(s, _AdditiveStub([0.1] * 14))


def test_guard_on_non_observed_item():
    rec = _record([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = init_state(rec)
    s["questions_remaining"] = 10
    s["budget"] = 10
    s = update_state(s, 0, 1, 9)
    with pytest.raises(ValueError, match="not OBSERVED"):
        exact_shapley(s, _AdditiveStub([0.1] * 10), items=[0, 3])


def test_empty_attribution_when_nothing_observed():
    rec = _record([0] * 10)
    s = init_state(rec)
    s["questions_remaining"] = 10
    s["budget"] = 10
    res = exact_shapley(s, _AdditiveStub([0.1] * 10))
    assert res.items == []
    assert res.n_attributed == 0
    assert res.baseline_p == pytest.approx(res.full_p, abs=1e-15)
