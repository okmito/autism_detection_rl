"""Regression tests for the 2026-09-30 demo-behaviour audit.

Locks the behaviours investigated in the audit (scripts/demo_app.py):
  1. Greedy policy is deterministic and only picks legal, unasked items.
  2. Random policy with seed=None differs across episodes; int seed = reproducible.
  3. Belief values stay in [0, 1] and are continuous, NOT confined to the
     isotonic step levels {0.0, 0.477, 1.0} reported in the audit.
  4. Platt calibration (now fitted on logits) keeps partial-evidence posteriors
     strictly inside (0, 1).
  5. The referral decision is exactly `p_hat >= tau` - no hidden mapping.
  6. The demo API returns the continuous risk value; the frontend renders it
     without bucketizing/rounding to {0, 60, 100}.

Training here uses a small noisy-synthetic cohort (labels = noisy sum-threshold
rule) so tests are fast and do not depend on data/raw.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.ingest import load_dataset
from src.env.environment import run_episode
from src.env.state import init_state, update_state
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from sklearn.model_selection import StratifiedKFold

N_ITEMS = 10
BUDGET = 6
TAU = 0.5
REPO = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- fixtures
def _mk_record(vals: list[int], label: int) -> dict:
    return {
        "item_responses": np.array(vals, dtype=float),
        "label": int(label),
        "label_source": "questionnaire",
        "covariates": {"age_band": "1-2", "sex": "M"},
        "missing_mask": np.zeros(N_ITEMS, dtype=bool),
        "provenance": "audit_synthetic",
    }


def _synthetic_cohort(n: int = 240, seed: int = 0) -> list[dict]:
    """Noisy deterministic labels (sum >= 4, 8% flipped) - mirrors the
    circular structure of the real Q-CHAT-10 datasets but is not separable,
    so calibrated posteriors stay interior."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        # skew toward mostly-typical answers, like the real cohorts
        vals = (rng.random(N_ITEMS) < 0.25).astype(int).tolist()
        label = int(sum(vals) >= 4)
        if rng.random() < 0.08:
            label = 1 - label
        recs.append(_mk_record(vals, label))
    return recs


@pytest.fixture(scope="module")
def trained_platt():
    recs = _synthetic_cohort()
    y = np.array([r["label"] for r in recs])
    idx = np.arange(len(recs))
    tr, te = next(StratifiedKFold(4, shuffle=True, random_state=0).split(idx, y))
    train = [recs[i] for i in tr]
    val = train[-40:]
    fit = train[:-40]
    pred = MaskedPredictor(n_items=N_ITEMS, hidden=[32, 16], calibration="platt")
    pred.fit(fit, epochs=40, lr=2e-3, batch_size=32, seed=0)
    pred.fit_calibrator(val, method="platt")
    return pred, fit


# ------------------------------------------------- 1. policy behaviour
def test_greedy_policy_deterministic_and_legal(trained_platt):
    pred, train = trained_platt
    rec = _mk_record([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 0)
    seqs = []
    for _ in range(3):
        ep = run_episode(rec, BUDGET, 0, GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)
        seqs.append(ep["items_asked"])
        assert len(set(ep["items_asked"])) == len(ep["items_asked"]), "item repeated in episode"
        assert all(0 <= j < N_ITEMS for j in ep["items_asked"]), "illegal item index"
    assert seqs[0] == seqs[1] == seqs[2], "greedy must be deterministic for identical inputs"


def test_random_policy_differs_across_episodes(trained_platt):
    pred, _ = trained_platt
    rec = _mk_record([0] * N_ITEMS, 0)
    seqs = {tuple(run_episode(rec, BUDGET, 0, RandomPolicy(seed=None), pred, 0.0, tau=TAU)["items_asked"])
            for _ in range(5)}
    assert len(seqs) >= 2, "RandomPolicy(seed=None) must vary across episodes"


def test_random_policy_int_seed_reproducible():
    a = [RandomPolicy(seed=7)({"mask": None, "value": None}, list(range(N_ITEMS))) for _ in range(6)]
    b = [RandomPolicy(seed=7)({"mask": None, "value": None}, list(range(N_ITEMS))) for _ in range(6)]
    assert a == b, "same int seed must give the same sequence (documented semantics)"


# ------------------------------------------------- 2. belief / risk sanity
def test_belief_trajectory_bounded_and_no_repeats(trained_platt):
    pred, _ = trained_platt
    rec = _mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1)
    ep = run_episode(rec, BUDGET, 0, GreedyIGPolicy([_mk_record([0] * N_ITEMS, 0)], N_ITEMS),
                     pred, 0.0, tau=TAU)
    for t in ep["trace"]:
        assert 0.0 <= t["belief_before"] <= 1.0
        assert 0.0 <= t["belief_after"] <= 1.0
    assert 0.0 <= ep["p_hat"] <= 1.0


def test_platt_partial_evidence_stays_interior(trained_platt):
    """One atypical answer = weak evidence -> posterior must NOT saturate.
    (The isotonic step calibrator mapped this to exactly 0.477/1.0.)"""
    pred, _ = trained_platt
    rec = _mk_record([1] + [0] * (N_ITEMS - 1), 0)
    s = init_state(rec)
    s["questions_remaining"] = BUDGET
    s["budget"] = BUDGET
    s = update_state(s, 0, 1, BUDGET - 1)
    s["questions_remaining"] = BUDGET - 1
    s["budget"] = BUDGET
    p = pred(s)
    assert 0.0 < p < 1.0, f"partial-evidence belief saturated to {p}"


def test_risk_not_confined_to_step_levels(trained_platt):
    """Across many answer patterns the risk must take many distinct values -
    rules out the {0.0, 0.477, 1.0} isotonic-step behaviour from the audit."""
    pred, _ = trained_platt
    vals = set()
    for k in range(N_ITEMS + 1):
        for pos in range(0, N_ITEMS, 3):
            v = [0] * N_ITEMS
            for j in range(k):
                v[(pos + j) % N_ITEMS] = 1
            rec = _mk_record(v, 0)
            s = init_state(rec)
            s["questions_remaining"] = 0
            s["budget"] = N_ITEMS
            for j in range(N_ITEMS):
                s = update_state(s, j, v[j], 0)
                s["budget"] = N_ITEMS
                s["questions_remaining"] = 0
            vals.add(round(float(pred(s)), 4))
    assert len(vals) > 8, f"risk values collapsed to {sorted(vals)}"


def test_different_patterns_different_trajectories(trained_platt):
    pred, train = trained_platt
    pa = run_episode(_mk_record([0] * N_ITEMS, 0), BUDGET, 0,
                     GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)["p_hat"]
    pb = run_episode(_mk_record([1] * N_ITEMS, 1), BUDGET, 0,
                     GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)["p_hat"]
    assert abs(pa - pb) > 0.3, "all-typical vs all-atypical must differ substantially"


# ------------------------------------------------- 3. decision logic
def test_decision_is_documented_threshold_rule(trained_platt):
    # environment layer documents: decision = "REFER" if p_hat >= tau else "NO_REFERRAL_INDICATED"
    pred, train = trained_platt
    for vals in ([0] * N_ITEMS, [1] * N_ITEMS, [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]):
        ep = run_episode(_mk_record(vals, 0), BUDGET, 0,
                         GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)
        expected = "REFER" if ep["p_hat"] >= TAU else "NO_REFERRAL_INDICATED"
        assert ep["decision"] == expected, "decision must be exactly p_hat >= tau"
    # demo API layer maps REFER -> REFERRAL_RECOMMENDED, same rule
    demo = _load_demo_app()
    pred2, train2 = trained_platt
    demo.S.update({"predictor": pred2, "train": train2,
                   "test": [_mk_record([1] * N_ITEMS, 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "auto", "policy": "greedy"})
    res = r["result"]
    expected_api = "REFERRAL_RECOMMENDED" if res["p_hat"] >= TAU else "NO_REFERRAL_INDICATED"
    assert res["decision"] == expected_api
    demo.S["sessions"].clear()


# ------------------------------------------------- 4. demo API contract
def _load_demo_app():
    spec = importlib.util.spec_from_file_location("demo_app", REPO / "scripts" / "demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_app"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_api_returns_continuous_risk(trained_platt):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train, "test": [_mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "interactive", "policy": "greedy"})
    assert isinstance(r["belief"], float) and 0.0 <= r["belief"] <= 1.0
    a = demo.api_answer({"session_id": r["session_id"], "value": 1})
    assert not a["finished"]
    assert isinstance(a["belief_after"], float)
    assert 0.0 <= a["belief_after"] <= 1.0
    assert a["next"]["code"] not in ("",) and a["trace_step"]["item"] != a["next"]["code"]
    demo.S["sessions"].clear()


# ------------------------------------------------- 5. frontend rendering
def test_frontend_renders_continuous_percent():
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "(p*100).toFixed(1)" in html, "frontend must render the continuous backend value"
    assert "Math.round(p" not in html, "frontend must not bucketize the risk value"


# ------------------------------------------------- 6. isotonic still available
def test_isotonic_option_still_works(trained_platt):
    pred, val = trained_platt
    pred.fit_calibrator(val, method="isotonic")
    rec = _mk_record([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 1)
    s = init_state(rec)
    s["questions_remaining"] = 0
    s["budget"] = N_ITEMS
    for j in range(N_ITEMS):
        s = update_state(s, j, int(rec["item_responses"][j]), 0)
        s["budget"] = N_ITEMS
        s["questions_remaining"] = 0
    p = pred(s)
    assert 0.0 <= p <= 1.0
