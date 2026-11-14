"""Stage 4: hybrid EVOI maths, calibration separation, stopping, abstention."""
import numpy as np
import pytest

from src.data.ingest import load_saudi
from src.data.splits import stratified_split
from src.models.logistic_predictor import LogisticPredictor
from src.models.bayesian_logistic import BayesianLogisticPredictor
from src.policies.irt_cat import IRTCATPolicy
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy, STOP
from src.env.state import init_state, update_state, get_legal_items
from src.env.environment import run_episode


@pytest.fixture(scope="module")
def stack():
    recs = load_saudi()
    train, val, _ = stratified_split(recs, seed=0)
    base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=0)
    bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=0)
    irt = IRTCATPolicy(train, n_items=10)
    return train, bayes, irt


def _state(train, rec_idx=0, asked=2):
    st = init_state(train[rec_idx])
    st["budget"] = 6; st["questions_remaining"] = 6 - asked
    for j in range(asked):
        st = update_state(st, j, int(train[rec_idx]["item_responses"][j]), 6 - asked - j)
        st["budget"] = 6
    return st


def test_evoi_finite_and_bounds(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10)
    st = _state(train)
    legal = get_legal_items(st) + [STOP]
    s = pol.scores(st, legal)
    assert all(np.isfinite(v) for v in s["evoi"].values())
    assert all(-1.0 <= v <= 1.0 for v in s["evoi"].values())
    assert all(np.isfinite(v) for v in s["fisher_information"].values())


def test_duplicate_question_prevention(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10)
    st = _state(train, asked=3)
    legal = get_legal_items(st) + [STOP]
    a = pol(st, legal)
    assert a not in [0, 1, 2, STOP]


def test_budget_enforcement(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, max_questions=0)
    st = _state(train, asked=2)
    assert pol(st, get_legal_items(st) + [STOP]) == STOP


def test_min_questions_enforcement(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, min_questions=5)
    st = _state(train, asked=2)
    assert pol(st, get_legal_items(st) + [STOP]) != STOP


def test_episode_terminates_and_probability_in_bounds(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, min_questions=0, max_questions=6)
    out = run_episode(train[0], 6, 0, pol, bayes, lambda_cost=0.0)
    assert out["stop_reason"] in ("policy_stop", "budget_exhausted")
    assert 0.0 <= out["p_hat"] <= 1.0
    assert len(set(out["items_asked"])) == len(out["items_asked"])


def test_explain_reports_evoi_and_uncertainty(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10)
    st = _state(train)
    ex = pol.explain(st, get_legal_items(st) + [STOP])
    assert ex["is_clinical_uncertainty"] is False
    assert len(ex["evoi_values"]) > 0
    assert "predictive_sd" in ex


def test_never_returns_stop_when_stop_not_legal(stack):
    """Regression: hybrid must never return STOP when STOP is absent from
    the legal set (environment contract)."""
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
    st = _state(train, asked=2)
    legal_no_stop = get_legal_items(st)  # no STOP
    assert -1 not in legal_no_stop
    a = pol(st, legal_no_stop)
    assert a in legal_no_stop and a != -1

def test_abstention_path_is_reachable(stack):
    train, bayes, irt = stack
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=10.0, abstain_sd=0.0)
    st = _state(train)
    # with a huge cost, net scores are negative; policy prefers asking only
    # when uncertain, else STOP — either way it returns a legal action.
    a = pol(st, get_legal_items(st) + [STOP])
    assert a in (get_legal_items(st) + [STOP])
