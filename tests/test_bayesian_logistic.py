"""Stage 3: Bayesian logistic (Laplace) — interface, uncertainty, no leak."""
import numpy as np
import pytest

from src.data.ingest import _synthetic_records, load_saudi
from src.data.splits import stratified_split
from src.models.logistic_predictor import LogisticPredictor
from src.models.bayesian_logistic import BayesianLogisticPredictor
from src.env.state import init_state, update_state


@pytest.fixture(scope="module")
def fitted():
    recs = load_saudi()
    train, val, _ = stratified_split(recs, seed=0)
    base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=0)
    bayes = BayesianLogisticPredictor(base, train, n_samples=200, seed=0)
    return recs, train, val, base, bayes


def test_same_predictor_interface(fitted):
    _, _, _, _, bayes = fitted
    st = init_state(fitted[1][0]); st["budget"] = 6; st["questions_remaining"] = 6
    p = bayes.predict_state(st)
    assert 0.0 <= p <= 1.0
    assert bayes.__call__(st) == p


def test_probabilities_in_bounds(fitted):
    recs, train, _, _, bayes = fitted
    rng = np.random.default_rng(0)
    for r in train[:20]:
        st = init_state(r)
        st = update_state(st, 0, int(r["item_responses"][0]), 5)
        p = bayes.predict_state(st)
        assert 0.0 <= p <= 1.0


def test_uncertainty_distinct_from_clinical(fitted):
    _, train, _, _, bayes = fitted
    st = init_state(train[0]); st["budget"] = 6; st["questions_remaining"] = 6
    u = bayes.predictive_uncertainty(st)
    assert u["available"] is True
    assert u["is_clinical_uncertainty"] is False
    assert 0.0 <= u["epistemic_sd"] <= 0.5
    lo, hi = u["predictive_interval_95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_coefficient_intervals_monotone_width(fitted):
    _, _, _, _, bayes = fitted
    cis = bayes.coefficient_credible_intervals()
    assert len(cis["items"]) == 10
    for it in cis["items"]:
        lo, hi = it["ci95"]
        assert lo <= hi


def test_not_saturated_on_partial_state(fitted):
    _, train, _, _, bayes = fitted
    st = init_state(train[0]); st["budget"] = 6; st["questions_remaining"] = 6
    p = bayes.predict_state(st)
    assert 0.0 < p < 1.0  # partial state must not collapse to 0/1


def test_rejects_nan_training_items(fitted):
    recs = _synthetic_records(10, 10, "synthetic", seed=1)
    recs[0]["item_responses"][0] = np.nan
    base = fitted[3]
    with pytest.raises(ValueError, match="fully observed"):
        BayesianLogisticPredictor(base, recs, n_samples=50, seed=0)
