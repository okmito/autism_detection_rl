"""Tests for predictor v3, the frozen RL protocol, and the external evaluation.

Locks the three properties that make the new experiment interpretable:
1. predictor v3 is Saudi-only, frozen, and correctly scored;
2. the protocol was frozen before the sealed cohort was opened, and the
   equivalence margin was NOT invented;
3. the evaluation reports every mandatory baseline and declares no equivalence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

from src.models.logistic_predictor import (CALIBRATION_METHOD, HYPERPARAMETERS,
                                           N_ITEMS, PREDICTOR_VERSION,
                                           LogisticPredictor)
from src.eval.baselines import (REQUIRED_BASELINES, MissingBaselineError,
                                assert_all_baselines, missing_baselines)


def _load(name: str):
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _predictor():
    path = RESULTS / "predictor_logistic_saudi_v3.pkl"
    if not path.exists():
        pytest.skip("run scripts/step12_new_predictor.py first")
    return LogisticPredictor.load(path)


# ============================================ 1. new primary predictor
def test_predictor_v3_is_logistic_l2_c01():
    assert PREDICTOR_VERSION == "logistic-saudi-v3"
    assert HYPERPARAMETERS["model"] == "LogisticRegression"
    assert HYPERPARAMETERS["penalty"] == "l2"
    assert HYPERPARAMETERS["C"] == 0.1


def test_predictor_v3_uses_platt_not_isotonic():
    assert CALIBRATION_METHOD == "platt_sigmoid_on_raw_probability_saudi_val"
    assert "isotonic" not in CALIBRATION_METHOD


def test_predictor_v3_trained_and_calibrated_on_saudi_only():
    meta = _predictor().metadata
    assert meta["trained_on"] == "saudi train split"
    assert meta["calibrated_on"] == "saudi validation split"
    assert meta["external_cohort_used"] is False
    assert meta["n_train"] == 284
    assert meta["n_val"] == 95


def test_predictor_v3_probabilities_are_sane():
    """Guards the two bugs that made v3 report ~0.001 everywhere."""
    p = _predictor()
    typical = p.predict_features(np.zeros(N_ITEMS))
    atypical = p.predict_features(np.ones(N_ITEMS))
    assert 0.0 < typical < 0.5
    assert 0.5 < atypical < 1.0
    assert typical < atypical
    # monotone in the number of atypical answers
    for k in range(N_ITEMS):
        lo = p.predict_features(np.array([1] * k + [0] * (N_ITEMS - k), float))
        hi = p.predict_features(np.array([1] * (k + 1) + [0] * (N_ITEMS - k - 1),
                                          float))
        assert lo <= hi + 1e-12


def test_full_vector_and_state_scoring_agree_exactly():
    """A reversed config-index convention once made these disagree."""
    from src.env.state import OBSERVED
    p = _predictor()
    rng = np.random.default_rng(0)
    for _ in range(20):
        row = rng.integers(0, 2, size=N_ITEMS).astype(float)
        via_features = p.predict_features(row)
        via_state = p.predict_state({"mask": np.full(N_ITEMS, OBSERVED),
                                     "value": row.astype(int), "n": N_ITEMS})
        assert abs(via_features - via_state) < 1e-12


def test_marginal_over_unobserved_items_is_not_the_naive_extremum():
    """With nothing observed the answer must be a prevalence-like middle value."""
    from src.env.state import UNASKED
    p = _predictor()
    empty = p.predict_state({"mask": np.zeros(N_ITEMS, dtype=int),
                             "value": np.full(N_ITEMS, -1), "n": N_ITEMS})
    assert 0.05 < empty < 0.95
    one_item = p.predict_state({"mask": np.array([1] + [0] * (N_ITEMS - 1)),
                                "value": np.array([1] + [-1] * (N_ITEMS - 1)),
                                "n": N_ITEMS})
    assert one_item > empty


def test_predictor_v3_explainability_reports_all_ten_items():
    expl = _predictor().explain()
    assert expl["form"] == "log-odds = beta0 + sum_i beta_i * x_i"
    assert len(expl["items"]) == N_ITEMS
    for row in expl["items"]:
        assert isinstance(row["coefficient_log_odds"], float)
        assert row["direction"]
        assert row["odds_multiplier_when_atypical"] > 0
    assert expl["platt_calibration"]["form"].startswith("P(y=1|p)")


def test_v2_legacy_artifacts_are_retained_unchanged():
    """v2 must NOT be deleted or overwritten."""
    for name, expected in [
            ("predictor_saudi_v2_isotonic.pt",
             "2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4"),
            ("predictor_saudi_v2_isotonic.pkl",
             "a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863")]:
        path = RESULTS / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def test_v3_manifest_records_the_rationale_and_no_clinical_claim():
    m = _load("predictor_manifest_v3.json")
    if not m:
        pytest.skip("run scripts/step12_new_predictor.py first")
    assert m["predictor_version"] == PREDICTOR_VERSION
    assert m["polish_touched"] is False
    assert m["external_cohort_used"] is False
    assert "FROZEN LEGACY" in m["v2_status"]
    assert "NOT a claim of clinical superiority" in \
        m["rationale"]["not_a_clinical_claim"]
    assert m["threshold"]["tuned_on_polish"] is False


# ============================================ 3. mandatory baselines
def test_all_four_baselines_are_mandatory():
    assert set(REQUIRED_BASELINES) == {
        "item_count", "random_fixed", "greedy_information_gain",
        "beta_greedy_evoi"}


def test_evaluation_fails_when_a_baseline_is_missing():
    assert missing_baselines(["item_count", "random_fixed"]) == [
        "greedy_information_gain", "beta_greedy_evoi"]
    with pytest.raises(MissingBaselineError, match="mandatory baseline"):
        assert_all_baselines(["item_count"], context="unit test")
    assert_all_baselines(list(REQUIRED_BASELINES))


# ============================================ 10. protocol frozen first
def test_protocol_was_frozen_before_polish():
    p = _load("polish_rl_external_protocol.json")
    if not p:
        pytest.skip("run scripts/step13_rl_protocol.py first")
    assert p["frozen_before_polish"] is True
    assert p["predictor"]["sha256"]
    assert p["predictor"]["polish_used"] is False
    assert p["rl"]["polish_used_in_training"] is False
    assert p["git_sha"]


def test_protocol_equivalence_margin_is_open_and_not_invented():
    p = _load("polish_rl_external_protocol.json")
    if not p:
        pytest.skip("run scripts/step13_rl_protocol.py first")
    eq = p["equivalence"]
    assert eq["acceptable_AUROC_margin"] == "OPEN"
    assert eq["state"] == "OPEN"
    assert "None has been invented" in eq["reason"]
    assert any("statistically indistinguishable" in f for f in eq["forbidden"])


def test_protocol_budgets_and_seed_are_fixed():
    p = _load("polish_rl_external_protocol.json")
    if not p:
        pytest.skip("run scripts/step13_rl_protocol.py first")
    assert p["budgets"] == [2, 3, 4, 5, 10]
    assert p["full_question_reference_budget"] == 10
    assert p["seed"] == 0
    assert p["metrics"]["tau_primary"] == 0.5
    assert p["metrics"]["tau_tuned_on_polish"] is False


def test_reward_structure_is_preserved():
    p = _load("polish_rl_external_protocol.json")
    if not p:
        pytest.skip("run scripts/step13_rl_protocol.py first")
    assert p["reward"]["formula"] == \
        "R = (1 - (p_hat - y)^2) - lambda_cost * sum_j c_j"
    assert "UNCHANGED" in p["reward"]["structure"]
    assert "must NOT be mixed" in p["reward"]["note"]


def test_all_dqn_policies_were_actually_trained_and_persisted():
    p = _load("polish_rl_external_protocol.json")
    if not p:
        pytest.skip("run scripts/step13_rl_protocol.py first")
    artifacts = p["rl"]["artifacts"]
    assert set(artifacts) == {"dqn_B2", "dqn_B3", "dqn_B4", "dqn_B5", "dqn_B10"}
    for name, info in artifacts.items():
        path = REPO / info["path"]
        assert path.exists(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        assert info["polish_used_in_training"] is False
        assert info["training"]["n_updates"] > 1000


def test_beta_greedy_is_not_described_as_rl():
    p = _load("polish_rl_external_protocol.json")
    if not p:
        pytest.skip("run scripts/step13_rl_protocol.py first")
    assert "never described as an RL policy" in p["rl"]["why_dqn_only"]
    assert "NOT an RL policy" in \
        p["policy_arms"]["beta_greedy_evoi"]["note"]


# ============================================ 14. external evaluation
def _evaluation():
    d = _load("polish_rl_external_evaluation.json")
    if not d:
        pytest.skip("run scripts/step14_rl_external_eval.py first")
    return d


def test_evaluation_verified_the_frozen_protocol():
    d = _evaluation()
    assert d["protocol_sha_verified"] is True
    assert d["protocol_frozen_before_polish"] is True
    assert d["predictor_version"] == PREDICTOR_VERSION


def test_evaluation_reports_every_mandatory_baseline():
    """Covers the RETAINED first evaluation.

    That run predates the corrected design, so its random arm was the
    stop-capable ``random_questioning``. The corrected fixed-budget run uses
    ``random_fixed`` and is checked in tests/test_corrected_design.py. Both
    artifacts are retained.
    """
    d = _evaluation()
    reported = set(d["mandatory_baselines"]["reported"])
    assert d["mandatory_baselines"]["all_present"] is True
    assert {"item_count", "random_questioning",
            "greedy_information_gain", "beta_greedy_evoi"} <= reported
    # the corrected mandatory set differs only in the random arm's semantics;
    # the retained first run additionally reports the learned DQN arm
    assert reported - set(REQUIRED_BASELINES) == {"random_questioning", "dqn"}


def test_evaluation_declares_no_equivalence_claim():
    d = _evaluation()
    assert d["equivalence"]["acceptable_AUROC_margin"] == "OPEN"
    for c in d["comparisons_vs_full_question"]:
        assert c["equivalence_declared"] == "NOT DECLARED"


def test_no_arm_below_full_budget_matches_the_item_count_reference():
    """The headline negative result, locked so it cannot quietly disappear."""
    d = _evaluation()
    below = [c for c in d["comparisons_vs_full_question"]
             if c["B"] < 10 and c["arm"] != "item_count"]
    assert below, "expected sub-full-budget comparisons"
    for c in below:
        v = c["vs_full_item_count"]
        assert v["ci_hi"] < 0, (
            f"{c['arm']} B={c['B']} interval no longer excludes zero: {v}")


def test_selection_criteria_are_recorded_not_invented():
    d = _evaluation()
    for row in d["results"]:
        assert row["selection_criterion"]
    dqn_rows = [r for r in d["results"] if r["arm"] == "dqn"]
    assert dqn_rows
    for row in dqn_rows:
        assert "Q-value" in row["selection_criterion"]


def test_every_result_row_declares_no_polish_training():
    for row in _evaluation()["results"]:
        assert row["polish_used_in_training"] is False


def test_evaluation_is_metadata_only():
    d = _evaluation()
    assert d["contains_participant_rows"] is False
    assert d["cohort"]["n"] == 252
    assert d["cohort"]["asd"] == 135
    assert d["cohort"]["control"] == 117


def test_limitations_are_recorded():
    text = " ".join(_evaluation()["limitations"]).lower()
    assert "one polish external cohort" in text
    assert "label-definition shift" in text
    assert "ordinal-to-binary projection" in text
    assert "finite precision" in text
    assert "not clinical diagnosis" in text
    assert "domain shift" in text