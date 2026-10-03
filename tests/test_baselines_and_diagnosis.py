"""Tests for the item-count reference baseline and the Step 11 diagnosis.

The item-count reference is now a permanent floor: no predictor or policy may claim
improvement without being measured against it. These tests lock its definition, its
independence from labels and from Polish tuning, and the recorded finding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

from src.data.qchat10_contract import (CANONICAL_MAPPING, MODEL_FEATURE_ORDER,
                                       build_feature_matrix)
from src.eval.baselines import (REQUIRED_BASELINES, baseline_comparison_table,
                                item_count_probabilities, item_count_scores,
                                missing_baselines)


def _polish_features():
    from src.data.ingest import POLISH_CSV
    import pandas as pd
    df = pd.read_csv(POLISH_CSV, encoding="utf-8")
    table = {e["polish_var"]: df[e["polish_var"]].tolist()
             for e in MODEL_FEATURE_ORDER}
    return df, np.array(build_feature_matrix(table, allow_missing=False))


# ================================================= 3. item-count audit
def test_item_count_definition_is_count_of_atypical_responses():
    features = np.array([[0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    counts = item_count_scores(features)
    assert counts.tolist() == [4.0, 0.0]


def test_item_count_uses_the_same_projection_as_the_model():
    """Same 252 participants, same 10 items, same binary encodings."""
    df, features = _polish_features()
    assert features.shape == (252, 10)
    assert set(np.unique(features)) == {0.0, 1.0}
    # identical projection object the frozen model consumes
    assert len(MODEL_FEATURE_ORDER) == 10
    assert features.shape[1] == len(CANONICAL_MAPPING) == 10


def test_item_count_uses_no_labels_and_no_polish_tuning():
    """The construction cannot see a label: it takes only a feature matrix."""
    features = np.array([[1] * 10, [0] * 10], dtype=float)
    with pytest.raises(TypeError):
        item_count_scores(features, np.array([1, 0]))
    # no threshold parameter exists at all
    import inspect
    assert list(inspect.signature(item_count_scores).parameters) == ["features"]


def test_item_count_is_defined_only_for_ten_binary_items():
    with pytest.raises(ValueError, match="10 projected"):
        item_count_scores(np.zeros((3, 9)))
    with pytest.raises(ValueError, match="binary"):
        item_count_scores(np.full((3, 10), 2.0))
    with pytest.raises(ValueError, match="2-D"):
        item_count_scores(np.zeros(10))


def test_item_count_probabilities_are_monotone_in_the_count():
    _, features = _polish_features()
    counts = item_count_scores(features)
    probs = item_count_probabilities(features)
    assert np.array_equal(counts, probs * 10.0)
    order_counts = np.argsort(counts)
    order_probs = np.argsort(probs)
    assert np.array_equal(order_counts, order_probs), (
        "scaling must not change the ranking")


def test_item_count_reproduces_the_recorded_auroc():
    """Locks the 0.9369 figure the whole baseline decision rests on."""
    from sklearn.metrics import roc_auc_score
    df, features = _polish_features()
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]])
    auroc = roc_auc_score(y, item_count_scores(features))
    assert abs(auroc - 0.9369) < 5e-4, auroc


# ========================================= 9. item count is permanent
def test_required_baselines_are_declared():
    """All four are mandatory; beta-greedy/EVOI was added after the diagnosis."""
    assert set(REQUIRED_BASELINES) == {
        "item_count", "random_fixed", "greedy_information_gain",
        "beta_greedy_evoi"}
    assert REQUIRED_BASELINES[0] == "item_count"
    assert "beta_greedy_evoi" in REQUIRED_BASELINES


def test_missing_baselines_detects_an_incomplete_result_set():
    assert missing_baselines(["item_count"]) == [
        "random_fixed", "greedy_information_gain", "beta_greedy_evoi"]
    assert missing_baselines(["item_count", "random_fixed",
                              "greedy_information_gain"]) == ["beta_greedy_evoi"]
    assert missing_baselines(list(REQUIRED_BASELINES)) == []


def test_baseline_comparison_flags_that_nothing_beats_item_count():
    _, features = _polish_features()
    df, _ = _polish_features()
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]])
    # a genuinely weaker predictor: the reversed count (anti-correlated)
    weak = 10.0 - item_count_scores(features)
    table = baseline_comparison_table(features, y, {"weak_model": weak})
    assert table["item_count_auroc"] > 0.9
    assert table["rows"][0]["beats_item_count"] is False
    assert table["rows"][0]["auroc_minus_item_count"] < 0
    assert table["no_predictor_beats_item_count"] is True


def test_scaling_a_predictor_does_not_change_its_auroc():
    """Guards a real trap: count/10 is not 'weaker' than count."""
    _, features = _polish_features()
    df, _ = _polish_features()
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]])
    from sklearn.metrics import roc_auc_score
    counts = item_count_scores(features)
    assert roc_auc_score(y, counts) == roc_auc_score(
        y, item_count_probabilities(features))


# ============================== 2. the difference is statistically resolved
def _diagnosis():
    path = RESULTS / "predictor_diagnosis.json"
    if not path.exists():
        pytest.skip("run scripts/step11_predictor_diagnosis.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_step11_reproduces_the_primary_result_exactly():
    v = _diagnosis()["verification"]
    assert v["exact_reproduction"] is True
    assert v["weights_match_recorded"] is True
    assert v["calibrator_match_recorded"] is True
    assert v["threshold"] == 0.5
    assert v["contract_complete"] is True


def test_paired_auroc_difference_ci_lies_entirely_below_zero():
    d = _diagnosis()["paired_auroc_bootstrap"]
    assert d["point_estimate"] < 0
    assert d["ci_hi"] < 0, "model can only be called worse if the whole CI is below 0"
    assert d["bootstrap_p_model_better"] < 0.01
    assert d["n_resamples_used"] >= 1000


def test_brier_difference_is_indeterminate_and_reported_as_such():
    """Prevents over-claiming: Brier did NOT differ significantly."""
    d = _diagnosis()["paired_brier_bootstrap"]
    assert d["ci_lo"] < 0 < d["ci_hi"]


def test_isotonic_calibration_is_the_worst_of_the_three_calibrators():
    rows = {r["name"]: r for r in _diagnosis()["calibration_ablation"]}
    assert rows["isotonic_saudi"]["auroc"] < rows["raw_uncalibrated"]["auroc"]
    assert rows["isotonic_saudi"]["auroc"] < rows["platt_saudi_network"]["auroc"]
    assert rows["isotonic_saudi"]["role"] == "PRIMARY"


def test_plain_logistic_on_saudi_beats_the_frozen_network():
    rows = {r["name"]: r for r in _diagnosis()["predictor_ablation"]}
    for name, row in rows.items():
        if name.startswith("logistic"):
            assert row["polish_used_in_training"] is False
            assert row["brier"] < rows["frozen_isotonic"]["brier"]
            assert row["auroc"] > rows["frozen_isotonic"]["auroc"]


def test_classification_is_confirmed_not_uncertain():
    c = _diagnosis()["classification"]
    assert c["ci_excludes_zero"] is True
    assert c["confirmed"].startswith("C:")


def test_primary_result_is_declared_unchanged():
    d = _diagnosis()
    assert d["primary_result_unchanged"]["auroc"] == pytest.approx(0.8959, abs=5e-4)
    assert d["primary_result_unchanged"]["brier"] == pytest.approx(0.1398, abs=5e-4)
    assert d["primary_result_unchanged"]["threshold"] == 0.5
    assert "does not recompute" in d["primary_result_unchanged"]["note"]
    assert d["contains_participant_rows"] is False


# ================================= 10. frozen artefacts untouched by diagnosis
@pytest.mark.parametrize("name,expected", [
    ("predictor_saudi_v2_isotonic.pt",
     "2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4"),
    ("predictor_saudi_v2_isotonic.pkl",
     "a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863"),
])
def test_frozen_artifacts_unchanged(name, expected):
    path = RESULTS / name
    if not path.exists():
        pytest.skip("artifact absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_rl_component_is_reported_separately_from_the_predictor():
    rl = _diagnosis()["rl_component"]
    assert "COMPONENT A" in rl["separation_note"]
    assert "COMPONENT B" in rl["separation_note"]
    assert rl["trained_rl_policies_persisted"] is False
    assert rl["saudi_benchmark_warning"]


def test_every_ablation_model_declares_saudi_only_training():
    for row in _diagnosis()["predictor_ablation"]:
        assert row["polish_used_in_training"] is False