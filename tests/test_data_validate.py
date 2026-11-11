"""Stage 2 regression tests: label provenance, synthetic/real separation,
duplicate leakage, split separation, deterministic circularity detection."""
import numpy as np
import pytest

from src.data.ingest import _synthetic_records, load_saudi
from src.data.schema import validate_dataset, VALID_SYNTHETIC_SOURCES
from src.data.validate import (
    check_split_separation,
    validate_datasets,
    validate_records,
)
from src.data.dedupe import find_duplicates


def _real_record(label=0, idx=0):
    return {
        "item_responses": np.array([0.0, 1.0, 0.0, 1.0]),
        "label": int(label),
        "label_source": "questionnaire",
        "covariates": {"age_band": "1-2", "sex": "M"},
        "missing_mask": np.array([False, False, False, False]),
        "provenance": f"real:{idx}",
    }


# --- synthetic/real separation -------------------------------------------

def test_synthetic_fixture_validates_as_synthetic():
    recs = _synthetic_records(5, 10, "synthetic", seed=0)
    assert all(r["label_source"] == "synthetic" for r in recs)
    validate_dataset(recs, n_items=10)


def test_synthetic_cannot_masquerade_as_questionnaire():
    recs = _synthetic_records(5, 10, "synthetic", seed=0)
    bad = [dict(r, label_source="questionnaire") for r in recs]
    with pytest.raises(ValueError, match="provenance|Provenance"):
        validate_dataset(bad, n_items=10)


def test_synthetic_rejects_questionnaire_and_clinical_labels():
    with pytest.raises(ValueError):
        _synthetic_records(3, 10, "questionnaire", seed=0)
    with pytest.raises(ValueError):
        _synthetic_records(3, 10, "clinical", seed=0)


def test_mixed_real_and_synthetic_is_refused():
    recs = [_real_record(0, 0), _real_record(1, 1)] + _synthetic_records(2, 4, "synthetic", seed=1)
    with pytest.raises(ValueError, match="Mixed real"):
        validate_records(recs, n_items=4)


def test_unverified_labels_are_rejected_for_validation():
    rec = dict(_real_record(0, 0), label_source="unverified")
    with pytest.raises(ValueError, match="unverified"):
        validate_records([rec], n_items=4)


def test_real_synthetic_pipeline_split_separation():
    recs = _synthetic_records(40, 10, "synthetic", seed=7)
    # synthetic fixtures share a dataset-level provenance tag; assign per-row
    # provenance for the split-identity check (real loaders already do this).
    for i, r in enumerate(recs):
        r["provenance"] = f"synthetic:synthetic_40_10_7:{i}"
    out = check_split_separation(recs, seed=0)
    assert out["n_train"] + out["n_val"] + out["n_test"] == 40


# --- dedupe label-disagreement --------------------------------------------

def test_cross_source_duplicate_with_disagreeing_labels_is_flagged():
    a = _real_record(0, 0)
    b = dict(_real_record(1, 1))  # same answers/covariates, different label
    dups = find_duplicates({"saudi": [a], "uci_child": [b]})
    assert len(dups) == 1
    assert dups[0]["label_disagreement"] is True
    assert dups[0]["labels"] == [0, 1]


def test_cross_source_duplicate_with_agreeing_labels_is_flagged():
    a = _real_record(0, 0)
    b = _real_record(0, 1)
    dups = find_duplicates({"saudi": [a], "uci_child": [b]})
    assert len(dups) == 1
    assert dups[0]["label_disagreement"] is False


# --- circularity must be detected, never hidden ----------------------------

def test_deterministic_saudi_labels_are_reported():
    recs = load_saudi()  # real local Saudi CSV (verified present)
    out = validate_datasets({"saudi": recs}, n_items=10, run_circularity=True)
    circ = out["datasets"]["saudi"]["circularity"]
    assert circ["classification"] == "Deterministic"
    assert circ["exact_match_rate"] == 1.0
    assert out["datasets"]["saudi"]["label_sources"] == ["questionnaire"]


def test_validate_records_reports_missing_and_prevalence():
    recs = [_real_record(0, 0), _real_record(1, 1), _real_record(1, 2)]
    out = validate_records(recs, n_items=4)
    assert out["n"] == 3
    assert out["label_sources"] == ["questionnaire"]
    assert out["missing_frac"] == 0.0
    assert out["prevalence"] == pytest.approx(2 / 3)
