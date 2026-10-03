"""V-4 / V-7 external-validation regression tests.

Covers the fourteen behaviours required for the external-validation phase. Every
test asserts a *property that must not regress*, not an implementation detail of
one particular script, so a future refactor cannot quietly reopen the sealed
cohort or fabricate a metric.

The tests that read the real `.sav` are module-scoped so the SPSS file is parsed
once. If the export is absent they skip rather than fail, because the export is a
local provenance artefact and is deliberately not tracked by git - the tests
document the expected result when it is present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SAV = REPO / "data" / "QCHAT_dataset2 mendeley.sav"
POLISH_CSV = REPO / "data" / "raw" / "Q-CHAT Polish" / "polish_qchat.csv"


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def provenance():
    from src.data.provenance import verify_same_cohort
    if not SAV.exists() or not POLISH_CSV.exists():
        pytest.skip("local SPSS export or integrated Polish CSV not present")
    return verify_same_cohort(SAV, POLISH_CSV)


@pytest.fixture(scope="module")
def polish_frame():
    import pandas as pd
    if not POLISH_CSV.exists():
        pytest.skip("integrated Polish CSV not present")
    return pd.read_csv(POLISH_CSV, encoding="utf-8")


@pytest.fixture(scope="module")
def gate_artifact():
    path = RESULTS / "v4_v7_validation_status.json"
    if not path.exists():
        pytest.skip("run scripts/step9_v4_v7_gates.py first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------- 1. provenance = same cohort
def test_sav_provenance_matches_integrated_polish_cohort(provenance):
    """The local SPSS export must be recognised as the SAME cohort, not a new one.

    This is the test that prevents a duplicate cohort being ingested as if it
    were a fresh external-validation dataset.
    """
    assert provenance["dataset_identity"] == "same_cohort"
    assert provenance["row_count_sav"] == provenance["row_count_csv"]
    assert provenance["column_count_sav"] == provenance["column_count_csv"]


def test_schema_matching_is_order_sensitive(provenance):
    assert provenance["schema_equal_ordered"] is True
    assert provenance["schema_equal_set"] is True


# ------------------------------------------------- 2. no duplicate keys
def test_no_duplicate_polish_participant_ids(provenance, polish_frame):
    assert provenance["duplicate_count"] == 0
    ids = polish_frame["child_id"].astype(str).str.strip()
    assert len(ids) == ids.nunique(), "duplicate participant key in Polish cohort"


def test_every_participant_matches_between_sav_and_csv(provenance):
    assert provenance["participant_mismatch_count"] == 0
    assert provenance["only_in_sav"] == 0
    assert provenance["only_in_csv"] == 0
    assert provenance["participant_match_count"] == provenance["row_count_sav"]


# ------------------------------------------------------ 3/4. label mapping
def test_group_maps_only_asd_and_control():
    from src.eval.external_validation import group_to_label
    assert group_to_label("ASD") == 1
    assert group_to_label("control") == 0
    assert group_to_label(" ASD ") == 1        # whitespace tolerated
    assert group_to_label(1.0) == 1            # declared SPSS code
    assert group_to_label(7.0) == 0            # declared SPSS control code


def test_unknown_group_value_raises():
    """An unrecognised code must raise, never silently fold into class 0."""
    from src.eval.external_validation import LabelError, group_to_label
    for bad in ("unknown", "ASD?", "", None, "2", 2.0, 3.5):
        with pytest.raises(LabelError):
            group_to_label(bad)


def test_polish_group_column_contains_only_known_values(polish_frame):
    values = set(polish_frame["group"].astype(str).str.strip().unique())
    assert values <= {"ASD", "control"}, f"unexpected group values: {values}"


# ------------------------- 5. clinical target is not questionnaire-derived
def test_clinical_group_not_reproduced_by_questionnaire_score(polish_frame):
    """The clinical target must NOT be a threshold over the questionnaire score.

    If this ever became true, the external cohort would carry no information the
    development cohort lacks, and the whole validation would be circular.
    """
    score = __import__("pandas").to_numeric(polish_frame["Sum_QCHAT"], errors="coerce")
    y = (polish_frame["group"].astype(str).str.strip() == "ASD").astype(int)
    best = max(float(((score >= t).astype(int) == y).mean())
               for t in sorted(score.dropna().unique()))
    assert best < 0.99, (
        f"clinical GROUP is reproducible from Sum_QCHAT at accuracy {best}; "
        f"the external cohort would be circular")


def test_development_cohort_remains_circular_for_contrast():
    """Guards the contrast the external cohort exists to provide."""
    from src.audits.circularity import audit_circularity
    from src.data.ingest import load_dataset
    saudi = audit_circularity(load_dataset("saudi", synthetic=False))
    assert saudi["exact_match_rate"] == pytest.approx(1.0)
    assert "Deterministic" in str(saudi["classification"])


# ------------------------------------------------------- 6. leakage guard
def test_polish_participants_never_appear_in_development_training():
    """Development and external cohorts must be disjoint by label source."""
    from src.data.ingest import load_dataset
    saudi = load_dataset("saudi", synthetic=False)
    polish = load_dataset("polish", synthetic=False)
    assert {r["label_source"] for r in saudi} == {"questionnaire"}
    assert {r["label_source"] for r in polish} == {"clinical"}
    # the development cohort exposes no participant key at all, so a key-based
    # intersection is impossible by construction rather than by filtering
    assert all("child_id" not in r for r in saudi)


def test_external_cohort_has_a_distinct_instrument_representation():
    from src.data.ingest import load_dataset
    saudi = load_dataset("saudi", synthetic=False)
    polish = load_dataset("polish", synthetic=False)
    assert saudi[0]["item_responses"].shape[0] == 10
    assert polish[0]["item_responses"].shape[0] == 25
    # Saudi is binary, Polish is multi-level: they cannot be silently pooled
    import numpy as np
    assert set(np.unique(saudi[0]["item_responses"])) <= {0.0, 1.0}
    assert len(set(np.unique(polish[0]["item_responses"]))) > 2


# ------------------------------------- 7/8. external validation cannot fit
class _Fittable:
    def fit(self): ...
    def fit_calibrator(self): ...

    def __call__(self, state):
        return 0.5


def test_external_validation_rejects_an_object_that_can_fit():
    from src.eval.external_validation import (ExternalValidationBlocked,
                                              assert_no_fitting)
    with pytest.raises(ExternalValidationBlocked, match="fitting method"):
        assert_no_fitting(_Fittable())


def test_external_validation_rejects_a_calibrator_fitter():
    from src.eval.external_validation import (ExternalValidationBlocked,
                                              assert_no_fitting)

    class OnlyCalibrator:
        def fit_calibrator(self): ...

        def __call__(self, state):
            return 0.5

    with pytest.raises(ExternalValidationBlocked, match="fit_calibrator"):
        assert_no_fitting(OnlyCalibrator())


def test_frozen_predictor_exposes_no_training_surface():
    from src.eval.external_validation import FrozenPredictor
    for name in ("fit", "fit_calibrator", "train", "partial_fit", "calibrate"):
        assert not hasattr(FrozenPredictor, name), f"FrozenPredictor exposes {name}"


def test_frozen_predictor_is_immutable():
    from src.eval.external_validation import FrozenPredictor
    fp = FrozenPredictor(object(), {"n_items": 10})
    with pytest.raises(AttributeError):
        fp.manifest = {"tampered": True}


# ------------------------------- 9. ordinal responses are never truncated
def _fake_frozen(n_items=10, m_list=None):
    """A FrozenPredictor wrapping a stub whose declared contract we control."""
    from src.eval.external_validation import FrozenPredictor

    class _Inner:
        pass

    inner = _Inner()
    inner.n_items = n_items
    inner.m_list = m_list
    inner.__call__ = lambda self, state: 0.5
    return FrozenPredictor(inner, {"n_items": n_items, "m_list": m_list})


#: The real Polish cardinality: 25 items, two of which have 4 levels.
POLISH_M_LIST = [5, 4] + [5] * 11 + [4] + [5] * 11
assert len(POLISH_M_LIST) == 25 and sum(POLISH_M_LIST) == 123


def test_ordinal_responses_are_not_silently_truncated():
    """A multi-level cohort must be refused, not squeezed into a binary slot."""
    from src.eval.external_validation import (ExternalValidationBlocked, preflight)
    with pytest.raises(ExternalValidationBlocked) as excinfo:
        preflight(_fake_frozen(), cohort_n_items=25, cohort_m_list=POLISH_M_LIST)
    message = str(excinfo.value)
    assert "item-count mismatch" in message
    assert "response-scale mismatch" in message
    assert "coercion" in message.lower()


def test_feature_compatibility_reports_widths_without_proposing_a_fix():
    from src.eval.gates import feature_compatibility
    result = feature_compatibility(10, None, 25, POLISH_M_LIST)
    assert result["frozen_input_dim"] == 41          # binary: 4n+1
    assert result["cohort_input_dim"] == 199         # ordinal: 3n + sum(m) + 1
    assert result["compatible"] is False
    assert result["blocking_reasons"]


def test_ordinal_encoder_width_matches_encode_state():
    """The m_list plumbing must agree with src/env/state.py, not invent a width."""
    import numpy as np
    from src.env.state import encode_state, update_state
    state = {"mask": np.zeros(25, dtype=int), "value": np.full(25, -1, dtype=int),
             "n": 25, "questions_remaining": 6, "budget": 6}
    state = update_state(state, 0, 2, 5)
    state["questions_remaining"], state["budget"] = 5, 6
    vector = encode_state(state, m_list=POLISH_M_LIST,
                          questions_remaining=5, budget=6)
    assert len(vector) == 3 * 25 + sum(POLISH_M_LIST) + 1 == 199


# ------------------------------- 10. artifacts carry no participant rows
def test_result_artifacts_contain_no_participant_rows():
    """Artifacts must not embed participant identifiers.

    Only unambiguous keys are checked. The cohort contains short purely-numeric
    keys such as '220', and substring-matching those against arbitrary JSON would
    fire on any number in the file, so a 3-character numeric key is not evidence
    of a leak. Keys of 6+ characters are unique enough to test on.
    """
    import pandas as pd
    if not POLISH_CSV.exists():
        pytest.skip("integrated Polish CSV not present")
    ids = {x.strip() for x in pd.read_csv(POLISH_CSV, encoding="utf-8")["child_id"]
           if len(x.strip()) >= 6}
    assert ids, "expected unambiguous participant keys in the cohort"
    for name in ("polish_provenance_verification.json",
                 "v4_v7_validation_status.json",
                 "polish_external_validation.json"):
        path = RESULTS / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        leaked = sorted(pid for pid in ids if pid in text)
        assert not leaked, f"{name} contains participant identifiers: {leaked[:5]}"
        assert '"participant_rows"' not in text


def test_provenance_payload_guard_rejects_row_shaped_fields():
    from src.data.provenance import _assert_no_participant_payload
    with pytest.raises(AssertionError):
        _assert_no_participant_payload({"participant_rows": [1, 2, 3]})
    _assert_no_participant_payload({"row_count": 252, "dataset_identity": "same_cohort"})


def test_artifacts_declare_they_hold_no_participant_rows():
    for name in ("polish_provenance_verification.json",
                 "polish_external_validation.json"):
        path = RESULTS / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("contains_participant_rows") is False


# ------------------------------------------------- 11. provenance hash kept
def test_provenance_hash_is_recorded(gate_artifact):
    path = RESULTS / "polish_provenance_verification.json"
    if not path.exists():
        pytest.skip("run scripts/step9_v4_v7_gates.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["source_sha256"]) == 64
    assert len(data["integrated_sha256"]) == 64
    assert data["verification_timestamp"]


def test_gate_artifact_records_git_sha_and_timestamp(gate_artifact):
    assert gate_artifact["git_sha"]
    assert gate_artifact["generated"]
    assert set(gate_artifact["gates"]) >= {"V-4", "V-7", "external_validation"}


# ------------------------------------------- 12/13. failures must be clear
def test_missing_model_artifact_fails_clearly(tmp_path):
    from src.eval.external_validation import (ExternalValidationBlocked,
                                              load_frozen_predictor)
    manifest = {"n_items": 10, "m_list": None, "hidden": [128, 64],
                "calibration": "platt", "seed": 0}
    with pytest.raises(ExternalValidationBlocked, match="weights missing"):
        load_frozen_predictor(tmp_path / "absent.pt", tmp_path / "absent.pkl", manifest)


def test_missing_calibrator_fails_clearly_and_refuses_to_fit(tmp_path):
    from src.eval.external_validation import (ExternalValidationBlocked,
                                              load_frozen_predictor)
    manifest = {"n_items": 10, "m_list": None, "hidden": [128, 64],
                "calibration": "platt", "seed": 0}
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not a real checkpoint")
    with pytest.raises(ExternalValidationBlocked):
        load_frozen_predictor(weights, tmp_path / "absent.pkl", manifest)


def test_provenance_missing_input_fails_clearly(tmp_path):
    from src.data.provenance import ProvenanceError, verify_same_cohort
    with pytest.raises(ProvenanceError, match="required input missing"):
        verify_same_cohort(tmp_path / "nope.sav", tmp_path / "nope.csv")


def test_schema_mismatch_is_detected(tmp_path, polish_frame):
    """A truncated, column-dropped CSV must NOT be reported as the same cohort."""
    from src.data.provenance import verify_same_cohort
    reduced = polish_frame.drop(columns=["Sum_QCHAT"]).head(50)
    path = tmp_path / "reduced.csv"
    reduced.to_csv(path, index=False)
    result = verify_same_cohort(SAV, path)
    assert result["dataset_identity"] != "same_cohort"
    assert result["schema_equal_set"] is False
    assert result["row_count_csv"] == 50
    assert result["row_count_sav"] == 252


def test_reordered_columns_are_not_silently_accepted(tmp_path, polish_frame):
    """Column order is part of the declared contract."""
    from src.data.provenance import verify_same_cohort
    shuffled = polish_frame[list(reversed(polish_frame.columns))]
    path = tmp_path / "reordered.csv"
    shuffled.to_csv(path, index=False)
    result = verify_same_cohort(SAV, path)
    assert result["schema_equal_ordered"] is False
    assert result["schema_equal_set"] is True
    assert result["dataset_identity"] != "same_cohort"


# -------------------------------------------------- 14. reproducibility
def test_provenance_verification_is_deterministic():
    from src.data.provenance import verify_same_cohort
    if not SAV.exists():
        pytest.skip("local SPSS export not present")
    first = verify_same_cohort(SAV, POLISH_CSV)
    second = verify_same_cohort(SAV, POLISH_CSV)
    for key in ("row_count_sav", "column_count_sav", "participant_match_count",
                "participant_mismatch_count", "duplicate_count",
                "dataset_identity", "schema_equal_ordered"):
        assert first[key] == second[key], f"{key} is not deterministic"


def test_mde_is_deterministic_and_assumption_labelled():
    from src.eval.gates import compute_v4_mde
    grid = {"moderate_1.00": 1.0}
    a = compute_v4_mde(252, grid)
    b = compute_v4_mde(252, grid)
    assert a["comparisons"] == b["comparisons"]
    assert "ASSUMED" in a["assumption_note"].upper()


def test_holm_alphas_are_monotone_and_correct():
    from src.eval.gates import holm_bonferroni_alpha
    adjusted = holm_bonferroni_alpha(5, 0.05)
    assert adjusted[0] == pytest.approx(0.01)
    assert adjusted[-1] == pytest.approx(0.05)
    assert all(a <= b for a, b in zip(adjusted, adjusted[1:]))


# ------------------------------------------------------- governance gates
def test_gates_reflect_recorded_decisions_not_inference(gate_artifact):
    """Governance guarantee: a gate may only read closed because a decision was
    RECORDED, never because an automated process inferred approval. Each closed
    gate must carry its supervisor decision."""
    gates = gate_artifact["gates"]
    assert gates["V-4"]["human_sign_off"] == "NOT_REQUIRED_FOR_PRIMARY"
    assert gates["V-4"]["automated_evidence"] == "PASS"
    assert gates["V-4"]["gates_primary_external_validation"] is False
    assert gates["V-4"]["evidence"]["v4"]["supervisor_decision"]["reference"]

    assert gates["V-7"]["human_sign_off"] == "PASS"
    assert gates["V-7"]["automated_baseline10"] == "OPEN"
    assert gates["V-7"]["baseline_10_blocks_primary"] is False

    projection = gates["external_validation"]["qchat10_projection_approval"]
    assert projection["human_approval"] == "PASS"
    assert projection["automated_evidence"] == "PASS"
    assert projection["supervisor_decision"]["reference"]
    assert "information-preserving" in projection["information_loss_accepted"]
    assert "NOT" in projection["information_loss_accepted"]

    # the MDE grid is retained, not deleted
    mde = gates["V-4"]["evidence"]["v4"]["comparisons"][0]["mde_by_assumed_sd_diff"]
    assert set(mde) == {"conservative_2.00", "moderate_1.00", "optimistic_0.50"}

    # the artifact must not claim to be the thing that opens the cohort
    assert "opened by Step 10" in gate_artifact["tag"]


def test_v7_denominator_is_resolved_against_the_observed_cohort(gate_artifact):
    v7 = gate_artifact["gates"]["V-7"]["evidence"]["v7"]
    assert v7["denominator_resolved"] is True
    assert v7["observed"]["total"] == 252
    assert v7["observed"]["asd"] == 135
    assert v7["observed"]["control"] == 117
    assert v7["automated_denominator_state"] == "PASS"


def test_baseline_10_is_left_open_not_fabricated(gate_artifact):
    v7 = gate_artifact["gates"]["V-7"]["evidence"]["v7"]
    assert v7["baseline_10_recomputation"]["status"] == "OPEN"
    assert v7["baseline_10_recomputation"]["required_to_close"]


def test_external_validation_reports_metrics_after_authorisation():
    """After the 2026-10-02 supervisor decisions the primary run completes. The
    invariants that must survive are: no fabricated metrics, a real threshold, and
    an artifact that stays metadata-only."""
    path = RESULTS / "polish_external_validation.json"
    if not path.exists():
        pytest.skip("run scripts/step10_external_validation.py first")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["state"] in {"COMPLETED", "BLOCKED"}
    if data["state"] == "BLOCKED":
        assert data["metrics"] is None, "a blocked run must not report metrics"
        assert data["blockers"]
        return

    assert data["blockers"] == []
    primary = data["primary_result"]
    assert primary["threshold"] == 0.5, "primary threshold must stay frozen at 0.5"
    metrics = primary["metrics"]
    assert metrics["n"] == 252
    assert metrics["n_asd"] == 135
    assert metrics["n_control"] == 117
    # Every reported point estimate must have a real bootstrap interval behind it.
    intervals = primary["confidence_intervals"]["intervals"]
    for key in ("brier", "auroc", "sensitivity", "specificity"):
        point = metrics[key]
        ci = intervals[key]
        assert point is not None and ci["lo"] is not None and ci["hi"] is not None
        assert ci["lo"] <= point <= ci["hi"]
    assert data["secondary_sensitivity"]["threshold"] == 0.3
    assert data["secondary_sensitivity"]["not_used_to_tune"] is True
    assert data["contains_participant_rows"] is False
    assert data["frozen_model"]["retrained"] is False


def test_claim_boundary_present_in_artifacts():
    for name in ("polish_external_validation.json", "v4_v7_validation_status.json"):
        path = RESULTS / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "not a diagnostic" in text or "not clinical" in text
        assert "research prototype" in text