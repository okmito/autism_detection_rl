"""Calibration-authority and Q-CHAT-10 subset feasibility regression tests.

These lock two findings that were previously only prose:

1. **Calibration authority.** Step 2 recorded `isotonic` in
   ``results/predictor_*_metrics.json`` but persisted no weights at all, so those
   metrics could not be reproduced from disk. The only saved predictor was the
   demo's Platt cache, which is a different artefact. Step 2 now persists a
   versioned, hashed isotonic artefact, and these tests assert the recorded
   metrics are regenerable from it.
2. **Subset feasibility.** All 25 Polish items are ordinal and the cohort carries
   no question wording, so a Q-CHAT-10 subset cannot be verified. These tests
   assert the analysis reports OPEN and asserts no mapping is fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
POLISH_CSV = REPO / "data" / "raw" / "Q-CHAT Polish" / "polish_qchat.csv"
STEPS = REPO / "scripts" / "step2_train_and_sweep.py"


# ------------------------------------------- calibration authority
@pytest.fixture(scope="module")
def saudi_metrics():
    path = RESULTS / "predictor_saudi_metrics.json"
    if not path.exists():
        pytest.skip("run scripts/step2_train_and_sweep.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_step2_persists_a_predictor_artifact(saudi_metrics):
    """Metrics must reference an artefact that actually exists on disk."""
    for key in ("artifact_weights", "artifact_calibrator",
                "artifact_weights_sha256", "artifact_calibrator_sha256",
                "artifact_version"):
        assert key in saudi_metrics, f"metrics do not record {key}"
    for key in ("artifact_weights", "artifact_calibrator"):
        assert (REPO / saudi_metrics[key]).exists(), f"{key} not found on disk"


def test_recorded_calibration_matches_the_persisted_artifact(saudi_metrics):
    """The filename encodes the calibration method; the metrics must agree."""
    weights = Path(saudi_metrics["artifact_weights"]).name
    assert saudi_metrics["calibration"] in weights
    assert f"v{saudi_metrics['artifact_version']}" in weights


def test_artifact_hashes_are_real_sha256(saudi_metrics):
    import hashlib
    for path_key, hash_key in (("artifact_weights", "artifact_weights_sha256"),
                               ("artifact_calibrator", "artifact_calibrator_sha256")):
        path = REPO / saudi_metrics[path_key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == saudi_metrics[hash_key], f"{path_key} hash is stale"


def test_step2_source_actually_persists_weights():
    """Guard the fix itself: a future refactor must not silently drop persistence."""
    source = STEPS.read_text(encoding="utf-8")
    assert "torch.save" in source, "step2 no longer persists model weights"
    assert "_persist_predictor" in source


def test_demo_platt_cache_is_not_mistaken_for_the_research_artifact(saudi_metrics):
    """The Platt cache is demo-only. If it ever became the research artefact the
    recorded isotonic metrics would silently describe the wrong model."""
    research = Path(saudi_metrics["artifact_weights"]).name
    demo = "demo_model_saudi_seed0_platt_v2.pt"
    assert research != demo
    assert "platt" not in research


def test_gate_artifact_records_calibration_authority():
    path = RESULTS / "v4_v7_validation_status.json"
    if not path.exists():
        pytest.skip("run scripts/step9_v4_v7_gates.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    authority = data["gates"]["external_validation"]["calibration_authority"]
    assert authority["authoritative_for_research"] == "isotonic"
    assert authority["research_metrics_reproducible_from_disk"] is True
    assert "DEMO ONLY" in authority["demo_cache_role"]


# ------------------------------------------ Q-CHAT-10 subset feasibility
@pytest.fixture(scope="module")
def subset_report():
    if not POLISH_CSV.exists():
        pytest.skip("integrated Polish CSV not present")
    import pandas as pd
    from src.eval.qchat10_subset import analyse_subset_feasibility
    frame = pd.read_csv(POLISH_CSV, encoding="utf-8")
    cols = [f"qchat{i}recode" for i in range(1, 26)]
    return analyse_subset_feasibility(frame, cols, canonical_map=None,
                                      binary_rule=None)


def test_no_polish_item_is_natively_binary(subset_report):
    """A 10-item binary subset cannot be drawn without a stated conversion rule."""
    evidence = subset_report["requirements"]["scale_compatibility"]["evidence"]
    assert evidence["binary_items"] == 0
    assert evidence["ordinal_items"] == 25
    assert subset_report["requirements"]["scale_compatibility"]["status"] == "OPEN"


def test_item_identity_is_open_because_the_cohort_has_no_wording(subset_report):
    req = subset_report["requirements"]["item_identity"]
    assert req["status"] == "OPEN"
    assert req["evidence"]["item_text_present_in_dataset"] is False
    assert req["evidence"]["canonical_map_supplied"] is False


def test_provenance_is_open(subset_report):
    assert subset_report["requirements"]["provenance"]["status"] == "OPEN"


def test_subset_is_reported_not_forced(subset_report):
    """With no evidence supplied, every requirement must fail closed."""
    assert subset_report["subset_defensible"] is False
    assert subset_report["state"] == "OPEN"
    assert subset_report["automated_evidence"] == "OPEN"
    assert set(subset_report["blocking_requirements"]) == {
        "scale_compatibility", "item_identity", "provenance",
        "supervisor_approval"}


def test_verified_evidence_passes_but_human_approval_stays_open():
    """The invariant that matters: complete automated evidence must never close
    the gate on its own. Supervisor approval is the one thing code cannot supply."""
    if not POLISH_CSV.exists():
        pytest.skip("integrated Polish CSV not present")
    import pandas as pd

    from src.data.qchat10_contract import CANONICAL_MAPPING, MODEL_FEATURE_ORDER
    from src.eval.qchat10_subset import analyse_subset_feasibility

    report = analyse_subset_feasibility(
        pd.read_csv(POLISH_CSV, encoding="utf-8"),
        [f"qchat{i}recode" for i in range(1, 26)],
        canonical_map={e["polish_var"]: f"Q{e['qchat10_item']}"
                        for e in MODEL_FEATURE_ORDER},
        binary_rule={"method": "printed-letter derivation"},
        primary_sources=[{"document": "QCHAT.pdf"}],
    )
    reqs = report["requirements"]
    assert reqs["scale_compatibility"]["status"] == "PASS"
    assert reqs["item_identity"]["status"] == "PASS"
    assert reqs["provenance"]["status"] == "PASS"
    assert reqs["supervisor_approval"]["status"] == "OPEN"
    assert report["automated_evidence"] == "PASS"
    assert report["supervisor_approval"] == "OPEN"
    assert report["approved_as_external_representation"] is False
    assert report["subset_defensible_on_evidence"] is True
    # still blocked overall
    assert report["state"] == "OPEN"
    assert report["blocking_requirements"] == ["supervisor_approval"]
    assert [CANONICAL_MAPPING[i] for i in range(1, 11)] == [
        1, 2, 5, 6, 9, 10, 15, 17, 19, 25]


def test_analysis_refuses_to_invent_a_mapping():
    """Supplying a partial map must not be enough to declare a subset defensible."""
    import pandas as pd
    from src.eval.qchat10_subset import analyse_subset_feasibility
    frame = pd.read_csv(POLISH_CSV, encoding="utf-8")
    cols = [f"qchat{i}recode" for i in range(1, 26)]
    partial = {c: f"QCHAT-10-{i}" for i, c in enumerate(cols[:3], 1)}
    report = analyse_subset_feasibility(frame, cols, canonical_map=partial)
    assert report["subset_defensible"] is False
    assert report["requirements"]["item_identity"]["status"] == "OPEN"


def test_encoder_widths_are_reported_not_bridged(subset_report):
    assert subset_report["development_input_dim"] == 41
    assert subset_report["external_input_dim"] == 199
    assert "No ordinal value is truncated" in subset_report["no_coercion_policy"]


# ------------------------------------------------------ governance honesty
def test_evidence_requests_document_every_open_item():
    path = REPO / "EVIDENCE_REQUESTS.md"
    assert path.exists(), "evidence-request document is missing"
    text = path.read_text(encoding="utf-8")
    for token in ("E-1", "E-2", "E-3", "OPEN", "Baseline 10"):
        assert token in text


def test_decision_memo_compares_all_three_options():
    path = REPO / "POLISH_VALIDATION_DECISION.md"
    assert path.exists(), "decision memo is missing"
    text = path.read_text(encoding="utf-8")
    for token in ("Option A", "Option B", "Option C",
                  "Scientific question answered", "Leakage risk",
                  "Retraining required", "Does it validate the current system?"):
        assert token in text, f"memo omits {token}"


def test_decision_memo_records_no_selection():
    """The memo must recommend without deciding."""
    text = (REPO / "POLISH_VALIDATION_DECISION.md").read_text(encoding="utf-8")
    assert "AWAITING SUPERVISOR DECISION" in text
    assert "recommendation, not a decision" in text


def test_v7_denominator_stays_resolved_not_reopened():
    path = RESULTS / "v4_v7_validation_status.json"
    if not path.exists():
        pytest.skip("run scripts/step9_v4_v7_gates.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    v7 = data["gates"]["V-7"]["evidence"]["v7"]
    assert v7["denominator_resolved"] is True
    assert v7["observed"]["total"] == 252
    assert v7["baseline_10_recomputation"]["status"] == "OPEN"


def test_external_validation_completed_after_supervisor_approval():
    """Supervisor approved the projection and denominator (2026-10-02), so the
    primary run legitimately completes. What must still hold is that it reports
    real metrics and that the gate remains capable of blocking."""
    path = RESULTS / "polish_external_validation.json"
    if not path.exists():
        pytest.skip("run scripts/step10_external_validation.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["state"] == "COMPLETED"
    assert data["blockers"] == []
    metrics = data["primary_result"]["metrics"]
    assert metrics["n"] == 252
    assert metrics["brier"] is not None
    assert metrics["auroc"] is not None


def test_gate_still_blocks_if_any_condition_fails():
    """The real invariant: approval removed two conditions, not the gate itself."""
    from src.eval.gates import primary_gate_open
    good = {name: "PASS" for name in
            ("qchat10_projection_evidence", "polish_provenance", "leakage_audit",
             "circularity_audit", "frozen_predictor_artifact",
             "calibration_artifact_reproducibility", "polish_denominator")}
    assert primary_gate_open(good)["open"] is True
    for failing in good:
        broken = dict(good)
        broken[failing] = "FAILED"
        result = primary_gate_open(broken)
        assert result["open"] is False
        assert failing in result["failed"]
    missing = dict(good)
    del missing["leakage_audit"]
    assert primary_gate_open(missing)["open"] is False


def test_mde_and_baseline_10_no_longer_block_primary():
    from src.eval.gates import NON_BLOCKING_FOR_PRIMARY
    assert "v4_sd_ratification" in NON_BLOCKING_FOR_PRIMARY
    assert "v7_baseline_10" in NON_BLOCKING_FOR_PRIMARY


def test_primary_threshold_is_frozen_at_0_5():
    from src.eval.gates import PRIMARY_TAU, SECONDARY_TAU
    assert PRIMARY_TAU == 0.5
    assert SECONDARY_TAU == 0.3


def test_rl_research_contract_untouched():
    """The external-validation work must not have altered the RL pipeline."""
    from src.models.masked_predictor import MaskedPredictor
    from src.env.environment import run_episode
    # 10 binary items, 41-dim encoder: the development contract
    from src.env.state import encode_state
    import numpy as np
    state = {"mask": np.zeros(10, dtype=int), "value": np.full(10, -1, dtype=int),
             "n": 10, "questions_remaining": 6, "budget": 6}
    assert len(encode_state(state, questions_remaining=6, budget=6)) == 41
    assert MaskedPredictor.VERSION == 2
    assert "ExternalValidationBlocked" not in run_episode.__doc__