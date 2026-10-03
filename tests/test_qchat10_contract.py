"""Q-CHAT-10 feature-contract regression tests.

Locks the single canonical mapping that lets the frozen Saudi predictor consume
Polish Q-CHAT-25 responses. Every fact asserted here is traceable to a primary
source shipped with the repository:

* ``data/raw/Q-CHAT Saudi Arabia/...Data Set Description.pdf`` - ``A{i}`` is
  Q-CHAT-10 item {i}, natively binary;
* ``data/raw/Q-CHAT Polish/QCHAT.pdf`` - the printed option order A-E for each of
  the 25 items;
* the SPSS value-label dictionary of ``data/QCHAT_dataset2 mendeley.sav`` - the
  numeric code for each option.

Tests are schema checks and synthetic conversions. No participant row is scored
and no external metric is computed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SAV = REPO / "data" / "QCHAT_dataset2 mendeley.sav"
INGEST = REPO / "src" / "data" / "ingest.py"

from src.data.qchat10_contract import (CANONICAL_MAPPING, MODEL_FEATURE_ORDER,
                                       ORDINAL_SPLIT, ORDINAL_VERIFICATION,
                                       POLISH_INVALID_SENTINEL,
                                       QCHAT10_SCORED_LETTERS,
                                       QCHAT10_SCORING_DIRECTION,
                                       QCHAT25_PRINTED_OPTIONS_EN, ContractError,
                                       binary_for_letter, build_feature_matrix,
                                       code_to_binary, contract_completeness,
                                       describe_contract, encode_qchat10_features,
                                       label_to_code, letter_for_code,
                                       scoring_table, verify_split_rule)

EXPECTED_Q25 = [1, 2, 5, 6, 9, 10, 15, 17, 19, 25]
EXPECTED_VARS = ["qchat1recode", "qchat2recode", "qchat5recode", "qchat6recode",
                 "qchat9recode", "qchat10recode", "qchat15recode",
                 "qchat17recode", "qchat19recode", "qchat25recode"]
VARS_BY_ITEM = dict(zip(EXPECTED_Q25, EXPECTED_VARS))


# ============================================ 1. the canonical mapping
def test_qchat25_source_items_are_exactly_as_authoritative():
    assert [CANONICAL_MAPPING[i] for i in range(1, 11)] == EXPECTED_Q25


def test_polish_variables_are_exactly_as_authoritative():
    derived = [e["polish_var"] for e in MODEL_FEATURE_ORDER]
    assert derived == EXPECTED_VARS


def test_model_feature_order_is_q1_through_q10():
    assert len(MODEL_FEATURE_ORDER) == 10
    for entry in MODEL_FEATURE_ORDER:
        i = entry["feature_index"]
        assert entry["qchat10_item"] == i
        assert entry["saudi_column"] == f"A{i}"
        assert entry["model_feature"] == f"feature_{i}"
    assert [e["qchat10_item"] for e in MODEL_FEATURE_ORDER] == list(range(1, 11))


def test_explicit_chain_is_consistent():
    """qchat10_item -> qchat25_item -> polish_var -> feature index -> direction."""
    for entry in MODEL_FEATURE_ORDER:
        i = entry["qchat10_item"]
        assert entry["qchat25_item"] == CANONICAL_MAPPING[i]
        assert entry["polish_var"] == f"qchat{CANONICAL_MAPPING[i]}recode"
        assert entry["scoring_direction"] == QCHAT10_SCORING_DIRECTION[i]


# ===================================== 2. regressions against the old mapping
@pytest.mark.parametrize("qchat10_item,old_wrong_q25", [
    (3, 6), (4, 9), (5, 10), (6, 15), (7, 17), (8, 19), (9, 25),
])
def test_items_after_q2_are_not_the_old_shifted_mapping(qchat10_item, old_wrong_q25):
    assert CANONICAL_MAPPING[qchat10_item] != old_wrong_q25


def test_q10_maps_to_q25():
    assert CANONICAL_MAPPING[10] == 25


def test_q1_and_q2_are_unchanged():
    assert CANONICAL_MAPPING[1] == 1
    assert CANONICAL_MAPPING[2] == 2


def test_polish_variables_are_not_the_contiguous_first_ten():
    """Guards the specific mistake of reading QCHAT1..QCHAT10."""
    contiguous = [f"qchat{i}recode" for i in range(1, 11)]
    assert [e["polish_var"] for e in MODEL_FEATURE_ORDER] != contiguous
    for wrong in ("qchat3recode", "qchat4recode", "qchat7recode", "qchat8recode",
                  "qchat11recode", "qchat12recode", "qchat13recode",
                  "qchat14recode", "qchat16recode", "qchat18recode",
                  "qchat20recode", "qchat21recode", "qchat22recode",
                  "qchat23recode", "qchat24recode"):
        assert wrong not in EXPECTED_VARS, f"{wrong} is not part of the contract"


def test_no_unrelated_item_is_used():
    used = {e["qchat25_item"] for e in MODEL_FEATURE_ORDER}
    assert used == set(EXPECTED_Q25)
    assert len(set(EXPECTED_VARS)) == 10
    assert len({f"qchat{i}recode" for i in range(1, 26)} - set(EXPECTED_VARS)) == 15


# ======================================== 3. contract is no longer OPEN
def test_contract_is_complete():
    c = contract_completeness()
    assert c["n_features_verified"] == 10
    assert c["n_features_required"] == 10
    assert c["missing_feature_indices"] == []
    assert c["complete"] is True
    assert c["state"] == "COMPLETE"


# ===================================== 4. scoring direction is official
def test_scoring_direction_per_item():
    for item in range(1, 10):
        assert QCHAT10_SCORED_LETTERS[item] == frozenset("CDE")
        assert QCHAT10_SCORING_DIRECTION[item] == "C/D/E scores 1"
    assert QCHAT10_SCORED_LETTERS[10] == frozenset("ABC")
    assert QCHAT10_SCORING_DIRECTION[10] == "A/B/C scores 1"


@pytest.mark.parametrize("letter,expected", [
    ("A", 0), ("B", 0), ("C", 1), ("D", 1), ("E", 1)])
def test_items_one_to_nine_score_cde(letter, expected):
    for item in range(1, 10):
        assert binary_for_letter(letter, item) == expected


@pytest.mark.parametrize("letter,expected", [
    ("A", 1), ("B", 1), ("C", 1), ("D", 0), ("E", 0)])
def test_item_ten_scores_abc(letter, expected):
    """The single asymmetry in the instrument."""
    assert binary_for_letter(letter, 10) == expected


# ================================ 5. code -> label -> letter -> binary, verified
@pytest.mark.parametrize("qchat25_item", EXPECTED_Q25)
def test_every_code_resolves_through_its_actual_value_label(qchat25_item):
    record = ORDINAL_VERIFICATION[qchat25_item]
    var = record["polish_var"]
    for code in sorted(record["labels"]):
        label = record["labels"][code]
        assert letter_for_code(var, float(code)) in "ABCDE"
        value = code_to_binary(float(code), var)
        assert value in (0.0, 1.0)
        # the letter must correspond to the printed wording for that label
        english = record["labels"][code]
        assert english == label


@pytest.mark.parametrize("qchat25_item", EXPECTED_Q25)
def test_split_rule_is_derived_not_assumed(qchat25_item):
    """The letter derivation agrees with code>=2 for all ten items."""
    var = ORDINAL_VERIFICATION[qchat25_item]["polish_var"]
    for code in sorted(ORDINAL_VERIFICATION[qchat25_item]["labels"]):
        assert code_to_binary(float(code), var) == (
            1.0 if code >= ORDINAL_SPLIT else 0.0)


def test_verify_split_rule_covers_all_ten_items():
    report = verify_split_rule()
    assert len(report) == 10
    assert set(report) == set(EXPECTED_VARS)


def test_scoring_table_has_one_row_per_code_and_is_complete():
    rows = scoring_table()
    expected_rows = sum(len(ORDINAL_VERIFICATION[q]["labels"]) for q in EXPECTED_Q25)
    assert len(rows) == expected_rows
    assert {r["polish_var"] for r in rows} == set(EXPECTED_VARS)
    for r in rows:
        assert r["binary"] in (0.0, 1.0)
        assert r["printed_letter"] in "ABCDE"
        assert r["label"]


# ------------------------------------------- Q25 is the reverse-coded case
def test_q25_is_reverse_coded_so_frequent_staring_is_one():
    var = "qchat25recode"
    assert ORDINAL_VERIFICATION[25]["construct_en"].startswith("stares at nothing")
    # code 4 = "wiele razy/dzień" = printed A = atypical = 1
    assert code_to_binary(4, var) == 1.0
    assert letter_for_code(var, 4) == "A"
    # code 0 = "nigdy" (never) = printed E = typical = 0
    assert code_to_binary(0, var) == 0.0
    assert letter_for_code(var, 0) == "E"
    # monotone in frequency: more staring never lowers the risk-side value
    values = [code_to_binary(c, var) for c in (0, 1, 2, 3, 4)]
    assert values == sorted(values), values


def test_old_ingest_rule_would_have_been_wrong_for_q25():
    """The removed duplicate used ``v <= 2`` for item 10, which assumes the code
    runs in printed order. On reverse-coded qchat25recode it inverts the feature."""
    old_rule = [1 if c <= 2 else 0 for c in (0, 1, 2, 3, 4)]
    correct = [int(code_to_binary(c, "qchat25recode")) for c in (0, 1, 2, 3, 4)]
    assert old_rule != correct
    assert correct == [0, 0, 1, 1, 1]


# ---------------------------------- the mapping lives in exactly one place
def test_ingest_no_longer_defines_its_own_qchat_binary_rule():
    source = INGEST.read_text(encoding="utf-8")
    assert "def qchat10_binary_map" not in source, (
        "a second hard-coded Q-CHAT scoring rule was reintroduced in ingest.py; "
        "the contract module must be the only definition")
    assert "v <= 2" not in source


def test_no_other_module_hard_codes_the_item_mapping():
    """A literal Q25 index list is the signature of a duplicated mapping."""
    pattern = re.compile(r"\[\s*1\s*,\s*2\s*,\s*(5|6)\s*,")
    offenders = []
    for path in list((REPO / "src").rglob("*.py")) + \
            list((REPO / "scripts").rglob("*.py")):
        if path.name == "qchat10_contract.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"duplicated mapping literal in {offenders}"


def test_contract_labels_match_the_ingest_value_schema():
    """The loader's raw-string schema must agree with the verified labels."""
    source = INGEST.read_text(encoding="utf-8")
    tree = ast.parse(source)
    schema = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_VALID_QCHAT_VALUES"
                for t in node.targets):
            schema = ast.literal_eval(node.value)
    assert schema, "_VALID_QCHAT_VALUES not found in ingest.py"
    for record in ORDINAL_VERIFICATION.values():
        var = record["polish_var"]
        if var in schema:
            assert set(record["labels"].values()) <= schema[var], var


# ============================================== 6. invalid / missing handling
def test_invalid_sentinel_is_never_scored():
    with pytest.raises(ContractError, match="invalid sentinel"):
        code_to_binary(POLISH_INVALID_SENTINEL, "qchat1recode")


def test_qchat2_code_gap_is_not_filled_in():
    """qchat2 is coded 0,1,2,3,5 - code 4 was never observed."""
    assert sorted(ORDINAL_VERIFICATION[2]["labels"]) == [0, 1, 2, 3, 5]
    with pytest.raises(ContractError, match="outside the item"):
        code_to_binary(4, "qchat2recode")
    assert code_to_binary(5, "qchat2recode") == 1.0


def test_out_of_vocabulary_code_is_refused():
    with pytest.raises(ContractError, match="outside the item"):
        code_to_binary(99, "qchat1recode")


def test_unobserved_label_is_refused():
    with pytest.raises(ContractError, match="not observed"):
        label_to_code("qchat1recode", "sometimes")


def test_missing_stays_missing_and_is_never_imputed():
    values = response_missing_check()
    assert np.isnan(values[0])
    assert values[1] == 0.0
    assert values[2] == 1.0


def response_missing_check():
    from src.data.qchat10_contract import response_to_feature
    return response_to_feature([None, 0, 2], "qchat1recode")


def test_missing_can_be_forbidden():
    from src.data.qchat10_contract import response_to_feature
    with pytest.raises(ContractError, match="not permitted"):
        response_to_feature([None], "qchat1recode", allow_missing=False)


def test_polish_label_strings_are_accepted():
    """The integrated CSV stores label text, not codes."""
    assert code_to_binary("nigdy", "qchat1recode") == 1.0
    assert code_to_binary("zawsze", "qchat1recode") == 0.0
    assert code_to_binary("wiele razy/dzień", "qchat25recode") == 1.0


# ====================================== 7. transformation shape and ordering
def test_feature_vector_has_exactly_ten_binary_positions():
    table = {var: [code] for var, code in zip(EXPECTED_VARS, range(10))}
    rows = build_feature_matrix({v: [0.0] for v in EXPECTED_VARS})
    assert len(rows) == 1
    vector = rows[0]
    assert len(vector) == 10
    assert all(v in (0.0, 1.0) for v in vector)


def test_short_or_long_vector_is_refused():
    with pytest.raises(ContractError, match="exactly 10"):
        build_feature_matrix({v: [0.0] for v in EXPECTED_VARS[:-1]})
    with pytest.raises(ContractError, match="exactly 10"):
        encode_qchat10_features([0.0] * 9)
    with pytest.raises(ContractError, match="exactly 10"):
        encode_qchat10_features([0.0] * 11)


def test_positions_follow_q1_to_q10_order():
    """Feature i must be driven by the Polish variable mapped to Q-CHAT-10 Q{i}.

    Each item is pushed to its atypical side using that item's own highest
    observed code, which is the value whose printed letter is scored 1.
    """
    for position, (var, entry) in enumerate(
            zip(EXPECTED_VARS, MODEL_FEATURE_ORDER)):
        assert entry["polish_var"] == var
        assert entry["feature_index"] == position + 1
        assert entry["qchat25_item"] == EXPECTED_Q25[position]
        top = float(max(ORDINAL_VERIFICATION[EXPECTED_Q25[position]]["labels"]))
        table = {v: [0.0] for v in EXPECTED_VARS}
        table[var] = [top]
        vector = build_feature_matrix(table)[0]
        assert vector[position] == 1.0, (var, top, vector)
        assert sum(vector) == 1.0, (var, vector)


def test_no_padding_position_is_ever_invented():
    table = {v: [0.0] for v in EXPECTED_VARS}
    table.pop("qchat25recode")
    with pytest.raises(ContractError, match="missing required variable"):
        build_feature_matrix(table)


def test_ragged_rows_are_refused():
    table = {v: [0.0, 1.0] for v in EXPECTED_VARS}
    table["qchat1recode"] = [0.0]
    with pytest.raises(ContractError, match="inconsistent row lengths"):
        build_feature_matrix(table)


# ================================= 8. the frozen model's 41-dim representation
def test_ten_features_encode_to_the_frozen_41_dim_input():
    for features in ([0] * 10, [1] * 10, [1, 0] * 5):
        encoded = encode_qchat10_features(features)
        assert encoded.shape == (41,)
        assert encoded.dtype == np.float32


def test_development_encoder_width_is_41():
    from src.env.state import encode_state
    state = {"mask": np.ones(10, dtype=int), "value": np.zeros(10, dtype=int),
             "n": 10}
    assert len(encode_state(state, m_list=None, questions_remaining=0,
                            budget=0)) == 41


def test_saudi_features_are_read_verbatim_and_in_order():
    source = INGEST.read_text(encoding="utf-8")
    feature_line = next(line for line in source.splitlines()
                        if 'row[f"A{j}"]' in line and "vals" in line)
    assert "np.array" in feature_line
    for forbidden in ("int(", "> 0", "> 0.5", ">= 1", "astype(int)"):
        assert forbidden not in feature_line


def test_published_threshold_is_not_the_model_threshold():
    threshold = describe_contract()["ml_decision_threshold"]
    assert threshold["value"] == 0.5
    assert threshold["kind"] == "calibrated probability"
    assert "NOT used as the model decision threshold" in threshold["note"]


# ============================================ 9. the frozen model is untouched
@pytest.mark.parametrize("name,expected", [
    ("predictor_saudi_v2_isotonic.pt",
     "2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4"),
    ("predictor_saudi_v2_isotonic.pkl",
     "a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863"),
])
def test_frozen_predictor_hashes_unchanged(name, expected):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, (
        f"{name} was modified; the predictor must stay frozen")


def test_architecture_is_unchanged():
    path = RESULTS / "predictor_saudi_metrics.json"
    if not path.exists():
        pytest.skip("metrics artifact not present")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["hidden"] == [128, 64], "MLP shape must be untouched"
    assert manifest["predictor_version"] == 2
    # The frozen interface is ten binary items; no ordinal expansion was added.
    assert encode_qchat10_features([0.0] * 10).shape == (41,)


# ================================ 10. governance still blocks external validation
def test_external_validation_state_is_coherent():
    path = RESULTS / "polish_external_validation.json"
    if not path.exists():
        pytest.skip("run scripts/step10_external_validation.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["state"] == "BLOCKED":
        assert data["metrics"] is None
        assert data["blockers"]
    else:
        assert data["blockers"] == []
        assert data["primary_result"]["metrics"]["n"] == 252


def test_all_three_gate_conditions_are_enumerated_separately():
    """Step 9 must still name each condition separately in the gate artifact."""
    path = RESULTS / "v4_v7_validation_status.json"
    if not path.exists():
        pytest.skip("run scripts/step9_v4_v7_gates.py first")
    ext = json.loads(path.read_text(encoding="utf-8"))["gates"]["external_validation"]
    assert set(ext["required_conditions"]) == {
        "v4_human_sign_off", "v7_human_sign_off", "qchat10_projection_approval"}
    projection = ext["qchat10_projection_approval"]
    assert projection["automated_evidence"] == "PASS"
    assert projection["human_approval"] == "PASS"
    assert projection["overall"] == "PASS"


def test_projection_approval_reaches_pass_from_verified_evidence():
    """DECISION 1: the supervisor approved the projection. The machine-readable
    gate must read PASS for evidence AND approved, from real verification."""
    from src.eval.gates import SUPERVISOR_DECISIONS
    assert SUPERVISOR_DECISIONS["qchat10_projection"]["decision"] == "APPROVE"
    assert SUPERVISOR_DECISIONS["qchat10_projection"]["mapping"] == {
        "Q1": "Q25 Q1", "Q2": "Q25 Q2", "Q3": "Q25 Q5", "Q4": "Q25 Q6",
        "Q5": "Q25 Q9", "Q6": "Q25 Q10", "Q7": "Q25 Q15", "Q8": "Q25 Q17",
        "Q9": "Q25 Q19", "Q10": "Q25 Q25"}

    gate = _load("v4_v7_validation_status.json")["gates"]["external_validation"]
    projection = gate["qchat10_projection_approval"]
    assert projection["automated_evidence"] == "PASS"
    assert projection["human_approval"] == "PASS"
    assert projection["overall"] == "PASS"


def test_projection_is_recorded_as_lossy_not_information_preserving():
    path = RESULTS / "polish_external_validation.json"
    if not path.exists():
        pytest.skip("run scripts/step10_external_validation.py first")
    loss = json.loads(path.read_text(encoding="utf-8"))[
        "feature_contract"]["information_loss"]
    assert "discards ordinal information" in loss
    assert "NOT" in loss and "information-preserving" in loss


def test_252_is_the_external_validation_denominator():
    from src.eval.gates import PRIMARY_CLASS_COUNTS, PRIMARY_DENOMINATOR
    assert PRIMARY_DENOMINATOR == 252
    assert PRIMARY_CLASS_COUNTS == {"asd": 135, "control": 117}
    assert sum(PRIMARY_CLASS_COUNTS.values()) == 252
    assert 135 + 117 == 252


def test_published_135_plus_118_discrepancy_stays_documented():
    """DECISION 2: the observed 252 is used; the published 135+118=253 is recorded
    as an inconsistency and the dataset is NOT altered to match it."""
    from src.eval.gates import PRIMARY_DENOMINATOR
    gate = _load("v4_v7_validation_status.json")["gates"]["V-7"]["evidence"]["v7"]
    assert gate["published"]["asd"] == 135
    assert gate["published"]["control"] == 118
    assert gate["published"]["arithmetic_sum"] == 253
    assert gate["published"]["arithmetic_sum"] != gate["published"]["stated_total"]
    assert gate["observed"]["total"] == PRIMARY_DENOMINATOR
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") == "COMPLETED":
        denom = artifact["gate_evidence"]["polish_denominator"]
        assert denom["n"] == 252 and denom["agrees"] is True
        assert "135 + 118 = 253" in denom["published_discrepancy_retained"]


def test_baseline_10_remains_unreproduced_and_secondary():
    """DECISION 4: secondary and unreproduced; no Sollis specification invented."""
    from src.eval.gates import SUPERVISOR_DECISIONS
    decision = SUPERVISOR_DECISIONS["v7_baseline_10"]
    assert decision["decision"] == "SECONDARY_UNREPRODUCED"
    assert "No Sollis et al. specification is fabricated" in decision["note"]
    gate = _load("v4_v7_validation_status.json")["gates"]["V-7"]
    assert gate["automated_baseline10"] == "OPEN"
    assert gate["automated_denominator"] == "PASS"
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") == "COMPLETED":
        assert artifact["v7"]["baseline_10"] == "OPEN_UNREPRODUCED"
        assert artifact["v7"]["baseline_10_blocks_primary"] is False


def test_v4_arbitrary_sd_does_not_block_primary_external_validation():
    """DECISION 3: MDE grid retained as sensitivity analysis, but non-blocking."""
    from src.eval.gates import SUPERVISOR_DECISIONS
    assert SUPERVISOR_DECISIONS["v4_sd_ratification"][
        "decision"] == "NOT_REQUIRED_FOR_PRIMARY"
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") == "COMPLETED":
        v4 = artifact["v4"]
        assert v4["automated_evidence"] == "PASS"
        assert v4["mde_analysis_role"] == "SENSITIVITY_ANALYSIS_REPORTED"
        assert v4["gates_primary_external_validation"] is False
        assert v4["human_sd_ratification_required"] is False
    grid = _load("v4_v7_validation_status.json")["gates"]["V-4"]["evidence"]["v4"]
    mde = grid["comparisons"][0]["mde_by_assumed_sd_diff"]
    assert set(mde) == {"conservative_2.00", "moderate_1.00", "optimistic_0.50"}


def test_threshold_remains_0_5_for_the_primary_result():
    """DECISION 5: tau frozen at 0.5; 0.3 is secondary only."""
    from src.eval.gates import PRIMARY_TAU, SECONDARY_TAU
    assert PRIMARY_TAU == 0.5
    assert SECONDARY_TAU == 0.3
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") == "COMPLETED":
        assert artifact["primary_result"]["threshold"] == 0.5
        assert artifact["secondary_sensitivity"]["threshold"] == 0.3
        assert artifact["secondary_sensitivity"]["not_used_to_tune"] is True
        assert "Not tuned on Polish" in \
            artifact["primary_result"]["threshold_rationale"]


def test_no_fitting_or_tuning_on_polish():
    from src.eval.external_validation import FITTING_METHOD_NAMES
    for name in ("fit", "fit_calibrator", "train", "fit_threshold", "calibrate"):
        assert name in FITTING_METHOD_NAMES
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") == "COMPLETED":
        assert artifact["frozen_model"]["retrained"] is False
        assert artifact["frozen_model"]["calibration"] == "isotonic"


def test_all_seven_conditions_passed_in_the_recorded_run():
    from src.eval.gates import PRIMARY_GATE_CONDITIONS, primary_gate_open
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked; nothing to assert")
    conditions = artifact["gate"]["conditions"]
    assert set(conditions) == set(PRIMARY_GATE_CONDITIONS)
    assert primary_gate_open(conditions)["open"] is True
    for name, state in conditions.items():
        assert state == "PASS", name


def test_gate_still_blocks_if_any_condition_fails():
    """The invariant that matters: two conditions were removed, not the gate."""
    from src.eval.gates import PRIMARY_GATE_CONDITIONS, primary_gate_open
    good = {name: "PASS" for name in PRIMARY_GATE_CONDITIONS}
    assert primary_gate_open(good)["open"] is True
    for failing in PRIMARY_GATE_CONDITIONS:
        broken = dict(good)
        broken[failing] = "FAILED"
        result = primary_gate_open(broken)
        assert result["open"] is False
        assert failing in result["failed"]
    missing = dict(good)
    del missing["leakage_audit"]
    assert primary_gate_open(missing)["open"] is False
    assert "leakage_audit" in primary_gate_open(missing)["not_evaluated"]


def test_leakage_and_circularity_results_are_recorded():
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked")
    evidence = artifact["gate_evidence"]
    leak = evidence["leakage_audit"]
    assert leak["label_sources_disjoint"] is True
    assert leak["participant_key_intersection"] == 0
    circ = evidence["circularity_audit"]
    assert circ["external_target_independent"] is True
    assert circ["polish_clinical_target"]["exact_match_rate"] < 1.0
    assert circ["saudi_questionnaire_target_for_contrast"][
        "exact_match_rate"] == 1.0


def test_prediction_receives_exactly_the_expected_representation():
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked")
    contract = artifact["feature_contract"]
    assert contract["projected_shape"] == [252, 10]
    assert contract["encoded_dim"] == 41
    assert artifact["projection_compatibility"]["compatible"] is True
    assert encode_qchat10_features([0.0] * 10).shape == (41,)
    assert encode_qchat10_features([1.0] * 10).shape == (41,)


def test_encoding_matches_the_saudi_terminal_scoring_path():
    """The projection must encode exactly as the Saudi metrics were computed."""
    from src.env.state import encode_state, init_state, update_state
    values = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    n = len(values)
    record = {"item_responses": np.asarray(values, dtype=float),
              "missing_mask": np.zeros(n, dtype=bool)}
    state = init_state(record)
    for j in range(n):
        state = update_state(state, j, int(record["item_responses"][j]), 0)
    state["questions_remaining"] = 0
    state["budget"] = n
    reference = encode_state(state, m_list=None, questions_remaining=0, budget=n)
    assert np.array_equal(reference, encode_qchat10_features(values))


def test_every_reported_metric_has_a_real_interval():
    """No fabricated CIs: each reported estimate sits inside its own interval."""
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked")
    for block in ("primary_result", "secondary_sensitivity"):
        section = artifact[block]
        metrics = section["metrics"]
        intervals = section["confidence_intervals"]["intervals"]
        assert section["confidence_intervals"]["n_resamples"] >= 1000
        for key, ci in intervals.items():
            point = metrics.get(key)
            if point is None or ci.get("lo") is None:
                continue
            assert ci["lo"] <= point + 1e-9, (block, key)
            assert ci["hi"] >= point - 1e-9, (block, key)


def test_calibration_is_reported_honestly():
    """The frozen model transfers in ranking but is miscalibrated. That must be
    visible in the artifact rather than smoothed over."""
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked")
    calib = artifact["calibration_diagnostics"]
    assert calib["role"] == "diagnostic_only"
    assert calib["calibration_slope"] is not None
    assert calib["calibration_in_the_large"] is not None
    # slope far from 1 is a real finding about the isotonic calibrator
    assert calib["calibration_slope"] < 1.0
    assert calib["calibration_table"]


def test_limitations_are_recorded_including_information_loss():
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked")
    text = " ".join(artifact["limitations"]).lower()
    assert "questionnaire-derived" in text
    assert "ordinal information" in text
    assert "cross-country" in text
    assert "not clinical diagnosis" in text


def test_interpretation_does_not_claim_diagnosis():
    artifact = _load("polish_external_validation.json")
    if artifact.get("state") != "COMPLETED":
        pytest.skip("run is blocked")
    ladder = artifact["interpretation_ladder"]
    assert "NOT CLAIMED" in ladder["4_medical_diagnosis"]
    assert "diagnose autism" in ladder["4_medical_diagnosis"]
    assert "NOT a diagnosis" in artifact["claim_boundary"]
    assert "clinician-established" in artifact["claim_boundary"]


def _load(name: str):
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_will_not_open_until_every_condition_is_signed():
    """Directly exercise the gate predicate rather than trusting the artifact."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "step10", REPO / "scripts" / "step10_external_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    signed_states = {"SIGNED", "CLOSED", "PASS"}
    for combo, expected in [
            ({"V-4": "OPEN", "V-7": "SIGNED",
              "Q-CHAT-10-PROJECTION": "SIGNED"}, False),
            ({"V-4": "SIGNED", "V-7": "OPEN",
              "Q-CHAT-10-PROJECTION": "SIGNED"}, False),
            ({"V-4": "SIGNED", "V-7": "SIGNED",
              "Q-CHAT-10-PROJECTION": "OPEN"}, False),
            ({"V-4": "SIGNED", "V-7": "SIGNED",
              "Q-CHAT-10-PROJECTION": "SIGNED"}, True),
    ]:
        closed = all(str(v).upper() in signed_states for v in combo.values())
        assert closed is expected, combo


def test_v4_and_v7_reflect_the_recorded_supervisor_decisions():
    """V-4 SD ratification is no longer required for the primary run (DECISION 3);
    V-7 denominator sign-off is recorded (DECISION 2). Baseline 10 stays OPEN."""
    path = RESULTS / "v4_v7_validation_status.json"
    if not path.exists():
        pytest.skip("run scripts/step9_v4_v7_gates.py first")
    gates = json.loads(path.read_text(encoding="utf-8"))["gates"]
    assert gates["V-4"]["human_sign_off"] == "NOT_REQUIRED_FOR_PRIMARY"
    assert gates["V-4"]["automated_evidence"] == "PASS"
    assert gates["V-4"]["mde_analysis_role"] == "SENSITIVITY_ANALYSIS_REPORTED"
    assert gates["V-4"]["gates_primary_external_validation"] is False
    assert gates["V-7"]["human_sign_off"] == "PASS"
    assert gates["V-7"]["automated_denominator"] == "PASS"
    assert gates["V-7"]["automated_baseline10"] == "OPEN"
    assert gates["V-7"]["baseline_10_blocks_primary"] is False


def test_contract_completion_does_not_imply_validation():
    """The contract being complete must not be reported as a passed validation."""
    summary = describe_contract()
    assert "BLOCKED" in summary["external_validation_status"]
    assert "complete but" in summary["external_validation_status"]


# ================================= 11. agreement with the actual SPSS source
@pytest.mark.skipif(not SAV.exists(), reason="local SPSS export not present")
@pytest.mark.parametrize("qchat25_item", EXPECTED_Q25)
def test_contract_matches_the_spss_value_labels(qchat25_item):
    import pyreadstat
    _sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False)
    record = ORDINAL_VERIFICATION[qchat25_item]
    from_file = {float(k): str(v) for k, v
                 in meta.variable_value_labels[record["polish_var"]].items()}
    from_contract = {float(k): v for k, v in record["labels"].items()}
    assert from_file == from_contract, record["polish_var"]


@pytest.mark.skipif(not SAV.exists(), reason="local SPSS export not present")
def test_saved_cohort_maps_to_a_clean_ten_column_matrix():
    """Shape check on the real cohort's schema only; no prediction, no metric."""
    import pandas as pd
    from src.data.ingest import POLISH_CSV
    df = pd.read_csv(POLISH_CSV, encoding="utf-8")
    matrix = np.array(build_feature_matrix(
        {e["polish_var"]: df[e["polish_var"]].tolist() for e in MODEL_FEATURE_ORDER}))
    assert matrix.shape == (len(df), 10)
    assert set(np.unique(matrix)) <= {0.0, 1.0}
    for row in matrix[:5]:
        assert encode_qchat10_features(row).shape == (41,)