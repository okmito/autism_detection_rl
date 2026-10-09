"""The questionnaire table must be the contract's, and the demo must agree — P4.

This is the test whose absence let ``scripts/demo_app.py`` drift: it carried
its own item wording, so from A3 onward the demo asked about one construct
while the answer was recorded as the feature for a different one. Deriving the
wording here, from the verified contract, and asserting the demo's list matches
turns that drift into a test failure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from src.data.qchat10_contract import (MODEL_FEATURE_ORDER,
                                       ORDINAL_VERIFICATION,
                                       QCHAT10_SCORING_DIRECTION)
from src.support.questionnaire import (ASKED_WITHOUT_SUGGESTION_NOTE,
                                       ATYPICAL_VALUE, ITEMS, ITEM_BY_CODE,
                                       ITEM_CODES, is_valid_item_code,
                                       item_code_for_feature_index,
                                       questionnaire_table,
                                       questionnaire_version)

REPO = Path(__file__).resolve().parent.parent


def _load_demo_app():
    spec = importlib.util.spec_from_file_location(
        "demo_app_questionnaire", REPO / "scripts" / "demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the table is the instrument
# ---------------------------------------------------------------------------

def test_ten_items_with_the_expected_codes():
    assert len(ITEMS) == 10
    assert ITEM_CODES == tuple(f"A{i}" for i in range(1, 11))
    assert list(ITEM_BY_CODE) == list(ITEM_CODES)


def test_every_item_comes_from_the_verified_contract():
    """Nothing here is restated: construct, feature id, Polish variable and
    scoring direction all have to equal what the contract says."""
    for item in ITEMS:
        entry = next(e for e in MODEL_FEATURE_ORDER
                     if e["saudi_column"] == item.item_code)
        assert item.feature_id == entry["model_feature"]
        assert item.feature_index == entry["feature_index"] - 1
        assert item.qchat10_item == entry["qchat10_item"]
        assert item.qchat25_item == entry["qchat25_item"]
        assert item.polish_var == entry["polish_var"]
        assert item.scoring_direction == \
            QCHAT10_SCORING_DIRECTION[item.qchat10_item]
        assert item.construct == \
            ORDINAL_VERIFICATION[item.qchat25_item]["construct_en"]


def test_item_code_resolution():
    assert is_valid_item_code("A3") and is_valid_item_code("A10")
    for bad in ("A0", "A11", "A", "a3", "B2", ""):
        assert not is_valid_item_code(bad)
    assert item_code_for_feature_index(0) == "A1"
    assert item_code_for_feature_index(9) == "A10"


def test_answer_polarity_is_the_contracts():
    for item in ITEMS:
        assert item.atypical_value == 1 and item.typical_value == 0
        assert "atypical" in item.answer_label(1)
        assert "typical" in item.answer_label(0)


def test_table_is_json_friendly_data():
    table = questionnaire_table()
    assert len(table) == 10
    assert {row["item_code"] for row in table} == set(ITEM_CODES)
    import json
    json.dumps(table)          # must not raise


# ---------------------------------------------------------------------------
# the drift that is now locked out
# ---------------------------------------------------------------------------

def test_demo_wording_matches_the_contract():
    """The anti-drift lock: the questions the user reads must be these items."""
    demo = _load_demo_app()
    assert demo.ITEMS == [i.question_text for i in ITEMS], (
        "the demo's displayed wording has drifted from the verified "
        "questionnaire contract")


def test_no_legacy_wording_survives():
    """The specific items the drift moved. Each assertion here is a regression
    test for one mislabelled question."""
    a3 = ITEM_BY_CODE["A3"].question_text.lower()
    assert "point" in a3 and "ask for something" in a3
    assert "pay attention" not in a3          # A3 is not the attention item
    assert ITEM_BY_CODE["A4"].construct == "points to share interest with you"
    assert "pretend" in ITEM_BY_CODE["A5"].question_text.lower()
    assert "follow where you are looking" in ITEM_BY_CODE["A6"].question_text
    assert "comfort" in ITEM_BY_CODE["A7"].question_text
    assert "first words" in ITEM_BY_CODE["A8"].question_text
    assert "simple gestures" in ITEM_BY_CODE["A9"].question_text
    assert "stare" in ITEM_BY_CODE["A10"].question_text


def test_every_asked_item_is_covered_or_disclosed():
    """Nothing is offered for items with no reviewed suggestion, and the report
    has a sentence ready that says so instead of staying silent."""
    assert "A5" in ASKED_WITHOUT_SUGGESTION_NOTE
    assert "A10" in ASKED_WITHOUT_SUGGESTION_NOTE
    assert "no curated" in ASKED_WITHOUT_SUGGESTION_NOTE


def test_questionnaire_version_is_the_contracts():
    assert questionnaire_version() == "qchat10-mapping/2.0.0"
