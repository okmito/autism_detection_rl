"""P4 demo-wiring tests: the report is generated from the answers already given.

Covers the two boundaries the support layer adds to the demo:

* **backend** — the finished session retains its episode result and
  explanation, the support report is assembled from them alone (no answers
  parameter, no second questionnaire), the screening values are unchanged, and
  the support endpoint is gone;
* **frontend** — the report panel renders only backend fields, no support
  questionnaire exists anywhere in the page, and the skip/dismiss control is
  local.

The demo backend is loaded the same way the existing audit tests load it, and
the predictor is a small synthetic one so no participant data is involved.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.models.masked_predictor import MaskedPredictor
from src.support.questionnaire import ITEM_BY_CODE, ITEM_CODES

N_ITEMS = 10
BUDGET = 6
TAU = 0.5
REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "scripts" / "demo_static" / "index.html"

ATYPICAL = [1] * 10
TYPICAL = [0] * 10


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


@pytest.fixture(scope="module")
def trained_platt():
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(240):
        vals = (rng.random(N_ITEMS) < 0.3).astype(int).tolist()
        label = int(sum(vals) >= 4)
        if rng.random() < 0.08:
            label = 1 - label
        recs.append(_mk_record(vals, label))
    y = np.array([r["label"] for r in recs])
    idx = np.arange(len(recs))
    tr, _te = next(StratifiedKFold(4, shuffle=True, random_state=0).split(idx, y))
    train = [recs[i] for i in tr]
    val = train[-40:]
    fit = train[:-40]
    pred = MaskedPredictor(n_items=N_ITEMS, hidden=[32, 16], calibration="platt")
    pred.fit(fit, epochs=40, lr=2e-3, batch_size=32, seed=0)
    pred.fit_calibrator(val, method="platt")
    return pred, fit


def _load_demo_app():
    spec = importlib.util.spec_from_file_location(
        "demo_app_support", REPO / "scripts" / "demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _session(trained_platt, responses, mode="auto", policy="greedy"):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train,
                   "test": [_mk_record(responses, 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": mode, "policy": policy})
    if mode == "interactive":
        for _ in range(BUDGET):
            a = demo.api_answer({"session_id": r["session_id"], "value": 1})
            if a.get("finished"):
                break
        r = {"session_id": r["session_id"], "result": a["result"]}
    sid = r["session_id"]
    return demo, sid, r["result"]


# ------------------------------------------------------------------ backend
def test_result_carries_the_generated_report(trained_platt):
    demo, sid, res = _session(trained_platt, ATYPICAL)
    assert "support_report" in res
    report = res["support_report"]
    assert report["schema_version"] == "support-report/2.0"
    assert report["recommendations"] or report["support_assessments"]
    json.dumps(report)


def test_report_is_built_from_the_session_alone(trained_platt):
    demo, sid, res = _session(trained_platt, ATYPICAL)
    session = demo.S["sessions"][sid]
    assert session.finished and session.episode_result is not None
    # no answers are used or stored anywhere
    assert not hasattr(session, "followup_payload")
    assert not hasattr(session, "support")
    import inspect
    assert "answers" not in inspect.signature(session.support_report).parameters


def test_screening_values_are_unchanged_by_the_report(trained_platt):
    """The support layer is a consumer: the same session with and without it
    must produce the same outcome."""
    demo, sid, res = _session(trained_platt, ATYPICAL)
    report = res["support_report"]
    assert report["screening_result"]["p_hat"] == res["p_hat"]
    assert report["screening_result"]["items_asked"] == res["items_asked"]
    assert report["screening_result"]["stop_reason"] == res["stop_reason"]
    assert report["explanation"] is not None


def test_report_evidence_uses_the_verified_instrument(trained_platt):
    _demo, _sid, res = _session(trained_platt, ATYPICAL)
    report = res["support_report"]
    for e in report["questionnaire_evidence"]:
        assert e["question_id"] in ITEM_CODES
        item = ITEM_BY_CODE[e["question_id"]]
        assert e["feature_id"] == item.feature_id
        assert e["question_text"] == item.question_text
        assert e["response"] in (0, 1)


def test_no_support_questionnaire_endpoint_remains(trained_platt):
    demo, sid, _res = _session(trained_platt, ATYPICAL)
    assert not hasattr(demo, "api_support")
    assert not hasattr(demo, "_question_payload")


def test_typical_answers_produce_no_suggestions(trained_platt):
    _demo, _sid, res = _session(trained_platt, TYPICAL)
    report = res["support_report"]
    assert report["recommendations"] == []
    assert report["unassessed_areas"]


def test_auto_and_interactive_both_carry_the_report(trained_platt):
    _demo, _sid, auto_res = _session(trained_platt, ATYPICAL, mode="auto")
    _demo2, _sid2, int_res = _session(trained_platt, ATYPICAL, mode="interactive")
    assert "support_report" in auto_res and "support_report" in int_res


def test_unfinished_session_cannot_be_reported(trained_platt):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train,
                   "test": [_mk_record(ATYPICAL, 1)], "records": None,
                   "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "interactive", "policy": "greedy"})
    with pytest.raises(ValueError, match="finish the screening session"):
        demo.api_support({"session_id": r["session_id"], "answers": []}) \
            if hasattr(demo, "api_support") else \
            demo.S["sessions"][r["session_id"]].support_report()
    demo.S["sessions"].clear()


# ----------------------------------------------------------------- frontend
def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_no_support_questionnaire_anywhere_in_the_page():
    html = _html()
    for token in ("supportScreen", "supportForm", "supportSubmit",
                  "supportBtn", "collectSupportAnswers", "renderSupportForm",
                  "/api/session/support", "followup"):
        assert token not in html, f"the second questionnaire left {token} behind"


def test_support_panel_renders_the_report_it_was_given():
    html = _html()
    assert 'id="resSupportBox"' in html
    assert 'id="resSupportBody"' in html
    assert "renderSupportReport(res.support_report);" in html
    assert "state.followup" not in html


def test_frontend_composes_no_suggestion_copy():
    """Every suggestion, status and reason must come from the backend."""
    html = _html()
    assert "SUPPORT_STATUS_PRESENTATION" in html       # enum -> label only
    for status in ("evidence_suggested", "no_evidence", "not_measured"):
        assert status in html
    for strategy_text in ("Pair words with something to see",
                          "Name what you see", "Get attention before you speak",
                          "extra processing time"):
        assert strategy_text not in html


def test_dismiss_is_local_and_returns_to_the_result():
    html = _html()
    assert "$('resSupportBox').removeAttribute('open');" in html
    block = html[html.find("$('supportSkipBtn')"):]
    block = block[:block.find("});")]
    assert "api(" not in block, "dismissing must not call the backend"
    assert "Suggestions skipped" in block


def test_panel_never_claims_a_diagnosis_or_a_need():
    html = _html()
    block = html[html.find('id="resSupportBox"'):]
    block = block[:block.find("</section>")]
    for banned in ("diagnos", "deficit", "disorder", "impair", "treatment",
                   "cure", "suffers", "probability of autism"):
        assert banned not in block.lower(), banned


def test_support_state_resets_with_the_session():
    html = _html()
    body = html[html.find("function resetAll"):
                html.find("function resetAll") + 1600]
    assert "$('resSupportBody').innerHTML = ''" in body
    assert "$('resSupportBox').removeAttribute('open');" in body
