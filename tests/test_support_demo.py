"""P4 demo-wiring tests: the browser demo's support path.

Covers the two boundaries the support layer adds to the demo:

* **backend** — the finished session retains its episode result and
  explanation, the follow-up payload splits suggested from optional exactly as
  the engine does, the report is assembled read-only and JSON-safe, and
  invalid answers surface as a readable error instead of a half-built report;
* **frontend** — the support screen renders only backend fields (no suggestion
  text, status or reason is composed client-side), keeps the skip path, and
  never makes a claim the screening result did not make.

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

N_ITEMS = 10
BUDGET = 6
TAU = 0.5
REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "scripts" / "demo_static" / "index.html"


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


ATYPICAL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)


# ------------------------------------------------------------------ backend
def test_finished_session_retains_result_and_explanation(trained_platt):
    demo, sid, res = _session(trained_platt, ATYPICAL)
    s = demo.S["sessions"][sid]
    assert s.finished and s.episode_result is not None
    assert s.explanation is not None
    assert res["explanation"] is s.explanation
    assert "followup" in res


def test_result_carries_the_followup_questionnaire(trained_platt):
    _demo, _sid, res = _session(trained_platt, ATYPICAL)
    fu = res["followup"]
    assert set(fu) >= {"note", "answer_values", "suggested", "optional"}
    assert "prefer_not_to_answer" in fu["answer_values"]
    # the atypical responses legitimately invite the two linked questions
    assert {q["question_id"] for q in fu["suggested"]} == {"q_comm_support",
                                                          "q_emotion_support"}
    # preference questions are never suggested by screening data
    assert all(not q["is_preference"] for q in fu["suggested"])
    optional = {q["question_id"] for q in fu["optional"]}
    assert {"q_sensory_support", "q_transitions_support",
            "q_comm_preference"} <= optional


def test_support_report_is_assembled_read_only(trained_platt):
    demo, sid, res = _session(trained_platt, ATYPICAL)
    p_before = res["p_hat"]
    out = demo.api_support({"session_id": sid,
                            "answers": [{"question_id": "q_comm_support",
                                         "value": "yes"}]})
    rep = out["report"]
    assert rep["screening_result"]["p_hat"] == p_before
    assert rep["schema_version"] == "support-report/1.0"
    assert rep["explanation"] is not None          # the session's own explanation
    assert {r["domain"] for r in rep["recommendations"]} == \
        {"social_communication"}
    json.dumps(rep)                                # JSON-safe for the frontend


def test_skipping_the_questionnaire_is_a_valid_outcome(trained_platt):
    demo, sid, _res = _session(trained_platt, ATYPICAL)
    rep = demo.api_support({"session_id": sid})["report"]
    assert rep["recommendations"] == []
    assert any(a["status"] == "hypothesis_from_observed"
               for a in rep["support_assessments"])


def test_unfinished_session_is_refused(trained_platt):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train,
                   "test": [_mk_record(ATYPICAL, 1)], "records": None,
                   "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "interactive", "policy": "greedy"})
    with pytest.raises(ValueError, match="finish the screening session"):
        demo.api_support({"session_id": r["session_id"], "answers": []})
    demo.S["sessions"].clear()


def test_invalid_answers_raise_a_readable_error(trained_platt):
    demo, sid, _res = _session(trained_platt, ATYPICAL)
    with pytest.raises(ValueError, match="invalid follow-up answers"):
        demo.api_support({"session_id": sid,
                          "answers": [{"question_id": "q_comm_preference",
                                       "value": "yes"}]})
    with pytest.raises(ValueError, match="must be a list"):
        demo.api_support({"session_id": sid, "answers": "yes"})


def test_unknown_session_is_refused(trained_platt):
    demo, _sid, _res = _session(trained_platt, ATYPICAL)
    with pytest.raises(ValueError, match="unknown session"):
        demo.api_support({"session_id": "nope", "answers": []})


# ----------------------------------------------------------------- frontend
def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_support_screen_exists_and_is_optional():
    html = _html()
    assert 'id="supportScreen"' in html
    assert 'id="supportBtn"' in html
    assert "Optional support ideas" in html
    assert "Skip" in html


def test_frontend_renders_no_invented_support_copy():
    """Every suggestion, status and reason must come from the backend."""
    html = _html()
    for backend_only in ("renderSupportReport", "SUPPORT_STATUS_PRESENTATION",
                         "collectSupportAnswers"):
        assert backend_only in html
    # the only presentation table maps backend enum values to labels
    block = html[html.find("const SUPPORT_STATUS_PRESENTATION"):
                 html.find("const BASIS_PRESENTATION")]
    for status in ("user_confirmed", "user_stated_preference",
                   "hypothesis_from_observed", "user_declined", "unknown",
                   "insufficient_evidence"):
        assert status in block
    # no suggestion text is hard-coded in the client
    assert "extra processing time" not in html
    assert "quieter space" not in html


def test_frontend_posts_to_the_support_endpoint():
    html = _html()
    assert "/api/session/support" in html
    assert "session_id: state.sessionId, answers: answers" in html


def test_frontend_never_implies_the_screening_result_found_a_need():
    html = _html()
    support = html[html.find('id="supportScreen"'):]
    support = support[:support.find("</section>")]
    for banned in ("diagnos", "deficit", "disorder", "impair", "treatment",
                   "cure", "suffers"):
        assert banned not in support.lower(), banned
    assert "not a diagnosis" in html.lower()


def test_unanswered_question_is_skipped_not_answered():
    html = _html()
    block = html[html.find("function collectSupportAnswers"):
                 html.find("function submitSupport")]
    assert "if (!pressed) return;" in block


def test_support_state_is_reset_with_the_session():
    html = _html()
    body = html[html.find("function resetAll"):
                html.find("function resetAll") + 1600]
    assert "state.followup = null" in body
    assert "$('supportScreen').hidden = true" in body
