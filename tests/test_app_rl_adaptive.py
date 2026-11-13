"""The adaptive RL policy in the web app: ASK/STOP decisions, explanations, skips, fallback, integrity.

The served artifact is built here from synthetic data in a temporary results directory, so these tests never depend
on (or touch) the real ``results/rl_adaptive`` artifacts. They need the Saudi CSV only because the app's screening
service loads it for its other policies (``screening_api`` skips without it).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import MDPConfig
from src.rl.served import SERVED_NAME, write_served_policy
from tests.test_rl_mdp import synth


def ok(resp, status=200):
    body = resp.get_json()
    assert resp.status_code == status, body
    return body["data"] if status == 200 else body


def build_artifact(root, *, passed=True, cost=0.01):
    X, y = synth(n_rec=400, seed=11)
    b = NaiveBayesBelief.fit(X, y, trained_on="synthetic test cohort")
    cfg = MDPConfig(cost=cost)
    payload = {"agent": "rl_value_iteration" if passed else None, "belief": b.to_dict(), "mdp": cfg.to_dict(),
               "gate": {"passed": passed, "reason": None if passed else "no RL agent passed the integration gate (test)"},
               "expected_performance": {"source": "test"}}
    write_served_policy(root / "results" / "rl_adaptive" / SERVED_NAME, payload)
    return root / "results" / "rl_adaptive" / SERVED_NAME


@pytest.fixture
def rl_root(tmp_path, monkeypatch, screening_api):
    from app.services import screening
    monkeypatch.setattr(screening, "_ROOT", tmp_path)
    screening._STATE.pop("rl", None)
    yield tmp_path
    screening._STATE.pop("rl", None)


@pytest.fixture
def rl_api(rl_root, screening_api):
    build_artifact(rl_root)
    return screening_api


def start(api, policy="rl_adaptive", mode="interactive", status=200):
    return ok(api.post("/api/screening/start", json={"mode": mode, "policy": policy}), status)


def answer(api, sid, item_index, value=None, status=200, **extra):
    return ok(api.post("/api/screening/answer",
                       json={"session_id": sid, "value": value, "item_index": item_index, **extra}), status)


def run(api, answer_fn):
    """Run a session answering with ``answer_fn(item_index) -> 0/1/None(skip)``; returns (result, items)."""
    s = start(api)
    sid, q = s["session_id"], s["question"]["index"]
    assert s["decision"]["action"] == {"type": "ASK", "item": f"A{q + 1}", "item_index": q}
    items = [q]
    for _ in range(10):
        v = answer_fn(q)
        r = answer(api, sid, q, v, skip=v is None)
        if r["finished"]:
            return r["result"], items, sid
        assert r["decision"]["action"]["item_index"] == r["next"]["index"]
        q = r["next"]["index"]
        items.append(q)
    raise AssertionError("RL session did not finish within the cap")


def test_meta_reports_the_rl_policy(rl_api):
    m = ok(rl_api.get("/api/meta"))
    assert "rl_adaptive" in m["policies_available"]
    assert m["rl_policy"]["available"] is True and m["rl_policy"]["agent"] == "rl_value_iteration"


def test_rl_session_asks_unique_questions_and_stops_with_an_explanation(rl_api):
    result, items, sid = run(rl_api, lambda j: 0)
    assert len(items) == len(set(items)) and 2 <= len(items) <= 10
    assert result["asked"] == len(items)
    assert result["stop_reason"] in ("policy_stop", "budget_exhausted")
    se = result["stop_explanation"]
    assert se["action"]["type"] == "STOP" and se["stop_reason"] == result["stop_reason"]
    assert result["policy_info"]["agent"] == "rl_value_iteration"
    assert result["explanation"] and result["support_report"]          # existing report layers preserved
    for step, item in zip(result["trace"], items):
        assert step["decision"]["action"]["item_index"] == item          # each step explains the choice it records


def test_question_sequence_depends_on_the_answers(rl_api):
    _, all_typical, _ = run(rl_api, lambda j: 0)
    _, all_atypical, _ = run(rl_api, lambda j: 1)
    _, mixed, _ = run(rl_api, lambda j: j % 2)
    assert len({tuple(all_typical), tuple(all_atypical), tuple(mixed)}) > 1


def test_clear_cases_stop_before_the_cap(rl_api):
    result, items, _ = run(rl_api, lambda j: 0)
    assert result["stop_reason"] == "policy_stop" and len(items) < 10


def test_rl_inference_is_deterministic(rl_api):
    a = run(rl_api, lambda j: j % 2)
    b = run(rl_api, lambda j: j % 2)
    assert a[1] == b[1] and a[0]["p_hat"] == b[0]["p_hat"]


def test_skip_is_recorded_as_a_question_without_evidence(rl_api):
    s = start(rl_api)
    q = s["question"]["index"]
    r = answer(rl_api, s["session_id"], q, None, skip=True)
    assert r["trace_step"]["skipped"] is True and r["trace_step"]["value"] is None
    assert r["belief_after"] == pytest.approx(r["belief_before"])
    assert r["asked"] == 1 and r["next"]["index"] != q


def test_invalid_answers_are_rejected(rl_api):
    s = start(rl_api)
    sid, q = s["session_id"], s["question"]["index"]
    answer(rl_api, sid, q, 1, status=400, skip=True)            # value and skip together
    answer(rl_api, sid, q, None, status=400)                     # neither value nor skip
    answer(rl_api, sid, (q + 1) % 10, 1, status=409)             # answers a question that is not pending
    ok(rl_api.post("/api/screening/answer", json={"session_id": sid, "value": 2, "item_index": q}), 400)


def test_skip_is_refused_for_fixed_policies(rl_api):
    s = start(rl_api, policy="greedy")
    answer(rl_api, s["session_id"], s["question"]["index"], None, status=400, skip=True)


def test_auto_mode_is_refused_for_rl(rl_api):
    start(rl_api, mode="auto", status=400)


def test_resume_returns_the_pending_decision(rl_api):
    s = start(rl_api)
    r = answer(rl_api, s["session_id"], s["question"]["index"], 1)
    g = ok(rl_api.get(f"/api/screening/sessions/{s['session_id']}"))
    assert g["pending_question"]["index"] == r["next"]["index"] == g["decision"]["action"]["item_index"]
    assert g["belief"] == pytest.approx(r["belief_after"])
    assert g["policy_info"]["agent"] == "rl_value_iteration"


def test_missing_artifact_falls_back_to_greedy_and_says_so(rl_root, screening_api):
    m = ok(screening_api.get("/api/meta"))
    assert "rl_adaptive" not in m["policies_available"] and m["rl_policy"]["available"] is False
    s = start(screening_api)
    assert s["policy"] == "greedy"
    assert s["policy_info"]["fallback"]["requested"] == "rl_adaptive"
    assert s["policy_info"]["fallback"]["used"] == "greedy" and s["policy_info"]["fallback"]["reason"]


def test_failed_gate_is_reported_as_the_reason(rl_root, screening_api):
    build_artifact(rl_root, passed=False)
    m = ok(screening_api.get("/api/meta"))
    assert m["rl_policy"]["available"] is False and "integration gate" in m["rl_policy"]["reason"]


def test_tampered_artifact_is_refused(rl_root, screening_api):
    path = build_artifact(rl_root)
    d = json.loads(path.read_text())
    d["mdp"]["cost"] = 0.5
    path.write_text(json.dumps(d))
    m = ok(screening_api.get("/api/meta"))
    assert m["rl_policy"]["available"] is False and "integrity" in m["rl_policy"]["reason"]


def test_policy_change_mid_session_is_a_conflict(rl_api, rl_root):
    from app.services import screening
    s = start(rl_api)
    build_artifact(rl_root, cost=0.02)
    screening._STATE.pop("rl", None)
    answer(rl_api, s["session_id"], s["question"]["index"], 1, status=409)
