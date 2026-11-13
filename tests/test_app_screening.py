"""Screening-session tests for the NeuroBridge app (audit finding C4 and the working flows).

C4: ``start_session`` used to offer STOP (-1) to the policy for the *opening* question. RandomPolicy
draws uniformly from the legal list, so about 1 session in 11 opened on index -1, which the UI showed
as "A0" with the text of item A10; the report builder then rejected "A0" and the session could never
finish. These tests need the (gitignored) Saudi dataset and are skipped without it.
"""
from __future__ import annotations

import pytest

from src.env.environment import STOP


class StopWhenOffered:
    """A policy whose draw lands on STOP whenever STOP is among the legal actions.

    That is the worst case of RandomPolicy (uniform over ``legal``) made deterministic.
    """

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        return STOP if STOP in legal else items[0]


def ok(resp, status=200):
    body = resp.get_json()
    assert resp.status_code == status, body
    return body["data"]


def start(api, mode="interactive", policy="greedy"):
    return ok(api.post("/api/screening/start", json={"mode": mode, "policy": policy}))


def answer(api, sid, value=None, item_index=None, **extra):
    """Answer the pending question; ``item_index`` says which question this answer is for."""
    return ok(api.post("/api/screening/answer",
                       json={"session_id": sid, "value": value, "item_index": item_index, **extra}))


def run_to_the_end(api, first):
    """Answer alternately 0/1 until the session finishes; returns (result, questions_seen)."""
    sid = first["session_id"]
    seen = [first["question"]["index"]]
    for i in range(10):
        r = answer(api, sid, i % 2, item_index=seen[-1])
        if r["finished"]:
            return r["result"], seen
        seen.append(r["next"]["index"])
    raise AssertionError("session did not finish within the budget")


# ------------------------------------------------------------------ C4


def test_opening_question_is_never_stop(screening_api, monkeypatch):
    from app.services import screening

    monkeypatch.setattr(screening, "make_policy", lambda name: StopWhenOffered())
    first = start(screening_api, policy="random")
    q = first["question"]
    assert 0 <= q["index"] < 10, "STOP (-1) must never become the opening question"
    assert q["code"] == f"A{q['index'] + 1}" and q["code"] != "A0"
    assert q["text"] == screening.ITEMS[q["index"]]


def test_a_session_that_opened_through_the_stub_completes_with_valid_items(screening_api, monkeypatch):
    from app.services import screening

    monkeypatch.setattr(screening, "make_policy", lambda name: StopWhenOffered())
    result, seen = run_to_the_end(screening_api, start(screening_api, policy="random"))
    assert result["asked"] == 6
    codes = result["items_asked"]
    assert [f"A{i + 1}" for i in seen] == codes, "the questions shown must be exactly the items recorded"
    assert len(codes) == 6 and len(set(codes)) == 6 and "A0" not in codes
    assert all(c.startswith("A") and 1 <= int(c[1:]) <= 10 for c in codes)


@pytest.mark.parametrize("bad", [STOP, -2, 10, 99, 3.5])
def test_require_legal_item_rejects_anything_that_is_not_an_askable_item(bad):
    from app.services import screening

    with pytest.raises(ValueError):
        screening._require_legal_item(bad, [0, 1, 2])


def test_require_legal_item_rejects_an_already_asked_item():
    from app.services import screening

    with pytest.raises(ValueError):
        screening._require_legal_item(5, [0, 1, 2])
    assert screening._require_legal_item(2, [0, 1, 2]) == 2


def test_a_policy_that_picks_an_illegal_question_leaves_no_session_behind(screening_api, monkeypatch):
    from app.db import get_conn, init_db
    from app.services import screening

    init_db()  # the service is called directly here, not through Flask's before_request hook
    monkeypatch.setattr(screening, "make_policy", lambda name: (lambda state, legal: 99))
    with pytest.raises(ValueError):
        screening.start_session("interactive", "greedy")
    assert get_conn().execute("SELECT COUNT(*) FROM screening_sessions").fetchone()[0] == 0


# ------------------------------------------------------------------ working flows (regression guard)


@pytest.mark.parametrize("policy", ["greedy", "beta_greedy"] + ["random"] * 8)
def test_every_served_policy_completes_an_interactive_session(screening_api, policy):
    first = start(screening_api, policy=policy)
    assert first["policy"] == policy and 0 <= first["question"]["index"] < 10
    result, seen = run_to_the_end(screening_api, first)
    assert result["asked"] == 6 and result["stop_reason"] == "budget_exhausted"
    assert len(set(result["items_asked"])) == 6
    assert 0.0 <= result["p_hat"] <= 1.0
    assert result["decision"] in ("REFERRAL_RECOMMENDED", "NO_REFERRAL_INDICATED")
    assert [f"A{i + 1}" for i in seen] == result["items_asked"], "shown questions == recorded items, in order"


@pytest.mark.parametrize("policy", ["greedy", "random", "beta_greedy"])
def test_auto_mode_returns_a_complete_result(screening_api, policy):
    res = start(screening_api, mode="auto", policy=policy)["result"]
    # random may legitimately STOP on its very first move (about 1 in 11), so zero questions is valid
    assert res["asked"] == len(res["items_asked"]) and 0.0 <= res["p_hat"] <= 1.0
    assert res["support_report"]["screening_result"]["decision"] in (
        "REFERRAL_RECOMMENDED", "NO_REFERRAL_INDICATED")


def test_user_can_stop_after_at_least_one_answer(screening_api):
    first = start(screening_api)
    sid = first["session_id"]
    assert screening_api.post("/api/screening/answer", json={"session_id": sid, "stop": True}).status_code == 400
    answer(screening_api, sid, 1, first["question"]["index"])
    r = answer(screening_api, sid, None, stop=True)
    assert r["finished"] and r["result"]["stop_reason"] == "user_stop" and r["result"]["asked"] == 1


def test_a_finished_session_stays_finished_and_is_listed(screening_api):
    first = start(screening_api)
    result, _ = run_to_the_end(screening_api, first)
    again = answer(screening_api, first["session_id"], 1)
    assert again["finished"] and again["result"]["p_hat"] == result["p_hat"]
    listed = ok(screening_api.get("/api/screening/sessions"))
    assert listed[0]["id"] == first["session_id"] and listed[0]["status"] == "completed"
    detail = ok(screening_api.get(f"/api/screening/sessions/{first['session_id']}"))
    assert detail["status"] == "completed" and detail["result"]["asked"] == 6


def test_invalid_answers_are_rejected(screening_api):
    first = start(screening_api)
    sid, idx = first["session_id"], first["question"]["index"]
    for bad in (2, -1, "1", None, True, 1.0):
        r = screening_api.post("/api/screening/answer", json={"session_id": sid, "value": bad, "item_index": idx})
        assert r.status_code == 400, bad
    r = screening_api.post("/api/screening/answer", json={"session_id": "ses_missing", "value": 1})
    assert r.status_code in (400, 404)
    assert screening_api.post("/api/screening/start", json={"mode": "nope"}).status_code == 400
    assert screening_api.post("/api/screening/start", json={"policy": "dqn"}).status_code == 400


def test_resuming_an_active_session_returns_the_pending_question(screening_api):
    first = start(screening_api)
    detail = ok(screening_api.get(f"/api/screening/sessions/{first['session_id']}"))
    assert detail["status"] == "active" and detail["pending_question"]["index"] == first["question"]["index"]


def test_an_auto_replay_that_stops_immediately_still_produces_a_valid_result_and_report(screening_api, monkeypatch):
    """Auto mode replays a policy, and a policy may STOP before asking anything. That must not crash report assembly."""
    from app.services import screening

    class StopImmediately:
        def __call__(self, state, legal):
            return STOP

    monkeypatch.setattr(screening, "make_policy", lambda name: StopImmediately())
    res = start(screening_api, mode="auto", policy="random")["result"]
    assert res["asked"] == 0 and res["items_asked"] == [] and res["stop_reason"] == "policy_stop"
    assert res["support_report"]["screening_result"]["decision"] in ("REFERRAL_RECOMMENDED", "NO_REFERRAL_INDICATED")
