"""Stale and duplicate screening answers (audit finding I2).

The answer endpoint used to take only a value, so a double-click answered the *next* question with
the previous answer, and two overlapping requests could overwrite each other (last write wins).
Needs the gitignored Saudi dataset (skipped without it).
"""
from __future__ import annotations

import threading

import pytest

from tests.test_app_screening import answer, ok, start


def raw_answer(api, sid, value, item_index):
    return api.post("/api/screening/answer", json={"session_id": sid, "value": value, "item_index": item_index})


def test_a_duplicate_answer_is_rejected_and_not_applied_to_the_next_question(screening_api):
    first = start(screening_api)
    sid, q1 = first["session_id"], first["question"]["index"]
    second = answer(screening_api, sid, 1, q1)
    q2 = second["next"]["index"]
    assert q2 != q1

    dup = raw_answer(screening_api, sid, 1, q1)               # the double-click
    assert dup.status_code == 409, dup.get_json()
    state = ok(screening_api.get(f"/api/screening/sessions/{sid}"))
    assert state["asked_count"] == 1 and len(state["trace"]) == 1, "the duplicate must not have recorded anything"
    assert state["pending_question"]["index"] == q2, "the pending question is unchanged"


def test_the_item_index_is_required_and_must_match(screening_api):
    first = start(screening_api)
    sid, q1 = first["session_id"], first["question"]["index"]
    assert raw_answer(screening_api, sid, 1, None).status_code == 400
    assert raw_answer(screening_api, sid, 1, (q1 + 1) % 10).status_code == 409
    assert raw_answer(screening_api, sid, 1, -1).status_code == 400
    assert raw_answer(screening_api, sid, 1, q1).status_code == 200


def test_a_conflicting_write_is_refused_not_overwritten(screening_api):
    """Two requests load the same session version; only the first may persist."""
    from app.services import screening
    first = start(screening_api)
    sid = first["session_id"]
    stale = screening._load_session(sid)
    fresh = screening._load_session(sid)
    fresh["asked_count"] = 1
    screening._persist_session(fresh)
    stale["asked_count"] = 99
    import pytest
    from app.errors import Conflict
    with pytest.raises(Conflict):
        screening._persist_session(stale)
    assert ok(screening_api.get(f"/api/screening/sessions/{sid}"))["asked_count"] == 1


def test_parallel_identical_answers_record_exactly_one(screening_api):
    first = start(screening_api)
    sid, q1 = first["session_id"], first["question"]["index"]
    codes, barrier = [], threading.Barrier(4)

    def fire():
        from app import db
        client = screening_api.application.test_client()
        barrier.wait()
        try:
            codes.append(raw_answer(client, sid, 1, q1).status_code)
        finally:
            db.close_connection()

    threads = [threading.Thread(target=fire) for _ in range(4)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert sorted(codes) == [200, 409, 409, 409], codes
    final = ok(screening_api.get(f"/api/screening/sessions/{sid}"))
    assert final["asked_count"] == 1 and [t["item"] for t in final["trace"]] == [f"A{q1 + 1}"]


# ----------------------------------------------------------------- abandoned-session expiry (T12)


def _insert_session(status, updated_at, sid):
    from app.db import tx
    with tx() as conn:
        conn.execute(
            "INSERT INTO screening_sessions(id,created_at,updated_at,status,policy,mode,budget,tau,asked_count) "
            "VALUES (?,?,?,?,?,?,?,?,?)", (sid, updated_at, updated_at, status, "greedy", "interactive", 6, 0.5, 0))


def _status(sid):
    from app.db import get_conn
    return get_conn().execute("SELECT status, version FROM screening_sessions WHERE id=?", (sid,)).fetchone()


def test_only_stale_active_sessions_expire_and_nothing_is_deleted(api):
    from datetime import datetime, timedelta, timezone
    from app.services import screening
    now = datetime(2030, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    iso = lambda h: (now - timedelta(hours=h)).replace(microsecond=0).isoformat()
    _insert_session("active", iso(30), "ses_old")
    _insert_session("active", iso(1), "ses_recent")
    _insert_session("active", iso(24), "ses_boundary")
    _insert_session("completed", iso(500), "ses_done")
    assert screening.expire_stale_sessions(now=now, ttl_hours=24) == 1
    assert _status("ses_old")["status"] == "expired"
    assert _status("ses_recent")["status"] == "active"
    assert _status("ses_boundary")["status"] == "active", "exactly at the TTL is not yet stale"
    assert _status("ses_done")["status"] == "completed", "a completed session is never touched"
    assert screening.expire_stale_sessions(now=now, ttl_hours=24) == 0, "idempotent"


def test_expiring_bumps_the_version_so_an_in_flight_write_cannot_resurrect_it(api):
    from datetime import datetime, timezone
    from app.services import screening
    _insert_session("active", "2000-01-01T00:00:00+00:00", "ses_x")
    before = _status("ses_x")["version"]
    screening.expire_stale_sessions(now=datetime(2030, 1, 1, tzinfo=timezone.utc), ttl_hours=1)
    assert _status("ses_x")["version"] == before + 1


def _age(sid):
    from app.db import tx
    with tx() as conn:
        conn.execute("UPDATE screening_sessions SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (sid,))


def test_listing_expires_and_the_api_refuses_to_answer_an_expired_session(screening_api):
    first = start(screening_api)
    sid, q = first["session_id"], first["question"]["index"]
    _age(sid)
    listed = ok(screening_api.get("/api/screening/sessions"))
    assert {s["id"]: s["status"] for s in listed}[sid] == "expired"
    r = screening_api.post("/api/screening/answer", json={"session_id": sid, "value": 1, "item_index": q})
    assert r.status_code == 409 and "expired" in r.get_json()["error"]
    got = ok(screening_api.get(f"/api/screening/sessions/{sid}"))
    assert got["status"] == "expired" and got["pending_question"] is None


def test_answering_a_stale_session_directly_expires_it_without_a_list_call(screening_api):
    first = start(screening_api)
    sid, q = first["session_id"], first["question"]["index"]
    _age(sid)
    r = screening_api.post("/api/screening/answer", json={"session_id": sid, "value": 1, "item_index": q})
    assert r.status_code == 409 and _status(sid)["status"] == "expired"
    assert ok(screening_api.get(f"/api/screening/sessions/{sid}"))["asked_count"] == 0, "nothing was recorded"


def test_an_expired_session_is_still_covered_by_export_and_delete(screening_api):
    sid = start(screening_api)["session_id"]
    _age(sid)
    screening_api.get("/api/screening/sessions")
    exported = ok(screening_api.post("/api/timeline/export", json={}))
    assert any(s["id"] == sid for s in exported["screening_sessions"])
    deleted = ok(screening_api.post("/api/privacy/delete_sessions", json={}))
    assert deleted["counts"]["screening_sessions"] >= 1


@pytest.mark.parametrize("raw, ok", [(None, 24.0), ("", 24.0), ("6", 6.0), ("0.5", 0.5)])
def test_ttl_setting_accepts_sensible_values(raw, ok):
    from app.config import _session_ttl_hours
    assert _session_ttl_hours(raw) == ok


@pytest.mark.parametrize("raw", ["0", "-3", "abc", "nan", "99999"])
def test_ttl_setting_rejects_nonsense(raw):
    from app.config import _session_ttl_hours
    with pytest.raises(ValueError):
        _session_ttl_hours(raw)
