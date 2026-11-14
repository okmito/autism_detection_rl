"""Privacy and timeline behaviour (audit findings C5, P5, T2, I9).

C5: timeline rows used to copy observation/strategy free text, so "delete" left the text readable on
the Timeline and in the export, sessions could not be deleted, and the export omitted tables.
These tests plant recognisable markers in every free-text field and assert they are gone from every
read surface after deletion.
"""
from __future__ import annotations

import json

import pytest

M = {"obs": "MARK-OBS-TEXT", "note": "MARK-OBS-NOTE", "ctx": "MARK-CONTEXT", "strat": "MARK-STRATEGY-TITLE",
     "snote": "MARK-STRATEGY-NOTE", "anote": "MARK-ATTEMPT-NOTE", "aout": "MARK-ATTEMPT-OUTCOME"}


def data(r):
    assert r.status_code == 200, r.get_json()
    return r.get_json()["data"]


def everything(api) -> str:
    """Every read surface a person can reach, as one string."""
    parts = [api.get(u).get_data(as_text=True) for u in
             ("/api/timeline", "/api/observations", "/api/support/strategies", "/api/consent",
              "/api/screening/sessions")]
    parts.append(api.post("/api/timeline/export", json={}).get_data(as_text=True))
    return "\n".join(parts)


@pytest.fixture
def populated(api):
    oid = data(api.post("/api/observations", json={
        "context": M["ctx"], "what_happened": M["obs"], "notes": M["note"]}))["id"]
    sid = data(api.post("/api/support/strategies", json={
        "title": M["strat"], "notes": M["snote"], "triggered_observation_id": oid}))["id"]
    data(api.post(f"/api/support/strategies/{sid}/attempts", json={
        "status": "tried", "notes": M["anote"], "outcome": M["aout"]}))
    data(api.put(f"/api/observations/{oid}", json={"notes": M["note"] + "-edited"}))
    return {"api": api, "oid": oid, "sid": sid}


def test_the_marker_text_is_visible_while_the_records_exist(populated):
    text = everything(populated["api"])
    for k in ("obs", "ctx", "strat"):
        assert M[k] in text


def test_timeline_events_hold_references_not_copies(populated):
    from app import db
    rows = db.get_conn().execute("SELECT kind, summary, notes, context, payload_json FROM timeline_events").fetchall()
    blob = json.dumps([dict(r) for r in rows])
    for marker in M.values():
        assert marker not in blob, f"{marker} was copied into timeline_events"


def test_timeline_resolves_live_text_and_reflects_edits(populated):
    api = populated["api"]
    events = data(api.get("/api/timeline?kind=observation"))
    assert events[0]["summary"] == M["obs"] and events[0]["context"] == M["ctx"]
    data(api.put(f"/api/observations/{populated['oid']}", json={"what_happened": "rewritten text"}))
    assert data(api.get("/api/timeline?kind=observation"))[0]["summary"] == "rewritten text"
    assert M["obs"] not in everything(api), "the old wording must not survive an edit"


def test_attempt_events_resolve_strategy_title_and_attempt_notes(populated):
    ev = data(populated["api"].get("/api/timeline?kind=support_attempt"))[0]
    assert ev["summary"] == M["strat"] and ev["notes"] == M["anote"]


def test_deleting_one_observation_removes_its_text_everywhere(populated):
    api = populated["api"]
    data(api.delete(f"/api/observations/{populated['oid']}", json={}))
    text = everything(api)
    for k in ("obs", "note", "ctx"):
        assert M[k] not in text, M[k]
    assert M["strat"] in text, "unrelated records stay"
    kinds = [e["kind"] for e in data(api.get("/api/timeline"))]
    assert "observation_delete" in kinds
    strat = data(api.get("/api/support/strategies"))[0]
    assert strat["triggered_observation_id"] is None, "no dangling reference"


def test_deleting_a_strategy_removes_it_and_its_attempts_everywhere(populated):
    api = populated["api"]
    data(api.delete(f"/api/support/strategies/{populated['sid']}", json={}))
    text = everything(api)
    for k in ("strat", "snote", "anote", "aout"):
        assert M[k] not in text, M[k]
    assert M["obs"] in text


def test_delete_observations_endpoint_removes_text_from_timeline_and_export(populated):
    api = populated["api"]
    data(api.post("/api/privacy/delete_observations", json={}))
    text = everything(api)
    for k in ("obs", "note", "ctx"):
        assert M[k] not in text, M[k]
    assert data(api.get("/api/observations")) == []
    assert data(api.get("/api/support/strategies"))[0]["triggered_observation_id"] is None


def _make_session_row(api, marker):
    from app import db
    now = db.utc_now()
    with db.tx() as conn:
        conn.execute("INSERT INTO screening_sessions(id,created_at,updated_at,status,policy,mode,budget,tau,state_json,"
                     "result_json,version) VALUES('ses_marker',?,?,'finished','greedy','interactive',6,0.5,?,?,0)",
                     (now, now, json.dumps({"x": marker}), json.dumps({"summary": marker})))


def test_sessions_can_be_deleted_and_observation_links_are_cleared(api):
    _make_session_row(api, "MARK-SESSION")
    oid = data(api.post("/api/observations", json={"context": "c", "what_happened": "w", "session_id": "ses_marker"}))["id"]
    assert "MARK-SESSION" in everything(api)
    r = data(api.post("/api/privacy/delete_sessions", json={}))
    assert r["counts"]["screening_sessions"] == 1
    assert "MARK-SESSION" not in everything(api)
    assert data(api.get("/api/observations"))[0]["session_id"] is None
    assert data(api.get("/api/observations"))[0]["id"] == oid, "observations themselves are kept"


def test_delete_all_removes_everything_but_the_consent_log(populated):
    api = populated["api"]
    _make_session_row(api, "MARK-SESSION")
    data(api.post("/api/consent", json={"scope": "general_research_use", "granted": True}))
    counts = data(api.post("/api/privacy/delete_all", json={}))["counts"]
    assert counts["observations"] == 1 and counts["support_strategies"] == 1 and counts["screening_sessions"] == 1
    text = everything(api)
    for marker in list(M.values()) + ["MARK-SESSION"]:
        assert marker not in text, marker
    assert data(api.get("/api/support/strategies")) == []
    assert len(data(api.get("/api/consent"))) == 1, "the consent log is deliberately kept"
    from app import db
    assert db.get_conn().execute("SELECT COUNT(*) FROM strategy_attempts").fetchone()[0] == 0
    events = data(api.get("/api/timeline"))
    assert [e["kind"] for e in events] == ["privacy"], "only the content-free deletion record remains"
    assert "removed by user request" in events[0]["summary"]


def test_privacy_endpoints_are_csrf_guarded(api):
    for path in ("delete_observations", "delete_sessions", "delete_all"):
        assert api.post(f"/api/privacy/{path}", data="x", headers={"Content-Type": "text/plain"}).status_code == 415
        assert api.post(f"/api/privacy/{path}", json={}, headers={"Origin": "http://evil.example"}).status_code == 403


def test_export_covers_every_table(populated):
    api = populated["api"]
    _make_session_row(api, "MARK-SESSION")
    data(api.post("/api/consent", json={"scope": "store_observations", "granted": False}))
    export = data(api.post("/api/timeline/export", json={}))
    assert set(export) >= {"exported_at", "observations", "support_strategies", "strategy_attempts",
                           "screening_sessions", "consent_records", "timeline", "note"}
    assert len(export["strategy_attempts"]) == 1 and len(export["consent_records"]) == 1
    session = export["screening_sessions"][0]
    assert session["result"] == {"summary": "MARK-SESSION"}, "stored JSON columns are exported parsed, not as strings"
    assert "result_json" not in session
    assert len(export["timeline"]) >= 4, "export is not capped at the UI's 500 rows"


# ---------------------------------------------------------------- P5: filters before limit


def test_timeline_filters_apply_before_the_limit(api):
    """Old code did LIMIT 500 first and filtered in Python, so old matches silently vanished."""
    from app import db
    from app.services.timeline import record_event
    with db.tx() as conn:
        record_event(conn, "privacy", "old special event", event_time="2020-01-01T00:00:00+00:00", summary="needle")
        for i in range(30):
            record_event(conn, "observation_delete", "recent", event_time=f"2026-01-{i % 28 + 1:02d}T00:00:00+00:00",
                         summary="hay")
    assert len(data(api.get("/api/timeline?limit=10"))) == 10
    hits = data(api.get("/api/timeline?limit=10&q=needle"))
    assert [h["title"] for h in hits] == ["old special event"]
    assert len(data(api.get("/api/timeline?limit=5&kind=privacy"))) == 1


def test_timeline_search_matches_resolved_text_and_is_case_insensitive(populated):
    api = populated["api"]
    assert data(api.get("/api/timeline?q=mark-obs-text")), "matches text that lives on the observation row"
    assert data(api.get("/api/timeline?q=mark-attempt-note"))
    assert data(api.get("/api/timeline?q=no-such-thing")) == []


def test_record_event_requires_no_user_text_and_is_one_row(api):
    from app import db
    from app.services.timeline import record_event
    with db.tx() as conn:
        eid = record_event(conn, "privacy", "T", ref_type="privacy", summary="constant")
    row = db.get_conn().execute("SELECT * FROM timeline_events WHERE id=?", (eid,)).fetchone()
    assert (row["kind"], row["summary"], row["notes"], row["context"]) == ("privacy", "constant", None, None)


# ---------------------------------------------------------------- I9: consent is a log, not a gate


def test_consent_is_recorded_but_gates_nothing_and_the_ui_says_so(api):
    data(api.post("/api/consent", json={"scope": "store_observations", "granted": False}))
    assert api.post("/api/observations", json={"context": "c", "what_happened": "w"}).status_code == 200
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent / "app/static/js/app.js").read_text(encoding="utf-8")
    assert "does not switch any feature on or off" in js
