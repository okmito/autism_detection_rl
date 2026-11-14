"""Data-layer tests (audit findings I6 and I7): connections, transactions, migrations, timestamps.

Everything runs against throw-away SQLite files; the real ``data/neurobridge.db`` is never opened.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from app import db


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "layer.db")
    db.close_connection()
    yield tmp_path / "layer.db"
    db.close_connection()


# ------------------------------------------------------------------ schema and migrations


def test_a_new_database_is_created_at_the_current_schema_version(fresh_db):
    conn = db.get_conn()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"screening_sessions", "observations", "support_strategies", "strategy_attempts",
            "timeline_events", "consent_records"} <= tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(screening_sessions)")}
    assert "version" in cols, "optimistic-concurrency column (I2)"
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_obs_occurred", "idx_tl_time", "idx_tl_ref", "idx_sess_updated"} <= indexes


def test_connection_pragmas(fresh_db):
    conn = db.get_conn()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("SELECT lower_u('ÉCOLE')").fetchone()[0] == "école"


def test_migrating_twice_changes_nothing(fresh_db):
    db.get_conn().execute("INSERT INTO consent_records VALUES('c','t','t','general_research_use',1,NULL)")
    db.close_connection()
    db._MIGRATED.discard(str(fresh_db.resolve()))
    conn = db.get_conn()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM consent_records").fetchone()[0] == 1


def _legacy_database(path):
    """A database as the app created it before versioning: old schema, ``user_version`` 0, and the
    problems the audit found (copied free text, empty strings, naive timestamps, dangling ids)."""
    conn = sqlite3.connect(str(path))
    for stmt in db._BASELINE[:6]:  # the six original tables, no indexes, no version column
        conn.execute(stmt)
    conn.execute("INSERT INTO observations(id,created_at,updated_at,context,what_happened,frequency,difficulty,notes,session_id,occurred_at)"
                 " VALUES('obs_live','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00','Daily routines','live text',"
                 "'', '  ', 'live note', 'ses_gone', '2026-03-01T10:30')")
    conn.execute("INSERT INTO support_strategies(id,created_at,updated_at,title,domain,triggered_observation_id,notes)"
                 " VALUES('sup_live','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00','Live title','','obs_deleted','strategy note')")
    conn.execute("INSERT INTO strategy_attempts VALUES('att_1','sup_live','2026-01-01T00:00:00+00:00','tried','helped','attempt note')")
    events = [
        ("e1", "observation", "obs_live", "observation", "LEAKED-OBS-TEXT", "LEAKED-NOTE", "Daily routines", '{"frequency":"often"}'),
        ("e2", "observation", "obs_deleted_long_ago", "observation", "LEAKED-DELETED-TEXT", "LEAKED-DELETED-NOTE", "Daily routines", None),
        ("e3", "support", "sup_live", "support", "Live title", "x", "Routines", '{"status":"saved"}'),
        ("e4", "support_attempt", "sup_live", "support_attempt", "Live title", "LEAKED-ATTEMPT", None,
         '{"attempt":"att_1","status":"tried","outcome":"LEAKED-OUTCOME"}'),
        ("e5", "observation_delete", "obs_deleted_long_ago", "observation", "An observation record was deleted.", None, None, None),
    ]
    for eid, kind, ref, ref_type, summary, notes, context, payload in events:
        conn.execute("INSERT INTO timeline_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (eid, "2026-01-01T00:00:00+00:00", "2026-03-01T10:30", kind, "title", summary, context, ref_type, ref, notes, payload))
    conn.commit()
    conn.close()


def test_a_legacy_database_is_repaired_without_touching_live_content(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    db.close_connection()
    try:
        conn = db.get_conn()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION

        # C5: the copied text is gone from every event that copied it ...
        blob = json.dumps([dict(r) for r in conn.execute("SELECT * FROM timeline_events")])
        for leaked in ("LEAKED-OBS-TEXT", "LEAKED-NOTE", "LEAKED-DELETED-TEXT", "LEAKED-DELETED-NOTE",
                       "LEAKED-ATTEMPT", "LEAKED-OUTCOME"):
            assert leaked not in blob, leaked
        # ... constant descriptions of deletions are kept ...
        assert conn.execute("SELECT summary FROM timeline_events WHERE id='e5'").fetchone()[0] == "An observation record was deleted."
        # ... and the attempt event now points at the attempt row, so its notes resolve at read time.
        e4 = conn.execute("SELECT ref_type, ref_id FROM timeline_events WHERE id='e4'").fetchone()
        assert (e4["ref_type"], e4["ref_id"]) == ("strategy_attempt", "att_1")

        # live content is untouched
        obs = conn.execute("SELECT * FROM observations WHERE id='obs_live'").fetchone()
        assert obs["what_happened"] == "live text" and obs["notes"] == "live note"
        assert conn.execute("SELECT notes FROM strategy_attempts").fetchone()[0] == "attempt note"
        assert conn.execute("SELECT title, notes FROM support_strategies").fetchone()[:] == ("Live title", "strategy note")

        # I7: empty/blank strings became NULL
        assert obs["frequency"] is None and obs["difficulty"] is None
        assert conn.execute("SELECT domain FROM support_strategies").fetchone()[0] is None
        # I6: naive timestamps are now UTC with an offset, references to missing rows are cleared
        expected = datetime(2026, 3, 1, 10, 30).astimezone(timezone.utc).replace(microsecond=0).isoformat()
        assert obs["occurred_at"] == expected
        assert conn.execute("SELECT event_time FROM timeline_events WHERE id='e1'").fetchone()[0] == expected
        assert obs["session_id"] is None
        assert conn.execute("SELECT triggered_observation_id FROM support_strategies").fetchone()[0] is None
    finally:
        db.close_connection()


# ------------------------------------------------------------------ transactions and connections


def test_tx_commits_and_rolls_back(fresh_db):
    with db.tx() as conn:
        conn.execute("INSERT INTO consent_records VALUES('a','t','t','general_research_use',1,NULL)")
    assert db.get_conn().execute("SELECT COUNT(*) FROM consent_records").fetchone()[0] == 1
    with pytest.raises(RuntimeError):
        with db.tx() as conn:
            conn.execute("INSERT INTO consent_records VALUES('b','t','t','general_research_use',1,NULL)")
            raise RuntimeError("boom")
    assert db.get_conn().execute("SELECT COUNT(*) FROM consent_records").fetchone()[0] == 1


def test_a_nested_tx_joins_the_outer_one(fresh_db):
    with pytest.raises(RuntimeError):
        with db.tx() as outer:
            outer.execute("INSERT INTO consent_records VALUES('a','t','t','general_research_use',1,NULL)")
            with db.tx() as inner:
                inner.execute("INSERT INTO consent_records VALUES('b','t','t','general_research_use',1,NULL)")
            raise RuntimeError("outer fails after inner finished")
    assert db.get_conn().execute("SELECT COUNT(*) FROM consent_records").fetchone()[0] == 0, \
        "the inner block must not have committed on its own"
    with db.tx():
        with db.tx() as inner:
            inner.execute("INSERT INTO consent_records VALUES('c','t','t','general_research_use',1,NULL)")
    assert db.get_conn().execute("SELECT COUNT(*) FROM consent_records").fetchone()[0] == 1


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            out["error"] = exc
        finally:
            db.close_connection()

    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    assert not t.is_alive() and "error" not in out, out.get("error")
    return out["value"]


def test_a_reader_never_sees_another_threads_uncommitted_rows(fresh_db):
    """I6: with the old single shared connection this read returned the uncommitted row."""
    count = "SELECT COUNT(*) FROM consent_records"
    with pytest.raises(RuntimeError):
        with db.tx() as conn:
            conn.execute("INSERT INTO consent_records VALUES('dirty','t','t','general_research_use',1,NULL)")
            assert conn.execute(count).fetchone()[0] == 1, "the writer sees its own row"
            assert _in_thread(lambda: db.get_conn().execute(count).fetchone()[0]) == 0, "a reader must not"
            raise RuntimeError("roll back")
    assert _in_thread(lambda: db.get_conn().execute(count).fetchone()[0]) == 0


def test_each_thread_has_its_own_connection(fresh_db):
    main = db.get_conn()
    other = _in_thread(lambda: id(db.get_conn()))
    assert other != id(main)
    assert db.get_conn() is main


def test_writers_in_different_threads_do_not_lose_rows(fresh_db):
    db.get_conn()
    errors = []

    def write(i):
        try:
            for j in range(10):
                with db.tx() as conn:
                    conn.execute("INSERT INTO consent_records VALUES(?,?,?,?,?,NULL)",
                                 (f"{i}-{j}", "t", "t", "general_research_use", 1))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            db.close_connection()

    threads = [threading.Thread(target=write, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert not errors, errors
    assert db.get_conn().execute("SELECT COUNT(*) FROM consent_records").fetchone()[0] == 60


# ------------------------------------------------------------------ timestamps (I6)


@pytest.mark.parametrize("raw, expected", [
    ("2026-03-01T10:00:00+00:00", "2026-03-01T10:00:00+00:00"),
    ("2026-03-01T10:00:00Z", "2026-03-01T10:00:00+00:00"),
    ("2026-03-01T10:00:00+05:30", "2026-03-01T04:30:00+00:00"),
    ("2026-03-01T10:00:00-05:00", "2026-03-01T15:00:00+00:00"),
    ("2026-03-01T10:00:00.987654+00:00", "2026-03-01T10:00:00+00:00"),
])
def test_offset_timestamps_are_normalised_to_utc(raw, expected):
    assert db.to_utc_iso(raw) == expected


def test_a_naive_timestamp_is_read_as_local_time():
    got = db.to_utc_iso("2026-03-01T10:30")
    assert got == datetime(2026, 3, 1, 10, 30).astimezone(timezone.utc).isoformat()


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_timestamps_are_none(raw):
    assert db.to_utc_iso(raw) is None


@pytest.mark.parametrize("raw", ["yesterday", "2026-13-45", "not a date at all"])
def test_garbage_timestamps_are_rejected(raw):
    with pytest.raises(ValueError):
        db.to_utc_iso(raw)


def test_observations_sort_by_the_real_instant_not_the_string(api):
    """I6: 09:00-05:00 is 14:00Z, later than 10:00Z, but sorts *earlier* as a raw string."""
    def create(when, text):
        r = api.post("/api/observations", json={"context": "Daily routines", "what_happened": text, "occurred_at": when})
        assert r.status_code == 200, r.get_json()

    create("2026-01-01T10:00:00+00:00", "earlier")
    create("2026-01-01T09:00:00-05:00", "later")
    texts = [o["what_happened"] for o in api.get("/api/observations").get_json()["data"]]
    assert texts == ["later", "earlier"]
    events = api.get("/api/timeline?kind=observation").get_json()["data"]
    assert [e["summary"] for e in events] == ["later", "earlier"]


def test_a_bad_timestamp_is_a_400_not_a_500(api):
    r = api.post("/api/observations", json={"context": "Daily routines", "what_happened": "x", "occurred_at": "soon"})
    assert r.status_code == 400 and "occurred_at" in r.get_json()["error"]


# ------------------------------------------------------------------ I7: blanks


def test_blank_optional_fields_are_stored_as_null_not_empty_strings(api):
    """I7: the edit form sends ``occurred_at: ""`` when the date box is empty; that must not be stored."""
    oid = api.post("/api/observations", json={
        "context": "Daily routines", "what_happened": "x", "occurred_at": "2026-02-01T08:00:00+00:00",
        "frequency": "", "notes": "   "}).get_json()["data"]["id"]
    row = api.get("/api/observations").get_json()["data"][0]
    assert row["frequency"] is None and row["notes"] is None

    r = api.put(f"/api/observations/{oid}", json={"occurred_at": "", "difficulty": "", "what_happened": "edited"})
    assert r.status_code == 200
    row = api.get("/api/observations").get_json()["data"][0]
    assert row["occurred_at"] is None and row["difficulty"] is None and row["what_happened"] == "edited"
    # a NULL date falls back to created_at for ordering/display instead of sorting as ""
    assert api.get("/api/observations").get_json()["data"][0]["id"] == oid
