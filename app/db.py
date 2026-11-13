"""SQLite access for the NeuroBridge app (local, single-user, but served by a threaded Flask server).

Design (audit finding I6):

* **One connection per thread.** The previous design shared a single connection across all Flask
  threads and only locked writers, so a reader could see another thread's uncommitted rows (which
  then vanished on rollback). Connections are now thread-local, in WAL mode, so readers always see
  the last committed state and never block writers.
* **Explicit transactions.** Connections run in autocommit; :func:`tx` opens ``BEGIN IMMEDIATE`` so
  a writer takes the write lock up front (no mid-transaction lock upgrade failures) and nested
  ``tx()`` calls join the outer transaction.
* **Versioned migrations** (``PRAGMA user_version``). ``CREATE TABLE IF NOT EXISTS`` alone could never
  change an existing database. Each migration runs in its own transaction together with its version
  bump, and is re-checked under the write lock, so concurrent processes cannot apply it twice.
* **One timestamp format**: UTC, ``YYYY-MM-DDTHH:MM:SS+00:00``. It sorts correctly as a string.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import DB_PATH

log = logging.getLogger("neurobridge.db")

#: Serialises writers inside this process and guards the migration bookkeeping.
_LOCK = threading.RLock()
_local = threading.local()
_MIGRATED: set = set()


# --------------------------------------------------------------------------- time


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_utc_iso(value: Optional[str]) -> Optional[str]:
    """Normalise a client-supplied timestamp to the stored UTC format.

    A timestamp with an offset (or ``Z``) is converted. A naive one - what ``<input
    type="datetime-local">`` produces - is read as local time of this machine: the app and its
    browser run on the same host, so that is the user's wall-clock time. ``None``/blank -> ``None``;
    anything unparseable raises ``ValueError``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[-1] in "Zz":  # fromisoformat only accepts a trailing Z from Python 3.11
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"not an ISO-8601 timestamp: {text[:40]!r}") from exc
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- connections


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # SQLite's built-in lower() is ASCII-only; searches must be case-insensitive for any script.
    conn.create_function("lower_u", 1, lambda v: v.lower() if isinstance(v, str) else v, deterministic=True)
    return conn


def get_conn() -> sqlite3.Connection:
    """This thread's connection to the current ``DB_PATH`` (migrated on first use of that path)."""
    path = Path(DB_PATH)
    conn = getattr(_local, "conn", None)
    if conn is not None and _local.path == path:
        return conn
    if conn is not None:
        conn.close()
    conn = _connect(path)
    _local.conn, _local.path, _local.depth = conn, path, 0
    _ensure_schema(conn, path)
    return conn


def close_connection() -> None:
    """Close this thread's connection (tests, and clean shutdown)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
    _local.conn, _local.path, _local.depth = None, None, 0


@contextmanager
def tx():
    """Write transaction: ``BEGIN IMMEDIATE`` ... ``COMMIT``, rolled back on any exception.

    Re-entrant: a ``tx()`` opened inside another one joins it (the outer block commits or rolls back).
    """
    conn = get_conn()
    if _local.depth > 0:
        _local.depth += 1
        try:
            yield conn
        finally:
            _local.depth -= 1
        return
    with _LOCK:
        conn.execute("BEGIN IMMEDIATE")
        _local.depth = 1
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover - connection already unusable
                pass
            raise
        finally:
            _local.depth = 0


@contextmanager
def snapshot():
    """Read-only transaction: every query inside sees one consistent committed state."""
    conn = get_conn()
    if _local.depth > 0:
        yield conn
        return
    conn.execute("BEGIN")
    _local.depth = 1
    try:
        yield conn
    finally:
        try:
            conn.execute("COMMIT")
        finally:
            _local.depth = 0


# --------------------------------------------------------------------------- schema + migrations

_BASELINE = [
    """CREATE TABLE IF NOT EXISTS screening_sessions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL,
        policy TEXT NOT NULL,
        mode TEXT NOT NULL,
        budget INTEGER NOT NULL,
        tau REAL NOT NULL,
        record_index INTEGER,
        true_label INTEGER,
        asked_count INTEGER NOT NULL DEFAULT 0,
        state_json TEXT,
        pending_item INTEGER,
        items_asked_json TEXT,
        trace_json TEXT,
        result_json TEXT,
        explanation_json TEXT,
        support_report_json TEXT,
        stop_reason TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        context TEXT NOT NULL,
        what_happened TEXT NOT NULL,
        frequency TEXT,
        difficulty TEXT,
        what_helped TEXT,
        consistency TEXT,
        insufficient_info INTEGER NOT NULL DEFAULT 0,
        notes TEXT,
        session_id TEXT,
        occurred_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS support_strategies (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        title TEXT NOT NULL,
        domain TEXT,
        description TEXT,
        triggered_observation_id TEXT,
        why_useful TEXT,
        how_to_try TEXT,
        how_to_record TEXT,
        status TEXT NOT NULL DEFAULT 'saved',
        outcome TEXT,
        notes TEXT,
        source TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS strategy_attempts (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        outcome TEXT,
        notes TEXT,
        FOREIGN KEY(strategy_id) REFERENCES support_strategies(id) ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS timeline_events (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        event_time TEXT NOT NULL,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT,
        context TEXT,
        ref_type TEXT,
        ref_id TEXT,
        notes TEXT,
        payload_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS consent_records (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        scope TEXT NOT NULL,
        granted INTEGER NOT NULL,
        details TEXT
    )""",
    # Indexes for the list/sort/filter queries the API actually runs.
    "CREATE INDEX IF NOT EXISTS idx_obs_occurred ON observations(occurred_at, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_obs_context ON observations(context)",
    "CREATE INDEX IF NOT EXISTS idx_strat_updated ON support_strategies(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_attempt_strategy ON strategy_attempts(strategy_id)",
    "CREATE INDEX IF NOT EXISTS idx_tl_time ON timeline_events(event_time, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tl_ref ON timeline_events(ref_type, ref_id)",
    "CREATE INDEX IF NOT EXISTS idx_sess_updated ON screening_sessions(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_consent_scope ON consent_records(scope, updated_at)",
]


def _m1_baseline(conn: sqlite3.Connection) -> None:
    """Baseline schema + indexes. Idempotent, so it is also correct for databases created before
    versioning existed (they have ``user_version`` 0 and already hold these tables)."""
    for stmt in _BASELINE:
        conn.execute(stmt)


def _m2_session_version(conn: sqlite3.Connection) -> None:
    """Optimistic-concurrency counter for screening sessions (audit finding I2)."""
    conn.execute("ALTER TABLE screening_sessions ADD COLUMN version INTEGER NOT NULL DEFAULT 0")


_OPTIONAL_OBS = ("frequency", "difficulty", "what_helped", "consistency", "notes", "session_id", "occurred_at")
_OPTIONAL_STRAT = ("domain", "description", "triggered_observation_id", "why_useful", "how_to_try",
                   "how_to_record", "outcome", "notes", "source")
_COPIED_TEXT_KINDS = ("observation", "observation_update", "support", "support_update", "support_attempt")


def _m3_privacy_and_consistency(conn: sqlite3.Connection) -> None:
    """Data repair for audit findings C5, I6 and I7. Touches no live observation or strategy content.

    * C5: timeline rows used to *copy* observation/strategy free text, so deleting or editing the
      source left the old text readable on the Timeline and in exports. The copies are blanked; the
      Timeline now reads the live text from the source row at query time.
    * Attempt events are re-pointed at the attempt row (so its notes can be read the same way).
    * I7: empty strings stored for optional fields become NULL.
    * I6: timestamps are normalised to UTC; references to rows that no longer exist are cleared.
    """
    marks = ",".join("?" * len(_COPIED_TEXT_KINDS))
    conn.execute(f"UPDATE timeline_events SET summary=NULL, notes=NULL, context=NULL "
                 f"WHERE kind IN ({marks})", _COPIED_TEXT_KINDS)
    conn.execute("UPDATE timeline_events SET payload_json=NULL "
                 "WHERE kind IN ('observation','observation_update','support','support_update')")
    for row in conn.execute("SELECT id, payload_json FROM timeline_events WHERE kind='support_attempt'").fetchall():
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except ValueError:
            payload = {}
        attempt_id = payload.get("attempt")
        conn.execute("UPDATE timeline_events SET ref_type='strategy_attempt', ref_id=COALESCE(?, ref_id), payload_json=? "
                     "WHERE id=?", (attempt_id, json.dumps({"attempt": attempt_id, "status": payload.get("status")}), row["id"]))

    for col in _OPTIONAL_OBS:
        conn.execute(f"UPDATE observations SET {col}=NULL WHERE trim({col})=''")
    for col in _OPTIONAL_STRAT:
        conn.execute(f"UPDATE support_strategies SET {col}=NULL WHERE trim({col})=''")

    def renormalise(table: str, column: str) -> None:
        for r in conn.execute(f"SELECT id, {column} AS v FROM {table} WHERE {column} IS NOT NULL").fetchall():
            try:
                fixed = to_utc_iso(r["v"])
            except ValueError:
                continue
            if fixed and fixed != r["v"]:
                conn.execute(f"UPDATE {table} SET {column}=? WHERE id=?", (fixed, r["id"]))

    renormalise("observations", "occurred_at")
    renormalise("timeline_events", "event_time")

    conn.execute("UPDATE observations SET session_id=NULL WHERE session_id IS NOT NULL "
                 "AND session_id NOT IN (SELECT id FROM screening_sessions)")
    conn.execute("UPDATE support_strategies SET triggered_observation_id=NULL WHERE triggered_observation_id IS NOT NULL "
                 "AND triggered_observation_id NOT IN (SELECT id FROM observations)")


#: (version, function). Append only; never edit a released migration.
MIGRATIONS: List[tuple] = [
    (1, _m1_baseline),
    (2, _m2_session_version),
    (3, _m3_privacy_and_consistency),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    key = str(path.resolve())
    if key in _MIGRATED:  # fast path without the writer lock: readers on new threads must not wait for writers
        return
    with _LOCK:
        if key in _MIGRATED:
            return
        for version, apply in MIGRATIONS:
            if conn.execute("PRAGMA user_version").fetchone()[0] >= version:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Re-check under the write lock: another process may have just applied it.
                if conn.execute("PRAGMA user_version").fetchone()[0] < version:
                    apply(conn)
                    conn.execute(f"PRAGMA user_version = {int(version)}")
                    log.info("database migrated to schema version %d", version)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        _MIGRATED.add(key)


def init_db() -> None:
    """Create/upgrade the schema now (it also happens lazily on first connection)."""
    get_conn()


# --------------------------------------------------------------------------- small helpers


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex[:12]


def jdump(v: Any) -> Optional[str]:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False)


def jload(v: Optional[str], default: Any = None) -> Any:
    if v is None:
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]
