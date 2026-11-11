from __future__ import annotations
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import DB_PATH

_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_CONN: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        with _LOCK:
            if _CONN is None:
                _CONN = _connect()
    return _CONN


@contextmanager
def tx():
    conn = get_conn()
    with _LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db() -> None:
    schema = r'''
    CREATE TABLE IF NOT EXISTS screening_sessions (
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
    );

    CREATE TABLE IF NOT EXISTS observations (
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
    );

    CREATE TABLE IF NOT EXISTS support_strategies (
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
    );

    CREATE TABLE IF NOT EXISTS strategy_attempts (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        outcome TEXT,
        notes TEXT,
        FOREIGN KEY(strategy_id) REFERENCES support_strategies(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS timeline_events (
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
    );

    CREATE TABLE IF NOT EXISTS consent_records (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        scope TEXT NOT NULL,
        granted INTEGER NOT NULL,
        details TEXT
    );
    '''
    with tx() as conn:
        conn.executescript(schema)


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
