"""Timeline events (audit findings C5 and T2).

The timeline records *that something happened* to a stored record, not a copy of what it said.
Events for observations, support strategies and strategy attempts therefore keep only a reference
(``ref_type``/``ref_id``); the text shown for them (summary, notes, context) is read from the live
source row when the timeline is listed or exported. Consequences, all of which the old
copy-the-text design got wrong:

* deleting an observation/strategy removes its text from the timeline and from exports;
* editing one is reflected immediately (no stale copy of the old wording);
* the 8 hand-written ``INSERT INTO timeline_events`` statements are one helper.

Events that have no source row (``observation_delete``, ``privacy`` ...) carry a constant summary
and never any user text.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from ..db import jdump, new_id, rows_to_dicts, utc_now

#: ``ref_type`` values whose display text is resolved from the source row.
RESOLVED_REFS = ("observation", "support_strategy", "strategy_attempt")


def record_event(conn: sqlite3.Connection, kind: str, title: str, *, ref_type: Optional[str] = None,
                 ref_id: Optional[str] = None, event_time: Optional[str] = None,
                 summary: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> str:
    """Append one event inside the caller's transaction and return its id.

    ``summary`` is for constant, user-text-free descriptions only. Never pass what a person typed:
    for referenced records that text is resolved from the source row at read time.
    """
    now = utc_now()
    eid = new_id("evt_")
    conn.execute(
        "INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) "
        "VALUES(?,?,?,?,?,?,NULL,?,?,NULL,?)",
        (eid, now, event_time or now, kind, title, summary, ref_type, ref_id, jdump(payload)))
    return eid


_RESOLVED = """
SELECT e.id, e.created_at, e.event_time, e.kind, e.title, e.ref_type, e.ref_id, e.payload_json,
       COALESCE(o.context, s.domain, s2.domain, e.context)                  AS context,
       COALESCE(substr(o.what_happened, 1, 180), s.title, s2.title, e.summary) AS summary,
       COALESCE(o.notes, s.notes, a.notes, e.notes)                         AS notes
FROM timeline_events e
LEFT JOIN observations      o  ON e.ref_type = 'observation'      AND o.id = e.ref_id
LEFT JOIN support_strategies s  ON e.ref_type = 'support_strategy' AND s.id = e.ref_id
LEFT JOIN strategy_attempts  a  ON e.ref_type = 'strategy_attempt' AND a.id = e.ref_id
LEFT JOIN support_strategies s2 ON s2.id = a.strategy_id
"""


def list_events(conn: sqlite3.Connection, *, q: str = "", kind: str = "", limit: Optional[int] = 500) -> List[Dict[str, Any]]:
    """Events newest first, with kind/search filters applied *before* the limit."""
    sql = (f"WITH ev AS ({_RESOLVED}) SELECT * FROM ev "
           "WHERE (:kind = '' OR kind = :kind) "
           "AND (:q = '' OR instr(lower_u(coalesce(title,'') || ' ' || coalesce(summary,'') || ' ' || "
           "coalesce(notes,'') || ' ' || coalesce(context,'')), :q) > 0) "
           "ORDER BY event_time DESC, created_at DESC")
    params: Dict[str, Any] = {"kind": kind, "q": q.lower()}
    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    return rows_to_dicts(conn.execute(sql, params).fetchall())
