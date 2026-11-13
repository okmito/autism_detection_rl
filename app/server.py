from __future__ import annotations
import logging
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from app import validation as v
from app.config import POLICY_DEFAULT, STATIC_DIR, SECRET_KEY
from app.db import get_conn, init_db, jload, new_id, rows_to_dicts, snapshot, tx, utc_now
from app.errors import ApiError, BadRequest, NotFound
from app.services import screening, research, rl_lab
from app.services.support import suggest as suggest_for_observation
from app.services.timeline import list_events, record_event

log = logging.getLogger("neurobridge.server")

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["SECRET_KEY"] = SECRET_KEY
# No CORS: the SPA is served from this same origin. A wildcard CORS policy would let any
# website the user visits read observation/timeline data from this local server.

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


@app.before_request
def _guard():
    """Reject DNS-rebinding (foreign Host) and cross-site state changes (CSRF).

    - Host must be a loopback name, otherwise a rebinding page could talk to us as same-origin.
    - Mutating requests must be application/json (a cross-site <form> cannot send that without a
      CORS preflight, which this server never approves) and any Origin header must be our own.
    """
    host = (request.host or "").strip().lower()
    hostname = urlsplit("//" + host).hostname or ""
    if hostname not in _LOOPBACK_HOSTS:
        log.warning("rejected request with non-loopback Host header")
        return err("forbidden host", 403)
    if request.method in _MUTATING:
        if not request.is_json:
            return err("Content-Type must be application/json", 415)
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() != host:
            log.warning("rejected cross-origin %s %s", request.method, request.path)
            return err("cross-origin request rejected", 403)


# Content-Security-Policy for the single-page app (audit finding E3). Scripts must come from this origin:
# there are no inline scripts or handlers (a test forbids them), so an injected <script> or onclick= cannot
# run. Inline style *attributes* are allowed because the page builds its markup from templates that use
# them; <style> elements are not. Nothing is loaded from another origin, and the page cannot be framed.
CSP = "; ".join([
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self'",
    "style-src-attr 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
])
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


@app.after_request
def _security_headers(resp):
    """Applied to every response, including the guard's and the error handlers' (they pass through here)."""
    for name, value in SECURITY_HEADERS.items():
        resp.headers.setdefault(name, value)
    if request.path.startswith("/api/"):
        # Observations and sessions are sensitive; keep them out of any browser or proxy cache.
        resp.headers["Cache-Control"] = "no-store"
    return resp


def ok(data=None, **kw):
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    payload.update(kw)
    return jsonify(payload)


def err(message, code=400, **kw):
    payload = {"ok": False, "error": str(message)}
    payload.update(kw)
    return jsonify(payload), code


# -------- Error handling: one JSON envelope, client mistakes -> 4xx, server faults -> opaque 500 --------
@app.errorhandler(ApiError)
def _api_error(exc: ApiError):
    return err(str(exc), exc.status, **exc.extra)


@app.errorhandler(HTTPException)
def _http_error(exc: HTTPException):
    return err(exc.description or exc.name, exc.code or 500)


@app.errorhandler(Exception)
def _unhandled(exc: Exception):
    # Never send the exception text to the client (it can hold paths or SQL); log it with an id
    # the person can quote. Request bodies are deliberately not logged: they are health/behaviour notes.
    error_id = uuid.uuid4().hex[:8]
    log.error("unhandled error %s during %s %s", error_id, request.method, request.path, exc_info=exc)
    return err("Internal error", 500, error_id=error_id)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return ok({"status": "up", "screening": screening.readiness()})


# -------- Screening / sessions --------
@app.get("/api/meta")
def api_meta():
    m = screening.meta()
    m["policies"] = research.available_policies()
    return ok(m)


@app.post("/api/screening/start")
def screening_start():
    body = v.parse_body(v.ScreeningStart)
    return ok(screening.start_session(body.mode, body.policy or POLICY_DEFAULT))


@app.post("/api/screening/answer")
def screening_answer():
    body = v.parse_body(v.ScreeningAnswer)
    return ok(screening.answer_session(body.session_id, value=body.value, stop=body.stop, item_index=body.item_index,
                                        skip=body.skip))


@app.get("/api/screening/sessions")
def screening_sessions():
    return ok(screening.list_sessions())


@app.get("/api/screening/sessions/<sid>")
def screening_session_get(sid):
    return ok(screening.get_session(sid))


# -------- Observations --------
def _require_row(conn, table: str, row_id, what: str) -> None:
    """A reference to another record must point at one that exists (no dangling ids)."""
    if row_id is not None and conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (row_id,)).fetchone() is None:
        raise BadRequest(f"unknown {what}")


@app.get("/api/observations")
def obs_list():
    q = v.query_text("q").lower()
    ctx = v.query_text("context")
    limit = v.query_limit(1000, 5000)
    rows = get_conn().execute(
        "SELECT * FROM observations WHERE (:ctx = '' OR context = :ctx) AND (:q = '' OR instr(lower_u("
        "coalesce(what_happened,'') || ' ' || coalesce(difficulty,'') || ' ' || coalesce(what_helped,'') || ' ' || "
        "coalesce(notes,'') || ' ' || context), :q) > 0) "
        "ORDER BY COALESCE(occurred_at, created_at) DESC, created_at DESC LIMIT :limit",
        {"ctx": ctx, "q": q, "limit": limit}).fetchall()
    return ok(rows_to_dicts(rows))


@app.post("/api/observations")
def obs_create():
    body = v.parse_body(v.ObservationIn)
    oid = new_id("obs_")
    now = utc_now()
    occurred = body.occurred_at or now
    with tx() as conn:
        _require_row(conn, "screening_sessions", body.session_id, "session_id")
        conn.execute(
            "INSERT INTO observations(id,created_at,updated_at,context,what_happened,frequency,difficulty,what_helped,"
            "consistency,insufficient_info,notes,session_id,occurred_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, now, now, body.context, body.what_happened, body.frequency, body.difficulty, body.what_helped,
             body.consistency, int(body.insufficient_info), body.notes, body.session_id, occurred))
        record_event(conn, "observation", "Observation recorded", ref_type="observation", ref_id=oid, event_time=occurred)
    return ok({"id": oid})


@app.put("/api/observations/<oid>")
def obs_update(oid):
    changes = v.parse_body(v.ObservationUpdate).provided()
    with tx() as conn:
        if conn.execute("SELECT 1 FROM observations WHERE id=?", (oid,)).fetchone() is None:
            raise NotFound("not found")
        if not changes:
            return ok({"id": oid})
        _require_row(conn, "screening_sessions", changes.get("session_id"), "session_id")
        if "insufficient_info" in changes:
            changes["insufficient_info"] = int(changes["insufficient_info"])
        cols = list(changes)  # keys come from the validated model's fields, never from the request
        conn.execute(f"UPDATE observations SET {', '.join(c + '=?' for c in cols)}, updated_at=? WHERE id=?",
                     [changes[c] for c in cols] + [utc_now(), oid])
        record_event(conn, "observation_update", "Observation updated", ref_type="observation", ref_id=oid)
    return ok({"id": oid})


@app.delete("/api/observations/<oid>")
def obs_delete(oid):
    with tx() as conn:
        if conn.execute("DELETE FROM observations WHERE id=?", (oid,)).rowcount == 0:
            raise NotFound("not found")
        conn.execute("UPDATE support_strategies SET triggered_observation_id=NULL WHERE triggered_observation_id=?", (oid,))
        record_event(conn, "observation_delete", "Observation removed", ref_type="observation", ref_id=oid,
                     summary="An observation record was deleted.")
    return ok({"deleted": True})


# -------- Support strategies --------
@app.get("/api/support/strategies")
def strategies_list():
    rows = get_conn().execute("SELECT * FROM support_strategies ORDER BY updated_at DESC").fetchall()
    return ok(rows_to_dicts(rows))


@app.post("/api/support/suggest")
def strategies_suggest():
    body = v.parse_body(v.SuggestIn)
    row = get_conn().execute("SELECT * FROM observations WHERE id=?", (body.observation_id,)).fetchone()
    if not row:
        raise NotFound("observation not found")
    return ok(suggest_for_observation(dict(row)))


@app.post("/api/support/strategies")
def strategies_create():
    body = v.parse_body(v.StrategyIn)
    sid = new_id("sup_")
    now = utc_now()
    with tx() as conn:
        _require_row(conn, "observations", body.triggered_observation_id, "triggered_observation_id")
        conn.execute(
            "INSERT INTO support_strategies(id,created_at,updated_at,title,domain,description,triggered_observation_id,"
            "why_useful,how_to_try,how_to_record,status,outcome,notes,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, now, now, body.title, body.domain, body.description, body.triggered_observation_id, body.why_useful,
             body.how_to_try, body.how_to_record, body.status, body.outcome, body.notes, body.source or "user"))
        record_event(conn, "support", "Support strategy saved", ref_type="support_strategy", ref_id=sid)
    return ok({"id": sid})


@app.put("/api/support/strategies/<sid>")
def strategies_update(sid):
    changes = v.parse_body(v.StrategyUpdate).provided()
    with tx() as conn:
        if conn.execute("SELECT 1 FROM support_strategies WHERE id=?", (sid,)).fetchone() is None:
            raise NotFound("not found")
        if not changes:
            return ok({"id": sid})
        _require_row(conn, "observations", changes.get("triggered_observation_id"), "triggered_observation_id")
        cols = list(changes)
        conn.execute(f"UPDATE support_strategies SET {', '.join(c + '=?' for c in cols)}, updated_at=? WHERE id=?",
                     [changes[c] for c in cols] + [utc_now(), sid])
        record_event(conn, "support_update", "Support strategy updated", ref_type="support_strategy", ref_id=sid)
    return ok({"id": sid})


@app.delete("/api/support/strategies/<sid>")
def strategies_delete(sid):
    with tx() as conn:
        # strategy_attempts rows go with it (ON DELETE CASCADE)
        if conn.execute("DELETE FROM support_strategies WHERE id=?", (sid,)).rowcount == 0:
            raise NotFound("not found")
        record_event(conn, "support_delete", "Support strategy removed", ref_type="support_strategy", ref_id=sid,
                     summary="A support strategy was deleted.")
    return ok({"deleted": True})


@app.post("/api/support/strategies/<sid>/attempts")
def strategies_attempt(sid):
    body = v.parse_body(v.AttemptIn)
    aid = new_id("att_")
    with tx() as conn:
        if conn.execute("SELECT 1 FROM support_strategies WHERE id=?", (sid,)).fetchone() is None:
            raise NotFound("not found")
        conn.execute("INSERT INTO strategy_attempts(id,strategy_id,created_at,status,outcome,notes) VALUES(?,?,?,?,?,?)",
                     (aid, sid, utc_now(), body.status, body.outcome, body.notes))
        record_event(conn, "support_attempt", "Support strategy tried/updated", ref_type="strategy_attempt", ref_id=aid,
                     payload={"attempt": aid, "status": body.status})
    return ok({"id": aid})


# -------- Timeline --------
@app.get("/api/timeline")
def timeline_list():
    return ok(list_events(get_conn(), q=v.query_text("q"), kind=v.query_text("kind"), limit=v.query_limit(500, 2000)))


def _export_sessions(conn) -> list:
    out = []
    for r in conn.execute("SELECT * FROM screening_sessions ORDER BY created_at DESC").fetchall():
        d = dict(r)
        for col in [c for c in d if c.endswith("_json")]:
            d[col[:-5]] = jload(d.pop(col))
        out.append(d)
    return out


@app.post("/api/timeline/export")
def timeline_export():
    # JSON payload for the client to download; nothing leaves the machine. One read transaction, so
    # the tables in the export are consistent with each other.
    with snapshot() as conn:
        payload = {
            "exported_at": utc_now(),
            "observations": rows_to_dicts(conn.execute("SELECT * FROM observations ORDER BY created_at DESC").fetchall()),
            "support_strategies": rows_to_dicts(conn.execute("SELECT * FROM support_strategies ORDER BY created_at DESC").fetchall()),
            "strategy_attempts": rows_to_dicts(conn.execute("SELECT * FROM strategy_attempts ORDER BY created_at DESC").fetchall()),
            "screening_sessions": _export_sessions(conn),
            "consent_records": rows_to_dicts(conn.execute("SELECT * FROM consent_records ORDER BY created_at DESC").fetchall()),
            "timeline": list_events(conn, limit=None),
        }
    payload["note"] = "Local export of every record this app stores. Handle as sensitive information."
    return ok(payload)


# -------- Consent / privacy --------
@app.get("/api/consent")
def consent_get():
    rows = get_conn().execute("SELECT * FROM consent_records ORDER BY updated_at DESC").fetchall()
    return ok(rows_to_dicts(rows))


@app.post("/api/consent")
def consent_set():
    # An acknowledgement log: it records what the person chose and when. Nothing in the app is gated on
    # it (the data stays on this machine either way), and the UI says so.
    body = v.parse_body(v.ConsentIn)
    cid = new_id("con_")
    now = utc_now()
    with tx() as conn:
        conn.execute("INSERT INTO consent_records(id,created_at,updated_at,scope,granted,details) VALUES(?,?,?,?,?,?)",
                     (cid, now, now, body.scope, int(body.granted), body.details))
    return ok({"id": cid, "scope": body.scope, "granted": body.granted})


@app.post("/api/privacy/delete_observations")
def privacy_delete_observations():
    with tx() as conn:
        conn.execute("UPDATE support_strategies SET triggered_observation_id=NULL WHERE triggered_observation_id IS NOT NULL")
        conn.execute("DELETE FROM observations")
        record_event(conn, "privacy", "Observations deleted", ref_type="privacy",
                     summary="All observation records were removed by user request.", payload={"action": "delete_observations"})
    return ok({"deleted": True})


@app.post("/api/privacy/delete_sessions")
def privacy_delete_sessions():
    with tx() as conn:
        n = conn.execute("DELETE FROM screening_sessions").rowcount
        conn.execute("UPDATE observations SET session_id=NULL WHERE session_id IS NOT NULL")
        record_event(conn, "privacy", "Screening sessions deleted", ref_type="privacy",
                     summary="All screening sessions were removed by user request.", payload={"action": "delete_sessions"})
    return ok({"deleted": True, "counts": {"screening_sessions": n}})


@app.post("/api/privacy/delete_all")
def privacy_delete_all():
    """Erase everything the app stored about people: observations, support strategies and attempts,
    screening sessions and the timeline. Consent records are kept as the log of the person's choices."""
    with tx() as conn:
        counts = {}
        for table in ("observations", "support_strategies", "screening_sessions"):
            counts[table] = conn.execute(f"DELETE FROM {table}").rowcount  # attempts cascade from strategies
        counts["timeline_events"] = conn.execute("DELETE FROM timeline_events").rowcount
        record_event(conn, "privacy", "All app data deleted", ref_type="privacy",
                     summary="All observations, support strategies, screening sessions and timeline events were removed by user request.",
                     payload={"action": "delete_all"})
    return ok({"deleted": True, "counts": counts})


# -------- Research dashboard --------
@app.get("/api/research/dashboard")
def research_dashboard_api():
    return ok(research.research_dashboard())


# -------- RL Lab --------
@app.get("/api/rl/scenarios")
def rl_scenarios():
    return ok({"scenarios": rl_lab.synthetic_scenarios(), "policies": research.available_policies(), "budgets": [1, 2, 3, 4, 5, 6, 8, 10]})


@app.post("/api/rl/replay")
def rl_replay():
    body = v.parse_body(v.RlReplay)
    return ok(rl_lab.replay(body.scenario or "syn_mixed", body.policy or "greedy", body.budget or 6))


@app.post("/api/rl/compare")
def rl_compare():
    body = v.parse_body(v.RlCompare)
    pids = body.policies or ["greedy", "random", "beta_greedy"]
    return ok(rl_lab.compare(body.scenario or "syn_mixed", pids, body.budget or 6))


def configure_logging() -> None:
    """Console logging for the app (only if the host process has not configured logging itself)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("NEUROBRIDGE_LOG_LEVEL", "INFO").upper(),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app():
    configure_logging()
    init_db()
    return app


if __name__ == "__main__":
    configure_logging()
    init_db()
    # Pre-warm screening resources can be slow; allow lazy load on first request in dev.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8010")), debug=False)
