from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory
try:
    from flask_cors import CORS
except Exception:  # optional
    CORS = None

from app.config import STATIC_DIR, SECRET_KEY
from app.db import init_db, tx, get_conn, new_id, utc_now, jdump, jload, rows_to_dicts
from app.services import screening, research, rl_lab

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["SECRET_KEY"] = SECRET_KEY
if CORS:
    CORS(app)


@app.before_request
def _init():
    # lazy init to keep startup responsive; idempotent
    if not getattr(app, "_db_ready", False):
        init_db()
        app._db_ready = True


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


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return ok({"status": "up"})


# -------- Screening / sessions --------
@app.get("/api/meta")
def api_meta():
    m = screening.meta()
    m["policies"] = research.available_policies()
    return ok(m)


@app.post("/api/screening/start")
def screening_start():
    req = request.get_json(silent=True) or {}
    try:
        res = screening.start_session(req.get("mode", "interactive"), req.get("policy") or "greedy")
        return ok(res)
    except Exception as e:
        return err(e)


@app.post("/api/screening/answer")
def screening_answer():
    req = request.get_json(silent=True) or {}
    sid = req.get("session_id", "")
    try:
        res = screening.answer_session(sid, value=req.get("value"), stop=bool(req.get("stop")))
        return ok(res)
    except Exception as e:
        return err(e)


@app.get("/api/screening/sessions")
def screening_sessions():
    return ok(screening.list_sessions())


@app.get("/api/screening/sessions/<sid>")
def screening_session_get(sid):
    try:
        return ok(screening.get_session(sid))
    except Exception as e:
        return err(e, 404)


# -------- Observations --------
@app.get("/api/observations")
def obs_list():
    q = (request.args.get("q") or "").strip().lower()
    ctx = (request.args.get("context") or "").strip()
    conn = get_conn()
    rows = conn.execute("SELECT * FROM observations ORDER BY COALESCE(occurred_at, created_at) DESC, created_at DESC").fetchall()
    items = rows_to_dicts(rows)
    if ctx:
        items = [x for x in items if x.get("context") == ctx]
    if q:
        def hit(x):
            blob = " ".join(str(x.get(k) or "") for k in ("what_happened","difficulty","what_helped","notes","context")).lower()
            return q in blob
        items = [x for x in items if hit(x)]
    return ok(items)


@app.post("/api/observations")
def obs_create():
    req = request.get_json(silent=True) or {}
    if not req.get("context") or not req.get("what_happened"):
        return err("context and what_happened are required")
    oid = new_id("obs_")
    now = utc_now()
    with tx() as conn:
        conn.execute(
            "INSERT INTO observations(id,created_at,updated_at,context,what_happened,frequency,difficulty,what_helped,consistency,insufficient_info,notes,session_id,occurred_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (oid, now, now, req.get("context"), req.get("what_happened"), req.get("frequency"), req.get("difficulty"), req.get("what_helped"), req.get("consistency"), 1 if req.get("insufficient_info") else 0, req.get("notes"), req.get("session_id"), req.get("occurred_at") or now)
        )
        conn.execute(
            "INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (new_id("evt_"), now, req.get("occurred_at") or now, "observation", "Observation recorded", req.get("what_happened")[:180], req.get("context"), "observation", oid, req.get("notes"), jdump({"frequency": req.get("frequency"), "consistency": req.get("consistency")}))
        )
    return ok({"id": oid})


@app.put("/api/observations/<oid>")
def obs_update(oid):
    req = request.get_json(silent=True) or {}
    conn = get_conn()
    row = conn.execute("SELECT id FROM observations WHERE id=?", (oid,)).fetchone()
    if not row:
        return err("not found", 404)
    fields = ["context","what_happened","frequency","difficulty","what_helped","consistency","insufficient_info","notes","session_id","occurred_at"]
    sets=[]; vals=[]
    for f in fields:
        if f in req:
            sets.append(f+"=?")
            v = req[f]
            if f=="insufficient_info": v = 1 if v else 0
            vals.append(v)
    if not sets:
        return ok({"id": oid})
    vals.extend([utc_now(), oid])
    with tx() as conn:
        conn.execute(f"UPDATE observations SET {', '.join(sets)}, updated_at=? WHERE id=?", vals)
        # timeline note
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), utc_now(), utc_now(), "observation_update", "Observation updated", (req.get("what_happened") or "Observation edited")[:180], req.get("context"), "observation", oid, req.get("notes"), jdump({"updated": True})))
    return ok({"id": oid})


@app.delete("/api/observations/<oid>")
def obs_delete(oid):
    with tx() as conn:
        cur = conn.execute("DELETE FROM observations WHERE id=?", (oid,))
        if cur.rowcount == 0:
            return err("not found", 404)
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), utc_now(), utc_now(), "observation_delete", "Observation removed", "An observation record was deleted.", None, "observation", oid, None, jdump({"deleted": True})))
    return ok({"deleted": True})


# -------- Support strategies --------
def _strategy_suggestions(observation: dict) -> list:
    # Non-diagnostic, observation-grounded suggestions. Each explains trigger/why/how/record.
    text = " ".join(str(observation.get(k) or "") for k in ("what_happened","difficulty","context","what_helped","notes")).lower()
    ctx = (observation.get("context") or "").lower()
    sug = []
    def add(title, domain, why, how, record):
        sug.append({"title": title, "domain": domain, "why_useful": why, "how_to_try": how, "how_to_record": record,
                    "triggered_observation_id": observation.get("id"), "source": "rule-based from recorded observation"})
    if "transition" in text or ctx == "transitions":
        add("Advance transition warnings", "Transitions", "The observation involves change/transition; advance warnings can make upcoming changes more predictable.", "Give a clear warning before a change (e.g. 5 and 1 minute), using a timer or visual cue.", "Note whether the warning was given and if the transition felt easier or harder.")
    if "routine" in text or "schedule" in text or ctx == "daily routines":
        add("Visual schedule", "Routines", "Recorded routine-related difficulty may benefit from making steps visible and predictable.", "Use a simple picture/step schedule for the routine and review it before starting.", "Record which steps were used and whether the routine was smoother.")
    if "noise" in text or "loud" in text or "sensory" in text or ctx == "sensory":
        add("Quieter environment options", "Sensory environment", "The observation mentions sensory/noise factors; reducing stimulation can lower discomfort for some people.", "Offer a quieter space, headphones, or reduced background noise when possible.", "Record the environment change and whether comfort/engagement improved.")
    if "instruction" in text or "understand" in text or "communication" in text or ctx == "communication":
        add("Clearer visual instructions", "Communication", "Communication/instruction difficulty was noted; visual supports can make expectations clearer.", "Pair spoken instructions with simple visuals, written steps, or demonstrations.", "Record which support was used and whether understanding improved.")
    if "distract" in text or "focus" in text or ctx == "classroom":
        add("Reduced environmental distractions", "Environment", "Distraction was noted; simplifying the environment can support focus.", "Reduce clutter/noise near the activity and provide a clear workspace.", "Record the change and any effect on focus or participation.")
    if "change" in text or "unexpected" in text or "flexib" in text:
        add("Flexible routine options", "Adaptability", "Difficulty with change was noted; offering choices can support flexibility.", "Prepare a simple alternative/choice when plans change and explain it clearly.", "Record the options offered and how the change was experienced.")
    if not sug:
        add("Structured support check-in", observation.get("context") or "General", "A recorded observation may benefit from a small, observable support to try.", "Choose one small environmental or communication adjustment related to what was observed.", "Record what was tried, context, and whether it seemed helpful.")
    # dedupe by title
    seen=set(); out=[]
    for s in sug:
        if s["title"] in seen: continue
        seen.add(s["title"]); out.append(s)
    return out


@app.get("/api/support/strategies")
def strategies_list():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM support_strategies ORDER BY updated_at DESC").fetchall()
    return ok(rows_to_dicts(rows))


@app.post("/api/support/suggest")
def strategies_suggest():
    req = request.get_json(silent=True) or {}
    oid = req.get("observation_id")
    if not oid:
        return err("observation_id required")
    conn = get_conn()
    row = conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()
    if not row:
        return err("observation not found", 404)
    obs = dict(row)
    return ok(_strategy_suggestions(obs))


@app.post("/api/support/strategies")
def strategies_create():
    req = request.get_json(silent=True) or {}
    if not req.get("title"):
        return err("title required")
    sid = new_id("sup_")
    now = utc_now()
    with tx() as conn:
        conn.execute(
            "INSERT INTO support_strategies(id,created_at,updated_at,title,domain,description,triggered_observation_id,why_useful,how_to_try,how_to_record,status,outcome,notes,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, now, now, req.get("title"), req.get("domain"), req.get("description"), req.get("triggered_observation_id"), req.get("why_useful"), req.get("how_to_try"), req.get("how_to_record"), req.get("status") or "saved", req.get("outcome"), req.get("notes"), req.get("source") or "user")
        )
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), now, now, "support", "Support strategy saved", req.get("title"), req.get("domain"), "support_strategy", sid, req.get("notes"), jdump({"status": req.get("status") or "saved"})))
    return ok({"id": sid})


@app.put("/api/support/strategies/<sid>")
def strategies_update(sid):
    req = request.get_json(silent=True) or {}
    fields = ["title","domain","description","triggered_observation_id","why_useful","how_to_try","how_to_record","status","outcome","notes","source"]
    sets=[]; vals=[]
    for f in fields:
        if f in req:
            sets.append(f+"=?"); vals.append(req[f])
    if not sets:
        return ok({"id": sid})
    vals.extend([utc_now(), sid])
    with tx() as conn:
        cur = conn.execute(f"UPDATE support_strategies SET {', '.join(sets)}, updated_at=? WHERE id=?", vals)
        if cur.rowcount == 0:
            return err("not found", 404)
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), utc_now(), utc_now(), "support_update", "Support strategy updated", req.get("title") or "Strategy updated", req.get("domain"), "support_strategy", sid, req.get("notes"), jdump({"status": req.get("status"), "outcome": req.get("outcome")})))
    return ok({"id": sid})


@app.delete("/api/support/strategies/<sid>")
def strategies_delete(sid):
    with tx() as conn:
        cur = conn.execute("DELETE FROM support_strategies WHERE id=?", (sid,))
        if cur.rowcount == 0:
            return err("not found", 404)
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), utc_now(), utc_now(), "support_delete", "Support strategy removed", "A support strategy was deleted.", None, "support_strategy", sid, None, jdump({"deleted": True})))
    return ok({"deleted": True})


@app.post("/api/support/strategies/<sid>/attempts")
def strategies_attempt(sid):
    req = request.get_json(silent=True) or {}
    aid = new_id("att_")
    with tx() as conn:
        row = conn.execute("SELECT id,title FROM support_strategies WHERE id=?", (sid,)).fetchone()
        if not row:
            return err("not found", 404)
        conn.execute("INSERT INTO strategy_attempts(id,strategy_id,created_at,status,outcome,notes) VALUES(?,?,?,?,?,?)",
                     (aid, sid, utc_now(), req.get("status") or "tried", req.get("outcome"), req.get("notes")))
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), utc_now(), utc_now(), "support_attempt", "Support strategy tried/updated", row["title"], None, "support_strategy", sid, req.get("notes"), jdump({"attempt": aid, "status": req.get("status"), "outcome": req.get("outcome")})))
    return ok({"id": aid})


# -------- Timeline --------
@app.get("/api/timeline")
def timeline_list():
    q = (request.args.get("q") or "").strip().lower()
    kind = (request.args.get("kind") or "").strip()
    conn = get_conn()
    rows = conn.execute("SELECT * FROM timeline_events ORDER BY event_time DESC, created_at DESC LIMIT 500").fetchall()
    items = rows_to_dicts(rows)
    if kind:
        items = [x for x in items if x.get("kind") == kind]
    if q:
        items = [x for x in items if q in " ".join(str(x.get(k) or "") for k in ("title","summary","notes","context")).lower()]
    return ok(items)


@app.post("/api/timeline/export")
def timeline_export():
    # Return JSON export payload (client can download); no external transfer.
    conn = get_conn()
    obs = rows_to_dicts(conn.execute("SELECT * FROM observations ORDER BY created_at DESC").fetchall())
    sups = rows_to_dicts(conn.execute("SELECT * FROM support_strategies ORDER BY created_at DESC").fetchall())
    events = rows_to_dicts(conn.execute("SELECT * FROM timeline_events ORDER BY event_time DESC").fetchall())
    return ok({"exported_at": utc_now(), "observations": obs, "support_strategies": sups, "timeline": events,
               "note": "Local export of recorded observations/support/timeline data. Handle as sensitive information."})


# -------- Consent / privacy --------
@app.get("/api/consent")
def consent_get():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM consent_records ORDER BY updated_at DESC").fetchall()
    return ok(rows_to_dicts(rows))


@app.post("/api/consent")
def consent_set():
    req = request.get_json(silent=True) or {}
    scope = req.get("scope") or "general_research_use"
    granted = 1 if req.get("granted") else 0
    details = req.get("details")
    cid = new_id("con_")
    now = utc_now()
    with tx() as conn:
        conn.execute("INSERT INTO consent_records(id,created_at,updated_at,scope,granted,details) VALUES(?,?,?,?,?,?)",
                     (cid, now, now, scope, granted, details))
    return ok({"id": cid, "scope": scope, "granted": bool(granted)})


@app.post("/api/privacy/delete_observations")
def privacy_delete_observations():
    # destructive within app data only
    with tx() as conn:
        conn.execute("DELETE FROM observations")
        conn.execute("INSERT INTO timeline_events(id,created_at,event_time,kind,title,summary,context,ref_type,ref_id,notes,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (new_id("evt_"), utc_now(), utc_now(), "privacy", "Observations deleted", "All observation records were removed by user request.", None, "privacy", None, None, jdump({"action": "delete_observations"})))
    return ok({"deleted": True})


# -------- Research dashboard --------
@app.get("/api/research/dashboard")
def research_dashboard_api():
    return ok(research.research_dashboard())


# -------- RL Lab --------
@app.get("/api/rl/scenarios")
def rl_scenarios():
    return ok({"scenarios": rl_lab.synthetic_scenarios(), "policies": research.available_policies(), "budgets": [1,2,3,4,5,6,8,10]})


@app.post("/api/rl/replay")
def rl_replay():
    req = request.get_json(silent=True) or {}
    try:
        res = rl_lab.replay(req.get("scenario") or "syn_mixed", req.get("policy") or "greedy", int(req.get("budget") or 6))
        return ok(res)
    except Exception as e:
        return err(e)


@app.post("/api/rl/compare")
def rl_compare():
    req = request.get_json(silent=True) or {}
    pids = req.get("policies") or ["greedy", "random", "beta_greedy"]
    try:
        res = rl_lab.compare(req.get("scenario") or "syn_mixed", pids, int(req.get("budget") or 6))
        return ok(res)
    except Exception as e:
        return err(e)


def create_app():
    init_db()
    return app


if __name__ == "__main__":
    init_db()
    # Pre-warm screening resources can be slow; allow lazy load on first request in dev.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8010")), debug=False)
