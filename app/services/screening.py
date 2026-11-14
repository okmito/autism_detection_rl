from __future__ import annotations
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# The package is imported from the repository root (``python -m app.server``) or after ``pip install -e .``;
# it no longer edits sys.path or changes the process working directory on import (audit finding T5).
ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from src.audits.circularity import audit_circularity
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from src.screening import model_cache, session as engine
from src.screening.diagnostics import ITEMS, item_payload

from app.config import SESSION_TTL_HOURS, BUDGET_DEFAULT, TAU_DEFAULT, N_ITEMS, SEED, DEMO_CALIBRATION, ROOT as _ROOT
from app.db import get_conn, tx, new_id, utc_now, jdump, jload, rows_to_dicts
from app.errors import BadRequest, Conflict, NotFound, Unavailable

log = logging.getLogger("neurobridge.screening")

_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {"ready": False}

def _build_predictor(retrain: bool = False):
    """Load or train the predictor through the shared, hash-verified cache (src/screening/model_cache.py)."""
    return model_cache.build_predictor(
        _ROOT / "results", seed=SEED, n_items=N_ITEMS, calibration=DEMO_CALIBRATION, retrain=retrain,
        info=log.info, warn=log.warning)


_MISSING_DATA_MESSAGE = "Screening data not found: place the Saudi Q-CHAT-10 CSV under data/raw/ (see README)."


def readiness() -> dict:
    """Cheap status for /api/health: is the dataset present, and is the model already loaded?"""
    from src.data import ingest
    present = (ROOT / ingest.SAUDI_CSV).exists()
    return {"data_present": bool(present), "model_loaded": bool(_STATE.get("ready")),
            "message": None if present else _MISSING_DATA_MESSAGE}


def ensure_ready():
    if _STATE.get("ready"):
        return
    with _LOCK:
        if _STATE.get("ready"):
            return
        try:
            records, train, _val, test, pred = _build_predictor(retrain=False)
        except FileNotFoundError as exc:
            # The exception text contains an absolute local path; log it, do not send it to the client.
            log.error("screening data unavailable: %s", exc)
            raise Unavailable(_MISSING_DATA_MESSAGE) from None
        _STATE.update({
            "ready": True,
            "records": records,
            "train": train,
            "test": test,
            "predictor": pred,
            "counter": 0,
        })


def predict(state: dict) -> float:
    ensure_ready()
    with _LOCK:
        return float(_STATE["predictor"](state))


def make_policy(name: str):
    """The policy object for ``name`` (greedy is the default)."""
    ensure_ready()
    train = _STATE["train"]
    if name in ("greedy", None):
        return GreedyIGPolicy(train, N_ITEMS)
    if name == "random":
        # seed=None: each session draws fresh entropy, so two random sessions differ.
        return RandomPolicy(seed=None)
    if name == "beta_greedy":
        from src.policies.beta_greedy import BetaGreedyPolicy
        return BetaGreedyPolicy(train, N_ITEMS)
    raise BadRequest(f"unsupported policy {name!r}")


def _context() -> engine.Context:
    ensure_ready()
    return engine.Context(
        predict=predict, predictor=_STATE["predictor"], train=_STATE["train"], records=_STATE["records"],
        n_items=N_ITEMS, budget=BUDGET_DEFAULT, tau=TAU_DEFAULT,
        tag="app session — research prototype, not a diagnostic device", input_reference="app-session")


RL_POLICY = "rl_adaptive"
LIVE_POLICIES = ("greedy", "random", "beta_greedy", RL_POLICY)


def rl_status() -> dict:
    """Load the served RL policy once. ``{"agent", "info", "reason", "records"}``; agent None when unavailable.

    Never raises: an unavailable policy is a state the UI and the session record report, not a server error.
    """
    with _LOCK:
        if "rl" in _STATE:
            return _STATE["rl"]
        from src.rl.served import ServedPolicyUnavailable, development_records, load_served_agent
        out = {"agent": None, "info": None, "reason": None, "records": []}
        try:
            out["agent"], out["info"] = load_served_agent(_ROOT / "results")
        except ServedPolicyUnavailable as exc:
            out["reason"] = str(exc)
            log.warning("RL policy unavailable: %s", exc)
        except Exception:   # a corrupt artifact must not take the screening service down
            log.exception("RL policy failed to load")
            out["reason"] = "the RL policy artifact could not be loaded"
        if out["agent"] is not None:
            try:
                from src.rl.data import load_polish_qchat10
                X, y = load_polish_qchat10()
                out["records"] = development_records(X, y)
            except Exception as exc:   # explanations then run without reference rows / circularity audit
                log.warning("RL development cohort unavailable for explanation reference rows: %s", exc)
            log.info("RL policy loaded: %s (artifact %s)", out["info"]["agent"], out["info"]["artifact_sha256"][:12])
        _STATE["rl"] = out
        return out


def rl_public_info() -> dict:
    st = rl_status()
    if st["agent"] is None:
        return {"available": False, "reason": st["reason"]}
    i = st["info"]
    return {"available": True, "agent": i["agent"], "trained_on": i["trained_on"], "cost_per_question": i["cost"],
            "max_questions": i["mdp"]["max_questions"], "min_questions": i["mdp"]["min_questions"],
            "gate": i["gate"], "expected_performance": i.get("expected_performance"),
            "artifact_sha256": i["artifact_sha256"]}


def _rl_context() -> engine.Context:
    st = rl_status()
    agent = st["agent"]
    return engine.Context(
        predict=agent.belief.predict_state, predictor=agent.belief, train=st["records"], records=st["records"],
        n_items=N_ITEMS, budget=agent.cfg.max_questions, tau=agent.cfg.tau,
        tag="app session (adaptive RL policy) — research prototype, not a diagnostic device",
        input_reference="app-session", dataset_tag="Polish Q-CHAT cohort projected to Q-CHAT-10 (clinician labels)",
        model_name="NaiveBayesBelief", model_version=agent.belief.metadata["predictor_version"])


def _policy_info(sess: dict) -> dict:
    return (sess.get("state") or {}).get("policy_info") or {"policy": sess.get("policy"), "fallback": None}


def _require_legal_item(j, legal) -> int:
    return engine.require_legal_item(j, legal, N_ITEMS)


def meta() -> dict:
    ensure_ready()
    recs = _STATE["records"]
    audit = audit_circularity(recs) if recs else {}
    y = np.array([r["label"] for r in recs]) if recs else np.array([0, 0])
    return {
        "budget": BUDGET_DEFAULT, "tau": TAU_DEFAULT, "n_items": N_ITEMS,
        "items": [{"code": f"A{i + 1}", "text": t} for i, t in enumerate(ITEMS)],
        "dataset": {"name": "Saudi toddler screening (Q-CHAT-10, real CSV)",
                    "n": int(len(recs)) if recs else 0,
                    "positive": int(y.sum()), "negative": int((1 - y).sum())},
        "circularity": {"threshold": audit.get("best_threshold"),
                        "exact_match": audit.get("exact_match_rate"),
                        "classification": audit.get("classification")},
        "policies_available": ["greedy", "random", "beta_greedy"] + ([RL_POLICY] if rl_status()["agent"] else []),
        "rl_policy": rl_public_info(),
    }


def _session_row(session_id: str) -> Optional[dict]:
    row = get_conn().execute("SELECT * FROM screening_sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def _state_to_jsonable(state: dict) -> dict:
    if state is None:
        return None
    s = dict(state)
    for k in ("mask", "value", "item_responses", "missing_mask"):
        if k in s and hasattr(s[k], "tolist"):
            s[k] = s[k].tolist()
    # ensure ints/floats native
    for k, v in list(s.items()):
        if isinstance(v, (np.integer,)):
            s[k] = int(v)
        elif isinstance(v, (np.floating,)):
            s[k] = float(v)
    return s

def _state_from_jsonable(state: dict) -> dict:
    if state is None:
        return None
    s = dict(state)
    for k in ("mask", "value"):
        if k in s and isinstance(s[k], list):
            s[k] = np.array(s[k], dtype=int)
    return s

def _to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _session_values(sess: dict) -> tuple:
    return (sess["status"], sess.get("asked_count", 0), jdump(_state_to_jsonable(sess.get("state"))),
            sess.get("pending_item"), jdump(_to_jsonable(sess.get("items_asked", []))),
            jdump(_to_jsonable(sess.get("trace", []))), jdump(_to_jsonable(sess.get("result"))),
            jdump(_to_jsonable(sess.get("explanation"))), jdump(_to_jsonable(sess.get("support_report"))),
            sess.get("stop_reason"))


def _persist_session(sess: dict):
    """Insert a new session, or update an existing one only if nobody else changed it meanwhile.

    Every update is ``WHERE id=? AND version=?`` and bumps ``version``. Two requests that load the same
    state (a double-clicked answer button, two tabs) can both compute a next step, but only one can
    persist it; the other gets a 409 instead of silently overwriting or answering twice.
    """
    with tx() as conn:
        if sess.get("version") is None:
            conn.execute(
                """INSERT INTO screening_sessions(id,created_at,updated_at,status,policy,mode,budget,tau,record_index,true_label,
                   asked_count,state_json,pending_item,items_asked_json,trace_json,result_json,explanation_json,support_report_json,
                   stop_reason,version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
                (sess["id"], sess["created_at"], utc_now(), sess["status"], sess["policy"], sess["mode"], sess["budget"],
                 sess["tau"], sess.get("record_index"), sess.get("true_label"), sess.get("asked_count", 0),
                 *_session_values(sess)[2:]))
            sess["version"] = 0
            return
        cur = conn.execute(
            """UPDATE screening_sessions SET updated_at=?,status=?,asked_count=?,state_json=?,pending_item=?,items_asked_json=?,
               trace_json=?,result_json=?,explanation_json=?,support_report_json=?,stop_reason=?,version=version+1
               WHERE id=? AND version=?""",
            (utc_now(), *_session_values(sess), sess["id"], sess["version"]))
        if cur.rowcount == 0:
            raise Conflict("This session was changed by another request; reload it and try again.")
        sess["version"] += 1


def _new_session(mode: str, policy_name: str) -> dict:
    return {"id": new_id("ses_"), "created_at": utc_now(), "status": "active",
            "policy": policy_name, "mode": mode, "budget": BUDGET_DEFAULT, "tau": TAU_DEFAULT,
            "record_index": None, "true_label": None, "asked_count": 0,
            "state": None, "pending_item": None, "items_asked": [], "trace": [],
            "result": None, "explanation": None, "support_report": None, "stop_reason": None}


def _complete(sess: dict, out: dict, state: dict, trace: list, items: list) -> dict:
    result = out["result"]
    sess.update({"status": "completed", "state": state, "pending_item": None, "trace": trace,
                 "items_asked": [int(x) for x in items], "asked_count": len(items), "result": result,
                 "explanation": out["explanation"], "support_report": out["support_report"],
                 "stop_reason": result["stop_reason"]})
    _persist_session(sess)
    return result


def start_session(mode: str, policy_name: str) -> dict:
    ensure_ready()
    if mode not in ("interactive", "auto"):
        raise BadRequest("mode must be interactive|auto")
    if policy_name not in LIVE_POLICIES:
        raise BadRequest("unsupported policy")
    fallback = None
    if policy_name == RL_POLICY:
        st = rl_status()
        if st["agent"] is None:
            # Explicit, recorded fallback: the session says which policy it really used and why.
            fallback = {"requested": RL_POLICY, "used": "greedy", "reason": st["reason"]}
            log.warning("rl_adaptive requested but unavailable; session falls back to greedy: %s", st["reason"])
            policy_name = "greedy"
        elif mode == "auto":
            raise BadRequest("auto replay is not offered for rl_adaptive: every record of its development cohort was "
                             "used to train it, so a replay would not be held out (see RL_ADAPTIVE_REPORT.md)")
        else:
            return _start_rl_session()
    ctx = _context()
    policy = make_policy(policy_name)
    sess = _new_session(mode, policy_name)
    head = {"session_id": sess["id"], "mode": mode, "budget": BUDGET_DEFAULT, "tau": TAU_DEFAULT, "policy": policy_name,
            "policy_info": {"policy": policy_name, "fallback": fallback}}
    if mode == "auto":
        # Auto mode *replays a held-out dataset record* so the policies can be compared on the same
        # people; it is the only mode bound to a record.
        with _LOCK:
            idx = _STATE["counter"] % len(_STATE["test"])
            _STATE["counter"] += 1
        rec = _STATE["test"][idx]
        out = engine.run_auto(ctx, policy, rec, result_id=sess["id"])
        out["result"]["policy_info"] = head["policy_info"]
        sess.update({"record_index": idx, "true_label": int(rec["label"])})
        ep = out["episode_result"]
        result = _complete(sess, out, ep["final_state"], ep["trace"], ep["items_asked"])
        return {**head, "result": result}
    # Interactive: a real respondent. Start from a blank state (T8): nothing is bound to a dataset record,
    # so no record's missingness can make a question unaskable and no unrelated label is attached.
    state = engine.blank_state(N_ITEMS, BUDGET_DEFAULT)
    state["policy_info"] = head["policy_info"]
    j, sel = engine.next_question(ctx, policy, state)   # never offers STOP as the opening action (C4)
    sess.update({"state": state, "pending_item": j})
    _persist_session(sess)
    return {**head, "question": item_payload(j), "belief": predict(state), "asked": 0, "selection": sel}


def _start_rl_session() -> dict:
    """Interactive adaptive-RL session: the agent picks every question and decides when to stop."""
    st = rl_status()
    agent, ctx = st["agent"], _rl_context()
    cfg = agent.cfg
    sess = _new_session("interactive", RL_POLICY)
    sess["budget"], sess["tau"] = cfg.max_questions, cfg.tau
    state = engine.blank_state(N_ITEMS, cfg.max_questions)
    info = {"policy": RL_POLICY, "fallback": None, "agent": st["info"]["agent"],
            "artifact_sha256": st["info"]["artifact_sha256"], "mdp": st["info"]["mdp"]}
    state["policy_info"] = info
    decision = engine.rl_decide(agent, state)
    if decision["action"]["type"] != "ASK":     # cannot happen with min_questions >= 1; never present STOP as a question
        raise Unavailable("the RL policy did not select an opening question")
    j = int(decision["action"]["item_index"])
    sess.update({"state": state, "pending_item": j})
    _persist_session(sess)
    return {"session_id": sess["id"], "mode": "interactive", "budget": cfg.max_questions, "tau": cfg.tau,
            "policy": RL_POLICY, "policy_info": info, "question": item_payload(j), "belief": ctx.predict(state),
            "asked": 0, "selection": None, "decision": decision, "min_questions": cfg.min_questions}


def _answer_rl(s: dict, value: Optional[int], stop: bool, item_index: Optional[int], skip: bool) -> dict:
    st = rl_status()
    if st["agent"] is None:
        raise Unavailable("This session used the adaptive RL policy, which is no longer available: "
                          f"{st['reason']}. Start a new session.")
    agent, ctx = st["agent"], _rl_context()
    if s["state"].get("policy_info", {}).get("artifact_sha256") != st["info"]["artifact_sha256"]:
        raise Conflict("The RL policy changed since this session started; start a new session.")
    state, asked = s["state"], int(s["asked_count"])
    trace, items_asked = list(s.get("trace") or []), list(s.get("items_asked") or [])

    def finish(reason: str, stop_explanation: Optional[dict]) -> dict:
        out = engine.finalize(ctx, state=state, trace=trace, items_asked=items_asked, stop_reason=reason,
                              result_id=s["id"])
        out["result"]["stop_explanation"] = stop_explanation
        out["result"]["policy_info"] = state.get("policy_info")
        return _complete(s, out, state, trace, items_asked)

    if stop:
        if asked < 1:
            raise BadRequest("answer at least one question before stopping")
        return {"finished": True, "result": finish("user_stop", None)}
    if skip and value is not None:
        raise BadRequest("send either a value or skip, not both")
    if not skip and value not in (0, 1):
        raise BadRequest("value must be 0 or 1 (or skip the question)")
    pending = s.get("pending_item")
    if pending is None:
        raise BadRequest("no pending question")
    j = int(pending)
    if item_index is None:
        raise BadRequest("item_index is required: say which question this answer is for")
    if int(item_index) != j:
        raise Conflict("That question has already been answered; reload the session to see the current one.")
    try:
        state, step = engine.rl_apply(ctx, agent, state, j, None if skip else int(value), asked)
    except RuntimeError:
        raise Conflict("The RL policy no longer selects this question; start a new session.") from None
    asked += 1
    items_asked.append(j)
    trace.append(step)
    decision = engine.rl_decide(agent, state)
    if decision["action"]["type"] == "STOP":
        return {"finished": True, "result": finish(decision["stop_reason"], decision)}
    nj = int(decision["action"]["item_index"])
    s.update({"state": state, "pending_item": nj, "items_asked": items_asked, "trace": trace,
              "asked_count": asked, "status": "active"})
    _persist_session(s)
    return {"finished": False, "belief_before": step["belief_before"], "belief_after": step["belief_after"],
            "trace_step": step, "asked": asked, "next": item_payload(nj), "selection": None, "decision": decision}


def _load_session(session_id: str) -> dict:
    row = _session_row(session_id)
    if not row:
        raise NotFound("unknown session - start a new one")
    row["state"] = _state_from_jsonable(jload(row.get("state_json")))
    row["items_asked"] = jload(row.get("items_asked_json"), [])
    row["trace"] = jload(row.get("trace_json"), [])
    row["result"] = jload(row.get("result_json"))
    row["explanation"] = jload(row.get("explanation_json"))
    row["support_report"] = jload(row.get("support_report_json"))
    return row


def answer_session(session_id: str, value: Optional[int] = None, stop: bool = False,
                   item_index: Optional[int] = None, skip: bool = False) -> dict:
    ensure_ready()
    expire_stale_sessions()
    s = _load_session(session_id)
    if s["status"] == "completed":
        return {"finished": True, "result": s["result"]}
    if s["status"] == "expired":
        raise Conflict("This session expired after a long pause. Start a new session.")
    if s["policy"] == RL_POLICY:
        return _answer_rl(s, value, stop, item_index, skip)
    if skip:
        raise BadRequest("skipping a question is supported by the rl_adaptive policy only")
    ctx = _context()
    state = s["state"]
    policy = make_policy(s["policy"])
    asked = int(s["asked_count"])
    trace = list(s.get("trace") or [])
    items_asked = list(s.get("items_asked") or [])
    pending = s.get("pending_item")

    def finish(reason: str) -> dict:
        out = engine.finalize(ctx, state=state, trace=trace, items_asked=items_asked, stop_reason=reason,
                              result_id=session_id)
        out["result"]["policy_info"] = _policy_info(s)
        return _complete(s, out, state, trace, items_asked)

    if stop:
        if asked < 1:
            raise BadRequest("answer at least one question before stopping")
        return {"finished": True, "result": finish("user_stop")}
    if value not in (0, 1):
        raise BadRequest("value must be 0 or 1")
    if pending is None:
        raise BadRequest("no pending question")
    j = int(pending)
    if item_index is None:
        raise BadRequest("item_index is required: say which question this answer is for")
    if int(item_index) != j:
        # A repeated or stale submission (double-click, second tab): the question it answers is no
        # longer the pending one. Answering the *current* one with this value would record an answer
        # the person never gave to a question they never saw.
        raise Conflict("That question has already been answered; reload the session to see the current one.")
    state, step = engine.apply_answer(ctx, policy, state, j, int(value), asked)
    asked += 1
    items_asked.append(j)
    trace.append(step)
    if engine.is_exhausted(ctx, state, asked):
        return {"finished": True, "result": finish("budget_exhausted")}
    nj, next_sel = engine.next_question(ctx, policy, state)
    s.update({"state": state, "pending_item": nj, "items_asked": items_asked, "trace": trace,
              "asked_count": asked, "status": "active"})
    _persist_session(s)
    return {"finished": False, "belief_before": step["belief_before"], "belief_after": step["belief_after"],
            "trace_step": step, "asked": asked, "next": item_payload(nj), "selection": next_sel}


def expire_stale_sessions(now: Optional[datetime] = None, ttl_hours: Optional[float] = None) -> int:
    """Mark unfinished sessions untouched for longer than the TTL as ``expired``; return how many.

    Nothing is deleted: the row, its answers and its trace stay (and remain covered by the privacy
    delete/export). Expiring only stops the Screening page from auto-resuming a session that was
    abandoned days ago, and stops it from being answered, because the person answering would no longer
    remember the context. Completed sessions are never touched. Timestamps are UTC ISO strings of one
    fixed format, so the string comparison is chronological.
    """
    now = now or datetime.now(timezone.utc)
    hours = SESSION_TTL_HOURS if ttl_hours is None else ttl_hours
    cutoff = (now - timedelta(hours=hours)).replace(microsecond=0).isoformat()
    with tx() as conn:
        cur = conn.execute(
            "UPDATE screening_sessions SET status='expired', version=version+1 "
            "WHERE status='active' AND updated_at < ?", (cutoff,))
        n = cur.rowcount
    if n:
        log.info("expired %d abandoned screening session(s) older than %s hours", n, hours)
    return n


def list_sessions(limit: int = 20) -> list:
    expire_stale_sessions()
    rows = get_conn().execute("SELECT id,created_at,updated_at,status,policy,mode,asked_count,stop_reason FROM screening_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    return rows_to_dicts(rows)


def get_session(session_id: str) -> dict:
    expire_stale_sessions()
    s = _load_session(session_id)
    # return safe summary for resume/UI
    out = {k: s.get(k) for k in ("id","created_at","updated_at","status","policy","mode","budget","tau","asked_count","stop_reason")}
    out["pending_question"] = item_payload(int(s["pending_item"])) if s.get("pending_item") is not None and s.get("status")=="active" else None
    out["trace"] = s.get("trace") or []
    out["result"] = s.get("result")
    if s.get("policy") == RL_POLICY:
        st = rl_status()
        out["belief"] = st["agent"].belief.predict_state(s["state"]) if (st["agent"] and s.get("state")) else None
        out["decision"] = (engine.rl_decide(st["agent"], s["state"])
                           if (st["agent"] and s.get("state") and s.get("status") == "active") else None)
        out["policy_info"] = (s.get("state") or {}).get("policy_info")
    else:
        out["belief"] = predict(s["state"]) if s.get("state") else None
        out["policy_info"] = _policy_info(s)
    return out
