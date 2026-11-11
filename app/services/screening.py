from __future__ import annotations
import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

from src.data.ingest import load_dataset
from src.data.splits import stratified_split
from src.audits.circularity import audit_circularity
from src.env.state import init_state, update_state, get_legal_items
from src.env.environment import STOP, run_episode
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from src.explain.counterfactual import find_counterfactual
from src.explain.outcome import explain_outcome
from src.support.questionnaire import ITEMS as _QCHAT_ITEMS

from app.config import BUDGET_DEFAULT, TAU_DEFAULT, N_ITEMS, SEED, DEMO_CALIBRATION, ROOT as _ROOT
from app.db import get_conn, tx, new_id, utc_now, jdump, jload, rows_to_dicts

_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {"ready": False}

ITEMS = [item.question_text for item in _QCHAT_ITEMS]
CACHE_STEM = f"demo_model_saudi_seed{SEED}_{DEMO_CALIBRATION}_v{MaskedPredictor.VERSION}"
CACHE_PT = _ROOT / "results" / f"{CACHE_STEM}.pt"
CACHE_PKL = _ROOT / "results" / f"{CACHE_STEM}.pkl"


def _build_predictor(retrain: bool = False):
    records = load_dataset("saudi")
    train, val, test = stratified_split(records, seed=SEED)
    pred = MaskedPredictor(n_items=N_ITEMS, hidden=[128, 64], calibration=DEMO_CALIBRATION, seed=SEED)
    if not retrain and CACHE_PT.exists() and CACHE_PKL.exists():
        try:
            pred.model.load_state_dict(torch.load(CACHE_PT, map_location="cpu"))
            with open(CACHE_PKL, "rb") as f:
                pred.calibrator = pickle.load(f)
        except Exception:
            pred.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=SEED)
            pred.fit_calibrator(val, method=DEMO_CALIBRATION)
    else:
        pred.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=SEED)
        pred.fit_calibrator(val, method=DEMO_CALIBRATION)
        try:
            CACHE_PT.parent.mkdir(exist_ok=True)
            torch.save(pred.model.state_dict(), CACHE_PT)
            with open(CACHE_PKL, "wb") as f:
                pickle.dump(pred.calibrator, f)
        except Exception:
            pass
    return records, train, val, test, pred


def ensure_ready():
    if _STATE.get("ready"):
        return
    with _LOCK:
        if _STATE.get("ready"):
            return
        records, train, _val, test, pred = _build_predictor(retrain=False)
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
    ensure_ready()
    train = _STATE["train"]
    if name in ("greedy", None):
        return GreedyIGPolicy(train, N_ITEMS)
    if name == "random":
        return RandomPolicy(seed=None)
    if name == "beta_greedy":
        from src.policies.beta_greedy import BetaGreedyPolicy
        return BetaGreedyPolicy(train, N_ITEMS)
    raise ValueError(f"unsupported policy {name!r}")


def item_payload(j: int) -> dict:
    return {"index": int(j), "code": f"A{j + 1}", "text": ITEMS[int(j)]}


# selection_diagnostic copied/adapted from demo_app for transparency
DEGENERACY_TOL = 1e-12

def selection_diagnostic(policy, state, legal) -> dict:
    items_only = [a for a in legal if a != STOP]
    out = {"applicable": hasattr(policy, "_support"), "n_legal": len(items_only),
           "support_size": None, "n_pos": None, "posterior": None,
           "support_is_pure": None, "ig_spread": None, "criterion_vacuous": None,
           "tie_at_max": None, "top_items": []}
    if not out["applicable"] or not items_only:
        return out
    idx = policy._support(state["mask"], state["value"])
    out["support_size"] = int(len(idx))
    if len(idx) == 0:
        out["criterion_vacuous"] = True
        return out
    ys = policy.y[idx]
    total = len(idx)
    n_pos = int(ys.sum())
    p_s = float(ys.mean())
    out["n_pos"] = n_pos
    out["posterior"] = p_s
    out["support_is_pure"] = bool(n_pos in (0, total))
    import math
    def entropy(p: float) -> float:
        if p <= 0 or p >= 1: return 0.0
        return -p*math.log2(p) - (1-p)*math.log2(1-p)
    h_s = entropy(p_s)
    ig = {}
    for j in items_only:
        sub = idx[~policy.missing_mask[idx, j]]
        if len(sub) == 0:
            continue
        exp_h = 0.0
        vals, counts = np.unique(policy.X[sub, j], return_counts=True)
        for v, c in zip(vals, counts):
            child = idx[(~policy.missing_mask[idx, j]) & (policy.X[idx, j] == v)]
            if len(child) == 0:
                continue
            exp_h += (c / total) * entropy(float(policy.y[child].mean()))
        n_miss = int(np.sum(policy.missing_mask[idx, j]))
        if n_miss > 0:
            child = idx[policy.missing_mask[idx, j]]
            p_child = float(policy.y[child].mean()) if len(child) > 0 else 0.5
            exp_h += (n_miss / total) * entropy(p_child)
        ig[j] = h_s - exp_h
    if not ig:
        out["criterion_vacuous"] = True
        return out
    best = max(ig.values()); worst = min(ig.values())
    out["ig_spread"] = best - worst
    out["criterion_vacuous"] = bool((best - worst) <= DEGENERACY_TOL)
    tied = sorted(j for j, v in ig.items() if best - v <= DEGENERACY_TOL)
    out["tie_at_max"] = len(tied)
    out["top_items"] = [{"code": f"A{j + 1}", "ig": ig[j]} for j in tied[:6]]
    return out


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
        "policies_available": ["greedy", "random", "beta_greedy"],
    }


def _session_row(session_id: str) -> Optional[dict]:
    conn = get_conn()
    with _LOCK:
        row = conn.execute("SELECT * FROM screening_sessions WHERE id=?", (session_id,)).fetchone()
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

def _persist_session(sess: dict):
    with tx() as conn:
        conn.execute(
            """INSERT INTO screening_sessions(id,created_at,updated_at,status,policy,mode,budget,tau,record_index,true_label,asked_count,state_json,pending_item,items_asked_json,trace_json,result_json,explanation_json,support_report_json,stop_reason)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,status=excluded.status,asked_count=excluded.asked_count,state_json=excluded.state_json,pending_item=excluded.pending_item,items_asked_json=excluded.items_asked_json,trace_json=excluded.trace_json,result_json=excluded.result_json,explanation_json=excluded.explanation_json,support_report_json=excluded.support_report_json,stop_reason=excluded.stop_reason""",
            (sess["id"], sess["created_at"], utc_now(), sess["status"], sess["policy"], sess["mode"], sess["budget"], sess["tau"], sess.get("record_index"), sess.get("true_label"), sess.get("asked_count",0), jdump(_state_to_jsonable(sess.get("state"))), sess.get("pending_item"), jdump(_to_jsonable(sess.get("items_asked",[]))), jdump(_to_jsonable(sess.get("trace",[]))), jdump(_to_jsonable(sess.get("result"))), jdump(_to_jsonable(sess.get("explanation"))), jdump(_to_jsonable(sess.get("support_report"))), sess.get("stop_reason"))
        )


def _record_for_index(i: int):
    ensure_ready()
    test = _STATE["test"]
    return test[i % len(test)]


def start_session(mode: str, policy_name: str) -> dict:
    ensure_ready()
    if mode not in ("interactive", "auto"):
        raise ValueError("mode must be interactive|auto")
    if policy_name not in ("greedy", "random", "beta_greedy"):
        raise ValueError("unsupported policy")
    with _LOCK:
        idx = _STATE["counter"] % len(_STATE["test"])
        _STATE["counter"] += 1
    rec = _record_for_index(idx)
    state = init_state(rec)
    state["questions_remaining"] = BUDGET_DEFAULT
    state["budget"] = BUDGET_DEFAULT
    sess = {
        "id": new_id("ses_"), "created_at": utc_now(), "status": "active",
        "policy": policy_name, "mode": mode, "budget": BUDGET_DEFAULT, "tau": TAU_DEFAULT,
        "record_index": idx, "true_label": int(rec["label"]), "asked_count": 0,
        "state": state, "pending_item": None, "items_asked": [], "trace": [],
        "result": None, "explanation": None, "support_report": None, "stop_reason": None,
    }
    policy = make_policy(policy_name)
    if mode == "auto":
        ep = run_episode(rec, BUDGET_DEFAULT, 0, policy, predict, 0.0, tau=TAU_DEFAULT)
        cf = find_counterfactual(ep["final_state"], ep["p_hat"], predict, tau=TAU_DEFAULT)
        decision = "REFERRAL_RECOMMENDED" if ep["p_hat"] >= TAU_DEFAULT else "NO_REFERRAL_INDICATED"
        X_train = np.array([np.asarray(r["item_responses"], dtype=float) for r in (_STATE["train"] or [])])
        explanation = explain_outcome(ep, _STATE["predictor"], tau=TAU_DEFAULT,
                                      costs=np.ones(N_ITEMS, dtype=float), prior=None,
                                      reference_rows=X_train if len(X_train) else None,
                                      dataset_tag="saudi", circularity=audit_circularity(_STATE["records"]).get("classification"),
                                      label_source=rec.get("label_source"), budget=BUDGET_DEFAULT,
                                      tag="app session — research prototype, not a diagnostic device")
        from src.support.report import build_support_report
        from src.support.schemas import to_json_safe
        ep_for_support = {**ep, "p_hat": float(ep["p_hat"]), "decision": decision}
        report = build_support_report(episode_result=ep_for_support, explanation=explanation,
                                      result_id=sess["id"], model_name="MaskedPredictor",
                                      model_version=str(MaskedPredictor.VERSION), input_reference="app-session")
        result = {"p_hat": float(ep["p_hat"]), "decision": decision, "asked": len(ep["items_asked"]),
                  "items_asked": [f"A{j + 1}" for j in ep["items_asked"]], "trace": ep["trace"],
                  "counterfactual": cf, "true_label": int(rec["label"]), "stop_reason": ep["stop_reason"],
                  "explanation": explanation, "support_report": to_json_safe(report)}
        sess.update({"status": "completed", "state": ep["final_state"], "trace": ep["trace"],
                     "items_asked": [int(x) for x in ep["items_asked"]], "asked_count": len(ep["items_asked"]),
                     "result": result, "explanation": explanation, "support_report": to_json_safe(report),
                     "stop_reason": ep["stop_reason"]})
        _persist_session(sess)
        return {"session_id": sess["id"], "mode": mode, "budget": BUDGET_DEFAULT, "tau": TAU_DEFAULT, "policy": policy_name, "result": result}
    else:
        legal = get_legal_items(state) + [STOP]
        j = policy(state, legal)
        sess["pending_item"] = int(j)
        sel = selection_diagnostic(policy, state, legal)
        _persist_session(sess)
        return {"session_id": sess["id"], "mode": mode, "budget": BUDGET_DEFAULT, "tau": TAU_DEFAULT, "policy": policy_name,
                "question": item_payload(j), "belief": predict(state), "asked": 0, "selection": sel}


def _load_session(session_id: str) -> dict:
    row = _session_row(session_id)
    if not row:
        raise ValueError("unknown session - start a new one")
    row["state"] = _state_from_jsonable(jload(row.get("state_json")))
    row["items_asked"] = jload(row.get("items_asked_json"), [])
    row["trace"] = jload(row.get("trace_json"), [])
    row["result"] = jload(row.get("result_json"))
    row["explanation"] = jload(row.get("explanation_json"))
    row["support_report"] = jload(row.get("support_report_json"))
    return row


def answer_session(session_id: str, value: Optional[int] = None, stop: bool = False) -> dict:
    ensure_ready()
    s = _load_session(session_id)
    if s["status"] == "completed":
        return {"finished": True, "result": s["result"]}
    rec = _record_for_index(int(s["record_index"]))
    state = s["state"]
    policy = make_policy(s["policy"])
    asked = int(s["asked_count"])
    trace = list(s.get("trace") or [])
    items_asked = list(s.get("items_asked") or [])
    pending = s.get("pending_item")

    def finish(reason: str):
        p = predict(state)
        decision = "REFERRAL_RECOMMENDED" if p >= TAU_DEFAULT else "NO_REFERRAL_INDICATED"
        cf = find_counterfactual(state, p, predict, tau=TAU_DEFAULT)
        ep_result = {"p_hat": p, "decision": decision, "trace": trace, "items_asked": items_asked,
                     "stop_reason": reason, "final_state": state}
        X_train = np.array([np.asarray(r["item_responses"], dtype=float) for r in (_STATE["train"] or [])])
        explanation = explain_outcome(ep_result, _STATE["predictor"], tau=TAU_DEFAULT,
                                      costs=np.ones(N_ITEMS, dtype=float), prior=None,
                                      reference_rows=X_train if len(X_train) else None,
                                      dataset_tag="saudi", circularity=audit_circularity(_STATE["records"]).get("classification"),
                                      label_source=rec.get("label_source"), budget=BUDGET_DEFAULT,
                                      tag="app session — research prototype, not a diagnostic device")
        from src.support.report import build_support_report
        from src.support.schemas import to_json_safe
        report = build_support_report(episode_result=ep_result, explanation=explanation,
                                      result_id=session_id, model_name="MaskedPredictor",
                                      model_version=str(MaskedPredictor.VERSION), input_reference="app-session")
        result = {"p_hat": p, "decision": decision, "asked": asked, "items_asked": [f"A{j + 1}" for j in items_asked],
                  "trace": trace, "counterfactual": cf, "true_label": int(rec["label"]), "stop_reason": reason,
                  "explanation": explanation, "support_report": to_json_safe(report)}
        s.update({"status": "completed", "state": state, "pending_item": None, "items_asked": items_asked,
                  "trace": trace, "asked_count": asked, "result": result, "explanation": explanation,
                  "support_report": to_json_safe(report), "stop_reason": reason})
        _persist_session(s)
        return result

    if stop:
        if asked < 1:
            raise ValueError("answer at least one question before stopping")
        return {"finished": True, "result": finish("user_stop")}
    if value not in (0, 1):
        raise ValueError("value must be 0 or 1")
    if pending is None:
        raise ValueError("no pending question")
    j = int(pending)
    p_before = predict(state)
    qr = state.get("questions_remaining", BUDGET_DEFAULT) - 1
    sel = selection_diagnostic(policy, state, get_legal_items(state) + [STOP])
    state = update_state(state, j, int(value), qr)
    state["questions_remaining"] = qr
    state["budget"] = BUDGET_DEFAULT
    asked += 1
    items_asked.append(j)
    p_after = predict(state)
    step = {"step": asked, "item": f"A{j + 1}", "value": int(value), "belief_before": p_before, "belief_after": p_after, "selection": sel}
    trace.append(step)
    if asked >= BUDGET_DEFAULT or not get_legal_items(state):
        return {"finished": True, "result": finish("budget_exhausted")}
    nj = int(policy(state, get_legal_items(state)))
    s.update({"state": state, "pending_item": nj, "items_asked": items_asked, "trace": trace, "asked_count": asked, "status": "active"})
    _persist_session(s)
    next_sel = selection_diagnostic(policy, state, get_legal_items(state) + [STOP])
    return {"finished": False, "belief_before": p_before, "belief_after": p_after, "trace_step": step,
            "asked": asked, "next": item_payload(nj), "selection": next_sel}


def list_sessions(limit: int = 20) -> list:
    conn = get_conn()
    with _LOCK:
        rows = conn.execute("SELECT id,created_at,updated_at,status,policy,mode,asked_count,stop_reason FROM screening_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    return rows_to_dicts(rows)


def get_session(session_id: str) -> dict:
    s = _load_session(session_id)
    # return safe summary for resume/UI
    out = {k: s.get(k) for k in ("id","created_at","updated_at","status","policy","mode","budget","tau","asked_count","stop_reason")}
    out["pending_question"] = item_payload(int(s["pending_item"])) if s.get("pending_item") is not None and s.get("status")=="active" else None
    out["trace"] = s.get("trace") or []
    out["result"] = s.get("result")
    out["belief"] = predict(s["state"]) if s.get("state") else None
    return out
