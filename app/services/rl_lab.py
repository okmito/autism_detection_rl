from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from src.data.ingest import load_dataset
from src.data.splits import stratified_split
from src.env.state import init_state, update_state, get_legal_items
from src.env.environment import STOP
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy, entropy as _entropy
from src.policies.random_policy import RandomPolicy
from src.policies.random_fixed import RandomFixedLengthPolicy

from app.errors import BadRequest
from src.screening.session import decision_for
from app.config import N_ITEMS, SEED, DEMO_CALIBRATION, BUDGET_DEFAULT, TAU_DEFAULT, ROOT as _ROOT
from app.services.screening import ensure_ready, predict as screening_predict, _STATE as SCREEN_STATE


def synthetic_scenarios() -> List[dict]:
    # Reproducible, clearly labeled synthetic demonstration cases (not real user data).
    # Constructed Q-CHAT-like binary response patterns.
    cases = [
        {"id": "syn_typical", "name": "Synthetic: mostly typical responses", "responses": [0,0,0,0,0,0,0,0,0,0], "label": 0},
        {"id": "syn_mixed", "name": "Synthetic: mixed responses", "responses": [1,0,1,0,0,1,0,1,0,0], "label": 0},
        {"id": "syn_elevated", "name": "Synthetic: many atypical responses", "responses": [1,1,1,0,1,1,0,1,1,1], "label": 1},
        {"id": "syn_borderline", "name": "Synthetic: borderline pattern", "responses": [1,0,1,0,1,0,1,0,0,1], "label": 1},
    ]
    out = []
    for c in cases:
        out.append({"id": c["id"], "name": c["name"], "note": "Synthetic demonstration case (reproducible). Not a real person and not clinical evidence."})
    return out


def _case_record(case_id: str) -> dict:
    mapping = {
        "syn_typical": ([0,0,0,0,0,0,0,0,0,0], 0),
        "syn_mixed": ([1,0,1,0,0,1,0,1,0,0], 0),
        "syn_elevated": ([1,1,1,0,1,1,0,1,1,1], 1),
        "syn_borderline": ([1,0,1,0,1,0,1,0,0,1], 1),
    }
    if case_id not in mapping:
        raise BadRequest("unknown scenario")
    resp, label = mapping[case_id]
    return {
        "item_responses": np.array(resp, dtype=float),
        "missing_mask": np.zeros(N_ITEMS, dtype=bool),
        "label": int(label),
        "label_source": "synthetic_demo",
    }


def _make_policy(pid: str):
    ensure_ready()
    train = SCREEN_STATE["train"]
    if pid == "greedy":
        return GreedyIGPolicy(train, N_ITEMS), "real"
    if pid == "random":
        return RandomPolicy(seed=SEED), "real"
    if pid == "random_fixed":
        return RandomFixedLengthPolicy(seed=SEED), "real"
    if pid == "beta_greedy":
        from src.policies.beta_greedy import BetaGreedyPolicy
        return BetaGreedyPolicy(train, N_ITEMS), "real"
    raise BadRequest(f"policy '{pid}' not available for live lab replay in this build")


def _legal_with_stop(state, asked, b_min=0):
    items = get_legal_items(state)
    stop_legal = (asked >= b_min) or (not items)
    legal = list(items)
    if stop_legal:
        legal.append(STOP)
    return legal


def replay(case_id: str, policy_id: str, budget: int = BUDGET_DEFAULT) -> dict:
    ensure_ready()
    rec = _case_record(case_id)
    budget = max(1, min(int(budget), N_ITEMS))
    try:
        policy, mode = _make_policy(policy_id)
    except BadRequest as e:  # an unknown/unavailable policy is a normal, displayable outcome in the lab
        return {"ok": False, "available": False, "error": str(e)}
    state = init_state(rec)
    state["questions_remaining"] = budget
    state["budget"] = budget
    asked = 0
    trace = []
    steps = []
    # initial snapshot
    steps.append({
        "step": 0,
        "phase": "state",
        "state": {"mask": state["mask"].tolist(), "value": state["value"].tolist()},
        "belief": screening_predict(state),
        "questions_remaining": budget,
        "available_actions": [f"A{j+1}" for j in get_legal_items(state)] + ["STOP"],
        "note": "Initial state (no responses observed)."
    })
    while True:
        legal = _legal_with_stop(state, asked)
        items = [a for a in legal if a != STOP]
        budget_exhausted = (state["questions_remaining"] == 0) or (not items)
        if budget_exhausted:
            action = STOP
            forced = True
        else:
            action = policy(state, legal)
            forced = False
        # policy evaluation snapshot (scores when available)
        eval_info = {"available_actions": ["STOP" if a==STOP else f"A{a+1}" for a in legal], "chosen": "STOP" if action==STOP else f"A{action+1}", "forced": forced}
        if hasattr(policy, "_support") and items:
            # compute IG scores summary for transparency
            idx = policy._support(state["mask"], state["value"])
            eval_info["support_size"] = int(len(idx))
            if len(idx) > 0:
                ys = policy.y[idx]
                p_s = float(ys.mean())
                eval_info["posterior_support"] = p_s
        steps.append({"step": asked, "phase": "decision", "evaluation": eval_info, "belief": screening_predict(state), "questions_remaining": state["questions_remaining"]})
        if action == STOP:
            p_hat = screening_predict(state)
            steps.append({"step": asked, "phase": "stop", "p_hat": p_hat, "decision": decision_for(p_hat, TAU_DEFAULT),
                          "stop_reason": "budget_exhausted" if budget_exhausted else "policy_stop"})
            break
        j = int(action)
        v = int(rec["item_responses"][j])
        p_before = screening_predict(state)
        # update_state already records the decremented budget; decrementing again here used to burn two
        # questions per answer, so the lab replayed half the budget it displayed.
        state = update_state(state, j, v, state["questions_remaining"] - 1)
        state["budget"] = budget
        asked += 1
        p_after = screening_predict(state)
        trace.append({"step": asked, "item": f"A{j+1}", "value": v, "belief_before": p_before, "belief_after": p_after})
        steps.append({"step": asked, "phase": "update", "observed": {"item": f"A{j+1}", "value": v}, "belief_before": p_before, "belief_after": p_after,
                      "state": {"mask": state["mask"].tolist(), "value": state["value"].tolist()}, "questions_remaining": state["questions_remaining"]})
    final_p = steps[-1].get("p_hat", screening_predict(state))
    return {
        "ok": True, "available": True, "mode": mode, "scenario": case_id, "policy": policy_id, "budget": budget,
        "tau": TAU_DEFAULT, "synthetic": True,
        "summary": {"questions_asked": asked, "final_p_hat": final_p,
                    "decision": decision_for(final_p, TAU_DEFAULT),
                    "stop_reason": steps[-1].get("stop_reason")},
        "trace": trace, "steps": steps,
        "disclaimer": "Synthetic demonstration replay using real policy/inference logic where available. Not a clinical result."
    }


def compare(case_id: str, policy_ids: List[str], budget: int = BUDGET_DEFAULT) -> dict:
    results = []
    for pid in policy_ids:
        results.append(replay(case_id, pid, budget))
    return {"scenario": case_id, "budget": budget, "results": results, "synthetic": True}
