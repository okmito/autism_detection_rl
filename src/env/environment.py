"""Episode loop — §10"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Callable, Tuple, List
from .state import UNASKED, OBSERVED, MISSING, init_state, update_state, get_legal_items
from .costs import get_costs

STOP = -1  # sentinel for STOP action (n is max item index)

def run_episode(
    record: Dict[str, Any],
    question_budget: int,
    b_min: int,
    policy: Callable[[Dict[str, Any], List[int]], int],
    predictor: Callable[[Dict[str, Any]], float],
    lambda_cost: float = 0.0,
    cost_mode: str = "uniform",
    cost_weights: list[float] | None = None,
    tau: float = 0.5,
) -> Dict[str, Any]:
    """Run one episode per §10 RUN_EPISODE pseudocode."""
    n = len(record["item_responses"])
    costs = get_costs(n, mode=cost_mode, weights=cost_weights)
    state = init_state(record)
    state["questions_remaining"] = question_budget
    state["budget"] = question_budget
    questions_remaining = question_budget
    items_asked: List[int] = []
    trace: List[Dict[str, Any]] = []

    # Keep raw values for observation
    raw_values = record["item_responses"]

    while True:
        legal_items = get_legal_items(state)
        stop_legal = (len(items_asked) >= b_min)
        if not legal_items:
            stop_legal = True
        legal = legal_items.copy()
        if stop_legal:
            legal.append(STOP)

        if questions_remaining == 0:
            action = STOP
        else:
            # policy receives state dict and legal list
            action = policy(state, legal)

        if action == STOP:
            p_hat = float(predictor(state))
            p_hat = float(np.clip(p_hat, 0.0, 1.0))
            y = int(record["label"])
            asked_cost = sum(costs[j] for j in items_asked)
            R = (1 - (p_hat - y) ** 2) - lambda_cost * asked_cost
            decision = "REFER" if p_hat >= tau else "NO_REFERRAL_INDICATED"
            stop_reason = "policy_stop" if stop_legal else "budget_exhausted"
            # counterfactual placeholder — filled by caller if needed
            return {
                "decision": decision,
                "p_hat": p_hat,
                "trace": trace,
                "R": float(R),
                "items_asked": items_asked.copy(),
                "stop_reason": stop_reason,
                "final_state": state,
            }

        # Validate action
        if action not in legal_items:
            raise ValueError(f"Illegal action {action}, legal={legal_items}, mask={state['mask']}")

        belief_before = float(predictor(state))
        # observed value — raw item response truncated to int for binary; preserve category for multi
        v_raw = raw_values[action]
        # For NaN cases should not happen because MISSING not legal
        if np.isnan(v_raw):
            raise ValueError(f"Observed NaN for legal item {action}")
        v = int(v_raw) if not np.isnan(v_raw) else 0

        state = update_state(state, action, v, questions_remaining - 1)
        state["questions_remaining"] = questions_remaining - 1
        state["budget"] = question_budget
        questions_remaining -= 1
        items_asked.append(action)
        belief_after = float(predictor(state))
        trace.append({
            "step": len(trace) + 1,
            "item": f"A{action+1}",
            "item_idx": action,
            "value": v,
            "belief_before": float(belief_before),
            "belief_after": float(belief_after),
        })

        # Safety: budget exhaustion forces STOP next loop
        if questions_remaining == 0:
            continue
