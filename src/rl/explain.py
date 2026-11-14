"""Explanations of individual ASK / STOP decisions, computed from the acting agent's own values.

Faithfulness contract
---------------------
* ``action_values`` are the Q-values the agent used, and ``action`` is their argmax under the same tie rule
  (``src/rl/planner.greedy_action``). Tests assert this for every decision, so an explanation can never describe
  an action the agent would not take.
* For the value-iteration agent every statement about *why* is exact: ``Q(ask j)`` is decomposed into its two
  answer branches (probability of each answer, belief after it, value after it, question cost), and those numbers
  reconstruct the Q-value to machine precision.
* For the DQN agent the Q-values are the network's estimates; the answer-branch numbers come from the belief model
  and are labelled ``source = "belief_model"``. They describe the information the question offers, not the network's
  internal computation, and ``faithful_to_agent_internals`` says so.

Value units: Q-values are the expected screening-decision accuracy (agreement with the clinician reference label
the agent was trained against) minus ``cost`` per question. "Points" below are those units x 100.

Every narrative sentence is filled from these numbers by a fixed template; nothing is generated free-form.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import C_NEG, C_POS, C_SKIP, C_UNASKED, MDPConfig, legal_mask, obs_val
from src.rl.planner import greedy_action

SCHEMA_VERSION = "rl-decision-explanation/1.0"
REFERRAL, NO_REFERRAL = "REFERRAL_RECOMMENDED", "NO_REFERRAL_INDICATED"


def _code(j: int) -> str:
    return f"A{j + 1}"


def _entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _decision(p: float, tau: float) -> str:
    return REFERRAL if p >= tau else NO_REFERRAL


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def information_offer(belief: NaiveBayesBelief, codes: np.ndarray, j: int, tau: float) -> Dict[str, Any]:
    """What asking item j can tell us, under the belief model (exact for that model)."""
    o, v = obs_val(codes)
    lo = float(belief.log_odds_arrays(o, v))
    p = 1 / (1 + np.exp(-lo))
    a1 = float(belief.theta[j, 1] * p + belief.theta[j, 0] * (1 - p))
    p1 = float(1 / (1 + np.exp(-(lo + belief.w * belief.llr1[j]))))
    p0 = float(1 / (1 + np.exp(-(lo + belief.w * belief.llr0[j]))))
    now = _decision(p, tau)
    flip1, flip0 = _decision(p1, tau) != now, _decision(p0, tau) != now
    return {"item": _code(j), "item_index": int(j), "source": "belief_model",
            "p_answer_atypical": a1, "p_after_atypical": p1, "p_after_typical": p0,
            "decision_if_atypical": _decision(p1, tau), "decision_if_typical": _decision(p0, tau),
            "could_change_decision": bool(flip1 or flip0),
            "p_decision_changes": float(a1 * flip1 + (1 - a1) * flip0),
            "expected_entropy_reduction_bits": _entropy(p) - (a1 * _entropy(p1) + (1 - a1) * _entropy(p0))}


def reachable_range(belief: NaiveBayesBelief, codes: np.ndarray, cfg: MDPConfig) -> Dict[str, float]:
    """Lowest / highest belief reachable within the remaining budget (each remaining answer chosen adversarially)."""
    codes = np.asarray(codes)
    o, v = obs_val(codes)
    lo = float(belief.log_odds_arrays(o, v))
    left = max(0, cfg.max_questions - int(np.count_nonzero(codes)))
    un = np.flatnonzero(codes == C_UNASKED)
    up = np.sort(np.maximum(belief.llr1[un], belief.llr0[un]) * belief.w)[::-1][:left]
    down = np.sort(np.minimum(belief.llr1[un], belief.llr0[un]) * belief.w)[:left]
    return {"p_min_reachable": float(1 / (1 + np.exp(-(lo + down[down < 0].sum())))),
            "p_max_reachable": float(1 / (1 + np.exp(-(lo + up[up > 0].sum())))),
            "questions_left_in_budget": int(min(left, len(un)))}


def answered_evidence(belief: NaiveBayesBelief, codes: np.ndarray, item_texts: Optional[Sequence[str]] = None
                      ) -> List[Dict[str, Any]]:
    out = []
    for j in np.flatnonzero(np.asarray(codes) != C_UNASKED):
        c = int(codes[j])
        if c == C_SKIP:
            out.append({"item": _code(j), "answer": "skipped", "log_odds_shift": 0.0, "direction": "none"})
            continue
        eff = belief.item_effect(int(j), 1 if c == C_POS else 0)
        row = {"item": _code(j), "answer": "atypical" if c == C_POS else "typical",
               "log_odds_shift": eff["log_odds_shift"], "odds_multiplier": eff["odds_multiplier"],
               "direction": "towards referral" if eff["log_odds_shift"] > 0 else "away from referral"}
        if item_texts is not None:
            row["text"] = item_texts[j]
        out.append(row)
    return sorted(out, key=lambda r: -abs(r["log_odds_shift"]))


def explain_decision(agent, codes: np.ndarray, *, stop_reason: Optional[str] = None,
                     item_texts: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Explain the action ``agent`` takes at history ``codes`` (shape (n,))."""
    codes = np.asarray(codes, dtype=np.int64)
    cfg: MDPConfig = agent.cfg
    belief: NaiveBayesBelief = agent.belief
    n = cfg.n_items
    Q = np.asarray(agent.q_values(codes[None]), float)[0]
    lm = legal_mask(codes[None], cfg)[0]
    action = int(greedy_action(Q[None])[0])
    forced = not lm[:-1].any()
    o, v = obs_val(codes)
    p = float(belief.prob_arrays(o, v))
    used = int(np.count_nonzero(codes))
    exact = hasattr(agent, "decompose")
    values = [{"action": "STOP" if a == n else _code(a), "q": float(Q[a])} for a in range(n + 1) if lm[a]]
    values.sort(key=lambda r: -r["q"])
    base = {
        "schema": SCHEMA_VERSION, "agent": getattr(agent, "name", type(agent).__name__),
        "faithful_to_agent_internals": bool(exact),
        "value_units": "expected screening-decision accuracy minus cost per question",
        "state": {"questions_used": used, "max_questions": cfg.max_questions, "min_questions": cfg.min_questions,
                  "p_positive": p, "decision_if_stopped_now": _decision(p, cfg.tau), "tau": cfg.tau,
                  "cost_per_question": cfg.cost},
        "action_values": values,
    }
    ask_q = np.where(lm[:-1], Q[:-1], -np.inf)
    if action != n:
        j = action
        offer = information_offer(belief, codes, j, cfg.tau)
        order = [a for a in np.argsort(-ask_q, kind="stable") if np.isfinite(ask_q[a])]
        runner = next((int(a) for a in order if a != j), None)
        stop_legal = bool(lm[-1])
        exp: Dict[str, Any] = {**base, "action": {"type": "ASK", "item": _code(j), "item_index": j},
                               "chosen_value": float(Q[j]), "information": offer,
                               "stop_was_allowed": stop_legal,
                               "margin_over_stop": float(Q[j] - Q[n]) if stop_legal else None}
        if exact:
            dec = agent.decompose(codes, j)
            dec["source"] = "agent (exact Bellman decomposition)"
            exp["value_decomposition"] = dec
        if runner is not None:
            r_off = information_offer(belief, codes, runner, cfg.tau)
            exp["runner_up"] = {"item": _code(runner), "item_index": runner, "value": float(Q[runner]),
                                "margin": float(Q[j] - Q[runner]), "information": r_off}
        exp["narrative"] = _ask_narrative(exp, item_texts)
        return exp

    best = int(np.argmax(ask_q)) if np.isfinite(ask_q).any() else None
    reason = stop_reason or ("budget_exhausted" if used >= cfg.max_questions
                             else "no_items_left" if forced else "policy_stop")
    exp = {**base, "action": {"type": "STOP"}, "stop_reason": reason, "forced": bool(forced),
           "stop_value": float(Q[n]),
           "evidence": answered_evidence(belief, codes, item_texts),
           "reachable_belief": reachable_range(belief, codes, cfg),
           "remaining_items": [_code(k) for k in np.flatnonzero(codes == C_UNASKED)]}
    if best is not None:
        exp["best_remaining_question"] = {"item": _code(best), "item_index": best, "value": float(Q[best]),
                                          "shortfall_vs_stop": float(Q[n] - Q[best]),
                                          "information": information_offer(belief, codes, best, cfg.tau)}
        if exact:
            dec = agent.decompose(codes, best)
            dec["source"] = "agent (exact Bellman decomposition)"
            exp["best_remaining_question"]["value_decomposition"] = dec
    exp["remaining_uncertainty"] = {
        "p_positive": p, "distance_to_threshold": abs(p - cfg.tau),
        "p_wrong_side_under_model": float(min(p, 1 - p)),
        "note": "Probability that the reference label differs from the current decision, under the belief model."}
    exp["limitations"] = _limitations()
    exp["narrative"] = _stop_narrative(exp)
    return exp


def _limitations() -> List[str]:
    return [
        "Screening estimate, not a diagnosis; a clinical assessment is needed to establish or rule out autism.",
        "The agent was trained on one case-control cohort (Polish toddlers, 18-24 months, n=252) where about half "
        "the children had an ASD diagnosis; in a general population far fewer children do, so the estimate is not "
        "a population risk.",
        "Probabilities come from a naive-Bayes model that treats answers as independent given the label, corrected "
        "by one cross-fitted evidence weight.",
        "Stopping early means some Q-CHAT-10 questions were not asked; the full questionnaire remains available.",
    ]


def _ask_narrative(e: Dict[str, Any], item_texts) -> List[str]:
    s, info = e["state"], e["information"]
    item = e["action"]["item"]
    lines = [f"Current estimate {_pct(s['p_positive'])} after {s['questions_used']} question(s); stopping now "
             f"would give '{s['decision_if_stopped_now'].replace('_', ' ').lower()}'."]
    lines.append(f"{item} was chosen because it had the highest value of all questions the agent could ask "
                 f"(value {e['chosen_value']:.3f} = expected chance the final screening decision matches the clinical "
                 f"reference, minus {s['cost_per_question']:.3f} for each further question expected).")
    lines.append(f"If answered atypical the estimate would move to {_pct(info['p_after_atypical'])}; if typical, to "
                 f"{_pct(info['p_after_typical'])} (atypical answer expected with probability "
                 f"{_pct(info['p_answer_atypical'])}).")
    lines.append("Either answer could change the screening outcome."
                 if info["could_change_decision"] else
                 "Neither single answer changes the outcome by itself; the agent asks it because it positions the "
                 "following questions to resolve the outcome.")
    if e["stop_was_allowed"]:
        lines.append(f"Asking was worth {100 * e['margin_over_stop']:.1f} points more than stopping, after the "
                     f"per-question cost of {100 * s['cost_per_question']:.1f} points.")
    else:
        lines.append(f"Stopping is not allowed before {s['min_questions']} questions (safety minimum).")
    if "runner_up" in e:
        r = e["runner_up"]
        lines.append(f"The next-best question, {r['item']}, was worth {100 * r['margin']:.2f} points less "
                     f"(could change the outcome: {'yes' if r['information']['could_change_decision'] else 'no'}).")
    return lines


def _stop_narrative(e: Dict[str, Any]) -> List[str]:
    s = e["state"]
    lines = []
    if e["stop_reason"] == "budget_exhausted":
        lines.append(f"Stopped because the maximum of {s['max_questions']} questions was reached (safety cap), not "
                     "because the agent judged the information sufficient.")
    elif e["stop_reason"] == "no_items_left":
        lines.append("Stopped because every question had been asked.")
    else:
        lines.append(f"The agent chose to stop after {s['questions_used']} question(s): stopping was worth more than "
                     "asking any remaining question.")
    b = e.get("best_remaining_question")
    if b and e["stop_reason"] == "policy_stop":
        lines.append(f"The most useful remaining question, {b['item']}, would change the outcome with probability "
                     f"{_pct(b['information']['p_decision_changes'])}, which did not repay its cost "
                     f"(short by {100 * b['shortfall_vs_stop']:.2f} points).")
    lines.append(f"Final estimate {_pct(s['p_positive'])} -> "
                 f"'{s['decision_if_stopped_now'].replace('_', ' ').lower()}' (threshold {_pct(s['tau'])}).")
    rr = e["reachable_belief"]
    if rr["questions_left_in_budget"]:
        lines.append(f"Within the remaining budget the estimate could range from {_pct(rr['p_min_reachable'])} to "
                     f"{_pct(rr['p_max_reachable'])} in the most extreme answer patterns.")
    ev = [x for x in e["evidence"] if x["direction"] != "none"][:3]
    if ev:
        lines.append("Answers that moved the estimate most: " +
                     "; ".join(f"{x['item']} ({x['answer']}, {x['direction']})" for x in ev) + ".")
    lines.append("This is a screening estimate, not a diagnosis.")
    return lines
