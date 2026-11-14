"""One implementation of a Q-CHAT-10 adaptive screening session (audit finding T1).

The web app (``app/services/screening.py``, persisted in SQLite) and the demo server
(``scripts/demo_app.py``, in memory) each used to carry their own copy of "ask a question, record the
answer, finish, explain, build the support report". The copies drifted (C4 was fixed in one only).
They now both call these functions; the callers keep only what really differs: where session state is
stored and which dataset record, if any, a session is bound to.

Everything here is pure with respect to storage: no database, no HTTP, no globals. The predictor and
training data arrive in a :class:`Context`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.env.environment import STOP, run_episode
from src.env.state import get_legal_items, init_state, update_state
from src.explain.counterfactual import find_counterfactual
from src.explain.outcome import explain_outcome
from src.audits.circularity import audit_circularity
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
from src.screening.diagnostics import selection_diagnostic, selection_summary
from src.support.report import build_support_report
from src.support.schemas import to_json_safe

REFERRAL = "REFERRAL_RECOMMENDED"
NO_REFERRAL = "NO_REFERRAL_INDICATED"


@dataclass
class Context:
    """Everything a session needs from its host that is not session state."""
    predict: Callable[[dict], float]          # state -> calibrated P(label=1)
    predictor: Any                            # the MaskedPredictor behind ``predict`` (for explanations)
    train: Sequence[dict]                     # training records (policy support + counterfactual reference rows)
    records: Sequence[dict]                   # all records (circularity audit)
    n_items: int
    budget: int
    tau: float
    tag: str                                  # human-readable provenance tag stored with the explanation
    input_reference: str                      # e.g. "app-session" / "demo-session"
    dataset_tag: str = "saudi"                # development dataset named in the limitations
    model_name: str = "MaskedPredictor"       # scorer named in the support report
    model_version: str = str(MaskedPredictor.VERSION)


def decision_for(p_hat: float, tau: float) -> str:
    return REFERRAL if p_hat >= tau else NO_REFERRAL


def blank_state(n_items: int, budget: int) -> dict:
    """An all-unasked state with no dataset record behind it (what a real respondent starts from)."""
    state = init_state({"item_responses": np.full(n_items, np.nan), "missing_mask": np.zeros(n_items, dtype=bool)})
    state["questions_remaining"] = budget
    state["budget"] = budget
    return state


def record_state(record: dict, budget: int) -> dict:
    """A state seeded from a dataset record's own missingness (replay/auto mode only)."""
    state = init_state(record)
    state["questions_remaining"] = budget
    state["budget"] = budget
    return state


def require_legal_item(j, legal, n_items: int) -> int:
    """A question to ask must be a real, currently-unasked item index (never STOP / -1).

    ``RandomPolicy`` draws uniformly from ``legal``; offering STOP as an *opening* action yielded index
    -1, rendered as "A0" with the text of item A10 (audit finding C4). A plain ``ValueError`` is
    deliberate: a policy returning this is a bug, not a client mistake.
    """
    j = int(j)
    if j not in legal or not (0 <= j < n_items):
        raise ValueError(f"policy selected an invalid question index {j}")
    return j


def next_question(ctx: Context, policy, state: dict) -> Tuple[int, dict]:
    """Ask the policy for the next item. Returns (item index, selection diagnostic)."""
    legal = get_legal_items(state)
    j = require_legal_item(policy(state, legal), legal, ctx.n_items)
    return j, selection_diagnostic(policy, state, legal)


def apply_answer(ctx: Context, policy, state: dict, j: int, value: int, asked_before: int) -> Tuple[dict, dict]:
    """Record ``value`` for item ``j``. Returns (new state, trace step).

    The diagnostic in the step explains the choice that was *just made*, evaluated at the state it was
    made from, so the trace explains each pick rather than the next one.
    """
    p_before = ctx.predict(state)
    remaining = state.get("questions_remaining", ctx.budget) - 1
    sel = selection_diagnostic(policy, state, get_legal_items(state) + [STOP])
    new_state = update_state(state, j, int(value), remaining)
    new_state["questions_remaining"] = remaining
    new_state["budget"] = ctx.budget
    step = {"step": asked_before + 1, "item": f"A{j + 1}", "value": int(value),
            "belief_before": p_before, "belief_after": ctx.predict(new_state), "selection": sel}
    return new_state, step


def is_exhausted(ctx: Context, state: dict, asked: int) -> bool:
    return asked >= ctx.budget or not get_legal_items(state)


def finalize(ctx: Context, *, state: dict, trace: List[dict], items_asked: Sequence[int], stop_reason: str,
             result_id: str, label_source: Optional[str] = None, true_label: Optional[int] = None,
             p_hat: Optional[float] = None) -> Dict[str, Any]:
    """Counterfactual + explanation + support report + the result dict, from a finished session.

    ``p_hat`` may be passed when the caller already has it (``run_episode``); otherwise it is predicted
    from ``state``. Returns ``{"result", "explanation", "support_report", "episode_result"}``.
    """
    p = float(ctx.predict(state) if p_hat is None else p_hat)
    decision = decision_for(p, ctx.tau)
    items = [int(j) for j in items_asked]
    episode = {"p_hat": p, "decision": decision, "trace": trace, "items_asked": items,
               "stop_reason": stop_reason, "final_state": state}
    counterfactual = find_counterfactual(state, p, ctx.predict, tau=ctx.tau)
    train_rows = np.array([np.asarray(r["item_responses"], dtype=float) for r in ctx.train])
    circularity = audit_circularity(list(ctx.records)).get("classification") if ctx.records else None
    explanation = explain_outcome(
        episode, ctx.predictor, tau=ctx.tau, costs=np.ones(ctx.n_items, dtype=float), prior=None,
        reference_rows=train_rows if len(train_rows) else None, dataset_tag=ctx.dataset_tag,
        circularity=circularity, label_source=label_source, budget=ctx.budget, tag=ctx.tag)
    report = to_json_safe(build_support_report(
        episode_result=episode, explanation=explanation, result_id=result_id, model_name=ctx.model_name,
        model_version=ctx.model_version, input_reference=ctx.input_reference))
    result = {"p_hat": p, "decision": decision, "asked": len(items), "items_asked": [f"A{j + 1}" for j in items],
              "trace": trace, "counterfactual": counterfactual, "stop_reason": stop_reason,
              "selection_summary": selection_summary(trace), "explanation": explanation, "support_report": report}
    if true_label is not None:
        result["true_label"] = int(true_label)
    return {"result": result, "explanation": explanation, "support_report": report, "episode_result": episode}


def replayed_diagnostics(ctx: Context, policy, record: dict, items_asked: Sequence[int]) -> List[dict]:
    """Re-walk a realised item sequence, diagnosing each decision point (greedy policy only).

    Used for auto/replay runs: ``run_episode`` keeps its own trace and the environment is shared with
    the RL trainers, so the per-step diagnostic is attached afterwards instead of changing it. Read-only
    with respect to the episode.
    """
    if not isinstance(policy, GreedyIGPolicy):
        return []
    state = record_state(record, ctx.budget)
    out: List[dict] = []
    for depth, a in enumerate(items_asked):
        sel = selection_diagnostic(policy, state, get_legal_items(state) + [STOP])
        out.append({"step": depth + 1, "item": f"A{a + 1}", "selection": sel})
        value = record["item_responses"][a]
        if np.isnan(value):
            break
        state = update_state(state, a, int(value), ctx.budget - depth - 1)
        state["budget"] = ctx.budget
        state["questions_remaining"] = ctx.budget - depth - 1
    return out


def run_auto(ctx: Context, policy, record: dict, *, result_id: str) -> Dict[str, Any]:
    """Replay a dataset record end to end with ``policy`` and finalise it like any other session."""
    ep = run_episode(record, ctx.budget, 0, policy, ctx.predict, 0.0, tau=ctx.tau)
    items = [int(a) for a in ep["items_asked"]]
    for st, diag in enumerate(replayed_diagnostics(ctx, policy, record, items)):
        if st < len(ep["trace"]):
            ep["trace"][st]["selection"] = diag.get("selection")
    return finalize(ctx, state=ep["final_state"], trace=ep["trace"], items_asked=items,
                    stop_reason=ep["stop_reason"], result_id=result_id, label_source=record.get("label_source"),
                    true_label=record.get("label"), p_hat=ep["p_hat"])


# --------------------------------------------------------------------------------------------------------------
# Adaptive RL sessions (src/rl): the agent decides ASK(item) or STOP after every answer.
# --------------------------------------------------------------------------------------------------------------
def rl_decide(agent, state: dict, *, stop_reason: Optional[str] = None) -> dict:
    """The agent's decision at ``state`` with its explanation (``src/rl/explain.py``)."""
    from src.rl.explain import explain_decision
    from src.rl.mdp import codes_from_state
    from src.screening.diagnostics import ITEMS as _texts
    return explain_decision(agent, codes_from_state(state), stop_reason=stop_reason, item_texts=_texts)


def rl_apply(ctx: Context, agent, state: dict, j: int, value: Optional[int], asked_before: int) -> Tuple[dict, dict]:
    """Record the answer to item ``j`` (``value`` None = skipped: the question is spent, no evidence).

    The step stores the explanation of the decision that *asked* ``j``, recomputed from the state it was asked in
    (inference is deterministic). If the agent would no longer ask ``j`` there, the policy changed under the
    session and this raises instead of attaching an explanation of a different decision.
    """
    from src.env.state import MISSING, OBSERVED
    decision = rl_decide(agent, state)
    if decision["action"].get("item_index") != int(j):
        raise RuntimeError("the RL policy no longer selects the pending question at this state")
    p_before = ctx.predict(state)
    mask = np.asarray(state["mask"], dtype=int).copy()
    val = np.asarray(state["value"], dtype=int).copy()
    if mask[j] != 0:
        raise ValueError(f"item {j} was already asked")
    if value is None:
        mask[j], val[j] = MISSING, -1
    else:
        mask[j], val[j] = OBSERVED, int(value)
    remaining = int(state.get("questions_remaining", ctx.budget)) - 1
    new_state = {**{k: v for k, v in state.items() if k not in ("mask", "value")}, "mask": mask, "value": val,
                 "questions_remaining": remaining, "budget": ctx.budget}
    step = {"step": asked_before + 1, "item": f"A{j + 1}", "value": None if value is None else int(value),
            "skipped": value is None, "belief_before": p_before, "belief_after": ctx.predict(new_state),
            "decision": decision}
    return new_state, step
