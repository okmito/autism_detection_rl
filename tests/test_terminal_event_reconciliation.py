"""Terminal-event invariants (no EVOI/threshold/budget changes).

Verified reconciliation (VAL, seed 0): hybrid = explicit policy STOP at
3.31-ish mean questions; environment budget exhaustion is the only other
terminal event. The prior audit trace counted *decisions* at each n_asked
depth (412 decisions), not episodes, and its loop never checked budget
exhaustion before querying the policy, so rows at n_asked=6 are decision
points, not six-question episodes.
"""
import numpy as np

from src.data.ingest import load_dataset
from src.data.splits import stratified_split
from src.env.environment import STOP, run_episode
from src.eval.metrics import acquisition_burden
from src.models.logistic_predictor import LogisticPredictor
from src.models.bayesian_logistic import BayesianLogisticPredictor
from src.policies.irt_cat import IRTCATPolicy
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy

BUDGET = 6


def _stack(seed=0, n_val=12):
    recs = load_dataset("saudi")
    train, val, _ = stratified_split(recs, seed=0)
    base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
    bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
    irt = IRTCATPolicy(train, n_items=10)
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
    return val[:n_val], bayes, pol


def test_terminal_events_partition_all_episodes():
    val, bayes, pol = _stack()
    outs = [run_episode(r, BUDGET, 0, pol, bayes, lambda_cost=0.0) for r in val]
    explicit = [o for o in outs if o["stop_reason"] == "policy_stop"]
    exhausted = [o for o in outs if o["stop_reason"] == "budget_exhausted"]
    assert len(explicit) + len(exhausted) == len(outs)
    assert all(o["stop_reason"] in ("policy_stop", "budget_exhausted") for o in outs)


def test_question_count_invariants_and_burden_reproduction():
    val, bayes, pol = _stack()
    outs = [run_episode(r, BUDGET, 0, pol, bayes, lambda_cost=0.0) for r in val]
    asked = [o["items_asked"] for o in outs]
    assert all(0 <= len(a) <= BUDGET for a in asked)
    assert all(len(set(a)) == len(a) for a in asked)  # no duplicate questions
    burden = acquisition_burden(asked)
    assert burden["mean"] == float(np.mean([len(a) for a in asked]))
    # budget-exhaustion means exactly BUDGET questions, policy STOP means < BUDGET
    assert all(len(o["items_asked"]) == BUDGET for o in outs if o["stop_reason"] == "budget_exhausted")
    assert all(len(o["items_asked"]) < BUDGET for o in outs if o["stop_reason"] == "policy_stop")


def test_policy_max_questions_cap_is_unreachable_under_env_budget():
    # benchmark construction: max_questions=10 > env budget=6, so the cap
    # can never trigger; termination is only explicit STOP or env exhaustion
    _, _, pol = _stack()
    assert pol.max_questions == 10 and pol.min_questions == 0
    assert pol.max_questions > BUDGET  # structural: cap unreachable at budget 6


def test_no_abstain_outcome_exists():
    # hybrid policy returns item or STOP only; env returns policy_stop or
    # budget_exhausted. Abstention is "keep asking the best item".
    val, bayes, pol = _stack()
    outs = [run_episode(r, BUDGET, 0, pol, bayes, lambda_cost=0.0) for r in val]
    assert all(o["stop_reason"] in ("policy_stop", "budget_exhausted") for o in outs)
    assert all("abstain" not in o["stop_reason"] for o in outs)
