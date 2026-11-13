"""Decision explanations must describe the decision the agent actually takes, with the numbers it actually used."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.rl.belief import NaiveBayesBelief
from src.rl.evaluate import rollout
from src.rl.explain import explain_decision, information_offer, reachable_range
from src.rl.mdp import C_NEG, C_POS, C_UNASKED, MDPConfig
from src.rl.planner import BeliefValueIteration, greedy_action
from tests.test_rl_mdp import synth


@pytest.fixture(scope="module")
def setup():
    X, y = synth(n_rec=300, seed=5)
    b = NaiveBayesBelief.fit(X, y)
    cfg = MDPConfig(cost=0.01)
    vi = BeliefValueIteration(b, cfg)
    return X, y, b, cfg, vi


def histories(vi, X, cfg):
    """Every intermediate history visited by the agent on the records (the states it really decides at)."""
    r = rollout(vi, X, vi.belief, cfg)
    out = []
    for seq, final in zip(r["seq"], r["codes"]):
        codes = np.zeros(cfg.n_items, int)
        out.append(codes.copy())
        for j in seq[seq >= 0]:
            codes[j] = final[j]
            out.append(codes.copy())
    return out


def test_every_explained_action_is_the_action_the_agent_takes(setup):
    X, y, b, cfg, vi = setup
    for codes in histories(vi, X[:80], cfg):
        e = explain_decision(vi, codes)
        act = int(vi.act_batch(codes[None])[0])
        if act == cfg.n_items:
            assert e["action"]["type"] == "STOP"
        else:
            assert e["action"] == {"type": "ASK", "item": f"A{act + 1}", "item_index": act}
        # the action is the argmax of the reported values under the agent's tie rule
        Q = np.full(cfg.n_items + 1, -np.inf)
        for row in e["action_values"]:
            Q[cfg.n_items if row["action"] == "STOP" else int(row["action"][1:]) - 1] = row["q"]
        assert int(greedy_action(Q[None])[0]) == act


def test_value_iteration_explanations_reconstruct_the_chosen_value_exactly(setup):
    X, y, b, cfg, vi = setup
    n_ask = 0
    for codes in histories(vi, X[:60], cfg):
        e = explain_decision(vi, codes)
        if e["action"]["type"] != "ASK":
            continue
        n_ask += 1
        d = e["value_decomposition"]
        q = -d["cost"] + (1 - d["p_answer_atypical"]) * d["value_after_typical"] + \
            d["p_answer_atypical"] * d["value_after_atypical"]
        assert q == pytest.approx(e["chosen_value"], abs=1e-12)
        assert e["faithful_to_agent_internals"] is True
        # the belief-model offer and the agent's decomposition agree on the answer branches
        assert d["p_after_atypical"] == pytest.approx(e["information"]["p_after_atypical"], abs=1e-12)
        assert d["p_after_typical"] == pytest.approx(e["information"]["p_after_typical"], abs=1e-12)
        if e["stop_was_allowed"]:
            assert e["margin_over_stop"] > 0                     # STOP wins ties, so a chosen ASK is strictly better
        if "runner_up" in e:
            assert e["runner_up"]["margin"] >= 0
    assert n_ask > 0


def test_stop_explanations(setup):
    X, y, b, cfg, vi = setup
    stops = [explain_decision(vi, c) for c in histories(vi, X[:80], cfg)]
    stops = [e for e in stops if e["action"]["type"] == "STOP"]
    assert stops
    for e in stops:
        rr = e["reachable_belief"]
        p = e["state"]["p_positive"]
        assert rr["p_min_reachable"] - 1e-12 <= p <= rr["p_max_reachable"] + 1e-12
        if e["stop_reason"] == "policy_stop":
            assert e["forced"] is False
            assert e["best_remaining_question"]["shortfall_vs_stop"] >= -1e-12
        assert any("not a diagnosis" in s for s in e["narrative"])
        json.dumps(e)                                            # serialisable for the session record


def test_budget_forced_stop_is_labelled_as_forced():
    X, y = synth(n_rec=200, seed=2)
    b = NaiveBayesBelief.fit(X, y)
    cfg = MDPConfig(cost=0.0, max_questions=3, min_questions=1)
    vi = BeliefValueIteration(b, cfg)
    codes = np.array([C_POS, C_NEG, C_POS] + [C_UNASKED] * 7)
    e = explain_decision(vi, codes)
    assert e["action"]["type"] == "STOP" and e["forced"] is True and e["stop_reason"] == "budget_exhausted"
    assert "safety cap" in e["narrative"][0]


def test_stop_not_allowed_before_minimum_is_explained(setup):
    X, y, b, cfg, vi = setup
    e = explain_decision(vi, np.zeros(cfg.n_items, int))
    assert e["action"]["type"] == "ASK" and e["stop_was_allowed"] is False and e["margin_over_stop"] is None
    assert any("not allowed before" in s for s in e["narrative"])


def test_information_offer_is_consistent_with_the_belief_model(setup):
    X, y, b, cfg, vi = setup
    codes = np.zeros(cfg.n_items, int)
    codes[2] = C_POS
    off = information_offer(b, codes, 5, cfg.tau)
    c1, c0 = codes.copy(), codes.copy()
    c1[5], c0[5] = C_POS, C_NEG
    from tests.test_rl_mdp import state_for
    assert off["p_after_atypical"] == pytest.approx(b.predict_state(state_for(c1)))
    assert off["p_after_typical"] == pytest.approx(b.predict_state(state_for(c0)))
    assert 0.0 <= off["p_decision_changes"] <= 1.0


def test_dqn_explanations_flag_that_branch_numbers_come_from_the_belief_model(setup):
    from src.rl.dqn import DQNAgent, DQNConfig, train_dqn
    X, y, b, cfg, vi = setup
    net = train_dqn(X[:100], y[:100], cfg, DQNConfig(epochs=5, updates_per_epoch=2, batch_size=32), seed=0)
    ag = DQNAgent([net], b, cfg)
    for codes in histories(vi, X[:10], cfg):
        e = explain_decision(ag, codes)
        assert e["faithful_to_agent_internals"] is False
        assert "value_decomposition" not in e
        act = int(ag.act_batch(codes[None])[0])
        assert (e["action"]["type"] == "STOP") == (act == cfg.n_items)
