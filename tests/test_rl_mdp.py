"""Adaptive-RL MDP: belief model, state codes, legal actions, rewards, value iteration, DQN, rollouts."""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from src.rl.baselines import FixedOrder, RandomFixedLength, ThresholdStopping
from src.rl.belief import NaiveBayesBelief
from src.rl.evaluate import adaptivity_profile, rollout, screening_metrics
from src.rl.experiment import DEFAULT_PROTOCOL, Study, holm, non_inferiority, paired_bootstrap
from src.rl.mdp import (C_NEG, C_POS, C_SKIP, MDPConfig, ReplayBatch, codes_from_state, expected_terminal_quality,
                        legal_mask, terminal_quality)
from src.rl.planner import BeliefValueIteration, greedy_action


def synth(n_rec=300, n=10, seed=0):
    """Correlated synthetic screening data with an independent latent label (no circular rule)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rec) < 0.5).astype(int)
    sens = np.linspace(0.45, 0.85, n)
    fpr = np.linspace(0.05, 0.4, n)[::-1]
    latent = rng.normal(size=n_rec)
    X = np.zeros((n_rec, n))
    for j in range(n):
        p = np.where(y == 1, sens[j], fpr[j])
        p = np.clip(p + 0.08 * latent, 0.01, 0.99)
        X[:, j] = (rng.random(n_rec) < p).astype(float)
    return X, y


@pytest.fixture(scope="module")
def data():
    return synth()


@pytest.fixture(scope="module")
def belief(data):
    X, y = data
    return NaiveBayesBelief.fit(X, y, trained_on="synthetic")


def state_for(codes):
    codes = np.asarray(codes)
    mask = np.zeros(len(codes), int)
    value = np.full(len(codes), -1, int)
    mask[(codes == C_NEG) | (codes == C_POS)] = 1
    mask[codes == C_SKIP] = 2
    value[codes == C_NEG] = 0
    value[codes == C_POS] = 1
    return {"mask": mask, "value": value, "n": len(codes)}


# ----------------------------------------------------------------------------- belief
def test_belief_unobserved_and_skipped_items_carry_no_evidence(belief):
    codes = np.zeros(10, int)
    p0 = belief.predict_state(state_for(codes))
    assert p0 == pytest.approx(belief.pi)
    codes[3] = C_SKIP
    assert belief.predict_state(state_for(codes)) == pytest.approx(p0)


def test_belief_answers_move_belief_in_the_direction_of_their_likelihood_ratio(belief):
    for j in range(10):
        c1, c0 = np.zeros(10, int), np.zeros(10, int)
        c1[j], c0[j] = C_POS, C_NEG
        p1, p0 = belief.predict_state(state_for(c1)), belief.predict_state(state_for(c0))
        assert (p1 > belief.pi) == (belief.llr1[j] > 0)
        assert (p0 < belief.pi) == (belief.llr0[j] < 0)
        eff = belief.item_effect(j, 1)
        assert np.log(p1 / (1 - p1)) - np.log(belief.pi / (1 - belief.pi)) == pytest.approx(eff["log_odds_shift"])


def test_belief_evidence_weight_is_fitted_out_of_fold_and_bounded(belief):
    assert belief.metadata_extra["calibrated"] is True
    assert 0.05 <= belief.w <= 1.5


def test_belief_round_trip_and_validation(belief):
    b2 = NaiveBayesBelief.from_dict(belief.to_dict())
    assert np.allclose(b2.theta, belief.theta) and b2.w == belief.w and b2.pi == belief.pi
    bad = belief.to_dict()
    bad["theta"][0][0] = 1.0
    with pytest.raises(ValueError):
        NaiveBayesBelief.from_dict(bad)


def test_answer_probability_is_the_mixture_of_class_conditionals(belief):
    a = belief.answer_prob(0.3)
    assert np.allclose(a, 0.3 * belief.theta[:, 1] + 0.7 * belief.theta[:, 0])


# ----------------------------------------------------------------------------- MDP
def test_legal_mask_never_offers_an_asked_item_and_gates_stop():
    cfg = MDPConfig(min_questions=2, max_questions=4)
    codes = np.array([[C_POS, 0, 0, C_SKIP, 0, 0, 0, 0, 0, 0]])
    lm = legal_mask(codes, cfg)[0]
    assert not lm[0] and not lm[3]
    assert lm[1] and lm[2]
    assert lm[-1]                                          # 2 used (a skip counts as a question used)
    codes = np.array([[C_POS, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert not legal_mask(codes, cfg)[0][-1]               # STOP illegal before min_questions
    codes = np.array([[C_POS, C_NEG, C_POS, C_NEG, 0, 0, 0, 0, 0, 0]])
    lm = legal_mask(codes, cfg)[0]
    assert not lm[:-1].any() and lm[-1]                     # cap reached: STOP forced


def test_codes_round_trip_with_app_state_dict():
    codes = np.array([0, C_NEG, C_POS, C_SKIP, 0, 0, C_POS, 0, 0, 0])
    assert np.array_equal(codes_from_state(state_for(codes)), codes)


def test_terminal_rewards():
    cfg = MDPConfig(terminal="accuracy")
    assert terminal_quality(np.array([0.7, 0.7, 0.2]), np.array([1, 0, 0]), cfg).tolist() == [1.0, 0.0, 1.0]
    assert expected_terminal_quality(np.array([0.7, 0.2]), cfg).tolist() == pytest.approx([0.7, 0.8])
    cfg = MDPConfig(terminal="brier")
    assert terminal_quality(np.array([0.7]), np.array([1]), cfg)[0] == pytest.approx(1 - 0.09)
    assert expected_terminal_quality(np.array([0.7]), cfg)[0] == pytest.approx(1 - 0.21)


def test_config_validation():
    for bad in ({"terminal": "x"}, {"max_questions": 11}, {"min_questions": 11}, {"cost": -1}, {"tau": 1.0}):
        with pytest.raises(ValueError):
            MDPConfig(**bad)


def test_replay_batch_rejects_illegal_actions_and_charges_cost(data, belief):
    X, y = data
    cfg = MDPConfig(cost=0.05, min_questions=1)
    env = ReplayBatch(X[:3], y[:3], cfg)
    with pytest.raises(ValueError):
        env.step(np.array([10, 10, 10]), belief)          # STOP before min_questions
    r, done = env.step(np.array([0, 1, 2]), belief)
    assert np.allclose(r, -0.05) and not done.any()
    with pytest.raises(ValueError):
        env.step(np.array([0, 3, 3]), belief)             # row 0 re-asks item 0
    r, done = env.step(np.array([10, 10, 10]), belief)
    assert done.all() and set(np.unique(r)) <= {0.0, 1.0}


# ----------------------------------------------------------------------------- value iteration
def brute_force_value(belief, cfg, codes):
    """Direct recursion of the Bellman equation (reference implementation)."""
    codes = np.asarray(codes)
    lm = legal_mask(codes[None], cfg)[0]
    o = (codes == C_NEG) | (codes == C_POS)
    p = float(belief.prob_arrays(o, (codes == C_POS).astype(int)))
    vals = []
    if lm[-1]:
        vals.append(float(expected_terminal_quality(np.array([p]), cfg)[0]))
    for j in np.flatnonzero(lm[:-1]):
        a1 = belief.theta[j, 1] * p + belief.theta[j, 0] * (1 - p)
        c0, c1 = codes.copy(), codes.copy()
        c0[j], c1[j] = C_NEG, C_POS
        vals.append(-cfg.cost + (1 - a1) * brute_force_value(belief, cfg, c0) + a1 * brute_force_value(belief, cfg, c1))
    return max(vals)


@pytest.mark.parametrize("terminal,cost", [("accuracy", 0.0), ("accuracy", 0.01), ("brier", 0.005)])
def test_value_iteration_matches_brute_force_bellman_recursion(terminal, cost):
    X, y = synth(n_rec=200, n=5, seed=3)
    b = NaiveBayesBelief.fit(X, y)
    cfg = MDPConfig(n_items=5, max_questions=4, min_questions=1, cost=cost, terminal=terminal)
    vi = BeliefValueIteration(b, cfg)
    for codes in ([0] * 5, [C_POS, 0, 0, 0, 0], [C_NEG, C_POS, 0, 0, 0], [0, 0, C_SKIP, 0, C_POS]):
        assert vi.V[vi.index_of(np.array(codes))] == pytest.approx(brute_force_value(b, cfg, codes), abs=1e-12)
        Q = vi.q_values(np.array(codes))
        assert np.max(Q) == pytest.approx(vi.V[vi.index_of(np.array(codes))], abs=1e-12)


def test_value_iteration_decomposition_reconstructs_q(belief):
    vi = BeliefValueIteration(belief, MDPConfig(cost=0.01))
    codes = np.array([C_POS, C_NEG, 0, 0, 0, 0, 0, 0, 0, 0])
    Q = vi.q_values(codes)
    for j in range(2, 10):
        d = vi.decompose(codes, j)
        q = -d["cost"] + (1 - d["p_answer_atypical"]) * d["value_after_typical"] + \
            d["p_answer_atypical"] * d["value_after_atypical"]
        assert q == pytest.approx(Q[j], abs=1e-12)


def test_greedy_action_tie_rules():
    Q = np.array([[0.5, 0.7, 0.7, 0.7], [0.5, 0.6, -np.inf, 0.4], [-np.inf, -np.inf, -np.inf, 0.1]])
    assert greedy_action(Q).tolist() == [3, 1, 3]          # STOP wins a tie; lowest ask index otherwise


def test_stop_is_a_learned_decision_driven_by_cost(data, belief):
    X, y = data
    q = {}
    for c in (0.0, 0.01, 0.6):
        cfg = MDPConfig(cost=c)
        r = rollout(BeliefValueIteration(belief, cfg), X, belief, cfg)
        q[c] = r["n_q"].mean()
        assert r["n_q"].min() >= cfg.min_questions and r["n_q"].max() <= cfg.max_questions
    assert q[0.6] == 2.0                                    # no question can repay 0.6 accuracy units: stop at the minimum
    assert q[0.0] > q[0.01] > q[0.6]
    r = rollout(BeliefValueIteration(belief, MDPConfig(cost=0.01)), X, belief, MDPConfig(cost=0.01))
    assert len(np.unique(r["n_q"])) > 1                     # stopping point depends on the person


def test_next_question_depends_on_previous_answers(belief):
    vi = BeliefValueIteration(belief, MDPConfig(cost=0.005))
    first = int(vi.act_batch(np.zeros((1, 10), int))[0])
    c0, c1 = np.zeros(10, int), np.zeros(10, int)
    c0[first], c1[first] = C_NEG, C_POS
    seen = set()
    for a, b in itertools.product((C_NEG, C_POS), repeat=2):
        codes = np.zeros(10, int)
        codes[first] = a
        nxt = int(vi.act_batch(codes[None])[0])
        codes[nxt] = b
        seen.add((a, b, int(vi.act_batch(codes[None])[0])))
    third_choices = {t for (_, _, t) in seen}
    assert len(third_choices) > 1, "the agent asks the same third question whatever was answered"


def test_value_iteration_is_deterministic(belief):
    a = BeliefValueIteration(belief, MDPConfig(cost=0.01))
    b = BeliefValueIteration(belief, MDPConfig(cost=0.01))
    assert np.array_equal(a.V, b.V)


# ----------------------------------------------------------------------------- rollouts / baselines
def test_rollout_never_repeats_an_item_and_respects_cap(data, belief):
    X, y = data
    cfg = MDPConfig(cost=0.0, max_questions=6)
    r = rollout(BeliefValueIteration(belief, cfg), X, belief, cfg)
    for s in r["seq"]:
        asked = s[s >= 0]
        assert len(asked) == len(set(asked.tolist())) <= 6
    assert (r["n_q"] <= 6).all()


def test_rollout_rejects_an_agent_that_reasks():
    class Bad:
        name = "bad"

        def act_batch(self, codes):
            return np.zeros(len(codes), dtype=int)
    X, y = synth(n_rec=20)
    b = NaiveBayesBelief.fit(X, y)
    with pytest.raises(ValueError):
        rollout(Bad(), X, b, MDPConfig())


def test_rollout_handles_missing_answers_as_skips():
    X, y = synth(n_rec=50)
    X[:, 0] = np.nan
    b = NaiveBayesBelief.fit(X, y)
    cfg = replace(MDPConfig(), min_questions=0)
    r = rollout(FixedOrder(range(10), 3), X, b, cfg)
    assert (r["codes"][:, 0] == C_SKIP).all() and (r["n_q"] == 3).all()


def test_fixed_and_random_baselines_ask_exactly_k(data, belief):
    X, y = data
    cfg = replace(MDPConfig(), min_questions=0)
    for k in (1, 4, 10):
        assert (rollout(FixedOrder(range(10), k), X, belief, cfg)["n_q"] == k).all()
        assert (rollout(RandomFixedLength(10, k, seed=1), X, belief, cfg)["n_q"] == k).all()


def test_threshold_stopping_baseline_stops_when_confident(data, belief):
    X, y = data
    cfg = MDPConfig()
    r = rollout(ThresholdStopping(belief, 0.9, cfg.min_questions, cfg.max_questions), X, belief, cfg)
    early = r["n_q"] < 10
    assert early.any()
    assert np.all((r["p"][early] >= 0.9) | (r["p"][early] <= 0.1))


def test_adaptivity_profile_detects_response_dependence(data, belief):
    X, y = data
    cfg = MDPConfig(cost=0.005)
    r = rollout(BeliefValueIteration(belief, cfg), X, belief, cfg)
    prof = adaptivity_profile(r["seq"], r["codes"])
    assert prof["distinct_sequences"] > 1 and prof["distinct_lengths"] > 1


def test_screening_metrics_basic():
    m = screening_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.1, 0.6]), np.array([2, 3, 4, 5]))
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5 and m["mean_questions"] == 3.5


# ----------------------------------------------------------------------------- DQN
def test_dqn_trains_acts_legally_and_is_seed_deterministic(data):
    from src.rl.dqn import DQNAgent, DQNConfig, train_dqn
    X, y = data
    cfg = MDPConfig(cost=0.01)
    dcfg = DQNConfig(epochs=8, updates_per_epoch=2, batch_size=64)
    n1 = train_dqn(X[:120], y[:120], cfg, dcfg, seed=3)
    n2 = train_dqn(X[:120], y[:120], cfg, dcfg, seed=3)
    for a, b in zip(n1.state_dict().values(), n2.state_dict().values()):
        assert np.array_equal(a.numpy(), b.numpy())
    b = NaiveBayesBelief.fit(X[:120], y[:120])
    ag = DQNAgent([n1, n2], b, cfg, dcfg)
    r = rollout(ag, X[120:], b, cfg)
    assert (r["n_q"] >= cfg.min_questions).all() and (r["n_q"] <= cfg.max_questions).all()
    q = ag.q_values(r["codes"])
    assert np.isinf(q[r["codes"][:, 0] != 0, 0]).all()     # asked items are masked out


# ----------------------------------------------------------------------------- experiment statistics
def test_constrained_selection_rule():
    st = Study(np.zeros((4, 10)), np.array([0, 1, 0, 1]), DEFAULT_PROTOCOL)
    table = {0.0: {"balanced_accuracy": 0.90, "mean_questions": 9.0},
             0.01: {"balanced_accuracy": 0.885, "mean_questions": 5.0},
             0.05: {"balanced_accuracy": 0.80, "mean_questions": 2.0}}
    assert st.select(table, full_ba=0.90) == (0.01, "constraint_met")
    assert st.select(table, full_ba=0.99) == (0.0, "constraint_unmet_max_ba")


def test_holm_and_non_inferiority():
    h = holm({"a": 0.01, "b": 0.04})
    assert h["a"]["p_holm"] == pytest.approx(0.02) and h["b"]["p_holm"] == pytest.approx(0.04)
    assert h["a"]["reject_h0"] and h["b"]["reject_h0"]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.random(200) * 0.4, 0, 1)
    a = {"p": p, "n_q": np.full(200, 4)}
    bt = paired_bootstrap(y, a, {"p": p, "n_q": np.full(200, 10)}, 0.5, 300, 0)
    ni = non_inferiority(bt, "balanced_accuracy", 0.05)
    assert ni["difference"] == 0 and ni["non_inferior"]
    assert bt["mean_questions"]["difference"] == -6
