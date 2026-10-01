"""RL training regression tests — Part 1 (2026-10-01).

These cover defects that existed in ``src/policies/dqn.py`` and
``src/policies/ppo.py`` but had never been caught, because no training code had
ever been executed anywhere in the repository:

  1. ``DQNPolicy.train_step`` built a legal-action mask and discarded it, so the
     bootstrap target was a max over *all* actions -- including illegal ones.
  2. ``PPOPolicy`` had no training method at all: the critic was constructed
     but never received a gradient.

(A third alleged defect -- a ``torch.cat`` shape error on terminal next-states --
was investigated and RETRACTED; see ``RL_TRAINING_REPORT.md`` §1.)

Plus coverage for the replay buffer that was missing entirely.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.state import init_state
from src.policies.dqn import DQNPolicy, STOP
from src.policies.ppo import PPOPolicy
from src.policies.replay import ReplayBuffer


def _record(n_items=4, label=1):
    return {
        "item_responses": np.array([1.0] * n_items),
        "label": label,
        "missing_mask": np.zeros(n_items, dtype=bool),
    }


def _state_with_mask(asked):
    """Build a valid env state with the given items already OBSERVED."""
    st = init_state(_record())
    for j in asked:
        st["mask"][j] = 1
        st["value"][j] = 1
    st["asked_count"] = len(asked)
    return st


# ---------------------------------------------------------------------------
# DQN defect 2 — terminal transition shape
# ---------------------------------------------------------------------------

def test_dqn_train_step_handles_terminal_batch():
    """A batch containing terminal transitions must train cleanly.

    Guards the terminal branch of ``train_step``. NOTE: this was originally
    written as a regression against a claimed ``torch.cat`` shape error. That
    claim was wrong -- ``torch.cat(dim=0)`` concatenates along dim 0 and each
    row contributes one ``(1, D)`` tensor, so the original code did not raise.
    Kept because mixed terminal/non-terminal batches are the common case in
    real collection (STOP is a legal action), so the branch deserves coverage.
    """
    pol = DQNPolicy(n_items=4)
    batch = []
    for i in range(3):
        s = _state_with_mask([0])
        batch.append((s, 1, 0.5, _state_with_mask([0, 1]), 0.0, [1, 2, 3, STOP]))
    # last two are terminal (s_next None)
    batch.append((_state_with_mask([0]), STOP, 1.0, None, 1.0, None))
    batch.append((_state_with_mask([1]), 2, 0.2, None, 1.0, None))

    loss = pol.train_step(batch)
    assert np.isfinite(loss)


def test_dqn_train_step_terminal_target_has_no_bootstrap():
    """With a terminal transition the regression target must be exactly r.

    ``train_step`` returns the loss computed *before* the optimiser step, so the
    returned value pins down the target y: loss == MSE(Q(s,a), y). For a terminal
    transition y must be r with no gamma*bootstrap term. Asserting on the
    post-step Q instead would be wrong -- one Adam step moves Q by about lr,
    not to the target.
    """
    pol = DQNPolicy(n_items=4)
    s = _state_with_mask([0])
    batch = [(s, STOP, 0.75, None, 1.0, None)]

    q_before = pol.q(pol._encode(s)).detach()[0, pol.n_items].item()
    loss = pol.train_step(batch, gamma=0.99)

    # If the terminal row leaked into the bootstrap, y would be
    # 0.75 + 0.99 * max_over_actions Q(s_next) and the loss would differ.
    # Tolerance is float32-scale (1e-6), not float64.
    assert loss == pytest.approx((q_before - 0.75) ** 2, abs=1e-6)


def test_dqn_train_step_rejects_empty_batch():
    pol = DQNPolicy(n_items=4)
    with pytest.raises(ValueError):
        pol.train_step([])


# ---------------------------------------------------------------------------
# DQN defect 1 — legal-action mask on the bootstrap target
# ---------------------------------------------------------------------------

def test_dqn_bootstrap_ignores_illegal_next_actions():
    """The target max must be taken over legal next actions only.

    Regression: the mask was constructed and thrown away, so the bootstrap
    optimised toward already-asked items.

    Construction: make the online and target nets constant functions of the last
    layer bias, then put a huge value on an ILLEGAL action. The correct
    (masked) target uses the best legal value; the unmasked target would use the
    illegal one. ``train_step`` returns the pre-step loss, which pins down y.
    """
    pol = DQNPolicy(n_items=4)
    s = _state_with_mask([0, 1])       # items 0,1 asked -> legal are 2, 3, STOP
    s_next = _state_with_mask([0, 1, 2])
    legal_next = [2, 3, STOP]          # indices 2, 3, 4

    # Make both nets constant: zero the last layer's weight, set its bias.
    # online argmax picks the legal index of highest bias (used for Double DQN
    # action selection); target supplies the value.
    with torch.no_grad():
        pol.q.net[-1].weight.zero_()
        pol.q.net[-1].bias.copy_(torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
        pol.target.net[-1].weight.zero_()
        # illegal action idx0 = 1000 (huge); legal idx2 = 5, idx3 = 5, STOP idx4 = 9
        pol.target.net[-1].bias.copy_(torch.tensor([1000.0, 1000.0, 5.0, 5.0, 9.0]))

    gamma = 0.5
    batch = [(s, 2, 0.0, s_next, 0.0, legal_next)]

    # Predict the two candidate targets before stepping.
    # Double DQN: online argmax over LEGAL actions -> idx3 (bias 2.0) is the
    # argmax among {2:1.0, 3:2.0, STOP:3.0} -> idx4 actually (3.0). So
    # next_actions = idx4 (STOP). target value at idx4 = 9.0.
    # masked y = 0 + 0.5*9 = 4.5 ; unmasked y would use max over all = 1000.
    q_before = pol.q(pol._encode(s)).detach()[0, 2].item()
    loss = pol.train_step(batch, gamma=gamma)

    assert loss == pytest.approx((q_before - 4.5) ** 2, abs=1e-6), (
        f"masked target not used (loss {loss}); unmasked would imply a much "
        f"larger target from the illegal action"
    )


def test_dqn_select_action_only_returns_legal():
    pol = DQNPolicy(n_items=4)
    s = _state_with_mask([0])
    legal = [1, 2, 3, STOP]
    for _ in range(30):
        a = pol.select_action(s, legal, epsilon=1.0)
        assert a in legal


# ---------------------------------------------------------------------------
# PPO defect 3 — the critic never received a gradient
# ---------------------------------------------------------------------------

def test_ppo_train_step_updates_critic():
    """PPOPolicy.train_step must produce a gradient on the critic.

    Regression: PPOPolicy had no training method at all, so its critic never
    received a gradient.
    """
    pol = PPOPolicy(n_items=4)
    batch = [
        (_state_with_mask([0]), 1, 0.7, None, 0.0, [1, 2, 3, STOP]),
        (_state_with_mask([1]), 2, 0.3, None, 0.0, [0, 2, 3, STOP]),
    ]
    before = [p.detach().clone() for p in pol.critic.parameters()]
    out = pol.train_step(batch)
    after = list(pol.critic.parameters())

    assert set(["loss", "policy_loss", "value_loss", "entropy"]) <= set(out)
    assert np.isfinite(out["loss"])
    changed = any(
        not torch.allclose(b, a.detach()) for b, a in zip(before, after)
    )
    assert changed, "critic parameters did not change — critic received no gradient"


def test_ppo_train_step_updates_actor():
    pol = PPOPolicy(n_items=4)
    batch = [(_state_with_mask([0]), 1, 0.5, None, 0.0, [1, 2, 3, STOP])]
    before = [p.detach().clone() for p in pol.actor.parameters()]
    pol.train_step(batch)
    after = list(pol.actor.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))


def test_ppo_assigns_zero_probability_to_illegal_actions():
    """Illegal actions must never receive probability mass."""
    pol = PPOPolicy(n_items=4)
    s = _state_with_mask([0])
    legal = [2, 3, STOP]
    # deterministic call must not choose an illegal action
    for _ in range(20):
        assert pol.select_action(s, legal, deterministic=True) in legal
    # sampled call must not choose an illegal action
    for _ in range(50):
        assert pol.select_action(s, legal, deterministic=False) in legal


def test_ppo_train_step_rejects_empty_batch():
    pol = PPOPolicy(n_items=4)
    with pytest.raises(ValueError):
        pol.train_step([])


# ---------------------------------------------------------------------------
# Replay buffer — did not exist before Part 1
# ---------------------------------------------------------------------------

def test_replay_buffer_stitch_and_capacity():
    buf = ReplayBuffer(capacity=10)
    s0, s1, s2 = _state_with_mask([0]), _state_with_mask([0, 1]), _state_with_mask([0, 1, 2])
    # Each step records the legal set of its OWN state (the set the agent chose
    # from). end_episode re-pairs it onto the transition that arrives there.
    buf.add_step(s0, 1, 0.0, s1, False, [1, 2, 3, STOP])   # legal at s0
    buf.add_step(s1, 2, 0.0, s2, False, [2, 3, STOP])      # legal at s1
    buf.add_step(s2, STOP, 0.0, None, True, [3, STOP])    # legal at s2
    assert buf.pending == 3
    n = buf.end_episode(terminal_reward=0.9)
    assert n == 3
    assert len(buf) == 3
    # last transition is terminal with the overridden reward
    s, a, r, sn, done, ln = buf.all()[-1]
    assert a == STOP and done == 1.0 and sn is None
    assert r == pytest.approx(0.9)
    # Transition 0 lands in s1, so it must carry s1's legal set (step 1's),
    # not s0's. Getting this backwards is the masking bug in original form.
    assert buf.all()[0][5] == [2, 3, STOP]
    assert buf.all()[1][5] == [3, STOP]


def test_replay_buffer_capacity_is_fifo():
    buf = ReplayBuffer(capacity=3)
    s = _state_with_mask([0])
    for i in range(5):
        buf.add(s, i % 4, float(i), None, True, None)
    assert len(buf) == 3


def test_replay_buffer_sample_respects_size():
    buf = ReplayBuffer(capacity=10, seed=0)
    s = _state_with_mask([0])
    for i in range(4):
        buf.add(s, i % 4, float(i), None, True, None)
    assert len(buf.sample(4)) == 4
    with pytest.raises(ValueError):
        buf.sample(99)


def test_replay_buffer_roundtrip_into_dqn_train_step():
    """A buffer built from real env states must drive a DQN update."""
    buf = ReplayBuffer(capacity=100, seed=0)
    s0, s1, s2 = _state_with_mask([0]), _state_with_mask([0, 1]), _state_with_mask([0, 1, 2])
    buf.add_step(s0, 1, 0.0, s1, False, [1, 2, 3, STOP])
    buf.add_step(s1, 2, 0.0, s2, False, [2, 3, STOP])
    buf.end_episode(terminal_reward=1.0)
    assert len(buf) == 2

    pol = DQNPolicy(n_items=4)
    loss = pol.train_step(buf.sample(2))
    assert np.isfinite(loss)


# ===========================================================================
# Part 2 (2026-10-01) — the rollout collector, which turned out to be the
# reason no policy in this repository had ever learned anything.
#
# `collect_episode` in scripts/step4_train_policies.py looked correct and ran
# without error, but it stored `r = 0.0` on every transition, attached the
# legal set of the state it LEFT rather than the state it ENTERED, and wrote
# terminal transitions into a `_pending` list that nothing ever drained.
# The combined effect was:
#
#   * DQN optimised `y = gamma * max_a Q(s', a)` with no reward anywhere.
#   * PPO's advantage was identically zero, so the policy gradient was exactly
#     zero and only the entropy bonus moved the actor.
#   * Every STOP transition was silently discarded.
#   * The bootstrap mask still permitted already-asked items.
#
# These tests pin each defect independently so a future refactor cannot silently
# reintroduce any of them.
# ===========================================================================

from scripts.step4_train_policies import _legal_for, _to_returns, collect_episode


def _collection_record(n_items=6, label=1):
    return {
        "item_responses": np.ones(n_items, dtype=float),
        "label": label,
        "missing_mask": np.zeros(n_items, dtype=bool),
    }


class _NeverStop:
    """Deterministic policy that always takes the lowest legal item."""

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        return items[0] if items else STOP


class _StopImmediately:
    def __call__(self, state, legal):
        return STOP if STOP in legal else legal[0]


def test_collect_episode_stores_the_terminal_reward():
    """P0-1: the §10 terminal utility must reach the replay buffer.

    Previously every stored reward was 0.0, including the terminal transition,
    so no learning signal existed anywhere in the buffer.
    """
    rec = _collection_record()
    buf = ReplayBuffer(capacity=100, seed=0)
    terminal_r, n_asked = collect_episode(
        rec, 4, 0, _NeverStop(), lambda s: 0.42, buf, 6, 0.0, 0.5
    )

    transitions = buf.all()
    # 4 item actions + 1 terminal STOP.
    assert len(transitions) == 5
    assert n_asked == 4

    terminal = transitions[-1]
    assert terminal[1] == STOP
    assert terminal[3] is None
    assert terminal[4] == 1.0

    # Exactly one transition carries a non-zero reward, and it is the terminal.
    nonzero = [i for i, t in enumerate(transitions) if t[2] != 0.0]
    assert nonzero == [len(transitions) - 1]
    assert terminal[2] == pytest.approx(terminal_r)

    # R = 1 - (p_hat - y)^2 = 1 - (0.42 - 1)^2
    assert terminal[2] == pytest.approx(1 - (0.42 - 1.0) ** 2)


def test_collect_episode_marks_exactly_one_terminal_transition():
    """Every episode must contribute exactly one `done == 1` row."""
    rec = _collection_record()
    buf = ReplayBuffer(capacity=100, seed=0)
    collect_episode(rec, 5, 0, _NeverStop(), lambda s: 0.3, buf, 6, 0.0, 0.5)
    transitions = buf.all()
    assert sum(1 for t in transitions if t[4] == 1.0) == 1
    # Only the terminal row may have a null next state.
    assert [t[3] is None for t in transitions] == [False] * (len(transitions) - 1) + [True]


def test_collect_episode_legal_next_is_the_state_it_enters():
    """P0-2: the bootstrap mask must be the legal set of `s_next`.

    Previously the stored set was the legal set of the state being *left*, so an
    item asked on the way into `s_next` was still marked legal.
    """
    rec = _collection_record()
    buf = ReplayBuffer(capacity=100, seed=0)
    collect_episode(rec, 6, 0, _NeverStop(), lambda s: 0.3, buf, 6, 0.0, 0.5)

    for s, a, r, s_next, done, legal_next in buf.all():
        if done:
            assert legal_next is None
            continue
        assert s_next is not None
        asked_on_the_way = {j for j in range(6) if s_next["mask"][j] == 1}
        # Nothing already observed may be offered as a legal next action.
        assert not (asked_on_the_way & set(legal_next))
        # Every still-unasked item must be present.
        unasked = {j for j in range(6) if s_next["mask"][j] == 0}
        assert unasked <= set(legal_next)


def test_collect_episode_leaves_nothing_pending():
    """P0-3: terminal transitions must be committed, not left in `_pending`.

    `ReplayBuffer.add_step` only stages transitions; `end_episode` is what
    commits them. The collector now writes with `add`, so `_pending` stays empty
    and nothing is silently lost or leaks memory.
    """
    rec = _collection_record()
    buf = ReplayBuffer(capacity=100, seed=0)
    for _ in range(10):
        collect_episode(rec, 4, 0, _StopImmediately(), lambda s: 0.3, buf, 6, 0.0, 0.5)
    assert buf.pending == 0
    # Every episode terminated at once, so each contributed exactly its STOP row.
    assert len(buf) == 10
    assert all(t[1] == STOP and t[4] == 1.0 for t in buf.all())


def test_collect_episode_honours_capacity():
    """Committed transitions must still respect the FIFO capacity bound."""
    rec = _collection_record()
    buf = ReplayBuffer(capacity=7, seed=0)
    for _ in range(10):
        collect_episode(rec, 6, 0, _NeverStop(), lambda s: 0.3, buf, 6, 0.0, 0.5)
    assert len(buf) == 7


def test_collect_episode_rejects_illegal_actions():
    rec = _collection_record()

    class _Illegal:
        def __call__(self, state, legal):
            return 5  # never legal on the first step under the mask below

    buf = ReplayBuffer(capacity=10, seed=0)
    with pytest.raises(ValueError):
        collect_episode(rec, 4, 0, _Illegal(), lambda s: 0.3, buf, 6, 0.0, 0.5)


def test_returns_to_go_propagates_terminal_utility_backwards():
    """`gamma=...` must store discounted returns, not the sparse terminal reward.

    §11.1 emits only a terminal reward. PPO's advantage is `rewards`, so storing
    the immediate reward would leave the advantage zero at every step except
    STOP and train the policy to stop immediately.
    """
    rec = _collection_record()
    buf = ReplayBuffer(capacity=100, seed=0)
    gamma = 0.9
    terminal_r, _ = collect_episode(
        rec, 4, 0, _NeverStop(), lambda s: 0.42, buf, 6, 0.0, 0.5, gamma=gamma
    )
    transitions = buf.all()
    assert len(transitions) == 5
    for i, t in enumerate(transitions):
        steps_remaining = len(transitions) - 1 - i
        assert t[2] == pytest.approx(terminal_r * gamma ** steps_remaining)
    # The default (DQN) path must keep immediate rewards.
    plain = ReplayBuffer(capacity=100, seed=0)
    collect_episode(rec, 4, 0, _NeverStop(), lambda s: 0.42, plain, 6, 0.0, 0.5)
    assert [t[2] for t in plain.all()[:-1]] == [0.0] * 4


def test_to_returns_handles_single_transition_episode():
    assert _to_returns([("s", 0, 0.75, None, True, None)], 0.99)[0][2] == pytest.approx(0.75)


def test_legal_for_forces_stop_when_nothing_remains():
    # A single-item record: once it is observed nothing is left to ask.
    rec = _collection_record(n_items=1)
    st = init_state(rec)
    st["mask"][0] = 1
    # b_min far above the number asked, but nothing is left to ask.
    assert _legal_for(st, asked=1, b_min=5) == [STOP]

    fresh = init_state(_collection_record(n_items=6))
    assert STOP not in _legal_for(fresh, asked=0, b_min=2)
    assert STOP in _legal_for(fresh, asked=2, b_min=2)


def test_collected_batch_drives_a_finite_dqn_update():
    """End-to-end: a collected batch must produce a finite loss."""
    rec = _collection_record()
    buf = ReplayBuffer(capacity=400, seed=0)
    pol = DQNPolicy(n_items=6, seed=0)
    # 40 episodes, not 20: exploration is now seeded (`DQNPolicy(seed=...)`), so
    # the transition count is deterministic rather than incidentally large. At 20
    # episodes this collected 28 transitions and fell one batch short of 32.
    for _ in range(40):
        collect_episode(rec, 4, 0, pol, lambda s: 0.42, buf, 6, 0.0, 0.5, epsilon=0.5)
    assert len(buf) >= 32
    losses = [pol.train_step(buf.sample(32), gamma=0.99) for _ in range(5)]
    assert all(np.isfinite(l) for l in losses)


def test_collected_batch_drives_a_finite_ppo_update():
    """P0 follow-on: PPO crashed on terminal rows before the P0-3 fix.

    Every collected batch now contains a `done == 1` row with
    `legal_next is None`. `PPOPolicy.legal_mask` used to raise `TypeError` on
    that row, and `train_step` failed — a latent crash that the discarded
    terminal transitions had been hiding.
    """
    rec = _collection_record()
    buf = ReplayBuffer(capacity=200, seed=0)
    pol = PPOPolicy(n_items=6)
    for k in range(8):
        collect_episode(rec, 4, 0, pol, lambda s: 0.42, buf, 6, 0.0, 0.5,
                        stochastic=True, gamma=0.99)
    data = buf.all()
    assert any(t[4] == 1.0 for t in data)

    old_logp = [float(pol.masked_log_probs(t[0], t[5])[6 if t[1] == STOP else t[1]])
                for t in data]
    out = pol.train_step(data, old_logp=old_logp)
    assert set(out) == {"loss", "policy_loss", "value_loss", "entropy"}
    assert all(np.isfinite(v) for v in out.values())
    # The advantage is no longer identically zero, so a policy gradient exists.
    assert abs(out["policy_loss"]) > 1e-12


def test_masked_log_probs_is_finite_for_every_legal_action():
    """The importance ratio must never see `-inf - -inf` and produce `nan`."""
    n_items = 6
    pol = PPOPolicy(n_items=n_items)
    st = init_state(_collection_record(n_items=n_items))
    st["mask"][0] = 1
    st["value"][0] = 1
    st["questions_remaining"] = 3
    st["budget"] = 3
    legal = [1, 2, 3, 4, 5, STOP]
    lp = pol.masked_log_probs(st, legal)
    assert np.all(np.isfinite(lp))
    # A terminal row (legal=None) must also be usable.
    assert np.all(np.isfinite(pol.masked_log_probs(st, None)))


def test_ppo_legal_mask_accepts_terminal_rows():
    pol = PPOPolicy(n_items=4)
    assert pol.legal_mask(None).all()
    mask = pol.legal_mask([0, 2, STOP])
    assert mask.tolist() == [True, False, True, False, True]


# ---------------------------------------------------------------------------
# P0-7 — reproducible epsilon-greedy exploration
# ---------------------------------------------------------------------------

def test_dqn_epsilon_greedy_is_reproducible_per_seed():
    """`select_action` used the unseeded module-level `random` generator, so
    `torch.manual_seed` / `np.random.seed` did not make rollouts reproducible
    and no DQN number in the repository could be regenerated."""
    st = init_state(_record(n_items=4))
    legal = [0, 1, 2, 3, STOP]

    def rollout(seed):
        pol = DQNPolicy(n_items=4, seed=seed)
        return [pol.select_action(st, legal, epsilon=1.0) for _ in range(30)]

    assert rollout(7) == rollout(7)
    assert rollout(11) == rollout(11)
    # Different seeds must still be able to produce different exploration.
    assert rollout(7) != rollout(11)
    # Every draw must remain legal.
    assert set(rollout(3)) <= set(legal)


def test_dqn_greedy_action_is_deterministic_at_zero_epsilon():
    pol = DQNPolicy(n_items=4, seed=0)
    st = init_state(_record(n_items=4))
    legal = [0, 1, 2, 3, STOP]
    first = pol.select_action(st, legal, epsilon=0.0)
    assert all(pol.select_action(st, legal, epsilon=0.0) == first for _ in range(5))
