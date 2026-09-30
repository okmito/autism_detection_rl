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
