"""Transition replay buffer — §11.2

There was no replay buffer anywhere in the repository before this module: nothing
collected transitions, so neither ``DQNPolicy.train_step`` nor
``PPOPolicy.train_step`` could ever be called.

The buffer stores the *legal-action set* alongside each transition. That is not
optional bookkeeping: the bootstrap target is a max over next-state Q-values,
and without the legal set the max is free to select an already-asked item or a
STOP that violates ``b_min``. The mask has to survive storage for the training
step to be correct.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Dict, List


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of ``(s, a, r, s_next, done, legal_next)``.

    Transitions arriving via :meth:`add_step` are stitched into full transitions
    by :meth:`end_episode`, which resolves the terminal reward and the final
    next-state. Collecting a whole episode before storing it keeps the reward
    attributable to the stop decision.
    """

    def __init__(self, capacity: int = 50_000, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf: Deque[tuple] = deque(maxlen=capacity)
        self._pending: List[Dict[str, Any]] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def pending(self) -> int:
        """Number of steps collected but not yet committed to an episode."""
        return len(self._pending)

    def add_step(self, state, action, reward: float, next_state, done: bool,
                 legal_next: List[int]):
        """Record one environment step within the current episode.

        ``legal_next`` is the legal-action set the agent chose from in ``state``.
        The set that constrains the bootstrap target belongs to ``next_state``,
        which :meth:`end_episode` picks up from the following step.
        """
        self._pending.append({
            "s": state,
            "a": action,
            "r": float(reward),
            "s_next": next_state,
            "done": bool(done),
            "legal_next": list(legal_next),
        })

    def end_episode(self, terminal_reward: float | None = None):
        """Commit the pending episode to the buffer.

        ``terminal_reward`` overrides the reward recorded on the final step,
        which is how the episode's terminal utility (Brier minus cost, per §10)
        is attached to the STOP transition.
        """
        if not self._pending:
            return 0
        steps = self._pending
        self._pending = []
        for i, st in enumerate(steps):
            last = i == len(steps) - 1
            r = st["r"]
            if last and terminal_reward is not None:
                r = float(terminal_reward)
            if last:
                self._buf.append((st["s"], st["a"], r, None, 1.0, None))
            else:
                # Step i+1's legal_next is the legal set of s_{i+1}: at that step
                # the agent chose an action *from* s_{i+1}, and s_{i+1} is this
                # transition's s_next. That is the set the bootstrap max must be
                # taken over.
                nxt = steps[i + 1]
                self._buf.append((st["s"], st["a"], r, nxt["s_next"], 0.0,
                                  nxt["legal_next"]))
        return len(steps)

    def add(self, s, a, r, s_next, done, legal_next):
        """Store a complete transition directly."""
        self._buf.append((s, a, float(r), None if done else s_next,
                          float(done), None if done else list(legal_next)))

    def sample(self, batch_size: int) -> List[tuple]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._buf) < batch_size:
            raise ValueError(
                f"buffer holds {len(self._buf)} transitions, need {batch_size}"
            )
        return self._rng.sample(list(self._buf), batch_size)

    def all(self) -> List[tuple]:
        return list(self._buf)

    def clear(self):
        self._buf.clear()
        self._pending.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._buf),
            "capacity": self.capacity,
            "pending": len(self._pending),
        }
