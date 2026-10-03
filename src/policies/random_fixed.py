"""RandomFixedLengthPolicy — budget-matched random question selection.

Why this exists
---------------
The primary fixed-budget experiment needs a random arm that asks **exactly** B
questions. The existing :class:`~src.policies.random_policy.RandomPolicy` chooses
uniformly from ``legal``, and the environment appends STOP to ``legal`` whenever
``b_min == 0``. The random arm therefore stopped early: it asked 1.73/2, 2.60/3,
3.12/4, 3.52/5 and 4.96/10 questions in the first external evaluation. That made
"adaptive vs random" an unequal-budget comparison.

This policy removes STOP from consideration, so it cannot terminate early. Combined
with ``b_min = B`` in the episode loop, every episode asks exactly B unique
questions: ``get_legal_items`` returns only ``UNASKED`` items, so no question is
repeated.

``RandomPolicy`` is deliberately left untouched and is reserved for a separate
learned/variable-length stopping experiment. The two must not be mixed in the
fixed-budget comparison.

MANDATORY CAVEAT — research prototype, not a diagnostic device.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np

from src.env.environment import STOP


class RandomFixedLengthPolicy:
    """Uniformly sample an unasked item. Never stops.

    Parameters
    ----------
    seed:
        Fixed seed so the sampled order is reproducible. Each episode draws from
        a single shared stream, exactly as ``RandomPolicy`` does, so the seed
        protocol matches the other arms.
    """

    def __init__(self, seed: int | None = 0):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.seed = seed
        self.requests_stop = False

    def __call__(self, state: Dict[str, Any], legal: List[int]) -> int:
        # Defensive: even if the environment offers STOP (it must not under
        # b_min == B before B questions are asked), this policy never takes it.
        items = [a for a in legal if a != STOP]
        if not items:
            return STOP
        return self.rng.choice(items)