"""Random baseline — §17 #6"""
from __future__ import annotations
import random
import numpy as np

STOP = -1

class RandomPolicy:
    def __init__(self, seed: int | None = 0):
        # seed=None -> non-deterministic across instances (fresh entropy);
        # seed=int  -> reproducible sequence, shared across instances with the same seed.
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def __call__(self, state, legal):
        # legal includes STOP if allowed
        return self.rng.choice(legal)
