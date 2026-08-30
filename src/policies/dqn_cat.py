"""RL-based CAT (DQN) — §17 #9 : psychometric reward = -posterior variance"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from .dqn import DQNPolicy

STOP = -1

class DQNCATPolicy(DQNPolicy):
    """Same machinery as DQN but intended to be trained with variance reward.
    Inference is identical to DQNPolicy; reward definition difference is training-time only.
    """
    pass
