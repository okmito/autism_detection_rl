"""PPO policy — §11.2 ablation AB-4 (stub with basic REINFORCE for testing)"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from src.env.state import encode_state
from typing import List, Dict, Any

STOP = -1

class PPOPolicy:
    def __init__(self, n_items: int, m_list=None, device: str="cpu"):
        self.n_items = n_items
        self.m_list = m_list
        if m_list is None:
            input_dim = 4*n_items+1
        else:
            input_dim = 3*n_items+sum(m_list)+1
        self.device = device
        self.actor = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128, n_items+1)).to(device)
        self.critic = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128,1)).to(device)
        self.opt = torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=3e-4)

    def _encode(self, state):
        v = encode_state(state, m_list=self.m_list, questions_remaining=state.get("questions_remaining",0), budget=state.get("budget", self.n_items))
        return torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)

    def __call__(self, state, legal):
        with torch.no_grad():
            logits = self.actor(self._encode(state)).cpu().numpy()[0]
            masked = np.full(self.n_items+1, float("-inf"))
            for a in legal:
                idx = self.n_items if a==STOP else a
                masked[idx]=logits[idx]
            # softmax over legal only
            # subtract max for stability
            m = np.max(masked[np.isfinite(masked)])
            exps = np.exp(masked - m); exps[~np.isfinite(masked)]=0
            probs = exps / exps.sum()
            # sample or greedy: greedy for eval
            best = int(np.argmax(masked))
            return STOP if best==self.n_items else best
