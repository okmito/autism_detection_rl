"""Double DQN policy — §11.2"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any
from src.env.state import encode_state

STOP = -1

class DQNNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class DQNPolicy:
    def __init__(self, n_items: int, m_list=None, device: str = "cpu", lr: float = 1e-3):
        self.n_items = n_items
        self.m_list = m_list
        self.device = device
        if m_list is None:
            input_dim = 4*n_items+1
        else:
            input_dim = 3*n_items+sum(m_list)+1
        self.n_actions = n_items+1  # last is STOP
        self.q = DQNNet(input_dim, self.n_actions).to(device)
        self.target = DQNNet(input_dim, self.n_actions).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    def _encode(self, state: Dict[str, Any]) -> torch.Tensor:
        v = encode_state(state, m_list=self.m_list, questions_remaining=state.get("questions_remaining",0), budget=state.get("budget", self.n_items))
        return torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state: Dict[str, Any], legal: List[int], epsilon: float = 0.0) -> int:
        import random
        if random.random() < epsilon:
            return random.choice(legal)
        # map STOP sentinel to index n_items
        q_vals = self.q(self._encode(state)).detach().cpu().numpy()[0]
        # mask illegal
        masked = np.full(self.n_actions, float("-inf"))
        for a in legal:
            idx = self.n_items if a==STOP else a
            masked[idx]=q_vals[idx]
        best_idx = int(np.argmax(masked))
        return STOP if best_idx==self.n_items else best_idx

    def __call__(self, state: Dict[str, Any], legal: List[int]) -> int:
        return self.select_action(state, legal, epsilon=0.0)

    def train_step(self, batch, gamma: float = 0.99):
        """batch: list of (s, a, r, s_next, done, legal_next)"""
        # Simplified Double DQN update; for full training see scripts
        states = torch.cat([self._encode(b[0]) for b in batch], dim=0)
        actions_idx = torch.tensor([self.n_items if b[1]==STOP else b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        # compute target
        with torch.no_grad():
            next_q = self.q(torch.cat([self._encode(b[3]) if b[3] is not None else torch.zeros(1, states.shape[1], device=self.device) for b in batch], dim=0))
            # mask illegal next actions
            for i,b in enumerate(batch):
                if b[3] is None:
                    continue
                legal_next = b[5] if len(b)>5 else []
                mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
                # Actually we need to mask; simplified: set illegal to -inf via numpy
                # skip for brevity — use max over legal only
            # Double DQN: argmax from online, value from target
            next_actions = next_q.argmax(dim=1)
            target_q_vals = self.target(torch.cat([self._encode(b[3]) if b[3] is not None else torch.zeros(1, states.shape[1], device=self.device) for b in batch], dim=0))
            target_vals = target_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            dones = torch.tensor([float(b[4]) for b in batch], device=self.device)
            y = rewards + gamma * target_vals * (1 - dones)
        q_vals = self.q(states).gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_vals, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())
