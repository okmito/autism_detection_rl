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
        """One Double-DQN update.

        ``batch`` is a list of ``(s, a, r, s_next, done, legal_next)`` tuples as
        produced by :class:`src.policies.replay.ReplayBuffer`. ``s_next`` is
        ``None`` for terminal transitions; ``legal_next`` is ignored then.
        """
        if not batch:
            raise ValueError("train_step requires a non-empty batch")

        states = torch.cat([self._encode(b[0]) for b in batch], dim=0)
        actions_idx = torch.tensor(
            [self.n_items if b[1]==STOP else b[1] for b in batch],
            dtype=torch.long, device=self.device,
        )
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(b[4]) for b in batch], device=self.device)

        with torch.no_grad():
            # Build the next-state tensor. Terminal rows are zero-filled so all
            # rows share a shape: an earlier revision used (1, D) for terminal
            # rows, which made torch.cat raise for any batch containing a
            # terminal transition -- the common case, since STOP is legal.
            next_states, legal_next = [], []
            for b in batch:
                if b[3] is None:
                    next_states.append(torch.zeros(1, states.shape[1], device=self.device))
                    legal_next.append(None)
                else:
                    next_states.append(self._encode(b[3]))
                    legal_next.append(b[5] if len(b) > 5 else None)
            next_states = torch.cat(next_states, dim=0)

            # Legal-action mask over next states. Illegal actions (already-asked
            # items, STOP before b_min) must never contribute to the bootstrap
            # target. An earlier revision built this mask and discarded it, so
            # the target optimised toward illegal actions -- the one behaviour a
            # budgeted questionnaire policy must not learn.
            next_mask = torch.full(
                (next_states.shape[0], self.n_actions), float("-inf"), device=self.device
            )
            for i, legal in enumerate(legal_next):
                if legal is None:
                    continue
                for a in legal:
                    next_mask[i, self.n_items if a == STOP else a] = 0.0
            illegal = next_mask == float("-inf")

            # Double DQN: argmax chosen by the online net, value taken from target.
            next_actions = self.q(next_states).masked_fill(illegal, float("-inf")).argmax(dim=1)
            target_vals = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Terminal rows contribute nothing: done == 1 zeroes the bootstrap.
            y = rewards + gamma * target_vals * (1 - dones)

        q_vals = self.q(states).gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_vals, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())
