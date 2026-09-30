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

    def select_action(self, state, legal, deterministic: bool = False):
        """Sample from the legal-action distribution (or take the argmax).

        ``deterministic=True`` reproduces the previous ``__call__`` behaviour.
        """
        with torch.no_grad():
            logits = self.actor(self._encode(state)).cpu().numpy()[0]
        masked = np.full(self.n_items + 1, float("-inf"))
        for a in legal:
            idx = self.n_items if a == STOP else a
            masked[idx] = logits[idx]
        if deterministic:
            best = int(np.argmax(masked))
            return STOP if best == self.n_items else best
        m = np.max(masked[np.isfinite(masked)])
        exps = np.exp(masked - m)
        exps[~np.isfinite(masked)] = 0.0
        probs = exps / exps.sum()
        idx = int(np.random.choice(len(probs), p=probs))
        return STOP if idx == self.n_items else idx

    def legal_mask(self, legal) -> torch.Tensor:
        """Boolean mask over actions, True where the action is legal."""
        mask = torch.zeros(self.n_items + 1, dtype=torch.bool, device=self.device)
        for a in legal:
            mask[self.n_items if a == STOP else a] = True
        return mask

    def __call__(self, state, legal):
        return self.select_action(state, legal, deterministic=True)

    def train_step(self, batch, clip_eps: float = 0.2, value_coef: float = 0.5,
                   entropy_coef: float = 0.01, old_logp: list | None = None):
        """One PPO update over ``(s, a, r, s_next, done, legal_next)`` transitions.

        The batch format matches :class:`~src.policies.dqn.DQNPolicy.train_step`
        and :class:`~src.policies.replay.ReplayBuffer`, so both algorithms can
        consume the same collected data.

        An earlier revision of this class had no training method at all: the
        critic was constructed but never received a gradient, so ``PPOPolicy``
        behaved as an untrained sampler. This implements the standard clipped
        surrogate objective plus value loss and an entropy bonus.

        ``old_logp`` is the log-probability of the taken action *before* the
        update. When omitted it is recomputed from the current policy, which
        makes the first epoch effectively a plain policy-gradient step; pass the
        pre-update values for a faithful multi-epoch PPO update.
        """
        if not batch:
            raise ValueError("train_step requires a non-empty batch")

        states = torch.cat([self._encode(b[0]) for b in batch], dim=0)
        actions_idx = torch.tensor(
            [self.n_items if b[1] == STOP else b[1] for b in batch],
            dtype=torch.long, device=self.device,
        )
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        legal_masks = torch.stack([self.legal_mask(b[5]) for b in batch], dim=0)

        logits = self.actor(states)
        # Never assign probability to an illegal action.
        logits = logits.masked_fill(~legal_masks, float("-inf"))
        logprobs_all = torch.log_softmax(logits, dim=-1)
        probs_all = torch.softmax(logits, dim=-1)
        entropy = -(probs_all * torch.nan_to_num(logprobs_all, neginf=0.0)).sum(dim=-1).mean()

        if old_logp is None:
            old_logp_t = logprobs_all.gather(1, actions_idx.unsqueeze(1)).squeeze(1).detach()
        else:
            old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=self.device)

        logp = logprobs_all.gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(logp - old_logp_t)
        adv = rewards
        if adv.numel() > 1:
            # Unbiased std is undefined for a single sample; skip normalising.
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        surrogate = torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
        ).mean()
        values = self.critic(states).squeeze(-1)
        value_loss = nn.MSELoss()(values, rewards)

        loss = -surrogate + value_coef * value_loss - entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(-surrogate.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
