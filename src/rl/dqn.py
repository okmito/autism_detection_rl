"""Model-free RL: Double DQN trained on replayed real response histories.

Why not the existing ``src/policies/dqn.py`` / ``ppo.py`` as-is
-------------------------------------------------------------
* Their state (``encode_state``: mask one-hot + answer bits + budget fraction) has no belief feature, and they
  were trained with a terminal reward against the *circular* Saudi label at a fixed budget with ``b_min = B``, so
  STOP was never a decision they could learn.
* ``DQNPolicy.train_step`` encodes one python dict per sample; a run over a few hundred records with a cost grid,
  seeds and nested cross-validation needs a vectorised loop.
* ``PPOPolicy.train_step`` uses the normalised raw reward as its advantage (no returns, no baseline from the
  critic), so it is a REINFORCE variant rather than PPO, and on-policy learning is the least sample-efficient
  choice for ~200 training records.

The Double-DQN update rule (online argmax, target evaluation, illegal next actions masked, terminal rows not
bootstrapped) is the same as ``DQNPolicy.train_step``. What changes is the MDP it is applied to (``src/rl/mdp.py``:
belief feature, per-question cost, a learnable STOP from ``min_questions`` on, a safety cap) and that terminal
rewards are computed with **cross-fitted** beliefs: a training record's belief comes from a model that did not see
that record, so the agent cannot learn that in-sample probabilities are more trustworthy than they are.

Inference is deterministic: an ensemble of seeds acts on the mean Q-value with the tie rule of
``src/rl/planner.greedy_action``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import (C_SKIP, FEATURE_DIM, MDPConfig, encode_features, legal_mask, obs_val,
                        terminal_quality)
from src.rl.planner import greedy_action


@dataclass(frozen=True)
class DQNConfig:
    hidden: int = 64
    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 256
    updates_per_epoch: int = 8
    buffer_size: int = 60000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6
    target_sync: int = 100
    gamma: float = 1.0
    crossfit_folds: int = 5
    weight_decay: float = 1e-4

    def to_dict(self):
        return asdict(self)


class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class _BeliefBank:
    """Row-specific beliefs: each training record is scored by the cross-fitted model that excluded it."""

    def __init__(self, models: Sequence[NaiveBayesBelief], row_model: np.ndarray):
        self.llr1 = np.stack([m.llr1 for m in models])
        self.llr0 = np.stack([m.llr0 for m in models])
        self.w = np.array([m.w for m in models])
        self.prior = np.array([m.prior_logit for m in models])
        self.row_model = np.asarray(row_model, int)

    def log_odds(self, codes: np.ndarray, rows: np.ndarray) -> np.ndarray:
        f = self.row_model[rows]
        o, v = obs_val(codes)
        contrib = np.where(v == 1, self.llr1[f], self.llr0[f])
        return self.prior[f] + self.w[f] * np.sum(np.where(o, contrib, 0.0), axis=1)


def _features(codes: np.ndarray, log_odds: np.ndarray, cfg: MDPConfig) -> np.ndarray:
    """Same layout as ``mdp.encode_features`` but with a supplied belief log-odds."""
    n = codes.shape[1]
    oh = np.zeros((codes.shape[0], 4 * n), dtype=np.float32)
    oh[np.arange(codes.shape[0])[:, None], np.arange(n)[None, :] * 4 + codes] = 1.0
    used = np.count_nonzero(codes, axis=1)
    extra = np.stack([np.clip(log_odds / 4.0, -3, 3), used / cfg.max_questions,
                      (cfg.max_questions - used) / cfg.max_questions], axis=1).astype(np.float32)
    return np.concatenate([oh, extra], axis=1)


def _stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def train_dqn(X: np.ndarray, y: np.ndarray, cfg: MDPConfig, dcfg: DQNConfig = DQNConfig(), *, seed: int = 0,
              belief_kwargs: Optional[Dict[str, Any]] = None, log: Optional[List[Dict[str, float]]] = None
              ) -> "QNet":
    """Train one Q-network on the records (X, y). Returns the online network (eval mode)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    M = np.isnan(X)
    Xi = np.where(M, 0, X).astype(np.int64)
    N, n = Xi.shape
    bk = dict(belief_kwargs or {})
    folds = _stratified_folds(y, dcfg.crossfit_folds, rng)
    models = [NaiveBayesBelief.fit(X[folds != f], y[folds != f], seed=seed, **bk) for f in range(dcfg.crossfit_folds)]
    bank = _BeliefBank(models, folds)

    net = QNet(FEATURE_DIM(n), n + 1, dcfg.hidden)
    tgt = QNet(FEATURE_DIM(n), n + 1, dcfg.hidden)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=dcfg.lr, weight_decay=dcfg.weight_decay)
    D = FEATURE_DIM(n)
    cap = dcfg.buffer_size
    buf_s = np.zeros((cap, D), np.float32)
    buf_s2 = np.zeros((cap, D), np.float32)
    buf_a = np.zeros(cap, np.int64)
    buf_r = np.zeros(cap, np.float32)
    buf_d = np.zeros(cap, np.float32)
    buf_l2 = np.zeros((cap, n + 1), bool)
    ptr = size = 0
    n_updates = 0
    all_rows = np.arange(N)

    for ep in range(dcfg.epochs):
        frac = min(1.0, ep / max(1, dcfg.epochs * dcfg.eps_decay_frac))
        eps = dcfg.eps_start + frac * (dcfg.eps_end - dcfg.eps_start)
        codes = np.zeros((N, n), dtype=np.int64)
        active = np.ones(N, bool)
        ep_return = np.zeros(N)
        while active.any():
            rows = all_rows[active]
            c = codes[rows]
            lo = bank.log_odds(c, rows)
            feat = _features(c, lo, cfg)
            lm = legal_mask(c, cfg)
            with torch.no_grad():
                q = net(torch.from_numpy(feat)).numpy().astype(float)
            q[~lm] = -np.inf
            act = greedy_action(q)
            explore = rng.random(len(rows)) < eps
            if explore.any():
                for i in np.flatnonzero(explore):
                    act[i] = rng.choice(np.flatnonzero(lm[i]))
            stop = act == n
            r = np.zeros(len(rows))
            nc = c.copy()
            ask_i = np.flatnonzero(~stop)
            if len(ask_i):
                rr, jj = rows[ask_i], act[ask_i]
                nc[ask_i, jj] = np.where(M[rr, jj], C_SKIP, 1 + Xi[rr, jj])
                r[ask_i] = -cfg.cost
            if stop.any():
                p = 1.0 / (1.0 + np.exp(-lo[stop]))
                r[stop] = terminal_quality(p, y[rows[stop]], cfg)
            lo2 = bank.log_odds(nc, rows)
            feat2 = _features(nc, lo2, cfg)
            lm2 = legal_mask(nc, cfg)
            k = len(rows)
            pos = (ptr + np.arange(k)) % cap
            buf_s[pos], buf_s2[pos], buf_a[pos], buf_r[pos] = feat, feat2, act, r
            buf_d[pos], buf_l2[pos] = stop.astype(np.float32), lm2
            ptr = (ptr + k) % cap
            size = min(cap, size + k)
            ep_return[rows] += r
            codes[rows] = nc
            active[rows[stop]] = False
        for _ in range(dcfg.updates_per_epoch):
            if size < dcfg.batch_size:
                break
            bi = rng.integers(0, size, dcfg.batch_size)
            s = torch.from_numpy(buf_s[bi])
            s2 = torch.from_numpy(buf_s2[bi])
            a = torch.from_numpy(buf_a[bi])
            r = torch.from_numpy(buf_r[bi])
            d = torch.from_numpy(buf_d[bi])
            l2 = torch.from_numpy(buf_l2[bi])
            with torch.no_grad():
                qn = net(s2).masked_fill(~l2, -1e9)
                a2 = qn.argmax(1, keepdim=True)
                qt = tgt(s2).gather(1, a2).squeeze(1)
                target = r + dcfg.gamma * (1 - d) * qt
            qsa = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(qsa, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_updates += 1
            if n_updates % dcfg.target_sync == 0:
                tgt.load_state_dict(net.state_dict())
        if log is not None and (ep % 25 == 0 or ep == dcfg.epochs - 1):
            log.append({"epoch": ep, "eps": float(eps), "mean_return": float(ep_return.mean())})
    net.eval()
    return net


class DQNAgent:
    """Deterministic ensemble of trained Q-networks acting on the full-training-set belief."""

    name = "rl_dqn"

    def __init__(self, nets: Sequence[QNet], belief: NaiveBayesBelief, cfg: MDPConfig,
                 dcfg: DQNConfig = DQNConfig(), seeds: Sequence[int] = ()):
        self.nets = list(nets)
        self.belief = belief
        self.cfg = cfg
        self.dcfg = dcfg
        self.seeds = list(seeds)

    @classmethod
    def train(cls, X, y, cfg: MDPConfig, dcfg: DQNConfig = DQNConfig(), seeds: Sequence[int] = (0, 1, 2),
              belief: Optional[NaiveBayesBelief] = None, belief_kwargs: Optional[Dict[str, Any]] = None,
              trained_on: str = "unknown") -> "DQNAgent":
        torch.set_num_threads(1)
        bk = dict(belief_kwargs or {})
        if belief is None:
            belief = NaiveBayesBelief.fit(X, y, trained_on=trained_on, **bk)
        nets = [train_dqn(X, y, cfg, dcfg, seed=s, belief_kwargs=bk) for s in seeds]
        return cls(nets, belief, cfg, dcfg, seeds)

    def q_values(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        flat = codes.reshape(-1, self.cfg.n_items)
        o, v = obs_val(flat)
        feat = _features(flat, self.belief.log_odds_arrays(o, v), self.cfg)
        with torch.no_grad():
            x = torch.from_numpy(feat)
            q = np.mean([net(x).numpy() for net in self.nets], axis=0).astype(float)
        q[~legal_mask(flat, self.cfg)] = -np.inf
        return q.reshape(codes.shape[:-1] + (self.cfg.n_items + 1,))

    def act_batch(self, codes: np.ndarray) -> np.ndarray:
        return greedy_action(self.q_values(codes))

    def describe(self) -> Dict[str, Any]:
        return {"agent": self.name, "kind": "model-free RL (Double DQN, cross-fitted terminal rewards)",
                "seeds": self.seeds, "config": self.cfg.to_dict(), "dqn": self.dcfg.to_dict()}

    # persistence
    def state_dicts(self):
        return [net.state_dict() for net in self.nets]

    @classmethod
    def from_state_dicts(cls, sds, belief: NaiveBayesBelief, cfg: MDPConfig, dcfg: DQNConfig, seeds=()):
        nets = []
        for sd in sds:
            net = QNet(FEATURE_DIM(cfg.n_items), cfg.n_items + 1, dcfg.hidden)
            net.load_state_dict(sd)
            net.eval()
            nets.append(net)
        return cls(nets, belief, cfg, dcfg, seeds)
