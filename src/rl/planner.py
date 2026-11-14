"""Model-based RL: exact value iteration on the learned belief MDP.

The environment model is learned from data (:class:`~src.rl.belief.NaiveBayesBelief`: P(y | history) and
P(next answer | history)). With ten binary items every history is one of 4^10 = 1,048,576 codes, so the Bellman
equation can be solved *exactly* by backward induction over the number of questions used::

    Q(h, STOP)  = E[q(p_hat(h), y) | h]
    Q(h, ask j) = -cost + P(x_j=0|h) V(h + (j,0)) + P(x_j=1|h) V(h + (j,1))
    V(h)        = max over legal actions

This is the Bayes-optimal sequential policy *under the learned model*. Its weaknesses are the model's: if the
naive-Bayes independence assumption or its calibration is wrong, so is the plan. The model-free agent
(``src/rl/dqn.py``) learns from real answer sequences and real labels and does not share that assumption, which
is why both are trained and compared.

Ties: STOP wins a tie with the best question (no question is asked unless it is worth strictly more than
stopping); among questions the lowest index wins. Both rules are deterministic.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import C_NEG, C_POS, C_UNASKED, MDPConfig, expected_terminal_quality

TIE_TOL = 1e-12


class BeliefValueIteration:
    """Solve the belief MDP once; then act by table lookup (deterministic)."""

    name = "rl_value_iteration"

    def __init__(self, belief: NaiveBayesBelief, cfg: MDPConfig):
        if belief.n_items != cfg.n_items:
            raise ValueError("belief and config disagree on n_items")
        if cfg.n_items > 12:
            raise ValueError("exact value iteration is limited to <= 12 items")
        self.belief = belief
        self.cfg = cfg
        self.n = cfg.n_items
        self.pow4 = (4 ** np.arange(self.n)).astype(np.int64)
        self._solve()

    # ------------------------------------------------------------------ solve
    def _digits(self, idx: np.ndarray) -> np.ndarray:
        return (idx[:, None] // self.pow4[None, :]) % 4

    def _solve(self):
        n, cfg, b = self.n, self.cfg, self.belief
        S = 4 ** n
        idx = np.arange(S, dtype=np.int64)
        digits = self._digits(idx).astype(np.int8)
        used = np.count_nonzero(digits, axis=1)
        contrib = np.zeros((n, 4))
        contrib[:, C_NEG] = b.llr0
        contrib[:, C_POS] = b.llr1
        ev = contrib[np.arange(n)[None, :], digits].sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-(b.prior_logit + b.w * ev)))
        stop_val = expected_terminal_quality(p, cfg)
        V = np.empty(S)
        for k in range(n, -1, -1):
            lvl = np.flatnonzero(used == k)
            if k >= cfg.max_questions:
                V[lvl] = stop_val[lvl]
                continue
            best_ask = np.full(len(lvl), -np.inf)
            pl = p[lvl]
            for j in range(n):
                can = digits[lvl, j] == C_UNASKED
                if not can.any():
                    continue
                s = lvl[can]
                a1 = b.theta[j, 1] * pl[can] + b.theta[j, 0] * (1 - pl[can])
                q = -cfg.cost + (1 - a1) * V[s + C_NEG * self.pow4[j]] + a1 * V[s + C_POS * self.pow4[j]]
                best_ask[can] = np.maximum(best_ask[can], q)
            stop_legal = (k >= cfg.min_questions) | ~np.isfinite(best_ask)
            V[lvl] = np.where(stop_legal, np.maximum(stop_val[lvl], best_ask), best_ask)
        self.V = V
        self.p = p
        self.stop_val = stop_val
        self.digits = digits

    # ------------------------------------------------------------------ act
    def index_of(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.int64) @ self.pow4

    def q_values(self, codes: np.ndarray) -> np.ndarray:
        """Q for every action (n asks + STOP); -inf for illegal actions. ``codes`` shape (..., n)."""
        from src.rl.mdp import legal_mask
        codes = np.asarray(codes, dtype=np.int64)
        flat = codes.reshape(-1, self.n)
        s = flat @ self.pow4
        p = self.p[s]
        lm = legal_mask(flat, self.cfg)
        Q = np.full((len(s), self.n + 1), -np.inf)
        for j in range(self.n):
            ok = lm[:, j]
            if not ok.any():
                continue
            a1 = self.belief.theta[j, 1] * p[ok] + self.belief.theta[j, 0] * (1 - p[ok])
            Q[ok, j] = (-self.cfg.cost + (1 - a1) * self.V[s[ok] + C_NEG * self.pow4[j]]
                        + a1 * self.V[s[ok] + C_POS * self.pow4[j]])
        Q[lm[:, -1], -1] = self.stop_val[s[lm[:, -1]]]
        return Q.reshape(codes.shape[:-1] + (self.n + 1,))

    def act_batch(self, codes: np.ndarray) -> np.ndarray:
        return greedy_action(self.q_values(codes))

    def decompose(self, codes: np.ndarray, j: int) -> Dict[str, Any]:
        """Exact decomposition of Q(h, ask j) into its two answer branches (for explanations)."""
        codes = np.asarray(codes, dtype=np.int64)
        s = int(codes @ self.pow4)
        p = float(self.p[s])
        a1 = float(self.belief.theta[j, 1] * p + self.belief.theta[j, 0] * (1 - p))
        s0, s1 = s + C_NEG * int(self.pow4[j]), s + C_POS * int(self.pow4[j])
        return {"p_answer_atypical": a1, "p_after_typical": float(self.p[s0]), "p_after_atypical": float(self.p[s1]),
                "value_after_typical": float(self.V[s0]), "value_after_atypical": float(self.V[s1]),
                "cost": self.cfg.cost}

    def describe(self) -> Dict[str, Any]:
        return {"agent": self.name, "kind": "model-based RL (exact value iteration on a learned belief MDP)",
                "n_states": int(4 ** self.n), "config": self.cfg.to_dict(), "V_root": float(self.V[0])}


def greedy_action(Q: np.ndarray) -> np.ndarray:
    """Deterministic argmax over (asks..., STOP): STOP wins ties with the best ask; lowest index among asks."""
    Q = np.asarray(Q, float)
    ask = Q[..., :-1]
    best_j = np.argmax(ask, axis=-1)                       # first maximum = lowest index
    best_q = np.take_along_axis(ask, best_j[..., None], axis=-1)[..., 0]
    stop_q = Q[..., -1]
    take_stop = np.isfinite(stop_q) & (stop_q >= best_q - TIE_TOL)
    take_stop |= ~np.isfinite(best_q)
    return np.where(take_stop, Q.shape[-1] - 1, best_j)
