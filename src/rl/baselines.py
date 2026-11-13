"""Comparison agents with the same ``act_batch(codes) -> actions`` interface as the RL agents.

Fixed-length arms never STOP before ``k`` questions and always STOP at ``k`` (``k = n`` is the full questionnaire).
Every arm is scored by the same belief model at STOP, so differences are differences in *which* and *how many*
questions were asked, never in the scorer.

Adapters wrap the pre-existing policies (``GreedyIGPolicy``, ``BetaGreedyPolicy``, the Saudi-trained DQN v3
artifacts) without modifying them; they are offered only the unasked items, never STOP, so they run fixed-length.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import C_NEG, C_POS, C_SKIP, C_UNASKED, obs_val


def _binary_entropy(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class FixedOrder:
    """Ask ``order[:k]`` in that order, then STOP."""

    def __init__(self, order: Sequence[int], k: int, name: Optional[str] = None):
        self.order = [int(j) for j in order]
        self.k = int(k)
        self.n = len(self.order)
        self.name = name or f"fixed_order_k{k}"

    def act_batch(self, codes):
        codes = np.asarray(codes)
        out = np.full(len(codes), self.n, dtype=np.int64)
        used = np.count_nonzero(codes, axis=1)
        for i in range(len(codes)):
            if used[i] >= self.k:
                continue
            for j in self.order:
                if codes[i, j] == C_UNASKED:
                    out[i] = j
                    break
        return out


class RandomFixedLength:
    """A fresh uniformly random item order per record (seeded), ``k`` questions, then STOP."""

    def __init__(self, n: int, k: int, seed: int = 0):
        self.n, self.k = n, k
        self.rng = np.random.default_rng(seed)
        self.name = f"random_fixed_k{k}"

    def act_batch(self, codes):
        codes = np.asarray(codes)
        out = np.full(len(codes), self.n, dtype=np.int64)
        used = np.count_nonzero(codes, axis=1)
        for i in range(len(codes)):
            if used[i] < self.k:
                out[i] = int(self.rng.choice(np.flatnonzero(codes[i] == C_UNASKED)))
        return out


def forward_selection_order(belief: NaiveBayesBelief, X: np.ndarray, y: np.ndarray, tau: float = 0.5) -> List[int]:
    """Best fixed subsets, nested: add the item that most improves training accuracy (ties: log-likelihood)."""
    X = np.where(np.isnan(X), 0, X).astype(int)
    n = X.shape[1]
    chosen: List[int] = []
    for _ in range(n):
        best, best_key = None, None
        for j in range(n):
            if j in chosen:
                continue
            obs = np.zeros_like(X, dtype=bool)
            obs[:, chosen + [j]] = True
            p = belief.prob_arrays(obs, X)
            acc = np.mean((p >= tau) == y)
            ll = np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
            key = (acc, ll, -j)
            if best_key is None or key > best_key:
                best, best_key = j, key
        chosen.append(best)
    return chosen


class BeliefGreedyIG:
    """Ask the item with the largest expected entropy reduction *under the belief model*; fixed length ``k``."""

    def __init__(self, belief: NaiveBayesBelief, k: int):
        self.b, self.k, self.n = belief, k, belief.n_items
        self.name = f"belief_greedy_ig_k{k}"

    def scores(self, codes):
        o, v = obs_val(codes)
        lo = self.b.log_odds_arrays(o, v)
        p = 1 / (1 + np.exp(-lo))
        a1 = self.b.answer_prob(p)
        p1 = 1 / (1 + np.exp(-(lo[:, None] + self.b.w * self.b.llr1[None, :])))
        p0 = 1 / (1 + np.exp(-(lo[:, None] + self.b.w * self.b.llr0[None, :])))
        ig = _binary_entropy(p)[:, None] - (a1 * _binary_entropy(p1) + (1 - a1) * _binary_entropy(p0))
        ig[np.asarray(codes) != C_UNASKED] = -np.inf
        return ig, p

    def act_batch(self, codes):
        codes = np.asarray(codes)
        ig, _ = self.scores(codes)
        used = np.count_nonzero(codes, axis=1)
        return np.where(used >= self.k, self.n, np.argmax(ig, axis=1))


class ThresholdStopping(BeliefGreedyIG):
    """Non-RL adaptive baseline: belief-greedy item choice, STOP once P(positive) leaves [1-hi, hi].

    A sequential-probability-ratio-style rule. It isolates what RL adds beyond "stop when confident".
    """

    def __init__(self, belief: NaiveBayesBelief, hi: float, min_questions: int, max_questions: int):
        super().__init__(belief, max_questions)
        self.hi, self.min_q = float(hi), int(min_questions)
        self.name = f"threshold_stop_{hi:g}"

    def act_batch(self, codes):
        codes = np.asarray(codes)
        ig, p = self.scores(codes)
        used = np.count_nonzero(codes, axis=1)
        confident = (p >= self.hi) | (p <= 1 - self.hi)
        stop = (used >= self.k) | ((used >= self.min_q) & confident) | ~np.isfinite(ig.max(axis=1))
        return np.where(stop, self.n, np.argmax(ig, axis=1))


def _state_dict(codes_row: np.ndarray, budget: int) -> Dict[str, Any]:
    mask = np.zeros(len(codes_row), dtype=int)
    value = np.full(len(codes_row), -1, dtype=int)
    mask[(codes_row == C_NEG) | (codes_row == C_POS)] = 1
    mask[codes_row == C_SKIP] = 2
    value[codes_row == C_NEG] = 0
    value[codes_row == C_POS] = 1
    used = int(np.count_nonzero(codes_row))
    return {"mask": mask, "value": value, "n": len(codes_row), "questions_remaining": budget - used, "budget": budget}


class LegacyPolicyAdapter:
    """Run an existing ``policy(state_dict, legal) -> item`` for exactly ``k`` questions (STOP never offered)."""

    def __init__(self, policy: Callable, n: int, k: int, name: str):
        self.policy, self.n, self.k, self.name = policy, n, k, name

    def act_batch(self, codes):
        codes = np.asarray(codes)
        out = np.full(len(codes), self.n, dtype=np.int64)
        for i in range(len(codes)):
            used = int(np.count_nonzero(codes[i]))
            if used >= self.k:
                continue
            legal = [int(j) for j in np.flatnonzero(codes[i] == C_UNASKED)]
            if not legal:
                continue
            j = int(self.policy(_state_dict(codes[i], self.k), legal))
            if j not in legal:
                raise ValueError(f"{self.name} chose illegal item {j}")
            out[i] = j
        return out


def legacy_records(X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    """The record format GreedyIGPolicy / BetaGreedyPolicy are fitted on."""
    X = np.asarray(X, float)
    return [{"item_responses": X[i], "label": int(y[i]), "missing_mask": np.isnan(X[i])} for i in range(len(y))]
