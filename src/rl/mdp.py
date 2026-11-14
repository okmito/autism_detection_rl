"""The adaptive-screening MDP: states, legal actions, rewards, and a vectorised replay environment.

State
-----
A history is a vector of per-item codes::

    0 = not asked      1 = answered "typical" (0)      2 = answered "atypical" (1)      3 = asked, no answer (skipped)

Everything an agent may condition on is a function of the codes: which items were asked, their answers, how many
questions were used (skips count: they cost the respondent a question), the remaining budget, and the belief
P(screen-positive | answers) from :class:`~src.rl.belief.NaiveBayesBelief`. Nothing that is only known after the
episode (the label, unasked answers) is part of the state.

Actions
-------
``0..n-1`` ASK that item (legal iff its code is 0) and ``n`` STOP. STOP is legal once ``min_questions`` have been
used; it is forced when ``max_questions`` are used or no askable item is left. An already-asked item is never legal.

Reward (primary)
----------------
``-cost`` for every question asked, and at STOP a terminal screening-quality term ``q(p_hat, y)``:

* ``accuracy`` (primary): ``1[decision(p_hat, tau) == y]``. Its expectation is the probability of a correct
  screening decision, so the episode return is ``accuracy - cost * E[questions]``: the Lagrangian of
  "minimise expected questions subject to a screening-accuracy constraint". On a near-balanced cohort accuracy and
  balanced accuracy coincide closely; for another prevalence the two class weights must be revisited.
* ``brier`` (sensitivity analysis): ``1 - (p_hat - y)^2``, a proper scoring rule that also rewards calibrated
  probabilities and ranking.

``y`` must be an independent reference label (clinician-established in the primary experiment). A reward computed
from a label that is itself a function of the questionnaire answers would only teach the agent to reproduce the
scoring formula; see ``RL_ADAPTIVE_REPORT.md``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.rl.belief import MISSING, OBSERVED, UNASKED, NaiveBayesBelief

C_UNASKED, C_NEG, C_POS, C_SKIP = 0, 1, 2, 3
TERMINALS = ("accuracy", "brier")


@dataclass(frozen=True)
class MDPConfig:
    n_items: int = 10
    max_questions: int = 10
    min_questions: int = 2
    cost: float = 0.01
    terminal: str = "accuracy"
    tau: float = 0.5

    def __post_init__(self):
        if self.terminal not in TERMINALS:
            raise ValueError(f"terminal must be one of {TERMINALS}")
        if not 1 <= self.max_questions <= self.n_items:
            raise ValueError("max_questions must be in 1..n_items")
        if not 0 <= self.min_questions <= self.max_questions:
            raise ValueError("min_questions must be in 0..max_questions")
        if self.cost < 0:
            raise ValueError("cost must be >= 0")
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be in (0, 1)")

    @property
    def stop_action(self) -> int:
        return self.n_items

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------- terminal quality
def terminal_quality(p: np.ndarray, y: np.ndarray, cfg: MDPConfig) -> np.ndarray:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if cfg.terminal == "accuracy":
        return ((p >= cfg.tau).astype(float) == y).astype(float)
    return 1.0 - (p - y) ** 2


def expected_terminal_quality(p: np.ndarray, cfg: MDPConfig) -> np.ndarray:
    """E[q(p, y)] when y ~ Bernoulli(p) — the value of stopping under the belief model."""
    p = np.asarray(p, float)
    if cfg.terminal == "accuracy":
        return np.where(p >= cfg.tau, p, 1.0 - p)
    return 1.0 - p * (1.0 - p)


# ---------------------------------------------------------------------------- codes <-> state dicts
def codes_from_state(state: Dict[str, Any]) -> np.ndarray:
    mask = np.asarray(state["mask"], int)
    val = np.asarray(state["value"], int)
    codes = np.zeros(mask.shape, dtype=np.int64)
    codes[mask == OBSERVED] = 1 + np.clip(val[mask == OBSERVED], 0, 1)
    codes[mask == MISSING] = C_SKIP
    return codes


def obs_val(codes: np.ndarray):
    codes = np.asarray(codes)
    return (codes == C_NEG) | (codes == C_POS), (codes == C_POS).astype(int)


def n_used(codes: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(codes) != C_UNASKED, axis=-1)


def legal_mask(codes: np.ndarray, cfg: MDPConfig) -> np.ndarray:
    """Boolean (..., n+1): askable items, then STOP."""
    codes = np.asarray(codes)
    used = n_used(codes)
    ask = codes == C_UNASKED
    budget_left = used < cfg.max_questions
    ask = ask & budget_left[..., None]
    any_ask = ask.any(axis=-1)
    stop = (used >= cfg.min_questions) | ~any_ask
    return np.concatenate([ask, stop[..., None]], axis=-1)


def is_forced_stop(codes: np.ndarray, cfg: MDPConfig) -> np.ndarray:
    lm = legal_mask(codes, cfg)
    return ~lm[..., :-1].any(axis=-1)


def encode_features(codes: np.ndarray, belief: NaiveBayesBelief, cfg: MDPConfig) -> np.ndarray:
    """Network input: one-hot codes (4n) + [belief log-odds / 4, used / B, remaining / B]."""
    codes = np.asarray(codes, dtype=np.int64)
    flat = codes.reshape(-1, codes.shape[-1])
    n = flat.shape[1]
    oh = np.zeros((flat.shape[0], 4 * n), dtype=np.float32)
    rows = np.arange(flat.shape[0])[:, None]
    oh[rows, np.arange(n)[None, :] * 4 + flat] = 1.0
    o, v = obs_val(flat)
    lo = belief.log_odds_arrays(o, v)
    used = n_used(flat)
    extra = np.stack([np.clip(lo / 4.0, -3, 3), used / cfg.max_questions,
                      (cfg.max_questions - used) / cfg.max_questions], axis=1).astype(np.float32)
    out = np.concatenate([oh, extra], axis=1)
    return out.reshape(codes.shape[:-1] + (out.shape[1],))


FEATURE_DIM = lambda n: 4 * n + 3  # noqa: E731


# ---------------------------------------------------------------------------- batch replay environment
class ReplayBatch:
    """Run many real response records in parallel. The answer to an ASK is the record's recorded answer.

    Records with a missing answer to an asked item get code 3 (skip): the question is spent, no evidence.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: MDPConfig, missing: Optional[np.ndarray] = None):
        X = np.asarray(X, float)
        self.M = np.isnan(X) if missing is None else np.asarray(missing, bool)
        self.X = np.where(self.M, 0, X).astype(np.int64)
        self.y = np.asarray(y, int)
        self.cfg = cfg
        self.N = len(self.y)
        self.reset()

    def reset(self):
        self.codes = np.zeros((self.N, self.cfg.n_items), dtype=np.int64)
        self.done = np.zeros(self.N, dtype=bool)
        return self.codes

    def answer_codes(self, rows: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.where(self.M[rows, items], C_SKIP, 1 + self.X[rows, items])

    def step(self, actions: np.ndarray, belief: NaiveBayesBelief):
        """Apply one action per active row. Returns (reward, done) arrays for all rows (0 for finished rows)."""
        a = np.asarray(actions, dtype=np.int64)
        lm = legal_mask(self.codes, self.cfg)
        active = ~self.done
        rows = np.flatnonzero(active)
        if not np.all(lm[rows, a[rows]]):
            bad = rows[~lm[rows, a[rows]]]
            raise ValueError(f"illegal action(s) for rows {bad[:5].tolist()}")
        reward = np.zeros(self.N)
        stop_rows = rows[a[rows] == self.cfg.stop_action]
        ask_rows = rows[a[rows] != self.cfg.stop_action]
        if len(ask_rows):
            items = a[ask_rows]
            self.codes[ask_rows, items] = self.answer_codes(ask_rows, items)
            reward[ask_rows] = -self.cfg.cost
        if len(stop_rows):
            o, v = obs_val(self.codes[stop_rows])
            p = belief.prob_arrays(o, v)
            reward[stop_rows] = terminal_quality(p, self.y[stop_rows], self.cfg)
            self.done[stop_rows] = True
        return reward, self.done.copy()
