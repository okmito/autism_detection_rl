"""Calibrated naive-Bayes belief over partial Q-CHAT-10 histories.

Why this model
--------------
An adaptive questionnaire is evaluated on *partial* response histories: after t answers the system must
state P(screen-positive | the answers so far), and an agent deciding whether to ask item j needs the
predictive probability of each answer, P(x_j = v | answers so far). A class-conditional (naive Bayes)
model provides both from one set of parameters:

    log-odds(h) = logit(pi) + w * sum_{j observed} LLR_j(x_j)
    LLR_j(1)    = log(theta_j1 / theta_j0),   LLR_j(0) = log((1 - theta_j1) / (1 - theta_j0))
    P(x_j=1|h)  = theta_j1 * P(y=1|h) + theta_j0 * P(y=0|h)

with ``theta_jy = P(x_j = 1 | y)`` (Beta(a, a) smoothed) and ``pi`` the training prevalence. Unobserved
and MISSING items are marginalised exactly (they contribute nothing), so a skipped question never counts as
evidence either way.

Q-CHAT items are correlated, so the independence assumption over-counts evidence and the raw log-odds is
over-confident. ``w`` (the *evidence weight*, w <= 1 shrinks) corrects this: it is a one-parameter logistic
recalibration fitted on cross-fitted (out-of-fold) partial-history log-odds of the training data only. The
recalibrated predictive P(x_j|h) is an approximation (it uses the recalibrated P(y|h)); this is stated in the
metadata and in the report.

Every quantity is cheap and closed-form, which is what makes exact value iteration over all 4^10 history
codes possible (``src/rl/planner.py``) and what makes per-answer explanations exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

UNASKED, OBSERVED, MISSING = 0, 1, 2   # same mask codes as src/env/state.py
_EPS = 1e-9


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_offset_logistic(score: np.ndarray, offset: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                        iters: int = 50) -> float:
    """Newton fit of y ~ Bernoulli(sigmoid(offset + w * score)) for the single slope ``w``."""
    score = np.asarray(score, float)
    offset = np.asarray(offset, float)
    y = np.asarray(y, float)
    w = 1.0
    for _ in range(iters):
        p = _sigmoid(offset + w * score)
        g = np.sum((p - y) * score) + l2 * (w - 1.0)
        h = np.sum(p * (1 - p) * score * score) + l2
        step = g / max(h, 1e-12)
        w -= step
        if abs(step) < 1e-10:
            break
    return float(w)


@dataclass
class NaiveBayesBelief:
    """theta[j, y] = P(x_j = 1 | y); pi = P(y = 1); w = evidence weight. Immutable after fit."""
    theta: np.ndarray
    pi: float
    w: float = 1.0
    smoothing: float = 1.0
    n_train: int = 0
    trained_on: str = "unknown"
    metadata_extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _counts(X: np.ndarray, M: np.ndarray, y: np.ndarray, a: float):
        n = X.shape[1]
        theta = np.zeros((n, 2))
        for cls in (0, 1):
            rows = y == cls
            obs = ~M[rows]
            ones = (X[rows] == 1) & obs
            theta[:, cls] = (ones.sum(0) + a) / (obs.sum(0) + 2 * a)
        return theta

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, missing: Optional[np.ndarray] = None, *, smoothing: float = 1.0,
            calibrate: bool = True, n_folds: int = 5, states_per_record: int = 20, seed: int = 0,
            trained_on: str = "unknown") -> "NaiveBayesBelief":
        """Fit theta/pi on (X, y); fit the evidence weight on cross-fitted partial histories.

        ``X`` holds 0/1 answers (NaN or any value where ``missing`` is True). Calibration draws, for every
        record, ``states_per_record`` histories whose size is uniform on 1..n and whose items are a uniform
        random subset — the same distribution of "how much has been asked" the agents face — and scores them
        with a model that never saw that record (``n_folds``-fold cross-fitting).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        M = np.isnan(X) if missing is None else np.asarray(missing, bool)
        Xi = np.where(M, 0, X).astype(int)
        theta = cls._counts(Xi, M, y, smoothing)
        pi = float((y.sum() + 1.0) / (len(y) + 2.0))
        w = 1.0
        info: Dict[str, Any] = {"calibrated": False}
        if calibrate and len(y) >= 2 * n_folds and len(np.unique(y)) == 2:
            rng = np.random.default_rng(seed)
            folds = np.empty(len(y), dtype=int)
            for cls_ in (0, 1):                      # stratified fold assignment
                idx = np.flatnonzero(y == cls_)
                rng.shuffle(idx)
                folds[idx] = np.arange(len(idx)) % n_folds
            n = X.shape[1]
            scores, offsets, ys = [], [], []
            for f in range(n_folds):
                tr, te = folds != f, folds == f
                th = cls._counts(Xi[tr], M[tr], y[tr], smoothing)
                p_f = float((y[tr].sum() + 1.0) / (tr.sum() + 2.0))
                l1, l0 = np.log(th[:, 1] / th[:, 0]), np.log((1 - th[:, 1]) / (1 - th[:, 0]))
                for r in np.flatnonzero(te):
                    for _ in range(states_per_record):
                        k = int(rng.integers(1, n + 1))
                        sub = rng.choice(n, size=k, replace=False)
                        sub = sub[~M[r, sub]]
                        s = float(np.sum(np.where(Xi[r, sub] == 1, l1[sub], l0[sub])))
                        scores.append(s)
                        offsets.append(_logit(p_f))
                        ys.append(y[r])
            w = fit_offset_logistic(np.array(scores), np.array(offsets), np.array(ys))
            w = float(np.clip(w, 0.05, 1.5))
            info = {"calibrated": True, "n_folds": n_folds, "states_per_record": states_per_record,
                    "n_calibration_states": len(ys), "seed": seed}
        return cls(theta=theta, pi=pi, w=w, smoothing=smoothing, n_train=int(len(y)), trained_on=trained_on,
                   metadata_extra=info)

    # ------------------------------------------------------------------ core quantities
    @property
    def n_items(self) -> int:
        return int(self.theta.shape[0])

    @property
    def llr1(self) -> np.ndarray:
        return np.log(self.theta[:, 1] / self.theta[:, 0])

    @property
    def llr0(self) -> np.ndarray:
        return np.log((1 - self.theta[:, 1]) / (1 - self.theta[:, 0]))

    @property
    def prior_logit(self) -> float:
        return float(_logit(self.pi))

    def evidence(self, obs: np.ndarray, val: np.ndarray) -> np.ndarray:
        """Sum of per-answer LLRs (before the evidence weight). ``obs`` bool (..., n), ``val`` 0/1."""
        obs = np.asarray(obs, bool)
        val = np.asarray(val)
        contrib = np.where(val == 1, self.llr1, self.llr0)
        return np.sum(np.where(obs, contrib, 0.0), axis=-1)

    def log_odds_arrays(self, obs: np.ndarray, val: np.ndarray) -> np.ndarray:
        return self.prior_logit + self.w * self.evidence(obs, val)

    def prob_arrays(self, obs: np.ndarray, val: np.ndarray) -> np.ndarray:
        return _sigmoid(self.log_odds_arrays(obs, val))

    def answer_prob(self, p_pos: np.ndarray | float) -> np.ndarray:
        """P(x_j = 1 | h) for every item j given P(y=1|h). Shape (..., n)."""
        p = np.asarray(p_pos, dtype=float)[..., None]
        return self.theta[:, 1] * p + self.theta[:, 0] * (1 - p)

    # ------------------------------------------------------------------ state-dict interface
    @staticmethod
    def _obs_val(state: Dict[str, Any]):
        mask = np.asarray(state["mask"], dtype=int)
        val = np.asarray(state["value"], dtype=int)
        return mask == OBSERVED, np.where(mask == OBSERVED, val, 0)

    def predict_state(self, state: Dict[str, Any]) -> float:
        obs, val = self._obs_val(state)
        return float(self.prob_arrays(obs, val))

    __call__ = predict_state

    def log_odds_state(self, state: Dict[str, Any]) -> float:
        obs, val = self._obs_val(state)
        return float(self.log_odds_arrays(obs, val))

    def item_effect(self, j: int, v: int) -> Dict[str, float]:
        """How one answer moves the belief: log-odds shift and odds multiplier (after the evidence weight)."""
        llr = float(self.llr1[j] if v == 1 else self.llr0[j]) * self.w
        return {"log_odds_shift": llr, "odds_multiplier": float(np.exp(llr))}

    # ------------------------------------------------------------------ persistence / provenance
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"predictor_version": "nb-belief/1", "calibration_method": "evidence-weight (cross-fitted)",
                "trained_on": self.trained_on, "external_cohort_used": None, "evidence_weight": self.w,
                "n_train": self.n_train}

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "naive_bayes_belief", "version": 1, "theta": self.theta.tolist(), "pi": self.pi,
                "w": self.w, "smoothing": self.smoothing, "n_train": self.n_train, "trained_on": self.trained_on,
                "fit_info": self.metadata_extra}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NaiveBayesBelief":
        if d.get("kind") != "naive_bayes_belief":
            raise ValueError("not a naive_bayes_belief payload")
        theta = np.asarray(d["theta"], dtype=float)
        if theta.ndim != 2 or theta.shape[1] != 2 or np.any(theta <= 0) or np.any(theta >= 1):
            raise ValueError("theta must be (n, 2) with entries strictly inside (0, 1)")
        return cls(theta=theta, pi=float(d["pi"]), w=float(d["w"]), smoothing=float(d.get("smoothing", 1.0)),
                   n_train=int(d.get("n_train", 0)), trained_on=str(d.get("trained_on", "unknown")),
                   metadata_extra=dict(d.get("fit_info") or {}))
