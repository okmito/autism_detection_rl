"""Hybrid IRT + Bayesian EVOI + uncertainty-aware stopping — Stage 4.

Composes three already-tested components, no new statistical machinery:
- ``BayesianLogisticPredictor`` (or any ``predict_state`` predictor) for
  posterior predictive probabilities and predictive-uncertainty,
- ``IRTCATPolicy`` 2PL for latent-trait EAP + Fisher information,
- ``BetaGreedyPolicy`` mathematics for EVOI on the Brier terminal utility.

EVOI(j) = E[utility after j] - utility now, evaluated exactly on the
empirical training support (the state space is small: <=10 binary items,
2^10 configs, support enumeration is cheap). Selection score:
    Score(j) = EVOI(j) - lambda * cost(j)
STOP is taken when the best net score <= 0 and an explicit uncertainty
gate passes; otherwise the report is ``inconclusive`` / requires_further_
assessment rather than a forced referral. All thresholds are configured
research parameters, never clinical thresholds, and the reward is the
existing proxy Brier objective (not clinical utility).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from src.policies.beta_greedy import entropy, beta_interval
from src.policies.irt_cat import IRTCATPolicy

STOP = -1


class HybridIRT_EVOIPolicy:
    def __init__(self, predictor, irt: IRTCATPolicy, records: Sequence[Dict[str, Any]],
                 n_items: int, lambda_cost: float = 0.0,
                 costs: Sequence[float] | None = None,
                 min_questions: int = 0, max_questions: int = 10,
                 uncertainty_stop_sd: float | None = None,
                 abstain_sd: float | None = None):
        if abstain_sd is None:
            abstain_sd = 0.25
        self.predictor = predictor
        self.irt = irt
        self.n_items = n_items
        self.lambda_cost = float(lambda_cost)
        self.costs = (np.ones(n_items) if costs is None else np.asarray(costs, dtype=float))
        if self.costs.shape != (n_items,):
            raise ValueError("costs must have one entry per item")
        self.min_questions = int(min_questions)
        self.max_questions = int(max_questions)
        self.uncertainty_stop_sd = uncertainty_stop_sd
        self.abstain_sd = float(abstain_sd)
        self.X = np.stack([r["item_responses"] for r in records])
        self.missing_mask = np.stack([r["missing_mask"] for r in records])
        self.y = np.array([r["label"] for r in records], dtype=float)
        self.n_train = len(records)

    # --- empirical support (same conditioning as BetaGreedyPolicy) -----
    def _support(self, mask, value) -> np.ndarray:
        idx = np.ones(self.n_train, dtype=bool)
        for j in range(self.n_items):
            m = mask[j]
            if m == 1:
                idx &= (~self.missing_mask[:, j]) & (self.X[:, j] == value[j])
            elif m == 2:
                idx &= self.missing_mask[:, j]
            if not idx.any():
                break
        return np.where(idx)[0]

    def _utility(self, idx: np.ndarray) -> tuple[float, float]:
        if len(idx) == 0:
            return 0.5, 1.0
        n_pos = int(self.y[idx].sum())
        p = (n_pos + 1) / (len(idx) + 2)
        util = float(np.mean(1.0 - (p - self.y[idx]) ** 2))
        return float(p), util

    def _evoi_for(self, idx: np.ndarray, j: int, u_now: float) -> tuple[float, float]:
        total = len(idx)
        expected = 0.0
        observed = idx[~self.missing_mask[idx, j]]
        if len(observed):
            for v in np.unique(self.X[observed, j]):
                child = observed[self.X[observed, j] == v]
                _, u_c = self._utility(child)
                expected += (len(child) / total) * u_c
        missing = idx[self.missing_mask[idx, j]]
        if len(missing):
            _, u_c = self._utility(missing)
            expected += (len(missing) / total) * u_c
        return expected - u_now, expected

    def predictive_sd(self, state) -> float:
        """Predictive uncertainty from the Bayesian predictor if available,
        else the Beta posterior sd on the empirical support. Never clinical."""
        if hasattr(self.predictor, "predictive_uncertainty"):
            try:
                u = self.predictor.predictive_uncertainty(state)
                if u.get("available"):
                    return float(u["epistemic_sd"])
            except Exception:
                pass
        idx = self._support(state["mask"], state["value"])
        if len(idx) == 0:
            return 1.0
        n_pos = int(self.y[idx].sum())
        a, b = n_pos + 1, len(idx) - n_pos + 1
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return float(math.sqrt(max(var, 0.0)))

    def scores(self, state, legal) -> Dict[str, Any]:
        from src.env.state import UNASKED
        idx = self._support(state["mask"], state["value"])
        p_now, u_now = self._utility(idx)
        theta = self.irt._ability_eap(state)
        evoi: Dict[int, float] = {}
        net: Dict[int, float] = {}
        fisher: Dict[int, float] = {}
        items_only = [a for a in legal if a != STOP]
        for a in items_only:
            gain, _ = self._evoi_for(idx, a, u_now)
            evoi[a] = float(gain)
            net[a] = float(gain - self.lambda_cost * float(self.costs[a]))
            fisher[a] = float(self.irt._fisher(theta, a))
        try:
            p_pred = float(self.predictor.predict_state(state))
        except Exception:
            p_pred = p_now
        return {
            "posterior": p_now,
            "predictor_probability": p_pred,
            "utility": u_now,
            "support_size": len(idx),
            "theta_eap": float(theta),
            "evoi": evoi,
            "net_score": net,
            "fisher_information": fisher,
            "predictive_sd": float(self.predictive_sd(state)),
            "n_asked": int((state["mask"] == 1).sum()),
        }

    def __call__(self, state, legal):
        items_only = [a for a in legal if a != STOP]
        if not items_only:
            return STOP
        s = self.scores(state, legal)
        ranked = sorted(items_only, key=lambda a: (-s["net_score"][a], a))
        best = ranked[0]
        n_asked = s["n_asked"]
        if n_asked < self.min_questions:
            return best
        # STOP is only available when the environment offers it. A
        # previously returned STOP unconditionally here, even when the
        # caller had removed STOP from the legal set to force budget use;
        # that violated the environment contract and broke ablation B.
        if STOP not in legal:
            return best
        if n_asked >= self.max_questions:
            return STOP
        best_net = s["net_score"][best]
        if best_net <= 0.0:
            sd = s["predictive_sd"]
            if sd > self.abstain_sd:
                return best  # keep asking: abstain from concluding, not from asking
            if self.uncertainty_stop_sd is not None and sd > self.uncertainty_stop_sd:
                return best
            return STOP
        return best

    def explain(self, state, legal) -> Dict[str, Any]:
        s = self.scores(state, legal)
        items_only = [a for a in legal if a != STOP]
        ranked = sorted(items_only, key=lambda a: (-s["net_score"][a], a))
        action = self(state, legal)
        out = {
            "selected": action,
            "selected_item": f"A{action + 1}" if action != STOP else "STOP",
            "theta_eap": s["theta_eap"],
            "predictor_probability": s["predictor_probability"],
            "posterior": s["posterior"],
            "support_size": s["support_size"],
            "predictive_sd": s["predictive_sd"],
            "lambda_cost": self.lambda_cost,
            "criterion_values": {f"A{a + 1}": s["net_score"][a] for a in items_only},
            "evoi_values": {f"A{a + 1}": s["evoi"][a] for a in items_only},
            "fisher_information": {f"A{a + 1}": s["fisher_information"][a] for a in items_only},
            "ranking": [f"A{a + 1}" for a in ranked],
            "min_questions": self.min_questions,
            "max_questions": self.max_questions,
            "abstain_sd": self.abstain_sd,
            "is_clinical_uncertainty": False,
            "interpretation": ("uncertainty is predictive/model uncertainty, not "
                               "clinical uncertainty; inconclusive means more "
                               "evidence needed, not a diagnosis"),
        }
        if action == STOP:
            out["stop_margin"] = max(s["net_score"].values()) if s["net_score"] else 0.0
        return out
