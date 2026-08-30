"""Generic IRT-CAT baseline — §17 #8 (unidimensional 2PL, max Fisher info)"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

STOP = -1

class IRTCATPolicy:
    """Simple 2PL IRT CAT: fit difficulty/discrimination on training data via heuristic,
    EAP ability prior N(0,1), select item with max expected Fisher info.
    """
    def __init__(self, records: List[Dict[str, Any]], n_items: int, seed: int=0):
        self.n_items = n_items
        # Fit 2PL parameters naively: discrimination ~1, difficulty = -logit(pos_rate for item=1 group)
        X = np.stack([r["item_responses"] for r in records])
        y = np.array([r["label"] for r in records])
        self.a = np.ones(n_items)  # discrimination
        self.b = np.zeros(n_items)  # difficulty
        for j in range(n_items):
            mask = ~np.isnan(X[:, j])
            if mask.sum()==0:
                continue
            # point-biserial proxy: difficulty approx -0.5*log odds difference
            pos_rate_1 = y[X[:,j]==1].mean() if np.any(X[:,j]==1) else 0.5
            pos_rate_0 = y[X[:,j]==0].mean() if np.any(X[:,j]==0) else 0.5
            # map to difficulty via simple heuristic
            self.b[j] = float(np.clip((pos_rate_0 - pos_rate_1), -2, 2))
            # discrimination proxy: difference
            self.a[j] = float(np.clip(abs(pos_rate_1 - pos_rate_0)*2 + 0.5, 0.3, 2.5))
        # EAP prior
        self.theta_grid = np.linspace(-3, 3, 61)
        self.prior = np.exp(-0.5*self.theta_grid**2); self.prior/=self.prior.sum()

    def _p_correct(self, theta, j):
        # 2PL: P = 1/(1+exp(-a*(theta-b)))
        return 1/(1+np.exp(-self.a[j]*(theta-self.b[j])))

    def _ability_eap(self, state) -> float:
        # likelihood of observed responses given theta
        mask = state["mask"]; value = state["value"]
        log_lik = np.zeros_like(self.theta_grid)
        for j in range(self.n_items):
            if mask[j]==1:
                v = value[j]
                p = self._p_correct(self.theta_grid, j)
                p = np.clip(p, 1e-6, 1-1e-6)
                log_lik += np.where(v==1, np.log(p), np.log(1-p))
        # posterior proportional prior * exp(log_lik)
        post_unnorm = self.prior * np.exp(log_lik - log_lik.max())
        post = post_unnorm / post_unnorm.sum()
        return float(np.sum(self.theta_grid * post))

    def _fisher(self, theta, j):
        p = self._p_correct(theta, j)
        return float(self.a[j]**2 * p * (1-p))

    def __call__(self, state, legal):
        items_only = [a for a in legal if a!=STOP]
        if not items_only:
            return STOP
        theta = self._ability_eap(state)
        # max expected Fisher (here deterministic at EAP)
        best = max(items_only, key=lambda j: self._fisher(theta, j))
        return best
