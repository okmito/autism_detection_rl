"""Generic IRT-CAT baseline — §17 #8 (unidimensional 2PL, max Fisher info).

Item direction
--------------
In this project's binary encoding, ``1`` means an **atypical / concerning**
response, not a correct answer. A 2PL model assumes ``1`` is correct, so the
latent trait recovered here is **symptom propensity**, not ability. That is
internally consistent — item selection only needs a monotone latent — and the
class is named IRT-CAT because it is the generic IRT-CAT *form*, per spec §17
#8, which explicitly does not claim to reproduce CAT-Autism. The direction is
stated here so the trait is not misread as ability.

Two defects in the original parameter estimation, both active on this instrument
(measured on the 284-record Saudi train split):

1. **Difficulty was mis-centred.** ``b[j] = clip(P(y=1|x_j=0) - P(y=1|x_j=1), -2, 2)``
   is *always* negative for any discriminative item here, because endorsing item
   ``j`` raises the total and therefore the probability of the positive label.
   Measured: ``b <= 0`` for **10/10** items. With every ``b`` below the origin and
   ``theta >= 0``, ``p_correct`` sits at 0.92-0.97 for every item, so
   ``a^2 p (1-p)`` is tiny and nearly flat — item selection degenerated to the
   ``a^2`` discrimination proxy and barely responded to the answers given.

   Fixed by estimating a latent trait first (the standardised endorsed count),
   then fitting a 1-D logistic model of each item on that trait and converting
   the result to 2PL parameters. ``b`` is then centred by construction.

2. **An all-missing column looked maximally informative.** ``continue`` left
   ``a=1.0, b=0.0``, giving ``I(0) = 0.25`` with no data behind it. This is
   *latent* rather than active here — the Saudi cohort has zero missing cells,
   verified 0/5060 — but it would bite on any cohort with a wholly unobserved
   item. Uninformative items now get ``a = 0``, which makes their Fisher
   information exactly 0.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

STOP = -1

#: Below this |slope| an item carries no usable signal and is scored as
#: uninformative (a = 0). Guards against perfect separation, where an item that
#: perfectly predicts the latent drives the logistic slope to infinity.
MIN_ABS_SLOPE = 1e-3


class IRTCATPolicy:
    """Unidimensional 2PL CAT on the project's binary items.

    Selects the unasked item with the highest Fisher information at the EAP
    estimate of the current latent trait, and never stops early — spec §17 #8
    requires the generic CAT baseline to spend the matched question budget, so
    that adaptive *stopping* is isolated in AB-7 rather than confounded with
    item selection.
    """

    def __init__(self, records: List[Dict[str, Any]], n_items: int, seed: int = 0):
        self.n_items = n_items
        X = np.stack([r["item_responses"] for r in records]).astype(float)
        y = np.array([r["label"] for r in records])

        # Latent trait: standardised count of endorsed (atypical) responses.
        # Used only to *fit* the item parameters, never at inference.
        total = np.nansum(X, axis=1)
        centred = total - np.nanmean(total)
        spread = float(np.nanstd(total))
        latent = centred / spread if spread > 1e-9 else np.zeros_like(centred)

        self.a = np.zeros(n_items)   # discrimination
        self.b = np.zeros(n_items)   # difficulty
        self.n_informative = 0
        for j in range(n_items):
            mask = ~np.isnan(X[:, j])
            if mask.sum() < 2:
                continue  # wholly unobserved, or a single observation
            responses = X[mask, j]
            if len(np.unique(responses)) < 2:
                continue  # constant item: no discrimination, no difficulty
            slope, intercept = _logistic_1d(latent[mask], responses)
            if abs(slope) < MIN_ABS_SLOPE:
                continue
            # P(x=1) = logistic(intercept + slope*z)  ==  2PL(theta) with
            # a = |slope| and b = -intercept/|slope|.
            self.a[j] = abs(slope)
            self.b[j] = -intercept / abs(slope)
            self.n_informative += 1

        self.theta_grid = np.linspace(-3, 3, 61)
        self.prior = np.exp(-0.5 * self.theta_grid ** 2)
        self.prior /= self.prior.sum()

    def _p_endorsed(self, theta, j):
        """P(item j endorsed | theta) under the 2PL form."""
        return 1.0 / (1.0 + np.exp(-self.a[j] * (theta - self.b[j])))

    def _ability_eap(self, state) -> float:
        """EAP estimate of the latent trait from the observed responses."""
        mask = state["mask"]
        value = state["value"]
        log_lik = np.zeros_like(self.theta_grid)
        for j in range(self.n_items):
            if mask[j] == 1 and self.a[j] > 0:
                p = np.clip(self._p_endorsed(self.theta_grid, j), 1e-6, 1 - 1e-6)
                log_lik += np.where(value[j] == 1, np.log(p), np.log(1 - p))
        shifted = log_lik - log_lik.max()
        post = self.prior * np.exp(shifted)
        total = post.sum()
        if not np.isfinite(total) or total <= 0:
            return 0.0
        post /= total
        return float(np.sum(self.theta_grid * post))

    def _fisher(self, theta, j) -> float:
        """Fisher information for item j at theta. Zero for uninformative items."""
        if self.a[j] <= 0:
            return 0.0
        p = self._p_endorsed(theta, j)
        return float(self.a[j] ** 2 * p * (1 - p))

    def information(self, state, legal) -> Dict[int, float]:
        """Fisher information per legal item at the current EAP estimate.

        Exposed so the benchmark and the explainability layer can report *why* an
        item was chosen rather than only which one was chosen.
        """
        theta = self._ability_eap(state)
        return {a: self._fisher(theta, a) for a in legal if a != STOP}

    def __call__(self, state, legal):
        items_only = [a for a in legal if a != STOP]
        if not items_only:
            return STOP
        theta = self._ability_eap(state)
        # Deterministic tie-break on the lowest index, so the policy is
        # reproducible and cannot be order-dependent.
        best = max(items_only, key=lambda j: (self._fisher(theta, j), -j))
        return best


def _logistic_1d(x: np.ndarray, y: np.ndarray, ridge: float = 1.0):
    """1-D ridge-penalised logistic fit of ``y`` on ``x``.

    Returns ``(slope, intercept)``. Hand-rolled (five Newton steps) so the module
    has no scikit-learn dependency and so the regularisation strength is explicit
    rather than hidden in a solver's defaults.
    """
    beta = np.zeros(2)  # [intercept, slope]
    for _ in range(25):
        z = beta[0] + beta[1] * x
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = np.array([
            np.sum(p - y) + ridge * beta[0],
            np.sum((p - y) * x) + ridge * beta[1],
        ])
        hess = np.array([
            [np.sum(w) + ridge, np.sum(w * x)],
            [np.sum(w * x), np.sum(w * x * x) + ridge],
        ])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[1]), float(beta[0])
