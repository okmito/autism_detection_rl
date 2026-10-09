"""Per-session uncertainty for the screening outcome — P3-outcome.

What this module is for
-----------------------
The outcome is a **calibrated screening probability** produced by the frozen
predictor. This module adds the one uncertainty statement the architecture can
support honestly: *how much the estimate moves when the prior over unobserved
items is resampled*.

What it is NOT
--------------
* It is **not** clinical uncertainty. A narrow band says the model is stable to
  its prior; it says nothing about whether the model is right.
* It is **not** a confidence interval over the person. Sampling variability of
  a fitted prior is not uncertainty about a child's traits.
* An RL action probability, a Q-value, or the cumulative reward is **not** a
  confidence either; none of those quantities enter this module.

Method
------
For ``LogisticPredictor`` (v3), partial states are scored by exact
marginalisation over the 2^10 item configurations under a factorised item
prior taken from the Saudi training split. Resample that prior from its
Dirichlet-multinomial posterior at the training sample size,

    q_j ~ Beta(kappa * q_j_hat, kappa * (1 - q_j_hat)),   kappa = n_train,

recompute the estimate, and report the percentile interval over ``n_draws``
resamples. ``kappa = n_train`` means: "how much would the estimate vary across
re-fits of the prior on comparable samples of this size" — a
resampling-uncertainty statement about the model, not about the person.

For ``MaskedPredictor`` (v2) there is no prior mechanism (and no
dropout/ensemble to draw on without retraining), so the band is reported as
**unavailable** rather than fabricated.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _predictor_version(predictor) -> str:
    return str(getattr(predictor, "metadata", {}).get("predictor_version",
                                                      type(predictor).__name__))


def prior_sensitivity_band(
    state: Dict[str, Any],
    predictor,
    prior=None,
    n_draws: int = 200,
    level: float = 0.95,
    seed: int = 0,
    concentration: Optional[float] = None,
) -> Dict[str, Any]:
    """Percentile band of the estimate under resampled item priors.

    Parameters
    ----------
    state:
        Terminal state of the episode.
    predictor:
        Frozen predictor. Must expose ``probability_with_prior`` (v3) for a band
        to be computable; anything else returns ``available=False``.
    prior:
        Item prevalence vector P(item atypical) on training data. Defaults to
        ``predictor.prior`` when the predictor carries one.
    n_draws, level, seed:
        Number of Dirichlet-multinomial resamples, interval level, RNG seed.
    concentration:
        Dirichlet concentration kappa. Defaults to the training sample size
        recorded in the predictor's metadata (fallback 200), i.e. the
        resample-a-comparable-training-set interpretation above.

    Returns
    -------
    dict with ``available``, ``method``, ``interval`` (or ``reason``), and the
    explicit assumption labels.
    """
    base: Dict[str, Any] = {
        "method": "prior_sensitivity_band",
        "is_clinical_uncertainty": False,
        "assumptions": [
            "resampled factorised item prior at the training sample size",
            "item independence across unobserved items (v3 marginalisation)",
        ],
    }
    if not hasattr(predictor, "probability_with_prior"):
        return {**base, "available": False,
                "reason": ("predictor version has no prior mechanism and no "
                           "dropout/ensemble; no per-session interval is "
                           "computed for it"),
                "predictor_version": _predictor_version(predictor)}
    if prior is None:
        prior = getattr(predictor, "prior", None)
    if prior is None:
        return {**base, "available": False,
                "reason": "no item prevalence vector available for resampling",
                "predictor_version": _predictor_version(predictor)}

    prior = np.asarray(prior, dtype=float).ravel()
    n_items = prior.size
    kappa = concentration
    if kappa is None:
        kappa = float(getattr(predictor, "metadata", {}).get("n_train", 200))
    if kappa <= 0:
        return {**base, "available": False,
                "reason": "non-positive concentration",
                "predictor_version": _predictor_version(predictor)}

    rng = np.random.default_rng(seed)
    a = np.clip(kappa * prior, 1e-6, None)
    b = np.clip(kappa * (1.0 - prior), 1e-6, None)

    point = float(predictor.predict_state(state)) \
        if hasattr(predictor, "predict_state") else float(predictor(state))
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        q = rng.beta(a, b)
        draws[i] = float(predictor.probability_with_prior(state, q))

    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    lo, hi = np.percentile(draws, [100 * lo_q, 100 * hi_q])
    return {
        **base,
        "available": True,
        "predictor_version": _predictor_version(predictor),
        "point_estimate": point,
        "interval": [float(lo), float(hi)],
        "draw_mean": float(draws.mean()),
        "draw_sd": float(draws.std(ddof=1)) if n_draws > 1 else 0.0,
        "n_draws": int(n_draws),
        "level": float(level),
        "concentration_kappa": float(kappa),
        "interpretation": (
            "width over plausible re-fits of the item prior at the training "
            "sample size; a model-stability statement, not clinical "
            "uncertainty and not confidence in the screening result"),
    }
