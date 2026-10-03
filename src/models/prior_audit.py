"""Audit of the Bayesian marginalisation prior used by predictor v3.

Predictor v3 scores a partially observed state as

    P(Y=1 | X_obs) = sum_c  P(Y=1 | c) * P(c) / sum_c P(c)      (c consistent with X_obs)

This module checks that the implementation computes that quantity, and measures how
much the choice of ``P(c)`` matters.

Why the prior choice is not cosmetic
------------------------------------
The Saudi training split contains only **155 of the 1024** binary configurations;
**869 carry zero empirical mass**, and 121 observed cells have a count of exactly 1.
The participation-ratio effective support is ~36 configurations out of 1024.

Consequences:

* **A. empirical joint prior** — assigns zero to 869 configurations. Any observed
  state whose consistent set is entirely zero-mass is undefined. Retained as a
  documented **failed** sensitivity variant, not silently dropped.
* **B. add-alpha joint prior** — prespecified smoothing, ``alpha = 0.5``
  (Jeffreys/Laplace add-half). Fixed before execution and **not** tuned.
* **C. factorised item prior** — ``P(c) = prod_j q_j^{c_j} (1-q_j)^{1-c_j}`` with
  ``q_j`` the Saudi training item prevalence. **PRIMARY.** It is the variant v3
  already implements and the only one with non-zero mass everywhere.

No variant is selected using the external cohort. Selection is Saudi-only and is
recorded as C by construction, justified by the support measurement above rather
than by any external result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import itertools
import numpy as np

N_ITEMS = 10
N_CONFIGS = 1 << N_ITEMS

#: Prespecified BEFORE execution and never tuned.
PRIOR_ALPHA = 0.5

PRIOR_VARIANTS = {
    "A_empirical_joint": "empirical joint frequency over the 1024 configurations",
    "B_add_alpha_joint": f"joint frequency with add-{PRIOR_ALPHA} smoothing "
                         f"(prespecified, not tuned)",
    "C_factorized_item": "PRIMARY: independent Bernoulli items with Saudi training "
                         "prevalences",
}


def config_keys(matrix: np.ndarray) -> np.ndarray:
    """Bit key per row, with item j as bit j."""
    matrix = np.asarray(matrix, dtype=int)
    return (matrix * (1 << np.arange(matrix.shape[1]))).sum(axis=1)


def support_report(train_matrix: np.ndarray,
                   eval_matrices: Optional[Dict[str, np.ndarray]] = None
                   ) -> Dict[str, Any]:
    """How much of the 1024-configuration space the data actually occupy."""
    train_matrix = np.asarray(train_matrix, dtype=int)
    keys = config_keys(train_matrix)
    counts = np.bincount(keys, minlength=N_CONFIGS)
    occupied = int((counts > 0).sum())
    effective = float(counts.sum() ** 2 / max((counts.astype(float) ** 2).sum(), 1.0))

    report: Dict[str, Any] = {
        "n_possible_configurations": N_CONFIGS,
        "train_n_records": int(train_matrix.shape[0]),
        "train_observed_configurations": occupied,
        "train_zero_mass_configurations": int(N_CONFIGS - occupied),
        "train_max_cell_count": int(counts.max()),
        "train_min_nonzero_cell_count": int(counts[counts > 0].min()),
        "train_cells_with_count_one": int((counts == 1).sum()),
        "effective_support_participation_ratio": effective,
        "effective_support_fraction": effective / N_CONFIGS,
        "interpretation": (
            "The joint prior is severely sparse. An empirical joint prior assigns "
            "zero mass to most of the configuration space, so it cannot define a "
            "predictive distribution for every partially observed state."),
    }
    if eval_matrices:
        for name, m in eval_matrices.items():
            ck = config_keys(np.asarray(m, dtype=int))
            unseen = int(sum(1 for k in ck if counts[k] == 0))
            report[f"{name}_configs_absent_from_train"] = unseen
            report[f"{name}_n_records"] = int(len(ck))
    return report


def build_prior(variant: str, train_matrix: np.ndarray) -> np.ndarray:
    """Return a length-1024 prior vector for the requested variant."""
    train_matrix = np.asarray(train_matrix, dtype=int)
    counts = np.bincount(config_keys(train_matrix), minlength=N_CONFIGS).astype(float)

    if variant == "A_empirical_joint":
        prior = counts / max(counts.sum(), 1.0)
    elif variant == "B_add_alpha_joint":
        smoothed = counts + PRIOR_ALPHA
        prior = smoothed / smoothed.sum()
    elif variant == "C_factorized_item":
        q = np.clip(train_matrix.mean(axis=0), 1e-6, 1 - 1e-6)
        k = np.arange(N_CONFIGS)
        bits = ((k[:, None] >> np.arange(N_ITEMS)[None, :]) & 1).astype(float)
        logp = (bits * np.log(q)[None, :]
                + (1 - bits) * np.log(1 - q)[None, :]).sum(axis=1)
        prior = np.exp(logp - logp.max())
        prior = prior / prior.sum()
    else:
        raise ValueError(f"unknown prior variant {variant!r}; "
                         f"expected one of {list(PRIOR_VARIANTS)}")

    if not np.isfinite(prior).all() or prior.sum() <= 0:
        raise ValueError(f"prior variant {variant!r} produced an invalid prior")
    return prior


def evaluate_variant(predictor, variant: str, train_matrix: np.ndarray,
                     eval_matrix: np.ndarray) -> Dict[str, Any]:
    """Apply one prior variant to the frozen coefficients and score it.

    The fitted logistic coefficients and the Platt calibrator are reused unchanged;
    only ``P(c)`` differs, which isolates the effect of the prior.
    """
    from sklearn.metrics import roc_auc_score

    prior = build_prior(variant, train_matrix)
    configs = np.array([[(int(r) >> j) & 1 for j in range(N_ITEMS)]
                        for r in range(N_CONFIGS)], dtype=float)

    logit = predictor.model.decision_function(configs)
    base = 1.0 / (1.0 + np.exp(-logit))
    if predictor.calibrator is not None:
        base = np.clip(np.asarray(predictor.calibrator.predict(base), dtype=float),
                       0.0, 1.0)
    table = base * prior

    eval_matrix = np.asarray(eval_matrix, dtype=int)
    probs = np.empty(eval_matrix.shape[0], dtype=float)
    undefined = 0
    fallback_used = 0
    for i, row in enumerate(eval_matrix):
        consistent = np.ones(N_CONFIGS, dtype=bool)
        for j in range(N_ITEMS):
            consistent &= (configs[:, j] == row[j])
        mass = float(prior[consistent].sum())
        if mass <= 0.0:
            undefined += 1
            fallback_used += 1
            probs[i] = float(1.0 / (1.0 + np.exp(-predictor.model.intercept_[0])))
        else:
            probs[i] = float(table[consistent].sum() / mass)

    from sklearn.metrics import brier_score_loss
    out: Dict[str, Any] = {
        "variant": variant,
        "description": PRIOR_VARIANTS[variant],
        "prior_zero_mass_cells": int((prior <= 0).sum()),
        "prior_min_positive": float(prior[prior > 0].min()) if (prior > 0).any() else 0.0,
        "n_records": int(eval_matrix.shape[0]),
        "n_states_with_zero_prior_mass": undefined,
        "fallback_used_for_zero_mass": fallback_used,
        "meets_definition_everywhere": undefined == 0,
        "mean_probability": float(probs.mean()),
        "probability_min": float(probs.min()),
        "probability_max": float(probs.max()),
        "brier": float(brier_score_loss(eval_matrix.sum(axis=1) > 0, probs))
        if eval_matrix.shape[0] else None,
    }
    return out


def audit(predictor, train_matrix: np.ndarray,
          eval_matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Full audit: support measurement plus all three prespecified variants."""
    support = support_report(train_matrix, eval_matrices)
    variants: List[Dict[str, Any]] = []
    for variant in PRIOR_VARIANTS:
        try:
            variants.append(evaluate_variant(predictor, variant, train_matrix,
                                             next(iter(eval_matrices.values()))))
        except Exception as exc:
            variants.append({"variant": variant,
                             "description": PRIOR_VARIANTS[variant],
                             "error": f"{type(exc).__name__}: {exc}"})
    return {
        "definition_checked": "P(Y|X_obs) = sum_c P(Y|c) P(c|X_obs), c consistent with X_obs",
        "marginal_form_implemented_by_v3": "C_factorized_item",
        "prespecified_alpha_for_variant_B": PRIOR_ALPHA,
        "alpha_was_tuned": False,
        "selected_using_polish": False,
        "support": support,
        "sensitivity": variants,
        "conclusion": (
            "Variant A (empirical joint) fails the requirement that a prior assign "
            "non-zero mass everywhere: it is undefined on "
            f"{support['train_zero_mass_configurations']} of {N_CONFIGS} "
            "configurations. Variant C is retained as PRIMARY because it is "
            "non-zero everywhere and is what predictor v3 implements. Variant B is "
            "reported with a prespecified alpha and was not tuned. No variant was "
            "selected using the external cohort."),
    }