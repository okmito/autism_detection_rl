"""Vectorised tie-aware AUROC and paired bootstrap comparisons.

The Step 14 comparison loop used a pure-Python rank loop, which is far too slow for
the corrected design: ~40 comparisons x 5000 resamples x 2 AUROC evaluations per
resample. ``scipy.stats.rankdata`` is C-implemented and handles ties with the same
average-rank convention, so results are identical but the run is fast enough to use
>= 5000 resamples everywhere.

Tie handling matters here specifically: the isotonic and factorised predictors
produce repeated probability values, and a naive rank implementation would
mis-measure discrimination for exactly the saturated cases this project cares about.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

__all__ = ["auroc", "paired_auroc_comparison", "holm_adjust"]


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Tie-aware AUROC. NaN when either class is absent."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0 or s.size == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def paired_auroc_comparison(y: Sequence[int], a: Sequence[float],
                            b: Sequence[float], n_resamples: int = 5000,
                            seed: int = 0,
                            alpha: float = 0.05) -> Dict[str, Any]:
    """Paired percentile bootstrap on ``AUROC(a) - AUROC(b)``.

    The SAME participant resample is applied to both predictors within a replicate,
    so the difference keeps its pairing and the interval is far tighter than
    differencing two independent intervals would give.

    Returns the point estimate, a 95% interval, and the bootstrap probability that
    ``a`` ranks above ``b``. Degenerate single-class resamples are excluded and
    counted, never imputed.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if y.size != a.size or y.size != b.size:
        raise ValueError("y, a and b must have the same length")
    if y.size < 2:
        return {"status": "UNDEFINED", "reason": "fewer than 2 observations"}

    point = auroc(y, a) - auroc(y, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y.size, size=(n_resamples, y.size))

    diffs = np.empty(n_resamples, dtype=float)
    valid = np.zeros(n_resamples, dtype=bool)
    for r in range(n_resamples):
        rows = idx[r]
        ys = y[rows]
        if np.unique(ys).size < 2:
            continue
        aa, bb = auroc(ys, a[rows]), auroc(ys, b[rows])
        if np.isfinite(aa) and np.isfinite(bb):
            diffs[r] = aa - bb
            valid[r] = True

    draws = diffs[valid]
    if draws.size < 2:
        return {"status": "UNDEFINED", "reason": "no usable resamples",
                "point_estimate": float(point),
                "n_degenerate_excluded": int((~valid).sum())}

    return {
        "status": "OK",
        "auroc_a": float(auroc(y, a)),
        "auroc_b": float(auroc(y, b)),
        "difference": float(point),
        "ci_lo": float(np.percentile(draws, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "bootstrap_p_a_greater": float((draws > 0).mean()),
        "bootstrap_p_a_less": float((draws < 0).mean()),
        "bootstrap_p_equal": float((draws == 0).mean()),
        "n_resamples": int(n_resamples),
        "n_usable_resamples": int(draws.size),
        "n_degenerate_excluded": int((~valid).sum()),
        "seed": int(seed),
        "confidence_level": 1.0 - alpha,
        "method": ("paired percentile bootstrap; identical participant resamples "
                   "for both predictors within each replicate"),
    }


def holm_adjust(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni step-down adjustment over a named family of comparisons."""
    items = [(k, v) for k, v in p_values.items() if v is not None and np.isfinite(v)]
    m = len(items)
    if m == 0:
        return {"family_size": 0, "adjusted": {}, "alpha": alpha}
    items.sort(key=lambda kv: kv[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        # Holm: multiply by (m - i), enforce monotonicity
        val = min(1.0, p * (m - i))
        running = max(running, val)
        adjusted[name] = running
    return {
        "family_size": m,
        "alpha": alpha,
        "method": "holm-bonferroni step-down",
        "adjusted_p": adjusted,
        "reject": {k: bool(v <= alpha) for k, v in adjusted.items()},
        "note": ("Applied ONLY to the prespecified confirmatory family F1. "
                 "Exploratory comparisons are not corrected and must not be "
                 "read as independent evidence."),
    }