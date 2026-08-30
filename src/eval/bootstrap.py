"""Paired bootstrap — §19.2"""
from __future__ import annotations
import numpy as np
from typing import Callable

def paired_bootstrap_ci(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                        metric_fn: Callable, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired bootstrap on matched evaluation instances.
    metric_fn(y_true, y_prob) -> float; we compute diff = metric_a - metric_b
    Returns dict with diff, ci_low, ci_high, se.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ma = metric_fn(y_true[idx], scores_a[idx])
        mb = metric_fn(y_true[idx], scores_b[idx])
        diffs.append(ma - mb)
    diffs = np.array(diffs)
    lo = float(np.percentile(diffs, 100*alpha/2))
    hi = float(np.percentile(diffs, 100*(1-alpha/2)))
    # observed diff
    obs = float(metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b))
    return {"observed_diff": obs, "ci_low": lo, "ci_high": hi, "se": float(diffs.std(ddof=1)), "n_resamples": n_resamples}
