"""Power / MDE — §19.3 stub"""
from __future__ import annotations
import numpy as np
from scipy import stats

def mde_paired(n: int, alpha: float = 0.05, power: float = 0.8, sd_diff: float = 1.0) -> float:
    """Minimum detectable effect for paired difference (normal approx)."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * sd_diff / np.sqrt(n))

def mde_report(n: int, sd_diffs: dict, alpha=0.05, power=0.8) -> dict:
    return {k: mde_paired(n, alpha, power, sd) for k, sd in sd_diffs.items()}
