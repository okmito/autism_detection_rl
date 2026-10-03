"""Permanent reference baselines for every predictive claim in this project.

Established after the primary Polish external validation (2026-10-02), where the
frozen predictor's AUROC (0.896) was significantly **below** a trivial item-count
reference (0.937; paired 95% CI on the difference [-0.064, -0.019]).

Why this module exists
----------------------
A claim of improvement is meaningless without a floor. Any future predictor or
adaptive policy must beat these references before "improvement" may be claimed.
They are deliberately trivial, fully transparent, and label-free in construction:

``item_count``
    The number of atypical answers among the ten projected Q-CHAT-10 items. This
    is the strongest simple reference found: on Polish it reached AUROC 0.937 with
    a calibration slope of 1.25, i.e. close to a linear probability model.
``random_questioning``
    The lower bound for adaptive selection: ask items in random order.
``greedy_information_gain``
    The standard adaptive heuristic: ask the item the frozen predictor is most
    uncertain about.

None of these may be tuned, calibrated or selected using the external cohort. They
are fixed definitions.

MANDATORY CAVEAT
----------------
Research prototype. Not a diagnostic device. Nothing here establishes that a
screening prediction is a clinical diagnosis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

#: Canonical baseline identifiers for the CORRECTED fixed-budget design.
#: All four are MANDATORY. An evaluation that omits any of them is incomplete and
#: must fail rather than quietly report a "win".
#:
#: ``random_fixed`` replaces ``random_questioning`` here. ``RandomPolicy`` is kept
#: untouched for a separate learned/variable-length STOPPING experiment and must
#: not be mixed into the fixed-budget comparison, because it stops early and is
#: therefore not budget-matched.
REQUIRED_BASELINES = ("item_count", "random_fixed",
                      "greedy_information_gain", "beta_greedy_evoi")

#: Reserved for the future stopping experiment; NOT part of the fixed-budget set.
STOPPING_EXPERIMENT_ONLY_BASELINES = ("random_questioning",)

#: The frozen primary result this module was introduced to guard against.
MOTIVATION = (
    "On the Polish external cohort the frozen predictor scored AUROC 0.8959 while a "
    "trivial item count scored 0.9369; the paired bootstrap 95% interval on the "
    "difference was [-0.0636, -0.0186], entirely below zero. Any later claim of "
    "improvement must be measured against this floor.")


def item_count_scores(features: np.ndarray) -> np.ndarray:
    """Raw count of atypical answers across the ten projected items."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D (n, 10) feature matrix, got "
                         f"{features.shape}")
    if features.shape[1] != 10:
        raise ValueError(
            f"item count is defined for the 10 projected Q-CHAT-10 items; got "
            f"{features.shape[1]} columns")
    if not np.isin(features, (0.0, 1.0)).all():
        raise ValueError("item count expects binary 0/1 features")
    return features.sum(axis=1).astype(float)


def item_count_probabilities(features: np.ndarray) -> np.ndarray:
    """Count scaled to [0, 1] so it can be compared with a probability.

    Monotone in :func:`item_count_scores`, so discrimination is identical; only the
    probability scale differs. Use the raw count for ranking comparisons.
    """
    return item_count_scores(features) / 10.0


def missing_baselines(reported: Sequence[str]) -> List[str]:
    """Which mandatory baselines a result set failed to report."""
    return [name for name in REQUIRED_BASELINES if name not in set(reported)]


class MissingBaselineError(RuntimeError):
    """Raised when a result set omits a mandatory reference."""


def assert_all_baselines(reported: Sequence[str], context: str = "") -> None:
    """Fail loudly rather than let an incomplete benchmark look like a win."""
    absent = missing_baselines(reported)
    if absent:
        where = f" ({context})" if context else ""
        raise MissingBaselineError(
            f"mandatory baseline(s) absent{where}: {absent}. Every future policy "
            f"experiment must report {list(REQUIRED_BASELINES)} before any "
            f"improvement claim is admissible.")


def baseline_comparison_table(features: np.ndarray, y: np.ndarray,
                              scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """AUROC of each supplied predictor against the item-count floor."""
    def auroc(yy: np.ndarray, ss: np.ndarray) -> float:
        yy = np.asarray(yy, dtype=int)
        ss = np.asarray(ss, dtype=float)
        n1 = int((yy == 1).sum())
        n0 = int((yy == 0).sum())
        if n1 == 0 or n0 == 0:
            return float("nan")
        order = np.argsort(ss, kind="mergesort")
        ranks = np.empty(ss.size, dtype=float)
        sorted_s = ss[order]
        i = 0
        while i < ss.size:
            j = i
            while j + 1 < ss.size and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    floor = auroc(y, item_count_scores(features))
    rows = []
    for name, s in scores.items():
        value = auroc(y, np.asarray(s, dtype=float))
        rows.append({
            "name": name,
            "auroc": value,
            "auroc_minus_item_count": (None if not np.isfinite(value)
                                       else float(value - floor)),
            "beats_item_count": (None if not np.isfinite(value)
                                else bool(value > floor)),
        })
    return {
        "item_count_auroc": floor,
        "motivation": MOTIVATION,
        "required_baselines": list(REQUIRED_BASELINES),
        "rows": rows,
        "no_predictor_beats_item_count": not any(
            r["beats_item_count"] for r in rows if r["beats_item_count"] is not None),
    }