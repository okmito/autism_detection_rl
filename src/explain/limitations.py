"""Limitation statements attached to every screening-outcome explanation.

Centralised so the honesty layer cannot drift between the demo, the batch
artifact and the tests. Every statement here is derived from what the caller
*measured* (state contents, dataset audit tags) — nothing is invented, and
nothing here is a clinical claim. Terminology follows spec §25: screening,
referral recommendation, risk estimate — never diagnosis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.env.state import UNASKED, OBSERVED, MISSING


def build_limitations(
    state: Dict[str, Any],
    predictor=None,
    *,
    stop_reason: Optional[str] = None,
    budget: Optional[int] = None,
    dataset_tag: Optional[str] = None,
    circularity: Optional[str] = None,
    label_source: Optional[str] = None,
    order_invariance_open: bool = True,
) -> List[str]:
    """Limitation statements for one screening outcome.

    Parameters mirror facts the caller already has; each is optional and each
    produces a statement only when it applies.
    """
    mask = np.asarray(state["mask"], dtype=int)
    n = int(mask.size)
    n_observed = int((mask == OBSERVED).sum())
    n_unasked = int((mask == UNASKED).sum())
    n_missing = int((mask == MISSING).sum())
    out: List[str] = []

    if dataset_tag:
        out.append(f"Dataset: {dataset_tag}.")
    if circularity:
        out.append(f"Label circularity status: {circularity}.")
    if label_source:
        out.append(f"Label source: {label_source}.")
    if n_unasked:
        out.append(
            f"{n_unasked} of {n} items were not asked. The estimate marginalises "
            f"them under the predictor's prior over unobserved items, which "
            f"assumes conditional independence across items; the marginalised "
            f"items contribute no observed evidence.")
    if n_missing:
        out.append(
            f"{n_missing} of {n} items are structurally missing (no response was "
            f"available) and carry no information in either direction.")
    if stop_reason == "policy_stop" and budget is not None and n_observed < budget:
        out.append(
            "The interview stopped early by policy before the question budget "
            "was spent; unasked items were never observed.")
    if order_invariance_open:
        out.append(
            "Adaptive question ordering assumes responses are order-invariant "
            "(verification gate V-5 remains open); retrospective answers cannot "
            "reproduce causal effects of administration order.")
    version = ""
    if predictor is not None:
        version = str(getattr(predictor, "metadata", {}).get("predictor_version",
                                                            type(predictor).__name__))
        if hasattr(predictor, "probability_with_prior"):
            out.append(
                "Partial states are scored by exact marginalisation over item "
                "configurations under a factorised training prior.")
        else:
            out.append(
                "No per-session uncertainty interval is available for this "
                "predictor version; the point estimate is reported without one.")
    if version:
        out.append(f"Predictor: {version}.")
    out.append(
        "The probability is calibrated on the development validation split of "
        "the training cohort; calibration is not guaranteed to transfer across "
        "cohorts or instruments.")
    out.append(
        "This is a screening result with a referral recommendation, not a "
        "diagnosis, and not a substitute for assessment by a qualified "
        "clinician.")
    return out


def screening_disclaimer() -> str:
    """The mandatory non-diagnostic disclaimer (spec §25 terminology lock)."""
    return ("Screening support only. This output is a research screening "
            "estimate, not a diagnosis and not a diagnostic device. It must not "
            "be used as a substitute for assessment by a qualified clinician.")
