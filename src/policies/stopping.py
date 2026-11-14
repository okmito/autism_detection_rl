"""Confidence-aware stopping & abstention — Stage 4/5.

A pure, testable decision function: given the predictive uncertainty,
EVOI net scores, question budget consumption and configurable
*research* thresholds, decide between STOP, CONTINUE (ask the best
item) and ABSTAIN (evidence insufficient — report inconclusive, ask
best item / escalate rather than forcing a category).

Nothing here is a clinical threshold. Until independently validated
thresholds exist, the values are experimental research parameters.
"""
from __future__ import annotations

from typing import Any, Dict, List


def decide_stop(*, predictive_sd: float, best_net_score: float,
                n_asked: int, min_questions: int, max_questions: int,
                abstain_sd: float, uncertainty_stop_sd: float | None = None) -> str:
    """Return 'STOP' | 'CONTINUE' | 'ABSTAIN'.

    - budget exhausted -> STOP
    - below min_questions -> CONTINUE
    - best question clearly worthwhile -> CONTINUE
    - best question not worthwhile and uncertainty is high -> ABSTAIN
      (do not force a confident category mid-interview)
    - best question not worthwhile and uncertainty is low -> STOP
    """
    if n_asked >= max_questions:
        return "STOP"
    if n_asked < min_questions:
        return "CONTINUE"
    if best_net_score > 0.0:
        return "CONTINUE"
    # No worthwhile question left.
    if predictive_sd > abstain_sd:
        return "ABSTAIN"
    if uncertainty_stop_sd is not None and predictive_sd > uncertainty_stop_sd:
        return "ABSTAIN"
    return "STOP"


def stopping_summary(*, decision: str, predictive_sd: float, best_net_score: float,
                     n_asked: int, budget: int) -> Dict[str, Any]:
    """Explainable stop summary: why the policy stopped/abstained."""
    return {
        "decision": decision,
        "predictive_sd": float(predictive_sd),
        "best_net_score": float(best_net_score),
        "n_questions_asked": int(n_asked),
        "budget": int(budget),
        "is_clinical_threshold": False,
        "interpretation": (
            "STOP means no remaining question repays its research cost; "
            "ABSTAIN means evidence is insufficient — the screening output "
            "should be reported as inconclusive / requiring further "
            "assessment rather than a forced referral category."
        ),
    }
