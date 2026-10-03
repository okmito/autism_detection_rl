"""V-4 / V-7 gate evidence — minimum detectable effect, confirmatory family,
and the Polish denominator reconciliation.

Background (Master-Project-Specification_FINAL.md §24)
-----------------------------------------------------
* **V-4** (Checkpoint 3) — "Pre-compute MDE for each confirmatory comparison and
  finalize the confirmatory comparison family". §19.3 requires this *before the
  Polish cohort is opened*, and §19.3 closes with "the exact family and
  hypotheses are frozen at Checkpoint 3 before the external cohort is accessed."
* **V-7** (Checkpoint 3) — "Recompute Baseline 10 from public data and resolve
  the Polish denominator discrepancy before locking results".

So both gates are *preconditions* for external validation. Neither can be
satisfied by running the model on Polish; they are computed **before** Polish is
opened, which is exactly what this module does.

Honesty constraints encoded here
--------------------------------
* The MDE depends on an assumed ``sd_diff``. That assumption is **not measured
  here and cannot be** - it is a property of the unobserved effect size. This
  module therefore reports a *sensitivity grid* over plausible values rather
  than a single number, and labels every figure ``assumption_dependent``.
* Nothing in this module opens the Polish cohort. It records the cohort *size*
  and *class balance*, which are published denominator facts, not per-participant
  predictions. The class balance is used only to size the power calculation.
* Gate states are ``PASS`` / ``OPEN`` / ``BLOCKED`` / ``NOT_APPLICABLE``.
  A gate is never silently marked complete: where a human must sign, the state
  is ``OPEN`` and ``sign_off`` stays ``required``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from src.data.qchat10_contract import CANONICAL_MAPPING

#: Gate states. Kept as literals so a typo becomes a test failure.
PASS = "PASS"
OPEN = "OPEN"
BLOCKED = "BLOCKED"
NOT_APPLICABLE = "NOT_APPLICABLE"


# --------------------------------------------------------------------------
# V-4 — minimum detectable effect
# --------------------------------------------------------------------------
def mde_paired_normal(n: int, sd_diff: float, alpha: float = 0.05,
                      power: float = 0.8) -> float:
    """Two-sided MDE for a paired difference under a normal approximation.

    Mirrors ``src/eval/power.py::mde_paired`` but takes the already-adjusted
    alpha so Holm-Bonferroni can be applied across the family.
    """
    if n < 2 or sd_diff <= 0:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * sd_diff / math.sqrt(n))


def holm_bonferroni_alpha(m: int, alpha: float = 0.05) -> List[float]:
    """Per-comparison adjusted alphas for a family of ``m`` ordered tests.

    Holm's step-down procedure: the smallest p-value is tested at ``alpha/m``,
    the next at ``alpha/(m-1)``, and so on. Returned in family order, which the
    caller must pre-sort by decreasing strictness.
    """
    return [alpha / (m - i) for i in range(m)]


#: The confirmatory family, pre-declared from spec §4 (H1), §4/H2 and §19.3.
#: Recorded here as data so the freeze is auditable and testable rather than
#: living in prose.
CONFIRMATORY_FAMILY: Dict[str, Any] = {
    "hypotheses": ["H1", "H2"],
    "multiple_comparison_correction": "holm-bonferroni",
    "family_alpha": 0.05,
    "primary_endpoint": "held-out Brier loss",
    "comparisons": [
        # H1: the primary adaptive policy against the exact best fixed subset,
        # at each budget in the pre-declared confirmatory family (spec §4: 3-6).
        {"id": "H1.B3", "hypothesis": "H1", "budget": 3,
         "arm_a": "greedy", "arm_b": "exact_fixed_subset", "metric": "brier"},
        {"id": "H1.B4", "hypothesis": "H1", "budget": 4,
         "arm_a": "greedy", "arm_b": "exact_fixed_subset", "metric": "brier"},
        {"id": "H1.B5", "hypothesis": "H1", "budget": 5,
         "arm_a": "greedy", "arm_b": "exact_fixed_subset", "metric": "brier"},
        {"id": "H1.B6", "hypothesis": "H1", "budget": 6,
         "arm_a": "greedy", "arm_b": "exact_fixed_subset", "metric": "brier"},
        # H2: the adaptive-minus-fixed difference is smaller on clinician-
        # established labels than on questionnaire-derived labels.
        {"id": "H2.transfer", "hypothesis": "H2", "budget": None,
         "arm_a": "polish_minus_saudi_gap", "arm_b": "0", "metric": "brier_gap"},
    ],
    "excluded_from_family": [
        "every other budget (1, 2, 8, 10)",
        "every policy other than the primary adaptive policy",
        "exploratory subgroup cells below the pre-declared minimum size",
        "UQ/AUPRC and any threshold-dependent secondary metric",
    ],
    "rationale": (
        "Spec §19.3 restricts the confirmatory family to comparisons that "
        "directly test H1-H2 and forbids multiplying every metric, budget, "
        "tier and subgroup into one undifferentiated p-value family."
    ),
}


def compute_v4_mde(n_polish: int, sd_diff_candidates: Dict[str, float],
                   alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
    """MDE for every confirmatory comparison, at the Polish sample size.

    Computed **before** the cohort is opened. ``sd_diff_candidates`` maps an
    assumed paired-difference SD to the resulting MDE; supplying several makes
    the assumption explicit instead of hiding it behind one number.
    """
    family = CONFIRMATORY_FAMILY["comparisons"]
    m = len(family)
    adjusted = holm_bonferroni_alpha(m, alpha)
    rows: List[Dict[str, Any]] = []
    for comparison, adj in zip(family, adjusted):
        entry = {
            "id": comparison["id"],
            "hypothesis": comparison["hypothesis"],
            "budget": comparison["budget"],
            "metric": comparison["metric"],
            "family_size": m,
            "alpha_family": alpha,
            "alpha_adjusted_holm": round(adj, 6),
            "target_power": power,
            "n_evaluation": int(n_polish),
            "mde_by_assumed_sd_diff": {
                label: round(mde_paired_normal(n_polish, sd, adj, power), 6)
                for label, sd in sorted(sd_diff_candidates.items())
            },
        }
        rows.append(entry)

    return {
        "gate": "V-4",
        "gate_title": "Pre-compute MDE and freeze the confirmatory comparison family",
        "spec_reference": "Master-Project-Specification_FINAL.md §19.3, §24 (V-4)",
        "computed_before_polish_opened": True,
        "n_evaluation": int(n_polish),
        "power": power,
        "family_alpha": alpha,
        "correction": "holm-bonferroni",
        "family_size": m,
        "confirmatory_family": CONFIRMATORY_FAMILY,
        "comparisons": rows,
        "assumption_note": (
            "Every MDE figure is conditional on an ASSUMED paired-difference "
            "standard deviation. That quantity is not measured here and cannot "
            "be, because it depends on the effect being detected. The grid "
            "above is a sensitivity analysis over plausible assumptions, not a "
            "measurement. A supervisor must select the assumption before the "
            "gate can be signed."
        ),
        "automated_evidence_state": PASS,
        "human_sign_off_state": OPEN,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# V-7 — Polish denominator reconciliation
# --------------------------------------------------------------------------
def resolve_v7_denominator(observed_asd: int, observed_control: int,
                           published_asd: int, published_control: int,
                           published_total: int) -> Dict[str, Any]:
    """Reconcile the observed cohort against the published class counts.

    Spec §15 records the discrepancy: the publication reports 252 cases and 135
    diagnosed with autism, but its text contains a 135 + 118 statement summing
    to 253. This function states exactly which numbers agree, which do not, and
    what the data support - without deciding what the publication *meant*.
    """
    observed_total = observed_asd + observed_control
    published_sum = published_asd + published_control

    findings = [
        {
            "statement": "published total matches observed row count",
            "published": int(published_total),
            "observed": int(observed_total),
            "agree": bool(published_total == observed_total),
        },
        {
            "statement": "published ASD count matches observed",
            "published": int(published_asd),
            "observed": int(observed_asd),
            "agree": bool(published_asd == observed_asd),
        },
        {
            "statement": "published control count matches observed",
            "published": int(published_control),
            "observed": int(observed_control),
            "agree": bool(published_control == observed_control),
        },
        {
            "statement": "published ASD+control arithmetic is internally consistent",
            "published": int(published_sum),
            "note": f"{published_asd}+{published_control}",
            "agree": bool(published_sum == published_total),
        },
    ]

    resolved = bool(
        published_total == observed_total
        and published_asd == observed_asd
        and published_control != observed_control
        and published_sum != published_total
    )

    return {
        "gate": "V-7",
        "gate_title": "Recompute Baseline 10 and resolve the Polish denominator discrepancy",
        "spec_reference": ("Master-Project-Specification_FINAL.md §15, §24 (V-7); "
                           "DATA_VERIFICATION_REPORT.md §Polish"),
        "observed": {"asd": int(observed_asd), "control": int(observed_control),
                     "total": int(observed_total)},
        "published": {"asd": int(published_asd), "control": int(published_control),
                      "stated_total": int(published_total),
                      "arithmetic_sum": int(published_sum)},
        "findings": findings,
        "denominator_resolved": resolved,
        "resolution": (
            f"The cohort contains {observed_asd} ASD and {observed_control} control "
            f"participants = {observed_total} rows, which matches the published "
            f"total and the published ASD count. The published control count "
            f"({published_control}) is one higher than the data "
            f"({observed_control}); the publication's own class counts therefore "
            f"sum to {published_sum}, not to its stated total of {published_total}. "
            f"The denominator used throughout this project is {observed_total}."
            if resolved else
            "The observed counts do not admit a single clean reconciliation "
            "against the published figures; a supervisor must resolve this "
            "against the source publication before results are locked."
        ),
        "baseline_10_recomputation": {
            "status": OPEN,
            "reason": (
                "Spec §19.5 requires Baseline 10 to reimplement the Sollis et al. "
                "model on the same public data. Doing so faithfully requires the "
                "publication's model specification (feature set, estimator, "
                "hyperparameters, preprocessing), which is not present in this "
                "repository. Implementing it from a paraphrase would fabricate a "
                "comparator, so it is left OPEN rather than guessed."
            ),
            "required_to_close": [
                "full citation and DOI for the Sollis et al. publication",
                "the paper's stated feature set and model specification",
                "the paper's stated class counts (to settle V-7's denominator)",
            ],
        },
        "automated_denominator_state": PASS if resolved else BLOCKED,
        "automated_baseline10_state": OPEN,
        "human_sign_off_state": OPEN,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# External-validation compatibility gate
# --------------------------------------------------------------------------
#: Gate states that permit a step to proceed.
SIGNED_STATES = {"SIGNED", "CLOSED", "PASS", "APPROVED", "NOT_BLOCKED"}

#: The primary decision threshold. Frozen on the development cohort and carried
#: unchanged into external validation. It is never tuned on Polish.
PRIMARY_TAU = 0.5

#: Secondary sensitivity threshold, reported separately and never fed back.
SECONDARY_TAU = 0.3

#: The authoritative external-validation denominator, verified from the data.
PRIMARY_DENOMINATOR = 252
PRIMARY_CLASS_COUNTS = {"asd": 135, "control": 117}

#: Supervisor research decisions, recorded verbatim as given. These are recorded
#: decisions, not signatures captured on a form: the approving party is the project
#: supervisor and the reference is the decision text supplied in this review.
SUPERVISOR_DECISIONS: Dict[str, Dict[str, Any]] = {
    "qchat10_projection": {
        "decision": "APPROVE",
        "reference": "DECISION 1 (2026-10-02 review)",
        "approved_by_role": "project supervisor",
        "scope": "PRIMARY external validation",
        "mapping": {f"Q{i}": f"Q25 Q{CANONICAL_MAPPING[i]}"
                    for i in range(1, 11)},
        "note": (
            "The verified Q-CHAT-10 -> Q-CHAT-25 projection is approved as the "
            "external-validation representation. The transformation intentionally "
            "discards ordinal information; this is a recorded limitation, not an "
            "information-preserving transformation. The frozen Saudi model is not "
            "modified."),
    },
    "v4_sd_ratification": {
        "decision": "NOT_REQUIRED_FOR_PRIMARY",
        "reference": "DECISION 3 (2026-10-02 review)",
        "approved_by_role": "project supervisor",
        "note": (
            "An assumed paired-difference SD is arbitrary unless independently "
            "justified. The MDE grid is retained as sensitivity/planning analysis "
            "and REPORTED, but it no longer gates primary external validation. "
            "Precision is conveyed by confidence intervals instead."),
    },
    "v7_baseline_10": {
        "decision": "SECONDARY_UNREPRODUCED",
        "reference": "DECISION 4 (2026-10-02 review)",
        "approved_by_role": "project supervisor",
        "note": (
            "Primary external validation does not depend on reproducing the "
            "historical Baseline-10 comparator. No Sollis et al. specification is "
            "fabricated and no substitute is constructed. Any later comparative "
            "claim that needs Baseline 10 stays blocked until the exact citation "
            "and model specification are obtained."),
    },
    "primary_threshold": {
        "decision": "KEEP_TAU_0.5",
        "reference": "DECISION 5 (2026-10-02 review)",
        "approved_by_role": "project supervisor",
        "note": (
            "tau = 0.5 for the primary result. A tau = 0.3 sensitivity analysis may "
            "be reported separately and must never be used to modify the frozen "
            "model or select the primary threshold."),
    },
    "primary_research_question": {
        "decision": "RECORDED",
        "reference": "DECISION 6 (2026-10-02 review)",
        "approved_by_role": "project supervisor",
        "statement": (
            "Does a predictor trained on Saudi Q-CHAT-10 screening-derived labels "
            "transfer to an independent Polish cohort with clinician-established "
            "labels?"),
        "prohibited_uses_of_the_external_cohort": [
            "training", "calibration fitting", "threshold tuning",
            "feature selection", "hyperparameter selection", "RL policy training",
            "reward tuning",
        ],
    },
}

#: Conditions that must ALL hold before PRIMARY external validation may run.
#: Deliberately excludes an arbitrary SD ratification and the unreproduced
#: Baseline-10 comparator. Genuine data/model compatibility and leakage conditions
#: remain, and any failure of these blocks the run.
PRIMARY_GATE_CONDITIONS: Tuple[str, ...] = (
    "qchat10_projection_evidence",
    "polish_provenance",
    "leakage_audit",
    "circularity_audit",
    "frozen_predictor_artifact",
    "calibration_artifact_reproducibility",
    "polish_denominator",
)

#: Conditions that must NOT gate primary external validation, with the reason.
NON_BLOCKING_FOR_PRIMARY: Dict[str, str] = {
    "v4_sd_ratification": (
        "DECISION 3: an assumed paired-difference SD is arbitrary unless "
        "independently justified; the MDE grid is reported as sensitivity analysis "
        "and precision is conveyed by confidence intervals."),
    "v7_baseline_10": (
        "DECISION 4: primary external validation does not depend on reproducing the "
        "historical comparator; Baseline 10 remains OPEN/UNREPRODUCED and blocks only "
        "later comparative claims."),
    "baseline_10_specification": (
        "DECISION 4: no Sollis et al. specification is fabricated and no substitute "
        "comparator is constructed."),
}


def primary_gate_open(conditions: Dict[str, str]) -> Dict[str, Any]:
    """Evaluate the primary external-validation gate. All conditions are required."""
    missing = [name for name in PRIMARY_GATE_CONDITIONS
               if name not in conditions]
    failed = [name for name in PRIMARY_GATE_CONDITIONS
              if name in conditions
              and str(conditions[name]).upper() not in SIGNED_STATES]
    return {
        "required": list(PRIMARY_GATE_CONDITIONS),
        "conditions": dict(conditions),
        "not_evaluated": missing,
        "failed": failed,
        "open": not missing and not failed,
        "non_blocking": dict(NON_BLOCKING_FOR_PRIMARY),
    }


def projection_compatibility(feature_matrix_shape: Sequence[int],
                             encoded_dim: int, expected_dim: int) -> Dict[str, Any]:
    """Can the approved projection feed the frozen encoder?

    This replaces the raw-representation comparison once the supervisor has
    approved the projection. The genuine question is no longer whether the
    cohort's native 25-item ordinal form matches, but whether the *approved
    projection* produces the exact width and binary scale the frozen encoder
    consumes. Both are checked here and either failure blocks the run.
    """
    reasons: List[str] = []
    if len(feature_matrix_shape) != 2 or feature_matrix_shape[1] != 10:
        reasons.append(
            f"projected feature matrix must be (n, 10); got {tuple(feature_matrix_shape)}")
    if encoded_dim != expected_dim:
        reasons.append(
            f"encoded width must be {expected_dim}; got {encoded_dim}")
    return {
        "projection": "Q-CHAT-25 -> Q-CHAT-10 -> binary -> 41-dim frozen input",
        "feature_matrix_shape": list(feature_matrix_shape),
        "expected_feature_width": 10,
        "encoded_dim": int(encoded_dim),
        "expected_dim": int(expected_dim),
        "compatible": not reasons,
        "blocking_reasons": reasons,
        "coercion_policy": (
            "The projection is item-specific and label-derived, not a positional cast. "
            "Each code is resolved through its own SPSS value label to a printed "
            "option letter before the official Q-CHAT-10 direction is applied. "
            "Ordinal information is discarded by design."),
    }


def feature_compatibility(frozen_n_items: int, frozen_m_list,
                          cohort_n_items: int, cohort_m_list) -> Dict[str, Any]:
    """Can the frozen predictor consume this cohort's items without corruption?

    Compares the *declared* feature contract of the frozen model against what
    the cohort requires. Never proposes a coercion: a mismatch is reported, not
    patched, because silently truncating an ordinal response to fit a binary
    encoder would corrupt the data without any visible failure.
    """
    frozen_m = frozen_m_list or [2] * frozen_n_items
    frozen_dim = (4 * frozen_n_items + 1) if frozen_m_list is None \
        else (3 * frozen_n_items + int(sum(frozen_m)) + 1)
    cohort_m = cohort_m_list or [2] * cohort_n_items
    cohort_dim = (4 * cohort_n_items + 1) if cohort_m_list is None \
        else (3 * cohort_n_items + int(sum(cohort_m)) + 1)

    reasons: List[str] = []
    if frozen_n_items != cohort_n_items:
        reasons.append(
            f"item-count mismatch: frozen model has {frozen_n_items} items, "
            f"cohort provides {cohort_n_items}")
    if (frozen_m_list is None) != (cohort_m_list is None):
        reasons.append(
            "response-scale mismatch: frozen model is "
            f"{'binary' if frozen_m_list is None else 'ordinal'}, cohort is "
            f"{'binary' if cohort_m_list is None else 'ordinal'}")
    if frozen_dim != cohort_dim:
        reasons.append(
            f"encoder width mismatch: frozen expects {frozen_dim}, cohort "
            f"requires {cohort_dim}")

    return {
        "frozen_n_items": int(frozen_n_items),
        "frozen_m_list": None if frozen_m_list is None else list(frozen_m_list),
        "frozen_input_dim": int(frozen_dim),
        "cohort_n_items": int(cohort_n_items),
        "cohort_m_list": None if cohort_m_list is None else list(cohort_m_list),
        "cohort_input_dim": int(cohort_dim),
        "compatible": not reasons,
        "blocking_reasons": reasons,
        "coercion_policy": (
            "No ad-hoc coercion is offered. Casting a 4-6 level ordinal response "
            "into a binary 0/1 slot without a verified item-level rule would "
            "silently discard information and is forbidden. Two routes are "
            "legitimate and both require an explicit, reviewed decision: (a) train "
            "a predictor over the cohort's own representation, or (b) project the "
            "cohort through the verified Q-CHAT-10 -> Q-CHAT-25 item mapping in "
            "src/data/qchat10_contract.py, whose 10 binary features encode to the "
            "frozen 41-dimensional interface. This function checks the *raw* "
            "representation and therefore still reports (b) as a mismatch; the "
            "projection is authorised separately, not here."
        ),
    }