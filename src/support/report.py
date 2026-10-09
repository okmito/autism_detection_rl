"""Assembly of the validated support report — P4 (screening + support).

This module is the single place where the three layers meet:

* the **screening outcome** produced by ``src.env.environment.run_episode``
  (the calibrated probability and the thresholded referral decision),
* its **explanation** produced by ``src.explain.outcome.explain_outcome``
  (schema ``outcome-explanation/1.0``), and
* the **support layer**'s own registries and rules (``src.support.*``).

What this module guarantees
---------------------------
* **Validation at every boundary.** Everything assembled here is a Pydantic
  model from ``src.support.schemas``: unknown fields are rejected, cross
  references must resolve inside the report, and a hypothesis can never be
  presented as a confirmed need. Adapters therefore *convert and verify*, they
  never invent.
* **No fabricated fields.** If the explanation object does not carry an
  uncertainty band (v2 MLP), the assembled report says so — the field is
  ``available=False`` with the reason preserved, not filled in.
* **Determinism.** Given the same episode result, the same explanation dict and
  the same follow-up answers, :func:`build_support_report` returns an identical
  report. The only varying field is ``created_at``, supplied by the schema as a
  timestamp for audit purposes.
* **The follow-up questionnaire is optional and separate.** Its answers never
  enter the RL state, the predictor input, or the screening outcome; they are
  consumed here and nowhere else.

Generation status rule (documented so it cannot drift)
------------------------------------------------------
``ExplanationResult.generation_status`` is derived, not chosen:

* ``PARTIAL`` — the explanation was produced but something the architecture
  could not supply is missing: no attributed items, no uncertainty band (every
  v2 explanation), or the explanation carries warnings (e.g. a decision/τ
  mismatch).
* ``COMPLETE`` — attributions, supporting/opposing evidence and an uncertainty
  band are all present and no warnings were raised.
* ``UNAVAILABLE`` — never emitted here; a missing explanation is represented by
  ``ScreeningReport.explanation = None``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.env.state import OBSERVED
from src.explain.limitations import screening_disclaimer
from src.support import engine
from src.support.schemas import (
    AssessmentStatus, EvidenceItem, ExplanationResult, FeatureContribution,
    FollowUpAnswers, GenerationStatus, ScreeningReport, ScreeningResult,
    SourceType, SupportNeedAssessment, SupportRecommendation, UncertaintyInfo)
from src.support.strategies import REVIEW_PENDING

#: The Shapley baseline every attribution in this project is measured against:
#: the predictor's prior-only estimate with all attributed items UNASKED.
BASELINE_REFERENCE = "prior_only_unasked_all_items"

#: The frozen operating threshold (spec §10); ``run_episode`` does not echo it,
#: so the screening result records 0.5 unless the caller passes what the
#: episode was actually run at.
TAU_DEFAULT = 0.5

#: Default model identity when a caller cannot supply the predictor's manifest.
DEFAULT_MODEL_NAME = "MaskedPredictor"
DEFAULT_MODEL_VERSION = "unknown"


# ---------------------------------------------------------------------------
# screening result
# ---------------------------------------------------------------------------

def _items_asked_codes(episode_result: Dict[str, Any]) -> List[str]:
    """Item codes (``A1``..) for the questions the episode actually asked."""
    items = episode_result.get("items_asked")
    if items:
        return [_item_code(j) for j in items]
    return [str(t["item"]) for t in (episode_result.get("trace") or [])
            if "item" in t]


def _item_code(j: Any) -> str:
    """Accept an item index (0-based int, as ``run_episode`` records) or an
    already-formatted code (``"A3"``, as stored explanations keep them)."""
    if isinstance(j, str):
        return j
    return f"A{int(j) + 1}"


def screening_result_from_episode(
    episode_result: Dict[str, Any],
    *,
    result_id: str = "sr-1",
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    input_reference: str = "",
    limitations: Sequence[str] = (),
    tau: Optional[float] = None,
) -> ScreeningResult:
    """Typed screening result from a ``run_episode`` output dict.

    ``run_episode`` does not record ``tau`` (it is a call argument), so the
    threshold defaults to the frozen value 0.5; pass ``tau`` explicitly to
    record the value the episode was actually run at.
    """
    for key in ("decision", "p_hat"):
        if key not in episode_result:
            raise ValueError(f"episode_result has no {key!r}; run_episode output "
                             f"is required to build a screening result")
    state = episode_result.get("final_state") or {}
    budget = state.get("budget")
    items = _items_asked_codes(episode_result)
    return ScreeningResult(
        result_id=result_id,
        decision=episode_result["decision"],       # both vocabularies normalise
        p_hat=float(episode_result["p_hat"]),
        tau=float(TAU_DEFAULT if tau is None else tau),
        stop_reason=episode_result.get("stop_reason"),
        items_asked=items,
        n_questions_asked=len(items),
        budget=(int(budget) if budget is not None else None),
        model_name=str(model_name),
        model_version=str(model_version),
        input_reference=input_reference or "episode",
        limitations=list(limitations),
        disclaimer="",
    )


def screening_result_from_explanation(
    explanation: Dict[str, Any],
    *,
    result_id: str = "sr-1",
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    input_reference: str = "",
    limitations: Sequence[str] = (),
) -> ScreeningResult:
    """Typed screening result from a stored ``outcome-explanation/1.0`` dict.

    Used when only the explanation artifact is available (batch replay); the
    explanation carries the outcome block it was produced from.
    """
    outcome = explanation.get("outcome")
    if not isinstance(outcome, dict):
        raise ValueError("explanation has no 'outcome' block; cannot rebuild the "
                         "screening result from it")
    for key in ("decision", "p_hat"):
        if key not in outcome:
            raise ValueError(f"explanation outcome block has no {key!r}")
    items = [_item_code(c) for c in (outcome.get("items_asked") or [])]
    model = explanation.get("model") or {}
    n_asked = outcome.get("n_questions_asked")
    return ScreeningResult(
        result_id=result_id,
        decision=outcome["decision"],
        p_hat=float(outcome["p_hat"]),
        tau=float(outcome.get("tau", TAU_DEFAULT)),
        stop_reason=outcome.get("stop_reason"),
        items_asked=items,
        n_questions_asked=int(n_asked if n_asked is not None else len(items)),
        budget=(int(outcome["budget"]) if outcome.get("budget") is not None
                else None),
        model_name=str(model_name),
        model_version=str(model_version or model.get("predictor_version",
                                                     DEFAULT_MODEL_VERSION)),
        input_reference=input_reference or "outcome-explanation",
        limitations=list(limitations),
        disclaimer="",
    )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def evidence_from_episode(episode_result: Dict[str, Any]) -> List[EvidenceItem]:
    """Observed-response evidence from a ``run_episode`` output.

    Uses the episode trace when present (each step names the item and the
    recorded response) and falls back to the terminal state's OBSERVED mask.
    Ids are deterministic (``ev-obs-A3``); a repeated item code keeps its first
    step so ids stay unique.
    """
    out: List[EvidenceItem] = []
    seen: set[str] = set()

    def _add(item: str, value: Any, reference: str) -> None:
        if item in seen:
            return
        seen.add(item)
        out.append(EvidenceItem(
            evidence_id=f"ev-obs-{item}",
            source_type=SourceType.OBSERVED_RESPONSE,
            item_code=item,
            observed_value=int(value),
            source_reference=reference,
            description=f"observed response to {item} recorded during the "
                        f"screening interview",
        ))

    trace = episode_result.get("trace") or []
    if trace:
        for i, t in enumerate(trace):
            if "item" not in t or t.get("value") is None:
                continue
            _add(str(t["item"]), t["value"], f"episode:trace[{i}]")
        return out

    state = episode_result.get("final_state") or {}
    mask = np.asarray(state.get("mask", []), dtype=int)
    value = np.asarray(state.get("value", []), dtype=int)
    for j in range(mask.size):
        if mask[j] == OBSERVED:
            _add(f"A{j + 1}", int(value[j]), "episode:final_state")
    return out


def _observed_evidence_from_explanation(
    explanation: Dict[str, Any]) -> List[EvidenceItem]:
    """Same evidence, recovered from a stored explanation dict.

    Only used when no episode result is available; the trace-position reference
    is not recoverable from the artifact, so the source reference names the
    block it came from.
    """
    observed = ((explanation.get("evidence") or {})
                .get("observed_responses") or [])
    out: List[EvidenceItem] = []
    for o in observed:
        item = o.get("item")
        if item is None or o.get("response") is None:
            continue
        out.append(EvidenceItem(
            evidence_id=f"ev-obs-{item}",
            source_type=SourceType.OBSERVED_RESPONSE,
            item_code=str(item),
            observed_value=int(o["response"]),
            source_reference="outcome-explanation:observed_responses",
            description=f"observed response to {item} recorded during the "
                        f"screening interview",
        ))
    return out


# ---------------------------------------------------------------------------
# explanation adapter
# ---------------------------------------------------------------------------

def _uncertainty_from_dict(block: Optional[Dict[str, Any]],
                           ) -> Tuple[UncertaintyInfo, bool]:
    """Convert the explanation's uncertainty block.

    Returns ``(info, band_available)``. An absent or unavailable band is
    reported as unavailable with the reason preserved — never fabricated.
    """
    if not isinstance(block, dict):
        return UncertaintyInfo(
            available=False,
            interpretation="the explanation did not carry an uncertainty "
                           "block"), False
    available = bool(block.get("available"))
    interval = block.get("interval") or [None, None]
    info = UncertaintyInfo(
        available=available,
        method=block.get("method"),
        interval_low=(float(interval[0]) if available else None),
        interval_high=(float(interval[1]) if available else None),
        interpretation=str(block.get("interpretation") or block.get("reason")
                           or ""),
    )
    return info, available


def explanation_from_dict(
    explanation: Dict[str, Any],
    *,
    result_id: str,
    observed_evidence: Optional[Sequence[EvidenceItem]] = None,
) -> ExplanationResult:
    """Typed explanation from an ``outcome-explanation/1.0`` dict.

    ``supporting``/``opposing`` evidence is resolved against the explanation's
    own attribution signs; contributions carry the same direction and the
    Shapley baseline reference. Generation status follows the documented rule
    in the module docstring.
    """
    if not isinstance(explanation, dict):
        raise TypeError("explanation must be the dict produced by "
                        "src.explain.outcome.explain_outcome")
    evidence_block = explanation.get("evidence") or {}
    if observed_evidence is None:
        observed_evidence = _observed_evidence_from_explanation(explanation)
    supporting_codes = {str(c) for c in (evidence_block.get("supporting") or [])}
    opposing_codes = {str(c) for c in (evidence_block.get("opposing") or [])}
    supporting = [e for e in observed_evidence if e.item_code in supporting_codes]
    opposing = [e for e in observed_evidence if e.item_code in opposing_codes]

    method = str((explanation.get("attribution") or {}).get("method",
                                                            "exact_group_shapley"))
    contributions = [
        FeatureContribution(
            item_code=str(c["item"]),
            contribution_value=float(c["phi"]),
            direction=str(c["direction"]),
            explanation_method=method,
            baseline_reference=BASELINE_REFERENCE,
        )
        for c in (explanation.get("contributions") or [])
    ]

    uncertainty, band_available = _uncertainty_from_dict(
        explanation.get("uncertainty"))

    if not contributions:
        status = GenerationStatus.PARTIAL
    elif not band_available:
        status = GenerationStatus.PARTIAL
    elif explanation.get("warnings"):
        status = GenerationStatus.PARTIAL
    else:
        status = GenerationStatus.COMPLETE

    return ExplanationResult(
        screening_result_id=result_id,
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        feature_contributions=contributions,
        explanation_method=method,
        limitations=[str(x) for x in (explanation.get("limitations") or [])],
        uncertainty=uncertainty,
        generation_status=status,
    )


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def _report_limitations(
    assessments: Sequence[SupportNeedAssessment],
    recommendations: Sequence[SupportRecommendation],
) -> List[str]:
    """Limitations of the *support* layer (the screening outcome's own
    limitations stay on the screening result and the explanation)."""
    out: List[str] = []
    n_hyp = sum(1 for a in assessments
                if a.status is AssessmentStatus.HYPOTHESIS_FROM_OBSERVED)
    if n_hyp:
        out.append(
            f"{n_hyp} area(s) are marked as hypotheses suggested by observed "
            f"responses. A hypothesis is not a finding of need: it only marks "
            f"a follow-up question that can be skipped, and it produced no "
            f"suggestions.")
    if any(r.review_status == REVIEW_PENDING for r in recommendations):
        out.append(
            "Every support suggestion here is curated project content that a "
            "human expert has not yet reviewed. None has been clinically "
            "evaluated, and none is offered as an intervention or a "
            "prescription.")
    out.append(
        "Support suggestions are optional and were selected by transparent "
        "rules from your own follow-up answers and stated preferences. No "
        "model personalised them to you, and nothing here is a diagnosis.")
    return out


def build_support_report(
    *,
    episode_result: Optional[Dict[str, Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
    followup_answers: Union[FollowUpAnswers, Sequence[Dict[str, Any]], None]
    = None,
    result_id: str = "sr-1",
    report_id: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    input_reference: str = "",
    tau: Optional[float] = None,
) -> ScreeningReport:
    """Assemble the validated support report.

    Parameters
    ----------
    episode_result:
        ``run_episode`` output (preferred: gives the exact trace positions for
        the observed-response evidence and the budget the episode ran at).
    explanation:
        ``explain_outcome`` output (schema ``outcome-explanation/1.0``). One of
        the two must be given; with both, the episode result is authoritative
        for the screening result and the explanation supplies the attribution.
    followup_answers:
        The optional follow-up questionnaire: a ``FollowUpAnswers`` model or a
        list of answer dicts. ``None`` (or an empty list) means the
        questionnaire was skipped, which is a valid outcome — every domain
        then reports an honest abstention or hypothesis.
    result_id, report_id:
        Correlation ids; ``report_id`` defaults to ``rep-<result_id>``.
    model_name, model_version, input_reference, tau:
        Provenance of the screening result, recorded verbatim.

    Raises
    ------
    ValueError
        If neither an episode result nor an explanation is given, or if the
        follow-up answers are semantically invalid (checked against the
        questionnaire definition, not just their types).
    """
    if episode_result is None and explanation is None:
        raise ValueError("build_support_report needs an episode_result or an "
                         "explanation; nothing can be assembled without one")

    answers = (FollowUpAnswers(answers=[]) if followup_answers is None
               else (followup_answers if isinstance(followup_answers,
                                                    FollowUpAnswers)
                     else FollowUpAnswers.model_validate(
                         {"answers": list(followup_answers)})))
    problems = engine.validate_followup_answers(answers)
    if problems:
        raise ValueError("invalid follow-up answers: " + "; ".join(problems))

    screening_limitations = [str(x) for x in
                             ((explanation or {}).get("limitations") or [])]
    if episode_result is not None:
        screening = screening_result_from_episode(
            episode_result, result_id=result_id, model_name=model_name,
            model_version=model_version, input_reference=input_reference,
            limitations=screening_limitations, tau=tau)
        evidence = evidence_from_episode(episode_result)
    else:
        screening = screening_result_from_explanation(
            explanation, result_id=result_id, model_name=model_name,
            model_version=model_version, input_reference=input_reference,
            limitations=screening_limitations)
        evidence = _observed_evidence_from_explanation(explanation)

    expl = (explanation_from_dict(explanation, result_id=screening.result_id,
                                  observed_evidence=evidence)
            if explanation is not None else None)

    assessments, recommendations = engine.evaluate(evidence, answers)

    limitations = _report_limitations(assessments, recommendations)
    disclaimer = (screening_disclaimer() + " Support suggestions are optional "
                  "and have not been clinically evaluated.")

    return ScreeningReport(
        report_id=report_id or f"rep-{screening.result_id}",
        screening_result=screening,
        explanation=expl,
        support_assessments=assessments,
        recommendations=recommendations,
        limitations=limitations,
        disclaimer=disclaimer,
    )
