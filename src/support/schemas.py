"""Pydantic data contracts for the support layer — P4 (screening + support).

One module owns every structured object that crosses a component boundary in
the support layer. The contracts enforce the invariants the design depends on:

* **Observed vs derived vs user-reported evidence is a type-level distinction**
  (``EvidenceItem.source_type``), never a convention.
* **A hypothesis is never a confirmed need** (``SupportNeedAssessment`` —
  status and ``user_confirmed`` are cross-validated).
* **Attribution direction matches the sign of the contribution value**
  (``FeatureContribution``), and all numerics are finite.
* **Recommendations are traceable** — every ``related_assessment_id`` and
  ``evidence_reference`` must resolve within the report (validated at assembly
  time in ``src/support/report.py``).
* **No invented semantics** — the only probability is the predictor's own
  calibrated ``p_hat``; there is no "autism probability" field, and no clinical
  cut-off is encoded anywhere in this module.

Nothing in this module knows about the RL environment, the predictor, or the
demo; it is pure data with validation, so it can be tested in isolation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

SCHEMA_VERSION = "support-report/1.0"

#: Instruments this project scores: 10 binary Q-CHAT-10 items (the primary
#: benchmark) or the 25-item Polish Q-CHAT. Item codes are A1..A{max_items}.
MAX_INSTRUMENT_ITEMS = 25


# ---------------------------------------------------------------------------
# enums — the closed vocabularies
# ---------------------------------------------------------------------------

class DecisionEnum(str, Enum):
    """The repository's two decision vocabularies, normalised.

    ``run_episode`` emits "REFER" (src/env/environment.py) while the demo emits
    "REFERRAL_RECOMMENDED"; both normalise to the same member here.
    """
    REFERRAL_RECOMMENDED = "REFERRAL_RECOMMENDED"
    NO_REFERRAL_INDICATED = "NO_REFERRAL_INDICATED"


class SourceType(str, Enum):
    OBSERVED_RESPONSE = "observed_response"   # a recorded questionnaire answer
    MODEL_DERIVED = "model_derived"           # produced by a fitted component
    USER_REPORTED = "user_reported"           # the person's own follow-up answer


class AttributionDirection(str, Enum):
    RAISES = "raises"
    LOWERS = "lowers"
    NEUTRAL = "neutral"


class GenerationStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class AssessmentStatus(str, Enum):
    """Lifecycle of a support-need assessment.

    ``hypothesis_from_observed`` and ``user_confirmed`` are strictly different:
    a hypothesis is something the observed responses suggest *asking about*;
    a confirmed need is something the person said they want help with.
    ``user_stated_preference`` is neither a hypothesis nor a need: it records a
    stated preference (e.g. preferred information format) that shapes *how*
    suggestions are offered.
    """
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HYPOTHESIS_FROM_OBSERVED = "hypothesis_from_observed"
    USER_CONFIRMED = "user_confirmed"
    USER_STATED_PREFERENCE = "user_stated_preference"
    USER_DECLINED = "user_declined"
    UNKNOWN = "unknown"


class AssessmentMethod(str, Enum):
    OBSERVED_ITEM_RULE = "observed_item_rule"   # transparent rule over item codes
    USER_REPORT = "user_report"                  # the person's own statement


class RecommendationBasis(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    HYPOTHESIS_OPTED_IN = "hypothesis_opted_in"
    GENERAL_GUIDANCE = "general_guidance"


class AnswerValue(str, Enum):
    """Follow-up answer vocabulary — every item allows abstention."""
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"
    NOT_APPLICABLE = "not_applicable"
    PREFER_NOT_TO_ANSWER = "prefer_not_to_answer"


# ---------------------------------------------------------------------------
# evidence and explanation
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """One verified piece of evidence: a recorded answer, a model output, or a
    user-reported experience. References must point at something that exists."""
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1, max_length=64)
    source_type: SourceType
    item_code: Optional[str] = Field(
        default=None, description="A1..A{MAX_INSTRUMENT_ITEMS}; required for "
                                  "observed responses, optional otherwise")
    observed_value: Optional[int] = Field(
        default=None, description="the recorded response, when applicable")
    source_reference: str = Field(
        ..., description="where this came from: e.g. 'episode:trace[2]' for an "
                         "observed answer, 'followup:q3' for a user report")
    description: str = Field(default="", max_length=500)

    @field_validator("item_code")
    @classmethod
    def _check_item_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.startswith("A") or not v[1:].isdigit():
            raise ValueError(f"item_code must look like 'A3', got {v!r}")
        if not 1 <= int(v[1:]) <= MAX_INSTRUMENT_ITEMS:
            raise ValueError(f"item_code {v!r} out of range "
                             f"A1..A{MAX_INSTRUMENT_ITEMS}")
        return v

    @model_validator(mode="after")
    def _observed_needs_value_and_code(self) -> "EvidenceItem":
        if self.source_type is SourceType.OBSERVED_RESPONSE:
            if self.item_code is None:
                raise ValueError("an observed_response evidence item must name "
                                 "its item_code")
            if self.observed_value is None:
                raise ValueError("an observed_response evidence item must "
                                 "record its observed_value")
        return self


class FeatureContribution(BaseModel):
    """A model-derived attribution for one questionnaire item.

    ``contribution_value`` is on the predictor's probability scale; it is a
    statement about the model, not a causal or clinical measure.
    """
    model_config = ConfigDict(extra="forbid")

    item_code: str
    contribution_value: float = Field(..., allow_inf_nan=False)
    direction: AttributionDirection
    explanation_method: str = Field(..., min_length=1, max_length=64)
    baseline_reference: Optional[str] = Field(
        default=None, description="the value function's baseline, e.g. "
                                  "'prior_only_unasked_all_items'")

    @field_validator("item_code")
    @classmethod
    def _check_item_code(cls, v: str) -> str:
        if not v.startswith("A") or not v[1:].isdigit():
            raise ValueError(f"item_code must look like 'A3', got {v!r}")
        if not 1 <= int(v[1:]) <= MAX_INSTRUMENT_ITEMS:
            raise ValueError(f"item_code {v!r} out of range")
        return v

    @model_validator(mode="after")
    def _direction_matches_sign(self) -> "FeatureContribution":
        v = self.contribution_value
        if self.direction is AttributionDirection.RAISES and not v > 0:
            raise ValueError("direction 'raises' requires contribution_value > 0")
        if self.direction is AttributionDirection.LOWERS and not v < 0:
            raise ValueError("direction 'lowers' requires contribution_value < 0")
        if self.direction is AttributionDirection.NEUTRAL and v != 0.0:
            raise ValueError("direction 'neutral' requires contribution_value "
                             "== 0.0")
        return self


class UncertaintyInfo(BaseModel):
    """The only uncertainty this architecture supports: model stability.

    Explicitly *not* clinical uncertainty and *not* confidence in the person.
    """
    model_config = ConfigDict(extra="forbid")

    available: bool
    method: Optional[str] = None
    interval_low: Optional[float] = Field(default=None, allow_inf_nan=False,
                                          ge=0.0, le=1.0)
    interval_high: Optional[float] = Field(default=None, allow_inf_nan=False,
                                           ge=0.0, le=1.0)
    interpretation: str = Field(default="", max_length=400)

    @model_validator(mode="after")
    def _bounds_consistent(self) -> "UncertaintyInfo":
        if not self.available:
            return self
        if self.interval_low is None or self.interval_high is None:
            raise ValueError("an available interval must provide both bounds")
        if self.interval_low > self.interval_high:
            raise ValueError("interval_low must be <= interval_high")
        if not self.method:
            raise ValueError("an available interval must name its method")
        return self


class ExplanationResult(BaseModel):
    """The existing outcome explanation, wrapped as a typed contract.

    ``screening_result_id`` links the explanation to the exact screening result
    it explains; the report assembler enforces the match.
    """
    model_config = ConfigDict(extra="forbid")

    screening_result_id: str
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    opposing_evidence: List[EvidenceItem] = Field(default_factory=list)
    feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    explanation_method: str = Field(default="exact_group_shapley")
    limitations: List[str] = Field(default_factory=list)
    uncertainty: Optional[UncertaintyInfo] = None
    generation_status: GenerationStatus = GenerationStatus.UNAVAILABLE

    @field_validator("supporting_evidence", "opposing_evidence")
    @classmethod
    def _unique_ids(cls, v: List[EvidenceItem]) -> List[EvidenceItem]:
        ids = [e.evidence_id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique within a list")
        return v


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------

class ScreeningResult(BaseModel):
    """The frozen screening outcome. The only probability is the predictor's
    own calibrated ``p_hat`` — no other probability semantics exist here."""
    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(..., min_length=1, max_length=64)
    decision: DecisionEnum
    p_hat: float = Field(..., allow_inf_nan=False, ge=0.0, le=1.0)
    tau: float = Field(default=0.5, allow_inf_nan=False, ge=0.0, le=1.0)
    stop_reason: Optional[str] = None
    items_asked: List[str] = Field(default_factory=list)
    n_questions_asked: int = Field(default=0, ge=0)
    budget: Optional[int] = Field(default=None, ge=0)
    model_name: str = Field(default="MaskedPredictor")
    model_version: str = Field(default="unknown")
    input_reference: str = Field(default="", description="e.g. 'episode:<id>'")
    limitations: List[str] = Field(default_factory=list)
    disclaimer: str = Field(default="")

    @field_validator("decision", mode="before")
    @classmethod
    def _normalise_decision(cls, v: Any) -> Any:
        """Accept both repository vocabularies; everything else is invalid."""
        if v == "REFER":
            return DecisionEnum.REFERRAL_RECOMMENDED
        return v

    @field_validator("items_asked")
    @classmethod
    def _check_item_codes(cls, v: List[str]) -> List[str]:
        import re
        for code in v:
            if not re.fullmatch(r"A(?:[1-9]|[1-9][0-9])", code) or \
                    not 1 <= int(code[1:]) <= MAX_INSTRUMENT_ITEMS:
                raise ValueError(f"items_asked entries must look like 'A3' "
                                 f"within A1..A{MAX_INSTRUMENT_ITEMS}, got "
                                 f"{code!r}")
        return v

    @model_validator(mode="after")
    def _items_consistent(self) -> "ScreeningResult":
        if len(self.items_asked) != self.n_questions_asked:
            raise ValueError("items_asked length must equal n_questions_asked")
        return self


# ---------------------------------------------------------------------------
# support needs
# ---------------------------------------------------------------------------

class SupportNeedAssessment(BaseModel):
    """A candidate area of support need.

    ``status`` and ``user_confirmed`` are cross-validated: a hypothesis is
    never presented as a confirmed need, and a declined/unknown answer is never
    interpreted as a need in either direction.
    """
    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(..., min_length=1, max_length=64)
    domain: str = Field(..., min_length=1, max_length=64)
    status: AssessmentStatus
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    assessment_method: AssessmentMethod
    user_confirmed: bool = False
    user_preference: Optional[str] = Field(default=None, max_length=300)
    followup_question_id: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_agrees_with_confirmation(self) -> "SupportNeedAssessment":
        if self.status is AssessmentStatus.USER_CONFIRMED and not self.user_confirmed:
            raise ValueError("status 'user_confirmed' requires "
                             "user_confirmed=True")
        if self.status is AssessmentStatus.HYPOTHESIS_FROM_OBSERVED and \
                self.user_confirmed:
            raise ValueError("a hypothesis must not be marked user_confirmed")
        if self.status is AssessmentStatus.USER_CONFIRMED and \
                self.assessment_method is not AssessmentMethod.USER_REPORT:
            raise ValueError("a confirmed need must come from a user report")
        if self.status is AssessmentStatus.USER_STATED_PREFERENCE and \
                self.user_preference is None:
            raise ValueError("a stated preference must record the preference")
        if self.status is AssessmentStatus.USER_STATED_PREFERENCE and \
                self.user_confirmed:
            raise ValueError("a preference is not a confirmed need")
        return self


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

class SupportRecommendation(BaseModel):
    """An optional support strategy. Never a prescription, never deficit-framed;
    every suggestion carries its basis and its provenance."""
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(..., min_length=1, max_length=64)
    domain: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=800)
    intended_purpose: str = Field(default="", max_length=300)
    evidence_references: List[str] = Field(default_factory=list)
    applicability_conditions: List[str] = Field(default_factory=list)
    related_assessment_ids: List[str] = Field(default_factory=list)
    basis: RecommendationBasis
    why_selected: str = Field(default="", max_length=400,
                              description="plain-language reason, referencing the "
                                          "triggering assessment/answer")
    limitations: List[str] = Field(default_factory=list)
    source_attribution: str = Field(default="", max_length=300)
    review_status: Literal["pending_expert_review", "approved"] = Field(
        default="pending_expert_review",
        description="curated content is pending expert review unless a human "
                    "has approved it; the report discloses the status")

    @model_validator(mode="after")
    def _confirmed_bases_need_triggering_assessments(self) -> "SupportRecommendation":
        if self.basis in (RecommendationBasis.USER_CONFIRMED,
                          RecommendationBasis.HYPOTHESIS_OPTED_IN) and \
                not self.related_assessment_ids:
            raise ValueError(
                "a need-based recommendation must reference the assessment(s) "
                "that triggered it")
        return self


class RecommendationFeedback(BaseModel):
    """Optional user feedback on one recommendation. Descriptive only — never
    evidence of clinical effectiveness."""
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(..., min_length=1, max_length=64)
    relevance_rating: Optional[int] = Field(default=None, ge=1, le=5)
    acceptability_rating: Optional[int] = Field(default=None, ge=1, le=5)
    usefulness_rating: Optional[int] = Field(default=None, ge=1, le=5)
    optional_comment: Optional[str] = Field(default=None, max_length=500)


# ---------------------------------------------------------------------------
# follow-up questionnaire
# ---------------------------------------------------------------------------

class FollowUpAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1, max_length=64)
    value: AnswerValue
    #: For choice (preference) questions only; validated against the
    #: question's declared vocabulary by src/support/engine.py.
    choice: Optional[str] = Field(default=None, max_length=40)
    optional_comment: Optional[str] = Field(default=None, max_length=300)


class FollowUpAnswers(BaseModel):
    """A completed (possibly partial) follow-up questionnaire.

    Abstentions are first-class values; a missing question is simply absent,
    and never treated as an answer.
    """
    model_config = ConfigDict(extra="forbid")

    answers: List[FollowUpAnswer] = Field(default_factory=list)

    @field_validator("answers")
    @classmethod
    def _unique_questions(cls, v: List[FollowUpAnswer]) -> List[FollowUpAnswer]:
        ids = [a.question_id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError("a question may be answered once")
        return v

    def answered(self, question_id: str) -> Optional[AnswerValue]:
        for a in self.answers:
            if a.question_id == question_id:
                return a.value
        return None


# ---------------------------------------------------------------------------
# the assembled report
# ---------------------------------------------------------------------------

class ScreeningReport(BaseModel):
    """The validated report returned to the user. Fully nested; unknown fields
    are rejected so the contract cannot drift silently."""
    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(..., min_length=1, max_length=64)
    screening_result: ScreeningResult
    explanation: Optional[ExplanationResult] = None
    support_assessments: List[SupportNeedAssessment] = Field(default_factory=list)
    recommendations: List[SupportRecommendation] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = Field(default=SCHEMA_VERSION)
    disclaimer: str = Field(default="")

    @model_validator(mode="after")
    def _references_resolve(self) -> "ScreeningReport":
        """Cross-entity integrity: every id reference inside the report must
        resolve to an entity inside the report."""
        if self.explanation is not None:
            if self.explanation.screening_result_id != self.screening_result.result_id:
                raise ValueError(
                    "explanation.screening_result_id must match the report's "
                    "screening_result.result_id")
        assessment_ids = {a.assessment_id for a in self.support_assessments}
        if len(assessment_ids) != len(self.support_assessments):
            raise ValueError("assessment ids must be unique within the report")
        evidence_ids: set[str] = set()
        for a in self.support_assessments:
            for e in a.supporting_evidence:
                if e.evidence_id in evidence_ids:
                    raise ValueError(f"duplicate evidence_id {e.evidence_id!r}")
                evidence_ids.add(e.evidence_id)
        for r in self.recommendations:
            for aid in r.related_assessment_ids:
                if aid not in assessment_ids:
                    raise ValueError(
                        f"recommendation {r.recommendation_id!r} references "
                        f"unknown assessment {aid!r}")
            for ref in r.evidence_references:
                # references may point at assessments or at approved external
                # guidance; the provenance registry (src/support/strategies.py)
                # validates the external kind at assembly time.
                if ref in assessment_ids:
                    continue
                if ref.startswith(("guidance:", "assessment:")):
                    continue
                raise ValueError(
                    f"recommendation {r.recommendation_id!r} has an "
                    f"unresolvable evidence_reference {ref!r}")
        return self


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python types.

    The report is serialised by the demo API and written to JSON artifacts;
    numpy scalars are not serialisable by the stdlib json module, so the layer
    that produces the data guarantees its own serialisability.
    """
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):          # pydantic models
        return to_json_safe(obj.model_dump(mode="json"))
    if isinstance(obj, bool):
        return bool(obj)
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [to_json_safe(v) for v in obj.tolist()]
    except ImportError:  # pragma: no cover - numpy is a project dependency
        pass
    return obj
