"""Pydantic data contracts for the support layer — P4 (screening + support).

One module owns every structured object that crosses a component boundary in
the support layer. The contracts enforce the invariants the design depends on:

* **Observed vs derived vs user-reported evidence is a type-level distinction**
  (``EvidenceItem.source_type``), never a convention.
* **A suggestion may only come from an answer that was actually given**
  (``SupportRecommendation.triggering_question_ids`` must resolve to the
  report's observed evidence, checked at assembly time).
* **Attribution direction matches the sign of the contribution value**
  (``FeatureContribution``), and all numerics are finite.
* **Question IDs are validated against the instrument itself** — an item code
  the Q-CHAT-10 contract does not define is rejected, so a typo or an invented
  question cannot enter a report.
* **Unmeasured areas are labelled, never inferred** — the report carries
  ``unassessed_areas`` for the support domains the questionnaire does not ask
  about, and nothing here can turn them into a need.
* **No invented semantics** — the only probability is the predictor's own
  calibrated ``p_hat``; there is no "autism probability" field, and no clinical
  cut-off is encoded anywhere in this module.

There is deliberately **no second questionnaire** in this module: every
recommendation is derived from the screening responses the person already
gave. The follow-up questionnaire that used to live here was removed in the
P4 revision — see ``docs/SUPPORT_LAYER_DESIGN.md``.

Nothing in this module knows about the RL environment, the predictor, or the
demo; it is pure data with validation, so it can be tested in isolation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from src.support.questionnaire import (ATYPICAL_VALUE, ITEM_BY_CODE,
                                       TYPICAL_VALUE)

SCHEMA_VERSION = "support-report/2.0"

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
    #: A statement the person made themselves. Retained so the three kinds of
    #: evidence stay a type-level distinction; nothing in support-report/2.0
    #: produces it, because the second questionnaire was removed.
    USER_REPORTED = "user_reported"


class AttributionDirection(str, Enum):
    RAISES = "raises"
    LOWERS = "lowers"
    NEUTRAL = "neutral"


class GenerationStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class AssessmentStatus(str, Enum):
    """What is known about one support area, from the screening responses alone.

    * ``EVIDENCE_SUGGESTED`` — at least one of this area's linked items was
      recorded as atypical. This is *not* a finding of need: it marks an area
      where an optional idea is offered because that kind of answer sometimes
      makes it useful. Only the person can say whether it does.
    * ``NO_EVIDENCE`` — the area's linked items were asked and every answer was
      typical (or the items were not asked in this session), so nothing is
      offered.
    * ``NOT_MEASURED`` — the questionnaire does not ask about this area at all.
      Nothing can be said, in either direction.
    """
    EVIDENCE_SUGGESTED = "evidence_suggested"
    NO_EVIDENCE = "no_evidence"
    NOT_MEASURED = "not_measured"


class AssessmentMethod(str, Enum):
    OBSERVED_ITEM_RULE = "observed_item_rule"   # transparent rule over item codes


class RecommendationBasis(str, Enum):
    """Why a suggestion was offered. There is exactly one basis now that the
    second questionnaire is gone: an observed response pattern.

    ``OBSERVED_RESPONSE_PATTERN`` means the person's own screening answer is the
    trigger, and the recommendation carries the item codes and the observed
    values that produced it. It never means a need was established.
    """
    OBSERVED_RESPONSE_PATTERN = "observed_response_pattern"


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


class QuestionnaireEvidence(BaseModel):
    """One answer the person actually gave to the screening questionnaire.

    This is the *only* input the support layer may reason about. Every field is
    validated against the instrument: ``question_id`` must be an item the
    Q-CHAT-10 contract defines, ``question_text`` must be that item's verified
    wording, ``feature_id`` must be the contract's feature name for it, and
    ``response`` must be one of the two binary values the instrument produces.
    A record that does not match is rejected rather than silently normalised.
    """
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1, max_length=64)
    question_text: str = Field(..., min_length=1, max_length=400)
    response: int
    feature_id: str = Field(..., min_length=1, max_length=64)

    @model_validator(mode="after")
    def _agrees_with_the_instrument(self) -> "QuestionnaireEvidence":
        item = ITEM_BY_CODE.get(self.question_id)
        if item is None:
            raise ValueError(
                f"question_id {self.question_id!r} is not an item of the "
                f"screening questionnaire (verified items: A1..A10)")
        if self.feature_id != item.feature_id:
            raise ValueError(
                f"feature_id {self.feature_id!r} does not match item "
                f"{self.question_id} (the contract defines {item.feature_id})")
        if self.question_text != item.question_text:
            raise ValueError(
                f"question_text for {self.question_id} does not match the "
                f"verified wording of that item")
        return self

    @field_validator("response")
    @classmethod
    def _binary_response(cls, v: int) -> int:
        if v not in (TYPICAL_VALUE, ATYPICAL_VALUE):
            raise ValueError(
                f"response must be the binary value the instrument produces "
                f"({TYPICAL_VALUE} = typical, {ATYPICAL_VALUE} = atypical), got {v}")
        return v


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
    """What is known about one support area, from the screening answers alone.

    ``status`` and the evidence list agree by construction: an assessment may
    only be ``EVIDENCE_SUGGESTED`` when it actually carries the atypical
    observed responses that triggered it, and a suggestion is a *possible*
    support idea — never a finding that a need exists.
    """
    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(..., min_length=1, max_length=64)
    domain: str = Field(..., min_length=1, max_length=64)
    label: str = Field(default="", max_length=120)
    status: AssessmentStatus
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    #: The questionnaire items whose observed responses triggered this
    #: assessment. Empty unless the status is EVIDENCE_SUGGESTED.
    triggering_question_ids: List[str] = Field(default_factory=list)
    assessment_method: AssessmentMethod
    limitations: List[str] = Field(default_factory=list)

    @field_validator("triggering_question_ids")
    @classmethod
    def _ids_are_real_items(cls, v: List[str]) -> List[str]:
        for code in v:
            if code not in ITEM_BY_CODE:
                raise ValueError(
                    f"triggering_question_id {code!r} is not an item of the "
                    f"screening questionnaire")
        if len(v) != len(set(v)):
            raise ValueError("triggering question ids must be unique")
        return v

    @field_validator("supporting_evidence")
    @classmethod
    def _unique_evidence_ids(cls, v: List[EvidenceItem]) -> List[EvidenceItem]:
        ids = [e.evidence_id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique within an assessment")
        return v

    @model_validator(mode="after")
    def _status_agrees_with_evidence(self) -> "SupportNeedAssessment":
        if self.status is AssessmentStatus.EVIDENCE_SUGGESTED:
            if not self.triggering_question_ids:
                raise ValueError("an evidence-suggested assessment must name the "
                                 "questionnaire items that suggested it")
            if not self.supporting_evidence:
                raise ValueError("an evidence-suggested assessment must carry the "
                                 "observed responses that suggested it")
            observed = {e.item_code for e in self.supporting_evidence}
            missing = set(self.triggering_question_ids) - observed
            if missing:
                raise ValueError(
                    f"assessment names triggering items {sorted(missing)} that are "
                    f"not in its own evidence {sorted(observed)}")
        else:
            if self.triggering_question_ids:
                raise ValueError(
                    f"status {self.status.value!r} carries triggering question "
                    f"ids but nothing was suggested from a response pattern")
        return self


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

class SupportRecommendation(BaseModel):
    """An optional support suggestion derived from recorded screening answers.

    Never a prescription, never deficit-framed, never personalised by a model:
    every suggestion carries the item codes and observed values that triggered
    it, the plain-language reason it was offered, its provenance and its review
    status. A suggestion with no recorded evidence behind it cannot be built.
    """
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(..., min_length=1, max_length=64)
    domain: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=800)
    intended_purpose: str = Field(default="", max_length=300)
    #: The questionnaire items whose recorded answers justify offering this.
    triggering_question_ids: List[str] = Field(default_factory=list)
    #: The observed response for each triggering item, same order as
    #: ``triggering_question_ids`` (so "A4 = atypical" is explicit, not implied).
    triggering_responses: List[int] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)
    applicability_conditions: List[str] = Field(default_factory=list)
    related_assessment_ids: List[str] = Field(default_factory=list)
    basis: RecommendationBasis
    #: Where this content comes from (registry key, e.g.
    #: "guidance:project-curated-v1"). Recorded so no source is invented.
    recommendation_source: str = Field(default="", max_length=64)
    why_selected: str = Field(default="", max_length=400,
                              description="plain-language reason, naming the "
                                          "triggering items and their answers")
    limitations: List[str] = Field(default_factory=list)
    source_attribution: str = Field(default="", max_length=300)
    review_status: Literal["pending_expert_review", "approved"] = Field(
        default="pending_expert_review",
        description="curated content is pending expert review unless a human "
                    "has approved it; the report discloses the status")

    @field_validator("triggering_question_ids")
    @classmethod
    def _ids_are_real_items(cls, v: List[str]) -> List[str]:
        for code in v:
            if code not in ITEM_BY_CODE:
                raise ValueError(
                    f"triggering_question_id {code!r} is not an item of the "
                    f"screening questionnaire")
        return v

    @field_validator("triggering_responses")
    @classmethod
    def _responses_are_binary(cls, v: List[int]) -> List[int]:
        for r in v:
            if r not in (TYPICAL_VALUE, ATYPICAL_VALUE):
                raise ValueError(
                    f"triggering responses must be binary instrument values, "
                    f"got {r}")
        return v

    @model_validator(mode="after")
    def _needs_its_own_evidence(self) -> "SupportRecommendation":
        if not self.triggering_question_ids:
            raise ValueError(
                "a suggestion must name the questionnaire items that triggered "
                "it; there is no basis-free suggestion in this layer")
        if len(self.triggering_responses) != len(self.triggering_question_ids):
            raise ValueError("each triggering question id needs its recorded "
                             "response")
        if not self.recommendation_source:
            raise ValueError("a suggestion must record where its content came "
                             "from")
        if self.basis is not RecommendationBasis.OBSERVED_RESPONSE_PATTERN:
            raise ValueError("suggestions only come from observed response "
                             "patterns")
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
# the assembled report
# ---------------------------------------------------------------------------

class ScreeningReport(BaseModel):
    """The validated report returned to the user. Fully nested; unknown fields
    are rejected so the contract cannot drift silently."""
    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(..., min_length=1, max_length=64)
    screening_result: ScreeningResult
    explanation: Optional[ExplanationResult] = None
    #: The answers the person actually gave, in the instrument's own wording.
    questionnaire_evidence: List[QuestionnaireEvidence] = Field(default_factory=list)
    support_assessments: List[SupportNeedAssessment] = Field(default_factory=list)
    recommendations: List[SupportRecommendation] = Field(default_factory=list)
    #: Support areas the screening questionnaire does not ask about. Labels
    #: only — nothing here may be read as a statement about the person.
    unassessed_areas: List[str] = Field(default_factory=list)
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
        # The same recorded answer may legitimately trigger more than one
        # support area (A1 is evidence for both "getting attention" and "extra
        # processing time"), so evidence ids are unique *within* an assessment
        # — enforced on SupportNeedAssessment — and not across the report.
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
            # Fail closed: a suggestion may only be justified by answers this
            # report can actually show. A trigger id outside the recorded
            # evidence is a bug in the rule table, not something to drop.
            observed = {e.question_id for e in self.questionnaire_evidence}
            observed |= {e.item_code for e in
                         (self.explanation.supporting_evidence if self.explanation
                          else [])}
            observed |= {e.item_code for e in
                         (self.explanation.opposing_evidence if self.explanation
                          else [])}
            for r in self.recommendations:
                unknown = set(r.triggering_question_ids) - observed
                if unknown:
                    raise ValueError(
                        f"recommendation {r.recommendation_id!r} is justified by "
                        f"question id(s) {sorted(unknown)} that are not part of "
                        f"this report's recorded evidence")
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
