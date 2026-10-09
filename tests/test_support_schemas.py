"""Tests for the Pydantic support-layer contracts — P4 (revised).

Covers the invariants the design depends on: the instrument-level validation of
question ids, the observed/model-derived/user-reported distinction, the rule
that a suggestion must be justified by recorded answers, attribution direction
matching the sign of the contribution, finite numerics, unknown-field
rejection, reference resolution inside the report, the fail-closed
evidence-sufficiency check, and JSON-safety. All fixtures are synthetic — no
real participant data anywhere.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.support.questionnaire import ITEM_BY_CODE
from src.support.schemas import (
    AssessmentMethod, AssessmentStatus, DecisionEnum, EvidenceItem,
    ExplanationResult, FeatureContribution, GenerationStatus,
    QuestionnaireEvidence, RecommendationBasis, RecommendationFeedback,
    ScreeningReport, ScreeningResult, SourceType, SupportNeedAssessment,
    SupportRecommendation, UncertaintyInfo, to_json_safe)


# ---------------------------------------------------------------------------
# helpers — every fixture is built from the verified instrument
# ---------------------------------------------------------------------------

def _evidence(code: str = "A3", value: int = 1,
              evidence_id: str = "") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id or f"ev-obs-{code}",
        source_type=SourceType.OBSERVED_RESPONSE,
        item_code=code, observed_value=value,
        source_reference="episode:trace[0]", description="synthetic")


def _question_evidence(code: str = "A3", value: int = 1) -> QuestionnaireEvidence:
    item = ITEM_BY_CODE[code]
    return QuestionnaireEvidence(question_id=item.item_code,
                                 question_text=item.question_text,
                                 response=value,
                                 feature_id=item.feature_id)


def _screening(**kw):
    base = dict(result_id="sr-1", decision="REFER", p_hat=0.58,
                items_asked=["A1", "A2", "A3", "A4", "A6", "A8"],
                n_questions_asked=6, budget=6, model_name="MaskedPredictor",
                model_version="2", input_reference="episode:e-1")
    base.update(kw)
    return ScreeningResult(**base)


def _assessment(**kw):
    base = dict(assessment_id="as-getting_message_across",
                domain="getting_message_across",
                label="Getting a message across",
                status="evidence_suggested",
                supporting_evidence=[_evidence("A4", 1)],
                triggering_question_ids=["A4"],
                assessment_method="observed_item_rule",
                limitations=["synthetic"])
    base.update(kw)
    return SupportNeedAssessment(**base)


def _recommendation(**kw):
    base = dict(recommendation_id="rec-as-x-visual_supports",
                domain="getting_message_across",
                title="Pair words with something to see",
                description="synthetic strategy",
                triggering_question_ids=["A4"],
                triggering_responses=[1],
                basis="observed_response_pattern",
                recommendation_source="guidance:project-curated-v1",
                why_selected="synthetic reason",
                evidence_references=["guidance:project-curated-v1"])
    base.update(kw)
    return SupportRecommendation(**base)


def _explanation(**kw):
    base = dict(screening_result_id="sr-1",
                supporting_evidence=[_evidence("A3", 1, "ev-1")],
                opposing_evidence=[_evidence("A2", 0, "ev-2")],
                feature_contributions=[
                    FeatureContribution(item_code="A3", contribution_value=0.2,
                                        direction="raises",
                                        explanation_method="exact_group_shapley",
                                        baseline_reference="prior_only_unasked_all_items"),
                    FeatureContribution(item_code="A2", contribution_value=-0.1,
                                        direction="lowers",
                                        explanation_method="exact_group_shapley"),
                ],
                limitations=["hypothetical"],
                uncertainty=UncertaintyInfo(available=False),
                generation_status="complete")
    base.update(kw)
    return ExplanationResult(**base)


def _report(**kw):
    base = dict(report_id="rep-1", screening_result=_screening(),
                explanation=_explanation(),
                questionnaire_evidence=[_question_evidence("A4", 1)],
                support_assessments=[_assessment()],
                recommendations=[_recommendation()],
                unassessed_areas=["Sensory comfort and environment"],
                limitations=["hypothetical"],
                disclaimer="Screening support only. Not a diagnosis.")
    base.update(kw)
    return ScreeningReport(**base)


# ---------------------------------------------------------------------------
# screening result (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_decision_vocabularies_normalise():
    assert _screening(decision="REFER").decision is \
        DecisionEnum.REFERRAL_RECOMMENDED
    assert _screening(decision="REFERRAL_RECOMMENDED").decision is \
        DecisionEnum.REFERRAL_RECOMMENDED
    assert _screening(decision="NO_REFERRAL_INDICATED").decision is \
        DecisionEnum.NO_REFERRAL_INDICATED
    with pytest.raises(ValidationError):
        _screening(decision="PROBABLY_AUTISTIC")


def test_p_hat_range_and_finiteness():
    for bad in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            _screening(p_hat=bad)


def test_items_asked_must_match_count():
    with pytest.raises(ValidationError, match="n_questions_asked"):
        _screening(items_asked=["A1", "A2"], n_questions_asked=3)
    with pytest.raises(ValidationError, match="items_asked entries"):
        _screening(items_asked=["B1"])


def test_no_invented_probability_field():
    fields = set(ScreeningResult.model_fields)
    for banned in ("autism_probability", "diagnosis", "clinical_probability",
                   "score"):
        assert banned not in fields


# ---------------------------------------------------------------------------
# evidence items
# ---------------------------------------------------------------------------

def test_observed_evidence_requires_code_and_value():
    base = dict(evidence_id="ev-1", source_type="observed_response",
                item_code="A3", observed_value=1,
                source_reference="synthetic")
    with pytest.raises(ValidationError, match="item_code"):
        EvidenceItem(**{**base, "item_code": None})
    with pytest.raises(ValidationError, match="observed_value"):
        EvidenceItem(**{**base, "observed_value": None})


def test_item_code_format_and_range():
    for good in ("A1", "A10", "A25"):
        EvidenceItem(evidence_id=f"ev-{good}", source_type="observed_response",
                     item_code=good, observed_value=1,
                     source_reference="synthetic")
    for bad in ("A0", "A26", "A", "3A", "a3", "A3 "):
        with pytest.raises(ValidationError):
            EvidenceItem(evidence_id="ev-bad", source_type="observed_response",
                         item_code=bad, observed_value=1,
                         source_reference="synthetic")


def test_unknown_fields_rejected_everywhere():
    base = dict(evidence_id="ev-1", source_type="observed_response",
                item_code="A3", observed_value=1,
                source_reference="synthetic")
    with pytest.raises(ValidationError):
        EvidenceItem(**{**base, "surprise_field": 1})
    with pytest.raises(ValidationError):
        _screening(extra="x")


# ---------------------------------------------------------------------------
# questionnaire evidence — validated against the instrument itself
# ---------------------------------------------------------------------------

def test_questionnaire_evidence_accepts_a_recorded_answer():
    ev = _question_evidence("A8", 1)
    assert ev.question_id == "A8"
    assert ev.feature_id == "feature_8"
    assert ev.response == 1
    assert ev.question_text == ITEM_BY_CODE["A8"].question_text


def test_questionnaire_evidence_rejects_unknown_question_ids():
    item = ITEM_BY_CODE["A3"]
    for bad in ("Q1", "A11", "A0", "a3"):
        with pytest.raises(ValidationError, match="not an item"):
            QuestionnaireEvidence(question_id=bad,
                                  question_text=item.question_text,
                                  response=1, feature_id=item.feature_id)


def test_questionnaire_evidence_rejects_wrong_feature_or_wording():
    item = ITEM_BY_CODE["A3"]
    with pytest.raises(ValidationError, match="feature_id"):
        QuestionnaireEvidence(question_id="A3",
                              question_text=item.question_text,
                              response=1, feature_id="feature_4")
    with pytest.raises(ValidationError, match="question_text"):
        QuestionnaireEvidence(question_id="A3", question_text="made up",
                              response=1, feature_id=item.feature_id)


def test_questionnaire_evidence_rejects_non_binary_responses():
    item = ITEM_BY_CODE["A3"]
    for bad in (2, -1, 7):
        with pytest.raises(ValidationError, match="binary"):
            QuestionnaireEvidence(question_id="A3",
                                  question_text=item.question_text,
                                  response=bad, feature_id=item.feature_id)


# ---------------------------------------------------------------------------
# feature contributions
# ---------------------------------------------------------------------------

def test_direction_must_match_sign():
    for value, direction in ((0.2, "raises"), (-0.2, "lowers"), (0.0, "neutral")):
        FeatureContribution(item_code="A3", contribution_value=value,
                            direction=direction,
                            explanation_method="exact_group_shapley")
    with pytest.raises(ValidationError, match="raises"):
        FeatureContribution(item_code="A3", contribution_value=-0.1,
                            direction="raises",
                            explanation_method="exact_group_shapley")
    with pytest.raises(ValidationError, match="lowers"):
        FeatureContribution(item_code="A3", contribution_value=0.1,
                            direction="lowers",
                            explanation_method="exact_group_shapley")


def test_contribution_must_be_finite():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            FeatureContribution(item_code="A3", contribution_value=bad,
                                direction="raises",
                                explanation_method="exact_group_shapley")


def test_contribution_item_codes_are_instrument_codes():
    for bad in ("A0", "A26", "A", "3A", "a3"):
        with pytest.raises(ValidationError):
            FeatureContribution(item_code=bad, contribution_value=0.1,
                                direction="raises",
                                explanation_method="exact_group_shapley")


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------

def test_uncertainty_bounds():
    UncertaintyInfo(available=True, method="prior_sensitivity_band",
                    interval_low=0.4, interval_high=0.6)
    with pytest.raises(ValidationError):
        UncertaintyInfo(available=True, method="x", interval_low=0.6,
                        interval_high=0.4)
    with pytest.raises(ValidationError):
        UncertaintyInfo(available=True, method="x", interval_low=None,
                        interval_high=0.5)
    with pytest.raises(ValidationError):
        UncertaintyInfo(available=True, method=None, interval_low=0.1,
                        interval_high=0.5)
    UncertaintyInfo(available=False)


# ---------------------------------------------------------------------------
# assessments
# ---------------------------------------------------------------------------

def test_evidence_suggested_needs_its_items_and_evidence():
    a = _assessment()
    assert a.status is AssessmentStatus.EVIDENCE_SUGGESTED
    assert a.triggering_question_ids == ["A4"]
    with pytest.raises(ValidationError, match="must name"):
        _assessment(triggering_question_ids=[])
    with pytest.raises(ValidationError, match="must carry"):
        _assessment(supporting_evidence=[])


def test_assessment_trigger_ids_must_be_in_its_own_evidence():
    with pytest.raises(ValidationError, match="not in its own evidence"):
        _assessment(triggering_question_ids=["A4", "A9"])


@pytest.mark.parametrize("status", ["no_evidence", "not_measured"])
def test_other_statuses_carry_no_triggers(status):
    with pytest.raises(ValidationError, match="carries triggering"):
        _assessment(status=status, supporting_evidence=[],
                    triggering_question_ids=["A4"])


def test_assessment_rejects_unknown_item_codes():
    with pytest.raises(ValidationError, match="not an item"):
        _assessment(triggering_question_ids=["Q4"])


def test_no_evidence_assessment_is_valid_without_evidence():
    a = _assessment(status="no_evidence", supporting_evidence=[],
                    triggering_question_ids=[], assessment_method="observed_item_rule")
    assert a.status is AssessmentStatus.NO_EVIDENCE


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def test_recommendation_must_name_its_triggering_evidence():
    with pytest.raises(ValidationError, match="must name"):
        _recommendation(triggering_question_ids=[], triggering_responses=[])
    with pytest.raises(ValidationError, match="each triggering"):
        _recommendation(triggering_question_ids=["A4"], triggering_responses=[])
    with pytest.raises(ValidationError, match="came from"):
        _recommendation(recommendation_source="")


def test_recommendation_rejects_unknown_items_and_responses():
    with pytest.raises(ValidationError, match="not an item"):
        _recommendation(triggering_question_ids=["Q4"], triggering_responses=[1])
    with pytest.raises(ValidationError, match="binary"):
        _recommendation(triggering_question_ids=["A4"], triggering_responses=[3])


def test_recommendation_basis_is_observed_responses_only():
    r = _recommendation()
    assert r.basis is RecommendationBasis.OBSERVED_RESPONSE_PATTERN
    with pytest.raises(ValidationError):
        _recommendation(basis="user_confirmed")
    with pytest.raises(ValidationError):
        _recommendation(basis="general_guidance")


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

def test_feedback_rating_ranges():
    RecommendationFeedback(recommendation_id="rec-1", relevance_rating=5,
                           usefulness_rating=1, acceptability_rating=3)
    for field in ("relevance_rating", "acceptability_rating",
                  "usefulness_rating"):
        with pytest.raises(ValidationError):
            RecommendationFeedback(recommendation_id="rec-1",
                                   **{field: 6})
        with pytest.raises(ValidationError):
            RecommendationFeedback(recommendation_id="rec-1",
                                   **{field: 0})


# ---------------------------------------------------------------------------
# the assembled report
# ---------------------------------------------------------------------------

def test_report_validates_and_serialises():
    rep = _report()
    dumped = to_json_safe(rep)
    assert dumped["screening_result"]["p_hat"] == 0.58
    assert dumped["recommendations"][0]["triggering_question_ids"] == ["A4"]
    assert dumped["questionnaire_evidence"][0]["question_id"] == "A4"
    assert dumped["unassessed_areas"]
    rep2 = ScreeningReport.model_validate(dumped)
    assert rep2.report_id == rep.report_id


def test_explanation_must_match_its_screening_result():
    expl = _explanation().model_copy(update={"screening_result_id": "sr-OTHER"})
    with pytest.raises(ValidationError, match="must match"):
        _report(explanation=expl)


def test_recommendation_must_reference_known_assessment():
    with pytest.raises(ValidationError, match="unknown assessment"):
        _report(recommendations=[_recommendation(
            recommendation_id="rec-2",
            related_assessment_ids=["as-missing"])])


def test_unresolvable_evidence_reference_rejected():
    with pytest.raises(ValidationError, match="unresolvable"):
        _report(recommendations=[_recommendation(
            evidence_references=["nowhere"])])


def test_duplicate_assessment_ids_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _report(support_assessments=[_assessment(), _assessment()])


def test_duplicate_evidence_ids_rejected_within_an_assessment():
    """Ids are unique within an assessment. The same recorded answer may
    legitimately appear in two areas (A1 triggers both "getting attention" and
    "extra processing time"), so the report may not require cross-assessment
    uniqueness."""
    with pytest.raises(ValidationError, match="unique within"):
        _assessment(supporting_evidence=[_evidence("A4", 1),
                                         _evidence("A4", 1)])
    # ... and the same answer in two areas is a valid report
    shared = _evidence("A1", 1)
    a1 = _assessment(assessment_id="as-getting_attention",
                     domain="getting_attention",
                     supporting_evidence=[shared],
                     triggering_question_ids=["A1"])
    a2 = _assessment(assessment_id="as-processing_time",
                     domain="processing_time",
                     supporting_evidence=[shared],
                     triggering_question_ids=["A1"])
    rep = _report(support_assessments=[a1, a2],
                  questionnaire_evidence=[_question_evidence("A1", 1)],
                  recommendations=[])
    assert len(rep.support_assessments) == 2


def test_suggestion_without_recorded_evidence_fails_closed():
    """The central safety check: a recommendation the report cannot justify
    with a recorded answer is a validation error, not a dropped field."""
    with pytest.raises(ValidationError, match="not part of this report"):
        _report(questionnaire_evidence=[_question_evidence("A1", 0)],
                recommendations=[_recommendation(
                    triggering_question_ids=["A9"],
                    triggering_responses=[1])])


def test_report_records_unassessed_areas_and_evidence():
    rep = _report()
    assert rep.unassessed_areas == ["Sensory comfort and environment"]
    assert [e.question_id for e in rep.questionnaire_evidence] == ["A4"]
    assert rep.schema_version == "support-report/2.0"


def test_to_json_safe_handles_numpy():
    import numpy as np
    payload = {"a": np.float64(0.5), "b": np.int64(3), "c": np.bool_(True),
               "d": np.array([1.0, 2.0]), "e": [np.float64(1.5)]}
    safe = to_json_safe(payload)
    import json
    json.dumps(safe)          # must not raise
    assert safe == {"a": 0.5, "b": 3, "c": True, "d": [1.0, 2.0],
                    "e": [1.5]}
