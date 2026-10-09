"""Tests for the Pydantic support-layer contracts — P4.

Covers the invariants the design depends on: the observed/derived/user-reported
distinction, hypothesis vs confirmed needs, attribution direction matching the
sign of the contribution value, finite numerics, unknown-field rejection,
reference resolution inside the report, and JSON-safety. All fixtures are
synthetic — no real participant data anywhere.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.support.schemas import (
    AnswerValue, AssessmentStatus,
    DecisionEnum, EvidenceItem, ExplanationResult, FeatureContribution,
    FollowUpAnswers, RecommendationBasis, RecommendationFeedback,
    ScreeningReport, ScreeningResult, SupportNeedAssessment,
    SupportRecommendation, UncertaintyInfo, to_json_safe)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _screening(**kw):
    base = dict(result_id="sr-1", decision="REFER", p_hat=0.58,
                items_asked=["A1", "A2", "A3", "A4", "A6", "A8"],
                n_questions_asked=6, budget=6, model_name="MaskedPredictor",
                model_version="2", input_reference="episode:e-1")
    base.update(kw)
    return ScreeningResult(**base)


def _evidence(**kw):
    base = dict(evidence_id="ev-1", source_type="observed_response",
                item_code="A3", observed_value=1,
                source_reference="episode:trace[1]", description="hypothetical")
    base.update(kw)
    return EvidenceItem(**base)


def _assessment(**kw):
    base = dict(assessment_id="as-1", domain="social_communication",
                status="user_confirmed",
                assessment_method="user_report", user_confirmed=True,
                supporting_evidence=[_evidence()],
                followup_question_id="q3",
                limitations=["hypothetical"])
    base.update(kw)
    return SupportNeedAssessment(**base)


def _recommendation(**kw):
    base = dict(recommendation_id="rec-1", domain="social_communication",
                title="Offer extra processing time",
                description="hypothetical strategy",
                basis="user_confirmed", related_assessment_ids=["as-1"],
                why_selected="triggered by the confirmed assessment as-1",
                evidence_references=["guidance:example-source-1"])
    base.update(kw)
    return SupportRecommendation(**base)


# ---------------------------------------------------------------------------
# screening result
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
    """There is no 'autism probability' — the schema must not grow one."""
    fields = set(ScreeningResult.model_fields)
    for banned in ("autism_probability", "diagnosis", "clinical_probability",
                   "score"):
        assert banned not in fields


# ---------------------------------------------------------------------------
# evidence items
# ---------------------------------------------------------------------------

def test_observed_evidence_requires_code_and_value():
    with pytest.raises(ValidationError, match="item_code"):
        _evidence(item_code=None)
    with pytest.raises(ValidationError, match="observed_value"):
        _evidence(observed_value=None)


def test_item_code_format_and_range():
    _evidence(item_code="A10")
    _evidence(item_code="A25")
    for bad in ("A0", "A26", "A", "3A", "a3", "A3 "):
        with pytest.raises(ValidationError):
            _evidence(item_code=bad)


def test_user_reported_evidence_can_reference_a_question_not_an_item():
    e = _evidence(evidence_id="ev-2", source_type="user_reported",
                  item_code=None, observed_value=None,
                  source_reference="followup:q3")
    assert e.source_reference == "followup:q3"


def test_unknown_fields_rejected_everywhere():
    with pytest.raises(ValidationError):
        _evidence(surprise_field=1)
    with pytest.raises(ValidationError):
        _screening(extra="x")


# ---------------------------------------------------------------------------
# feature contributions
# ---------------------------------------------------------------------------

def test_direction_must_match_sign():
    FeatureContribution(item_code="A3", contribution_value=0.2,
                        direction="raises", explanation_method="exact_group_shapley")
    FeatureContribution(item_code="A3", contribution_value=-0.2,
                        direction="lowers", explanation_method="exact_group_shapley")
    FeatureContribution(item_code="A3", contribution_value=0.0,
                        direction="neutral", explanation_method="exact_group_shapley")
    with pytest.raises(ValidationError, match="raises"):
        FeatureContribution(item_code="A3", contribution_value=-0.1,
                            direction="raises",
                            explanation_method="exact_group_shapley")
    with pytest.raises(ValidationError, match="lowers"):
        FeatureContribution(item_code="A3", contribution_value=0.1,
                            direction="lowers",
                            explanation_method="exact_group_shapley")
    with pytest.raises(ValidationError, match="neutral"):
        FeatureContribution(item_code="A3", contribution_value=0.1,
                            direction="neutral",
                            explanation_method="exact_group_shapley")


def test_contribution_must_be_finite():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            FeatureContribution(item_code="A3", contribution_value=bad,
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
    # unavailable must carry no bounds
    UncertaintyInfo(available=False)


# ---------------------------------------------------------------------------
# support-need assessments: the hypothesis/confirmed invariant
# ---------------------------------------------------------------------------

def test_confirmed_requires_user_report():
    with pytest.raises(ValidationError, match="user report"):
        _assessment(assessment_method="observed_item_rule")


def test_hypothesis_cannot_be_user_confirmed():
    with pytest.raises(ValidationError, match="hypothesis"):
        _assessment(status="hypothesis_from_observed",
                    assessment_method="observed_item_rule",
                    user_confirmed=True)


def test_hypothesis_without_confirmation_is_legal():
    a = _assessment(status="hypothesis_from_observed",
                    assessment_method="observed_item_rule",
                    user_confirmed=False, followup_question_id="q3")
    assert a.user_confirmed is False


def test_insufficient_evidence_assessment_has_no_evidence():
    a = _assessment(status="insufficient_evidence",
                    assessment_method="observed_item_rule",
                    supporting_evidence=[], followup_question_id=None)
    assert a.status is AssessmentStatus.INSUFFICIENT_EVIDENCE


def test_declined_is_not_a_need():
    a = _assessment(status="user_declined", user_confirmed=False,
                    assessment_method="user_report")
    assert a.user_confirmed is False


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def test_need_based_recommendation_must_reference_its_assessment():
    with pytest.raises(ValidationError, match="triggered it"):
        _recommendation(related_assessment_ids=[])


def test_general_guidance_may_stand_alone():
    r = _recommendation(basis="general_guidance", related_assessment_ids=[])
    assert r.basis is RecommendationBasis.GENERAL_GUIDANCE


def test_hypothesis_opted_in_still_needs_the_triggering_assessment():
    with pytest.raises(ValidationError):
        _recommendation(basis="hypothesis_opted_in", related_assessment_ids=[])


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
# follow-up answers
# ---------------------------------------------------------------------------

def test_followup_abstentions_are_first_class():
    ans = FollowUpAnswers(answers=[
        {"question_id": "q1", "value": "yes"},
        {"question_id": "q2", "value": "prefer_not_to_answer"},
        {"question_id": "q3", "value": "unsure"},
        {"question_id": "q4", "value": "not_applicable"},
    ])
    assert ans.answered("q2") is AnswerValue.PREFER_NOT_TO_ANSWER
    assert ans.answered("q9") is None          # unanswered, not "no"
    with pytest.raises(ValidationError):
        FollowUpAnswers(answers=[{"question_id": "q1", "value": "yes"},
                                 {"question_id": "q1", "value": "no"}])


# ---------------------------------------------------------------------------
# the assembled report
# ---------------------------------------------------------------------------

def _explanation():
    return ExplanationResult(
        screening_result_id="sr-1",
        supporting_evidence=[_evidence(evidence_id="ev-1")],
        opposing_evidence=[_evidence(evidence_id="ev-2", item_code="A2",
                                     observed_value=0)],
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
        generation_status="complete",
    )


def _report(**kw):
    base = dict(report_id="rep-1", screening_result=_screening(),
                explanation=_explanation(), support_assessments=[_assessment()],
                recommendations=[_recommendation()],
                limitations=["hypothetical"],
                disclaimer="Screening support only. Not a diagnosis.")
    base.update(kw)
    return ScreeningReport(**base)


def test_report_validates_and_serialises():
    rep = _report()
    dumped = to_json_safe(rep)
    assert dumped["screening_result"]["p_hat"] == 0.58
    assert dumped["recommendations"][0]["basis"] == "user_confirmed"
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


def test_duplicate_evidence_ids_rejected():
    dup = _assessment(support_evidence=None) if False else _assessment(
        supporting_evidence=[_evidence(), _evidence()])
    with pytest.raises(ValidationError, match="duplicate evidence"):
        _report(support_assessments=[dup])


def test_to_json_safe_handles_numpy():
    import numpy as np
    payload = {"a": np.float64(0.5), "b": np.int64(3), "c": np.bool_(True),
               "d": np.array([1.0, 2.0]), "e": [np.float64(1.5)]}
    safe = to_json_safe(payload)
    import json
    json.dumps(safe)  # must not raise
    assert safe == {"a": 0.5, "b": 3, "c": True, "d": [1.0, 2.0],
                    "e": [1.5]}
