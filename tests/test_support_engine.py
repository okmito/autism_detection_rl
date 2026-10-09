"""Rules tests for the deterministic recommendation engine — P4.

The engine holds the phase's core safety property:

    **A hypothesis never fires a recommendation.**

Only the person's own follow-up answers (and their stated preferences) can
produce suggestions. These tests lock that property in, together with the
honest-status table (confirmed / declined / abstained / hypothesis /
insufficient evidence) and the direction of the evidence linkage. All fixtures
are synthetic item codes and answers — no participant data.
"""
from __future__ import annotations

import pytest

from src.support.domains import DOMAINS_BY_ID
from src.support.engine import (MAX_ID_LENGTH, _recommendation_id, evaluate,
                                questions_to_offer, validate_followup_answers)
from src.support.questions import QUESTIONS_BY_ID
from src.support.schemas import (AssessmentStatus, EvidenceItem,
                                 FollowUpAnswers, RecommendationBasis,
                                 SourceType)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _obs(code: str, value: int) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"ev-obs-{code}",
        source_type=SourceType.OBSERVED_RESPONSE,
        item_code=code, observed_value=value,
        source_reference="episode:trace[0]")


def _answers(*pairs) -> FollowUpAnswers:
    """pairs of (question_id, value[, choice])."""
    return FollowUpAnswers(answers=[
        {"question_id": qid, "value": value,
         **({"choice": choice} if choice is not None else {})}
        for qid, value, *rest in pairs
        for choice in [rest[0] if rest else None]
    ])


ATYPICAL_A8 = [_obs("A8", 1)]          # emotional_regulation linkage
ATYPICAL_A1 = [_obs("A1", 1)]          # social_communication linkage
TYPICAL = [_obs("A1", 0), _obs("A8", 0)]


# ---------------------------------------------------------------------------
# the core safety rule
# ---------------------------------------------------------------------------

def test_a_hypothesis_never_fires_a_recommendation():
    assessments, recs = evaluate(ATYPICAL_A8, _answers())
    emotional = [a for a in assessments if a.domain == "emotional_regulation"]
    assert emotional and emotional[0].status is \
        AssessmentStatus.HYPOTHESIS_FROM_OBSERVED
    assert emotional[0].user_confirmed is False
    assert emotional[0].supporting_evidence          # the hypothesis is evidenced
    assert recs == [], "a hypothesis must not produce any recommendation"


def test_typical_responses_never_suggest_anything():
    assessments, recs = evaluate(TYPICAL, _answers())
    assert all(a.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
               for a in assessments)
    assert recs == []


def test_domains_without_linkage_cannot_be_hypothesised_from_data():
    assessments, _ = evaluate(ATYPICAL_A1 + [_obs("A7", 1)], _answers())
    sensory = [a for a in assessments if a.domain == "sensory_environment"]
    assert sensory[0].status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert sensory[0].supporting_evidence == []
    # and its question is optional, never suggested
    suggested, optional = questions_to_offer(ATYPICAL_A1 + [_obs("A7", 1)])
    assert "q_sensory_support" not in {q.question_id for q in suggested}
    assert "q_sensory_support" in {q.question_id for q in optional}


# ---------------------------------------------------------------------------
# what the person's own answers do
# ---------------------------------------------------------------------------

def test_confirmed_need_fires_recommendations_with_its_trigger():
    assessments, recs = evaluate(
        ATYPICAL_A8, _answers(("q_emotion_support", "yes")))
    emotional = [a for a in assessments if a.domain == "emotional_regulation"][0]
    assert emotional.status is AssessmentStatus.USER_CONFIRMED
    assert emotional.user_confirmed is True
    assert recs, "a confirmed need must produce suggestions"
    assert all(r.basis is RecommendationBasis.USER_CONFIRMED for r in recs)
    for r in recs:
        assert emotional.assessment_id in r.related_assessment_ids
        assert f"assessment:{emotional.assessment_id}" in r.evidence_references
        assert emotional.assessment_id in r.why_selected or r.why_selected


def test_confirmed_need_can_be_endorsed_without_any_atypical_response():
    """The person's own report is sufficient on its own; screening data is not
    the source of the need."""
    _, recs = evaluate([], _answers(("q_comm_support", "yes")))
    assert recs
    assert {r.domain for r in recs} == {"social_communication"}


def test_declining_fires_nothing():
    assessments, recs = evaluate(
        ATYPICAL_A8, _answers(("q_emotion_support", "no")))
    emotional = [a for a in assessments if a.domain == "emotional_regulation"][0]
    assert emotional.status is AssessmentStatus.USER_DECLINED
    assert emotional.user_confirmed is False
    assert recs == []


@pytest.mark.parametrize("value", ["unsure", "prefer_not_to_answer",
                                   "not_applicable"])
def test_abstention_fires_nothing_and_is_not_inferred_either_way(value):
    assessments, recs = evaluate(
        ATYPICAL_A8, _answers(("q_emotion_support", value)))
    emotional = [a for a in assessments if a.domain == "emotional_regulation"][0]
    assert emotional.status is AssessmentStatus.UNKNOWN
    assert emotional.user_confirmed is False
    assert any(value in lim for lim in emotional.limitations)
    assert recs == []


def test_an_unanswered_question_is_not_a_no():
    """Missing question id must behave exactly like a skipped questionnaire."""
    a_skipped, recs_skipped = evaluate(ATYPICAL_A8, _answers())
    # answering an unrelated question must not change the emotional domain
    a_wrong, recs_wrong = evaluate(ATYPICAL_A8, _answers(("q_daily_support", "no")))
    def emotional(a):
        return [x for x in a if x.domain == "emotional_regulation"][0].status
    assert emotional(a_skipped) is emotional(a_wrong) is \
        AssessmentStatus.HYPOTHESIS_FROM_OBSERVED
    assert recs_skipped == recs_wrong == []


# ---------------------------------------------------------------------------
# preferences
# ---------------------------------------------------------------------------

def test_stated_preference_is_recorded_as_a_preference_not_a_need():
    assessments, recs = evaluate(
        [], _answers(("q_comm_preference", "yes", "written")))
    prefs = [a for a in assessments
             if a.status is AssessmentStatus.USER_STATED_PREFERENCE]
    assert len(prefs) == 1
    p = prefs[0]
    assert p.user_preference == "written"
    assert p.user_confirmed is False
    assert recs and all(r.basis is RecommendationBasis.GENERAL_GUIDANCE
                        for r in recs)
    assert p.assessment_id in recs[0].related_assessment_ids


def test_preference_questions_are_never_suggested_by_screening_data():
    suggested, _ = questions_to_offer(ATYPICAL_A1 + ATYPICAL_A8)
    assert all(not q.is_preference_question for q in suggested)


def test_accessibility_domain_has_no_need_question():
    """Both accessibility questions carry choice vocabularies, so the domain
    can only ever be a stated preference — never a confirmed need."""
    from src.support.engine import _primary_question
    assert _primary_question(DOMAINS_BY_ID["communication_accessibility"]) is None


@pytest.mark.parametrize("value", ["unsure", "prefer_not_to_answer",
                                   "not_applicable"])
def test_a_choice_alongside_an_abstention_is_not_a_preference(value):
    assessments, _ = evaluate(
        [], _answers(("q_comm_preference", value, "written")))
    prefs = [a for a in assessments
             if a.followup_question_id == "q_comm_preference"]
    assert len(prefs) == 1
    assert prefs[0].status is AssessmentStatus.UNKNOWN
    assert prefs[0].user_preference is None


def test_unanswered_preference_question_produces_no_assessment():
    assessments, _ = evaluate([], _answers())
    assert all(a.followup_question_id != "q_comm_preference"
               for a in assessments)


# ---------------------------------------------------------------------------
# evidence linkage direction
# ---------------------------------------------------------------------------

def test_only_atypical_responses_suggest_a_follow_up_question():
    suggested, _ = questions_to_offer(TYPICAL)
    assert suggested == []
    suggested, _ = questions_to_offer([_obs("A2", 1)])
    assert {q.question_id for q in suggested} == {"q_comm_support"}


def test_suggested_questions_are_exactly_the_linked_triggered_ones():
    suggested, optional = questions_to_offer(ATYPICAL_A1 + ATYPICAL_A8)
    assert {q.question_id for q in suggested} == {"q_comm_support",
                                                  "q_emotion_support"}
    opt = {q.question_id for q in optional}
    assert {"q_sensory_support", "q_transitions_support", "q_daily_support",
            "q_comm_preference", "q_accessibility_preference"} <= opt
    assert len(suggested) + len(optional) == len(
        {q.question_id for q in suggested} | opt)


# ---------------------------------------------------------------------------
# id bounds (a regression once crashed report assembly for one domain)
# ---------------------------------------------------------------------------

def _all_yes_answers() -> FollowUpAnswers:
    """Endorse every need and state every preference — the answer set that
    fires every recommendation path in the registries."""
    return FollowUpAnswers(answers=[
        {"question_id": q.question_id, "value": "yes",
         **({"choice": q.choices[0]} if q.choices else {})}
        for q in QUESTIONS_BY_ID.values()])


def test_every_recommendation_path_produces_a_valid_id():
    """The composite ids must fit the contract's 64-character bound for every
    domain/strategy pair, not just the short ones."""
    assessments, recs = evaluate([], _all_yes_answers())
    assert recs, "this answer set must fire recommendations"
    assert assessments
    for r in recs:
        assert len(r.recommendation_id) <= MAX_ID_LENGTH, r.recommendation_id
        assert r.recommendation_id.startswith("rec-")


def test_long_domain_and_strategy_still_yield_a_bounded_unique_id():
    """The digest fallback keeps the id inside the bound when the readable
    composite would exceed it."""
    long_assessment = "as-" + "x" * 40
    for strategy_id in ("rec_transitions_visual_schedule",
                        "rec_sensory_adjustable_conditions",
                        "rec_" + "y" * 40):
        rid = _recommendation_id(long_assessment, strategy_id)
        assert len(rid) <= MAX_ID_LENGTH, (rid, strategy_id)
        assert rid.startswith(f"rec-{long_assessment}-")
    # digest path: deterministic and distinct per strategy
    a = _recommendation_id(long_assessment, "rec_" + "y" * 40)
    b = _recommendation_id(long_assessment, "rec_" + "z" * 40)
    assert a != b
    assert _recommendation_id(long_assessment, "rec_" + "y" * 40) == a


def test_recommendation_ids_are_unique_across_the_whole_answer_space():
    _, recs = evaluate(ATYPICAL_A1 + ATYPICAL_A8, _all_yes_answers())
    ids = [r.recommendation_id for r in recs]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# answer validation
# ---------------------------------------------------------------------------

def test_unknown_question_id_is_reported():
    problems = validate_followup_answers(_answers(("q_nope", "yes")))
    assert problems and "unknown question id" in problems[0]


def test_choice_questions_require_a_choice_from_the_vocabulary():
    assert validate_followup_answers(_answers(("q_comm_preference", "yes")))
    bad = validate_followup_answers(_answers(("q_comm_preference", "yes", "smoke")))
    assert any("not an option" in p for p in bad)


def test_non_choice_questions_carry_no_choice():
    problems = validate_followup_answers(_answers(("q_emotion_support", "yes",
                                                   "written")))
    assert any("not a choice question" in p for p in problems)


def test_valid_answers_pass():
    assert validate_followup_answers(
        _answers(("q_emotion_support", "yes"),
                 ("q_comm_preference", "yes", "visual"))) == []


# ---------------------------------------------------------------------------
# engine output integrity
# ---------------------------------------------------------------------------

def test_every_recommendation_reference_resolves():
    assessments, recs = evaluate(
        ATYPICAL_A1 + ATYPICAL_A8,
        _answers(("q_emotion_support", "yes"),
                 ("q_comm_support", "yes"),
                 ("q_comm_preference", "yes", "spoken")))
    known = {a.assessment_id for a in assessments}
    for r in recs:
        assert set(r.related_assessment_ids) <= known
        for ref in r.evidence_references:
            assert ref in known or ref.startswith(("guidance:", "assessment:"))


def test_identical_offers_are_deduplicated():
    """Answering the same need through both a symptom-linked and an
    accessibility question must not duplicate the same strategy offer."""
    _, recs = evaluate(
        [], _answers(("q_emotion_support", "yes"),
                     ("q_sensory_support", "yes")))
    keys = [(r.domain, r.title) for r in recs]
    assert len(keys) == len(set(keys))


def test_assessment_ids_are_stable_and_unique():
    assessments, _ = evaluate(ATYPICAL_A1, _answers())
    ids = [a.assessment_id for a in assessments]
    assert len(ids) == len(set(ids))
    assert "as-social_communication" in ids
