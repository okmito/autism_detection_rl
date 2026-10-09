"""The recommendation engine — P4.

Deterministic, auditable rules that connect *supported findings* and *user
statements* to optional support strategies. There are no trained components in
this module and no demographic input can reach it: rules read only

* the observed questionnaire responses (via the explanation's evidence), and
* the person's own follow-up answers.

The core safety rule, enforced structurally:

    **A hypothesis never fires a recommendation.**

Only ``user_confirmed`` needs produce need-based recommendations; a stated
preference produces general-guidance suggestions that keep the triggering
assessment reference. Abstention (``insufficient_evidence`` / ``unknown``)
produces an assessment and nothing else.

``RecommendationBasis.HYPOTHESIS_OPTED_IN`` exists in the schema but is never
emitted here, by construction: opting in to a hypothesis *is* answering "yes"
to its follow-up question, which is a user report and therefore the stronger
classification ``USER_CONFIRMED``. Nothing in this module is trained, and no
demographic field can reach it: rules read only observed responses (via
evidence items) and the person's own follow-up answers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.support.domains import DOMAINS, DOMAINS_BY_ID, Domain
from src.support.questions import QUESTIONS_BY_ID, FollowUpQuestion
from src.support.schemas import (
    AnswerValue, AssessmentMethod, AssessmentStatus, EvidenceItem,
    FollowUpAnswer, FollowUpAnswers, RecommendationBasis, SourceType,
    SupportNeedAssessment, SupportRecommendation)
from src.support.strategies import STRATEGIES_BY_ID

#: The Q-CHAT-10 binary projection used throughout this project:
#: ``1`` = atypical/concerning response (see README §15 and the contract).
ATYPICAL_VALUE = 1


# ---------------------------------------------------------------------------
# which follow-up questions to offer
# ---------------------------------------------------------------------------

def questions_to_offer(evidence: List[EvidenceItem]
                       ) -> Tuple[List[FollowUpQuestion], List[FollowUpQuestion]]:
    """Split the questionnaire into *suggested* and *optional* questions.

    A question is **suggested** only when its domain has screening-data linkage
    *and* at least one observed response in that domain was atypical — the sole
    permitted direction from screening data to a support question. Everything
    else (including every preference question and every domain with no
    linkage) is offered as an **optional** extra, and skipping any question is
    always valid.

    Returns ``(suggested, optional)``, both in questionnaire order.
    """
    suggested: List[FollowUpQuestion] = []
    optional: List[FollowUpQuestion] = []
    for q in QUESTIONS_BY_ID.values():
        domain = DOMAINS_BY_ID[q.domain_id]
        triggered = (domain.has_data_linkage and not q.is_preference_question
                     and bool(_observed_atypical_items(domain, evidence)))
        (suggested if triggered else optional).append(q)
    return suggested, optional


# ---------------------------------------------------------------------------
# answer validation
# ---------------------------------------------------------------------------

def validate_followup_answers(answers: FollowUpAnswers) -> List[str]:
    """Semantic validation of answers against the questionnaire definition.

    Returns a list of human-readable problems (empty when valid). The Pydantic
    layer guarantees types; this guarantees the answers mean what they claim.
    """
    problems: List[str] = []
    for a in answers.answers:
        q = QUESTIONS_BY_ID.get(a.question_id)
        if q is None:
            problems.append(f"unknown question id {a.question_id!r}")
            continue
        if q.is_preference_question:
            if a.choice is None:
                problems.append(
                    f"question {a.question_id!r} is a choice question and "
                    f"requires a choice value")
            elif a.choice not in q.choices:
                problems.append(
                    f"choice {a.choice!r} is not an option for question "
                    f"{a.question_id!r} (allowed: {list(q.choices)})")
            if a.value not in (AnswerValue.YES, AnswerValue.UNSURE,
                               AnswerValue.NOT_APPLICABLE,
                               AnswerValue.PREFER_NOT_TO_ANSWER):
                problems.append(
                    f"question {a.question_id!r} expects an answer value of "
                    f"yes/unsure/not_applicable/prefer_not_to_answer alongside "
                    f"any choice")
        elif a.choice is not None:
            problems.append(
                f"question {a.question_id!r} is not a choice question, so it "
                f"carries no choice value (got {a.choice!r})")
    return problems


def _answer_for(answers: FollowUpAnswers, question_id: str) -> FollowUpAnswer | None:
    for a in answers.answers:
        if a.question_id == question_id:
            return a
    return None


# ---------------------------------------------------------------------------
# assessments
# ---------------------------------------------------------------------------

def _observed_atypical_items(domain: Domain, evidence: List[EvidenceItem]
                             ) -> List[EvidenceItem]:
    """Evidence items for this domain's items whose observed response was the
    atypical/concerning value. Typical responses are not evidence of anything
    and never suggest a support question."""
    out = []
    for e in evidence:
        if (e.source_type is SourceType.OBSERVED_RESPONSE
                and e.item_code in domain.evidence_item_codes
                and e.observed_value == ATYPICAL_VALUE):
            out.append(e)
    return out


def _primary_question(domain: Domain) -> FollowUpQuestion | None:
    """The question that can establish a need for this domain (the first
    non-choice question)."""
    for qid in domain.followup_question_ids:
        q = QUESTIONS_BY_ID[qid]
        if not q.is_preference_question:
            return q
    return None


def build_assessments(evidence: List[EvidenceItem],
                      answers: FollowUpAnswers) -> List[SupportNeedAssessment]:
    """One assessment per domain, with an honest status for every outcome:
    confirmed / stated preference / declined / unknown / hypothesis / abstain.
    """
    assessments: List[SupportNeedAssessment] = []
    for domain in DOMAINS:
        q = _primary_question(domain)
        if q is None:
            continue
        a = _answer_for(answers, q.question_id)
        atypical = _observed_atypical_items(domain, evidence)
        domain_note = domain.neutral_framing_note

        if a is None:
            # No follow-up answer: the only thing screening data can produce is
            # an explicitly-labelled hypothesis, or an abstention.
            if atypical and domain.has_data_linkage:
                assessments.append(SupportNeedAssessment(
                    assessment_id=f"as-{domain.domain_id}",
                    domain=domain.domain_id,
                    status=AssessmentStatus.HYPOTHESIS_FROM_OBSERVED,
                    supporting_evidence=list(atypical),
                    assessment_method=AssessmentMethod.OBSERVED_ITEM_RULE,
                    user_confirmed=False,
                    followup_question_id=q.question_id,
                    limitations=[
                        "Suggested only by observed responses; no follow-up "
                        "answer was given, so nothing is established.",
                        domain_note,
                    ],
                ))
            else:
                assessments.append(SupportNeedAssessment(
                    assessment_id=f"as-{domain.domain_id}",
                    domain=domain.domain_id,
                    status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
                    supporting_evidence=[],
                    assessment_method=AssessmentMethod.OBSERVED_ITEM_RULE,
                    user_confirmed=False,
                    followup_question_id=None,
                    limitations=[
                        ("The screening questionnaire does not cover this "
                         "area and no follow-up answer was given; nothing can "
                         "be said about it." if not domain.has_data_linkage else
                         "No relevant observed responses and no follow-up "
                         "answer; nothing can be said about this area."),
                        domain_note,
                    ],
                ))
            continue

        # A follow-up answer exists: the person's own report decides.
        user_evidence = [EvidenceItem(
            evidence_id=f"ev-user-{q.question_id}",
            source_type=SourceType.USER_REPORTED,
            item_code=None,
            observed_value=None,
            source_reference=f"followup:{q.question_id}",
            description=(f"answer '{a.value.value}' to the follow-up question "
                         f"about {domain.label.lower()}"),
        )]

        if a.value is AnswerValue.YES:
            assessments.append(SupportNeedAssessment(
                assessment_id=f"as-{domain.domain_id}",
                domain=domain.domain_id,
                status=AssessmentStatus.USER_CONFIRMED,
                supporting_evidence=list(atypical) + user_evidence,
                assessment_method=AssessmentMethod.USER_REPORT,
                user_confirmed=True,
                followup_question_id=q.question_id,
                limitations=[
                    "Established by your own answer, not by the screening "
                    "result.",
                    domain_note,
                ],
            ))
        elif a.value is AnswerValue.NO:
            assessments.append(SupportNeedAssessment(
                assessment_id=f"as-{domain.domain_id}",
                domain=domain.domain_id,
                status=AssessmentStatus.USER_DECLINED,
                supporting_evidence=user_evidence,
                assessment_method=AssessmentMethod.USER_REPORT,
                user_confirmed=False,
                followup_question_id=q.question_id,
                limitations=[
                    "You indicated you do not want support with this right "
                    "now; nothing is offered.",
                    domain_note,
                ],
            ))
        else:
            abstained = a.value in (AnswerValue.PREFER_NOT_TO_ANSWER,
                                    AnswerValue.NOT_APPLICABLE,
                                    AnswerValue.UNSURE)
            assessments.append(SupportNeedAssessment(
                assessment_id=f"as-{domain.domain_id}",
                domain=domain.domain_id,
                status=AssessmentStatus.UNKNOWN,
                supporting_evidence=user_evidence,
                assessment_method=AssessmentMethod.USER_REPORT,
                user_confirmed=False,
                followup_question_id=q.question_id,
                limitations=[
                    (f"Your answer was '{a.value.value}'; nothing is inferred "
                     f"in either direction." if abstained else
                     "Your answer left this open; nothing is inferred."),
                    domain_note,
                ],
            ))

    assessments.extend(_preference_assessments(answers))
    return assessments


def _preference_assessments(answers: FollowUpAnswers
                            ) -> List[SupportNeedAssessment]:
    """Stated preferences are recorded as preferences, never as needs.

    A preference counts only when the person answered ``yes`` *and* named a
    choice. An abstention alongside a choice is recorded as ``unknown`` — the
    engine does not read a preference into a non-answer — and a domain whose
    only question is a preference question gets no assessment at all when it
    was not answered.
    """
    out: List[SupportNeedAssessment] = []
    for q in QUESTIONS_BY_ID.values():
        if not q.is_preference_question:
            continue
        a = _answer_for(answers, q.question_id)
        if a is None or a.choice is None:
            continue
        if a.value is AnswerValue.YES:
            status = AssessmentStatus.USER_STATED_PREFERENCE
            limitation = ("A preference about how information is shared; not a "
                          "support need and not a difficulty.")
        else:
            status = AssessmentStatus.UNKNOWN
            limitation = (f"Your answer was '{a.value.value}'; no preference is "
                          f"inferred in either direction.")
        out.append(SupportNeedAssessment(
            assessment_id=f"as-pref-{q.question_id}",
            domain=q.domain_id,
            status=status,
            supporting_evidence=[EvidenceItem(
                evidence_id=f"ev-user-{q.question_id}",
                source_type=SourceType.USER_REPORTED,
                item_code=None, observed_value=None,
                source_reference=f"followup:{q.question_id}",
                description=f"preference stated: {a.choice}",
            )],
            assessment_method=AssessmentMethod.USER_REPORT,
            user_confirmed=False,
            user_preference=(a.choice if status is
                             AssessmentStatus.USER_STATED_PREFERENCE else None),
            followup_question_id=q.question_id,
            limitations=[limitation],
        ))
    return out


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def build_recommendations(assessments: List[SupportNeedAssessment],
                          answers: FollowUpAnswers
                          ) -> List[SupportRecommendation]:
    """Recommendations fire only on user-confirmed needs and stated
    preferences. A hypothesis fires nothing; abstention fires nothing."""
    by_id = {a.assessment_id: a for a in assessments}
    recs: List[SupportRecommendation] = []

    def _why(text: str) -> str:
        return text

    for a in assessments:
        if a.status is AssessmentStatus.USER_CONFIRMED:
            q = QUESTIONS_BY_ID.get(a.followup_question_id or "")
            domain = DOMAINS_BY_ID[a.domain]
            for rid in domain.recommendation_ids:
                s = STRATEGIES_BY_ID[rid]
                recs.append(SupportRecommendation(
                    recommendation_id=f"rec-{a.assessment_id}-{rid}",
                    domain=s.domain,
                    title=s.title,
                    description=s.description,
                    intended_purpose=s.intended_purpose,
                    evidence_references=list(s.evidence_references) +
                                        [f"assessment:{a.assessment_id}"],
                    applicability_conditions=list(
                        s.applicability_conditions),
                    related_assessment_ids=[a.assessment_id],
                    basis=RecommendationBasis.USER_CONFIRMED,
                    why_selected=_why(
                        f"Offered because you said you would like support with "
                        f"{domain.label.lower()}"
                        + (f" (follow-up question {q.question_id})."
                           if q else ".")),
                    limitations=list(s.limitations) + [
                        "Optional: use it only if it fits your situation."],
                    source_attribution=s.source_attribution,
                    review_status=s.review_status,
                ))
        elif a.status is AssessmentStatus.USER_STATED_PREFERENCE:
            domain = DOMAINS_BY_ID[a.domain]
            for rid in domain.recommendation_ids:
                s = STRATEGIES_BY_ID[rid]
                recs.append(SupportRecommendation(
                    recommendation_id=f"rec-{a.assessment_id}-{rid}",
                    domain=s.domain,
                    title=s.title,
                    description=s.description,
                    intended_purpose=s.intended_purpose,
                    evidence_references=list(s.evidence_references) +
                                        [f"assessment:{a.assessment_id}"],
                    applicability_conditions=list(
                        s.applicability_conditions),
                    related_assessment_ids=[a.assessment_id],
                    # A preference is not a confirmed need, so this is general
                    # practice content *triggered by* the stated preference —
                    # not a response to a need. The assessment reference keeps
                    # the trigger auditable either way.
                    basis=RecommendationBasis.GENERAL_GUIDANCE,
                    why_selected=_why(
                        f"Offered because you told us you prefer "
                        f"'{a.user_preference}'; it changes how suggestions are "
                        f"shared with you."),
                    limitations=list(s.limitations),
                    source_attribution=s.source_attribution,
                    review_status=s.review_status,
                ))

    # de-duplicate identical (domain, strategy) offers, keeping the first
    seen: Dict[Tuple[str, str], SupportRecommendation] = {}
    for r in recs:
        key = (r.domain, r.title)
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def evaluate(evidence: List[EvidenceItem],
             answers: FollowUpAnswers) -> Tuple[List[SupportNeedAssessment],
                                                List[SupportRecommendation]]:
    """Full engine pass: assessments first, then recommendations that reference
    only assessments that exist."""
    assessments = build_assessments(evidence, answers)
    recommendations = build_recommendations(assessments, answers)
    # references must resolve (defence in depth; the report validator also
    # checks this, and a failure here is a bug in the registries).
    known = {a.assessment_id for a in assessments}
    for r in recommendations:
        for aid in r.related_assessment_ids:
            assert aid in known, f"engine emitted a dangling assessment ref {aid}"
    return assessments, recommendations
