"""Wording guardrails for the follow-up questionnaire — P4.

The questionnaire is *data*, reviewed like data: every item must be about the
person's own experience and what they would find useful, must never repeat a
screening item, must not assert a difficulty, and must disclose when an area is
outside what the screening questions cover. A question that fails these tests
cannot be offered.
"""
from __future__ import annotations

import re

from src.support.domains import DOMAINS_BY_ID
from src.support.questions import QUESTIONS, QUESTIONS_BY_ID

#: Wording that must never appear in user-facing support content.
BANNED_WORDS = (
    "suffers", "suffering", "deficit", "deficient", "abnormal", "broken",
    "cure", "treat", "treatment", "normalise", "normalize", "fix", "defect",
    "patient", "diagnos", "disorder", "severity", "impair",
)

#: Phrases the screening instrument itself uses; a follow-up item must not
#: re-ask one of them (the design keeps the two questionnaires separate).
SCREENING_QUESTIONS = [
    "look at you when you call", "easily get eye contact",
    "look at you and pay attention", "point to ask for something",
    "point to share interest", "engage in pretend play",
    "follow where you are looking", "try to comfort them",
    "first words typical", "simple gestures typical",
]


def _banned_hits(text: str) -> list:
    """Banned deficit words as whole words (so 'fixed place' is fine, 'fix'
    alone and 'fixes' are not)."""
    hits = []
    lowered = text.lower()
    for banned in BANNED_WORDS:
        for suffix in ("", "s", "es", "ed", "ing", "is", "osis", "e"):
            if re.search(rf"\b{re.escape(banned + suffix)}\b", lowered):
                hits.append(banned)
                break
    return hits


def test_question_ids_unique_and_domains_exist():
    ids = [q.question_id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in QUESTIONS:
        assert q.domain_id in DOMAINS_BY_ID, f"{q.question_id} -> unknown domain"


def test_every_question_is_a_real_question():
    for q in QUESTIONS:
        assert "?" in q.text, f"{q.question_id} is not phrased as a question"
        assert not q.text.lower().startswith("do you agree"), (
            f"{q.question_id} must not be a leading question")


def test_questions_do_not_repeat_screening_items():
    for q in QUESTIONS:
        lowered = q.text.lower()
        for s in SCREENING_QUESTIONS:
            assert s not in lowered, f"{q.question_id} repeats screening wording"


def test_question_wording_guardrails():
    for q in QUESTIONS:
        text = (q.text + " " + q.helper_text).lower()
        assert not _banned_hits(text), f"{q.question_id}: {_banned_hits(text)}"


def test_unlinked_domains_disclose_that_screening_does_not_cover_them():
    """A question whose domain has no screening-data linkage must say so, so a
    user is never led to believe the screening result raised it. Preference
    questions disclose the same thing from the other side: they are about
    preferences, not about any difficulty."""
    for q in QUESTIONS:
        domain = DOMAINS_BY_ID[q.domain_id]
        if domain.has_data_linkage:
            continue
        blob = (q.text + " " + q.helper_text + " " +
                domain.neutral_framing_note).lower()
        not_covered = ("does not ask" in blob or "not covered" in blob
                       or "does not cover" in blob)
        preference_framing = ("preference" in blob or "prefer" in blob)
        assert not_covered or preference_framing, (
            f"{q.question_id} sits in domain {q.domain_id} which has no "
            f"screening-data linkage, but neither the question nor the domain "
            f"note discloses that")


def test_linked_domain_questions_frame_the_link_as_exploration():
    for q in QUESTIONS:
        domain = DOMAINS_BY_ID[q.domain_id]
        if not domain.has_data_linkage:
            continue
        blob = (q.text + " " + q.helper_text).lower()
        # the one question that leans on the link must not assert a difficulty
        if q.question_id == "q_emotion_support":
            assert "suggested" in blob and "you decide" in blob, (
                "a data-linked question must frame the link as something the "
                "user decides about, never as a finding")


def test_choice_questions_declare_their_vocabulary():
    choice_qs = [q for q in QUESTIONS if q.choices]
    assert choice_qs, "the preference question should exist"
    for q in choice_qs:
        assert q.choices and len(q.choices) >= 2
        assert len(q.choices) == len(set(q.choices))
    # exactly the accessibility domain carries choice questions
    for q in choice_qs:
        assert q.domain_id == "communication_accessibility"


def test_every_domain_has_a_question_that_can_establish_a_need():
    """Each domain's primary (non-choice) question is the only one that can
    establish a need; a domain with only preference questions cannot."""
    from src.support.engine import _primary_question
    for domain in DOMAINS_BY_ID.values():
        if domain.domain_id == "communication_accessibility":
            continue          # preference-only by design
        q = _primary_question(domain)
        assert q is not None, f"domain {domain.domain_id} has no need question"
        assert QUESTIONS_BY_ID[q.question_id].domain_id == domain.domain_id
