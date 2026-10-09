"""Wording and provenance guardrails for the strategy library — P4 (revised).

The curated strategies ship as *unreviewed* data: nothing may claim approval a
human has not given, nothing may assert an external source that was not
verified, and the wording must stay optional, non-prescriptive and
non-deficit. A strategy that fails these tests cannot be offered.

The wording rule added with the revision: applicability conditions reference
the *observed response pattern* (the questionnaire never asks about a
preference, and there is no second form to answer).
"""
from __future__ import annotations

import re

from src.support.domains import DOMAINS
from src.support.questionnaire import ITEM_BY_CODE
from src.support.strategies import (DEPLOYMENT_REQUIRED_STATUS, GUIDANCE_SOURCES,
                                    NON_TRIGGERABLE_STRATEGIES,
                                    REVIEW_APPROVED, REVIEW_PENDING, STRATEGIES,
                                    STRATEGIES_BY_ID)

#: Wording that must never appear in user-facing support content.
BANNED_WORDS = (
    "suffers", "suffering", "deficit", "deficient", "abnormal", "broken",
    "cure", "treat", "treatment", "normalise", "normalize", "fix", "defect",
    "patient", "diagnos", "disorder", "severity", "impair",
)


def _banned_hits(text: str) -> list:
    """Banned deficit words as whole words."""
    hits = []
    lowered = text.lower()
    for banned in BANNED_WORDS:
        for suffix in ("", "s", "es", "ed", "ing", "is", "osis", "e"):
            if re.search(rf"\b{re.escape(banned + suffix)}\b", lowered):
                hits.append(banned)
                break
    return hits


def _all_strategies():
    return list(STRATEGIES) + list(NON_TRIGGERABLE_STRATEGIES)


def test_strategy_ids_unique_across_both_books():
    ids = [s.recommendation_id for s in _all_strategies()]
    assert len(ids) == len(set(ids))


def test_strategy_wording_guardrails():
    for s in _all_strategies():
        text = " ".join([s.title, s.description, s.intended_purpose]).lower()
        hits = _banned_hits(text)
        assert not hits, f"{s.recommendation_id}: {hits}"


def test_strategies_carry_provenance_review_and_limits():
    for s in _all_strategies():
        assert s.limitations, f"{s.recommendation_id} must state its limits"
        assert s.recommendation_source in GUIDANCE_SOURCES
        assert s.review_status == REVIEW_PENDING, (
            f"{s.recommendation_id} must not claim approval; a human sets it")
        assert not s.is_approved


def test_triggerable_strategies_state_when_they_apply():
    for s in STRATEGIES:
        assert s.trigger_item_codes, (
            f"{s.recommendation_id} is in the triggerable book but has no "
            f"linked items")
        assert s.applicability_conditions
        for code in s.trigger_item_codes:
            assert code in ITEM_BY_CODE
        # applicability must describe the observed pattern, not a questionnaire
        # that no longer exists
        blob = " ".join(s.applicability_conditions).lower()
        for phrase in ("you asked for", "you said", "you told us"):
            assert phrase not in blob, (s.recommendation_id, phrase)


def test_non_triggerable_strategies_explain_why_they_are_unreachable():
    for s in NON_TRIGGERABLE_STRATEGIES:
        assert s.no_link_reason
        assert s.recommendation_id not in STRATEGIES_BY_ID
        # and their area is not one the engine can offer anything in
        assert s.domain not in {d.domain_id for d in DOMAINS}, (
            f"{s.recommendation_id} is parked but {s.domain!r} is still a "
            f"triggerable area")
        # the reason must say what the questionnaire actually measures
        blob = s.no_link_reason.lower()
        assert "does not ask" in blob or "not ask" in blob


def test_nothing_ships_as_approved_without_human_review():
    for g in GUIDANCE_SOURCES.values():
        assert g.review_status == REVIEW_PENDING
    assert DEPLOYMENT_REQUIRED_STATUS == REVIEW_APPROVED
    assert not any(s.review_status == REVIEW_APPROVED
                   for s in _all_strategies())


def test_no_fabricated_external_urls():
    for g in GUIDANCE_SOURCES.values():
        blob = g.description + g.source_kind + g.reviewer_note
        assert "http://" not in blob and "https://" not in blob and "www." not in blob, (
            f"guidance {g.guidance_id} must not assert an external URL that has "
            f"not been verified by a human")


def test_every_domain_recommendation_exists_as_a_strategy():
    for d in DOMAINS:
        for rid in d.recommendation_ids:
            assert STRATEGIES_BY_ID[rid].domain == d.domain_id, (
                f"{rid} is declared by domain {d.domain_id} but belongs to "
                f"{STRATEGIES_BY_ID[rid].domain}")
