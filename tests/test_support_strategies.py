"""Wording and provenance guardrails for the strategy library — P4.

The curated strategies ship as *unreviewed* data: nothing may claim approval a
human has not given, nothing may assert an external source that was not
verified, and the wording must stay optional, non-prescriptive and
non-deficit. A strategy that fails these tests cannot be offered.
"""
from __future__ import annotations

import re

from src.support.domains import DOMAINS_BY_ID
from src.support.strategies import (DEPLOYMENT_REQUIRED_STATUS, GUIDANCE_SOURCES,
                                    REVIEW_APPROVED, REVIEW_PENDING,
                                    STRATEGIES, STRATEGIES_BY_ID)

#: Wording that must never appear in user-facing support content.
BANNED_WORDS = (
    "suffers", "suffering", "deficit", "deficient", "abnormal", "broken",
    "cure", "treat", "treatment", "normalise", "normalize", "fix", "defect",
    "patient", "diagnos", "disorder", "severity", "impair",
)


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


def test_strategy_ids_unique():
    ids = [s.recommendation_id for s in STRATEGIES]
    assert len(ids) == len(set(ids))


def test_strategy_wording_guardrails():
    for s in STRATEGIES:
        text = " ".join([s.title, s.description, s.intended_purpose,
                         s.source_attribution]).lower()
        hits = _banned_hits(text)
        assert not hits, f"{s.recommendation_id}: {hits}"


def test_strategies_carry_applicability_and_limitations():
    for s in STRATEGIES:
        assert s.applicability_conditions, (
            f"{s.recommendation_id} must state when it applies")
        assert s.limitations, f"{s.recommendation_id} must state its limits"
        assert s.evidence_references, f"{s.recommendation_id} needs provenance"
        assert "pending" in s.source_attribution.lower(), (
            f"{s.recommendation_id} must disclose its unreviewed status")


def test_strategy_domains_and_guidance_resolve():
    for s in STRATEGIES:
        assert s.domain in DOMAINS_BY_ID, f"{s.recommendation_id} -> unknown domain"
        for ref in s.evidence_references:
            assert ref in GUIDANCE_SOURCES, (
                f"{s.recommendation_id} references unknown guidance {ref!r}")


def test_every_domain_recommendation_exists_as_a_strategy():
    for d in DOMAINS_BY_ID.values():
        for rid in d.recommendation_ids:
            assert STRATEGIES_BY_ID[rid].domain == d.domain_id, (
                f"{rid} is declared by domain {d.domain_id} but belongs to "
                f"{STRATEGIES_BY_ID[rid].domain}")


def test_nothing_ships_as_approved_without_human_review():
    for s in STRATEGIES:
        assert s.review_status == REVIEW_PENDING, (
            f"{s.recommendation_id} must not claim approval; a human sets it")
        assert not s.is_approved
    for g in GUIDANCE_SOURCES.values():
        assert g.review_status == REVIEW_PENDING
    # the deployment gate is the approved state, and nothing currently meets it
    assert DEPLOYMENT_REQUIRED_STATUS == REVIEW_APPROVED
    assert not any(s.review_status == REVIEW_APPROVED for s in STRATEGIES)


def test_no_fabricated_external_urls():
    for g in GUIDANCE_SOURCES.values():
        blob = g.description + g.source_kind + g.reviewer_note
        assert "http://" not in blob and "https://" not in blob and "www." not in blob, (
            f"guidance {g.guidance_id} must not assert an external URL that has "
            f"not been verified by a human")
