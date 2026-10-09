"""Registry integrity for the support layer — P4.

The domain registry and the strategy library are *data*, so they are tested
like data: cross-references must resolve and evidence linkage must be honest.
Wording guardrails live next to the data they guard, in
``tests/test_support_questions.py`` and ``tests/test_support_strategies.py``.
"""
from __future__ import annotations

import re

from src.support.domains import DOMAINS, domain_for_item, linked_item_codes
from src.support.questions import QUESTIONS_BY_ID, questions_with_data_linkage
from src.support.strategies import STRATEGIES_BY_ID

ITEM_CODE_RE = re.compile(r"^A(?:[1-9]|[1-9][0-9])$")


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------

def test_domain_ids_unique():
    ids = [d.domain_id for d in DOMAINS]
    assert len(ids) == len(set(ids))
    assert len(DOMAINS) >= 6, "candidate domains from the design should exist"


def test_domain_cross_references_resolve():
    for d in DOMAINS:
        for qid in d.followup_question_ids:
            assert qid in QUESTIONS_BY_ID, f"{d.domain_id} -> unknown question {qid}"
        for rid in d.recommendation_ids:
            assert rid in STRATEGIES_BY_ID, f"{d.domain_id} -> unknown strategy {rid}"


def test_domain_item_codes_are_valid_instrument_codes():
    for d in DOMAINS:
        for code in d.evidence_item_codes:
            assert ITEM_CODE_RE.match(code), f"{d.domain_id}: bad code {code!r}"
            assert 1 <= int(code[1:]) <= 25
    # every linked code is a real Q-CHAT-10 item (A1..A10)
    for code in linked_item_codes():
        assert 1 <= int(code[1:]) <= 10, f"{code} is not a Q-CHAT-10 item"


def test_domains_without_linkage_can_never_fire_from_data():
    unlinked = [d for d in DOMAINS if not d.has_data_linkage]
    assert unlinked, "the design requires some user-reported-only domains"
    for d in unlinked:
        for code in linked_item_codes():
            assert d not in domain_for_item(code), (
                f"domain {d.domain_id} declares no linkage yet responds to "
                f"item {code}")
    # and their questions are not data-linked
    for d in unlinked:
        assert questions_with_data_linkage(d.domain_id) == (), (
            f"domain {d.domain_id} has no data linkage, so none of its "
            f"questions may be suggested by observed responses")


def test_linked_domain_questions_are_linked():
    for d in DOMAINS:
        if d.has_data_linkage:
            qs = questions_with_data_linkage(d.domain_id)
            assert qs, f"domain {d.domain_id} has data linkage but no linked question"
            # every linked question belongs to this domain
            for q in qs:
                assert q.domain_id == d.domain_id


def test_domain_for_item():
    assert "social_communication" in {d.domain_id for d in domain_for_item("A1")}
    assert "emotional_regulation" in {d.domain_id for d in domain_for_item("A8")}
    assert domain_for_item("A9") == () or all(
        d.evidence_item_codes for d in domain_for_item("A9"))
