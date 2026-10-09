"""Registry integrity for the support layer — P4 (revised).

The domain registry and the strategy library are *data*, so they are tested
like data: cross-references must resolve, the evidence linkage must agree with
the verified instrument, and unmeasured areas must stay unmeasured. Wording
guardrails live next to the data they guard, in
``tests/test_support_strategies.py``; the instrument itself is locked in
``tests/test_support_questionnaire.py``.
"""
from __future__ import annotations

from src.support.domains import (ASKED_WITHOUT_SUGGESTION, DOMAINS,
                                 DOMAINS_BY_ID, UNASSESSED_AREAS,
                                 domain_for_item, linked_item_codes)
from src.support.questionnaire import ITEM_BY_CODE
from src.support.strategies import (GUIDANCE_SOURCES, NON_TRIGGERABLE_STRATEGIES,
                                    STRATEGIES, STRATEGIES_BY_ID)


# ---------------------------------------------------------------------------
# domains
# ---------------------------------------------------------------------------

def test_domain_ids_unique():
    ids = [d.domain_id for d in DOMAINS]
    assert len(ids) == len(set(ids))
    assert len(DOMAINS) == 4, "the four areas Q-CHAT-10 can honestly suggest"


def test_domain_cross_references_resolve():
    for d in DOMAINS:
        assert d.recommendation_ids, f"{d.domain_id} offers no suggestion"
        for rid in d.recommendation_ids:
            assert rid in STRATEGIES_BY_ID, f"{d.domain_id} -> unknown strategy {rid}"
            assert STRATEGIES_BY_ID[rid].domain == d.domain_id
        for code in d.trigger_item_codes:
            assert code in ITEM_BY_CODE, (
                f"{d.domain_id} links item {code!r}, which the verified "
                f"questionnaire contract does not define")


def test_domain_item_codes_are_qchat10_items():
    for code in linked_item_codes():
        assert code in ITEM_BY_CODE
        assert 1 <= int(code[1:]) <= 10
    # the instrument has ten items; every one either suggests something, is
    # disclosed as asked-without-suggestion, or does not exist
    assert set(linked_item_codes()) | set(ASKED_WITHOUT_SUGGESTION) == \
        set(ITEM_BY_CODE)


def test_ask_without_suggestion_items_are_real_and_unlinked():
    for code in ASKED_WITHOUT_SUGGESTION:
        assert code in ITEM_BY_CODE
        assert domain_for_item(code) == ()


def test_every_linked_item_offers_something():
    for code in linked_item_codes():
        assert domain_for_item(code), code


def test_domain_for_item():
    assert {d.domain_id for d in domain_for_item("A1")} == {"getting_attention",
                                                            "processing_time"}
    assert {d.domain_id for d in domain_for_item("A7")} == {"naming_feelings"}
    assert domain_for_item("A5") == ()
    assert domain_for_item("A10") == ()


# ---------------------------------------------------------------------------
# unmeasured areas
# ---------------------------------------------------------------------------

def test_unassessed_areas_have_reasons_and_never_overlap_domains():
    assert len(UNASSESSED_AREAS) == 4
    for area in UNASSESSED_AREAS:
        assert area.reason
        assert area.area_id not in DOMAINS_BY_ID
    labels = [a.label for a in UNASSESSED_AREAS]
    assert len(labels) == len(set(labels))


def test_the_four_areas_the_questionnaire_does_not_ask_about():
    """The specific areas the design says must never be inferred."""
    areas = {a.area_id for a in UNASSESSED_AREAS}
    assert areas == {"sensory_environment", "predictability_transitions",
                     "daily_living_organization", "communication_accessibility"}


def test_no_domain_shares_an_item_with_an_unassessed_area():
    unmeasured_words = ("sensory", "routines", "organisation", "preference")
    for d in DOMAINS:
        for w in unmeasured_words:
            assert w not in d.domain_id, (d.domain_id, w)
