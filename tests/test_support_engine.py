"""Rules tests for the deterministic suggestion engine — P4 (revised).

The engine's safety property, restated for the no-second-questionnaire design:

    A suggestion may only be offered because of an answer that was actually
    given, and it must name the items and the responses that triggered it.

These tests lock that property, the honest-status table, the unmeasured areas,
and the direction of the evidence linkage. All fixtures are synthetic item
codes and answers — no participant data.
"""
from __future__ import annotations

import pytest

from src.support.domains import (ASKED_WITHOUT_SUGGESTION, DOMAINS,
                                 DOMAINS_BY_ID, UNASSESSED_AREAS,
                                 domain_for_item, linked_item_codes)
from src.support.engine import (MAX_ID_LENGTH, _recommendation_id,
                                asked_without_suggestion_codes, evaluate,
                                unassessed_area_labels)
from src.support.questionnaire import ITEM_BY_CODE
from src.support.schemas import (AssessmentStatus, EvidenceItem,
                                 RecommendationBasis, SourceType)
from src.support.strategies import (NON_TRIGGERABLE_BY_ID, NON_TRIGGERABLE_STRATEGIES,
                                    STRATEGIES, STRATEGIES_BY_ID)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _obs(code: str, value: int) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"ev-obs-{code}",
        source_type=SourceType.OBSERVED_RESPONSE,
        item_code=code, observed_value=value,
        source_reference="episode:trace[0]")


def _evidence(*pairs) -> list:
    """Build evidence from (item_code, response) pairs."""
    return [_obs(code, value) for code, value in pairs]


ATYPICAL_A1 = [("A1", 1)]
ATYPICAL_A7 = [("A7", 1)]
TYPICAL = [("A1", 0), ("A7", 0), ("A3", 0)]


# ---------------------------------------------------------------------------
# the safety property
# ---------------------------------------------------------------------------

def test_a_suggestion_always_names_its_recorded_evidence():
    assessments, recs = evaluate(_evidence(*ATYPICAL_A1, *ATYPICAL_A7))
    assert recs
    observed = {e.item_code: e.observed_value for e in
               [x for a in assessments for x in a.supporting_evidence]}
    for r in recs:
        assert r.triggering_question_ids, r.recommendation_id
        assert r.triggering_responses
        for code, value in zip(r.triggering_question_ids,
                               r.triggering_responses):
            assert code in observed, (r.recommendation_id, code)
            assert observed[code] == value
            assert value == 1, "only an atypical answer may trigger a suggestion"


def test_a_typical_answer_never_triggers_anything():
    assessments, recs = evaluate(_evidence(*TYPICAL))
    assert recs == []
    assert all(a.status is AssessmentStatus.NO_EVIDENCE
               for a in assessments
               if a.domain in {d.domain_id for d in DOMAINS})


def test_no_answers_at_all_is_a_valid_outcome():
    assessments, recs = evaluate([])
    assert recs == []
    statuses = {a.domain: a.status for a in assessments}
    assert statuses["getting_message_across"] is AssessmentStatus.NO_EVIDENCE
    assert all(a.limitations for a in assessments)


def test_the_same_answers_always_give_the_same_suggestions():
    ev = _evidence(*ATYPICAL_A1, *ATYPICAL_A7)
    first = evaluate(ev)
    for _ in range(3):
        assert evaluate(ev) == first


# ---------------------------------------------------------------------------
# the honest-status table
# ---------------------------------------------------------------------------

def test_statuses_match_the_evidence_they_have():
    assessments, _ = evaluate(_evidence(*ATYPICAL_A1, *ATYPICAL_A7))
    by_domain = {a.domain: a for a in assessments}
    assert by_domain["getting_attention"].status is \
        AssessmentStatus.EVIDENCE_SUGGESTED
    assert by_domain["processing_time"].status is \
        AssessmentStatus.EVIDENCE_SUGGESTED       # A1 is one of its triggers
    assert by_domain["getting_message_across"].status is \
        AssessmentStatus.NO_EVIDENCE               # A3/A4/A8/A9 not answered
    assert by_domain["naming_feelings"].status is AssessmentStatus.EVIDENCE_SUGGESTED


def test_unmeasured_areas_are_always_not_measured():
    assessments, _ = evaluate(_evidence(*ATYPICAL_A1))
    by_domain = {a.domain: a for a in assessments}
    for area in UNASSESSED_AREAS:
        a = by_domain[area.area_id]
        assert a.status is AssessmentStatus.NOT_MEASURED
        assert a.supporting_evidence == []
        assert a.triggering_question_ids == []
        assert any(area.reason in lim for lim in a.limitations)


def test_unasked_items_are_not_treated_as_an_absence_of_need():
    assessments, _ = evaluate(_evidence(*ATYPICAL_A1))
    by_domain = {a.domain: a for a in assessments}
    msg = " ".join(by_domain["getting_message_across"].limitations)
    assert "not reached" in msg


def test_items_asked_without_any_suggestion_are_disclosed():
    ev = _evidence(("A5", 1), ("A10", 1))
    assessments, recs = evaluate(ev)
    assert asked_without_suggestion_codes(ev) == ["A5", "A10"]
    # they are asked and atypical, yet trigger nothing at all
    assert recs == []
    assert not any(e.item_code in ASKED_WITHOUT_SUGGESTION
                   for a in assessments for e in a.supporting_evidence)


# ---------------------------------------------------------------------------
# evidence linkage direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", sorted(linked_item_codes()))
def test_a_linked_item_offers_its_area(code):
    assessments, recs = evaluate(_evidence((code, 1)))
    domains = {a.domain for a in assessments
               if a.status is AssessmentStatus.EVIDENCE_SUGGESTED}
    assert domains == {d.domain_id for d in domain_for_item(code)}
    assert recs, f"an atypical answer to {code} should offer something"


def test_unlinked_items_offer_nothing_even_when_atypical():
    unlinked = [c for c in ITEM_BY_CODE if c not in linked_item_codes()]
    assert set(unlinked) == set(ASKED_WITHOUT_SUGGESTION)
    for code in unlinked:
        _, recs = evaluate(_evidence((code, 1)))
        assert recs == [], f"{code} has no curated suggestion but fired one"


def test_unassessed_areas_can_never_be_fired():
    """No recorded answer may produce a suggestion in an area the
    questionnaire does not ask about."""
    import itertools
    codes = sorted(ITEM_BY_CODE)
    for combo in itertools.combinations(codes, 3):
        _, recs = evaluate(_evidence(*[(c, 1) for c in combo]))
        offered = {r.domain for r in recs}
        assert offered <= {d.domain_id for d in DOMAINS}
        assert not offered & {a.area_id for a in UNASSESSED_AREAS}


def test_every_strategy_fires_only_on_its_own_items():
    for strategy in STRATEGIES:
        # its own items atypical -> it fires
        _, recs = evaluate(_evidence(
            *[(c, 1) for c in strategy.trigger_item_codes]))
        assert any(r.title == strategy.title for r in recs), strategy.recommendation_id
        # other items atypical -> it does not
        others = [c for c in sorted(ITEM_BY_CODE)
                  if c not in strategy.trigger_item_codes]
        _, recs = evaluate(_evidence(*[(c, 1) for c in others[:4]]))
        assert all(r.title != strategy.title for r in recs), strategy.recommendation_id


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------

def test_recommendation_ids_fit_the_contract_bound():
    ev = _evidence(*[(c, 1) for c in sorted(ITEM_BY_CODE)])
    _, recs = evaluate(ev)
    assert recs
    for r in recs:
        assert len(r.recommendation_id) <= MAX_ID_LENGTH
        assert r.recommendation_id.startswith("rec-")
    ids = [r.recommendation_id for r in recs]
    assert len(ids) == len(set(ids))


def test_id_helper_is_bounded_and_deterministic():
    long_assessment = "as-" + "x" * 40
    a = _recommendation_id(long_assessment, "rec_" + "y" * 40)
    b = _recommendation_id(long_assessment, "rec_" + "z" * 40)
    assert len(a) <= MAX_ID_LENGTH and len(b) <= MAX_ID_LENGTH
    assert a != b
    assert a == _recommendation_id(long_assessment, "rec_" + "y" * 40)


# ---------------------------------------------------------------------------
# report content the engine owns
# ---------------------------------------------------------------------------

def test_unassessed_labels_name_the_areas_and_the_reason():
    labels = unassessed_area_labels()
    assert len(labels) == len(UNASSESSED_AREAS)
    for area, label in zip(UNASSESSED_AREAS, labels):
        assert area.label in label and area.reason in label


def test_non_triggerable_strategies_are_reachable_nowhere():
    """They stay in the library as reviewed-as data, but no rule can fire one."""
    for s in NON_TRIGGERABLE_STRATEGIES:
        assert s.recommendation_id not in STRATEGIES_BY_ID
        assert s.no_link_reason
        assert s.review_status == "pending_expert_review"
    # exhaustive: no answer pattern produces one
    import itertools
    codes = sorted(ITEM_BY_CODE)
    for combo in itertools.chain.from_iterable(
            itertools.combinations(codes, r) for r in range(0, 5)):
        _, recs = evaluate(_evidence(*[(c, 1) for c in combo]))
        for r in recs:
            assert r.title not in {s.title for s in NON_TRIGGERABLE_STRATEGIES}


def test_basis_is_observed_response_pattern():
    _, recs = evaluate(_evidence(*ATYPICAL_A1))
    assert recs
    assert all(r.basis is RecommendationBasis.OBSERVED_RESPONSE_PATTERN
               for r in recs)
    assert all(r.recommendation_source.startswith("guidance:")
               for r in recs)
    assert all(r.review_status == "pending_expert_review" for r in recs)


def test_registry_agrees_with_the_instrument():
    for domain in DOMAINS:
        for code in domain.trigger_item_codes:
            assert code in ITEM_BY_CODE
        for rid in domain.recommendation_ids:
            assert rid in STRATEGIES_BY_ID
            assert STRATEGIES_BY_ID[rid].domain == domain.domain_id
    for s in STRATEGIES:
        for code in s.trigger_item_codes:
            assert code in ITEM_BY_CODE
