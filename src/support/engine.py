"""The recommendation engine — P4 (revised: no second questionnaire).

Deterministic, auditable rules that connect recorded screening answers to
optional support suggestions. There are no trained components in this module
and no demographic input can reach it: rules read only the observed
questionnaire responses.

The core safety rule, enforced structurally:

    A suggestion may only be offered because of an answer that was actually
    given, and it must name the items and the responses that triggered it.

What this engine deliberately does **not** do
---------------------------------------------
* It does not establish a need. ``EVIDENCE_SUGGESTED`` means "an optional idea
  is offered because that kind of answer sometimes makes it useful", nothing
  more. The person decides whether it fits.
* It does not infer anything the questionnaire never measured. Areas in
  ``src/support/domains.py::UNASSESSED_AREAS`` (sensory, routines and
  transitions, daily organisation, information-format preference) are disclosed
  as unmeasured and can never produce a suggestion.
* It does not score, rank or weight suggestions by severity, and it produces
  no probability of any kind.
* It does not personalise content with the model: the same answers always give
  the same suggestions.

Determinism and bounds: ids are built from the assessment and strategy
registries and are capped at the contract's 64 characters
(``src/support/engine.py::_recommendation_id``); a digest fallback keeps them
unique and bounded if a future registry entry would run long.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from src.support.domains import (ASKED_WITHOUT_SUGGESTION, DOMAINS, DOMAINS_BY_ID,
                                 Domain, UNASSESSED_AREAS, UnassessedArea)
from src.support.questionnaire import (ATYPICAL_VALUE, ITEM_BY_CODE,
                                       ITEM_CODES, QuestionnaireItem)
from src.support.schemas import (AssessmentMethod, AssessmentStatus,
                                 EvidenceItem, RecommendationBasis, SourceType,
                                 SupportNeedAssessment, SupportRecommendation)
from src.support.strategies import STRATEGIES, STRATEGIES_BY_ID, SupportStrategy

#: Ids in this layer are bounded by the contracts in ``src.support.schemas``
#: (``recommendation_id``: 64 characters).
MAX_ID_LENGTH = 64


def _recommendation_id(assessment_id: str, strategy_id: str) -> str:
    """Composite recommendation id, bounded and deterministic.

    The strategy's ``rec_`` registry prefix is redundant inside the composite
    (the id already starts with ``rec-``); dropping it buys the length the
    longest domain/strategy pairs need, and a digest fallback keeps the id
    inside the bound if a future registry entry would run over.
    """
    stem = (strategy_id[len("rec_"):] if strategy_id.startswith("rec_")
            else strategy_id)
    rid = f"rec-{assessment_id}-{stem}"
    if len(rid) <= MAX_ID_LENGTH:
        return rid
    digest = hashlib.sha1(strategy_id.encode("utf-8")).hexdigest()[:8]
    return f"rec-{assessment_id}-{digest}"


# ---------------------------------------------------------------------------
# observed evidence
# ---------------------------------------------------------------------------

def _is_atypical(e: EvidenceItem) -> bool:
    return (e.source_type is SourceType.OBSERVED_RESPONSE
            and e.observed_value == ATYPICAL_VALUE)


def _atypical_for(evidence: List[EvidenceItem],
                  item_codes: Tuple[str, ...]) -> List[EvidenceItem]:
    """Observed atypical responses among ``item_codes``, in instrument order."""
    want = set(item_codes)
    return [e for e in evidence
            if _is_atypical(e) and e.item_code in want]


def _asked_codes(evidence: List[EvidenceItem]) -> List[str]:
    """Item codes that were actually asked, in instrument order."""
    observed = {e.item_code for e in evidence
                if e.source_type is SourceType.OBSERVED_RESPONSE}
    return [c for c in ITEM_CODES if c in observed]


# ---------------------------------------------------------------------------
# assessments
# ---------------------------------------------------------------------------

def build_assessments(evidence: List[EvidenceItem]
                      ) -> List[SupportNeedAssessment]:
    """One assessment per triggerable support area, plus one per unmeasured area.

    * a triggerable area with at least one atypical linked answer is
      ``EVIDENCE_SUGGESTED`` and carries those answers;
    * a triggerable area whose linked items were all answered in the typical
      direction is ``NO_EVIDENCE``;
    * a triggerable area whose items were not asked is ``NO_EVIDENCE`` with a
      limitation saying so, rather than being treated as an absence of need;
    * an unmeasured area is ``NOT_MEASURED``, always, with the reason.
    """
    assessments: List[SupportNeedAssessment] = []

    for domain in DOMAINS:
        atypical = _atypical_for(evidence, domain.trigger_item_codes)
        asked = _asked_codes(evidence)
        linked_asked = [c for c in domain.trigger_item_codes if c in asked]
        if atypical:
            status = AssessmentStatus.EVIDENCE_SUGGESTED
            limitations = [
                "An optional idea is offered because of the recorded answer(s) "
                "named above. This is not a finding that a difficulty exists.",
                domain.framing_note,
            ]
        elif linked_asked:
            status = AssessmentStatus.NO_EVIDENCE
            limitations = [
                "Answers about this area were all recorded in the typical "
                "direction, so nothing is offered.",
                domain.framing_note,
            ]
        else:
            status = AssessmentStatus.NO_EVIDENCE
            limitations = [
                "This area was not reached in this session, so nothing is "
                "offered and nothing is inferred in either direction.",
                domain.framing_note,
            ]
        assessments.append(SupportNeedAssessment(
            assessment_id=f"as-{domain.domain_id}",
            domain=domain.domain_id,
            label=domain.label,
            status=status,
            supporting_evidence=list(atypical),
            triggering_question_ids=[e.item_code for e in atypical],
            assessment_method=AssessmentMethod.OBSERVED_ITEM_RULE,
            limitations=limitations,
        ))

    for area in UNASSESSED_AREAS:
        assessments.append(SupportNeedAssessment(
            assessment_id=f"as-unassessed-{area.area_id}",
            domain=area.area_id,
            label=area.label,
            status=AssessmentStatus.NOT_MEASURED,
            supporting_evidence=[],
            triggering_question_ids=[],
            assessment_method=AssessmentMethod.OBSERVED_ITEM_RULE,
            limitations=[f"{area.reason}; nothing can be said about this area "
                         f"from this questionnaire, in either direction."],
        ))

    return assessments


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def _why_selected(domain: Domain, strategy: SupportStrategy,
                  triggers: List[EvidenceItem]) -> str:
    """Plain-language reason naming the item codes and the recorded answers."""
    parts = [f"{e.item_code} ({e.observed_value and 'atypical' or 'typical'})"
             for e in triggers]
    items = ", ".join(parts)
    label = domain.label[0].lower() + domain.label[1:]
    return (f"Offered because your answer to {items} was recorded, and this "
            f"idea is sometimes useful for that kind of answer. Optional; it is "
            f"not a finding about your child.")


def build_recommendations(assessments: List[SupportNeedAssessment],
                          evidence: List[EvidenceItem]
                          ) -> List[SupportRecommendation]:
    """Suggestions fire only from recorded atypical answers.

    Each strategy declares its own trigger items, so a strategy fires when its
    items — not merely its domain's — were answered atypically, and it carries
    exactly those item codes and their recorded values.
    """
    recs: List[SupportRecommendation] = []
    for a in assessments:
        if a.status is not AssessmentStatus.EVIDENCE_SUGGESTED:
            continue
        domain = DOMAINS_BY_ID[a.domain]
        for rid in domain.recommendation_ids:
            s = STRATEGIES_BY_ID[rid]
            triggers = _atypical_for(evidence, s.trigger_item_codes)
            if not triggers:
                # Domain-level evidence exists, but this strategy's own items
                # were not the ones answered atypically. Do not fire it.
                continue
            recs.append(SupportRecommendation(
                recommendation_id=_recommendation_id(a.assessment_id,
                                                     s.recommendation_id),
                domain=s.domain,
                title=s.title,
                description=s.description,
                intended_purpose=s.intended_purpose,
                triggering_question_ids=[e.item_code for e in triggers],
                triggering_responses=[int(e.observed_value) for e in triggers],
                evidence_references=[s.recommendation_source],
                applicability_conditions=list(s.applicability_conditions),
                related_assessment_ids=[a.assessment_id],
                basis=RecommendationBasis.OBSERVED_RESPONSE_PATTERN,
                recommendation_source=s.recommendation_source,
                why_selected=_why_selected(domain, s, triggers),
                limitations=list(s.limitations) + [
                    domain.framing_note,
                    "Optional: use it only if it fits your situation."],
                source_attribution=strategy_source_attribution(s),
                review_status=s.review_status,
            ))

    # de-duplicate identical (domain, title) offers, keeping the first
    seen: Dict[Tuple[str, str], SupportRecommendation] = {}
    for r in recs:
        key = (r.domain, r.title)
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def strategy_source_attribution(strategy: SupportStrategy) -> str:
    """Human-readable provenance for a triggerable strategy."""
    return ("curated project content (pending expert review)")


def evaluate(evidence: List[EvidenceItem]
             ) -> Tuple[List[SupportNeedAssessment],
                        List[SupportRecommendation]]:
    """Full engine pass: assessments first, then suggestions that reference
    only assessments that exist."""
    assessments = build_assessments(evidence)
    recommendations = build_recommendations(assessments, evidence)
    # references must resolve (defence in depth; the report validator also
    # checks this, and a failure here is a bug in the registries).
    known = {a.assessment_id for a in assessments}
    for r in recommendations:
        for aid in r.related_assessment_ids:
            assert aid in known, f"engine emitted a dangling assessment ref {aid}"
        for code in r.triggering_question_ids:
            assert code in ITEM_BY_CODE, (
                f"engine emitted a trigger id {code!r} that is not a "
                f"questionnaire item")
    return assessments, recommendations


# ---------------------------------------------------------------------------
# report content the engine owns
# ---------------------------------------------------------------------------

def unassessed_area_labels() -> List[str]:
    """Labels of the areas the questionnaire does not ask about."""
    return [f"{a.label} — {a.reason}" for a in UNASSESSED_AREAS]


def asked_without_suggestion_codes(evidence: List[EvidenceItem]) -> List[str]:
    """Of the items asked that carry no suggestion, which were asked here."""
    asked = set(_asked_codes(evidence))
    return [c for c in ASKED_WITHOUT_SUGGESTION if c in asked]
