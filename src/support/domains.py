"""Candidate support areas and their evidence linkage — P4 (revised).

This module is *data, not logic*: each support area declares which
questionnaire items (if any) can legitimately justify offering it, and the
linkage is the only bridge from screening evidence to a suggestion. It is
one-way by construction:

    observed response  ->  an area where that kind of answer sometimes makes
    (screening data)       an optional idea useful  ->  the person decides
                                                                whether it does

Everything here is keyed on the **verified** questionnaire contract
(``src/data/qchat10_contract.py`` via ``src/support/questionnaire.py``). The
domain table that lived here before was written against a hand-copied list of
item wording that had drifted from the contract, so its links pointed at the
wrong constructs; the codes below are the corrected ones.

What the instrument measures, and what it does not
--------------------------------------------------
Q-CHAT-10 asks about looking when called, ease of eye contact, pointing to
request, pointing to share interest, pretending, following gaze, wanting to
comfort an upset person, first words, simple gestures, and staring at nothing.
It does **not** ask about sensory sensitivities, routines or transitions, daily
organisation, or how someone prefers information to be delivered. Those areas
are listed in :data:`UNASSESSED_AREAS` and are disclosed as unmeasured; no rule
in this project can turn an unrelated answer into a need in one of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.support.questionnaire import ITEM_BY_CODE


@dataclass(frozen=True)
class Domain:
    """A support area this project can offer suggestions in.

    ``trigger_item_codes`` is the *only* evidence that may offer this area, and
    every code in it is validated against the instrument at import time below.
    """
    domain_id: str
    label: str
    trigger_item_codes: Tuple[str, ...]
    recommendation_ids: Tuple[str, ...]
    #: Shown wherever this area appears, to frame it as an optional idea.
    framing_note: str


DOMAINS: Tuple[Domain, ...] = (
    Domain(
        domain_id="getting_message_across",
        label="Getting a message across",
        trigger_item_codes=("A3", "A4", "A8", "A9"),
        recommendation_ids=("rec_comm_visual_supports",),
        framing_note=(
            "Offered because answers about pointing, first words or simple "
            "gestures are the kind of answer this idea is sometimes useful "
            "for. Only you can say whether it fits your child."),
    ),
    Domain(
        domain_id="getting_attention",
        label="Getting attention before you speak",
        trigger_item_codes=("A1", "A2"),
        recommendation_ids=("rec_comm_attention_cues",),
        framing_note=(
            "Offered because answers about looking when called and ease of eye "
            "contact are the kind of answer this idea is sometimes useful for."),
    ),
    Domain(
        domain_id="processing_time",
        label="Extra processing time",
        trigger_item_codes=("A1", "A2", "A6"),
        recommendation_ids=("rec_comm_processing_time",),
        framing_note=(
            "Offered because answers about attention and following your gaze "
            "are the kind of answer this idea is sometimes useful for."),
    ),
    Domain(
        domain_id="naming_feelings",
        label="Noticing and naming feelings",
        trigger_item_codes=("A7",),
        recommendation_ids=("rec_emotion_name_what_you_see",),
        framing_note=(
            "Offered because the answer about comforting someone who is upset "
            "is the kind of answer this idea is sometimes useful for. It says "
            "nothing about how a child experiences feelings."),
    ),
)

DOMAINS_BY_ID = {d.domain_id: d for d in DOMAINS}


@dataclass(frozen=True)
class UnassessedArea:
    """A support area the questionnaire does not ask about.

    Present so the report can name these areas as unmeasured. A recommendation
    can never be derived from one: there is no evidence to derive it from.
    """
    area_id: str
    label: str
    reason: str


UNASSESSED_AREAS: Tuple[UnassessedArea, ...] = (
    UnassessedArea(area_id="sensory_environment",
                   label="Sensory comfort and environment",
                   reason="the questionnaire does not ask about sounds, "
                          "lights, textures or places"),
    UnassessedArea(area_id="predictability_transitions",
                   label="Predictability, changes and routines",
                   reason="the questionnaire does not ask about routines, "
                          "advance notice or transitions"),
    UnassessedArea(area_id="daily_living_organization",
                   label="Daily routines and organisation",
                   reason="the questionnaire does not ask about planning or "
                          "organising daily activities"),
    UnassessedArea(area_id="communication_accessibility",
                   label="How information reaches you",
                   reason="the questionnaire does not ask how someone prefers "
                          "information to be shared"),
)

#: Items the questionnaire asks that carry no curated, reviewed suggestion in
#: this project. Offered nothing, and disclosed rather than quietly dropped.
ASKED_WITHOUT_SUGGESTION: Tuple[str, ...] = ("A5", "A10")


def domain_for_item(item_code: str) -> Tuple[Domain, ...]:
    """Domains whose evidence linkage includes ``item_code`` (may be empty)."""
    return tuple(d for d in DOMAINS if item_code in d.trigger_item_codes)


def linked_item_codes() -> Tuple[str, ...]:
    """Every item code used by any domain's evidence linkage."""
    codes = {c for d in DOMAINS for c in d.trigger_item_codes}
    return tuple(sorted(codes))


def triggerable_item_codes() -> Tuple[str, ...]:
    """Items that can offer at least one suggestion (the rest cannot)."""
    return linked_item_codes()


# ---------------------------------------------------------------------------
# the registry must agree with the instrument, or nothing else can be trusted
# ---------------------------------------------------------------------------

def _validate_registry() -> None:
    for domain in DOMAINS:
        if not domain.trigger_item_codes:
            raise ValueError(f"domain {domain.domain_id} declares no trigger "
                             f"items, so it could never offer anything")
        if not domain.recommendation_ids:
            raise ValueError(f"domain {domain.domain_id} offers no suggestion")
        for code in domain.trigger_item_codes:
            if code not in ITEM_BY_CODE:
                raise ValueError(
                    f"domain {domain.domain_id} links item {code!r}, which the "
                    f"verified questionnaire contract does not define "
                    f"(known: A1..A10)")
    codes = [d.domain_id for d in DOMAINS]
    if len(codes) != len(set(codes)):
        raise ValueError("domain ids must be unique")
    area_ids = [a.area_id for a in UNASSESSED_AREAS]
    if len(area_ids) != len(set(area_ids)):
        raise ValueError("unassessed area ids must be unique")
    overlap = {d.domain_id for d in DOMAINS} & set(area_ids)
    if overlap:
        raise ValueError(f"an area cannot be both triggerable and unassessed: "
                         f"{sorted(overlap)}")
    for code in ASKED_WITHOUT_SUGGESTION:
        if code not in ITEM_BY_CODE:
            raise ValueError(f"ASKED_WITHOUT_SUGGESTION names {code!r}, which "
                             f"is not a questionnaire item")


_validate_registry()
