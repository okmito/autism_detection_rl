"""Candidate support domains and their evidence linkage — P4.

This module is *data, not logic*: each domain declares which questionnaire
items (if any) can legitimately justify **asking** about it. The linkage is the
only bridge from screening evidence to a support question, and it is one-way
by design:

    observed response  ->  "may be worth exploring" question  ->  the person's
                                                                 own answer
    (screening data)       (never an assertion of difficulty)   (the only thing
                                                                 that establishes
                                                                 a need)

Domains whose ``evidence_item_codes`` are empty have **no screening-data
linkage**: they can only ever be user-reported. A test enforces that they never
fire on their own.

Item codes refer to the Q-CHAT-10 projection used across this project
(``src/data/qchat10_contract.py``); the wording the demo shows lives in
``scripts/demo_app.py::ITEMS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Domain:
    domain_id: str
    label: str
    #: Item codes whose observation may justify ASKING about this domain.
    #: Empty means no screening-data linkage exists.
    evidence_item_codes: Tuple[str, ...]
    #: Follow-up questions that can establish (or rule out) a need here.
    followup_question_ids: Tuple[str, ...]
    #: Curated strategies that may be offered for this domain.
    recommendation_ids: Tuple[str, ...]
    #: Shown wherever this domain appears, to frame it as exploration.
    neutral_framing_note: str

    @property
    def has_data_linkage(self) -> bool:
        return bool(self.evidence_item_codes)


DOMAINS: Tuple[Domain, ...] = (
    Domain(
        domain_id="social_communication",
        label="Everyday communication",
        evidence_item_codes=("A1", "A2", "A3", "A4", "A5", "A7", "A10"),
        followup_question_ids=("q_comm_support",),
        recommendation_ids=("rec_comm_processing_time", "rec_comm_visual_supports",
                            "rec_comm_attention_cues"),
        neutral_framing_note=(
            "Answers like these can be worth exploring together. They do not "
            "say anything about a person on their own."),
    ),
    Domain(
        domain_id="emotional_regulation",
        label="Big feelings and stressful moments",
        evidence_item_codes=("A8",),
        followup_question_ids=("q_emotion_support",),
        recommendation_ids=("rec_emotion_predictable_scripts",
                            "rec_emotion_quiet_break_option"),
        neutral_framing_note=(
            "Responses to others' distress are one small piece of a much bigger "
            "picture."),
    ),
    Domain(
        domain_id="sensory_environment",
        label="Sensory comfort and environment",
        evidence_item_codes=(),          # no Q-CHAT-10 item covers this
        followup_question_ids=("q_sensory_support",),
        recommendation_ids=("rec_sensory_quieter_space",
                            "rec_sensory_adjustable_conditions"),
        neutral_framing_note=(
            "The screening questionnaire does not ask about sensory "
            "experiences, so this area can only come from what you tell us."),
    ),
    Domain(
        domain_id="predictability_transitions",
        label="Predictability, changes and routines",
        evidence_item_codes=(),          # no Q-CHAT-10 item covers this
        followup_question_ids=("q_transitions_support",),
        recommendation_ids=("rec_transitions_advance_notice",
                            "rec_transitions_visual_schedule"),
        neutral_framing_note=(
            "Not covered by the screening questions; only your own report can "
            "raise it."),
    ),
    Domain(
        domain_id="daily_living_organization",
        label="Daily routines and organisation",
        evidence_item_codes=(),          # no Q-CHAT-10 item covers this
        followup_question_ids=("q_daily_support",),
        recommendation_ids=("rec_daily_optional_tools",),
        neutral_framing_note=(
            "Not covered by the screening questions; only your own report can "
            "raise it."),
    ),
    Domain(
        domain_id="communication_accessibility",
        label="How information reaches you",
        evidence_item_codes=(),          # a preference, not a screening finding
        followup_question_ids=("q_comm_preference", "q_accessibility_preference"),
        recommendation_ids=("rec_comm_preference_choice",),
        neutral_framing_note=(
            "This is about your preferences, not about any difficulty."),
    ),
)

DOMAINS_BY_ID = {d.domain_id: d for d in DOMAINS}


def domain_for_item(item_code: str) -> Tuple[Domain, ...]:
    """Domains whose evidence linkage includes ``item_code`` (may be empty)."""
    return tuple(d for d in DOMAINS if item_code in d.evidence_item_codes)


def linked_item_codes() -> Tuple[str, ...]:
    """Every item code used by any domain's evidence linkage."""
    codes = {c for d in DOMAINS for c in d.evidence_item_codes}
    return tuple(sorted(codes))
