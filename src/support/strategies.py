"""Curated support strategies with provenance — P4 (revised).

Every strategy is *data* with an auditable basis:

* ``trigger_item_codes`` — the questionnaire items that may trigger it. An
  **empty tuple means the strategy has no Q-CHAT-10 linkage**: the
  questionnaire never asks about what the suggestion addresses, so it can never
  be offered. Those entries stay in the library (they are reviewed as data) but
  they are structurally unreachable — see ``NON_TRIGGERABLE_STRATEGIES`` and
  the test that asserts so.
* ``recommendation_source`` — the provenance key the recommendation reports;
  resolves against :data:`GUIDANCE_SOURCES`. No external URL is asserted, so no
  citation can be fabricated by accident.
* ``review_status`` — nothing here ships as ``approved`` until a human reviews
  it. The library ships entirely as ``pending_expert_review``, and the report
  says so.

Wording guardrails (enforced in ``tests/test_support_strategies.py``): optional,
non-prescriptive, non-deficit; no "treatment", "cure", "normal", "fix"; no claim
about any individual.

Wording rule introduced with the P4 revision: applicability conditions no longer
reference a follow-up questionnaire (there is none). They describe the *observed
response pattern* that triggers the suggestion, in the instrument's own terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.support.questionnaire import ITEM_BY_CODE

#: Review states for curated content and guidance sources.
REVIEW_PENDING = "pending_expert_review"
REVIEW_APPROVED = "approved"

#: The review status required before this content may support any claim beyond
# a research demonstration. It gates *deployment claims*, not demo emission:
# the engine may emit pending-review strategies for demonstration, but the
# assembled report must disclose the pending status (see src/support/report.py).
DEPLOYMENT_REQUIRED_STATUS = REVIEW_APPROVED


@dataclass(frozen=True)
class GuidanceSource:
    guidance_id: str
    source_kind: str                 # what kind of source this is
    description: str
    review_status: str = REVIEW_PENDING
    #: Set when a human has reviewed it (free text for the audit trail).
    reviewer_note: str = ""


GUIDANCE_SOURCES = {
    "guidance:project-curated-v1": GuidanceSource(
        guidance_id="guidance:project-curated-v1",
        source_kind="curated_project_content",
        description=("Support strategies drafted by the project team from "
                     "general accessibility and communication-practice "
                     "principles. Not clinically evaluated. Awaiting expert "
                     "review before any use beyond research demonstration."),
        review_status=REVIEW_PENDING,
    ),
    "guidance:accessibility-practice-v1": GuidanceSource(
        guidance_id="guidance:accessibility-practice-v1",
        source_kind="accessibility_practice",
        description=("General accessibility and information-format practices "
                     "(offering choice of format, extra processing time). "
                     "Awaiting expert review."),
        review_status=REVIEW_PENDING,
    ),
    "guidance:person-centred-practice-v1": GuidanceSource(
        guidance_id="guidance:person-centred-practice-v1",
        source_kind="person_centred_practice",
        description=("Person-centred practice principles: ask the person, "
                     "offer optional strategies, respect stated preferences. "
                     "Awaiting expert review."),
        review_status=REVIEW_PENDING,
    ),
}


@dataclass(frozen=True)
class SupportStrategy:
    recommendation_id: str
    domain: str
    title: str
    description: str
    intended_purpose: str
    #: The questionnaire items that may trigger this suggestion. Empty means the
    #: questionnaire does not measure what this suggestion addresses.
    trigger_item_codes: Tuple[str, ...]
    applicability_conditions: Tuple[str, ...]
    limitations: Tuple[str, ...]
    recommendation_source: str
    review_status: str = REVIEW_PENDING

    @property
    def is_triggerable(self) -> bool:
        return bool(self.trigger_item_codes)

    @property
    def is_approved(self) -> bool:
        return self.review_status == REVIEW_APPROVED


# ---------------------------------------------------------------------------
# triggerable strategies — every one of these is linked to items the
# questionnaire actually asks, and only fires on an atypical answer to one of
# them.
# ---------------------------------------------------------------------------

STRATEGIES: Tuple[SupportStrategy, ...] = (
    SupportStrategy(
        recommendation_id="rec_comm_visual_supports",
        domain="getting_message_across",
        title="Pair words with something to see",
        description=("Add a gesture, a picture, the real object or a "
                     "demonstration alongside your words when it helps — for "
                     "example holding up the coat while saying it is time to go "
                     "out. Two routes for the same message often beat one."),
        intended_purpose="Give the message more than one way to land",
        trigger_item_codes=("A3", "A4", "A8", "A9"),
        applicability_conditions=(
            "Answers about pointing to ask for something, pointing to share "
            "interest, first words or simple gestures were recorded in the "
            "atypical direction",),
        limitations=("A general strategy, not tailored to any individual",
                     "An answer about early communication says nothing "
                     "definite about what a child needs next"),
        recommendation_source="guidance:project-curated-v1",
    ),
    SupportStrategy(
        recommendation_id="rec_comm_attention_cues",
        domain="getting_attention",
        title="Get attention before you speak",
        description=("Say your child's name and wait for them to look or turn "
                     "before starting. If that does not work, move into their "
                     "line of sight rather than repeating louder."),
        intended_purpose="Make it easier to know when something is being shared",
        trigger_item_codes=("A1", "A2"),
        applicability_conditions=(
            "Answers about looking when their name is called, or about how easy "
            "eye contact is, were recorded in the atypical direction",),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:project-curated-v1",
    ),
    SupportStrategy(
        recommendation_id="rec_comm_processing_time",
        domain="processing_time",
        title="Offer extra processing time",
        description=("When giving instructions or asking a question, pause "
                     "longer than usual before repeating or rephrasing. "
                     "Counting slowly to ten in your head is enough."),
        intended_purpose="Reduce time pressure in everyday exchanges",
        trigger_item_codes=("A1", "A2", "A6"),
        applicability_conditions=(
            "Answers about attention or following your gaze were recorded in "
            "the atypical direction",),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:accessibility-practice-v1",
    ),
    SupportStrategy(
        recommendation_id="rec_emotion_name_what_you_see",
        domain="naming_feelings",
        title="Name what you see, out loud",
        description=("When someone is upset, say plainly what you notice — "
                     "'Sam is crying; he looks sad' — and keep it short. "
                     "Predictable, low-key wording about feelings gives a child "
                     "something concrete to hold on to, whether or not they "
                     "respond to it."),
        intended_purpose="Make other people's feelings easier to notice and name",
        trigger_item_codes=("A7",),
        applicability_conditions=(
            "The answer about showing signs of wanting to comfort someone who "
            "is visibly upset was recorded in the atypical direction",),
        limitations=(
            "A general strategy, not tailored to any individual",
            "One screening answer about responding to another person's "
            "distress says nothing definite about how a child experiences "
            "their own feelings"),
        recommendation_source="guidance:person-centred-practice-v1",
    ),
)

STRATEGIES_BY_ID = {s.recommendation_id: s for s in STRATEGIES}

# ---------------------------------------------------------------------------
# non-triggerable strategies — the questionnaire does not measure what these
# address, so no recorded answer can justify offering them. They are kept as
# reviewed-as data with an explicit reason, and a test asserts the engine can
# never emit one.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonTriggerableStrategy:
    recommendation_id: str
    domain: str
    title: str
    description: str
    intended_purpose: str
    #: Why no recorded answer can justify offering this.
    no_link_reason: str
    limitations: Tuple[str, ...]
    recommendation_source: str
    review_status: str = REVIEW_PENDING

    @property
    def is_approved(self) -> bool:
        return self.review_status == REVIEW_APPROVED


NON_TRIGGERABLE_STRATEGIES: Tuple[NonTriggerableStrategy, ...] = (
    NonTriggerableStrategy(
        recommendation_id="rec_emotion_predictable_scripts",
        domain="emotional_regulation",
        title="Simple words for hard moments",
        description=("Short, predictable phrases used the same way each time "
                     "can lower the guesswork in a stressful moment."),
        intended_purpose="Make stressful moments more predictable",
        no_link_reason=("the questionnaire asks about wanting to comfort an "
                        "upset person; it does not ask about the child's own "
                        "moments of distress"),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:person-centred-practice-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_emotion_quiet_break_option",
        domain="emotional_regulation",
        title="Offer a quiet break as an option",
        description=("Agree in advance on a signal that means 'I need a "
                     "quieter moment', and make a low-stimulation spot "
                     "available."),
        intended_purpose="Give an exit ramp when things get loud",
        no_link_reason=("the questionnaire does not ask about the child's own "
                        "stress or need for low-stimulation space"),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:person-centred-practice-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_sensory_quieter_space",
        domain="sensory_environment",
        title="Make a quieter space available",
        description=("Pick one room or corner that can stay calmer — fewer "
                     "sounds, softer light."),
        intended_purpose="Have somewhere comfortable to go",
        no_link_reason=("the questionnaire does not ask about sounds, lights, "
                        "textures or places"),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:project-curated-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_sensory_adjustable_conditions",
        domain="sensory_environment",
        title="Adjust what can be adjusted",
        description=("Where you control the setting — lighting, volume, "
                     "clothing textures — make those adjustments and keep them "
                     "if they help."),
        intended_purpose="Reduce avoidable discomfort in controlled settings",
        no_link_reason=("the questionnaire does not ask about sounds, lights, "
                        "textures or places"),
        limitations=("Not every environment is adjustable",),
        recommendation_source="guidance:accessibility-practice-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_transitions_advance_notice",
        domain="predictability_transitions",
        title="Give changes a heads-up",
        description=("Mention what is coming next before it happens, and keep "
                     "the wording steady from day to day."),
        intended_purpose="Make change predictable rather than sudden",
        no_link_reason=("the questionnaire does not ask about routines, advance "
                        "notice or transitions"),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:project-curated-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_transitions_visual_schedule",
        domain="predictability_transitions",
        title="Show the day, not just say it",
        description=("A short visual list of the main parts of the day that can "
                     "be checked and ticked off together."),
        intended_purpose="Make the shape of the day visible",
        no_link_reason=("the questionnaire does not ask about routines, advance "
                        "notice or transitions"),
        limitations=("A general strategy, not tailored to any individual",),
        recommendation_source="guidance:accessibility-practice-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_daily_optional_tools",
        domain="daily_living_organization",
        title="Optional tools for daily steps",
        description=("If it helps: a visible checklist, one shared calendar, "
                     "or one agreed spot for the things that always go "
                     "missing."),
        intended_purpose="Lower the memory load of routines",
        no_link_reason=("the questionnaire does not ask about planning or "
                        "organising daily activities"),
        limitations=("Entirely optional; remove anything that does not help",),
        recommendation_source="guidance:project-curated-v1",
    ),
    NonTriggerableStrategy(
        recommendation_id="rec_comm_preference_choice",
        domain="communication_accessibility",
        title="Choose how suggestions reach you",
        description=("This tool can present the same information as written "
                     "steps, spoken summaries or visual summaries."),
        intended_purpose="Match the format to a stated preference",
        no_link_reason=("the questionnaire does not ask how someone prefers "
                        "information to be shared, so no preference exists to "
                        "honour"),
        limitations=("Applies to how this tool presents information",),
        recommendation_source="guidance:accessibility-practice-v1",
    ),
)

NON_TRIGGERABLE_BY_ID = {s.recommendation_id: s
                         for s in NON_TRIGGERABLE_STRATEGIES}


def _validate_registry() -> None:
    seen = set()
    for s in list(STRATEGIES) + list(NON_TRIGGERABLE_STRATEGIES):
        if s.recommendation_id in seen:
            raise ValueError(f"duplicate recommendation_id {s.recommendation_id}")
        seen.add(s.recommendation_id)
        for code in getattr(s, "trigger_item_codes", ()):
            if code not in ITEM_BY_CODE:
                raise ValueError(
                    f"{s.recommendation_id} links item {code!r}, which the "
                    f"verified questionnaire contract does not define")
        if s.recommendation_source not in GUIDANCE_SOURCES:
            raise ValueError(
                f"{s.recommendation_id} references unknown guidance "
                f"{s.recommendation_source!r}")


_validate_registry()
