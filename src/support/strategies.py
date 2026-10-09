"""Curated support strategies with provenance — P4.

Every strategy is *data* with an auditable basis:

* ``basis`` — why it was offered (user-confirmed need / opted-in hypothesis /
  general guidance); set by the engine, never by the strategy itself.
* ``evidence_references`` — resolve against :data:`GUIDANCE_SOURCES` or against
  an assessment id in the report.
* ``review_status`` — nothing here ships as ``approved`` until a human reviews
  it. The library ships entirely as ``pending_expert_review``, and the report
  says so. No external URLs are asserted: guidance entries name the *kind* of
  source and its review state, so no citation can be fabricated by accident.

Wording guardrails (enforced in ``tests/test_support_strategies.py``): optional,
non-prescriptive, non-deficit; no "treatment", "cure", "normal", "fix"; no claim
about any individual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

#: Review states for curated content and guidance sources.
REVIEW_PENDING = "pending_expert_review"
REVIEW_APPROVED = "approved"

#: The review status required before this content may support any claim beyond
#: a research demonstration. It gates *deployment claims*, not demo emission:
#: the engine may emit pending-review strategies for demonstration, but the
#: assembled report must disclose the pending status (see src/support/report.py).
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
    applicability_conditions: Tuple[str, ...]
    limitations: Tuple[str, ...]
    evidence_references: Tuple[str, ...]
    source_attribution: str
    review_status: str = REVIEW_PENDING

    @property
    def is_approved(self) -> bool:
        return self.review_status == REVIEW_APPROVED


STRATEGIES: Tuple[SupportStrategy, ...] = (
    # ---- social communication -------------------------------------------
    SupportStrategy(
        recommendation_id="rec_comm_processing_time",
        domain="social_communication",
        title="Offer extra processing time",
        description=("When giving instructions or asking a question, pause "
                     "longer than usual before repeating or rephrasing. "
                     "Counting slowly to ten in your head is enough."),
        intended_purpose="Reduce time pressure in everyday exchanges",
        applicability_conditions=(
            "You asked for ideas with everyday communication moments",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:project-curated-v1",),
        source_attribution="curated project content (pending expert review)",
    ),
    SupportStrategy(
        recommendation_id="rec_comm_visual_supports",
        domain="social_communication",
        title="Pair words with something visual",
        description=("Add a gesture, picture, object or demonstration alongside "
                     "spoken words when it helps — for example holding up the "
                     "coat while saying it is time to go out."),
        intended_purpose="Give the message more than one route to land",
        applicability_conditions=(
            "You asked for ideas with everyday communication moments",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:project-curated-v1",),
        source_attribution="curated project content (pending expert review)",
    ),
    SupportStrategy(
        recommendation_id="rec_comm_attention_cues",
        domain="social_communication",
        title="Get attention before you speak",
        description=("Say the person's name and wait for them to look or turn "
                     "before starting. If that does not work, move into their "
                     "line of sight rather than repeating louder."),
        intended_purpose="Make easier to know when something is being shared",
        applicability_conditions=(
            "You asked for ideas with everyday communication moments",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:project-curated-v1",),
        source_attribution="curated project content (pending expert review)",
    ),
    SupportStrategy(
        recommendation_id="rec_comm_preference_choice",
        domain="communication_accessibility",
        title="Choose how suggestions reach you",
        description=("You told us how you prefer to receive information. This "
                     "tool can present the same suggestions as written steps, "
                     "spoken summaries or visual summaries — your choice, "
                     "changeable at any time."),
        intended_purpose="Match the format to your stated preference",
        applicability_conditions=("You shared a format preference",),
        limitations=("Applies to how this tool presents information",),
        evidence_references=("guidance:accessibility-practice-v1",),
        source_attribution="accessibility practice (pending expert review)",
    ),
    # ---- emotional regulation -------------------------------------------
    SupportStrategy(
        recommendation_id="rec_emotion_predictable_scripts",
        domain="emotional_regulation",
        title="Simple words for hard moments",
        description=("Short, predictable phrases used the same way each time "
                     "can lower the guesswork in a stressful moment — for "
                     "example 'that was loud; we can step out when you want'."),
        intended_purpose="Make stressful moments more predictable",
        applicability_conditions=(
            "You asked for ideas with big feelings or stressful moments",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:person-centred-practice-v1",),
        source_attribution="person-centred practice (pending expert review)",
    ),
    SupportStrategy(
        recommendation_id="rec_emotion_quiet_break_option",
        domain="emotional_regulation",
        title="Offer a quiet break as an option",
        description=("Agree in advance on a signal or phrase that means 'I need "
                     "a quieter moment', and make a low-stimulation spot "
                     "available without asking for an explanation."),
        intended_purpose="Give an exit ramp when things get loud",
        applicability_conditions=(
            "You asked for ideas with big feelings or stressful moments",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:person-centred-practice-v1",),
        source_attribution="person-centred practice (pending expert review)",
    ),
    # ---- sensory / environment -------------------------------------------
    SupportStrategy(
        recommendation_id="rec_sensory_quieter_space",
        domain="sensory_environment",
        title="Make a quieter space available",
        description=("Pick one room or corner that can stay calmer — fewer "
                     "sounds, softer light — and let it be used without "
                     "explanation when wanted."),
        intended_purpose="Have somewhere comfortable to go",
        applicability_conditions=(
            "You said sounds, lights, textures or places matter to you",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:project-curated-v1",),
        source_attribution="curated project content (pending expert review)",
    ),
    SupportStrategy(
        recommendation_id="rec_sensory_adjustable_conditions",
        domain="sensory_environment",
        title="Adjust what can be adjusted",
        description=("Where you control the setting — lighting, volume, "
                     "clothing textures, background noise — make those "
                     "adjustments and keep them if they help."),
        intended_purpose="Reduce avoidable discomfort in controlled settings",
        applicability_conditions=(
            "You said sounds, lights, textures or places matter to you",),
        limitations=("Not every environment is adjustable",),
        evidence_references=("guidance:accessibility-practice-v1",),
        source_attribution="accessibility practice (pending expert review)",
    ),
    # ---- predictability / transitions ------------------------------------
    SupportStrategy(
        recommendation_id="rec_transitions_advance_notice",
        domain="predictability_transitions",
        title="Give changes a heads-up",
        description=("Mention what is coming next before it happens — 'in five "
                     "minutes we will pack up' — and keep the wording steady "
                     "from day to day."),
        intended_purpose="Make change predictable rather than sudden",
        applicability_conditions=(
            "You said advance notice or steadier routines would help",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:project-curated-v1",),
        source_attribution="curated project content (pending expert review)",
    ),
    SupportStrategy(
        recommendation_id="rec_transitions_visual_schedule",
        domain="predictability_transitions",
        title="Show the day, not just say it",
        description=("A short visual list of the main parts of the day "
                     "(pictures or objects works too) that can be checked and "
                     "ticked off together."),
        intended_purpose="Make the shape of the day visible",
        applicability_conditions=(
            "You said advance notice or steadier routines would help",),
        limitations=("A general strategy, not tailored to any individual",),
        evidence_references=("guidance:accessibility-practice-v1",),
        source_attribution="accessibility practice (pending expert review)",
    ),
    # ---- daily living -----------------------------------------------------
    SupportStrategy(
        recommendation_id="rec_daily_optional_tools",
        domain="daily_living_organization",
        title="Optional tools for daily steps",
        description=("If it helps: a visible checklist, a single calendar "
                     "everyone uses, or one agreed spot for the things that "
                     "always go missing. Use the pieces that work; ignore the "
                     "rest."),
        intended_purpose="Lower the memory load of routines",
        applicability_conditions=(
            "You said organisational tools would be useful",),
        limitations=("Entirely optional; remove anything that does not help",),
        evidence_references=("guidance:project-curated-v1",),
        source_attribution="curated project content (pending expert review)",
    ),
)

STRATEGIES_BY_ID = {s.recommendation_id: s for s in STRATEGIES}
