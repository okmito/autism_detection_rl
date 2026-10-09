"""The optional follow-up questionnaire — P4.

Asked **after** the screening result and its explanation are shown, on its own
screen, always skippable. Rules this questionnaire is written to:

* every item is about the person's (or caregiver's) **own experience and what
  they would find useful** — never an assertion that a difficulty exists;
* every item allows ``unsure`` / ``not_applicable`` / ``prefer_not_to_answer``;
* it does not repeat any screening question, and is stored separately from
  screening inputs (it never enters the RL state or the predictor input);
* items whose domain has no screening-data linkage say so in ``helper_text``.

Wording is data, reviewed like data; see ``test_support_questions.py`` for the
wording guardrails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FollowUpQuestion:
    question_id: str
    domain_id: str
    text: str
    helper_text: str = ""
    #: For preference questions: the closed vocabulary of answers. When set,
    #: ``choice`` (not the yes/no vocabulary) carries the answer.
    choices: Optional[Tuple[str, ...]] = None

    @property
    def is_preference_question(self) -> bool:
        return self.choices is not None


QUESTIONS: Tuple[FollowUpQuestion, ...] = (
    FollowUpQuestion(
        question_id="q_comm_support",
        domain_id="social_communication",
        text=("Are there everyday communication moments you would like ideas "
              "or support with — for example getting attention, using gestures "
              "or first words?"),
        helper_text=("This asks what you would find useful. It is not a "
                     "question about whether anything is 'wrong'."),
    ),
    FollowUpQuestion(
        question_id="q_comm_preference",
        domain_id="communication_accessibility",
        text=("How would you prefer information and suggestions to be shared "
              "with you?"),
        helper_text="Your preference is recorded and used only for you.",
        choices=("written", "spoken", "visual", "demonstration",
                 "no_preference"),
    ),
    FollowUpQuestion(
        question_id="q_emotion_support",
        domain_id="emotional_regulation",
        text=("Would you like ideas for supporting big feelings or stressful "
              "moments, if they come up?"),
        helper_text=("Your screening answers suggested this area might be "
                     "worth exploring — you decide whether it is."),
    ),
    FollowUpQuestion(
        question_id="q_sensory_support",
        domain_id="sensory_environment",
        text=("Are there sounds, lights, textures or places where small "
              "adjustments would make things more comfortable for you?"),
        helper_text=("The screening questionnaire does not ask about sensory "
                     "experiences; this comes only from you."),
    ),
    FollowUpQuestion(
        question_id="q_transitions_support",
        domain_id="predictability_transitions",
        text=("Would knowing about changes ahead of time, or steadier "
              "routines, help your day?"),
        helper_text=("The screening questionnaire does not ask about routines; "
                     "this comes only from you."),
    ),
    FollowUpQuestion(
        question_id="q_daily_support",
        domain_id="daily_living_organization",
        text=("Would optional tools for planning or organising daily "
              "activities be useful to you?"),
        helper_text=("Not covered by the screening questions, so this comes "
                     "only from you — and optional means exactly that."),
    ),
    FollowUpQuestion(
        question_id="q_accessibility_preference",
        domain_id="communication_accessibility",
        text=("Is there anything about how this tool presents information that "
              "would work better for you?"),
        helper_text=("For example written steps, spoken summaries, simpler "
                     "layout, or more time. Your preference is recorded and "
                     "used only for you."),
        choices=("written_steps", "spoken_summaries", "simpler_layout",
                 "more_time", "no_preference"),
    ),
)

QUESTIONS_BY_ID = {q.question_id: q for q in QUESTIONS}


def questions_for_domain(domain_id: str) -> Tuple[FollowUpQuestion, ...]:
    return tuple(q for q in QUESTIONS if q.domain_id == domain_id)


def questions_with_data_linkage(domain_id: str) -> Tuple[FollowUpQuestion, ...]:
    """Questions whose domain has screening-data linkage (can be *suggested*
    by observed responses). The others are offered only as optional extras."""
    from src.support.domains import DOMAINS_BY_ID
    d = DOMAINS_BY_ID[domain_id]
    return tuple(q for q in questions_for_domain(domain_id)
                 if d.has_data_linkage)
