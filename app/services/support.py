"""Observation-based support suggestions (audit findings T2 and T3).

Rule-based and deliberately simple: a rule fires when one of its terms appears, as a *whole word*, in the
text of the observation, or when the observation's context label is one the rule names. Every suggestion
reports which terms fired and in which field (``matched_terms``) so a person can see why it was offered
and dismiss it when the match is wrong. That is the same provenance principle as the questionnaire-driven
report in ``src/support``, which this module is separate from.

What changed from the old substring matching (T3):

* whole-word matching: "exchange" and "unchanged" no longer fire the *change* rule, "unfocused" no longer
  fires *focus*, "screenshot" ... no longer fires anything;
* simple negation: a term directly preceded by no/not/never/without/none (within two words, same clause)
  does not fire, so "no noise issue" does not suggest a quieter environment;
* the fired terms are returned and shown.

Limits, stated rather than hidden: negation handling is the two-word window above and nothing smarter
("didn't seem to struggle with transitions" still fires), and the vocabulary is English only. These are
prompts for a person to consider, not findings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

SOURCE = "rule-based keyword match on the recorded observation"
FALLBACK_SOURCE = "no keyword rule matched the recorded observation"

#: Observation fields searched, in the order matches are reported.
FIELDS = ("what_happened", "difficulty", "context", "what_helped", "notes")

_NEGATORS = {"no", "not", "never", "without", "none", "neither", "nor"}
_TOKEN = re.compile(r"[a-z0-9']+|[.;:!?]", re.IGNORECASE)


@dataclass(frozen=True)
class Rule:
    title: str
    domain: str
    why: str
    how: str
    record: str
    patterns: Tuple[str, ...]      # regexes; each is wrapped in \b...\b
    contexts: Tuple[str, ...] = ()  # exact (case-insensitive) context labels that also fire the rule

    def compiled(self) -> List[re.Pattern]:
        return [re.compile(r"\b(?:" + p + r")\b", re.IGNORECASE) for p in self.patterns]


RULES: Tuple[Rule, ...] = (
    Rule("Advance transition warnings", "Transitions",
         "The observation involves change/transition; advance warnings can make upcoming changes more predictable.",
         "Give a clear warning before a change (e.g. 5 and 1 minute), using a timer or visual cue.",
         "Note whether the warning was given and if the transition felt easier or harder.",
         (r"transition(?:s|ed|ing)?",), ("transitions",)),
    Rule("Visual schedule", "Routines",
         "Recorded routine-related difficulty may benefit from making steps visible and predictable.",
         "Use a simple picture/step schedule for the routine and review it before starting.",
         "Record which steps were used and whether the routine was smoother.",
         (r"routines?", r"schedul(?:e|es|ed|ing)"), ("daily routines",)),
    Rule("Quieter environment options", "Sensory environment",
         "The observation mentions sensory/noise factors; reducing stimulation can lower discomfort for some people.",
         "Offer a quieter space, headphones, or reduced background noise when possible.",
         "Record the environment change and whether comfort/engagement improved.",
         (r"nois(?:e|es|y)", r"loud(?:ly)?", r"sensory"), ("sensory",)),
    Rule("Clearer visual instructions", "Communication",
         "Communication/instruction difficulty was noted; visual supports can make expectations clearer.",
         "Pair spoken instructions with simple visuals, written steps, or demonstrations.",
         "Record which support was used and whether understanding improved.",
         (r"instructions?", r"understand(?:s|ing)?", r"communicat(?:e|es|ed|ing|ion)"),
         ("communication",)),
    Rule("Reduced environmental distractions", "Environment",
         "Distraction was noted; simplifying the environment can support focus.",
         "Reduce clutter/noise near the activity and provide a clear workspace.",
         "Record the change and any effect on focus or participation.",
         (r"distract(?:s|ed|ing|ion|ions)?", r"focus(?:es|ed|ing)?"), ("classroom",)),
    Rule("Flexible routine options", "Adaptability",
         "Difficulty with change was noted; offering choices can support flexibility.",
         "Prepare a simple alternative/choice when plans change and explain it clearly.",
         "Record the options offered and how the change was experienced.",
         (r"chang(?:e|es|ed|ing)", r"unexpected", r"flexib(?:le|ility)")),
)
_COMPILED = [(r, r.compiled()) for r in RULES]


def _negated(text: str, start: int) -> bool:
    """True when a negator sits within two words before ``start`` in the same clause."""
    before = _TOKEN.findall(text[:start].lower())
    window: List[str] = []
    for tok in reversed(before):
        if tok in ".;:!?":
            break
        window.append(tok)
        if len(window) == 2:
            break
    return any(w in _NEGATORS for w in window)


def find_terms(rule_patterns: List[re.Pattern], text: str) -> List[str]:
    """Whole-word terms from ``rule_patterns`` found in ``text`` and not negated, in order of appearance."""
    hits: List[Tuple[int, str]] = []
    for pat in rule_patterns:
        for m in pat.finditer(text):
            if not _negated(text, m.start()):
                hits.append((m.start(), m.group(0).lower()))
    seen, out = set(), []
    for _, term in sorted(hits):
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def suggest(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Support suggestions for one recorded observation, each with the terms that fired it."""
    obs_id = observation.get("id")
    context = str(observation.get("context") or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for rule, patterns in _COMPILED:
        matched: List[Dict[str, str]] = []
        for field in FIELDS:
            text = str(observation.get(field) or "")
            for term in find_terms(patterns, text):
                matched.append({"term": term, "field": field})
        if context and context in rule.contexts and not any(m["field"] == "context" for m in matched):
            matched.append({"term": context, "field": "context"})
        if matched:
            out.append({"title": rule.title, "domain": rule.domain, "why_useful": rule.why, "how_to_try": rule.how,
                        "how_to_record": rule.record, "triggered_observation_id": obs_id, "source": SOURCE,
                        "matched_terms": matched})
    if not out:
        out.append({"title": "Structured support check-in", "domain": observation.get("context") or "General",
                    "why_useful": "A recorded observation may benefit from a small, observable support to try.",
                    "how_to_try": "Choose one small environmental or communication adjustment related to what was observed.",
                    "how_to_record": "Record what was tried, context, and whether it seemed helpful.",
                    "triggered_observation_id": obs_id, "source": FALLBACK_SOURCE, "matched_terms": []})
    return out
