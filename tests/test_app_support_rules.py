"""Observation-based suggestion rules (audit findings T2/T3): whole words, simple negation, returned matches."""
from __future__ import annotations

import pytest

from app.services.support import FALLBACK_SOURCE, RULES, SOURCE, find_terms, suggest


def titles(**obs):
    return [s["title"] for s in suggest({"id": "obs_1", **obs})]


def fired(**obs):
    return {s["title"]: s["matched_terms"] for s in suggest({"id": "obs_1", **obs}) if s["source"] == SOURCE}


# ---------------------------------------------------------------- the T3 substring false positives are gone


@pytest.mark.parametrize("text", [
    "He exchanged toys with a friend",       # "exchange" contains "change"
    "The result was unchanged all week",     # "unchanged" contains "change"
    "She was unfocused and sleepy",          # "unfocused" contains "focus"
    "Tried a screenshot of the timetable",   # nothing relevant
    "A misunderstanding with the neighbour", # "misunderstanding" contains "understand"
    "The route was fine",                    # "route" is not "routine"
    "He was proud of the loudspeaker project",  # "loudspeaker" contains "loud"
])
def test_words_that_merely_contain_a_trigger_do_not_fire(text):
    assert fired(what_happened=text) == {}
    assert titles(what_happened=text) == ["Structured support check-in"]


@pytest.mark.parametrize("text, title, term", [
    ("Hard time with the transition to bedtime", "Advance transition warnings", "transition"),
    ("Transitions between lessons were hard", "Advance transition warnings", "transitions"),
    ("The bedtime routine fell apart", "Visual schedule", "routine"),
    ("Upset when the schedule moved", "Visual schedule", "schedule"),
    ("It was very loud in the canteen", "Quieter environment options", "loud"),
    ("Covered ears because of the noisy hall", "Quieter environment options", "noisy"),
    ("Did not follow the instructions", "Clearer visual instructions", "instructions"),
    ("Seemed distracted during the task", "Reduced environmental distractions", "distracted"),
    ("Hard to focus on homework", "Reduced environmental distractions", "focus"),
    ("Upset by the unexpected visitor", "Flexible routine options", "unexpected"),
    ("Plans changed at the last minute", "Flexible routine options", "changed"),
])
def test_whole_word_matches_still_fire_and_report_the_term(text, title, term):
    hits = fired(what_happened=text)
    assert title in hits
    assert {"term": term, "field": "what_happened"} in hits[title]


def test_matching_is_case_insensitive():
    assert "Visual schedule" in fired(what_happened="THE ROUTINE WAS HARD")


# ---------------------------------------------------------------- negation


@pytest.mark.parametrize("text", [
    "No noise issue today", "Not loud at all", "It stayed calm without any transitions",
    "There was never a routine problem", "There were no changes at all",
])
def test_a_directly_negated_term_does_not_fire(text):
    assert fired(what_happened=text) == {}


def test_negation_does_not_cross_a_sentence_boundary():
    # "no" is inside the two-word window of "loud" but belongs to the previous sentence
    assert "Quieter environment options" in fired(what_happened="She said no. Loud bus ride.")


def test_negation_only_affects_the_negated_occurrence():
    hits = fired(what_happened="No noise at lunch but the assembly was loud")
    assert [m["term"] for m in hits["Quieter environment options"]] == ["loud"]


def test_the_documented_negation_limit_is_real_not_hidden():
    """'didn't seem to struggle with transitions' still fires (a two-word window cannot see 'didn't'). The match
    is shown to the person, which is why the limitation is acceptable and documented in the module docstring."""
    assert "Advance transition warnings" in fired(what_happened="He didn't seem to struggle with transitions")


# ---------------------------------------------------------------- fields and context


def test_matches_report_the_field_they_came_from():
    hits = fired(what_happened="Quiet day", what_helped="A visual schedule helped", notes="Loud bus ride")
    assert {"term": "schedule", "field": "what_helped"} in hits["Visual schedule"]
    assert {"term": "loud", "field": "notes"} in hits["Quieter environment options"]


@pytest.mark.parametrize("context, title", [
    ("Transitions", "Advance transition warnings"), ("Daily routines", "Visual schedule"),
    ("Sensory", "Quieter environment options"), ("Communication", "Clearer visual instructions"),
    ("Classroom", "Reduced environmental distractions"),
])
def test_context_labels_fire_their_rule_even_when_the_text_has_no_keyword(context, title):
    hits = fired(context=context, what_happened="Had a difficult afternoon")
    assert title in hits and any(m["field"] == "context" for m in hits[title])


def test_a_context_is_reported_once_even_if_its_word_also_matches():
    hits = fired(context="Daily routines", what_happened="x")
    assert sum(1 for m in hits["Visual schedule"] if m["field"] == "context") == 1


def test_unknown_context_with_no_keywords_gets_the_labelled_fallback():
    out = suggest({"id": "o", "context": "Playground", "what_happened": "Played near the swings"})
    assert len(out) == 1 and out[0]["source"] == FALLBACK_SOURCE and out[0]["matched_terms"] == []
    assert out[0]["domain"] == "Playground" and out[0]["triggered_observation_id"] == "o"


def test_missing_or_null_fields_do_not_crash():
    assert titles(what_happened=None, notes=None) == ["Structured support check-in"]
    assert suggest({})[0]["triggered_observation_id"] is None


def test_every_suggestion_has_the_fields_the_ui_and_save_flow_need():
    for s in suggest({"id": "o", "what_happened": "loud transition routine change focus instructions"}):
        assert {"title", "domain", "why_useful", "how_to_try", "how_to_record", "triggered_observation_id",
                "source", "matched_terms"} <= set(s)
        assert all(set(m) == {"term", "field"} for m in s["matched_terms"])
    assert len({s["title"] for s in suggest({"what_happened": "loud transition routine change focus instructions"})}) == len(RULES)


def test_find_terms_orders_by_appearance_and_dedupes():
    import re
    pats = [re.compile(r"\b(?:loud|noise)\b", re.I)]
    assert find_terms(pats, "Noise, then loud, then noise again") == ["noise", "loud"]


# ---------------------------------------------------------------- through the API


def test_the_suggest_endpoint_returns_matched_terms(api):
    oid = api.post("/api/observations", json={"context": "Playground", "what_happened": "The assembly was loud"}).get_json()["data"]["id"]
    data = api.post("/api/support/suggest", json={"observation_id": oid}).get_json()["data"]
    assert data[0]["title"] == "Quieter environment options"
    assert data[0]["matched_terms"] == [{"term": "loud", "field": "what_happened"}]
    assert data[0]["triggered_observation_id"] == oid
