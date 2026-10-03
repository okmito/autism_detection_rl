"""Guards for the redesigned demo front end.

These lock the properties the redesign was required to deliver, so a later edit
cannot quietly reintroduce them:

* no hard-coded budget/threshold/instrument constant;
* budget, policy list and threshold all sourced from /api/meta;
* positioning is screening research, never a diagnosis claim;
* explainability renders backend fields and is never invented client-side;
* policy descriptions are not duplicated per-policy in the markup;
* error text is humanised and raw detail is behind a disclosure;
* accessibility affordances (semantic buttons, labels, aria state, focus);
* responsive breakpoints exist.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "scripts" / "demo_static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    if not HTML.exists():
        pytest.skip("frontend not present")
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def flat(html) -> str:
    """Whitespace-collapsed copy.

    Copy in the markup is hard-wrapped for readability, so prose assertions must
    not depend on where the line breaks fall.
    """
    return re.sub(r"\s+", " ", html)


# ================================== no hard-coded research configuration
def test_no_hardcoded_budget_constant(html):
    assert "const BUDGET" not in html
    assert "let BUDGET" not in html
    assert not re.search(r"\bBUDGET\s*=\s*\d", html)


def test_budget_threshold_and_items_come_from_meta(html):
    for token in ("/api/meta", "policies_available", "m.budget", "m.tau",
                  "m.n_items"):
        assert token in html, token


def test_progress_uses_backend_budget(html):
    """Progress must be asked/max-budget, never asked/10 or a literal."""
    assert "' of ' + budget" in html
    assert "askedSoFar / budget" in html


def test_no_literal_question_count_assumptions(html):
    assert "of 10 questions" not in html
    assert "Question 10" not in html


# ==================================================== product positioning
def test_titles_are_screening_research(html, flat):
    assert "Adaptive Autism Screening" in html
    assert "Research Prototype" in html
    assert "Adaptive questionnaire-based screening using intelligent question selection." \
        in html
    assert "This research prototype is intended for screening research and does not provide a medical diagnosis." in flat


def test_no_diagnosis_claims(html):
    lowered = html.lower()
    for banned in ("autism diagnosed", "medical diagnosis:", "diagnoses autism",
                   "diagnosis of autism", "confirmed autism", "autistic diagnosis"):
        assert banned not in lowered, banned


def test_result_screen_avoids_diagnosis_wording(html, flat):
    assert "Screening result" in html
    assert "Screening risk estimate" in html
    assert "should not be interpreted as a medical diagnosis" in flat


def test_backend_decision_label_is_shown_as_provided(html):
    """The referral decision comes from the backend and is not reworded as a verdict."""
    assert "REFERRAL_RECOMMENDED" in html or "res.decision" in html


# ========================================================= explainability
def test_why_this_question_section_exists(html):
    assert "Why this question?" in html


def test_explainability_uses_backend_diagnostic_fields(html):
    for field in ("criterion_vacuous", "ig_spread", "support_size",
                  "support_is_pure", "tie_at_max", "n_legal", "posterior",
                  "top_items"):
        assert field in html, field


def test_no_invented_explanation_copy(html):
    """The vacuous/informative wording must be driven by the backend flag."""
    assert "sel.criterion_vacuous" in html
    assert "did not discriminate" in html


def test_vacuous_criterion_is_explained_honestly(html, flat):
    assert "label-pure" in html
    assert "property of the data" in html or "fallback ordering" in html


# ===================================================== policy presentation
def test_policy_labels_are_readable(html):
    for label in ("Greedy Information Gain", "Random Fixed-Length",
                  "Beta-Greedy / EVOI", "DQN"):
        assert label in html, label


def test_beta_greedy_is_not_described_as_rl(html):
    """beta-greedy is a Bayesian heuristic; it must not be labelled an RL policy."""
    idx = html.find("Beta-Greedy / EVOI")
    block = html[idx:idx + 400] if idx >= 0 else ""
    assert "reinforcement" not in block.lower()
    assert "DQN" in block or "learned" in block.lower()  # only DQN is learned
    assert "'beta_greedy': {\n    label: 'Beta-Greedy / EVOI'," in html or \
        "kind: 'Heuristic'" in html


def test_policy_list_comes_from_backend(html):
    """Policies are rendered from meta.policies_available, not a fixed markup list."""
    assert "state.meta.policies_available" in html
    assert "POLICY_PRESENTATION" in html
    # a single presentation table, not per-policy duplicated markup
    assert html.count("data-policy=") == 1, "policy buttons must be generated, not hard-coded"


def test_policy_details_only_shows_available_metadata(html):
    assert "Policy details" in html
    assert "renderPolicyDetails" in html


# ==================================================== errors and loading
def test_errors_are_humanised(html):
    assert "friendlyError" in html
    assert "Something went wrong while processing this answer." in html
    assert "Technical details" in html


def test_raw_backend_error_is_not_shown_bare(html):
    """Raw exception text must sit behind a disclosure, not as the primary message."""
    assert "showError" in html
    assert "<pre>" in html
    # the friendly sentence is rendered as the strong lead, raw text inside <pre>
    assert re.search(r"<strong>.*friendlyError", html, re.S)


def test_loading_states_present(html):
    assert "Selecting next question" in html
    assert "Updating screening estimate" in html
    assert 'class="spin"' in html


def test_no_page_reload(html):
    assert "location.reload" not in html
    assert "form.submit" not in html


# ============================================== reset / session handling
def test_new_screening_is_the_completion_action(html):
    assert "Start new screening" in html
    assert "newBtn" in html


def test_reset_clears_all_stale_state(html):
    body = html[html.find("function resetAll"):html.find("function resetAll") + 1400]
    for token in ("state.sessionId = null", "state.liveTrace = []",
                  "state.lastSelection = null", "$('resultScreen').hidden = true",
                  "$('quizScreen').hidden = true", "$('progBar').style.width = '0%'"):
        assert token in body, token


def test_reset_does_not_call_a_backend_endpoint(html):
    """Backend session semantics are unchanged: the client simply starts fresh."""
    body = html[html.find("function resetAll"):html.find("function resetAll") + 1400]
    assert "api(" not in body


# ==================================================== answers / double submit
def test_answer_buttons_are_semantic_and_labelled(html):
    assert 'data-value="0"' in html
    assert 'data-value="1"' in html
    assert "No / typical" in html
    assert "Yes / atypical" in html
    assert html.count('class="ansbtn"') == 2


def test_answer_values_unchanged(html):
    """0/1 encoding must not be altered by a UI change."""
    assert "value: 0" in html or "data-value=\"0\"" in html
    assert "Number(b.dataset.value)" in html


def test_double_submission_prevented(html):
    assert "setAnswersEnabled(false)" in html
    assert "if (b && !b.disabled)" in html
    assert "state.finished) return" in html


# ==================================================== accessibility / layout
def test_focus_visible_styles_present(html):
    assert ":focus-visible" in html


def test_form_controls_have_accessible_names(html):
    assert 'aria-labelledby="polLabel"' in html
    assert 'aria-labelledby="modeLabel"' in html
    assert 'role="group"' in html


def test_live_region_for_status(html):
    assert 'aria-live="polite"' in html
    assert 'role="status"' in html


def test_selected_state_not_colour_only(html):
    assert 'aria-pressed' in html
    assert ".dot" in html


def test_responsive_breakpoints_present(html):
    assert "@media(max-width:520px)" in html
    assert "viewport" in html


def test_reduced_motion_respected(html):
    assert "prefers-reduced-motion" in html


def test_decorative_mark_hidden_from_screen_readers(html):
    assert 'aria-hidden="true"' in html


# ================================================ no framework introduced
def test_stays_dependency_free(html):
    """Single self-contained file, matching the existing architecture."""
    assert "<script src=" not in html
    assert "cdn." not in html
    assert "import " not in html.split("<script>")[-1]