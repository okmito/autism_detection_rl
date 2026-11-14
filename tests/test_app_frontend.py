"""Frontend regression tests for the NeuroBridge SPA (audit findings C3, I1 and the limitations UI).

Two layers:

* static checks of ``app/static/js/app.js`` that need nothing installed;
* DOM tests that run the real ``app.js`` in jsdom (``tests/js/frontend_harness.js``). They are
  skipped, with a message, unless ``node`` and ``npm install`` (jsdom) are available.

C3: ``$$('[data-x]', '#id')`` passed a *string* where the helper needs an element, so Support Studio
    threw on every load and the router replaced the page with "Failed to load module".
I1: the support-report card read field names that do not exist on the backend schema, so "Triggered
    by", "Why" and the guidance were always blank.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APP_JS = REPO / "app" / "static" / "js" / "app.js"
HARNESS = REPO / "tests" / "js" / "frontend_harness.js"

ROUTES = ["overview", "screening", "context", "support", "timeline", "rl-lab", "explain", "research", "privacy"]


# ============================================================ static checks (no node needed)

STRING_SCOPE = re.compile(r"""\$\$\(\s*(['"`])(?:(?!\1).)*\1\s*,\s*(['"`])""")


def test_the_string_scope_detector_actually_detects_the_c3_pattern():
    assert STRING_SCOPE.search("$$('[data-edit]','#stratList').forEach(f)")
    assert STRING_SCOPE.search('$$("[data-i]", "#sugList")')
    assert not STRING_SCOPE.search("$$('[data-i]', $('#sugList')).forEach(f)")
    assert not STRING_SCOPE.search("$$('#nav a')")


def test_dollar_dollar_is_never_given_a_selector_string_as_its_scope():
    """C3: the second argument of ``$$`` must be an element, not a selector string."""
    offenders = [m.group(0) for m in STRING_SCOPE.finditer(APP_JS.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_renderer_only_reads_fields_that_exist_on_the_report_schema():
    """I1: ``rec.x`` must be a SupportRecommendation field and ``support.x`` a ScreeningReport field."""
    from src.support.schemas import ScreeningReport, SupportRecommendation

    js = APP_JS.read_text(encoding="utf-8")
    rec_fields = set(re.findall(r"\brec\.([A-Za-z_]\w*)", js))
    report_fields = set(re.findall(r"\bsupport\.([A-Za-z_]\w*)", js))
    assert rec_fields, "renderer no longer reads recommendation fields? update this test"
    assert rec_fields <= set(SupportRecommendation.model_fields), rec_fields - set(SupportRecommendation.model_fields)
    assert report_fields <= set(ScreeningReport.model_fields), report_fields - set(ScreeningReport.model_fields)
    # the fields that carry the "named evidence" guarantee must be read
    assert {"triggering_question_ids", "triggering_responses", "why_selected", "limitations"} <= rec_fields
    assert {"limitations", "disclaimer", "unassessed_areas", "recommendations"} <= report_fields


# ============================================================ jsdom (needs node + npm install)


def run_scenario(tmp_path, scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed (frontend DOM tests are optional)")
    if not (REPO / "node_modules" / "jsdom").exists():
        pytest.skip("jsdom missing: run `npm install` in the repo root to enable the frontend DOM tests")
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    proc = subprocess.run([node, str(HARNESS), str(path)], capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def make_report():
    """A real, schema-valid support report built from fixed answers (no dataset or model needed)."""
    from src.support.report import build_support_report
    from src.support.schemas import to_json_safe

    atypical = {0, 2, 3, 4, 6, 8}
    trace = [{"step": i + 1, "item": f"A{i + 1}", "item_idx": i, "value": int(i in atypical),
              "belief_before": 0.3, "belief_after": 0.4} for i in range(10)]
    episode = {"decision": "REFERRAL_RECOMMENDED", "p_hat": 0.8, "trace": trace,
               "items_asked": list(range(10)), "stop_reason": "budget_exhausted"}
    return to_json_safe(build_support_report(episode_result=episode, result_id="fixture-1"))


BASE_API = {
    "/api/meta": {"ok": True, "data": {"budget": 6, "tau": 0.5, "n_items": 10, "policies_available": ["greedy", "random", "beta_greedy"]}},
    "/api/rl/scenarios": {"ok": True, "data": {"scenarios": [{"id": "syn_mixed", "name": "Synthetic: mixed"}], "policies": [], "budgets": [1, 6]}},
    "/api/research/dashboard": {"ok": True, "data": {"artifacts": {}, "tables": {}, "limitations": ["a limitation"]}},
}


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_renders_without_the_failure_notice(tmp_path, route):
    out = run_scenario(tmp_path, {"hash": f"#/{route}", "api": BASE_API})
    assert out["errors"] == []
    assert "Failed to load module" not in out["viewText"], out["viewText"][:300]
    assert len(out["viewText"].strip()) > 20, "route rendered nothing"


SUPPORT_API = {
    **BASE_API,
    "/api/support/strategies": {"ok": True, "data": [
        {"id": "sup_1", "title": "Visual schedule", "status": "saved", "domain": "Routines"}]},
    "/api/observations": {"ok": True, "data": [
        {"id": "obs_1", "context": "Daily routines", "what_happened": "x"}]},
    "/api/support/suggest": {"ok": True, "data": [
        {"title": "Visual schedule", "domain": "Routines", "why_useful": "w", "how_to_try": "h",
         "how_to_record": "r", "triggered_observation_id": "obs_1"}]},
}


def test_support_studio_buttons_work(tmp_path):
    """C3 end to end: suggestions load into the form and saved strategies can be edited."""
    out = run_scenario(tmp_path, {
        "hash": "#/support", "api": SUPPORT_API, "prompt": "helpful",
        "steps": [
            {"set": ["#supObs", "obs_1"]}, {"click": "#genSug"},
            {"collect": ["use_btn", "#sugList [data-i]", "exists"]},
            {"click": "#sugList [data-i]"},
            {"collect": ["title_after_use", "#stTitle", "value"]},
            {"click": "#stratList [data-edit]"},
            {"collect": ["edit_id", "#stId", "value"]},
            {"click": "#stratList [data-tried]"},
        ]})
    assert out["errors"] == []
    assert "Failed to load module" not in out["viewText"]
    assert out["collected"]["use_btn"] is True
    assert out["collected"]["title_after_use"] == "Visual schedule"
    assert out["collected"]["edit_id"] == "sup_1"
    methods = [(c["method"], c["path"]) for c in out["calls"]]
    assert ("PUT", "/api/support/strategies/sup_1") in methods
    assert ("POST", "/api/support/strategies/sup_1/attempts") in methods


def completed_session(report):
    return {
        "id": "ses_fixture", "status": "completed", "policy": "greedy", "mode": "auto", "asked_count": 10,
        "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
        "result": {"p_hat": 0.8, "decision": "REFERRAL_RECOMMENDED", "asked": 10, "stop_reason": "budget_exhausted",
                   "items_asked": [f"A{i}" for i in range(1, 11)], "trace": [], "counterfactual": None,
                   "explanation": {"contributions": []}, "support_report": report},
    }


def open_result(tmp_path, report):
    sess = completed_session(report)
    api = {**BASE_API,
           "/api/screening/sessions": {"ok": True, "data": [{k: sess[k] for k in ("id", "status", "policy", "mode", "asked_count", "updated_at")}]},
           "/api/screening/sessions/ses_fixture": {"ok": True, "data": sess}}
    return run_scenario(tmp_path, {"hash": "#/screening", "api": api, "steps": [
        {"click": "[data-view]"},
        {"collect": ["cards", "#screeningArea .list .item", "count"]},
        {"collect": ["first_card", "#screeningArea .list .item", "text"]},
        {"collect": ["rec_limits", "#screeningArea .list .item .rec-limitations li", "count"]},
        {"collect": ["report_limits", "#screeningArea .report-limitations li", "count"]},
        {"collect": ["report_limits_text", "#screeningArea .report-limitations", "text"]},
        {"collect": ["disclaimer", "#screeningArea .report-disclaimer", "text"]},
    ]})


def test_result_page_shows_triggering_answers_reason_and_guidance(tmp_path):
    """I1: every suggestion names the items and responses that triggered it."""
    report = make_report()
    out = open_result(tmp_path, report)
    assert out["errors"] == []
    rec0 = report["recommendations"][0]
    assert out["collected"]["cards"] == len(report["recommendations"])
    card = out["collected"]["first_card"]
    assert rec0["title"] in card
    for code in rec0["triggering_question_ids"]:
        assert f"{code} (atypical)" in card
    assert rec0["why_selected"] in card
    assert rec0["description"] in card
    assert rec0["source_attribution"] in card and rec0["review_status"] in card


def test_result_page_shows_limitations_and_the_disclaimer(tmp_path):
    """The limitations the backend already returns must be visible, per suggestion and per report."""
    report = make_report()
    out = open_result(tmp_path, report)
    assert out["collected"]["rec_limits"] == sum(len(r["limitations"]) for r in report["recommendations"])
    assert out["collected"]["report_limits"] == len(report["limitations"]) > 0
    for lim in report["limitations"]:
        assert lim in out["collected"]["report_limits_text"]
    assert report["disclaimer"] in out["collected"]["disclaimer"]


def test_result_page_never_prints_undefined_or_object_placeholders(tmp_path):
    out = open_result(tmp_path, make_report())
    for bad in ("undefined", "[object Object]", "null", "NaN"):
        assert bad not in out["viewText"], bad


def test_result_page_without_recommendations_is_honest_about_it(tmp_path):
    report = make_report()
    report["recommendations"] = []
    out = open_result(tmp_path, report)
    assert out["collected"]["cards"] == 0
    assert "No evidence-linked suggestions" in out["viewText"]
    assert out["collected"]["report_limits"] == len(report["limitations"])


def test_html_in_report_text_is_escaped(tmp_path):
    report = make_report()
    report["recommendations"][0]["limitations"] = ["<img src=x onerror=alert(1)>"]
    report["limitations"] = ["<script>alert(1)</script>"]
    out = open_result(tmp_path, report)
    assert "<img src=x" not in out["viewHtml"] and "<script>alert(1)" not in out["viewHtml"]
    assert "&lt;img src=x" in out["viewHtml"]


# ============================================================ answer flow (audit finding I2) and privacy controls (C5)

def screening_api(question_index, extra=None):
    q = {"index": question_index, "code": f"A{question_index + 1}", "text": f"Question text {question_index + 1}"}
    return {
        **BASE_API,
        "/api/screening/sessions": {"ok": True, "data": [{"id": "ses_1", "status": "active", "policy": "greedy", "mode": "interactive"}]},
        "/api/screening/sessions/ses_1": {"ok": True, "data": {
            "id": "ses_1", "status": "active", "mode": "interactive", "policy": "greedy", "budget": 6, "tau": 0.5,
            "asked_count": 0, "pending_question": q, "belief": 0.3, "trace": []}},
        **(extra or {}),
    }


def posts(out, path):
    return [c for c in out["calls"] if c["path"] == path and c["method"] == "POST"]


def test_each_answer_says_which_question_it_is_for(tmp_path):
    api = screening_api(4, {"/api/screening/answer": {"ok": True, "data": {"finished": True, "result": {}}}})
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": api, "steps": [{"wait": 200}, {"click": '[data-ans="1"]'}]})
    sent = posts(out, "/api/screening/answer")
    assert len(sent) == 1
    assert json.loads(sent[0]["body"]) == {"session_id": "ses_1", "value": 1, "stop": False, "item_index": 4}


def test_a_double_click_sends_one_request_and_disables_the_buttons(tmp_path):
    slow = {"__delay": 400, "body": {"ok": True, "data": {"finished": True, "result": {}}}}
    api = screening_api(4, {"/api/screening/answer": slow})
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": api, "steps": [
        {"wait": 200},
        {"click": '[data-ans="1"]', "after": 10},
        {"collect": ["busy", '#screeningArea[aria-busy="true"]', "exists"]},
        {"collect": ["disabled", "#screeningArea button:disabled", "count"]},
        {"collect": ["buttons", "#screeningArea button", "count"]},
        {"click": '[data-ans="1"]', "after": 10},       # the second click while the first is in flight
        {"click": '[data-ans="0"]', "after": 10},       # even the other button
        {"wait": 600},
    ]})
    assert len(posts(out, "/api/screening/answer")) == 1, out["calls"]
    assert out["collected"]["busy"] is True
    assert out["collected"]["buttons"] == 3 and out["collected"]["disabled"] == 3, "yes, no and stop are all locked"


def test_a_409_resyncs_to_the_current_question_instead_of_failing_silently(tmp_path):
    conflict = {"__status": 409, "body": {"ok": False, "error": "That question has already been answered; reload the session."}}
    api = screening_api(4, {"/api/screening/answer": conflict})
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": api, "steps": [
        {"wait": 200}, {"click": '[data-ans="1"]'}, {"wait": 200},
        {"collect": ["area", "#screeningArea", "text"]},
        {"collect": ["enabled", "#screeningArea button:not(:disabled)", "count"]},
    ]})
    assert out["errors"] == []
    gets = [c for c in out["calls"] if c["path"] == "/api/screening/sessions/ses_1"]
    assert len(gets) >= 2, "the session was re-fetched after the 409"
    assert "A5" in out["collected"]["area"]
    assert out["collected"]["enabled"] >= 2, "buttons are usable again after the failure"


def test_the_stop_button_does_not_need_an_item_index(tmp_path):
    api = screening_api(4, {"/api/screening/answer": {"ok": True, "data": {"finished": True, "result": {}}}})
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": api, "steps": [{"wait": 200}, {"click": "#stopBtn"}]})
    body = json.loads(posts(out, "/api/screening/answer")[0]["body"])
    assert body["stop"] is True and body["item_index"] is None


def test_privacy_page_offers_every_delete_and_calls_the_right_endpoint(tmp_path):
    for button, endpoint in (("#delObs", "/api/privacy/delete_observations"),
                             ("#delSess", "/api/privacy/delete_sessions"),
                             ("#delAll", "/api/privacy/delete_all")):
        out = run_scenario(tmp_path, {"hash": "#/privacy", "api": BASE_API, "steps": [{"wait": 150}, {"click": button}]})
        assert out["errors"] == [], button
        assert len(posts(out, endpoint)) == 1, (button, out["calls"])


def test_privacy_page_states_that_consent_is_a_record_only(tmp_path):
    out = run_scenario(tmp_path, {"hash": "#/privacy", "api": BASE_API})
    assert "does not switch any feature on or off" in out["viewText"]
    assert "Delete all app data" in out["viewText"]


def test_the_edit_form_shows_stored_utc_times_as_local_wall_clock(tmp_path):
    api = {**BASE_API, "/api/observations": {"ok": True, "data": [{
        "id": "obs_1", "context": "Daily routines", "what_happened": "x", "occurred_at": "2026-03-01T10:30:00+00:00",
        "created_at": "2026-03-01T10:30:00+00:00"}]}}
    out = run_scenario(tmp_path, {"hash": "#/context", "api": api, "steps": [
        {"wait": 200}, {"click": "[data-edit]"}, {"collect": ["when", "#obsWhen", "value"]}]})
    assert out["errors"] == []
    from datetime import datetime, timezone
    expected = datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M")
    assert out["collected"]["when"] == expected


# ============================================================ small hardening items

def test_no_inline_event_handlers_remain_in_the_app():
    """Inline handlers block adopting a Content-Security-Policy."""
    html = (REPO / "app" / "static" / "index.html").read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    assert not re.search(r"""\son[a-z]+\s*=\s*["']""", html)
    assert not re.search(r"""\son[a-z]+\s*=\s*["']""", js)


def test_the_app_still_boots_when_localstorage_throws(tmp_path):
    out = run_scenario(tmp_path, {"hash": "#/overview", "api": BASE_API, "breakStorage": True,
                                  "steps": [{"wait": 150}, {"click": "#themeBtn"}]})
    assert out["errors"] == []
    assert "Failed to load module" not in out["viewText"]


def test_new_session_button_has_a_script_handler_not_an_inline_one(tmp_path):
    report = make_report()
    out = open_result(tmp_path, report)
    assert out["errors"] == []
    assert 'id="newSessBtn"' in out["viewHtml"] and "onclick" not in out["viewHtml"]


def research_with(artifact):
    return {**BASE_API, "/api/research/dashboard": {"ok": True, "data": {
        "artifacts": {"step5_policy_benchmark_saudi.json": artifact}, "tables": {}, "limitations": []}}}


def test_research_card_does_not_print_undefined_when_an_artifact_has_no_h1(tmp_path):
    """T10: the step-5 artifact in this repo carries no ``h1`` key; the card used to print 'H1 supported: undefined'."""
    out = run_scenario(tmp_path, {"hash": "#/research", "api": research_with({"policies": ["greedy"], "per_budget": []})})
    assert out["errors"] == []
    assert "does not record an H1 verdict" in out["viewText"]
    assert "undefined" not in out["viewText"].replace("Raw (truncated)", "")


def test_research_card_still_reports_a_recorded_h1(tmp_path):
    art = {"policies": ["greedy"], "h1": {"supported": True, "greedy_brier": 0.075, "exact_fixed_subset_brier": 0.081}}
    out = run_scenario(tmp_path, {"hash": "#/research", "api": research_with(art)})
    assert "H1 supported: true" in out["viewText"] and "0.0750" in out["viewText"]


# ============================================================ adaptive RL policy (decision / stop explanations)

RL_DECISION = {"action": {"type": "ASK", "item": "A6", "item_index": 5}, "faithful_to_agent_internals": True,
               "value_units": "expected screening-decision accuracy minus cost per question",
               "narrative": ["Current estimate 54% after 0 question(s).", "A6 was chosen because it had the highest value."],
               "action_values": [{"action": "A6", "q": 0.81}, {"action": "A9", "q": 0.80}]}
RL_STOP = {"action": {"type": "STOP"}, "stop_reason": "policy_stop",
           "narrative": ["The agent chose to stop after 3 question(s).", "This is a screening estimate, not a diagnosis."],
           "limitations": ["Screening estimate, not a diagnosis."]}


def rl_api(session):
    return {**BASE_API,
            "/api/screening/sessions": {"ok": True, "data": [{k: session[k] for k in ("id", "status", "policy", "mode", "asked_count", "updated_at")}]},
            f"/api/screening/sessions/{session['id']}": {"ok": True, "data": session}}


def test_active_rl_session_shows_the_agents_reason_and_a_skip_option(tmp_path):
    sess = {"id": "ses_rl", "status": "active", "policy": "rl_adaptive", "mode": "interactive", "asked_count": 0,
            "updated_at": "2026-01-01T00:00:00+00:00", "budget": 10, "tau": 0.5, "trace": [], "belief": 0.54,
            "pending_question": {"index": 5, "code": "A6", "text": "question six"}, "decision": RL_DECISION,
            "policy_info": {"policy": "rl_adaptive", "fallback": None, "agent": "rl_value_iteration"}}
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": rl_api(sess), "steps": [
        {"collect": ["skip", "#skipBtn", "exists"]}]})
    assert out["errors"] == []
    assert out["collected"]["skip"] is True
    for line in RL_DECISION["narrative"]:
        assert line in out["viewText"]
    assert "agent decides when to stop" in out["viewText"]


def test_completed_rl_session_explains_why_it_stopped(tmp_path):
    sess = completed_session(make_report())
    sess.update({"id": "ses_rl", "policy": "rl_adaptive", "mode": "interactive"})
    sess["result"].update({"stop_reason": "policy_stop", "stop_explanation": RL_STOP, "asked": 3,
                           "policy_info": {"policy": "rl_adaptive", "agent": "rl_value_iteration",
                                           "artifact_sha256": "abcdef0123456789", "fallback": None}})
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": rl_api(sess), "steps": [{"click": "[data-view]"}]})
    assert out["errors"] == []
    assert "Why the questionnaire stopped" in out["viewText"]
    for line in RL_STOP["narrative"]:
        assert line in out["viewText"]
    assert "rl_value_iteration" in out["viewText"]


def test_fallback_session_is_labelled(tmp_path):
    sess = {"id": "ses_fb", "status": "active", "policy": "greedy", "mode": "interactive", "asked_count": 0,
            "updated_at": "2026-01-01T00:00:00+00:00", "budget": 6, "tau": 0.5, "trace": [], "belief": 0.6,
            "pending_question": {"index": 0, "code": "A1", "text": "question one"}, "decision": None,
            "policy_info": {"policy": "greedy", "fallback": {"requested": "rl_adaptive", "used": "greedy",
                                                             "reason": "artifact missing"}}}
    out = run_scenario(tmp_path, {"hash": "#/screening", "api": rl_api(sess)})
    assert out["errors"] == []
    assert "Fallback in use" in out["viewText"] and "artifact missing" in out["viewText"]
