"""Input validation and error handling (audit findings I3, I7, I11).

Client mistakes must be 4xx with a JSON body that names the field; server faults must be an opaque
500 (error id only, no exception text or paths). Nothing here needs the screening dataset.
"""
from __future__ import annotations

import pytest

OBS = {"context": "Daily routines", "what_happened": "Struggled with the transition to bedtime."}


def body(resp):
    return resp.get_json()


def make_obs(api, **over):
    r = api.post("/api/observations", json={**OBS, **over})
    assert r.status_code == 200, body(r)
    return body(r)["data"]["id"]


# ---------------------------------------------------------------- observations


@pytest.mark.parametrize("payload, field", [
    ({"what_happened": 123}, "what_happened"),                 # was a TypeError -> HTML 500
    ({"what_happened": None}, "what_happened"),
    ({"what_happened": "   "}, "what_happened"),
    ({"what_happened": "x" * 4001}, "what_happened"),
    ({"context": None}, "context"),
    ({"context": ["a"]}, "context"),
    ({"context": "c" * 121}, "context"),
    ({"insufficient_info": "yes"}, "insufficient_info"),
    ({"insufficient_info": 1}, "insufficient_info"),
    ({"frequency": {"a": 1}}, "frequency"),
    ({"occurred_at": 20260101}, "occurred_at"),
    ({"session_id": "x" * 65}, "session_id"),
])
def test_bad_observation_fields_are_400_and_named(api, payload, field):
    r = api.post("/api/observations", json={**OBS, **payload})
    assert r.status_code == 400, body(r)
    assert field in body(r)["error"]
    assert api.get("/api/observations").get_json()["data"] == [], "nothing may be stored"


def test_missing_required_fields_are_400(api):
    assert api.post("/api/observations", json={}).status_code == 400
    assert api.post("/api/observations", json={"context": "x"}).status_code == 400


def test_error_messages_do_not_echo_the_submitted_value(api):
    secret = "ZZ-private-note-ZZ"
    r = api.post("/api/observations", json={**OBS, "insufficient_info": secret})
    assert r.status_code == 400
    assert secret not in r.get_data(as_text=True)


@pytest.mark.parametrize("raw", ["[1,2]", '"text"', "42", "{not json"])
def test_non_object_or_malformed_json_is_400(api, raw):
    r = api.post("/api/observations", data=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and body(r)["ok"] is False


def test_update_cannot_null_required_columns(api):
    oid = make_obs(api)
    for payload in ({"context": None}, {"what_happened": None}, {"what_happened": ""}, {"insufficient_info": None}):
        r = api.put(f"/api/observations/{oid}", json=payload)
        assert r.status_code == 400, (payload, body(r))   # was IntegrityError -> 500
    row = api.get("/api/observations").get_json()["data"][0]
    assert row["context"] == OBS["context"] and row["what_happened"] == OBS["what_happened"]


def test_update_ignores_unknown_and_protected_keys(api):
    oid = make_obs(api)
    r = api.put(f"/api/observations/{oid}", json={"id": "hacked", "created_at": "1999", "nonsense": 1, "notes": "ok"})
    assert r.status_code == 200
    row = api.get("/api/observations").get_json()["data"][0]
    assert row["id"] == oid and row["created_at"] != "1999" and row["notes"] == "ok"


def test_update_and_delete_unknown_ids_are_404(api):
    assert api.put("/api/observations/nope", json={"notes": "x"}).status_code == 404
    assert api.delete("/api/observations/nope", json={}).status_code == 404
    assert api.put("/api/support/strategies/nope", json={"notes": "x"}).status_code == 404
    assert api.delete("/api/support/strategies/nope", json={}).status_code == 404
    assert api.post("/api/support/strategies/nope/attempts", json={"status": "tried"}).status_code == 404
    assert api.post("/api/support/suggest", json={"observation_id": "nope"}).status_code == 404


def test_references_must_point_at_existing_records(api):
    assert api.post("/api/observations", json={**OBS, "session_id": "ses_missing"}).status_code == 400
    r = api.post("/api/support/strategies", json={"title": "t", "triggered_observation_id": "obs_missing"})
    assert r.status_code == 400 and "triggered_observation_id" in body(r)["error"]


def test_failed_requests_leave_no_timeline_event(api):
    api.post("/api/observations", json={**OBS, "insufficient_info": "yes"})
    api.put("/api/observations/nope", json={"notes": "x"})
    assert api.get("/api/timeline").get_json()["data"] == []


# ---------------------------------------------------------------- strategies / attempts / consent


@pytest.mark.parametrize("payload", [
    {"title": ""}, {"title": None}, {"title": 5}, {"title": "t" * 201},
    {"title": "ok", "status": "bogus"}, {"title": "ok", "status": None}, {"title": "ok", "domain": ["x"]},
])
def test_bad_strategy_payloads_are_400(api, payload):
    assert api.post("/api/support/strategies", json=payload).status_code == 400


def test_strategy_status_enum_on_create_update_and_attempt(api):
    sid = body(api.post("/api/support/strategies", json={"title": "Visual schedule", "status": "tried"}))["data"]["id"]
    for st in ("saved", "tried", "helpful", "not_helpful", "mixed"):
        assert api.put(f"/api/support/strategies/{sid}", json={"status": st}).status_code == 200
        assert api.post(f"/api/support/strategies/{sid}/attempts", json={"status": st}).status_code == 200
    assert api.put(f"/api/support/strategies/{sid}", json={"status": "great"}).status_code == 400
    assert api.post(f"/api/support/strategies/{sid}/attempts", json={"status": "great"}).status_code == 400
    assert api.put(f"/api/support/strategies/{sid}", json={"title": None}).status_code == 400


@pytest.mark.parametrize("payload", [
    {"scope": "everything"}, {"scope": None}, {"granted": "true"}, {"granted": 1}, {"details": "d" * 2001},
])
def test_bad_consent_payloads_are_400(api, payload):
    assert api.post("/api/consent", json=payload).status_code == 400


def test_consent_defaults_and_roundtrip(api):
    r = api.post("/api/consent", json={"scope": "store_observations", "granted": True, "details": ""})
    assert r.status_code == 200 and body(r)["data"]["granted"] is True
    rows = api.get("/api/consent").get_json()["data"]
    assert rows[0]["scope"] == "store_observations" and rows[0]["details"] is None


# ---------------------------------------------------------------- query parameters


def test_query_parameters_are_bounded(api):
    assert api.get("/api/observations?q=" + "a" * 201).status_code == 400
    assert api.get("/api/timeline?q=" + "a" * 201).status_code == 400
    assert api.get("/api/timeline?limit=abc").status_code == 400
    assert api.get("/api/timeline?limit=0").status_code == 400
    assert api.get("/api/timeline?limit=999999").status_code == 400
    assert api.get("/api/timeline?limit=5").status_code == 200


def test_search_is_parameterised_and_case_insensitive(api):
    make_obs(api, what_happened="Loud ÉCOLE bell")
    make_obs(api, what_happened="Quiet afternoon")
    hits = api.get("/api/observations?q=école").get_json()["data"]
    assert [h["what_happened"] for h in hits] == ["Loud ÉCOLE bell"]
    injection = "x' OR '1'='1"
    assert api.get("/api/observations", query_string={"q": injection}).get_json()["data"] == []
    assert api.get("/api/observations", query_string={"q": "100%_"}).status_code == 200


# ---------------------------------------------------------------- RL lab


@pytest.mark.parametrize("payload", [
    {"budget": 0}, {"budget": 11}, {"budget": "6"}, {"budget": True},
    {"policies": ["greedy"] * 7}, {"policies": "greedy"}, {"policies": [1]}, {"scenario": 5},
])
def test_rl_compare_rejects_unbounded_or_mistyped_input(api, payload):
    assert api.post("/api/rl/compare", json=payload).status_code == 400


def test_rl_replay_unknown_scenario_is_400_not_500(api):
    assert api.post("/api/rl/replay", json={"scenario": "nope"}).status_code == 400


# ---------------------------------------------------------------- generic error handling


def test_unknown_routes_and_wrong_methods_are_json(api):
    r = api.get("/api/does-not-exist")
    assert r.status_code == 404 and body(r)["ok"] is False
    r = api.post("/api/health", json={})
    assert r.status_code == 405 and body(r)["ok"] is False


def test_unexpected_exceptions_become_an_opaque_500_with_an_error_id(api, monkeypatch, caplog):
    from app import server
    marker = "/home/secret/path/db-error-text"

    def boom(*a, **k):
        raise RuntimeError(marker)

    monkeypatch.setattr(server, "list_events", boom)
    with caplog.at_level("ERROR", logger="neurobridge.server"):
        r = api.get("/api/timeline")
    assert r.status_code == 500
    payload = body(r)
    assert payload["ok"] is False and payload["error"] == "Internal error" and len(payload["error_id"]) == 8
    assert marker not in r.get_data(as_text=True), "exception text must not reach the client"
    assert any(payload["error_id"] in rec.getMessage() for rec in caplog.records), "but it must be logged with the id"


def test_a_plain_valueerror_inside_a_service_is_a_server_fault_not_a_400(api, monkeypatch):
    """Only ApiError is a client mistake. A bare ValueError is an internal bug (e.g. the policy-output check)."""
    from app import server

    def bug(*a, **k):
        raise ValueError("internal invariant broken")

    monkeypatch.setattr(server.screening, "list_sessions", bug)
    r = api.get("/api/screening/sessions")
    assert r.status_code == 500 and "invariant" not in r.get_data(as_text=True)


def test_missing_dataset_is_a_503_that_does_not_leak_the_path(api, monkeypatch):
    from app.services import screening

    def missing(*a, **k):
        raise FileNotFoundError("/home/someone/projects/data/raw/secret_dataset.csv")

    monkeypatch.setattr(screening, "_build_predictor", missing)
    monkeypatch.setitem(screening._STATE, "ready", False)
    r = api.post("/api/screening/start", json={})
    assert r.status_code == 503
    text = r.get_data(as_text=True)
    assert "secret_dataset" not in text and "/home/" not in text
    assert "data/raw" in body(r)["error"], "tells the user what to do"


def test_health_reports_readiness_without_loading_the_model(api, monkeypatch):
    from app.services import screening
    monkeypatch.setitem(screening._STATE, "ready", False)
    monkeypatch.setattr(screening, "_build_predictor", lambda *a, **k: pytest.fail("health must not build the model"))
    h = body(api.get("/api/health"))["data"]["screening"]
    assert h["model_loaded"] is False and isinstance(h["data_present"], bool)
