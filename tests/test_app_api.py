"""Baseline contract tests for the NeuroBridge JSON API (observations, support, timeline, consent).

These pin the behaviour the SPA depends on, so later changes to validation, storage and the data
layer can be made without silently breaking it. Everything runs against a throw-away database
(see the ``api`` fixture); no dataset or model is needed.
"""
from __future__ import annotations

OBS = {
    "context": "Classroom environments",
    "what_happened": "Covered ears when the bell rang",
    "frequency": "daily",
    "difficulty": "loud noise",
    "what_helped": "quiet corner",
    "consistency": "Consistent",
    "notes": "private note",
}


def data(resp, status=200):
    body = resp.get_json()
    assert resp.status_code == status, body
    assert body["ok"] is (status < 400)
    return body.get("data")


def delete(api, url):
    """DELETE as the SPA sends it: JSON content type, no body (the C2 guard requires the header)."""
    return api.delete(url, headers={"Content-Type": "application/json"})


def make_obs(api, **over):
    return data(api.post("/api/observations", json={**OBS, **over}))["id"]


# ------------------------------------------------------------------ health / envelope


def test_health_and_index(api):
    health = data(api.get("/api/health"))
    assert health["status"] == "up"
    assert set(health["screening"]) == {"data_present", "model_loaded", "message"}
    r = api.get("/")
    assert r.status_code == 200 and b"NeuroBridge" in r.data
    r.close()


# ------------------------------------------------------------------ observations


def test_observation_requires_context_and_what_happened(api):
    assert data(api.post("/api/observations", json={"what_happened": "x"}), 400) is None
    assert data(api.post("/api/observations", json={"context": "Daily routines"}), 400) is None


def test_observation_create_list_update_delete(api):
    oid = make_obs(api)
    rows = data(api.get("/api/observations"))
    assert [r["id"] for r in rows] == [oid]
    assert rows[0]["what_happened"] == OBS["what_happened"] and rows[0]["insufficient_info"] == 0

    data(api.put(f"/api/observations/{oid}", json={"what_happened": "changed", "insufficient_info": True}))
    row = data(api.get("/api/observations"))[0]
    assert row["what_happened"] == "changed" and row["insufficient_info"] == 1

    assert data(delete(api, f"/api/observations/{oid}")) == {"deleted": True}
    assert data(api.get("/api/observations")) == []


def test_observation_unknown_id_is_404(api):
    assert data(api.put("/api/observations/nope", json={"notes": "x"}), 404) is None
    assert data(delete(api, "/api/observations/nope"), 404) is None


def test_observation_list_filters_by_context_and_search(api):
    make_obs(api, context="Daily routines", what_happened="Resisted the bedtime routine")
    make_obs(api, context="Sensory environments", what_happened="Loud hand dryer", difficulty="noise")
    assert len(data(api.get("/api/observations"))) == 2
    only = data(api.get("/api/observations?context=Daily routines"))
    assert [r["context"] for r in only] == ["Daily routines"]
    hit = data(api.get("/api/observations?q=HAND DRYER"))
    assert [r["context"] for r in hit] == ["Sensory environments"]


# ------------------------------------------------------------------ support strategies


def test_strategy_lifecycle(api):
    assert data(api.post("/api/support/strategies", json={"domain": "x"}), 400) is None
    sid = data(api.post("/api/support/strategies", json={"title": "Visual schedule", "domain": "Routines"}))["id"]
    rows = data(api.get("/api/support/strategies"))
    assert rows[0]["id"] == sid and rows[0]["status"] == "saved" and rows[0]["source"] == "user"

    data(api.put(f"/api/support/strategies/{sid}", json={"status": "tried", "outcome": "helped"}))
    assert data(api.get("/api/support/strategies"))[0]["status"] == "tried"

    aid = data(api.post(f"/api/support/strategies/{sid}/attempts", json={"outcome": "helped"}))["id"]
    assert aid.startswith("att_")

    assert data(delete(api, f"/api/support/strategies/{sid}")) == {"deleted": True}
    assert data(api.get("/api/support/strategies")) == []


def test_strategy_unknown_id_is_404(api):
    assert data(api.put("/api/support/strategies/nope", json={"notes": "x"}), 404) is None
    assert data(delete(api, "/api/support/strategies/nope"), 404) is None
    assert data(api.post("/api/support/strategies/nope/attempts", json={}), 404) is None


def test_support_suggestions_come_from_the_observation(api):
    oid = make_obs(api, context="Transitions and unexpected changes", what_happened="Upset by the change of plan",
                   difficulty="", what_helped="", notes="")
    sugs = data(api.post("/api/support/suggest", json={"observation_id": oid}))
    assert sugs and all(s["triggered_observation_id"] == oid for s in sugs)
    assert "Advance transition warnings" in {s["title"] for s in sugs}
    assert data(api.post("/api/support/suggest", json={}), 400) is None
    assert data(api.post("/api/support/suggest", json={"observation_id": "nope"}), 404) is None


# ------------------------------------------------------------------ timeline / export


def test_timeline_records_activity_and_filters(api):
    oid = make_obs(api)
    sid = data(api.post("/api/support/strategies", json={"title": "Quiet space"}))["id"]
    events = data(api.get("/api/timeline"))
    kinds = {e["kind"] for e in events}
    assert {"observation", "support"} <= kinds
    assert {e["ref_id"] for e in events} >= {oid, sid}
    assert {e["kind"] for e in data(api.get("/api/timeline?kind=support"))} == {"support"}
    # Search sees the content that the timeline shows for these events.
    assert data(api.get("/api/timeline?q=nothing-matches-this")) == []


def test_export_contains_the_recorded_data(api):
    make_obs(api)
    data(api.post("/api/support/strategies", json={"title": "Quiet space"}))
    out = data(api.post("/api/timeline/export", json={}))
    assert {"exported_at", "observations", "support_strategies", "timeline", "note"} <= set(out)
    assert len(out["observations"]) == 1 and len(out["support_strategies"]) == 1


# ------------------------------------------------------------------ consent / privacy


def test_consent_records(api):
    assert data(api.get("/api/consent")) == []
    out = data(api.post("/api/consent", json={"scope": "store_observations", "granted": True, "details": "ok"}))
    assert out["scope"] == "store_observations" and out["granted"] is True
    data(api.post("/api/consent", json={"scope": "store_observations", "granted": False}))
    rows = data(api.get("/api/consent"))
    assert len(rows) == 2 and {r["granted"] for r in rows} == {0, 1}
    # defaults
    assert data(api.post("/api/consent", json={}))["scope"] == "general_research_use"


def test_privacy_delete_removes_all_observations(api):
    make_obs(api)
    make_obs(api, context="Daily routines")
    assert data(api.post("/api/privacy/delete_observations", json={})) == {"deleted": True}
    assert data(api.get("/api/observations")) == []
    assert any(e["kind"] == "privacy" for e in data(api.get("/api/timeline")))
