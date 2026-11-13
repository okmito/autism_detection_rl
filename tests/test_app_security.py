"""Regression tests for the local-server security guards (audit findings C1 and C2).

C1: the app database holds records about real people and must never be tracked by git.
C2: a local server must not be readable or writable by arbitrary websites. The SPA is same-origin,
    so there is no CORS; state changes need JSON + a same-origin (or absent) Origin; and the Host
    header must be a loopback name (DNS rebinding).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EVIL = {"Origin": "http://evil.example"}


# --------------------------------------------------------------------------- C1


def test_gitignore_covers_the_app_database_and_its_sidecars():
    patterns = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    for needed in ("data/*.db", "data/*.db-wal", "data/*.db-shm"):
        assert needed in patterns, f".gitignore must list {needed!r}"


def test_no_database_file_is_tracked_by_git():
    try:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=str(REPO), text=True).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    dbs = [p for p in tracked if re.search(r"\.(db|sqlite3?)(-wal|-shm|-journal)?$", p)]
    assert dbs == [], f"database files must never be tracked: {dbs}"


# --------------------------------------------------------------------------- C2


def test_flask_cors_is_not_used_by_the_app():
    for path in (REPO / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "flask_cors" not in text, f"{path.name} must not import flask_cors"
        assert not re.search(r"\bCORS\(", text), f"{path.name} must not enable CORS"


def test_no_cors_headers_are_ever_sent(api):
    for resp in (
        api.get("/api/health", headers=EVIL),
        api.get("/api/observations", headers=EVIL),
        api.options("/api/observations", headers={**EVIL, "Access-Control-Request-Method": "POST"}),
    ):
        assert "Access-Control-Allow-Origin" not in resp.headers
        assert "Access-Control-Allow-Credentials" not in resp.headers


def _seed_observation(api):
    r = api.post("/api/observations", json={"context": "Daily routines", "what_happened": "seeded for csrf test"})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["data"]["id"]


def test_cross_site_form_post_cannot_delete_observations(api):
    """The CSRF the audit described: a plain <form method=POST> to a body-less endpoint."""
    _seed_observation(api)
    for ctype in ("application/x-www-form-urlencoded", "text/plain", "multipart/form-data; boundary=x"):
        r = api.post("/api/privacy/delete_observations", data="", content_type=ctype, headers=EVIL)
        assert r.status_code == 415, ctype
    r = api.post("/api/privacy/delete_observations")  # no content type at all
    assert r.status_code == 415
    assert len(api.get("/api/observations").get_json()["data"]) == 1, "forged request must not delete data"


def test_json_post_from_a_foreign_origin_is_rejected(api):
    _seed_observation(api)
    r = api.post("/api/privacy/delete_observations", json={}, headers=EVIL)
    assert r.status_code == 403
    assert len(api.get("/api/observations").get_json()["data"]) == 1


def test_forged_consent_record_is_rejected(api):
    r = api.post("/api/consent", json={"scope": "general_research_use", "granted": False}, headers=EVIL)
    assert r.status_code == 403
    assert api.get("/api/consent").get_json()["data"] == []


@pytest.mark.parametrize("method", ["put", "delete"])
def test_other_mutating_methods_are_guarded_too(api, method):
    oid = _seed_observation(api)
    r = getattr(api, method)(f"/api/observations/{oid}", headers=EVIL, json={})
    assert r.status_code == 403
    r = getattr(api, method)(f"/api/observations/{oid}")  # no content type
    assert r.status_code == 415
    assert len(api.get("/api/observations").get_json()["data"]) == 1


def test_same_origin_and_origin_less_json_requests_work(api):
    """What the SPA (same origin) and scripts/curl (no Origin header) actually send."""
    r = api.post("/api/consent", json={"granted": True}, headers={"Origin": "http://localhost"})
    assert r.status_code == 200
    r = api.post("/api/consent", json={"granted": True})
    assert r.status_code == 200
    # The SPA sends the JSON content type even for body-less calls (DELETE, export, bulk delete).
    r = api.delete("/api/observations/does-not-exist", headers={"Content-Type": "application/json"})
    assert r.status_code == 404, "must reach the handler, not be stopped by the guard"
    r = api.post("/api/timeline/export", headers={"Content-Type": "application/json"})
    assert r.status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8010", "127.0.0.1.evil.example", "10.0.0.5:8010"])
def test_dns_rebinding_hosts_are_rejected(api, host):
    r = api.get("/api/observations", headers={"Host": host})
    assert r.status_code == 403
    assert "ok" in r.get_json() and r.get_json()["ok"] is False


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8010", "localhost", "localhost:8010", "[::1]:8010"])
def test_loopback_hosts_are_accepted(api, host):
    assert api.get("/api/health", headers={"Host": host}).status_code == 200
