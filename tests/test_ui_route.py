"""The React + TypeScript UI (frontend/, built into app/static/ui/) is served at /ui/ under the same CSP as the legacy page.

The build output is generated (``npm run build`` in frontend/), so the tests that inspect it skip when it is absent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "app" / "static" / "ui"
built = pytest.mark.skipif(not (UI / "index.html").is_file(), reason="React UI not built (cd frontend && npm install && npm run build)")


def test_without_a_build_the_route_says_how_to_build_it(api, monkeypatch, tmp_path):
    from app import server
    monkeypatch.setattr(server.app, "static_folder", str(tmp_path))
    resp = api.get("/ui/")
    assert resp.status_code == 404
    assert "npm run build" in resp.get_json()["error"]
    assert "Content-Security-Policy" in resp.headers


@built
def test_ui_is_served_with_the_security_headers(api):
    for path in ("/ui", "/ui/"):
        resp = api.get(path)
        assert resp.status_code == 200 and b'id="root"' in resp.data
        assert "script-src 'self'" in resp.headers["Content-Security-Policy"]
        assert resp.headers["X-Frame-Options"] == "DENY"


@built
def test_the_built_page_obeys_the_policy():
    """No inline script or style element, no handler attributes, nothing from another origin."""
    html = (UI / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, flags=re.S | re.I)
    assert scripts and all("src=" in attrs and not body.strip() for attrs, body in scripts), "inline <script> block"
    assert not re.search(r"<style\b", html, flags=re.I)
    assert not re.search(r"\son[a-z]+\s*=", html, flags=re.I)
    for ref in re.findall(r'(?:src|href)="([^"#][^"]*)"', html):
        assert ref.startswith("/static/ui/") or ref.startswith("data:"), f"unexpected reference {ref}"
    assert not re.search(r"https?://", html)


@built
def test_the_theme_guard_is_an_external_blocking_script_in_head(api):
    """Inline scripts are forbidden, so the no-flash theme guard must be a classic (non-module, non-deferred) script in <head>."""
    html = (UI / "index.html").read_text(encoding="utf-8")
    head = html.split("</head>")[0]
    tag = re.search(r'<script\b[^>]*src="/static/ui/theme-init\.js"[^>]*>', head)
    assert tag, "theme-init.js must be loaded from <head>"
    assert not re.search(r"\b(?:type=\"module\"|defer|async)\b", tag.group(0))
    assert head.index("theme-init.js") < head.index('type="module"'), "guard must run before the app bundle"
    resp = api.get("/static/ui/theme-init.js")
    assert resp.status_code == 200 and b"data-theme" in resp.data and b"nb_ui_theme" in resp.data


@built
def test_every_asset_the_page_references_is_served(api):
    html = (UI / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="(/static/ui/[^"]+)"', html)
    assert len(refs) >= 3
    for ref in refs:
        assert api.get(ref).status_code == 200, ref
