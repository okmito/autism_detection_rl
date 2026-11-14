"""Security response headers and Content-Security-Policy (audit finding E3). Temp DB; no dataset needed."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


def directives(resp) -> dict:
    policy = resp.headers["Content-Security-Policy"]
    return {d.split()[0]: d.split()[1:] for d in (x.strip() for x in policy.split(";")) if d}


@pytest.mark.parametrize("path", ["/", "/api/health", "/api/privacy/nope", "/static/js/app.js", "/static/css/app.css"])
def test_every_kind_of_response_carries_the_security_headers(api, path):
    r = api.get(path)
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert r.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert r.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert "camera=()" in r.headers["Permissions-Policy"]
    assert "Content-Security-Policy" in r.headers


def test_rejected_requests_also_get_the_headers(api):
    forged_host = api.get("/api/health", headers={"Host": "evil.example"})
    assert forged_host.status_code == 403 and "Content-Security-Policy" in forged_host.headers
    wrong_type = api.post("/api/consent", data="x", headers={"Content-Type": "text/plain"})
    assert wrong_type.status_code == 415 and wrong_type.headers["X-Content-Type-Options"] == "nosniff"
    boom = api.get("/api/definitely-not-a-route")
    assert boom.status_code == 404 and boom.headers["X-Frame-Options"] == "DENY"


def test_policy_blocks_inline_script_and_foreign_origins(api):
    d = directives(api.get("/"))
    assert d["default-src"] == ["'none'"]
    assert d["script-src"] == ["'self'"], "no inline or eval script, no other origin"
    assert "'unsafe-inline'" not in d["style-src"] and "'unsafe-eval'" not in d["script-src"]
    assert d["style-src-attr"] == ["'unsafe-inline'"]
    assert d["connect-src"] == ["'self'"]
    assert d["frame-ancestors"] == ["'none'"] and d["object-src"] == ["'none'"] and d["base-uri"] == ["'none'"]
    for name, sources in d.items():
        assert not any(s.startswith(("http:", "https:", "*")) for s in sources), f"{name} allows a foreign origin"


def test_api_responses_are_not_cacheable_but_static_files_are_untouched(api):
    assert api.get("/api/health").headers["Cache-Control"] == "no-store"
    assert api.get("/api/meta").headers["Cache-Control"] == "no-store"
    assert api.post("/api/consent", data="x", headers={"Content-Type": "text/plain"}).headers["Cache-Control"] == "no-store"
    assert api.get("/static/js/app.js").headers.get("Cache-Control") != "no-store"


def test_the_page_is_compatible_with_the_policy():
    """The policy only helps if the shipped page obeys it. A page that needs inline script would be broken by it."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, flags=re.S | re.I)
    assert scripts and all("src=" in attrs and not body.strip() for attrs, body in scripts), "inline <script> block"
    assert not re.search(r"<style\b", html, flags=re.I), "inline <style> element"
    assert not re.search(r"\son[a-z]+\s*=", html, flags=re.I), "inline event handler attribute"
    for ref in re.findall(r'(?:src|href)="([^"#][^"]*)"', html):
        assert ref.startswith("/") or ref.startswith("#"), f"external reference {ref}"
    js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"\beval\(|new Function\(|setTimeout\(\s*['\"]", js), "eval-like code needs 'unsafe-eval'"
    assert not re.search(r"<style\b|createElement\(['\"]style|<script\b", js), "script/style injection"
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    for name, text in (("index.html", html), ("app.js", js), ("app.css", css)):
        assert not re.search(r"https?://", text, flags=re.I), f"{name} references another origin"
    assert not re.search(r"@import|url\(\s*['\"]?https?:", css, flags=re.I)


def test_the_page_still_loads_its_own_assets(api):
    page = api.get("/")
    assert page.status_code == 200 and b"/static/js/app.js" in page.data
    assert api.get("/static/js/app.js").status_code == 200
    assert api.get("/static/css/app.css").status_code == 200
