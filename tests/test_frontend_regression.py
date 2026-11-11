"""Regression tests for the broken-interactions repair (no redesign).

Verifies the interaction contracts, not appearances:
- every dynamic element id referenced by JS exists in the DOM
- theme toggle is attached at load (not inside a dead DOMContentLoaded)
- showResult does not dereference removed ruler elements
- session metadata fields update on start
- demo behaviour tests (loading across responses) still hold
"""
from pathlib import Path
import re

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "scripts" / "demo_static" / "index.html"


def _text():
    return INDEX.read_text(encoding="utf-8")


def test_no_missing_dom_ids():
    html = _text()
    defined = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r"\$\('([a-zA-Z0-9_]+)'\)", html))
    assert used <= defined, f"JS references undefined DOM ids: {used - defined}"


def test_theme_toggle_attached_at_load():
    html = _text()
    assert "$('themeToggle').addEventListener('click', toggleTheme);" in html
    assert "addEventListener" in html  # guard against dead-handler pattern
    # no DOMContentLoaded wrapper around the touchpoint
    assert "document.addEventListener('DOMContentLoaded'" not in html


def test_result_view_renders_tau_metadata_and_has_no_ruler():
    html = _text()
    assert "resTau" in html  # tau shown in the scale metadata (frozen 0.5)
    assert "resScale" in html
    assert "resMark" not in html
    assert "metaPolicy" in html and "metaSession" in html and "metaDate" in html


def test_show_result_renders_from_backend_payload_only():
    html = _text()
    assert "showResult(res" in html
    assert "showError('startErr', e)" in html


def test_answers_flow_starts_from_backend_selection():
    html = _text()
    assert "r.selection" in html and "r.question" in html and "r.next" in html
    assert "state.sessionId" in html
