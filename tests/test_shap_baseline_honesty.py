"""Honesty contract for the deprecated ``shap_baseline`` module — open item 9.

``src/explain/shap_baseline.py`` is *named* SHAP but computes a single-flip
delta. These tests lock the disclosure so the module cannot drift into being
mistaken for Shapley attribution, and assert its actual documented semantics:

1. the ``note`` field discloses that it explains the final prediction only,
   and is not additive;
2. the numbers are exactly single-flip deltas (its true, documented semantics);
3. the new outcome-explanation layer does not import this module.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.state import init_state, update_state
from src.explain.shap_baseline import shap_for_static_subset

REPO_SRC = __import__("pathlib").Path(__file__).resolve().parent.parent / "src" / "explain"


def _state():
    rec = {"item_responses": np.array([1.0, 0.0, 1.0, 0.0]), "label": 1,
           "label_source": "questionnaire",
           "covariates": {"age_band": "1-2", "sex": "M"},
           "missing_mask": np.array([False] * 4)}
    s = init_state(rec)
    s["questions_remaining"] = 4
    s["budget"] = 4
    s = update_state(s, 0, 1, 3)
    s = update_state(s, 1, 0, 2)
    s["questions_remaining"] = 2
    s["budget"] = 4
    return s


def _predictor(state):
    n = sum(int(state["value"][j]) for j in range(state["n"]) if state["mask"][j] == 1)
    return 0.2 + 0.2 * n


def test_note_discloses_scope_and_non_additivity():
    out = shap_for_static_subset([], _state(), _predictor)
    note = out["note"].lower()
    assert "not" in note and "sequential" in note, note
    assert "shap" in note  # the comparison-baseline framing is retained


def test_values_are_exactly_single_flip_deltas():
    """The documented (weak) semantics: base_p minus one-flip probability."""
    s = _state()
    out = shap_for_static_subset([], s, _predictor)
    base_p = out["base_p"]
    assert abs(base_p - _predictor(s)) < 1e-12
    for j in (0, 1):  # the two observed items
        s2 = {"mask": s["mask"].copy(), "value": s["value"].copy(), "n": s["n"],
              "questions_remaining": s["questions_remaining"], "budget": s["budget"]}
        s2["value"][j] = 1 - int(s["value"][j])
        assert abs(out["attributions"][f"A{j+1}"] - (base_p - _predictor(s2))) < 1e-12


def test_module_is_not_used_by_the_new_explanation_layer():
    """The new layer must not silently inherit the non-SHAP attribution.

    Checks for actual *imports* of the deprecated module (prose mentions in
    docstrings that point readers at the deprecation are intentional).
    """
    import re
    src = " ".join(p.read_text() for p in REPO_SRC.glob("*.py")
                   if p.name != "shap_baseline.py")
    imports = re.findall(r"^\s*(?:from\s+\S*shap_baseline\s+import|import\s+\S*shap_baseline)",
                         src, flags=re.M)
    assert not imports, (
        "new explanation modules must not import the deprecated module")
