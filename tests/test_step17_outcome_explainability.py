"""Tests for Step 17 — outcome-explainability artifact and faithfulness helpers.

Two layers:

* **Unit tests of the pure helpers** (deletion/insertion curves, AUC, rank
  correlation, kept_state) on stubs — no artifacts required.
* **Contract tests on the generated artifact** (skipped when
  ``results/outcome_explainability_saudi.json`` has not been generated, the
  repo's established pattern for results-dependent tests).

The regression-guard unit test for the deletion curve exists because a bug in
it once made every step evaluate the fully-deleted state, which produced a
flat curve and a perfect agreement between guided and random orderings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "results" / "outcome_explainability_saudi.json"

from scripts.step17_outcome_explainability import (
    curve_auc, deletion_curve, insertion_curve, kept_state, rank_correlation)
from src.env.state import UNASKED, OBSERVED, init_state, update_state


def _terminal_state(observed: dict, n: int = 10, budget: int = 6):
    rec = {"item_responses": np.zeros(n, dtype=float), "label": 1,
           "label_source": "questionnaire",
           "covariates": {"age_band": "1-2", "sex": "M"},
           "missing_mask": np.zeros(n, dtype=bool)}
    s = init_state(rec)
    s["questions_remaining"] = budget - len(observed)
    s["budget"] = budget
    for k, (j, v) in enumerate(observed.items()):
        s = update_state(s, j, v, budget - k - 1)
        s["questions_remaining"] = budget - k - 1
        s["budget"] = budget
    return s


class _LinearStub:
    """p = 0.1 + 0.08 * sum(observed); strictly additive."""

    def predict_state(self, state):
        s = sum(int(state["value"][j]) for j in range(state["n"])
                if state["mask"][j] == OBSERVED)
        return 0.1 + 0.08 * s

    __call__ = predict_state


# --------------------------------------------------------------------------
# unit tests: pure helpers
# --------------------------------------------------------------------------

def test_deletion_curve_actually_deletes_one_item_per_step():
    s = _terminal_state({0: 1, 1: 1, 2: 1})
    curve = deletion_curve(s, _LinearStub(), order=[0, 1, 2], tau=0.5)
    assert len(curve) == 3
    # strictly additive stub: each deletion removes exactly one contribution
    assert curve[0]["p"] == pytest.approx(0.1 + 0.08 * 2)
    assert curve[1]["p"] == pytest.approx(0.1 + 0.08 * 1)
    assert curve[2]["p"] == pytest.approx(0.1)
    # the curve must not be flat: distinct estimates per step
    assert len({round(c["p"], 12) for c in curve}) == 3
    # |p_t - p_final| increases monotonically for descending-coefficient order
    deltas = [c["abs_delta_from_final"] for c in curve]
    assert deltas == sorted(deltas)


def test_deletion_curve_rejects_non_observed_items():
    s = _terminal_state({0: 1, 1: 1})
    with pytest.raises(ValueError, match="not an observed item"):
        deletion_curve(s, _LinearStub(), order=[0, 5], tau=0.5)


def test_guided_deletion_beats_random_on_additive_stub():
    """On a strictly additive value function the strongest contributor first
    must grow |p - p_final| at least as fast as a random order."""
    from scripts.step17_outcome_explainability import _mean

    s = _terminal_state({0: 1, 1: 1, 2: 1, 3: 0, 4: 1})
    # coefficients are equal here; on equal coefficients ordering cannot beat
    # random, so use the phi-style order anyway and require >= random mean.
    order_guided = [0, 1, 2, 4, 3]
    auc_guided = curve_auc(deletion_curve(s, _LinearStub(), order_guided), "abs_delta_from_final")
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(20):
        perm = [int(x) for x in rng.permutation(order_guided)]
        aucs.append(curve_auc(deletion_curve(s, _LinearStub(), perm), "abs_delta_from_final"))
    assert auc_guided >= _mean(aucs) - 1e-12


def test_insertion_curve_ends_at_final_estimate():
    s = _terminal_state({0: 1, 2: 1, 4: 1})
    order = [0, 2, 4]
    curve = insertion_curve(s, _LinearStub(), order, tau=0.5)
    assert len(curve) == 3
    assert curve[-1]["p"] == pytest.approx(float(_LinearStub().predict_state(s)))
    assert curve[0]["p"] < curve[-1]["p"]
    assert curve[-1]["progress"] == pytest.approx(1.0, abs=1e-9)


def test_curve_auc_and_rank_correlation():
    # the documented convention: both faithfulness curves start at 0
    flat = [{"y": 0.5}] * 4
    assert curve_auc(flat, "y") == pytest.approx(0.4375)   # 0.25 * 0.5*0.5... trapezoid with 0 start
    assert curve_auc(flat, "y", start_value=0.5) == pytest.approx(0.5)
    assert curve_auc([], "y") == 0.0
    assert rank_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert not np.isfinite(rank_correlation([1.0], [1.0]))


def test_kept_state_only_touches_observed_items():
    s = _terminal_state({0: 1, 1: 0, 2: 1})
    rec = {"item_responses": np.array([1, 0, 1, np.nan] + [0] * 6, dtype=float),
           "label": 1, "label_source": "questionnaire",
           "covariates": {"age_band": "1-2", "sex": "M"},
           "missing_mask": np.array([False, False, False, True] + [False] * 6)}
    s2 = init_state(rec)
    s2["questions_remaining"] = 2
    s2["budget"] = 6
    for k, (j, v) in enumerate({0: 1, 2: 1}.items()):
        s2 = update_state(s2, j, v, 6 - k - 1)
        s2["questions_remaining"] = 6 - k - 1
        s2["budget"] = 6
    kept = kept_state(s2, [0])
    assert kept["mask"][0] == OBSERVED and kept["value"][0] == 1
    assert kept["mask"][2] == UNASKED and kept["value"][2] == -1
    assert kept["mask"][3] == 2          # MISSING preserved
    assert kept["mask"][5] == UNASKED    # untouched
    assert kept["n"] == 10


# --------------------------------------------------------------------------
# artifact contract (skipped when not generated)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not ART.exists(), reason="run scripts/step17_outcome_explainability.py")
def test_artifact_contract():
    d = json.loads(ART.read_text())
    assert d["artifact"] == "outcome_explainability_saudi"
    assert d["dataset"] == "saudi"
    assert d["n_episodes"] > 0
    assert d["episode_config"]["tau"] == 0.5
    # efficiency identity held on every session
    assert d["attribution"]["additivity_max_error"] < 1e-9
    # per-item attributions present for all ten items
    assert set(d["attribution"]["abs_phi_mean_per_item"]) == {f"A{j+1}" for j in range(10)}
    cf = d["counterfactuals"]
    assert 0.0 <= cf["found_rate"] <= 1.0
    assert abs(cf["found_rate"] + cf["robust_rate"] - 1.0) < 1e-9
    f = d["faithfulness"]
    for key in ("deletion_auc_guided", "deletion_auc_random_mean",
                "insertion_auc_guided", "insertion_auc_random_mean"):
        assert np.isfinite(f[key]), key
    assert f["n_random_orders"] >= 1
    # overhead is a measured number, finite and positive
    assert 0.0 < d["overhead_ms"]["mean"] < 60_000
    # honesty fields
    assert any("not a diagnosis" in l.lower() for l in d["limitations"])
    assert d["tag"]
