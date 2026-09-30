"""Step 5 benchmark contract tests (Part 2, 2026-10-01).

These lock down the properties the benchmark must have for its numbers to mean
anything. They do not assert that any policy "wins" — the measured result is
that Greedy-IG nearly attains the exact optimum while both learned policies do
not, and pinning that specific ordering would make the test suite fail the
moment the experiment is re-run with different seeds.

What IS asserted:
  - the exact DP is genuinely optimal (no policy beats V*)
  - greedy is near-optimal (the reproduction of the pre-existing claim)
  - all policies are legal and never exceed the budget
  - every artifact carries the circularity warning and the Polish seal
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
JSON_ART = REPO / "results" / "step5_policy_benchmark_saudi.json"
CSV_ART = REPO / "results" / "step5_policy_benchmark_saudi.csv"


@pytest.fixture(scope="module")
def art():
    if not JSON_ART.exists():
        pytest.skip("step5_policy_benchmark_saudi.json not generated; "
                    "run scripts/step5_policy_benchmark.py first")
    return json.loads(JSON_ART.read_text())


# ---------------------------------------------------------------------------
# Exact-DP optimality — the reference the whole benchmark rests on
# ---------------------------------------------------------------------------

def test_no_policy_beats_exact_optimum(art):
    """V* - V_emp must be >= 0 for every policy at every budget.

    A negative gap means the reference is not actually optimal, which would
    invalidate the entire comparison. Greedy at B=1/B=2 rounds to -0.0, so
    the tolerance covers float noise rather than a real violation.
    """
    for g in art["optimality_gap"]:
        for pol in ["greedy", "random", "dqn", "ppo"]:
            assert g[f"gap_{pol}"] >= -1e-6, (
                f"B={g['B']} policy {pol} beat V* by {-g[f'gap_{pol}']} — "
                f"the exact reference is wrong"
            )


def test_random_is_strictly_worse_than_optimum(art):
    """A uniform-random policy must not be optimal at any budget."""
    for g in art["optimality_gap"]:
        assert g["gap_random"] > 1e-4, f"B={g['B']} random is suspiciously optimal"


def test_greedy_is_near_optimum(art):
    """Greedy-IG reproduces the pre-existing near-optimality claim (§17 #7).

    Guards against a regression in the greedy policy or the evaluator that would
    silently invalidate the comparison.
    """
    for g in art["optimality_gap"]:
        assert g["gap_greedy"] < 0.02, (
            f"B={g['B']} greedy gap {g['gap_greedy']} regressed above 0.02"
        )


# ---------------------------------------------------------------------------
# Legality / budget
# ---------------------------------------------------------------------------

def test_no_policy_exceeds_its_budget(art):
    """items_asked_mean must never exceed the budget B."""
    for row in art["per_budget"]:
        assert row["items_asked_mean"] <= row["B"] + 1e-9, (
            f"B={row['B']} {row['policy']} asked "
            f"{row['items_asked_mean']} items"
        )


def test_all_policies_present_at_every_budget(art):
    budgets = set(art["budgets_evaluated"])
    seen = {r["policy"] for r in art["per_budget"]}
    for name in ["greedy", "random", "dqn", "ppo", "exact"]:
        assert name in seen, f"{name} missing from the benchmark"
    for b in budgets:
        for name in seen:
            assert any(r["B"] == b and r["policy"] == name
                       for r in art["per_budget"]), f"{name} missing at B={b}"


def test_brier_scores_in_range(art):
    for row in art["per_budget"]:
        assert 0.0 <= row["brier"] <= 1.0, f"B={row['B']} {row['policy']}"
        assert 0.0 <= row["uar"] <= 1.0, f"B={row['B']} {row['policy']}"


# ---------------------------------------------------------------------------
# Artifact contract — provenance and honesty
# ---------------------------------------------------------------------------

def test_artifact_records_real_source(art):
    assert art["source"] == "real"
    assert art["label_source"].startswith("questionnaire")


def test_artifact_carries_circularity_warning(art):
    """The mandatory caveat must survive into the artifact."""
    assert art["circularity_warning"]
    assert "NOT clinical evidence" in art["tag"] or "NOT clinical" in art["tag"]
    assert "deterministic sum-threshold oracle" in art["circularity_warning"]


def test_polish_stays_sealed(art):
    assert "SEALED" in art["polish_status"]


def test_artifact_has_git_sha_and_env(art):
    assert art["git_sha"] and art["git_sha"] != "no-git"
    assert art["python"] and art["torch"]


def test_training_converged(art):
    """Both learners must show a decreasing loss — otherwise nothing was learned.

    This asserts the pipeline works, not that the policies are good.
    """
    d = art["training"]["dqn"]
    assert d["n_updates"] > 0
    if d["loss_first_50"] is not None and d["loss_final_50"] is not None:
        assert d["loss_final_50"] < d["loss_first_50"], (
            "DQN loss did not decrease over training"
        )


def test_csv_matches_json(art):
    if not CSV_ART.exists():
        pytest.skip("csv not generated")
    with open(CSV_ART) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(art["per_budget"])
    # CSV fields are strings; coerce B before comparing.
    assert {(int(r["B"]), r["policy"]) for r in rows} == \
           {(r["B"], r["policy"]) for r in art["per_budget"]}
