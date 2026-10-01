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


#: Every policy the benchmark must produce at every budget. `exact_fixed_subset`,
#: `static_rfe` and `irt_cat` are spec §17 #4/#5/#8; they were written but never
#: instantiated anywhere, so H1 ("adaptive beats the exact best fixed subset at
#: matched budget") had no runnable comparator until P1-d.
#: `beta_greedy` is the P1-f arm: a Beta(1,1) posterior with EVOI selection and
#: cost-based stopping, added so H1 is testable against a non-degenerate adaptive
#: criterion. `greedy` stays in the list and `GreedyIGPolicy` is unchanged.
REQUIRED_POLICIES = [
    "greedy", "random", "dqn", "ppo", "exact",
    "exact_fixed_subset", "static_rfe", "irt_cat",
    "beta_greedy",
]


def test_all_policies_present_at_every_budget(art):
    budgets = set(art["budgets_evaluated"])
    seen = {r["policy"] for r in art["per_budget"]}
    for name in REQUIRED_POLICIES:
        assert name in seen, f"{name} missing from the benchmark"
    for b in budgets:
        for name in seen:
            assert any(r["B"] == b and r["policy"] == name
                       for r in art["per_budget"]), f"{name} missing at B={b}"


def test_h1_comparator_is_present_and_spends_the_full_budget(art):
    """H1 compares the adaptive policy against the exact best *fixed* subset.

    A fixed subset cannot stop early, so it must spend exactly B questions at
    every budget. If it ever reports fewer, the comparator has silently become an
    adaptive policy and H1 would be measuring the wrong thing.
    """
    for row in art["per_budget"]:
        if row["policy"] in ("exact_fixed_subset", "static_rfe", "irt_cat"):
            assert row["items_asked_mean"] == pytest.approx(float(row["B"])), (
                f"{row['policy']} asked {row['items_asked_mean']} at B={row['B']}; "
                f"a fixed/generic-CAT arm must spend the whole budget"
            )
            assert row["stopped_early_frac"] == 0.0


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
    """Both learners must have received a learning signal at all.

    Assertion changed 2026-10-01. This previously required
    ``loss_final_50 < loss_first_50``, which was only satisfiable because the
    rollout collector stored no reward anywhere: DQN was fitting
    ``y = gamma * max_a Q(s', a)`` against an all-zero target, which it can
    drive to zero by shrinking every Q-value. Once ``collect_episode`` started
    attaching the §10 terminal reward the target became a real, high-variance
    Monte Carlo value, and the loss legitimately rises before settling. A
    monotonically falling loss is therefore not a correctness property of DQN and
    is no longer asserted.

    What is asserted instead is the thing that actually matters: that the
    terminal reward reached the buffer. That is verified directly, without the
    benchmark artifact, in ``tests/test_rl_training.py::
    test_collect_episode_stores_the_terminal_reward``.
    """
    d = art["training"]["dqn"]
    assert d["n_updates"] > 0
    assert d["loss_first_50"] is not None
    assert d["loss_final_50"] is not None


def test_stopped_early_frac_is_not_a_constant(art):
    """P0-4 guard: `stopped_early_frac` must be informative again.

    `run_episode` used to label every episode `"policy_stop"`, so this metric was
    1.0 for every policy at every budget and carried no information. It is now
    computed from the real termination cause, so at least one policy must differ
    from 1.0 somewhere in the grid — the exact reference is the one that stops
    early. This guards against the label regressing to a constant.
    """
    fracs = [r["stopped_early_frac"] for r in art["per_budget"]]
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert min(fracs) < 1.0, (
        "stopped_early_frac is 1.0 for every policy/budget; the voluntary-vs-"
        "exhausted distinction has regressed to a constant"
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
