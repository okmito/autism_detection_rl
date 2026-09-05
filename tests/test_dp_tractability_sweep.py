"""Test the DP tractability sweep invariants — Step 2.

§14.1, §21: the empirical tractability sweep must obey:
  1. Monotonically non-decreasing in B for fixed N.
  2. Monotonically non-decreasing in N for fixed B (asymptotic upper bound).
  3. Status=optimal for all in-scope (N, B) combinations (no tractability violation).
  4. n_evals >= n_states.
  5. Wall-clock time strictly positive and within the spec bound (24h).

This test does NOT run the sweep itself — it reads the JSON artifact produced
by `scripts/step2_train_and_sweep.py`. This is a stable contract on the
artifact, not on the sweep.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"
SWEEP = RESULTS / "dp_tractability_sweep.json"

pytestmark = pytest.mark.skipif(
    not SWEEP.exists(),
    reason="dp_tractability_sweep.json not generated; run scripts/step2_train_and_sweep.py first",
)


def _load_sweep() -> list[dict]:
    return json.loads(SWEEP.read_text())


def test_sweep_artifact_exists_and_nonempty():
    assert SWEEP.exists()
    rows = _load_sweep()
    assert len(rows) > 0, "sweep artifact is empty"


def test_sweep_all_optimal():
    """No tractability violation should be reported (status=optimal for all in-scope runs)."""
    rows = _load_sweep()
    for r in rows:
        assert r["status"] == "optimal", (
            f"non-optimal run: N={r['N']} B={r['B']} λ={r['lambda_cost']} "
            f"status={r['status']}"
        )


def test_sweep_states_monotone_in_B():
    """For fixed N and λ, n_states is non-decreasing in B."""
    rows = _load_sweep()
    by_N = {}
    for r in rows:
        by_N.setdefault((r["N"], r["lambda_cost"]), []).append(r)
    for key, group in by_N.items():
        group.sort(key=lambda r: r["B"])
        for a, b in zip(group, group[1:]):
            assert b["n_states"] >= a["n_states"], (
                f"n_states not monotone in B at {key}: "
                f"B={a['B']} → {a['n_states']}, B={b['B']} → {b['n_states']}"
            )


def test_sweep_states_monotone_in_N_at_fixed_B():
    """For fixed B and λ, n_states is non-decreasing in N (upper bound)."""
    rows = _load_sweep()
    by_B = {}
    for r in rows:
        by_B.setdefault((r["B"], r["lambda_cost"]), []).append(r)
    for key, group in by_B.items():
        group.sort(key=lambda r: r["N"])
        for a, b in zip(group, group[1:]):
            assert b["n_states"] >= a["n_states"], (
                f"n_states not monotone in N at {key}: "
                f"N={a['N']} → {a['n_states']}, N={b['N']} → {b['n_states']}"
            )


def test_sweep_n_evals_ge_n_states():
    """Each Bellman step evaluates at least one action; n_evals >= n_states."""
    rows = _load_sweep()
    for r in rows:
        assert r["n_evals"] >= r["n_states"], (
            f"n_evals < n_states for N={r['N']} B={r['B']}: "
            f"{r['n_evals']} < {r['n_states']}"
        )


def test_sweep_time_positive():
    rows = _load_sweep()
    for r in rows:
        assert r["time_sec"] > 0.0, f"zero time for N={r['N']} B={r['B']}"


def test_sweep_within_tractability_bound():
    """Wall-clock must respect the spec's 24h bound; in practice minutes."""
    rows = _load_sweep()
    for r in rows:
        assert r["time_sec"] < 24 * 3600, (
            f"tractability time violation at N={r['N']} B={r['B']}: {r['time_sec']}s"
        )


def test_sweep_n_states_within_spec_bound():
    """n_states must be within the spec's 50M hard cap."""
    rows = _load_sweep()
    for r in rows:
        assert r["n_states"] < 50_000_000, (
            f"n_states > 50M at N={r['N']} B={r['B']}: {r['n_states']}"
        )


def test_sweep_budgets_3_through_6():
    rows = _load_sweep()
    budgets = sorted({r["B"] for r in rows})
    assert budgets == [3, 4, 5, 6], f"missing budgets: {budgets}"


def test_sweep_labels_record_source():
    """Every row must record `source` ('real' or 'synthetic')."""
    rows = _load_sweep()
    for r in rows:
        assert r["source"] in ("real", "synthetic"), f"unknown source: {r['source']!r}"
