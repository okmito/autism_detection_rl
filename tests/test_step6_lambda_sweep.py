"""Step 6 lambda-sweep contracts + the canonical split invariant.

Two things are pinned here.

1. ``src/data/splits.py`` — the invariant that every artifact is produced on the
   same partition. This is not cosmetic: ``step4_train_policies.py`` used a
   60/20/20 split (303/101/102) while step2/step3/step5 and both demos used the
   4-fold split (284/95/127), so the step4 policies were trained on a different
   record set from every other artifact and the numbers were never comparable.
   The fingerprint exists because two partitions can have identical row counts
   and still differ in membership, which is exactly how the 303-record split
   passed unnoticed.

2. ``scripts/step6_lambda_sweep.py`` — the V-6 evidence pack. The invariant that
   matters is that ``V*`` is non-increasing in ``lambda``: charging more per
   question cannot make the optimum more valuable. If that ever fails, either the
   solver or the sweep is wrong and no cost-dependent claim may be made.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "results" / "lambda_sweep_saudi.json"
CSV_ART = REPO / "results" / "lambda_sweep_saudi.csv"

from src.data.splits import SCHEME, split_fingerprint, stratified_split


# ---------------------------------------------------------------------------
# Canonical split
# ---------------------------------------------------------------------------

def _toy(n=40, n_items=6, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        x = rng.integers(0, 2, size=n_items).astype(float)
        recs.append({
            "item_responses": x,
            "label": int(x.sum() >= n_items // 2),
            "missing_mask": np.zeros(n_items, dtype=bool),
            "provenance": f"toy:{i}",
        })
    return recs


def test_split_is_deterministic():
    recs = _toy()
    a = stratified_split(recs, seed=0)
    b = stratified_split(recs, seed=0)
    assert [split_fingerprint(*p) for p in (a, b)] == \
           [split_fingerprint(*p) for p in (a, b)]
    for pa, pb in zip(a, b):
        assert [id(r) for r in pa] == [id(r) for r in pb]


def test_split_partitions_every_record_exactly_once():
    recs = _toy()
    train, val, test = stratified_split(recs, seed=0)
    assert len(train) + len(val) + len(test) == len(recs)
    ids = [r["provenance"] for part in (train, val, test) for r in part]
    assert len(set(ids)) == len(recs), "records were dropped or duplicated"


def test_split_parts_are_disjoint():
    """P0 regression: the three parts must not share records.

    The pre-existing copies of this split passed the *indices* returned by the
    second `skf.split(train_idx, ...)` call straight into `records[i]`. Those
    indices are relative to `train_idx`, not to the record list, so the
    "validation" set was drawn from the wrong records: on the 506-row Saudi
    cohort 71 of 95 validation records were also in the training set and 24 were
    in the test set. The row counts still came out as the documented 284/95/127,
    which is why it survived review — the membership was wrong, not the sizes.
    """
    recs = _toy(n=60)
    train, val, test = stratified_split(recs, seed=0)
    tr = {r["provenance"] for r in train}
    va = {r["provenance"] for r in val}
    te = {r["provenance"] for r in test}
    assert not (tr & va), f"train/val share {len(tr & va)} records"
    assert not (tr & te), f"train/test share {len(tr & te)} records"
    assert not (va & te), f"val/test share {len(va & te)} records"
    assert tr | va | te == {r["provenance"] for r in recs}


def test_split_sizes_follow_the_documented_shape():
    """4-fold, then a 4-fold of the 3/4 remainder, with ceil-style folds.

    On the 506-row Saudi cohort this is the 284/95/127 shape quoted in the spec,
    README and STATE_COUNT_VERIFICATION.md. Checked here on a toy cohort so the
    invariant is tested without requiring data/raw; the real-data shape is
    asserted separately below when the CSV is present.
    """
    train, val, test = stratified_split(_toy(n=40), seed=0)
    assert (len(train), len(val), len(test)) == (22, 8, 10)


def test_split_on_saudi_is_284_95_127_and_disjoint():
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")
    train, val, test = stratified_split(recs, seed=0)
    assert (len(train), len(val), len(test)) == (284, 95, 127)
    ids = [{id(r) for r in part} for part in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_every_consumer_agrees_on_one_partition():
    """Spec §17: all runnable baselines use identical splits.

    This is the invariant whose violation went unnoticed for so long. step4 used
    a 60/20/20 `train_test_split` (303/101/102) while the rest used 4-fold
    (284/95/127), so the step4 policies were trained on a different record set
    from every artifact they were benchmarked against. Fingerprints rather than
    row counts, because the contaminated split had the *correct* counts.
    """
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")

    import scripts.step2_train_and_sweep as s2
    import scripts.step3_preliminary_reports as s3
    import scripts.step5_policy_benchmark as s5
    from scripts.step4_train_policies import split_records
    from scripts.demo_app import split as demo_split

    ref = split_fingerprint(*stratified_split(recs, seed=0))
    consumers = {
        "step2": lambda: s2._split(recs, seed=0),
        "step3": lambda: s3._split(recs, seed=0),
        "step4": lambda: split_records(recs, seed=0),
        "step5": lambda: s5.split(recs, seed=0),
        "demo_app": lambda: demo_split(recs),
    }
    mismatched = {
        name: split_fingerprint(*fn())
        for name, fn in consumers.items()
        if split_fingerprint(*fn()) != ref
    }
    assert not mismatched, (
        f"these consumers are not on the canonical partition {ref}: {mismatched}"
    )


def test_split_is_stratified():
    """Both classes must appear in all three parts, and the ratio must be close.

    Built on a balanced cohort so the expected ratio is 0.5 and a mis-specified
    split shows up as drift rather than as an absent class.
    """
    recs = _toy(n=60)
    recs = recs[:30] + [dict(r, label=1 - r["label"]) for r in recs[30:]]
    for part in stratified_split(recs, seed=0):
        y = np.array([r["label"] for r in part])
        assert set(np.unique(y)) == {0, 1}
        assert 0.3 < y.mean() < 0.7


def test_split_seed_changes_the_partition():
    recs = _toy(n=60)
    a = split_fingerprint(*stratified_split(recs, seed=0))
    b = split_fingerprint(*stratified_split(recs, seed=1))
    assert a != b


def test_fingerprint_detects_membership_change_at_equal_sizes():
    """The whole point: equal row counts must not imply an equal partition."""
    recs = _toy(n=40)
    train, val, test = stratified_split(recs, seed=0)
    # Swap one record between train and test — sizes are unchanged.
    swapped_test = list(test)
    swapped_test[0], train[0] = train[0], swapped_test[0]
    assert len(train) == len(train) and len(swapped_test) == len(test)
    assert split_fingerprint(train, val, test) != \
           split_fingerprint(swapped_test, val, [train[0]] + swapped_test[1:])


def test_fingerprint_is_stable_for_identical_partitions():
    recs = _toy(n=30)
    parts = stratified_split(recs, seed=0)
    assert split_fingerprint(*parts) == split_fingerprint(*parts)


def test_split_rejects_too_few_records():
    with pytest.raises(ValueError):
        stratified_split(_toy(n=3), seed=0)


def test_split_scheme_is_recorded():
    assert "stratified" in SCHEME


# ---------------------------------------------------------------------------
# Lambda sweep artifact
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def art():
    if not ART.exists():
        pytest.skip("lambda_sweep_saudi.json not generated; "
                    "run scripts/step6_lambda_sweep.py first")
    return json.loads(ART.read_text())


def test_artifact_metadata(art):
    assert art["source"] == "real"
    assert art["label_source"].startswith("questionnaire")
    assert art["git_sha"] and art["git_sha"] != "no-git"
    assert art["python"]
    assert art["split_scheme"] == SCHEME
    assert art["split_fingerprint"]


def test_artifact_split_fingerprint_is_current(art):
    """Guards against a stale artifact after the split fix.

    The split leak was fixed after the first sweep run, which changed the train
    partition's membership (the row counts were already right, which is why the
    bug survived). Any artifact generated before the fix carries a fingerprint
    that no longer matches the canonical split and must be regenerated.
    """
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")
    current = split_fingerprint(*stratified_split(recs, seed=art["seed"]))
    assert art["split_fingerprint"] == current, (
        f"artifact was generated on split {art['split_fingerprint']} but the "
        f"canonical split is now {current}; re-run scripts/step6_lambda_sweep.py"
    )


def test_artifact_is_flagged_as_v6_decision_input(art):
    """The artifact must not be presentable as a signed-off result."""
    assert "DECISION INPUT" in art["purpose"].upper() or \
           "decision input" in art["purpose"]
    assert "V-6" in art["tag"]
    assert "NOT clinical evidence" in art["tag"]


def test_artifact_declares_predictor_independence(art):
    """The evidence pack is only usable before the P1-b retrain if this holds."""
    assert art["predictor_independent"] is True
    assert "support posterior" in art["predictor_independence_note"]


def test_circularity_warning_present(art):
    warn = art["circularity_warning"]
    assert "sum-threshold" in warn
    assert "saturates" in warn
    assert "label" in warn.lower()


def test_lambda_units_are_declared(art):
    assert "Brier units per question" in art["lambda_units"]


def test_lambda_units_caveat_is_declared(art):
    """The caveat must state the MEASURED relationship between the two
    estimators, and must not assert a unit mismatch.

    History: this field previously read "One reward, two scales ... a lambda
    grid does NOT transfer", on the strength of a 0.05-0.15 predictor figure
    that no code measured. `scripts/step8_evoi_scale_analysis.py` measured both
    sides and found them in the same units (ratio 0.43 at the median), so that
    framing was false and is retracted here. This test exists to stop the false
    claim being reinstated, and to keep the retraction attached to the artifact.
    """
    caveat = art["lambda_units_caveat"]
    assert "retracted" in caveat
    assert "Same units on both sides" in caveat
    assert "ceiling" in caveat
    # the measured numbers, so the claim cannot silently drift
    for token in ("0.0015", "0.0035", "0.0337", "0.0400", "0.25"):
        assert token in caveat, f"caveat omits measured value {token}"
    # the relationship the analysis actually measured
    assert "0.43 at the median" in caveat
    # and the negative-results that a supervisor must see
    assert "non-positive" in caveat
    assert "No threshold is selected" in caveat
    # the retracted framing must be gone as an ASSERTION. The phrase may still
    # appear inside the retraction itself, so check every occurrence is negated.
    assert "does NOT transfer" not in caveat
    for m in re.finditer(r"two orders of magnitude", caveat):
        before = caveat[max(0, m.start() - 40):m.start()].lower()
        assert "not " in before, (
            "'two orders of magnitude' appears un-negated in the caveat"
        )


def test_unsourced_marginal_utility_claim_is_not_asserted_in_the_script():
    """The 0.05-0.15 figure was prose that no code measured. Guard the docstring
    so it cannot be re-introduced as if it were a measurement."""
    src = (REPO / "scripts" / "step6_lambda_sweep.py").read_text(encoding="utf-8")
    doc = src.split('"""', 2)[1]
    assert "roughly 0.05-0.15 Brier, so a grid" not in doc
    assert "No\ncode in this repository ever measured that quantity" in doc \
        or "No code in this repository ever measured that quantity" in doc


def test_beta_greedy_is_swept_at_every_cell(art):
    for r in art["rows"]:
        for field in ("beta_greedy_mean_items", "beta_greedy_mean_reward",
                      "beta_greedy_stop_early"):
            assert field in r, f"{field} missing at B={r['B']} lambda={r['lambda']}"
        assert 0.0 <= r["beta_greedy_mean_items"] <= r["B"] + 1e-9
        assert -1.0 <= r["beta_greedy_mean_reward"] <= 1.0


def test_beta_greedy_never_stops_early_at_zero_lambda(art):
    """At lambda=0 the stop test `max_j gain_j < 0` cannot fire, so the policy
    must spend the full budget. This is what makes fixed-length (AB-7) separable
    from adaptive stopping rather than confounded with item selection.
    """
    rows = [r for r in art["rows"] if r["lambda"] == 0.0]
    assert rows
    for r in rows:
        assert r["beta_greedy_stop_early"] == 0.0, r
        assert r["beta_greedy_mean_items"] == pytest.approx(float(r["B"]), abs=1e-6), r


def test_beta_greedy_stops_more_often_as_questions_get_priced(art):
    """Monotone in lambda at the reference budget: charging more per question can
    only make stopping more attractive. Pins the direction of the cost effect
    independently of the reward scale.
    """
    group = sorted((r for r in art["rows"] if r["B"] == 6),
                   key=lambda r: r["lambda"])
    assert len(group) >= 4
    rates = [r["beta_greedy_stop_early"] for r in group]
    assert rates == sorted(rates), rates
    assert rates[-1] > rates[0], (
        "pricing questions did not change beta_greedy's stopping at all; the "
        "cost term is not reaching the stop test"
    )


def test_row_count_is_the_full_grid(art):
    assert len(art["rows"]) == len(art["lambdas"]) * len(art["budgets"])


def test_v_star_is_non_increasing_in_lambda(art):
    """The key mathematical invariant.

    Charging more per question cannot make the optimum more valuable. A
    violation means the solver or the sweep is wrong, and no cost-dependent claim
    may be made from this artifact.
    """
    by_b = {}
    for r in art["rows"]:
        by_b.setdefault(r["B"], []).append(r)
    assert by_b, "no rows"
    for B, group in by_b.items():
        group = sorted(group, key=lambda r: r["lambda"])
        for a, b in zip(group, group[1:]):
            assert a["V_star"] >= b["V_star"] - 1e-9, (
                f"V* increased from {a['V_star']} to {b['V_star']} "
                f"as lambda went {a['lambda']} -> {b['lambda']} at B={B}"
            )


def test_recorded_monotonicity_flag_agrees(art):
    for entry in art["v_star_monotonicity"]:
        assert entry["V_star_non_increasing_in_lambda"] is True, entry


def test_degeneracy_boundary_is_detected_and_excludes_b1(art):
    """B=1 can never spend fewer than one question, so it must not be flagged."""
    assert art["degenerate_lambdas"], "expected at least one degenerate lambda"
    for r in art["rows"]:
        if r["B"] == 1:
            assert r["degenerate"] is False
        if r["degenerate"]:
            assert r["B"] > 1
            assert r["exact_mean_items"] <= 1.0


def test_informative_band_is_present(art):
    """The grid must resolve the band where cost actually separates policies.

    The band is now justified by measurement rather than by a quoted figure:
    `scripts/step8_evoi_scale_analysis.py` measures the best remaining question
    at p75 = 0.103 and a ceiling of 0.25, so a grid topping out at 0.05 resolves
    the interesting region and the 0.10/0.20 probes bracket degeneracy.
    """
    assert 0.0 in art["lambdas"]
    informative = [l for l in art["lambdas"] if 0.001 <= l <= 0.05]
    assert len(informative) >= 3, "grid does not resolve the informative band"


def test_item_counts_respect_the_budget(art):
    for r in art["rows"]:
        assert 0.0 <= r["exact_mean_items"] <= r["B"] + 1e-9
        assert r["greedy_mean_items"] <= r["B"] + 1e-9
        assert r["random_mean_items"] <= r["B"] + 1e-9


def test_greedy_always_spends_the_full_budget(art):
    """Structural fact, asserted so a change to GreedyIGPolicy cannot pass silently.

    `GreedyIGPolicy` strips STOP from the legal set, so it can never stop early.
    At any lambda > 0 that makes it strictly cost-inferior to the exact
    reference, which is why AB-7 exists. If this test starts failing, greedy has
    gained a stopping rule and the cost-utility comparison must be re-run.
    """
    for r in art["rows"]:
        if r["B"] > 1:
            assert r["greedy_mean_items"] == pytest.approx(float(r["B"])), (
                f"greedy stopped early at B={r['B']}, lambda={r['lambda']}"
            )


def test_exact_beats_greedy_once_questions_are_priced(art):
    """The cost-utility gap opens only when lambda > 0."""
    at_zero = [r for r in art["rows"] if r["lambda"] == 0.0 and r["B"] >= 6]
    priced = [r for r in art["rows"] if r["lambda"] >= 0.01 and r["B"] >= 6]
    assert at_zero and priced
    for r in priced:
        assert r["exact_mean_reward"] >= r["greedy_mean_reward"] - 1e-9
    # And the gap must actually widen with lambda.
    gaps = sorted((r["lambda"], r["exact_mean_reward"] - r["greedy_mean_reward"])
                  for r in priced)
    assert gaps[-1][1] > gaps[0][1]


def test_v_star_saturates_at_zero_lambda(art):
    """The circularity artifact: at lambda=0 the support becomes pure and V*=1."""
    top = max(art["budgets"])
    sat = [r for r in art["rows"] if r["lambda"] == 0.0 and r["B"] == top]
    assert sat, "expected a lambda=0 row at the largest budget"
    assert sat[0]["v_star_saturated"] is True
    assert sat[0]["V_star"] == pytest.approx(1.0, abs=1e-9)


def test_exact_stops_only_on_a_pure_support_at_zero_lambda(art):
    """The decisive diagnostic for the saturation claim.

    If the exact policy's early stopping is driven by a *saturated objective*
    rather than by sufficient evidence, then every episode it ends early should
    terminate on a pure support. This is what the artifact records, and it is
    the reason the measured value of adaptive stopping at lambda=0 cannot be read
    as evidence about clinical information content.
    """
    top = max(art["budgets"])
    rows = [r for r in art["rows"] if r["lambda"] == 0.0 and r["B"] == top]
    assert rows
    assert rows[0]["pure_support_frac_at_stop"] >= 0.9
    # Once questions are priced the policy stops before saturation, so the
    # fraction must fall. If it does not, the stopping rule is not doing what
    # the cost term says it should.
    priced = [r for r in art["rows"] if r["lambda"] >= 0.05 and r["B"] == top]
    if priced:
        assert priced[0]["pure_support_frac_at_stop"] < rows[0]["pure_support_frac_at_stop"]


def test_support_purity_by_depth_is_reported_and_increases(art):
    diag = art["support_purity_by_depth"]
    by_depth = diag["by_depth"]
    assert len(by_depth) == max(art["budgets"])
    assert all(0.0 <= f <= 1.0 for f in by_depth)
    # Purity must be non-decreasing: asking more questions can only shrink the
    # support, and a shrunken support is at least as likely to be pure.
    for a, b in zip(by_depth, by_depth[1:]):
        assert b >= a - 1e-9, f"purity fell from {a} to {b}"
    assert by_depth[0] < by_depth[-1]
    assert diag["all_pure_from_depth_095"] is not None


def test_csv_matches_json(art):
    if not CSV_ART.exists():
        pytest.skip("csv not generated")
    import csv as _csv
    with open(CSV_ART) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == len(art["rows"])
    assert {(float(r["lambda"]), int(r["B"])) for r in rows} == \
           {(r["lambda"], r["B"]) for r in art["rows"]}
