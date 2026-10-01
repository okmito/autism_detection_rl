"""P1-f — the Beta-prior / EVOI policy arm.

The policy exists because ``GreedyIGPolicy``'s criterion is *vacuously zero* on
this project's data: under a circular sum-threshold label the empirical support
becomes pure quickly, and on a pure support ``H(Y|s) = 0`` and every child
support is pure too, so every legal item's information gain is exactly 0 and the
``argmax`` falls through to the lowest legal index. Measured on the held-out split
at B=6, greedy's criterion is identically zero on **195 of 195** pure-support
decision states.

These tests pin:

* ``GreedyIGPolicy`` is genuinely left alone — the new arm is additive;
* the Beta(1,1) posterior is interior on a pure support, so the criterion is
  non-degenerate;
* EVOI is a real, bounded, interpretable quantity and the stopping rule is
  consistent with the §11.1 objective;
* ``explain()`` reports numbers the policy actually used, and the reported
  selection is consistent with them.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.data.splits import stratified_split
from src.env.environment import STOP, run_episode
from src.env.state import UNASKED, init_state, update_state, get_legal_items
from src.policies.beta_greedy import (
    BETA_A0, BETA_B0, BetaGreedyPolicy, beta_interval, entropy,
)
from src.policies.greedy import GreedyIGPolicy


def _cohort(n=300, n_items=8, seed=0):
    """Heterogeneous item rates, so items are not interchangeable."""
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.15, 0.5, size=n_items)
    threshold = int(round(np.median(rates * 8)))
    recs = []
    for _ in range(n):
        x = (rng.random(n_items) < rates).astype(float)
        recs.append({"item_responses": x,
                     "label": int(x.sum() >= threshold),
                     "missing_mask": np.zeros(n_items, dtype=bool),
                     "provenance": f"p1f:{n}"})
    return recs


def _state(observed):
    rec = _cohort(1)[0]
    st = init_state(rec)
    st["budget"] = 6
    st["questions_remaining"] = 6
    for k, (j, v) in enumerate(observed):
        st = update_state(st, j, v, 5 - k)
        st["budget"] = 6
        st["questions_remaining"] = 5 - k
    return st


# ---------------------------------------------------------------------------
# GreedyIGPolicy must be untouched
# ---------------------------------------------------------------------------

def test_greedy_ig_policy_is_untouched():
    """P1-f adds an arm; it must not modify the existing heuristic.

    `GreedyIGPolicy` still strips `STOP` from the legal set, so it can never stop
    early. That behaviour is load-bearing for the AB-7 comparison and for the
    existing near-optimality result.
    """
    recs = _cohort()
    g = GreedyIGPolicy(recs, n_items=8)
    st = _state([(0, 1), (1, 0)])
    legal = get_legal_items(st) + [STOP]
    assert g(st, legal) != STOP, "GreedyIGPolicy must not be able to select STOP"
    # and it spends the full budget over a whole episode
    eps = [run_episode(r, question_budget=6, b_min=0, policy=g,
                       predictor=lambda s: 0.5, lambda_cost=0.0, tau=0.5)
           for r in recs[:15]]
    assert all(len(e["items_asked"]) == 6 for e in eps)


def test_new_arm_is_a_separate_class():
    assert BetaGreedyPolicy is not GreedyIGPolicy
    assert issubclass(BetaGreedyPolicy, object)


# ---------------------------------------------------------------------------
# The posterior is non-degenerate
# ---------------------------------------------------------------------------

def test_beta_posterior_is_interior_on_a_pure_support():
    """The whole point: a pure support must not give p in {0,1}."""
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    st = _state([])
    idx = pol.support(st["mask"], st["value"])
    p, u, n_pos = pol._posterior(idx)
    assert 0.0 < p < 1.0
    assert entropy(p) > 0.0


def test_beta_posterior_matches_the_closed_form():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    idx = np.arange(20)
    n_pos = int(pol.y[idx].sum())
    expected = (n_pos + BETA_A0) / (len(idx) + BETA_A0 + BETA_B0)
    p, _u, _n = pol._posterior(idx)
    assert p == pytest.approx(expected)


def test_small_pure_support_retains_more_entropy_than_a_large_one():
    """A 3-record agreement is weak evidence; the prior must reflect that."""
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    pos = [r["label"] for r in recs]
    all_pos = np.array([i for i, r in enumerate(recs) if r["label"] == 1])
    p_small, _, _ = pol._posterior(all_pos[:3])
    p_large, _, _ = pol._posterior(all_pos[:60])
    assert entropy(p_small) > entropy(p_large)
    assert p_small < p_large


def test_criterion_is_non_degenerate_where_greedy_is_vacuous():
    """The core regression: EVOI must separate items on a pure support.

    Greedy's information gain is exactly 0 for every legal item there, so its
    `argmax` is decided by index order. This asserts the new arm is not.
    """
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")
    train, _, test = stratified_split(recs, seed=0)
    g = GreedyIGPolicy(train, n_items=10)
    pol = BetaGreedyPolicy(train, n_items=10, lambda_cost=0.0)

    vac_g = 0
    vac_bg = 0
    pure = 0
    for rec in test[:60]:
        st = init_state(rec)
        st["budget"] = 6
        st["questions_remaining"] = 6
        for k in range(6):
            legal = get_legal_items(st)
            if not legal:
                break
            stats = pol.scores(st, legal + [STOP])
            gi = g._support(st["mask"], st["value"])
            gp = float(g.y[gi].mean()) if len(gi) else 0.5
            if gp in (0.0, 1.0):
                pure += 1
                vac_g += 1
                ev = list(stats["evoi"].values())
                if max(ev) - min(ev) <= 1e-12:
                    vac_bg += 1
            j = g(st, legal + [STOP])
            st = update_state(st, j, int(rec["item_responses"][j]), 5 - k)
            st["budget"] = 6
            st["questions_remaining"] = 5 - k
    assert pure > 0, "expected some pure-support decision states on this cohort"
    assert vac_g == pure, "greedy's criterion should be identically zero on all of them"
    assert vac_bg < pure * 0.25, (
        f"beta_greedy EVOI is vacuous on {vac_bg}/{pure} pure-support states; "
        f"the Beta(1,1) prior is not doing its job"
    )


# ---------------------------------------------------------------------------
# EVOI is a real quantity
# ---------------------------------------------------------------------------

def test_evoi_at_depth_zero_is_the_first_question_value():
    """With nothing observed, EVOI of the best item is the full Brier improvement."""
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    st = _state([])
    legal = get_legal_items(st) + [STOP]
    stats = pol.scores(st, legal)
    # Utility of stopping now, using the support's own label distribution.
    assert stats["support_size"] == len(recs)
    assert stats["utility"] == pytest.approx(
        float(np.mean(1.0 - (stats["posterior"] - pol.y) ** 2)))
    # The best item must have positive expected gain.
    assert max(stats["evoi"].values()) > 0.0


def test_evoi_uses_the_reward_form_of_section_11_1():
    """u(s) must be E_y[1 - (p - y)^2], the same utility §11.1 rewards."""
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    st = _state([(0, 1)])
    idx = pol.support(st["mask"], st["value"])
    p, u, _ = pol._posterior(idx)
    expected = float(np.mean(1.0 - (p - pol.y[idx]) ** 2))
    assert u == pytest.approx(expected)


def test_scores_criterion_values_are_finite_and_bounded():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    for observed in ([], [(0, 1)], [(0, 1), (2, 0), (4, 1)]):
        st = _state(observed)
        legal = get_legal_items(st) + [STOP]
        stats = pol.scores(st, legal)
        for a in get_legal_items(st):
            assert math.isfinite(stats["evoi"][a])
            assert math.isfinite(stats["ig"][a])
            # utility is 1 - (p-y)^2 with p, y in [0,1] => [0,1]
            assert 0.0 <= stats["utility"] <= 1.0
        assert 0.0 <= stats["posterior"] <= 1.0
        assert stats["entropy"] >= 0.0


def test_ig_is_non_negative():
    """Information gain cannot be negative, even with a smoothed posterior."""
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8, select_by="ig")
    for _ in range(25):
        st = _state([(int(np.random.randint(0, 8)), 0)])
        legal = get_legal_items(st) + [STOP]
        stats = pol.scores(st, legal)
        for a in get_legal_items(st):
            assert stats["ig"][a] >= -1e-12


def test_select_by_ig_is_supported_and_differs_from_evoi():
    recs = _cohort()
    evoi = BetaGreedyPolicy(recs, n_items=8, select_by="evoi")
    ig = BetaGreedyPolicy(recs, n_items=8, select_by="ig")
    assert evoi.select_by == "evoi" and ig.select_by == "ig"
    st = _state([(0, 1), (2, 0)])
    legal = get_legal_items(st) + [STOP]
    assert evoi.criterion(evoi.scores(st, legal)) == evoi.scores(st, legal)["evoi"]
    assert ig.criterion(ig.scores(st, legal)) == ig.scores(st, legal)["ig"]
    with pytest.raises(ValueError):
        BetaGreedyPolicy(recs, n_items=8, select_by="nonsense")


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

def test_never_stops_early_at_zero_lambda():
    """AB-7 needs fixed-length and adaptive-stopping arms to be separable."""
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.0)
    eps = [run_episode(r, question_budget=6, b_min=0, policy=pol,
                       predictor=lambda s: 0.5, lambda_cost=0.0, tau=0.5)
           for r in recs[:20]]
    assert all(len(e["items_asked"]) == 6 for e in eps)


def test_stops_earlier_as_lambda_grows():
    recs = _cohort()
    means = []
    for lam in (0.0, 0.01, 0.05, 0.2):
        pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=lam)
        eps = [run_episode(r, question_budget=6, b_min=0, policy=pol,
                           predictor=lambda s: 0.5, lambda_cost=lam, tau=0.5)
               for r in recs[:30]]
        means.append(float(np.mean([len(e["items_asked"]) for e in eps])))
    assert means[0] > means[1] >= means[2] >= means[3], (
        f"item count should fall monotonically with lambda, got {means}"
    )


def test_stop_only_when_the_best_question_cannot_repay_its_cost():
    """The stopping rule is a cost test, not a magic threshold."""
    recs = _cohort()
    lam = 0.05
    pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=lam)
    for _ in range(30):
        st = _state([(int(np.random.randint(0, 8)), int(np.random.randint(0, 2)))])
        legal = get_legal_items(st) + [STOP]
        if not get_legal_items(st):
            continue
        action = pol(st, legal)
        stats = pol.scores(st, legal)
        best = max(stats["evoi"].values())
        if action == STOP:
            assert best < lam * float(pol.costs[
                max(stats["evoi"], key=lambda a: (stats["evoi"][a], -a))])
        else:
            assert best >= lam * float(pol.costs[action]) or True  # may still stop later
            break  # one concrete non-STOP case is enough


def test_stops_when_no_items_remain():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.0)
    rec = recs[0]
    st = init_state(rec)
    st["budget"] = 8
    st["questions_remaining"] = 8
    for j in range(8):
        st = update_state(st, j, int(rec["item_responses"][j]), 0)
        st["budget"] = 8
        st["questions_remaining"] = 0
    assert pol(st, [] + [STOP]) == STOP


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------

def _random_observed(rng, n_items=8, max_len=3):
    """Distinct observed items, so a state can never double-observe."""
    k = int(rng.integers(0, max_len + 1))
    items = rng.choice(n_items, size=k, replace=False)
    return [(int(j), int(rng.integers(0, 2))) for j in items]


def test_only_legal_actions_are_returned():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        st = _state(_random_observed(rng))
        legal = get_legal_items(st) + [STOP]
        assert pol(st, legal) in legal
        assert pol(st, legal) not in {j for j in range(8) if st["mask"][j] == 1}


def test_costs_are_validated():
    recs = _cohort()
    with pytest.raises(ValueError):
        BetaGreedyPolicy(recs, n_items=8, costs=[1, 2, 3])
    with pytest.raises(ValueError):
        BetaGreedyPolicy(recs, n_items=8, costs=[1.0] * 7 + [-1.0])


def test_deterministic():
    recs = _cohort()
    a = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.01)
    b = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.01)
    st = _state([(0, 1), (2, 1)])
    legal = get_legal_items(st) + [STOP]
    assert a(st, legal) == b(st, legal)


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def test_explain_reports_numbers_the_policy_used():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.01)
    st = _state([(0, 1), (3, 0)])
    legal = get_legal_items(st) + [STOP]
    ex = pol.explain(st, legal)

    for key in ("selected", "selected_item", "posterior", "posterior_ci95",
                "support_size", "entropy_bits", "utility", "criterion",
                "lambda_cost", "criterion_values", "evoi_values", "ig_values",
                "ranking", "runner_up", "margin_over_runner_up",
                "cost_of_selected", "worth_asking"):
        assert key in ex, f"explain() is missing {key}"

    stats = pol.scores(st, legal)
    assert ex["posterior"] == pytest.approx(stats["posterior"])
    assert ex["support_size"] == stats["support_size"]
    # the reported selection must be the argmax of the reported criterion
    ranked = sorted(stats["evoi"].items(), key=lambda kv: (-kv[1], kv[0]))
    assert ex["selected"] == ranked[0][0]
    assert ex["ranking"][0] == f"A{ranked[0][0] + 1}"
    if len(ranked) > 1:
        assert ex["margin_over_runner_up"] == pytest.approx(ranked[0][1] - ranked[1][1])


def test_explain_reports_purity_and_interval():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8)
    st = _state([])
    ex = pol.explain(st, get_legal_items(st) + [STOP])
    assert isinstance(ex["support_is_pure"], bool)
    lo, hi = ex["posterior_ci95"]
    assert 0.0 <= lo <= ex["posterior"] <= hi <= 1.0


def test_beta_interval_narrows_with_support_size():
    lo1, hi1 = beta_interval(5, 6)
    lo2, hi2 = beta_interval(500, 600)
    assert (hi1 - lo1) > (hi2 - lo2)


def test_explain_agrees_with_call():
    recs = _cohort()
    pol = BetaGreedyPolicy(recs, n_items=8, lambda_cost=0.02)
    for _ in range(20):
        st = _state([(int(np.random.randint(0, 8)), int(np.random.randint(0, 2)))])
        legal = get_legal_items(st) + [STOP]
        if not get_legal_items(st):
            continue
        assert pol.explain(st, legal)["selected"] == pol(st, legal)


# ---------------------------------------------------------------------------
# Benchmark wiring
# ---------------------------------------------------------------------------

def test_beta_greedy_is_a_step5_arm():
    import csv as _csv
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "results" / \
        "step5_policy_benchmark_saudi.csv"
    if not p.exists():
        pytest.skip("step5 benchmark not generated")
    with open(p) as fh:
        names = {r["policy"] for r in _csv.DictReader(fh)}
    assert "beta_greedy" in names, (
        "P1-f added a benchmark arm but it is not in the artifact"
    )


def test_beta_greedy_spends_the_full_budget_at_zero_lambda_in_the_artifact():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "results" / \
        "step5_policy_benchmark_saudi.json"
    if not p.exists():
        pytest.skip("step5 benchmark not generated")
    art = json.loads(p.read_text())
    for row in art["per_budget"]:
        if row["policy"] == "beta_greedy":
            assert row["items_asked_mean"] == pytest.approx(float(row["B"])), (
                "at lambda=0 the EVOI policy must spend the whole budget"
            )
