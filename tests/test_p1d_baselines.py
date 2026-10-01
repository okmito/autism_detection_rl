"""P1-d — the reference baselines, and predictor construction reproducibility.

Three things are pinned here.

1. **Predictor construction is reproducible in its own right.** ``MaskedMLP`` builds
   ``nn.Linear`` layers, which draw from torch's *global* RNG. ``step3`` and
   ``step5`` constructed the predictor before seeding torch, so the initial
   weights were unseeded and every run produced a different network — the
   benchmark drifted between runs regardless of ``--seed``. Regression-tested
   below.

2. **The three §17 baselines that were never wired** now are, and H1 ("the
   adaptive policy beats the exact best fixed subset at matched budget") finally
   has a runnable comparator.

3. **The defects fixed in each baseline.** A silent ``except`` that made a
   degraded RFE selection indistinguishable from a converged one; a difficulty
   estimate that was ``<= 0`` for every item; a wholly-unobserved item that
   scored maximum Fisher information; a ``lambda_cost`` that could not change
   the argmax of a fixed-size subset search.
"""
from __future__ import annotations

import hashlib
import itertools
import warnings

import numpy as np
import pytest
import torch

from src.data.splits import stratified_split
from src.env.environment import STOP, run_episode
from src.env.state import init_state, update_state, get_legal_items
from src.models.masked_predictor import MaskedPredictor
from src.policies.irt_cat import IRTCATPolicy
from src.policies.static_fixed import ExactFixedSubsetPolicy
from src.policies.static_rfe import StaticRFEPolicy

N_ITEMS = 10


def _cohort(n=240, n_items=N_ITEMS, seed=0, missing_rate=0.0):
    """A cohort with *heterogeneous* item properties, as real questionnaire data has.

    The item endorsement rates are drawn per item rather than shared. This
    matters: with identical rates and an identical relationship to the label, the
    items genuinely are interchangeable, and any item-information criterion will
    correctly return the same ranking regardless of the answers given. Testing
    state sensitivity against such a cohort would be testing the fixture, not the
    policy.
    """
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.1, 0.5, size=n_items)
    threshold = int(round(np.median(rates * 10)))
    recs = []
    for _ in range(n):
        x = (rng.random(n_items) < rates).astype(float)
        if missing_rate:
            x[rng.random(n_items) < missing_rate] = np.nan
        recs.append({
            "item_responses": x,
            "label": int(np.nansum(x) >= threshold),
            "missing_mask": np.isnan(x),
            "provenance": f"p1d:{n}",
        })
    return recs


def _weights_hash(pred):
    h = hashlib.sha256()
    for k, v in sorted(pred.model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().numpy().tobytes())
    return h.hexdigest()[:24]


def _run(policy, records, B):
    return [run_episode(r, question_budget=B, b_min=0, policy=policy,
                        predictor=lambda s: 0.5, lambda_cost=0.0, tau=0.5)
            for r in records]


# ---------------------------------------------------------------------------
# 1. Predictor construction reproducibility (P0-10)
# ---------------------------------------------------------------------------

def test_predictor_construction_is_seed_deterministic():
    a = _weights_hash(MaskedPredictor(n_items=N_ITEMS, hidden=[16, 8], seed=0))
    b = _weights_hash(MaskedPredictor(n_items=N_ITEMS, hidden=[16, 8], seed=0))
    assert a == b
    c = _weights_hash(MaskedPredictor(n_items=N_ITEMS, hidden=[16, 8], seed=1))
    assert a != c, "different seeds must give different initial weights"


def test_predictor_construction_does_not_disturb_global_torch_rng():
    """Construction must not consume the caller's stream.

    The fix seeds torch only around the `nn.Linear` construction and then
    restores the prior global state, so a script that seeds once at the top of
    `main()` still gets the stream it expects.
    """
    torch.manual_seed(1234)
    expected = torch.randn(5)
    torch.manual_seed(1234)
    MaskedPredictor(n_items=N_ITEMS, hidden=[8], seed=0)
    assert torch.equal(expected, torch.randn(5))


def test_predictor_seed_none_uses_caller_state():
    """`seed=None` opts out and honours whatever the caller set."""
    torch.manual_seed(7)
    a = _weights_hash(MaskedPredictor(n_items=N_ITEMS, hidden=[16, 8], seed=None))
    torch.manual_seed(7)
    b = _weights_hash(MaskedPredictor(n_items=N_ITEMS, hidden=[16, 8], seed=None))
    assert a == b


def test_predictor_version_is_stamped():
    assert MaskedPredictor.VERSION >= 2
    assert MaskedPredictor(n_items=4, hidden=[8], seed=0).version == MaskedPredictor.VERSION


# ---------------------------------------------------------------------------
# 2. ExactFixedSubsetPolicy — the H1 comparator
# ---------------------------------------------------------------------------

def test_fixed_subset_selects_exactly_b_distinct_items():
    recs = _cohort()
    for B in (2, 3, 6):
        p = ExactFixedSubsetPolicy(recs, n_items=N_ITEMS, budget=B)
        assert len(p.best_subset) == B
        assert len(set(p.best_subset)) == B
        assert all(0 <= j < N_ITEMS for j in p.best_subset)
        assert p.fallback_used is False


def test_fixed_subset_actually_beats_random_subsets():
    """The search must be exhaustive, not a heuristic."""
    recs = _cohort()
    B = 3
    p = ExactFixedSubsetPolicy(recs, n_items=N_ITEMS, budget=B)
    assert p.best_value is not None
    # The selected subset must score at least as well as any other subset.
    X = np.stack([r["item_responses"] for r in recs])
    y = np.array([r["label"] for r in recs])

    def utility(subset):
        groups = {}
        for i, row in enumerate(X):
            key = tuple("NA" if np.isnan(row[j]) else int(row[j]) for j in subset)
            groups.setdefault(key, []).append(i)
        brier = 0.0
        for idx in groups.values():
            pp = float(y[idx].mean()) if idx else 0.5
            for i in idx:
                brier += (pp - y[i]) ** 2
        return 1.0 - brier / len(y)

    best = utility(tuple(p.best_subset))
    for subset in itertools.combinations(range(N_ITEMS), B):
        assert best >= utility(subset) - 1e-12, (
            f"selected {p.best_subset} scores {best:.6f} but {subset} scores "
            f"{utility(subset):.6f} — the search is not exhaustive"
        )


def test_fixed_subset_lambda_cost_cannot_change_the_argmax():
    """Every candidate has size B, so a uniform cost is a constant offset.

    An earlier revision added `lambda_cost` to the subset comparison, where it
    silently did nothing. It must not change the selection.
    """
    recs = _cohort()
    plain = ExactFixedSubsetPolicy(recs, n_items=N_ITEMS, budget=4, lambda_cost=0.0)
    pricey = ExactFixedSubsetPolicy(recs, n_items=N_ITEMS, budget=4, lambda_cost=0.5)
    assert plain.best_subset == pricey.best_subset
    # It must still move the reported value, so it is not entirely dead.
    assert pricey.best_value < plain.best_value


def test_fixed_subset_budget_above_n_items_warns_and_records_fallback():
    recs = _cohort(n_items=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = ExactFixedSubsetPolicy(recs, n_items=4, budget=9)
    assert p.fallback_used is True
    assert any("exceeds n_items" in str(w.message) for w in caught)
    assert len(p.best_subset) == 4


def test_fixed_subset_and_rfe_are_legal_and_spend_the_budget():
    recs = _cohort()
    for B in (3, 6):
        for p in (ExactFixedSubsetPolicy(recs, n_items=N_ITEMS, budget=B),
                  StaticRFEPolicy(recs, n_items=N_ITEMS, budget=B)):
            eps = _run(p, recs[:20], B)
            for ep in eps:
                asked = ep["items_asked"]
                assert len(asked) == B, "static baselines must spend the budget"
                assert len(set(asked)) == B, "an item was asked twice"
                assert ep["stop_reason"] == "budget_exhausted"


# ---------------------------------------------------------------------------
# 3. StaticRFEPolicy
# ---------------------------------------------------------------------------

def test_rfe_records_whether_it_fell_back():
    """A degraded selection must be visible, not silent.

    The original used a bare `except Exception:` and fell back to index order
    with no record, so a failed RFE was indistinguishable from a converged one.
    """
    recs = _cohort()
    p = StaticRFEPolicy(recs, n_items=N_ITEMS, budget=4)
    assert p.fallback_used is False
    assert len(p.selected) == 4
    assert p._order == p.selected and p._order is not p.selected, \
        "_order must be a copy, not an alias of selected"


def test_rfe_degenerate_input_warns_instead_of_failing_silently():
    """A single-class cohort makes RFE raise; that must surface."""
    recs = _cohort()
    for r in recs:
        r["label"] = 0  # LogisticRegression cannot fit a single class
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = StaticRFEPolicy(recs, n_items=N_ITEMS, budget=3)
    assert p.fallback_used is True
    assert any("RFE failed" in str(w.message) for w in caught)
    assert len(p.selected) == 3


def test_rfe_missing_mode_is_recorded():
    """The two static baselines use different missing-data conventions.

    That divergence was undocumented and made their results incomparable, so it
    is now recorded on the object and can be stamped into an artifact.
    """
    recs = _cohort(missing_rate=0.2)
    rfe = StaticRFEPolicy(recs, n_items=N_ITEMS, budget=4)
    fixed = ExactFixedSubsetPolicy(recs, n_items=N_ITEMS, budget=4)
    assert rfe.missing_mode == "nan_to_zero"
    assert fixed.missing_mode == "na_is_own_group"
    assert rfe.missing_mode != fixed.missing_mode, \
        "if these ever converge, the note in each docstring is stale"


# ---------------------------------------------------------------------------
# 4. IRTCATPolicy
# ---------------------------------------------------------------------------

def test_irt_difficulty_is_not_all_one_sign():
    """P0 regression: the old estimate gave b <= 0 for 10/10 items with no spread.

    With every difficulty below the origin, `p` sat at 0.92-0.97 for every item
    and `a^2 p (1-p)` was nearly flat, so selection degenerated to the `a^2`
    proxy and barely responded to the answers given.
    """
    recs = _cohort(n=400)
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    b = p.b[p.a > 0]
    assert len(b) > 0
    assert not np.all(b <= 0), "every difficulty is still <= 0"
    # The invariant that matters is spread, not sign: b is measured in units of
    # the latent, so its sign depends on how that latent happens to be scaled.
    # The old bug was a one-sided axis with ~zero range, which collapsed the
    # information function.
    assert np.ptp(b) > 0.5, f"difficulties are nearly identical: {b}"


def test_irt_difficulty_straddles_the_origin_on_saudi():
    """On the real cohort the difficulty axis is centred by construction.

    The pre-fix fit put every item below the origin (measured 10/10 negative,
    range 0.47). A centred axis is what lets Fisher information separate items.
    """
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")
    train, _, _ = stratified_split(recs, seed=0)
    p = IRTCATPolicy(train, n_items=N_ITEMS)
    b = p.b[p.a > 0]
    assert b.min() < 0.0 < b.max(), (
        f"difficulty does not straddle the origin: min={b.min():.3f} "
        f"max={b.max():.3f}"
    )
    assert np.ptp(b) > 0.3, f"difficulty range too narrow: {b}"


def test_irt_uninformative_item_has_zero_information():
    """P0 regression: a wholly-unobserved column used to look maximally informative.

    `continue` left a=1.0, b=0.0, giving I(0)=0.25 with no data behind it. Saudi
    has no missing cells so this was latent, not active — but it would bite on
    any cohort with an unobserved item.
    """
    recs = _cohort()
    for r in recs:
        r["item_responses"] = r["item_responses"].copy()
        r["item_responses"][0] = np.nan
        r["missing_mask"] = r["missing_mask"].copy()
        r["missing_mask"][0] = True
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    assert p.a[0] == 0.0
    assert p._fisher(0.0, 0) == 0.0
    # A real item must carry strictly positive information.
    assert p._fisher(0.0, 1) > 0.0


def test_irt_constant_item_is_uninformative():
    recs = _cohort()
    for r in recs:
        r["item_responses"] = r["item_responses"].copy()
        r["item_responses"][3] = 0.0
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    assert p.a[3] == 0.0


def test_irt_fisher_information_varies_across_items():
    """The old fit was nearly flat across items at any theta."""
    recs = _cohort(n=400)
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    for theta in (-1.0, 0.0, 1.0):
        info = [p._fisher(theta, j) for j in range(N_ITEMS) if p.a[j] > 0]
        assert max(info) - min(info) > 0.05, f"information too flat at theta={theta}"


def test_irt_selection_is_state_sensitive():
    """A CAT that ignores the answers is not adaptive."""
    recs = _cohort()
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    chosen = set()
    for combo in itertools.product([0, 1], repeat=2):
        st = init_state(recs[0])
        st["questions_remaining"] = 6
        st["budget"] = 6
        for k, (j, v) in enumerate([(5, combo[0]), (8, combo[1])]):
            st = update_state(st, j, v, 5 - k)
            st["budget"] = 6
            st["questions_remaining"] = 5 - k
        chosen.add(p(st, get_legal_items(st) + [STOP]))
    assert len(chosen) > 1, f"next item never changed: {chosen}"


def test_irt_is_deterministic():
    recs = _cohort()
    a = IRTCATPolicy(recs, n_items=N_ITEMS)
    b = IRTCATPolicy(recs, n_items=N_ITEMS)
    assert np.allclose(a.a, b.a) and np.allclose(a.b, b.b)
    st = init_state(recs[0])
    st["questions_remaining"] = 6
    st["budget"] = 6
    st = update_state(st, 5, 1, 5)
    st["budget"] = 6
    legal = get_legal_items(st) + [STOP]
    assert a(st, legal) == b(st, legal)


def test_irt_never_stops_early():
    """Spec §17 #8: the generic CAT baseline spends the matched budget.

    Adaptive *stopping* is isolated in AB-7, so the CAT arm must not also stop.
    """
    recs = _cohort()
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    for B in (3, 6):
        eps = _run(p, recs[:15], B)
        for ep in eps:
            assert len(ep["items_asked"]) == B
            assert ep["stop_reason"] == "budget_exhausted"


def test_irt_exposes_per_item_information_for_explanations():
    recs = _cohort()
    p = IRTCATPolicy(recs, n_items=N_ITEMS)
    st = init_state(recs[0])
    st["questions_remaining"] = 6
    st["budget"] = 6
    st = update_state(st, 5, 1, 5)
    st["budget"] = 6
    legal = get_legal_items(st)
    info = p.information(st, legal + [STOP])
    assert set(info) == set(legal)
    assert all(v >= 0 for v in info.values())
    assert max(info, key=info.get) == p(st, legal + [STOP])


# ---------------------------------------------------------------------------
# 5. H1 relationship on real data
# ---------------------------------------------------------------------------

def test_h1_comparator_exists_and_is_a_real_fixed_subset():
    """H1 is written as "adaptive beats the exact best fixed subset at B".

    Before P1-d the comparator had no runnable implementation, so H1 could not
    be evaluated at all. Guard that it stays runnable and genuinely fixed.
    """
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")
    train, _, _ = stratified_split(recs, seed=0)
    p = ExactFixedSubsetPolicy(train, n_items=N_ITEMS, budget=6)
    assert p.fallback_used is False
    assert len(p.best_subset) == 6
    # A fixed subset must produce the same path for every record.
    paths = {tuple(_run(p, [r], 6)[0]["items_asked"]) for r in recs[:25]}
    assert len(paths) == 1, f"a fixed-subset policy followed different paths: {paths}"


def test_rfe_and_exhaustive_search_agree_on_saudi():
    """Recorded as an observation, not a requirement.

    They select the same subset on this cohort, which is a useful consistency
    signal between the cheap and exhaustive selectors. If this starts failing it
    is worth understanding rather than updating the expectation blindly.
    """
    from src.data.ingest import load_dataset
    try:
        recs = load_dataset("saudi")
    except FileNotFoundError:
        pytest.skip("Saudi CSV not present under data/raw/")
    train, _, _ = stratified_split(recs, seed=0)
    for B in (3, 6):
        a = ExactFixedSubsetPolicy(train, n_items=N_ITEMS, budget=B)
        b = StaticRFEPolicy(train, n_items=N_ITEMS, budget=B)
        assert a.best_subset == b.selected, (
            f"B={B}: exhaustive {a.best_subset} vs RFE {b.selected}"
        )
