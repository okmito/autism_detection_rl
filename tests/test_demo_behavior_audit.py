"""Regression tests for the 2026-09-30 demo-behaviour audit.

Locks the behaviours investigated in the audit (scripts/demo_app.py):
  1. Greedy policy is deterministic and only picks legal, unasked items.
  2. Random policy with seed=None differs across episodes; int seed = reproducible.
  3. Belief values stay in [0, 1] and are continuous, NOT confined to the
     isotonic step levels {0.0, 0.477, 1.0} reported in the audit.
  4. Platt calibration (now fitted on logits) keeps partial-evidence posteriors
     strictly inside (0, 1).
  5. The referral decision is exactly `p_hat >= tau` - no hidden mapping.
  6. The demo API returns the continuous risk value; the frontend renders it
     without bucketizing/rounding to {0, 60, 100}.

Training here uses a small noisy-synthetic cohort (labels = noisy sum-threshold
rule) so tests are fast and do not depend on data/raw.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.ingest import load_dataset
from src.env.environment import run_episode
from src.env.state import init_state, update_state
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from sklearn.model_selection import StratifiedKFold

N_ITEMS = 10
BUDGET = 6
TAU = 0.5
REPO = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- fixtures
def _mk_record(vals: list[int], label: int) -> dict:
    return {
        "item_responses": np.array(vals, dtype=float),
        "label": int(label),
        "label_source": "questionnaire",
        "covariates": {"age_band": "1-2", "sex": "M"},
        "missing_mask": np.zeros(N_ITEMS, dtype=bool),
        "provenance": "audit_synthetic",
    }


def _synthetic_cohort(n: int = 240, seed: int = 0) -> list[dict]:
    """Noisy deterministic labels (sum >= 4, 8% flipped) - mirrors the
    circular structure of the real Q-CHAT-10 datasets but is not separable,
    so calibrated posteriors stay interior."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        # skew toward mostly-typical answers, like the real cohorts
        vals = (rng.random(N_ITEMS) < 0.25).astype(int).tolist()
        label = int(sum(vals) >= 4)
        if rng.random() < 0.08:
            label = 1 - label
        recs.append(_mk_record(vals, label))
    return recs


@pytest.fixture(scope="module")
def trained_platt():
    recs = _synthetic_cohort()
    y = np.array([r["label"] for r in recs])
    idx = np.arange(len(recs))
    tr, te = next(StratifiedKFold(4, shuffle=True, random_state=0).split(idx, y))
    train = [recs[i] for i in tr]
    val = train[-40:]
    fit = train[:-40]
    pred = MaskedPredictor(n_items=N_ITEMS, hidden=[32, 16], calibration="platt")
    pred.fit(fit, epochs=40, lr=2e-3, batch_size=32, seed=0)
    pred.fit_calibrator(val, method="platt")
    return pred, fit


# ------------------------------------------------- 1. policy behaviour
def test_greedy_policy_deterministic_and_legal(trained_platt):
    pred, train = trained_platt
    rec = _mk_record([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 0)
    seqs = []
    for _ in range(3):
        ep = run_episode(rec, BUDGET, 0, GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)
        seqs.append(ep["items_asked"])
        assert len(set(ep["items_asked"])) == len(ep["items_asked"]), "item repeated in episode"
        assert all(0 <= j < N_ITEMS for j in ep["items_asked"]), "illegal item index"
    assert seqs[0] == seqs[1] == seqs[2], "greedy must be deterministic for identical inputs"


def test_random_policy_differs_across_episodes(trained_platt):
    pred, _ = trained_platt
    rec = _mk_record([0] * N_ITEMS, 0)
    seqs = {tuple(run_episode(rec, BUDGET, 0, RandomPolicy(seed=None), pred, 0.0, tau=TAU)["items_asked"])
            for _ in range(5)}
    assert len(seqs) >= 2, "RandomPolicy(seed=None) must vary across episodes"


def test_random_policy_int_seed_reproducible():
    a = [RandomPolicy(seed=7)({"mask": None, "value": None}, list(range(N_ITEMS))) for _ in range(6)]
    b = [RandomPolicy(seed=7)({"mask": None, "value": None}, list(range(N_ITEMS))) for _ in range(6)]
    assert a == b, "same int seed must give the same sequence (documented semantics)"


# ------------------------------------------------- 2. belief / risk sanity
def test_belief_trajectory_bounded_and_no_repeats(trained_platt):
    pred, _ = trained_platt
    rec = _mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1)
    ep = run_episode(rec, BUDGET, 0, GreedyIGPolicy([_mk_record([0] * N_ITEMS, 0)], N_ITEMS),
                     pred, 0.0, tau=TAU)
    for t in ep["trace"]:
        assert 0.0 <= t["belief_before"] <= 1.0
        assert 0.0 <= t["belief_after"] <= 1.0
    assert 0.0 <= ep["p_hat"] <= 1.0


def test_platt_partial_evidence_stays_interior(trained_platt):
    """One atypical answer = weak evidence -> posterior must NOT saturate.
    (The isotonic step calibrator mapped this to exactly 0.477/1.0.)"""
    pred, _ = trained_platt
    rec = _mk_record([1] + [0] * (N_ITEMS - 1), 0)
    s = init_state(rec)
    s["questions_remaining"] = BUDGET
    s["budget"] = BUDGET
    s = update_state(s, 0, 1, BUDGET - 1)
    s["questions_remaining"] = BUDGET - 1
    s["budget"] = BUDGET
    p = pred(s)
    assert 0.0 < p < 1.0, f"partial-evidence belief saturated to {p}"


def test_risk_not_confined_to_step_levels(trained_platt):
    """Across many answer patterns the risk must take many distinct values -
    rules out the {0.0, 0.477, 1.0} isotonic-step behaviour from the audit."""
    pred, _ = trained_platt
    vals = set()
    for k in range(N_ITEMS + 1):
        for pos in range(0, N_ITEMS, 3):
            v = [0] * N_ITEMS
            for j in range(k):
                v[(pos + j) % N_ITEMS] = 1
            rec = _mk_record(v, 0)
            s = init_state(rec)
            s["questions_remaining"] = 0
            s["budget"] = N_ITEMS
            for j in range(N_ITEMS):
                s = update_state(s, j, v[j], 0)
                s["budget"] = N_ITEMS
                s["questions_remaining"] = 0
            vals.add(round(float(pred(s)), 4))
    assert len(vals) > 8, f"risk values collapsed to {sorted(vals)}"


def test_different_patterns_different_trajectories(trained_platt):
    pred, train = trained_platt
    pa = run_episode(_mk_record([0] * N_ITEMS, 0), BUDGET, 0,
                     GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)["p_hat"]
    pb = run_episode(_mk_record([1] * N_ITEMS, 1), BUDGET, 0,
                     GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)["p_hat"]
    assert abs(pa - pb) > 0.3, "all-typical vs all-atypical must differ substantially"


# ------------------------------------------------- 3. decision logic
def test_decision_is_documented_threshold_rule(trained_platt):
    # environment layer documents: decision = "REFER" if p_hat >= tau else "NO_REFERRAL_INDICATED"
    pred, train = trained_platt
    for vals in ([0] * N_ITEMS, [1] * N_ITEMS, [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]):
        ep = run_episode(_mk_record(vals, 0), BUDGET, 0,
                         GreedyIGPolicy(train, N_ITEMS), pred, 0.0, tau=TAU)
        expected = "REFER" if ep["p_hat"] >= TAU else "NO_REFERRAL_INDICATED"
        assert ep["decision"] == expected, "decision must be exactly p_hat >= tau"
    # demo API layer maps REFER -> REFERRAL_RECOMMENDED, same rule
    demo = _load_demo_app()
    pred2, train2 = trained_platt
    demo.S.update({"predictor": pred2, "train": train2,
                   "test": [_mk_record([1] * N_ITEMS, 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "auto", "policy": "greedy"})
    res = r["result"]
    expected_api = "REFERRAL_RECOMMENDED" if res["p_hat"] >= TAU else "NO_REFERRAL_INDICATED"
    assert res["decision"] == expected_api
    demo.S["sessions"].clear()


# ------------------------------------------------- 4. demo API contract
# ------------------------------------------------- 7. selection diagnostic
def test_selection_diagnostic_agrees_with_policy_choice(trained_platt):
    """The demo's diagnostic recomputes GreedyIGPolicy's IG formula (the policy
    is deliberately left unmodified), so the two must never drift apart: the
    reported IG ordering must pick the same item the policy actually picks.
    """
    demo = _load_demo_app()
    pred, train = trained_platt
    policy = GreedyIGPolicy(train, N_ITEMS)
    for vals in ([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]):
        rec = _mk_record(vals, 0)
        s = init_state(rec)
        s["questions_remaining"] = BUDGET
        s["budget"] = BUDGET
        for _ in range(4):
            legal = [j for j in range(N_ITEMS) if s["mask"][j] == 0]
            if not legal:
                break
            sel = demo.selection_diagnostic(policy, s, legal)
            assert sel["applicable"] is True
            assert sel["support_size"] > 0
            if sel["criterion_vacuous"]:
                # Vacuous criterion => the policy must fall back to the lowest
                # legal index, which is the behaviour the UI now explains.
                assert policy(s, legal) == legal[0], (
                    "diagnostic says the criterion is vacuous but the policy did "
                    "not fall back to the lowest legal index"
                )
            else:
                best = min(sel["top_items"], key=lambda t: t["code"])
                assert policy(s, legal) == int(best["code"][1:]) - 1, (
                    f"diagnostic ranks {[t['code'] for t in sel['top_items']]} "
                    f"first but the policy chose A{policy(s, legal) + 1}"
                )
            a = policy(s, legal)
            s = update_state(s, a, int(rec["item_responses"][a]), 0)
            s["budget"] = BUDGET
            s["questions_remaining"] = 0


def test_selection_diagnostic_flags_vacuous_criterion_on_pure_support(trained_platt):
    """A label-pure support must be reported as a vacuous criterion. This is the
    measured degeneracy the frontend exists to surface (195/195 pure-support
    states in the audit, 61.3% of decision states at B=6)."""
    demo = _load_demo_app()
    pred, train = trained_platt
    policy = GreedyIGPolicy(train, N_ITEMS)
    # a single record is trivially label-pure
    pure = [_mk_record([1] * N_ITEMS, 1)]
    pol_pure = GreedyIGPolicy(pure, N_ITEMS)
    s = init_state(_mk_record([0] * N_ITEMS, 0))
    s["questions_remaining"] = BUDGET
    s["budget"] = BUDGET
    sel = demo.selection_diagnostic(pol_pure, s, list(range(N_ITEMS)))
    assert sel["support_is_pure"] is True
    assert sel["criterion_vacuous"] is True, sel
    assert sel["ig_spread"] is not None and sel["ig_spread"] <= 1e-12
    # and the mixed support is not vacuous
    sel2 = demo.selection_diagnostic(policy, s, list(range(N_ITEMS)))
    assert sel2["support_size"] == len(train)
    assert sel2["support_is_pure"] is False
    assert sel2["criterion_vacuous"] is False, sel2


def test_selection_diagnostic_not_applicable_to_random(trained_platt):
    demo = _load_demo_app()
    s = init_state(_mk_record([0] * N_ITEMS, 0))
    s["questions_remaining"] = BUDGET
    s["budget"] = BUDGET
    sel = demo.selection_diagnostic(RandomPolicy(seed=0), s, list(range(N_ITEMS)))
    assert sel["applicable"] is False
    assert sel["criterion_vacuous"] is None


def test_api_exposes_selection_diagnostic_on_every_step(trained_platt):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train,
                   "test": [_mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "interactive", "policy": "greedy"})
    assert "selection" in r and r["selection"]["applicable"] is True
    a = demo.api_answer({"session_id": r["session_id"], "value": 1})
    assert not a["finished"]
    assert "selection" in a["trace_step"], "each trace step must explain its own pick"
    assert a["selection"]["applicable"] is True
    demo.S["sessions"].clear()


def test_api_auto_reports_selection_summary(trained_platt):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train,
                   "test": [_mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "auto", "policy": "greedy"})
    res = r["result"]
    s = res["selection_summary"]
    assert s["steps_diagnosed"] == res["asked"], (s, res["asked"])
    assert 0 <= s["steps_criterion_vacuous"] <= s["steps_diagnosed"]
    for t in res["trace"]:
        assert "selection" in t, "auto trace steps must carry the diagnostic too"
    demo.S["sessions"].clear()


def test_meta_reports_v6_unsigned_and_h1(trained_platt):
    """The UI must read gate status from artifacts, so it cannot claim a
    threshold exists or that the adaptive policy won."""
    demo = _load_demo_app()
    o = demo.read_offline()
    assert "v6" in o
    assert o["v6"]["signed_off"] is False
    assert o["v6"]["threshold_selected"] is False
    if (demo.REPO / "results" / "evoi_scale_saudi.json").exists():
        assert o["v6"]["no_threshold_selected"] is True


def test_meta_h1_is_populated_from_the_real_artifact(trained_platt):
    """Guards a real bug: read_offline() originally looked for a `test_brier`
    key that step5 does not emit, so every H1 cell came back null and the UI
    silently reported 'not supported' at every budget including B=3/B=4 where
    it IS supported. This asserts the values are present and that the supported
    set matches the audited result (B=3,4 supported; B=5,6 not)."""
    demo = _load_demo_app()
    art = demo.REPO / "results" / "step5_policy_benchmark_saudi.json"
    if not art.exists():
        pytest.skip("step5 artifact not generated")
    h1 = demo.read_offline()["h1"]
    assert h1, "H1 must not be silently empty when the artifact exists"
    for b, row in h1.items():
        assert row["greedy"] is not None, f"B={b} greedy brier missing"
        assert row["fixed"] is not None, f"B={b} fixed-subset brier missing"
        assert 0.0 <= row["greedy"] <= 1.0 and 0.0 <= row["fixed"] <= 1.0
        # a lower brier is better, so "supported" means greedy < fixed
        assert row["supported"] == (row["greedy"] < row["fixed"]), row
    assert h1["3"]["supported"] is True, h1["3"]
    assert h1["4"]["supported"] is True, h1["4"]
    assert h1["5"]["supported"] is False, h1["5"]
    assert h1["6"]["supported"] is False, h1["6"]


# ------------------------------------------------- 8. frontend honesty
def test_frontend_does_not_claim_rl_policy():
    """The demo serves greedy-information-gain and random only. A header that
    advertises reinforcement learning misdescribes what the viewer is seeing."""
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "RL Question Selection" not in html
    assert "Reinforcement-learning policy picks" not in html
    assert "No reinforcement-learning policy is served here" in html


def test_frontend_explains_the_degeneracy():
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "Why does it often ask A1, A2, A3" in html
    assert "label-pure" in html
    assert "criterion_vacuous" in html, "UI must render the vacuity flag from the API"
    assert "information-gain and random baselines" in html


def test_frontend_discloses_the_random_baseline_stops_early(trained_platt):
    """RandomPolicy draws uniformly from `legal`, which contains STOP, so it can
    end an episode early (~9% ask nothing, ~45% reach the full budget of 6).
    Presenting its question count beside greedy's without saying so invites a
    false comparison, so the UI must state it."""
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "Why did Random ask fewer questions" in html
    assert "not</b> comparable with the greedy policy" in html
    assert 'id="randNote"' in html
    assert "$('randNote').style.display" in html, "the note must be wired to the policy toggle"

    # The claim itself is measured here rather than asserted in prose.
    pred, train = trained_platt
    asked = [len(run_episode(_mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1), BUDGET, 0,
                             RandomPolicy(seed=None), pred, 0.0, tau=TAU)["items_asked"])
             for _ in range(60)]
    assert min(asked) < BUDGET, "random baseline should sometimes stop early"
    assert max(asked) <= BUDGET


def test_frontend_states_v6_is_unsigned():
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "V-6 gate is unsigned" in html
    assert "No question-cost threshold is in force" in html
    assert "not supported at B" in html, "UI must be able to show an H1 failure"


def test_frontend_budget_wording_is_consistent():
    """"asked 3 of 10" conflated the item pool with the budget of 6."""
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "of 10 questions" not in html
    assert "of 10 ·" not in html
    assert "of ${BUDGET} allowed" in html


def _load_demo_app():
    spec = importlib.util.spec_from_file_location("demo_app", REPO / "scripts" / "demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["demo_app"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_api_returns_continuous_risk(trained_platt):
    demo = _load_demo_app()
    pred, train = trained_platt
    demo.S.update({"predictor": pred, "train": train, "test": [_mk_record([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], 1)],
                   "records": None, "counter": 0, "sessions": {}})
    r = demo.api_start({"mode": "interactive", "policy": "greedy"})
    assert isinstance(r["belief"], float) and 0.0 <= r["belief"] <= 1.0
    a = demo.api_answer({"session_id": r["session_id"], "value": 1})
    assert not a["finished"]
    assert isinstance(a["belief_after"], float)
    assert 0.0 <= a["belief_after"] <= 1.0
    assert a["next"]["code"] not in ("",) and a["trace_step"]["item"] != a["next"]["code"]
    demo.S["sessions"].clear()


# ------------------------------------------------- 5. frontend rendering
def test_frontend_renders_continuous_percent():
    html = (REPO / "scripts" / "demo_static" / "index.html").read_text(encoding="utf-8")
    assert "(p*100).toFixed(1)" in html, "frontend must render the continuous backend value"
    assert "Math.round(p" not in html, "frontend must not bucketize the risk value"


# ------------------------------------------------- 6. isotonic still available
def test_isotonic_option_still_works(trained_platt):
    pred, val = trained_platt
    pred.fit_calibrator(val, method="isotonic")
    rec = _mk_record([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 1)
    s = init_state(rec)
    s["questions_remaining"] = 0
    s["budget"] = N_ITEMS
    for j in range(N_ITEMS):
        s = update_state(s, j, int(rec["item_responses"][j]), 0)
        s["budget"] = N_ITEMS
        s["questions_remaining"] = 0
    p = pred(s)
    assert 0.0 <= p <= 1.0
