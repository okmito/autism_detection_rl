"""Step 7 contracts — the bounded RL-collapse diagnosis.

What is pinned here is deliberately narrow. The diagnosis itself is a
measurement, and measurements change as seeds and budgets change; the tests below
pin the *structural* claims that must not silently regress:

* the two objectives really are different code paths (`bootstrap` vs `returns`);
* a small reproduction of the diagnosis actually shows the effect, so the claim
  in ``RL_TRAINING_REPORT.md`` is not taken on trust;
* the artifact carries the provenance needed to interpret it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "results" / "rl_diagnosis_saudi.json"
CSV_ART = REPO / "results" / "rl_diagnosis_saudi.csv"

from scripts.step7_rl_diagnosis import train_dqn_curve


@pytest.fixture(scope="module")
def art():
    if not ART.exists():
        pytest.skip("rl_diagnosis_saudi.json not generated; "
                    "run scripts/step7_rl_diagnosis.py first")
    return json.loads(ART.read_text())


# ---------------------------------------------------------------------------
# The two objectives are genuinely different
# ---------------------------------------------------------------------------

def test_bootstrapped_and_returns_targets_differ():
    """`bootstrap=True` adds a future term; `bootstrap=False` must not.

    If these were equivalent, the whole diagnosis would be measuring nothing.
    """
    import numpy as np
    import torch
    from src.policies.dqn import DQNPolicy, STOP
    from src.policies.replay import ReplayBuffer
    from src.env.state import init_state, update_state

    def batch(n=16):
        buf = ReplayBuffer(capacity=100, seed=0)
        st = init_state({"item_responses": np.zeros(4, dtype=float), "label": 0,
                         "missing_mask": np.zeros(4, dtype=bool)})
        # 4 items, so items 0..3 then STOP; each item observed exactly once.
        for i in range(3):
            nxt = update_state(st, i, i % 2, 3)
            nxt["budget"] = 4
            buf.add(st, i, 0.0, nxt, False, [0, 1, 2, 3, STOP])
            st = nxt
        final = update_state(st, 3, 1, 2)
        final["budget"] = 4
        buf.add(st, 3, 0.0, final, False, [STOP])
        buf.add(final, STOP, 0.75, None, True, None)
        return buf.all()

    data = batch()
    torch.manual_seed(0)
    a = DQNPolicy(n_items=4, seed=0)
    torch.manual_seed(0)
    b = DQNPolicy(n_items=4, seed=0)
    with_boot = a.train_step(list(data), gamma=0.99, bootstrap=True)
    without = b.train_step(list(data), gamma=0.99, bootstrap=False)
    assert with_boot != pytest.approx(without), (
        "bootstrap=True and bootstrap=False produced identical losses, so the "
        "returns arm of the diagnosis is not actually disabling the bootstrap"
    )


def test_returns_objective_reaches_behaviour_faster_than_bootstrap():
    """The core claim, reproduced at small scale.

    §11.1 emits a reward only at the end of an episode, so a B-step episode with
    immediate rewards needs a B-step bootstrap chain. This asserts the effect
    that motivates the `returns` arm: at a small episode budget the return-to-go
    objective has already stopped collapsing, while the bootstrapped one has
    not.

    Kept small (a few hundred episodes on synthetic data) so the suite stays
    fast; the full sweep lives in the artifact.
    """
    import numpy as np
    from src.data.splits import stratified_split
    from src.env.environment import STOP, run_episode
    from src.env.state import init_state, update_state
    from src.models.masked_predictor import MaskedPredictor
    from src.policies.replay import ReplayBuffer
    from src.policies.dqn import DQNPolicy
    from scripts.step4_train_policies import collect_episode

    def cohort(n=160, n_items=6, seed=0):
        rng = np.random.default_rng(seed)
        recs = []
        for _ in range(n):
            x = (rng.random(n_items) < 0.35).astype(float)
            recs.append({"item_responses": x,
                         "label": int(x.sum() >= 3),
                         "missing_mask": np.zeros(n_items, dtype=bool),
                         "provenance": "s7"})
        return recs

    recs = cohort()
    train, val, test = stratified_split(recs, seed=0)
    pred = MaskedPredictor(n_items=6, hidden=[16, 8], calibration="isotonic", seed=0)
    pred.fit(train, epochs=5, lr=1e-3, batch_size=32, seed=0)
    pred.fit_calibrator(val, method="isotonic")

    def mean_items(objective, episodes, seed=0):
        buf = ReplayBuffer(capacity=50_000, seed=seed)
        pol = DQNPolicy(n_items=6, seed=seed)
        for ep in range(episodes):
            collect_episode(train[ep % len(train)], 4, 0, pol,
                            pred.predict_state, buf, 6, 0.0, 0.5,
                            epsilon=1.0 + (0.05 - 1.0) * ep / max(episodes - 1, 1),
                            gamma=1.0 if objective == "returns" else None)
            if len(buf) >= 32:
                pol.train_step(buf.sample(32), gamma=0.99,
                               bootstrap=objective != "returns")
            if (ep + 1) % 100 == 0:
                pol.update_target()
        out = []
        for r in test[:20]:
            ep_ = run_episode(r, question_budget=4, b_min=0, policy=pol,
                              predictor=pred.predict_state, lambda_cost=0.0, tau=0.5)
            out.append(len(ep_["items_asked"]))
        return float(np.mean(out))

    boot = mean_items("bootstrap", 400)
    ret = mean_items("returns", 400)
    assert ret > boot, (
        f"return-to-go asked {ret:.2f} items vs bootstrap {boot:.2f}; the "
        f"diagnosis claims the opposite ordering at a small budget"
    )


# ---------------------------------------------------------------------------
# Artifact contracts
# ---------------------------------------------------------------------------

def test_artifact_metadata(art):
    assert art["source"] == "real"
    assert art["label_source"].startswith("questionnaire")
    assert art["git_sha"] and art["git_sha"] != "no-git"
    assert art["python"] and art["torch"]
    assert art["split_fingerprint"]
    assert art["predictor_version"] >= 2


def test_artifact_states_hypothesis_and_objectives(art):
    assert "bootstrap" in art["hypothesis_under_test"].lower()
    assert set(art["objectives"]) == {"bootstrap", "returns"}
    assert "NOT clinical evidence" in art["tag"]


def test_artifact_covers_both_objectives_over_the_budget_grid(art):
    for objective in ("bootstrap", "returns"):
        got = {r["episodes"] for r in art["rows"] if r["objective"] == objective}
        assert got == set(art["episode_budgets"]), (
            f"{objective} missing budgets: {set(art['episode_budgets']) - got}"
        )


def test_artifact_reports_seed_variance(art):
    for r in art["rows"]:
        assert r["seeds"] == len(art["seeds"])
        assert r["items_sd"] >= 0.0
        assert r["uar_sd"] >= 0.0
    assert len(art["seeds"]) >= 3, "a single seed cannot support a variance claim"


def test_verdict_records_both_coverage_and_behaviour(art):
    """Both metrics must be present, because they disagree.

    Coverage keeps rising long after behaviour plateaus, so reporting coverage
    alone would have concluded the policies were "still improving".
    """
    v = art["verdict"]
    assert "criterion_correction" in v
    assert "pre_committed_criterion" in v
    for objective in ("bootstrap", "returns"):
        e = v[objective]
        assert "coverage_still_growing" in e
        assert "behaviour_plateaued" in e
        assert "behaviour_uar_delta_last_two_budgets" in e
        assert "first_budget_with_plateaued_behaviour" in e


def test_return_to_go_converges_no_later_than_bootstrap(art):
    """The central measured claim, as recorded in the artifact."""
    v = art["verdict"]
    assert v["returns"]["first_budget_with_plateaued_behaviour"] <= \
           v["bootstrap"]["first_budget_with_plateaued_behaviour"], (
        "return-to-go should reach plateaued behaviour no later than the "
        "bootstrapped objective"
    )


def test_collapse_at_400_episodes_is_reproduced(art):
    """The step4/step5 default of 400 episodes must still be visibly undertrained.

    If this stops holding, the benchmark defaults should be revisited again.
    """
    rows = {(r["objective"], r["episodes"]): r for r in art["rows"]}
    if (("bootstrap", 400) in rows) and (("bootstrap", 2000) in rows):
        assert rows[("bootstrap", 400)]["items_mean"] < \
               rows[("bootstrap", 2000)]["items_mean"], (
            "the bootstrap objective no longer looks undertrained at 400 episodes; "
            "re-check the step4/step5 default episode budget"
        )


def test_benchmark_uses_the_diagnosed_episode_budget(art):
    """The step-5 artifact must not be produced from an undertrained budget."""
    bench = REPO / "results" / "step5_policy_benchmark_saudi.json"
    if not bench.exists():
        pytest.skip("step5 artifact not generated")
    b = json.loads(bench.read_text())
    plateau = art["verdict"]["bootstrap"]["first_budget_with_plateaued_behaviour"]
    assert b["train_episodes"] >= plateau, (
        f"step5 trains for {b['train_episodes']} episodes but the bootstrap "
        f"objective does not plateau until {plateau}"
    )


def test_reference_arms_are_recorded(art):
    """Greedy and the degenerate lowest-index policy anchor the comparison."""
    assert "greedy" in art["reference_arms"]
    assert "lowest_index" in art["reference_arms"]
    g = art["reference_arms"]["greedy"]
    assert g["mean_items"] == pytest.approx(art["budget"])
    assert 0.0 <= g["uar"] <= 1.0


def test_curves_are_recorded_per_seed(art):
    assert art["curves"], "no training curve recorded"
    objectives = {c["objective"] for c in art["curves"]}
    assert objectives == {"bootstrap", "returns"}
    for c in art["curves"]:
        assert c["distinct_states"] > 0
        assert c["episode"] > 0


def test_csv_matches_json(art):
    if not CSV_ART.exists():
        pytest.skip("csv not generated")
    import csv as _csv
    with open(CSV_ART) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == len(art["rows"])
    assert {(r["objective"], int(r["episodes"])) for r in rows} == \
           {(r["objective"], r["episodes"]) for r in art["rows"]}
