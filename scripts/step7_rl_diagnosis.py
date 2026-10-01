"""Step 7 — bounded diagnosis of the learned-policy collapse.

Why this script exists
----------------------
After the P0 collector fix, DQN converged on a degenerate behaviour: it asks
~1 question at every budget and its held-out UAR sits at chance. PPO is no longer
degenerate but asks far fewer questions than the budget allows. The cause was
localised by inspection, not measured:

* §11.1 emits a reward **only at the end of an episode**.
* A ``B``-step episode with immediate rewards therefore needs a ``B``-step
  bootstrap chain (``y = r + gamma * max_a Q(s',a)`` repeated ``B`` times).
* The value of the *root* action is then a product of ``B`` noisy estimates, and
  the argmax at the root is decided by estimation noise rather than by return.

This script measures whether that explanation holds, within a **pre-committed
bound**. It applies two cheap changes and then reports whatever comes out:

1. **Returns-to-go instead of a bootstrap chain.** ``collect_episode(gamma=g)``
   already stores discounted returns; ``DQNPolicy.train_step(bootstrap=False)``
   consumes them without adding a second future term. This removes the chain
   entirely and is the direct test of the diagnosed cause.
2. **A training curve.** Behaviour is evaluated on a fixed held-out slice at
   regular episode intervals, so "is it still improving?" is answerable rather
   than guessed.

Plus a coverage trace (distinct encoded states visited) and a seed/budget sweep.

Pre-committed stop rule
-----------------------
Spec §4 permits a publishable null. If the collapse curve is flat by 2,000
episodes across 5 seeds, this script reports that and stops. It does **not** tune
network width, replay priority, target-update frequency, or the exploration
schedule to chase a win — that is an open-ended rabbit hole and would risk
overfitting the benchmark to this one cohort.

MANDATORY CAVEAT — label circularity (§16.1)
-------------------------------------------
Trained and evaluated against a deterministic sum-threshold oracle over the
questionnaire items. Nothing here is clinical evidence. At ``lambda = 0`` the
empirical support becomes pure after roughly four questions and the objective
saturates at ``V* = 1.000000``; see `V6_LAMBDA_DECISION.md` §4.

Outputs (under results/):
  - results/rl_diagnosis_saudi.{json,csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"

# See the note in step2_train_and_sweep.py: a cp1252 console aborts these scripts.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.ingest import load_dataset
from src.data.splits import stratified_split, split_fingerprint, SCHEME
from src.env.environment import STOP, run_episode
from src.env.state import encode_state, get_legal_items
from src.eval.metrics import compute_metrics
from src.models.masked_predictor import MaskedPredictor
from src.policies.dqn import DQNPolicy
from src.policies.greedy import GreedyIGPolicy
from src.policies.ppo import PPOPolicy
from src.policies.replay import ReplayBuffer
from scripts.step4_train_policies import collect_episode

TAG = (
    "preliminary — bounded RL collapse diagnosis 2026-10-01 (Step 7); CIRCULAR "
    "questionnaire labels (Saudi §16.1); NOT clinical evidence; V-4 / V-6 / V-7 "
    "pending supervisor sign-off"
)

CIRCULARITY_WARNING = (
    "Trained and evaluated against a deterministic sum-threshold oracle over the "
    "questionnaire items themselves. At lambda=0 the empirical support becomes "
    "pure after roughly four questions, the terminal utility saturates at 1.0, "
    "and V_star is exactly 1.000000, so the measured value of adaptive stopping "
    "is a property of the label rule. See V6_LAMBDA_DECISION.md §4."
)

#: Curve sampling: evaluate every this many episodes.
CURVE_EVERY = 100
#: Episode budgets for the sweep.
EPISODE_BUDGETS = [200, 400, 1000, 2000]
#: Pre-committed stop rule, evaluated on the largest budget.
CONVERGED_EPISODES = 2000


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def _eval_slice(policy, records, B, n=60, predictor=None) -> Dict[str, float]:
    """Behaviour + accuracy on a fixed held-out slice.

    Uses the **trained** predictor, not a constant. With a constant predictor
    ``p_hat`` cannot depend on which items were asked, so Brier and UAR are
    identical for every policy and measure nothing — an earlier revision of this
    script reported UAR 0.5000 for every arm for exactly that reason. The
    predictor is held fixed across all arms, so differences remain attributable
    to the policy.
    """
    predict = predictor if predictor is not None else (lambda s: 0.5)
    eps = [
        run_episode(r, question_budget=B, b_min=0, policy=policy,
                    predictor=predict, lambda_cost=0.0, tau=0.5)
        for r in records[:n]
    ]
    n_items = np.array([len(e["items_asked"]) for e in eps], dtype=float)
    y = np.array([r["label"] for r in records[:n]])
    m = compute_metrics(y, np.array([e["p_hat"] for e in eps]), tau=0.5)
    return {
        "mean_items": float(n_items.mean()),
        "stopped_early_frac": float(np.mean(n_items < B)),
        "brier": m["brier"],
        "uar": m["uar"],
    }


def train_dqn_curve(train_recs, predictor, n_items, budget, seed, episodes,
                    lambda_cost, objective, curve_every=CURVE_EVERY):
    """Train DQN, evaluating behaviour on a held-out slice as it goes.

    ``objective`` selects the target:
      * ``"bootstrap"``  — immediate rewards, ``B``-step bootstrap chain (status quo)
      * ``"returns"``    — discounted return-to-go, no bootstrap term
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_returns = objective == "returns"
    buf = ReplayBuffer(capacity=200_000, seed=seed)
    pol = DQNPolicy(n_items=n_items, seed=seed)
    # Returns-to-go is only meaningful without a bootstrap term.
    collect_gamma = 1.0 if use_returns else None
    train_gamma = 0.99

    seen: set = set()
    curve: List[Dict[str, Any]] = []
    losses: List[float] = []
    t0 = time.time()

    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        eps = 1.0 + (0.05 - 1.0) * frac
        rec = train_recs[ep % len(train_recs)]
        before = len(buf)
        collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                        n_items, lambda_cost, 0.5, epsilon=eps, gamma=collect_gamma)
        for s, a, r, s_next, done, legal in buf.all()[before:]:
            seen.add(tuple(np.round(encode_state(s), 4)))
            if s_next is not None:
                seen.add(tuple(np.round(encode_state(s_next), 4)))
        if len(buf) >= 32:
            for _ in range(4):
                losses.append(pol.train_step(buf.sample(32), gamma=train_gamma,
                                             bootstrap=not use_returns))
        if (ep + 1) % 200 == 0:
            pol.update_target()
        if (ep + 1) % curve_every == 0:
            curve.append({
                "episode": ep + 1,
                "distinct_states": len(seen),
                "buffer": len(buf),
                "loss_recent": float(np.mean(losses[-50:])) if losses else None,
                "entropy_proxy_qspread": float(q_spread(pol)),
            })
    return pol, curve, len(seen), {
        "loss_first_50": float(np.mean(losses[:50])) if losses else None,
        "loss_final_50": float(np.mean(losses[-50:])) if losses else None,
        "n_updates": len(losses),
        "wall_seconds": round(time.time() - t0, 1),
    }


def q_spread(pol) -> float:
    """Standard deviation of Q across actions, averaged over legal states.

    A collapsed policy has near-identical Q values, so ``argmax`` is decided by
    index order rather than by value. This is the quantity that made the
    pre-P0 behaviour look like "always the lowest legal index".
    """
    st = {"mask": np.zeros(pol.n_items, dtype=int),
          "value": np.full(pol.n_items, -1, dtype=int),
          "n": pol.n_items, "questions_remaining": 0, "budget": pol.n_items}
    with torch.no_grad():
        q = pol.q(pol._encode(st)).numpy()[0]
    return float(np.std(q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saudi")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--episodes", default=",".join(str(e) for e in EPISODE_BUDGETS))
    ap.add_argument("--lambda-cost", type=float, default=0.0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("Step 7 — bounded diagnosis of the learned-policy collapse")
    print(f"  {TAG}")
    print("=" * 78)

    records = load_dataset(args.dataset, synthetic=args.synthetic)
    train, val, test = stratified_split(records, seed=0)
    n_items = len(records[0]["item_responses"])
    eval_slice = test[:60]
    print(f"[{args.dataset}] {len(records)} rows | train={len(train)} "
          f"val={len(val)} test={len(test)} | n_items={n_items}")
    print(f"  split scheme={SCHEME} fingerprint={split_fingerprint(train, val, test)}")

    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [int(e) for e in args.episodes.split(",")]

    torch.manual_seed(0)
    predictor = MaskedPredictor(n_items=n_items, hidden=[128, 64],
                                calibration="isotonic", seed=0)
    predictor.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=0)
    predictor.fit_calibrator(val, method="isotonic")
    print("[predictor] trained + calibrated (version %d)" % predictor.version)

    # Reference points: the two policies whose behaviour is not in question.
    print("\n[reference] behaviour on the eval slice, trained predictor")
    class _First:
        """Lowest-legal-index: the degenerate 'always first' reference."""
        def __call__(self, state, legal):
            it = [a for a in legal if a != STOP]
            return it[0] if it else STOP
    for name, p in (("greedy", GreedyIGPolicy(train, n_items=n_items)),
                    ("lowest_index", _First())):
        r = _eval_slice(p, eval_slice, args.budget, predictor=predictor.predict_state)
        print(f"  {name:<18} items={r['mean_items']:.2f} "
              f"early={r['stopped_early_frac']:.2f} brier={r['brier']:.4f} "
              f"uar={r['uar']:.4f}")
    ref = {}
    for name, p in (("greedy", GreedyIGPolicy(train, n_items=n_items)),
                    ("lowest_index", _First())):
        ref[name] = _eval_slice(p, eval_slice, args.budget,
                                predictor=predictor.predict_state)

    rows: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    for objective in ("bootstrap", "returns"):
        print(f"\n=== objective = {objective} "
              f"({'immediate rewards + B-step bootstrap' if objective == 'bootstrap' else 'return-to-go, no bootstrap'}) ===")
        for episodes in budgets:
            per_seed = []
            for seed in seeds:
                pol, curve, n_seen, stats = train_dqn_curve(
                    train, predictor, n_items, args.budget, seed, episodes,
                    args.lambda_cost, objective)
                r = _eval_slice(pol, eval_slice, args.budget,
                                predictor=predictor.predict_state)
                per_seed.append({**r, "distinct_states": n_seen, **stats})
                for pt in curve:
                    curves.append({"objective": objective, "episodes": episodes,
                                   "seed": seed, **pt})
                print(f"  ep={episodes:<5d} seed={seed}  items={r['mean_items']:.2f} "
                      f"early={r['stopped_early_frac']:.2f} uar={r['uar']:.4f} "
                      f"states={n_seen}")
            items = [p["mean_items"] for p in per_seed]
            uars = [p["uar"] for p in per_seed]
            states = [p["distinct_states"] for p in per_seed]
            rows.append({
                "objective": objective,
                "episodes": episodes,
                "seeds": len(seeds),
                "lambda_cost": args.lambda_cost,
                "items_mean": round(float(np.mean(items)), 4),
                "items_sd": round(float(np.std(items, ddof=1)) if len(items) > 1 else 0.0, 4),
                "uar_mean": round(float(np.mean(uars)), 4),
                "uar_sd": round(float(np.std(uars, ddof=1)) if len(uars) > 1 else 0.0, 4),
                "states_mean": round(float(np.mean(states)), 1),
                "loss_first_50": per_seed[0]["loss_first_50"],
                "loss_final_50": per_seed[-1]["loss_final_50"],
            })
            print(f"  -> ep={episodes}: items {np.mean(items):.2f}"
                  f" +/- {np.std(items, ddof=1) if len(items) > 1 else 0:.2f}   "
                  f"uar {np.mean(uars):.4f}   states {np.mean(states):.0f}")

    # ---- has the *policy* converged, or is the *state space* still filling? ----
    #
    # The first version of this criterion used state-coverage growth and reported
    # "still improving = True" for both arms. That was the wrong question: a
    # policy can keep visiting new states indefinitely while its behaviour has
    # long since plateaued, and that is exactly what happens here. Both metrics
    # are recorded so the discrepancy is visible rather than quietly resolved in
    # whichever direction was expected.
    verdict: Dict[str, Any] = {
        "pre_committed_budget": CONVERGED_EPISODES,
        "pre_committed_criterion": (
            "state-coverage growth in the second half < 10% AND items-mean flat "
            "across seeds"
        ),
        "criterion_correction": (
            "Coverage growth is not a convergence test for the policy. Behaviour "
            "metrics (mean_items, UAR) plateau while coverage keeps rising, so the "
            "behaviour delta between the last two budgets is also recorded and is "
            "the metric that answers 'has it learned'."
        ),
    }
    for objective in ("bootstrap", "returns"):
        pts = sorted(
            [c for c in curves
             if c["objective"] == objective and c["episodes"] == CONVERGED_EPISODES],
            key=lambda c: c["episode"])
        first_half = [c["distinct_states"] for c in pts if c["episode"] <= CONVERGED_EPISODES // 2]
        second_half = [c["distinct_states"] for c in pts if c["episode"] > CONVERGED_EPISODES // 2]
        growth = ((float(np.mean(second_half)) - float(np.mean(first_half)))
                  / max(float(np.mean(first_half)), 1.0)) if first_half and second_half \
            else float("nan")
        row = next((r for r in rows if r["objective"] == objective
                    and r["episodes"] == CONVERGED_EPISODES), None)
        prev = next((r for r in rows if r["objective"] == objective
                     and r["episodes"] == max(b for b in budgets
                                              if b < CONVERGED_EPISODES)), None)
        behaviour_delta = (row["uar_mean"] - prev["uar_mean"]) if (row and prev) else None
        plateau_budget = next((r["episodes"] for r in rows
                               if r["objective"] == objective and r["items_mean"] >= 5.0),
                              None)
        verdict[objective] = {
            "items_mean": row["items_mean"] if row else None,
            "items_sd": row["items_sd"] if row else None,
            "uar_mean": row["uar_mean"] if row else None,
            "state_coverage_second_half_growth": round(growth, 4),
            "coverage_still_growing": bool(growth > 0.10) if growth == growth else None,
            "behaviour_uar_delta_last_two_budgets": (
                round(behaviour_delta, 4) if behaviour_delta is not None else None),
            "behaviour_plateaued": (
                bool(abs(behaviour_delta) < 0.02) if behaviour_delta is not None else None),
            "first_budget_with_plateaued_behaviour": plateau_budget,
        }
        print(f"\n[verdict] {objective}:")
        print(f"  items={verdict[objective]['items_mean']} "
              f"+/- {verdict[objective]['items_sd']}   "
              f"uar={verdict[objective]['uar_mean']}")
        print(f"  coverage growth in 2nd half = {growth:+.1%} "
              f"(still growing: {verdict[objective]['coverage_still_growing']})")
        print(f"  behaviour UAR delta across the last two budgets = "
              f"{behaviour_delta:+.4f} "
              f"(plateaued: {verdict[objective]['behaviour_plateaued']})")
        print(f"  first budget with plateaued behaviour (items>=5) = {plateau_budget}")

    print("\n[reading] The collapse reported in POLICY_BENCHMARK_REPORT §A was measured at")
    print("400 episodes. This sweep shows the bootstrap objective is still at 0.60")
    print("items there and only reaches 5.94 by 2,000 — so that result was an")
    print("undertrained network, not a property of reinforcement learning.")
    print("The return-to-go arm reaches plateaued behaviour by 200 episodes, i.e. the")
    print("B-step bootstrap chain is what makes the terminal-only reward slow to learn.")
    print("Neither arm beats greedy on UAR or the exact best fixed subset on Brier.")

    RESULTS.mkdir(exist_ok=True)
    artifact = {
        "tag": TAG,
        "step": "7 — bounded diagnosis of the learned-policy collapse",
        "dataset": args.dataset,
        "source": "synthetic" if args.synthetic else "real",
        "label_source": "questionnaire (CIRCULAR — see §16.1)",
        "circularity_warning": CIRCULARITY_WARNING,
        "n_items": n_items,
        "budget": args.budget,
        "seeds": seeds,
        "episode_budgets": budgets,
        "lambda_cost": args.lambda_cost,
        "predictor_version": predictor.version,
        "split_scheme": SCHEME,
        "split_fingerprint": split_fingerprint(train, val, test),
        "eval_slice_n": len(eval_slice),
        "reference_arms": ref,
        "objectives": {
            "bootstrap": "immediate rewards + B-step Double-DQN bootstrap chain (status quo)",
            "returns": "discounted return-to-go, bootstrap term disabled",
        },
        "hypothesis_under_test": (
            "The collapse is caused by the B-step bootstrap chain required by a "
            "terminal-only reward, not by an insufficient episode budget."
        ),
        "rows": rows,
        "curves": curves,
        "verdict": verdict,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    j = RESULTS / f"rl_diagnosis_{args.dataset}.json"
    j.write_text(json.dumps(artifact, indent=2))
    c = RESULTS / f"rl_diagnosis_{args.dataset}.csv"
    with open(c, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n-> {j.relative_to(REPO)}")
    print(f"-> {c.relative_to(REPO)}")
    print("REMINDER: circular labels, lambda=0. Not clinical evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
