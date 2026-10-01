"""Step 6 — lambda (question-cost) sweep. Decision input for the V-6 gate.

Why this script exists
----------------------
Every number in the repository so far was produced at ``lambda = 0``. With no
acquisition-cost term the terminal utility of stopping at depth 0 and at depth B
are close in expectation, so (a) a sequential policy has almost nothing to gain
from good question *ordering* and (b) never stopping early is near-optimal. V-6
in §24 requires a supervisor sign-off on the lambda grid before any
cost-dependent claim is made. This script produces the evidence for that
sign-off. It does **not** make the sign-off.

This script is deliberately **predictor-independent**
--------------------------------------------
``ExactDP`` reads the empirical training support directly and never calls the
masked neural predictor, so V\\* and V_emp are functions of the training labels
alone. That is what makes this evidence pack usable *before* the predictor is
retrained (P1-b): the cost-utility structure of the problem does not depend on
how well the neural network happens to be fitted, and would be the same on
Polish once that cohort is unsealed.

What it measures
----------------
For each (lambda, budget) cell on the **train** split:

* ``V_star``                    — the exactly-solved finite-sample optimum
* ``mean_items_asked``          — how many questions the exact policy spends
* ``stopped_early_frac``        — how often it stops before the budget
* ``pure_support_frac``         — fraction of reachable states whose empirical
  support is *pure* (p_emp in {0,1})

The last one is the important diagnostic and is the reason this script exists in
this form. Under a deterministic sum-threshold label (``label = 1[sum(A) >= 4]``,
verified 506/506) the support becomes pure after roughly four questions, so
``u_stop = 1 - (p_emp - y)^2`` reaches exactly 1.0, ``V_star`` becomes exactly
1.000000, and **no further question can improve the reward**. The apparent value
of adaptive stopping at lambda = 0 is therefore a property of the label, not
evidence that a short interview retains sufficient diagnostic information. See
`POLICY_BENCHMARK_REPORT.md` and `diagnosisReady.md`.

Units
-----
``lambda`` is in **Brier units per question**: the reward is
``R = (1 - (p_hat - y)^2) - lambda * sum_j c_j`` with ``c_j = 1``, and the first
term is bounded by 1. An informative question is worth at most
``max_p [1 - p(1-p)] = 0.25``.

CORRECTION 2026-10-01 (measured in ``scripts/step8_evoi_scale_analysis.py``)
---------------------------------------------------------------------------
An earlier revision of this docstring justified the grid against a *"measured
marginal utility of an informative question ... roughly 0.05-0.15 Brier"*. **No
code in this repository ever measured that quantity**; it was prose only. It is
now measured, and the figure was wrong as stated:

* expected Brier improvement of the best remaining question, neural predictor,
  test split: **median 0.0035, mean 0.0400, p75 0.103, p90/p95 0.108**. So
  0.05-0.15 describes roughly the **p75-p95** band, not the typical value.
* the same quantity on the Beta-smoothed support posterior: **median 0.0015,
  mean 0.0337, p75 0.105**. The ratio of the two is **0.43 at the median and
  0.84 at the mean**, not two orders of magnitude.

Both estimators are the same functional form on the same support labels, so they
share units and the earlier "one reward, two scales" framing was wrong. What
survives is a different and sharper problem: on a *pure* support the
support-posterior EVOI is **analytically non-positive**
(``gain = 1/(n+2)^2 - sum_v w_v/(n_v+2)^2 <= 0``), so the statistic is
effectively a support-purity indicator rather than a graded information measure.
See ``V6_STOPPING_THRESHOLD_DECISION.md``.

MANDATORY CAVEAT — label circularity (§16.1)
-------------------------------------------
The Saudi cohort's labels are a deterministic sum-threshold oracle over the
questionnaire items themselves. Nothing here is clinical evidence.

Outputs (under results/):
  - results/lambda_sweep_saudi.{json,csv}
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
from src.data.splits import SCHEME, split_fingerprint, stratified_split
from src.env.environment import STOP, run_episode
from src.env.state import get_legal_items, init_state, update_state
from src.policies.beta_greedy import BetaGreedyPolicy
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from src.solvers.exact_custom import ExactDP

TAG = (
    "preliminary — lambda sweep 2026-10-01 (Step 6); CIRCULAR questionnaire "
    "labels (Saudi §16.1); NOT clinical evidence; V-4 / V-6 / V-7 pending "
    "supervisor sign-off; this artifact is DECISION INPUT for V-6 and is not a "
    "signed-off result"
)

CIRCULARITY_WARNING = (
    "Every number here is measured against a deterministic sum-threshold oracle "
    "over the questionnaire items themselves (label == 1[sum(A) >= threshold], "
    "verified 506/506 on Saudi). At lambda=0 the empirical support becomes pure "
    "after roughly four questions, u_stop = 1 - (p_emp - y)^2 saturates at "
    "exactly 1.0, and V_star becomes exactly 1.000000 — so no further question "
    "can improve the reward. The measured value of adaptive stopping at "
    "lambda=0 is therefore a property of the label rule, not evidence that a "
    "short interview retains sufficient diagnostic information."
)

#: Costs are uniform, so lambda is directly "Brier units per question".
DEFAULT_LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
#: Above this the sweep is flagged degenerate: no question is worth its cost.
DEGENERATE_ITEMS = 1.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def exact_as_policy(dp: ExactDP):
    """Adapt ``ExactDP`` to the ``(state, legal) -> action`` policy protocol.

    Never emits an action the environment considers illegal.
    """
    def policy(state, legal):
        action = dp.get_action(state)
        if action not in legal:
            items = [a for a in legal if a != STOP]
            return items[0] if items else STOP
        return action
    return policy


def support_purity_by_depth(records, n_items, budget, policy=None) -> List[float]:
    """Fraction of states whose empirical support is *pure*, by depth.

    A pure support means every training record still consistent with the observed
    answers shares the same label, so ``p_emp`` is exactly 0 or 1, the entropy
    term driving information gain collapses to zero, and
    ``u_stop = 1 - (p_emp - y)^2`` reaches exactly 1. Under the circular label
    this happens after roughly four questions, which is the mechanism behind
    ``V_star == 1.000000`` at lambda = 0.

    ``policy`` selects the exploration order; greedy-IG is used by default so
    this matches the depth profile the benchmark actually visits.
    """
    X = np.stack([r["item_responses"] for r in records])
    mm = np.stack([r["missing_mask"] for r in records])
    y = np.array([r["label"] for r in records], dtype=int)
    if policy is None:
        policy = GreedyIGPolicy(records, n_items=n_items)

    pure = [0] * budget
    total = [0] * budget
    for rec in records:
        state = init_state(rec)
        state["questions_remaining"] = budget
        state["budget"] = budget
        for depth in range(budget):
            legal = get_legal_items(state)
            if not legal:
                break
            action = policy(state, legal + [STOP])
            if action == STOP:
                break
            value = rec["item_responses"][action]
            if np.isnan(value):
                break
            state = update_state(state, action, int(value), budget - depth - 1)
            state["budget"] = budget
            state["questions_remaining"] = budget - depth - 1

            idx = _support(state, X, mm, n_items)
            total[depth] += 1
            if len(idx) and float(y[idx].mean()) in (0.0, 1.0):
                pure[depth] += 1
    return [(pure[d] / total[d]) if total[d] else float("nan") for d in range(budget)]


def pure_support_at_stop(policy, records, n_items, budget) -> float:
    """Fraction of episodes whose *terminal* state has a pure support.

    This is the decision-relevant quantity: it says whether the policy stopped
    because the evidence was sufficient or for some other reason.
    """
    X = np.stack([r["item_responses"] for r in records])
    mm = np.stack([r["missing_mask"] for r in records])
    y = np.array([r["label"] for r in records], dtype=int)

    hits = 0
    for rec in records:
        state = init_state(rec)
        state["questions_remaining"] = budget
        state["budget"] = budget
        for depth in range(budget):
            legal = get_legal_items(state)
            if not legal:
                break
            action = policy(state, legal + [STOP])
            if action == STOP:
                break
            value = rec["item_responses"][action]
            if np.isnan(value):
                break
            state = update_state(state, action, int(value), budget - depth - 1)
            state["budget"] = budget
            state["questions_remaining"] = budget - depth - 1
        idx = _support(state, X, mm, n_items)
        if len(idx) and float(y[idx].mean()) in (0.0, 1.0):
            hits += 1
    return hits / len(records) if records else float("nan")


def _support(state, X, mm, n_items) -> np.ndarray:
    idx = np.ones(len(X), dtype=bool)
    for j in range(n_items):
        m = state["mask"][j]
        if m == 1:
            idx &= (~mm[:, j]) & (X[:, j] == state["value"][j])
        elif m == 2:
            idx &= mm[:, j]
        if not idx.any():
            break
    return np.where(idx)[0]


def support_posterior_predictor(records, n_items):
    """A ``state -> p`` predictor backed by the empirical training support.

    Used instead of the neural predictor so the held-out reward is
    **predictor-independent** *and* meaningful.

    An earlier revision of this script passed a constant ``lambda s: 0.5``. That
    makes ``1 - (p_hat - y)^2`` identically 0.75 for every record, so
    ``mean_reward`` collapsed to ``0.75 - lambda * n_items`` and measured nothing
    but acquisition cost. The empirical support posterior is the same quantity
    ``ExactDP`` optimises, so the held-out numbers become directly comparable to
    ``V_star`` and the artifact stays free of the P1-b predictor dependency.
    """
    X = np.stack([r["item_responses"] for r in records])
    mm = np.stack([r["missing_mask"] for r in records])
    y = np.array([r["label"] for r in records], dtype=float)

    def predict(state) -> float:
        idx = _support(state, X, mm, n_items)
        if len(idx) == 0:
            return 0.5
        return float(y[idx].mean())

    return predict


def evaluate(records, policy, B, lambda_cost, predictor) -> Dict[str, float]:
    """Run every record at budget B and summarise the §10 utility."""
    episodes = [
        run_episode(rec, question_budget=B, b_min=0, policy=policy,
                    predictor=predictor, lambda_cost=lambda_cost,
                    cost_mode="uniform", tau=0.5)
        for rec in records
    ]
    n = np.array([len(e["items_asked"]) for e in episodes], dtype=float)
    rewards = np.array([e["R"] for e in episodes], dtype=float)
    return {
        "mean_items": float(n.mean()),
        "median_items": float(np.median(n)),
        "mean_reward": float(rewards.mean()),
        "stopped_early_frac": float(np.mean(n < B)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saudi")
    ap.add_argument("--budgets", default="1,2,3,4,5,6,8,10")
    ap.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--eval-n", type=int, default=127,
                    help="records used for the held-out item-count comparison")
    args = ap.parse_args()

    print("=" * 78)
    print("Step 6 — lambda (question-cost) sweep — DECISION INPUT for V-6")
    print(f"  {TAG}")
    print("=" * 78)

    records = load_dataset(args.dataset, synthetic=args.synthetic)
    train, val, test = stratified_split(records, seed=args.seed)
    fingerprint = split_fingerprint(train, val, test)
    n_items = len(records[0]["item_responses"])
    print(f"[{args.dataset}] {len(records)} rows | train={len(train)} "
          f"val={len(val)} test={len(test)} | n_items={n_items}")
    print(f"  split scheme={SCHEME} fingerprint={fingerprint}")

    budgets = [int(b) for b in args.budgets.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    rows: List[Dict[str, Any]] = []
    for lam in lambdas:
        print(f"\n[lambda={lam}]")
        for B in budgets:
            t0 = time.time()
            try:
                dp = ExactDP(train, n_items=n_items, budget=B, b_min=0,
                             lambda_cost=lam, cost_mode="uniform")
                sol = dp.solve()
            except Exception as e:  # tractability bound, not a crash
                print(f"  B={B:<2d} ExactDP {type(e).__name__}: {e}")
                continue
            if sol["status"] != "optimal":
                print(f"  B={B:<2d} ExactDP status={sol['status']} — omitted")
                continue
            pol = exact_as_policy(dp)
            v_emp = dp.evaluate_policy(pol, train)
            # Predictor-independent posterior: the same quantity ExactDP
            # optimises, so held-out rewards are comparable to V_star and this
            # artifact does not depend on the P1-b predictor retrain.
            predictor = support_posterior_predictor(train, n_items)
            held = evaluate(test[: args.eval_n], pol, B, lam, predictor)
            greedy = evaluate(test[: args.eval_n],
                              GreedyIGPolicy(train, n_items=n_items), B, lam,
                              predictor)
            rnd = evaluate(test[: args.eval_n], RandomPolicy(seed=args.seed), B,
                           lam, predictor)
            # P1-f: EVOI stopping, so the cost-utility frontier has a principled
            # stopping arm alongside the fixed-length ones. Its lambda is in
            # *support-posterior utility* units, which is a different scale from
            # the neural-predictor reward used elsewhere — see the artifact note.
            beta = BetaGreedyPolicy(train, n_items=n_items, lambda_cost=lam,
                                    select_by="evoi")
            beta_stats = evaluate(test[: args.eval_n], beta, B, lam, predictor)
            pure = pure_support_at_stop(pol, train[:100], n_items, B)
            # A budget of 1 can never spend fewer than one question, so
            # mean_items == 1 there is the budget, not a policy degeneracy.
            degenerate = (B > 1) and (held["mean_items"] <= DEGENERATE_ITEMS)
            row = {
                "lambda": lam,
                "B": B,
                "V_star": round(sol["V_star"], 8),
                "gap_exact_train": round(sol["V_star"] - v_emp, 8),
                "exact_mean_items": round(held["mean_items"], 4),
                "exact_median_items": round(held["median_items"], 2),
                "exact_stop_early": round(held["stopped_early_frac"], 4),
                "exact_mean_reward": round(held["mean_reward"], 6),
                "greedy_mean_items": round(greedy["mean_items"], 4),
                "greedy_mean_reward": round(greedy["mean_reward"], 6),
                "random_mean_items": round(rnd["mean_items"], 4),
                "random_mean_reward": round(rnd["mean_reward"], 6),
                "beta_greedy_mean_items": round(beta_stats["mean_items"], 4),
                "beta_greedy_stop_early": round(beta_stats["stopped_early_frac"], 4),
                "beta_greedy_mean_reward": round(beta_stats["mean_reward"], 6),
                "pure_support_frac_at_stop": round(float(pure), 4),                "v_star_saturated": bool(abs(sol["V_star"] - 1.0) < 1e-9),
                "degenerate": bool(degenerate),
                "n_states": sol["n_states"],
                "n_evals": sol["n_evals"],
                "dp_seconds": round(time.time() - t0, 2),
            }
            rows.append(row)
            print(f"  B={B:<2d} V*={sol['V_star']:.6f}  exact items="
                  f"{held['mean_items']:.2f} (early {held['stopped_early_frac']:.2f})"
                  f"  greedy items={greedy['mean_items']:.2f}"
                  f"  pure_support={pure:.2f}"
                  f"{'  DEGENERATE' if degenerate else ''}")

    if not rows:
        raise SystemExit("no (lambda, B) cell produced an exact solution")

    # ---- invariants the sweep must satisfy, checked rather than assumed ----
    by_b: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_b.setdefault(r["B"], []).append(r)
    monotonicity: List[Dict[str, Any]] = []
    for B, group in sorted(by_b.items()):
        group = sorted(group, key=lambda r: r["lambda"])
        ok = all(group[i]["V_star"] >= group[i + 1]["V_star"] - 1e-9
                 for i in range(len(group) - 1))
        monotonicity.append({"B": B, "V_star_non_increasing_in_lambda": ok,
                             "V_star": [r["V_star"] for r in group]})
        print(f"\n[invariant] B={B}: V* non-increasing in lambda -> {ok}")

    degenerate_lambdas = sorted({r["lambda"] for r in rows if r["degenerate"]})
    print(f"[boundary] degenerate lambdas (B>1 and mean items <= "
          f"{DEGENERATE_ITEMS}): {degenerate_lambdas}")

    # The saturation diagnostic: how fast does the empirical support become pure
    # under greedy-IG ordering? This is what makes V* hit exactly 1.0 at lambda=0
    # and therefore what the measured value of adaptive stopping rests on.
    purity = support_purity_by_depth(train, n_items, max(budgets))
    print("\n[diagnostic] pure-support fraction by depth (greedy-IG order, train):")
    for d, frac in enumerate(purity, start=1):
        print(f"    after {d:2d} question(s): {frac:.3f}")
    first_all = next((d for d, f in enumerate(purity, start=1) if f >= 0.99), None)
    first_95 = next((d for d, f in enumerate(purity, start=1) if f >= 0.95), None)
    print("  -> support is >=95% pure from depth "
          f"{first_95 if first_95 is not None else 'n/a'}; "
          f">=99% pure from depth {first_all if first_all is not None else 'n/a'}")

    RESULTS.mkdir(exist_ok=True)
    artifact = {
        "tag": TAG,
        "step": "6 — lambda (question-cost) sweep, V-6 decision input",
        "purpose": (
            "Evidence for the V-6 supervisor sign-off on the lambda grid. This is "
            "decision input, not a signed-off result."
        ),
        "dataset": args.dataset,
        "source": "synthetic" if args.synthetic else "real",
        "label_source": "questionnaire (CIRCULAR — see §16.1)",
        "circularity_warning": CIRCULARITY_WARNING,
        "predictor_independent": True,
        "predictor_independence_note": (
            "ExactDP reads the empirical training support directly and never calls "
            "the masked neural predictor, so V* and V_emp depend only on the "
            "training labels. The held-out policy comparison likewise uses the "
            "empirical support posterior rather than the neural predictor, so this "
            "evidence pack is valid independently of the P1-b predictor retrain. "
            "(An earlier revision passed a constant 0.5 predictor, which made "
            "mean_reward measure nothing but acquisition cost.)"
        ),
        "lambda_units": "Brier units per question (c_j = 1, reward term in [0,1])",
        "lambda_units_caveat": (
            "Same units on both sides; the earlier 'one reward, two scales' "
            "framing in this field was WRONG and is retracted. Both the "
            "Beta-smoothed support posterior and the neural predictor are scored "
            "with the same functional form mean_y[1-(p-y)^2] on the same support "
            "labels, so both are expected Brier improvement, dimensionless, with "
            "ceiling max_p[1-p(1-p)]=0.25. Measured on the test split, the best "
            "remaining question is worth median 0.0015 (support posterior) vs "
            "0.0035 (neural predictor), mean 0.0337 vs 0.0400, p75 0.105 vs "
            "0.103 -- a ratio of 0.43 at the median and 0.84 at the mean, not "
            "two orders of magnitude. The '0.05-0.15 marginal utility' previously "
            "quoted here was prose that no code measured; measured, it is the "
            "p75-p95 band, not the median. What DOES matter: on a pure support "
            "the support-posterior EVOI is analytically non-positive "
            "(gain = 1/(n+2)^2 - sum_v w_v/(n_v+2)^2 <= 0), so the statistic "
            "acts as a support-purity indicator, and the threshold grid spans "
            "only ~20 distinct behaviours with a sharp cliff between 0.001 and "
            "0.003. No threshold is selected. See "
            "V6_STOPPING_THRESHOLD_DECISION.md and "
            "scripts/step8_evoi_scale_analysis.py."
        ),
        "n_items": n_items,
        "seed": args.seed,
        "split_scheme": SCHEME,
        "split_fingerprint": fingerprint,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "eval_n": args.eval_n,
        "budgets": budgets,
        "lambdas": lambdas,
        "degenerate_lambdas": degenerate_lambdas,
        "rows": rows,
        "v_star_monotonicity": monotonicity,
        "support_purity_by_depth": {
            "description": (
                "Fraction of greedy-IG-ordered train states whose empirical support "
                "is pure (p_emp in {0,1}) after k questions. Once the support is "
                "pure, u_stop = 1 - (p_emp - y)^2 equals exactly 1 and no further "
                "question can raise the reward."
            ),
            "by_depth": [round(float(f), 4) for f in purity],
            "all_pure_from_depth_099": first_all,
            "all_pure_from_depth_095": first_95,
        },
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    j = RESULTS / f"lambda_sweep_{args.dataset}.json"
    j.write_text(json.dumps(artifact, indent=2))
    c = RESULTS / f"lambda_sweep_{args.dataset}.csv"
    with open(c, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n-> {j.relative_to(REPO)}")
    print(f"-> {c.relative_to(REPO)}")
    print("\nThis artifact is DECISION INPUT for V-6. A supervisor must sign off on")
    print("the lambda grid before any cost-dependent claim is made.")
    print("REMINDER: circular labels. Not clinical evidence. See diagnosisReady.md §4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
