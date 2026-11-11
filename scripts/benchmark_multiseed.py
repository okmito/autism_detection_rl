"""Multi-seed policy benchmark runner — Stage 5/6.

Reproducible experiment harness: every eligible policy is evaluated
across multiple independent seeds (default 10 for the development
benchmark) on held-out data only. Reports mean, standard deviation,
95% confidence intervals, completed/failed runs, average questions
asked, STOP frequency, and performance by budget.

Usage:
    python scripts/benchmark_multiseed.py --seeds 10 --budget 6 --lambda 0.0

Historical Step-5 artifacts are never overwritten: this script writes
to ``results/benchmark_multiseed_saudi.{json,csv}``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.ingest import load_dataset  # noqa: E402
from src.data.splits import stratified_split, split_fingerprint, SCHEME  # noqa: E402
from src.env.environment import STOP, run_episode  # noqa: E402
from src.eval.metrics import compute_metrics, compute_metrics_with_ci, acquisition_burden  # noqa: E402
from src.models.masked_predictor import MaskedPredictor  # noqa: E402
from src.policies.beta_greedy import BetaGreedyPolicy  # noqa: E402
from src.policies.greedy import GreedyIGPolicy  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.random_policy import RandomPolicy  # noqa: E402
from src.policies.static_fixed import ExactFixedSubsetPolicy  # noqa: E402
from src.policies.static_rfe import StaticRFEPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402


def _eval_policy(policy_factory, test_records, predictor, budget, seed):
    """Evaluate one policy factory over the test split. Returns per-seed
    metrics + burden; failures are captured, never swallowed."""
    policy = policy_factory()
    y_true, y_prob, asked_list = [], [], []
    for rec in test_records:
        out = run_episode(rec, budget, 0, policy, predictor, lambda_cost=0.0)
        y_true.append(rec["label"])
        y_prob.append(out["p_hat"])
        asked_list.append(out["items_asked"])
    metrics = compute_metrics(np.asarray(y_true), np.asarray(y_prob))
    burden = acquisition_burden(asked_list)
    stop_frac = float(np.mean([1.0 if len(a) < budget else 0.0 for a in asked_list])) if asked_list else 0.0
    return {
        "metrics": metrics,
        "burden": burden,
        "stop_early_frac": stop_frac,
        "n_completed": len(test_records),
    }


def benchmark(*, seeds: int = 10, budgets: list[int] | None = None,
              lambda_: float = 0.0, verbose: bool = True) -> dict:  # noqa: F821
    if budgets is None:
        budgets = [1, 2, 3, 4, 5, 6]
    records = load_dataset("saudi")
    train, val, test = stratified_split(records, seed=0)
    fingerprint = split_fingerprint(train, val, test)
    if verbose:
        print(f"split {SCHEME} fingerprint={fingerprint} train={len(train)} val={len(val)} test={len(test)}")

    # Train one shared predictor (v3 logistic + Bayesian wrapper) per seed
    # so policies are compared on equal footing; v2 MLP is kept out of the
    # multi-seed loop to keep runtime bounded — see docs.
    results: dict = {
        "artifact": "benchmark_multiseed_saudi",
        "split_scheme": SCHEME,
        "split_fingerprint": fingerprint,
        "seeds": list(range(seeds)),
        "budgets": budgets,
        "lambda": float(lambda_),
        "label_source": "questionnaire (CIRCULAR)",
        "circularity": "Deterministic",
        "polish_used": False,
        "per_seed": {},
    }
    failed_total, completed_total = 0, 0
    for B in budgets:
        if verbose:
            print(f"\n=== budget B={B} ===")
        arm_stats: dict[str, list] = {}
        for seed in range(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
            bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
            irt = IRTCATPolicy(train, n_items=10)
            arms = {
                "greedy": lambda: GreedyIGPolicy(train, n_items=10),
                "beta_greedy": lambda: BetaGreedyPolicy(train, 10, lambda_cost=lambda_),
                "irt_cat": lambda: IRTCATPolicy(train, n_items=10),
                "static_rfe": lambda: StaticRFEPolicy(train, n_items=10, budget=B),
                "exact_fixed_subset": lambda: ExactFixedSubsetPolicy(train, n_items=10, budget=B),
                "hybrid_irt_evoi": lambda: HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=lambda_),
                "random": lambda: RandomPolicy(seed=seed),
            }
            results["per_seed"].setdefault(f"B{B}", {})[f"seed{seed}"] = {}
            for name, factory in arms.items():
                try:
                    r = _eval_policy(factory, test, bayes, B, seed)
                    arm_stats.setdefault(name, []).append(r["metrics"])
                    completed_total += 1
                    results["per_seed"][f"B{B}"][f"seed{seed}"][name] = r
                except Exception as e:  # keep going; record the failure
                    failed_total += 1
                    if verbose:
                        print(f"  [fail] B={B} seed={seed} {name}: {e}")
                    results["per_seed"][f"B{B}"][f"seed{seed}"][name] = {"error": str(e)}
        # aggregate means across seeds
        summary_b: dict[str, Any] = {}
        for name, mets in arm_stats.items():
            summary_b[name] = {}
            for k in ("brier", "auroc", "uar", "sensitivity", "specificity", "ece"):
                vals = [m[k] for m in mets if np.isfinite(m.get(k, np.nan))]
                if vals:
                    summary_b[name][k] = {
                        "mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                    }
        results.setdefault("summary", {})[f"B{B}"] = summary_b
        if verbose:
            for name in summary_b:
                s = summary_b[name]
                print(f"  {name}: brier={s['brier']['mean']:.4f} uar={s['uar']['mean']:.4f}")

    results["n_completed_runs"] = completed_total
    results["n_failed_runs"] = failed_total
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--budget", type=int, default=6)
    p.add_argument("--lambda", type=float, default=0.0, dest="lambda_")
    args = p.parse_args()
    t0 = time.time()
    results = benchmark(seeds=args.seeds, budgets=[args.budget], lambda_=args.lambda_)
    results["wall_seconds"] = time.time() - t0
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / "benchmark_multiseed_saudi.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for bkey, sdata in results.get("summary", {}).items():
        for arm, s in sdata.items():
            rows.append({"budget": bkey, "arm": arm, **{f"{k}_mean": v["mean"] for k, v in s.items()}})
    with open(RESULTS / "benchmark_multiseed_saudi.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_json} ({results['n_completed_runs']} completed, {results['n_failed_runs']} failed)")


if __name__ == "__main__":
    main()
