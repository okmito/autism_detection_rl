"""Paired bootstrap: hybrid IRT+EVOI vs each principal comparator.

Subject-level bootstrap: resample test subjects (paired by subject index)
using predictions recorded per run. Per-subject hybrid vs comparator
predictions are recovered from the stored per-episode outputs? No: the
benchmark_multiseed artifact stores only aggregate metrics. To get
per-subject pairing, this script re-runs ONLY the evaluation portion
(one shared predictor per seed, same seed grid) and records per-subject
p_hat for each arm, then paired-bootstraps the differences.

This is an analysis script — it trains nothing new and touches only the
held-out test split for evaluation (same as the runner).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.ingest import load_dataset  # noqa: E402
from src.data.splits import stratified_split  # noqa: E402
from src.env.environment import run_episode  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402
from src.policies.beta_greedy import BetaGreedyPolicy  # noqa: E402
from src.policies.greedy import GreedyIGPolicy  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy  # noqa: E402
from src.policies.static_fixed import ExactFixedSubsetPolicy  # noqa: E402


def main():
    records = load_dataset("saudi")
    train, val, test = stratified_split(records, seed=0)
    y = np.array([r["label"] for r in test])
    n = len(y)
    seeds = list(range(10))
    arms = ["hybrid_irt_evoi", "greedy", "beta_greedy", "exact_fixed_subset", "irt_cat"]
    # accumulate per-subject p_hat per arm, averaged over seeds
    acc = {a: np.zeros(n) for a in arms}
    for seed in seeds:
        base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
        bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
        irt = IRTCATPolicy(train, n_items=10)
        pols = {
            "greedy": GreedyIGPolicy(train, n_items=10),
            "beta_greedy": BetaGreedyPolicy(train, 10),
            "irt_cat": IRTCATPolicy(train, n_items=10),
            "exact_fixed_subset": ExactFixedSubsetPolicy(train, n_items=10, budget=6),
            "hybrid_irt_evoi": HybridIRT_EVOIPolicy(bayes, irt, train, 10),
        }
        for a in arms:
            for i, rec in enumerate(test):
                out = run_episode(rec, 6, 0, pols[a], bayes, lambda_cost=0.0)
                acc[a][i] += out["p_hat"] / len(seeds)

    def brier(y_, p):
        return float(np.mean((p - y_) ** 2))

    rng = np.random.default_rng(123)
    n_boot = 2000
    out = {}
    for comp in [a for a in arms if a != "hybrid_irt_evoi"]:
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            diffs.append(brier(y[idx], acc["hybrid_irt_evoi"][idx]) - brier(y[idx], acc[comp][idx]))
        diffs = np.array(diffs)
        out[f"hybrid_vs_{comp}"] = {
            "observed_diff_brier": float(brier(y, acc["hybrid_irt_evoi"]) - brier(y, acc[comp])),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "se": float(diffs.std(ddof=1)),
            "n_resamples": int(n_boot),
            "interpretation": "positive diff = hybrid worse (higher Brier); negative = hybrid better",
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/paired_bootstrap_hybrid.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
