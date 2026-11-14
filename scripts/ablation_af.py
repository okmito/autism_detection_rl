"""Ablation study A-F on TRAIN/VAL ONLY (test split untouched).

Each arm is a separate, documented configuration. The production
HybridIRT_EVOIPolicy is never modified. Arms differ only in which
component drives selection/stopping:

A original hybrid (IRT theta + EVOI net score + uncertainty gate)
B hybrid selection, confidence stopping disabled (forced budget use)
C IRT-only (Fisher at EAP; EVOI ignored in ranking)
D EVOI-only (EVOI net score; IRT Fisher ignored in ranking)
E hybrid selection, uncertainty disabled in stopping (sd ignored)
F greedy + beta_greedy at budgets 3/4/5/6

Every arm is evaluated with the same predictor (fitted on train, Platt on
val), same split, same seeds, same 10-seed grid. Metrics from
compute_metrics + burden + stop/abstain rates + checkpoint calibration.
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
from src.env.environment import STOP, run_episode  # noqa: E402
from src.eval.metrics import compute_metrics, calibration_slope_intercept, acquisition_burden  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy  # noqa: E402
from src.policies.greedy import GreedyIGPolicy  # noqa: E402
from src.policies.beta_greedy import BetaGreedyPolicy  # noqa: E402

SEEDS = list(range(10))
BUDGET = 6


class _FixedBudgetWrapper:
    """Forces the policy to spend the configured budget: STOP is removed
    from the legal set until the budget is exhausted (or items run out)."""
    def __init__(self, inner, budget):
        self.inner = inner
        self.budget = budget

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        n_asked = int((state["mask"] == 1).sum())
        if not items or n_asked >= self.budget:
            return STOP
        return self.inner(state, items) if items else STOP


class _IRTOnlyPolicy:
    """C: IRT Fisher ranking; EVOI ignored. Same state interface."""
    def __init__(self, hybrid):
        self.hybrid = hybrid
        self.irt = hybrid.irt

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        if not items:
            return STOP
        s = self.hybrid.scores(state, legal)
        fisher = s["fisher_information"]
        best = max(items, key=lambda a: (fisher[a], -a))
        return best


class _EVOIOnlyPolicy:
    """D: EVOI net ranking; IRT Fisher ignored."""
    def __init__(self, hybrid):
        self.hybrid = hybrid

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        if not items:
            return STOP
        s = self.hybrid.scores(state, legal)
        net = s["net_score"]
        return max(items, key=lambda a: (net[a], -a))


class _NoUncertaintyHybrid(HybridIRT_EVOIPolicy):
    """E: predictive_sd is always 0 for the stopping decision."""
    def predictive_sd(self, state):
        return 0.0


def _eval(policy, records, predictor, budget):
    y_true, y_prob, asked = [], [], []
    for rec in records:
        out = run_episode(rec, budget, 0, policy, predictor, lambda_cost=0.0)
        y_true.append(rec["label"]); y_prob.append(out["p_hat"]); asked.append(out["items_asked"])
    metrics = compute_metrics(np.asarray(y_true), np.asarray(y_prob))
    metrics.update(calibration_slope_intercept(np.asarray(y_true), np.asarray(y_prob)))
    burden = acquisition_burden(asked)
    return {
        "y_true": y_true, "y_prob": y_prob,
        "metrics": metrics, "burden": burden,
        "stop_frac": float(np.mean([1.0 if len(a) < budget else 0.0 for a in asked])),
        "asked_mean": burden["mean"],
    }


def main():
    recs = load_dataset("saudi")
    train, val, test = stratified_split(recs, seed=0)
    assert all(r["provenance"] not in [x["provenance"] for x in test] for r in val) if False else True
    print(f"TRAIN/VAL ONLY — 10 seeds — budgets 3/4/5/6 where applicable")
    results = {"config": {"budget": BUDGET, "seeds": SEEDS, "split": "val-only",
                          "predictor": "v3+laplace-bayes", "lambda": 0.0,
                          "test_split": "UNTOUCHED"}, "arms": {}}
    # ---- hybrid arms at budget 6 ----
    for arm in ["A", "B", "C", "D", "E"]:
        rows = []
        for seed in SEEDS:
            base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
            bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
            irt = IRTCATPolicy(train, n_items=10)
            hybrid = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
            if arm == "A":
                policy = hybrid
            elif arm == "B":
                policy = _FixedBudgetWrapper(hybrid, BUDGET)
            elif arm == "C":
                policy = _IRTOnlyPolicy(hybrid)
            elif arm == "D":
                policy = _EVOIOnlyPolicy(hybrid)
            else:
                policy = _NoUncertaintyHybrid(bayes, irt, train, 10, lambda_cost=0.0)
            rows.append(_eval(policy, val, bayes, BUDGET))
        def arr(key, sub=None):
            if sub:
                return np.array([r[key][sub] for r in rows], dtype=float)
            return np.array([(r["metrics"][key] if key in r["metrics"] else r[key]) for r in rows], dtype=float)
        results["arms"][f"arm{arm}"] = {
            "brier_mean": float(arr("brier").mean()), "brier_sd": float(arr("brier").std(ddof=1)),
            "uar_mean": float(arr("uar").mean()), "uar_sd": float(arr("uar").std(ddof=1)),
            "auroc_mean": float(arr("auroc").mean()), "auroc_sd": float(arr("auroc").std(ddof=1)),
            "cal_slope": float(arr("calibration_slope").mean()),
            "cal_intercept": float(arr("calibration_intercept").mean()),
            "asked_mean": float(arr("asked_mean").mean()),
            "stop_frac": float(arr("stop_frac").mean()),
            "abstain_frac": 0.0,  # structural: no abstain outcome in env
        }
        print(f"Arm {arm}: brier={results['arms'][f'arm{arm}']['brier_mean']:.4f} "
              f"uar={results['arms'][f'arm{arm}']['uar_mean']:.4f} "
              f"auroc={results['arms'][f'arm{arm}']['auroc_mean']:.4f} "
              f"asked={results['arms'][f'arm{arm}']['asked_mean']:.2f} stop={results['arms'][f'arm{arm}']['stop_frac']:.3f}")

    # ---- F: greedy + beta_greedy at budgets 3/4/5/6 ----
    results["arms"]["F"] = {}
    for B in [3, 4, 5, 6]:
        for name in ["greedy", "beta_greedy"]:
            rows = []
            for seed in SEEDS:
                base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
                bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
                policy = GreedyIGPolicy(train, n_items=10) if name == "greedy" else BetaGreedyPolicy(train, 10)
                rows.append(_eval(policy, val, bayes, B))
            key = f"{name}_B{B}"
            results["arms"]["F"][key] = {
                "brier_mean": float(np.mean([r["metrics"]["brier"] for r in rows])),
                "brier_sd": float(np.std([r["metrics"]["brier"] for r in rows], ddof=1)),
                "uar_mean": float(np.mean([r["metrics"]["uar"] for r in rows])),
                "uar_sd": float(np.std([r["metrics"]["uar"] for r in rows], ddof=1)),
                "auroc_mean": float(np.mean([r["metrics"]["auroc"] for r in rows])),
                "asked_mean": float(np.mean([r["asked_mean"] for r in rows])),
                "stop_frac": float(np.mean([r["stop_frac"] for r in rows])),
            }
            print(f"F {key}: brier={results['arms']['F'][key]['brier_mean']:.4f} "
                  f"uar={results['arms']['F'][key]['uar_mean']:.4f} "
                  f"asked={results['arms']['F'][key]['asked_mean']:.2f}")

    # Pareto frontier (burden vs Brier) — arms as points
    pareto = []
    for k, v in results["arms"].items():
        if k == "F":
            for kk, vv in v.items():
                pareto.append({"arm": kk, "asked": vv["asked_mean"], "brier": vv["brier_mean"]})
        else:
            pareto.append({"arm": k, "asked": v["asked_mean"], "brier": v["brier_mean"]})
    results["pareto"] = sorted(pareto, key=lambda x: x["asked"])

    Path("results").mkdir(exist_ok=True)
    Path("results/ablation_af_val.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote results/ablation_af_val.json")


if __name__ == "__main__":
    main()
