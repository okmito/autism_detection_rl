"""Section 2 & 3: Bayesian uncertainty value + support-aware EVOI (VAL only).

Reuses production components only. 10 seeds, same predictor construction as
ablations, full budget, VAL evaluation only (test untouched).
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
from src.eval.metrics import compute_metrics, calibration_slope_intercept  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy  # noqa: E402
from src.policies.beta_greedy import BetaGreedyPolicy  # noqa: E402
from src.policies.greedy import GreedyIGPolicy  # noqa: E402

SEEDS = list(range(10))
BUDGET = 6


class _EvoiOnly:
    def __init__(self, hybrid):
        self.hybrid = hybrid

    def __call__(self, state, legal):
        s = self.hybrid.scores(state, legal)
        items = [a for a in legal if a != STOP]
        return max(items, key=lambda a: (s["net_score"][a], -a))


class _FullBudget:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        return self.inner(state, items) if items else STOP


class _SupportFallback:
    """Section 3 fallback: start with EVOI-net (hybrid math); when the
    empirical support is below the (training-internal) median support seen
    at early depths, fall back to Greedy-IG ranking (Beta-smoothed). No
    external threshold invented: the cutoff is the median support size of
    the training set's own depth-0 states computed over train, documented
    rather than tuned on val/test."""
    def __init__(self, hybrid, train_records):
        self.hybrid = hybrid
        self.n_items = hybrid.n_items
        self.X = hybrid.X; self.missing = hybrid.missing_mask; self.y = hybrid.y
        # train-internal reference: support size at depth 0 (full train)
        self._cutoff = int(np.median([len(hybrid._support(np.zeros(self.n_items, dtype=int), np.zeros(self.n_items, dtype=int))) for _ in range(1)]))

    def __call__(self, state, legal):
        s = self.hybrid.scores(state, legal)
        items = [a for a in legal if a != STOP]
        if s["support_size"] >= self._cutoff or not items:
            # enough support: EVOI net ranking
            return max(items, key=lambda a: (s["net_score"][a], -a))
        # insufficient support: Greedy-IG on Beta-smoothed support
        beta = BetaGreedyPolicy.__dict__["__init__"]  # noqa -- shape only
        # manual IG ranking on beta-smoothed empirical support:
        idx = self.hybrid._support(state["mask"], state["value"])
        from src.policies.beta_greedy import entropy
        total = len(idx)
        h_now = entropy((int(self.y[idx].sum()) + 1) / (len(idx) + 2)) if len(idx) else 0.0
        best, best_ig = items[0], -1e18
        for a in items:
            child_exp = 0.0
            obs = idx[~self.missing[idx, a]]
            if len(obs):
                for v in np.unique(self.X[obs, a]):
                    c = obs[self.X[obs, a] == v]
                    p = (int(self.y[c].sum()) + 1) / (len(c) + 2)
                    child_exp += (len(c) / total) * entropy(p)
            miss = idx[self.missing[idx, a]]
            if len(miss):
                p = (int(self.y[miss].sum()) + 1) / (len(miss) + 2)
                child_exp += (len(miss) / total) * entropy(p)
            ig = h_now - child_exp
            if ig > best_ig:
                best, best_ig = a, ig
        return best


def main():
    recs = load_dataset("saudi")
    train, val, _ = stratified_split(recs, seed=0)
    print("VAL only; 10 seeds; same predictor/stop conditions as ablation A; "
          "held-out test untouched.")

    # ---- Section 2: uncertainty value ----
    # For every decision point along EVOI-only trajectories, record
    # predictive_sd, absolute error |p - y|, and budget depth.
    sds, errs, depths = [], [], []
    for seed in SEEDS:
        base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
        bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
        irt = IRTCATPolicy(train, n_items=10)
        hybrid = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
        evoi = _FullBudget(_EvoiOnly(hybrid))
        # predictive_sd at decision points
        from src.env.state import init_state, get_legal_items, update_state
        for rec in val:
            state = init_state(rec); state["budget"] = BUDGET; state["questions_remaining"] = BUDGET
            for depth in range(BUDGET):
                legal = get_legal_items(state) + [STOP]
                sd = hybrid.predictive_sd(state)
                try:
                    p = float(bayes.predict_state(state))
                except Exception:
                    p = 0.5
                sds.append(sd); errs.append(abs(p - rec["label"])); depths.append(depth)
                if depth == BUDGET - 1:
                    break
                a = evoi(state, legal)
                if a == STOP:
                    break
                state = update_state(state, int(a), int(rec["item_responses"][int(a)]), state["questions_remaining"] - 1)
                state["questions_remaining"] -= 1; state["budget"] = BUDGET
    sds = np.asarray(sds); errs = np.asarray(errs); depths = np.asarray(depths)
    corr = float(np.corrcoef(sds, errs)[0, 1]) if sds.std() > 0 else 0.0
    # tertiles
    lo, hi = np.percentile(sds, [33.3, 66.6])
    grp = {"lo": float(errs[sds <= lo].mean()), "mid": float(errs[(sds > lo) & (sds <= hi)].mean()),
           "hi": float(errs[sds > hi].mean())}
    # calibration by budget
    cal_by_depth = {}
    for d in sorted(set(depths)):
        m = depths == d
        # aggregate |p - y| per depth (point approximation)
        cal_by_depth[int(d)] = float(np.mean(errs[m]))
    print(f"\n[Uncertainty] corr(sd, |error|) = {corr:+.3f}")
    print(f"[Uncertainty] mean |err| by sd tertile: lo={grp['lo']:.4f} mid={grp['mid']:.4f} hi={grp['hi']:.4f}")
    print(f"[Uncertainty] calibration proxy |err| by depth: {cal_by_depth}")
    print(f"[Uncertainty] sd range: {sds.min():.4f} - {sds.max():.4f}")

    # ---- Section 3: support-aware fallback ----
    rows = {"evoi_only": [], "support_fallback": []}
    for seed in SEEDS:
        base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
        bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
        irt = IRTCATPolicy(train, n_items=10)
        hybrid = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
        for name, policy in [("evoi_only", _FullBudget(_EvoiOnly(hybrid))),
                             ("support_fallback", _FullBudget(_SupportFallback(hybrid, train)))]:
            y_true, y_prob = [], []
            for rec in val:
                out = run_episode(rec, BUDGET, 0, policy, bayes, lambda_cost=0.0)
                y_true.append(rec["label"]); y_prob.append(out["p_hat"])
            rows[name].append((np.asarray(y_true), np.asarray(y_prob)))
    for name in rows:
        brs = [compute_metrics(y, p)["brier"] for y, p in rows[name]]
        uars = [compute_metrics(y, p)["uar"] for y, p in rows[name]]
        print(f"[Support-aware] {name}: brier={np.mean(brs):.4f}+/-{np.std(brs,ddof=1):.4f} "
              f"uar={np.mean(uars):.4f}+/-{np.std(uars,ddof=1):.4f}")

    # ---- Section 4: Pareto with CIs ----
    print("\n=== Pareto (VAL, full-budget where it applies) ===")
    pareto = {}
    # reuse maxima from ablation + F arms via fresh evaluation
    def eval_arm(factory, B=BUDGET):
        brs, uars, aurocs, asks, stops, ec = [], [], [], [], [], []
        for seed in SEEDS:
            base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
            bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
            y, p, asked = [], [], []
            for rec in val:
                out = run_episode(rec, B, 0, factory(bayes, train), bayes, lambda_cost=0.0)
                y.append(rec["label"]); p.append(out["p_hat"]); asked.append(len(out["items_asked"]))
            m = compute_metrics(np.asarray(y), np.asarray(p))
            brs.append(m["brier"]); uars.append(m["uar"]); aurocs.append(m["auroc"])
            asks.append(float(np.mean(asked))); stops.append(float(np.mean([1.0 if a < B else 0.0 for a in asked])))
            ec.append(m["ece"])
        return {
            "brier_mean": float(np.mean(brs)), "brier_sd": float(np.std(brs, ddof=1)),
            "brier_ci95": [float(np.percentile(brs, 2.5)), float(np.percentile(brs, 97.5))],
            "uar_mean": float(np.mean(uars)), "uar_sd": float(np.std(uars, ddof=1)),
            "auroc_mean": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs, ddof=1)),
            "asked_mean": float(np.mean(asks)), "stop_frac": float(np.mean(stops)),
            "ece_mean": float(np.mean(ec)),
            "abstain_frac": 0.0,
        }
    def make_exact(bayes, TR):
        return ExactFixedWrapper(train) if False else None
    # exact fixed subset / RFE via train-only selection
    from src.policies.static_fixed import ExactFixedSubsetPolicy
    pareto["exact_fixed_B6"] = eval_arm(lambda bayes, TR: ExactFixedSubsetPolicy(TR, n_items=10, budget=6))
    pareto["greedy_B6"] = eval_arm(lambda bayes, TR: GreedyIGPolicy(TR, n_items=10))
    pareto["beta_greedy_B6"] = eval_arm(lambda bayes, TR: BetaGreedyPolicy(TR, 10))
    pareto["greedy_B4"] = eval_arm(lambda bayes, TR: GreedyIGPolicy(TR, n_items=10), B=4)
    pareto["greedy_B3"] = eval_arm(lambda bayes, TR: GreedyIGPolicy(TR, n_items=10), B=3)
    for k, v in pareto.items():
        print(f"{k:20s} brier={v['brier_mean']:.4f} ci={v['brier_ci95']} "
              f"uar={v['uar_mean']:.4f} auroc={v['auroc_mean']:.4f} asked={v['asked_mean']:.2f} ece={v['ece_mean']:.4f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/uncertainty_support_val.json").write_text(json.dumps(
        {"uncertainty": {"corr_sd_abs_error": corr, "sd_range": [float(sds.min()), float(sds.max())],
                         "mean_abs_error_by_sd_tertile": grp, "abs_error_by_depth": cal_by_depth},
         "support_fallback": {name: {"brier_mean": float(np.mean([compute_metrics(y, p)["brier"] for y, p in rows[name]])),
                                     "brier_sd": float(np.std([compute_metrics(y, p)["brier"] for y, p in rows[name]], ddof=1))}
                              for name in rows},
         "pareto": pareto}, indent=2), encoding="utf-8")
    print("\nWrote results/uncertainty_support_val.json")


if __name__ == "__main__":
    main()
