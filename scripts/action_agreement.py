"""Action-level agreement between hybrid components (VAL only).

Reuses production HybridIRT_EVOIPolicy (never modified). Implements IRT-only
and EVOI-only as thin wrappers over the production policy's own scores(),
i.e. NO independent reimplementation of EVOI/Fisher. Trains once per seed,
evaluates on VAL with full-budget use (STOP removed from env via
_fixed-budget wrapper) so selection differences are isolated from stopping.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.ingest import load_dataset  # noqa: E402
from src.data.splits import stratified_split  # noqa: E402
from src.env.environment import STOP, run_episode  # noqa: E402
from src.eval.metrics import compute_metrics, acquisition_burden  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy  # noqa: E402
from src.policies.greedy import GreedyIGPolicy  # noqa: E402
from src.policies.beta_greedy import BetaGreedyPolicy  # noqa: E402

SEEDS = list(range(10))
BUDGET = 6


class _FixedBudget:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, state, legal):
        items = [a for a in legal if a != STOP]
        if not items:
            return STOP
        return self.inner(state, items)


def _sequences(policy, records, predictor):
    """Run episodes, collecting only (sequence, p_hat, asked) — no ids."""
    seqs, probs, labels = [], [], []
    for rec in records:
        out = run_episode(rec, BUDGET, 0, policy, predictor, lambda_cost=0.0)
        seqs.append(tuple(out["items_asked"]))
        probs.append(out["p_hat"])
        labels.append(rec["label"])
    return seqs, np.asarray(probs), np.asarray(labels)


def main():
    recs = load_dataset("saudi")
    train, val, _ = stratified_split(recs, seed=0)
    print("VAL only (95 subjects); same predictor per seed; full budget enforced.")

    arms = ["hybrid_full", "evoi_only", "irt_only", "beta_greedy", "greedy"]
    # accumulate: per-subject across seeds? Per policy: list of sequences per seed.
    per_arm_seqs = {a: [] for a in arms}
    per_arm_metrics = {a: [] for a in arms}
    for seed in SEEDS:
        base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
        bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
        irt = IRTCATPolicy(train, n_items=10)
        hybrid = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)

        class _EvoiOnly:
            def __call__(self, state, legal):
                s = hybrid.scores(state, legal)
                items = [a for a in legal if a != STOP]
                return max(items, key=lambda a: (s["net_score"][a], -a))

        class _IrtOnly:
            def __call__(self, state, legal):
                s = hybrid.scores(state, legal)
                items = [a for a in legal if a != STOP]
                return max(items, key=lambda a: (s["fisher_information"][a], -a))

        pols = {
            "hybrid_full": _FixedBudget(hybrid),
            "evoi_only": _FixedBudget(_EvoiOnly()),
            "irt_only": _FixedBudget(_IrtOnly()),
            "beta_greedy": BetaGreedyPolicy(train, 10),
            "greedy": GreedyIGPolicy(train, n_items=10),
        }
        for a in arms:
            seqs, probs, labels = _sequences(pols[a], val, bayes)
            per_arm_seqs[a].append(seqs)
            per_arm_metrics[a].append((probs, labels))

    # ---- agreement ----
    def flat(arm):
        return [tuple(s) for seed in per_arm_seqs[arm] for s in seed]

    print("\n=== Exact full-sequence agreement ===")
    for a, b in [("hybrid_full", "evoi_only"), ("hybrid_full", "irt_only"),
                 ("evoi_only", "irt_only"), ("hybrid_full", "beta_greedy"),
                 ("hybrid_full", "greedy"), ("evoi_only", "greedy")]:
        seq_a, seq_b = flat(a), flat(b)
        agree = float(np.mean([x == y for x, y in zip(seq_a, seq_b)]))
        print(f"{a} vs {b}: exact={agree:.3f}")

    print("\n=== First-question agreement ===")
    for a, b in [("hybrid_full", "evoi_only"), ("hybrid_full", "irt_only"),
                 ("hybrid_full", "beta_greedy"), ("hybrid_full", "greedy")]:
        fa = [s[0] if s else -1 for s in flat(a)]
        fb = [s[0] if s else -1 for s in flat(b)]
        print(f"{a} vs {b}: first={float(np.mean(np.array(fa)==np.array(fb))):.3f}")

    print("\n=== Per-depth prefix agreement (hybrid vs EVOI-only / IRT-only / greedy) ===")
    for b in ["evoi_only", "irt_only", "greedy", "beta_greedy"]:
        print(f"-- hybrid vs {b} --")
        for d in [1, 2, 3, 6]:
            fa = [s[:d] for s in flat("hybrid_full")]
            fb = [s[:d] for s in flat(b)]
            agree = float(np.mean([x == y for x, y in zip(fa, fb)]))
            print(f"  depth {d}: {agree:.3f}")

    # ---- action-ranking differences ----
    print("\n=== Action-ranking: mean agreement on top item ===")
    for b in ["evoi_only", "irt_only", "beta_greedy", "greedy"]:
        # compare first action only (decision point 0) — exact sequences above
        # already cover full ranking; this reports top-1 at depth 0.
        fa = [s[0] if s else -1 for s in flat("hybrid_full")]
        fb = [s[0] if s else -1 for s in flat(b)]
        print(f"hybrid vs {b}: top@0 {float(np.mean(np.array(fa)==np.array(fb))):.3f}")

    # ---- disagreement counts ----
    print("\n=== Disagreement (episodes where hybrid != evoi_only, any depth) ===")
    all_h = flat("hybrid_full"); all_e = flat("evoi_only")
    disag = [i for i, (x, y) in enumerate(zip(all_h, all_e)) if x != y]
    print(f"disagreeing episodes: {len(disag)}/950")

    # ---- metrics summary ----
    print("\n=== VAL metrics (10 seeds, full budget) ===")
    for a in arms:
        brs, uars, asks = [], [], []
        for probs, labels in per_arm_metrics[a]:
            m = compute_metrics(labels, probs)
            brs.append(m["brier"]); uars.append(m["uar"])
        print(f"{a:14s} brier={np.mean(brs):.4f}+/-{np.std(brs,ddof=1):.4f} "
              f"uar={np.mean(uars):.4f}+/-{np.std(uars,ddof=1):.4f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/action_agreement_val.json").write_text(json.dumps(
        {"note": "VAL-only. Actions = item sequences (full budget, STOP disabled).",
         "exact_hybrid_vs_evoi": float(np.mean([x==y for x,y in zip(flat('hybrid_full'),flat('evoi_only'))])),
         "exact_evoi_vs_irt": float(np.mean([x==y for x,y in zip(flat('evoi_only'),flat('irt_only'))])),
         "exact_hybrid_vs_irt": float(np.mean([x==y for x,y in zip(flat('hybrid_full'),flat('irt_only'))]))},
        indent=2), encoding="utf-8")
    print("\nWrote results/action_agreement_val.json")


if __name__ == "__main__":
    main()
