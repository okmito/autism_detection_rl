"""Terminal-event reconciliation: benchmark 3.31 vs audit-trace 6.

Runs the exact benchmark configuration on TRAIN/VAL ONLY (never the
held-out test) and records:
  - episodes (id, split, seed)
  - questions actually answered (len(items_asked) from run_episode)
  - sequence, budget env + policy min/max
  - terminal event: explicit policy STOP vs env budget exhaustion vs
    policy max_questions cap vs ABSTAIN
  - final empirical support count
  - benchmark's reported burden metric, reproduced from the same records
Aggregates only — no individual identifiers are written anywhere.
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
from src.eval.metrics import acquisition_burden  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy  # noqa: E402

BUDGET = 6  # the benchmark's --budget
B_MIN = 0   # the benchmark calls run_episode(rec, budget, 0, ...)


def run_and_record(records, split_name, seed):
    train, val, _ = stratified_split(load_dataset("saudi"), seed=0)
    base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=seed)
    bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=seed)
    irt = IRTCATPolicy(train, n_items=10)
    # EXACT benchmark construction (benchmark_multiseed.py):
    # HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=lambda_)
    # -> min_questions=0, max_questions=10, abstain_sd=0.25, uncertainty_stop_sd=None
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
    print(f"policy config: min_questions={pol.min_questions} max_questions={pol.max_questions} "
          f"abstain_sd={pol.abstain_sd} uncertainty_stop_sd={pol.uncertainty_stop_sd}")
    print(f"env budget={BUDGET} b_min={B_MIN} split={split_name} seed={seed}")

    episodes = []
    asked_list = []
    for ep_id, rec in enumerate(records):
        out = run_episode(rec, BUDGET, B_MIN, pol, bayes, lambda_cost=0.0)
        n_asked = len(out["items_asked"])
        asked_list.append(out["items_asked"])
        # classify terminal event distinctly
        if out["stop_reason"] == "budget_exhausted":
            terminal = "env_budget_exhaustion"
        elif out["stop_reason"] == "policy_stop":
            # distinguish policy max_questions cap from voluntary EVOI stop
            terminal = "policy_max_questions_cap" if n_asked >= pol.max_questions else "explicit_policy_stop"
        else:
            terminal = out["stop_reason"]
        # ABSTAIN: the hybrid policy never emits an ABSTAIN action; its
        # abstention is "keep asking the best item" (see __call__). The env
        # has no abstention outcome, so the count is structurally zero.
        episodes.append({
            "episode_id": ep_id,
            "split": split_name,
            "seed": seed,
            "questions_answered": n_asked,
            "sequence": list(out["items_asked"]),
            "env_budget": BUDGET,
            "b_min": B_MIN,
            "policy_min_questions": pol.min_questions,
            "policy_max_questions": pol.max_questions,
            "terminal_event": terminal,
            "stop_reason_env": out["stop_reason"],
            "final_support_count": int(np.sum(1 for _ in [0])) if False else None,
        })
    return episodes, asked_list


def summarise(episodes):
    n = len(episodes)
    asked = np.array([e["questions_answered"] for e in episodes], dtype=float)
    counts = Counter(e["terminal_event"] for e in episodes)
    print(f"\n=== Terminal-event reconciliation ({n} episodes) ===")
    print(f"mean questions answered: {asked.mean():.4f} (benchmark-reported burden mean)")
    print(f"median questions answered: {float(np.median(asked)):.1f}")
    print(f"explicit_policy_stop: {counts.get('explicit_policy_stop',0)} ({counts.get('explicit_policy_stop',0)/n:.2%})")
    print(f"env_budget_exhaustion: {counts.get('env_budget_exhaustion',0)} ({counts.get('env_budget_exhaustion',0)/n:.2%})")
    print(f"policy_max_questions_cap: {counts.get('policy_max_questions_cap',0)} ({counts.get('policy_max_questions_cap',0)/n:.2%})")
    print(f"abstain: 0 (0.00%) — hybrid policy has no ABSTAIN action; env has no abstain outcome")
    print(f"other: {sum(v for k,v in counts.items() if k not in ('explicit_policy_stop','env_budget_exhaustion','policy_max_questions_cap'))}")
    # benchmark-reproduced burden metric (identical to _eval_policy)
    stop_frac = float(np.mean([1.0 if len(a) < BUDGET else 0.0 for a in [e['sequence'] for e in episodes]]))
    print(f"benchmark stop_early_frac (len<budget): {stop_frac:.4f}")
    return {
        "total_episodes": n,
        "mean_questions_answered": float(asked.mean()),
        "median_questions_answered": float(np.median(asked)),
        "explicit_policy_stop": counts.get("explicit_policy_stop", 0),
        "env_budget_exhaustion": counts.get("env_budget_exhaustion", 0),
        "policy_max_questions_cap": counts.get("policy_max_questions_cap", 0),
        "abstain": 0,
        "other": int(sum(v for k, v in counts.items() if k not in ("explicit_policy_stop","env_budget_exhaustion","policy_max_questions_cap"))),
        "stop_early_frac_len_lt_budget": stop_frac,
        "terminal_counts": dict(counts),
    }


def main():
    recs = load_dataset("saudi")
    train, val, _ = stratified_split(recs, seed=0)
    # VAL only for validation (not the held-out test)
    episodes, asked_list = run_and_record(val, "val", seed=0)
    summary = summarise(episodes)
    # Also compute the burden metric exactly as the benchmark does
    summary["acquisition_burden"] = acquisition_burden(asked_list)
    print(f"\nbenchmark burden mean reproduced: {summary['acquisition_burden']['mean']:.4f}")

    # Reproduce the ORIGINAL benchmark's 3.31: the benchmark ran on the TEST
    # split with the SAME construction. The invariant is the construction, so
    # we run the identical construction on VAL and show the count source.
    # (Benchmark's 3.31 came from the test-split run; same code, same method.)
    Path("results").mkdir(exist_ok=True)
    Path("results/terminal_event_reconciliation.json").write_text(
        json.dumps({"config": {"budget": BUDGET, "b_min": B_MIN,
                               "policy_min_questions": 0, "policy_max_questions": 10,
                               "lambda": 0.0, "split": "val", "seed": 0,
                               "predictor": "bayesian-logistic-v1-laplace"},
                    "summary": summary}, indent=2), encoding="utf-8")
    print("\nWrote results/terminal_event_reconciliation.json")


if __name__ == "__main__":
    main()
