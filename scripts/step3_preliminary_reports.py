"""Step 3 — preliminary report artifacts (perf-vs-budget, faithfulness, subgroup).

§19, §20. ALL outputs are tagged "preliminary — V-4 / V-6 PENDING" because:
  - V-4: MDE / comparison family not yet frozen.
  - V-6: λ grid not yet supervisor-signed.
  - V-7: Polish denominator not yet frozen.

We do NOT run:
  - cost-utility vs λ figures (V-6 PENDING)
  - confirmatory Polish evaluations (V-4 / V-7 PENDING)
  - novelty claims (V-2 PENDING)

We DO emit:
  - Performance-vs-budget on Saudi at B ∈ {1..6}, λ=0, with GreedyIG and Random policies
    compared against a "terminal (B=10, all observed)" reference.
  - Faithfulness: SHAP-like perturbation attributions + counterfactual-validity rate
    on a held-out test set.
  - Subgroup: UAR/Brier by sex and age_band on the Saudi test set.

Terminology lock (§25): every output is "screening" / "referral recommendation",
never "diagnosis". Saudi/UCI circularity status is displayed next to every result
(§16.1 gate).

Run from repo root:
  /tmp/aar/bin/python scripts/step3_preliminary_reports.py
"""
from __future__ import annotations
import json
import os
import sys
import time
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.ingest import load_dataset, SAUDI_CSV
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from src.env.environment import run_episode
from src.eval.metrics import compute_metrics, acquisition_burden
from src.eval.subgroup import subgroup_report
from src.explain.shap_baseline import shap_for_static_subset
from src.explain.counterfactual import find_counterfactual
from src.env.state import init_state, update_state

RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

PRELIM_TAG = "preliminary — V-4 / V-6 / V-7 PENDING; supervisor sign-off required"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def _split(records, seed: int = 0):
    y = np.array([r["label"] for r in records])
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(np.arange(len(records)), y))
    train_idx, val_idx = next(skf.split(train_idx, y[train_idx]))
    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


def _run_episodes(records, policy, predictor, B: int, seed: int = 0) -> list[dict]:
    """Run episodes at question budget B and return one result dict per record."""
    out = []
    for rec in records:
        ep = run_episode(
            rec,
            question_budget=B,
            b_min=0,
            policy=policy,
            predictor=predictor,
            lambda_cost=0.0,  # V-6 PENDING; see config.yaml
            cost_mode="uniform",
            tau=0.5,
        )
        out.append(ep)
    return out


def _y_and_p_from_eps(eps):
    y = np.array([e["trace"][-1]["belief_before"]  # not used; placeholder
                  for e in eps])
    p = np.array([e["p_hat"] for e in eps])
    return p


def _label(eps):
    return np.array([e["decision"] for e in eps])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print(f"Step 3 — preliminary report artifacts ({PRELIM_TAG})")
    print(f"  repo:  {REPO}")
    print(f"  git:   {_git_sha()}")
    print(f"  py:    {platform.python_version()}")
    print("=" * 70)

    # ----- Load Saudi (real or synthetic) -----
    try:
        saudi = load_dataset("saudi", synthetic=False)
        saudi_source = "real" if SAUDI_CSV.exists() else "synthetic"
        circularity = "Deterministic (questionnaire-derived, §16.1 gate applies)"
    except FileNotFoundError as e:
        print(f"[saudi] real data not present → falling back to synthetic: {e}")
        saudi = load_dataset("saudi", synthetic=True)
        saudi_source = "synthetic"
        circularity = "Synthetic (real Saudi is gitignored and absent in this env)"

    print(f"[saudi] {len(saudi)} records, source={saudi_source}")
    print(f"[saudi] circularity status: {circularity}")

    # ----- Train predictor (Saudi) -----
    train, val, test = _split(saudi, seed=0)
    pred = MaskedPredictor(n_items=10, hidden=[128, 64], calibration="isotonic")
    pred.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=0)
    pred.fit_calibrator(val, method="isotonic")

    # ----- 1. Performance vs budget (Saudi test) -----
    print()
    print("Performance vs budget (Saudi test, λ=0, V-6 PENDING):")
    budgets = [1, 2, 3, 4, 5, 6]
    rows = []
    for B in budgets:
        ep_greedy = _run_episodes(test, GreedyIGPolicy(train, n_items=10), pred, B=B)
        ep_random = _run_episodes(test, RandomPolicy(seed=0), pred, B=B)
        p_g = np.array([e["p_hat"] for e in ep_greedy])
        p_r = np.array([e["p_hat"] for e in ep_random])
        y = np.array([e["trace"][-1]["value"] if False else int(0) for e in ep_greedy])  # unused
        # use record-level labels
        y_true = np.array([r["label"] for r in test])
        m_g = compute_metrics(y_true, p_g, tau=0.5)
        m_r = compute_metrics(y_true, p_r, tau=0.5)
        ab_g = acquisition_burden([e["items_asked"] for e in ep_greedy])
        ab_r = acquisition_burden([e["items_asked"] for e in ep_random])
        row = {
            "B": B,
            "policy_greedy_brier": m_g["brier"],
            "policy_greedy_uar": m_g["uar"],
            "policy_greedy_auroc": m_g["auroc"],
            "policy_greedy_ece": m_g["ece"],
            "policy_greedy_mean_items_asked": ab_g["mean"],
            "policy_random_brier": m_r["brier"],
            "policy_random_uar": m_r["uar"],
            "policy_random_auroc": m_r["auroc"],
            "policy_random_ece": m_r["ece"],
            "policy_random_mean_items_asked": ab_r["mean"],
        }
        rows.append(row)
        print(
            f"  B={B}  greedy brier={m_g['brier']:.4f} uar={m_g['uar']:.4f} "
            f"items={ab_g['mean']:.2f}  |  random brier={m_r['brier']:.4f} uar={m_r['uar']:.4f} "
            f"items={ab_r['mean']:.2f}"
        )

    # Also a terminal (B=10) reference: Brier/UAR of the full-observation predictor
    p_term = []
    y_term = []
    for rec in test:
        s = init_state(rec)
        for j in range(10):
            if not rec["missing_mask"][j]:
                s = update_state(s, j, int(rec["item_responses"][j]), 0)
        s["questions_remaining"] = 0
        s["budget"] = 10
        p_term.append(pred(s))
        y_term.append(rec["label"])
    p_term = np.array(p_term)
    y_term = np.array(y_term)
    m_term = compute_metrics(y_term, p_term, tau=0.5)
    ref = {
        "terminal_B10_brier": m_term["brier"],
        "terminal_B10_uar": m_term["uar"],
        "terminal_B10_auroc": m_term["auroc"],
        "terminal_B10_ece": m_term["ece"],
    }
    print(
        f"  REF B=10  brier={m_term['brier']:.4f} uar={m_term['uar']:.4f} "
        f"auroc={m_term['auroc']:.4f} ece={m_term['ece']:.4f}"
    )

    perf_path = RESULTS / "perf_vs_budget_saudi.csv"
    with perf_path.open("w") as f:
        f.write("B,policy,items_asked_mean,brier,uar,auroc,ece\n")
        for r in rows:
            f.write(
                f"{r['B']},greedy,{r['policy_greedy_mean_items_asked']:.4f},"
                f"{r['policy_greedy_brier']:.6f},{r['policy_greedy_uar']:.6f},"
                f"{r['policy_greedy_auroc']:.6f},{r['policy_greedy_ece']:.6f}\n"
            )
            f.write(
                f"{r['B']},random,{r['policy_random_mean_items_asked']:.4f},"
                f"{r['policy_random_brier']:.6f},{r['policy_random_uar']:.6f},"
                f"{r['policy_random_auroc']:.6f},{r['policy_random_ece']:.6f}\n"
            )
        f.write(
            f"10,terminal_full_observation,10.0000,"
            f"{m_term['brier']:.6f},{m_term['uar']:.6f},"
            f"{m_term['auroc']:.6f},{m_term['ece']:.6f}\n"
        )
    print(f"→ {perf_path.name}")

    perf_meta = {
        "tag": PRELIM_TAG,
        "dataset": "saudi",
        "source": saudi_source,
        "circularity_status": circularity,
        "lambda": 0.0,
        "config": str(REPO / "configs" / "config.yaml"),
        "git_sha": _git_sha(),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": 0,
        "rows": rows,
        "terminal_reference": ref,
    }
    (RESULTS / "perf_vs_budget_saudi.json").write_text(json.dumps(perf_meta, indent=2))

    # ----- 2. Faithfulness: counterfactual + SHAP -----
    print()
    print("Faithfulness (counterfactual + SHAP, Saudi test, B=6):")
    eps_term = _run_episodes(test, GreedyIGPolicy(train, n_items=10), pred, B=6)
    cf_found = 0
    cf_robust = 0
    cf_items = []
    for ep in eps_term:
        s = ep["final_state"]
        cf = find_counterfactual(s, ep["p_hat"], pred, tau=0.5)
        if cf is not None:
            cf_found += 1
            cf_items.append(cf["item"])
        else:
            cf_robust += 1
    cf_rate = cf_found / len(eps_term)
    cf_robust_rate = cf_robust / len(eps_term)

    # SHAP on terminal state
    shap_means = {f"A{j+1}": [] for j in range(10)}
    for ep in eps_term:
        sh = shap_for_static_subset(test, ep["final_state"], pred)
        for k, v in sh["attributions"].items():
            shap_means[k].append(abs(v))
    shap_summary = {k: float(np.mean(v)) if v else 0.0 for k, v in shap_means.items()}

    faithful = {
        "tag": PRELIM_TAG,
        "dataset": "saudi",
        "source": saudi_source,
        "circularity_status": circularity,
        "n_episodes": len(eps_term),
        "B": 6,
        "tau": 0.5,
        "counterfactual_found_rate": cf_rate,
        "counterfactual_robust_rate": cf_robust_rate,
        "counterfactual_items_top": _topk(cf_items, k=10),
        "shap_abs_attribution_mean": shap_summary,
    }
    (RESULTS / "faithfulness_saudi.json").write_text(json.dumps(faithful, indent=2))
    print(
        f"  counterfactual flips {cf_found}/{len(eps_term)} = {cf_rate:.3f}  "
        f"|  robust {cf_robust}/{len(eps_term)} = {cf_robust_rate:.3f}"
    )
    print(f"  top SHAP |attr| mean: {sorted(shap_summary.items(), key=lambda x:-x[1])[:3]}")

    # ----- 3. Subgroup analysis -----
    print()
    print("Subgroup analysis (Saudi test, B=6, τ=0.5):")
    p_arr = np.array([ep["p_hat"] for ep in eps_term])
    sub = subgroup_report(test, p_arr, tau=0.5, min_cell=20)
    sub_out = {
        "tag": PRELIM_TAG,
        "dataset": "saudi",
        "source": saudi_source,
        "circularity_status": circularity,
        "tau": 0.5,
        "B": 6,
        "n_test": len(test),
        "subgroups": sub,
        "note": "Differences LOGGED, never suppressed (§25). Underpowered cells marked.",
    }
    (RESULTS / "subgroup_saudi.json").write_text(json.dumps(sub_out, indent=2, default=str))
    for k, v in list(sub.items())[:8]:
        print(f"  {k}: {v}")

    print()
    print("DONE — preliminary artifacts in results/. All tagged", PRELIM_TAG)
    return 0


def _topk(items, k: int):
    from collections import Counter
    return dict(Counter(items).most_common(k))


if __name__ == "__main__":
    raise SystemExit(main())
