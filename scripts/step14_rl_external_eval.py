"""Step 14 — evaluate adaptive question selection on the sealed Polish cohort.

Runs ONCE, after scripts/step13_rl_protocol.py froze the protocol. Nothing here may
influence the protocol: every hyperparameter, seed, budget, baseline and metric is
read from the frozen artifact, and this script tunes nothing.

The research question is question EFFICIENCY:

    How many of the 10 Q-CHAT-10 questions can be removed while retaining
    acceptable predictive performance against clinician-established labels?

Success is NOT "beating the model". The reference is the full-question item count.

MANDATORY CAVEAT — this is external validation against clinician-established ASD
labels. It is NOT a diagnosis, NOT clinical validation, and NOT a basis for
deployment.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.provenance import git_sha, write_artifact

TAG = ("adaptive question-selection external evaluation; Polish clinician-"
       "established labels; research prototype, NOT a diagnostic device; protocol "
       "frozen before the cohort was opened; nothing tuned here")


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ss = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def paired_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray,
              n_boot: int, seed: int) -> Dict[str, Any]:
    """Paired percentile bootstrap on the AUROC difference a - b."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    diffs: List[float] = []
    degenerate = 0
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, size=y.size)
        ys = y[idx]
        if np.unique(ys).size < 2:
            degenerate += 1
            continue
        aa, bb = fast_auroc(ys, np.asarray(a, dtype=float)[idx]), \
            fast_auroc(ys, np.asarray(b, dtype=float)[idx])
        if np.isfinite(aa) and np.isfinite(bb):
            diffs.append(aa - bb)
    arr = np.asarray(diffs)
    if arr.size == 0:
        return {"status": "UNDEFINED", "reason": "no usable resamples"}
    return {
        "point_estimate": float(fast_auroc(y, np.asarray(a, dtype=float))
                                - fast_auroc(y, np.asarray(b, dtype=float))),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_resamples": int(n_boot),
        "n_degenerate_excluded": int(degenerate),
        "seed": int(seed),
        "method": "paired percentile bootstrap, identical resamples",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("Step 14 - adaptive question selection on the sealed Polish cohort")
    print(f"  {TAG}")
    print("=" * 78)

    from src.data.ingest import POLISH_CSV
    from src.data.qchat10_contract import (MODEL_FEATURE_ORDER,
                                           build_feature_matrix)
    from src.env.environment import STOP, run_episode
    from src.eval.baselines import (REQUIRED_BASELINES, assert_all_baselines,
                                    missing_baselines)
    from src.eval.external_metrics import calibration_diagnostics, core_metrics
    from src.models.logistic_predictor import (N_ITEMS, PREDICTOR_VERSION,
                                               LogisticPredictor, sha256_of)
    from src.policies.beta_greedy import BetaGreedyPolicy
    from src.policies.dqn import DQNPolicy
    from src.policies.greedy import GreedyIGPolicy
    from src.policies.random_policy import RandomPolicy

    protocol = json.loads(
        (RESULTS / "polish_rl_external_protocol.json").read_text(encoding="utf-8"))
    budgets = list(protocol["budgets"])
    tau = float(protocol["metrics"]["tau_primary"])
    tau_secondary = float(protocol["metrics"]["tau_secondary_sensitivity_only"])
    n_boot = args.bootstrap or int(protocol["statistical_method"]["n_resamples"])
    seed = int(protocol["seed"])
    lambda_cost = float(protocol["reward"]["lambda_cost"])

    print("\n[1/6] verify the frozen protocol and predictor")
    predictor_path = REPO / protocol["predictor"]["artifact"]
    actual_sha = sha256_of(predictor_path)
    sha_ok = actual_sha == protocol["predictor"]["sha256"]
    print(f"   protocol frozen before Polish : {protocol['frozen_before_polish']}")
    print(f"   predictor {PREDICTOR_VERSION}")
    print(f"   sha256 matches protocol       : {sha_ok}")
    print(f"   budgets                       : {budgets}")
    print(f"   primary tau                   : {tau}")
    print(f"   equivalence margin            : "
          f"{protocol['equivalence'].get('acceptable_AUROC_margin')} "
          f"({protocol['equivalence']['state']})")
    if not sha_ok:
        print("   -> BLOCKED: predictor does not match the frozen protocol")
        return 2

    predictor = LogisticPredictor.load(predictor_path)

    print("\n[2/6] project the sealed cohort")
    import pandas as pd
    df = pd.read_csv(POLISH_CSV, encoding="utf-8")
    features = np.array(build_feature_matrix(
        {e["polish_var"]: df[e["polish_var"]].tolist()
         for e in MODEL_FEATURE_ORDER}, allow_missing=False))
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]])
    records = [{"item_responses": features[i].astype(float),
                "label": int(y[i]),
                "missing_mask": np.zeros(N_ITEMS, dtype=bool)} for i in range(len(y))]
    print(f"   cohort {len(y)} (ASD {int((y == 1).sum())} / "
          f"control {int((y == 0).sum())})  features {features.shape}")

    from src.data.ingest import load_dataset
    from src.data.splits import stratified_split
    saudi = load_dataset("saudi", synthetic=False)
    saudi_train, _, _ = stratified_split(saudi, seed=seed)

    def dqn_for(B: int) -> DQNPolicy:
        import torch
        blob = torch.load(REPO / protocol["rl"]["artifacts"][f"dqn_B{B}"]["path"],
                          map_location="cpu", weights_only=False)
        pol = DQNPolicy(n_items=blob["n_items"], m_list=blob["m_list"],
                        seed=blob["seed"])
        pol.q.load_state_dict(blob["q_state_dict"])
        pol.target.load_state_dict(blob["target_state_dict"])
        pol.q.eval()
        pol.target.eval()
        return pol

    # Selection criteria, taken from the implementations. Nothing is invented.
    CRITERION = {
        "random_questioning": "an item drawn uniformly at random from the legal set",
        "greedy_information_gain":
            "the item with the highest empirical information gain "
            "IG(s,j) = H(Y|s) - sum_v P(v|s,j) H(Y|s,j,v) on the Saudi training support",
        "beta_greedy_evoi":
            "the item with the highest expected value of information under the "
            "Beta(1,1)-smoothed Saudi training posterior",
        "dqn":
            "the item with the highest learned Q-value for the current state",
    }

    print("\n[3/6] evaluate every arm at every budget")
    rows: List[Dict[str, Any]] = []
    for B in budgets:
        arms = {
            "random_questioning": RandomPolicy(seed=seed),
            "greedy_information_gain": GreedyIGPolicy(saudi_train, n_items=N_ITEMS),
            "beta_greedy_evoi": BetaGreedyPolicy(saudi_train, n_items=N_ITEMS,
                                                 lambda_cost=lambda_cost,
                                                 select_by="evoi"),
            "dqn": dqn_for(B),
        }
        for name, pol in arms.items():
            eps = [run_episode(rec, question_budget=B, b_min=0, policy=pol,
                               predictor=predictor.predict_state,
                               lambda_cost=lambda_cost, cost_mode="uniform",
                               tau=tau) for rec in records]
            p = np.array([e["p_hat"] for e in eps])
            asked = [len(e["items_asked"]) for e in eps]
            # item count restricted to the items this arm actually asked
            subset_counts = np.array([
                float(sum(int(features[i][j]) for j in e["items_asked"]))
                / max(len(e["items_asked"]), 1) for i, e in enumerate(eps)])
            m = core_metrics(y, p, tau)
            m2 = core_metrics(y, p, tau_secondary)
            rows.append({
                "arm": name, "B": B,
                "auroc": m["auroc"], "brier": m["brier"],
                "ece_10bin": m["ece_10bin"],
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "ppv": m["ppv"], "npv": m["npv"],
                "questions_used": float(np.mean(asked)),
                "cumulative_reward": float(np.mean([e["R"] for e in eps])),
                "auroc_tau_secondary": m2["auroc"],
                "item_count_asked_subset_auroc": fast_auroc(y, subset_counts),
                "selection_criterion": CRITERION[name],
                "trained_on": "saudi train split",
                "polish_used_in_training": False,
                "predictor_version": PREDICTOR_VERSION,
                "_p": p,
            })
            print(f"   B={B:<3} {name:<26} AUROC {m['auroc']:.4f}  "
                  f"Brier {m['brier']:.4f}  q {np.mean(asked):.2f}  "
                  f"R {np.mean([e['R'] for e in eps]):.4f}")

    print("\n[4/6] full-question item-count reference (B=10, all 10 items)")
    ref_p = np.array([predictor.predict_features(row) for row in features])
    ref_item_count = features.sum(axis=1).astype(float)
    ref_metrics = {
        "predictor_v3_on_all_10": core_metrics(y, ref_p, tau),
        "raw_item_count": {
            "auroc": fast_auroc(y, ref_item_count),
            "brier_raw_count": float(np.mean(
                ((ref_item_count / 10.0) - y) ** 2)),
        },
    }
    print(f"   v3 predictor on all 10 : AUROC "
          f"{ref_metrics['predictor_v3_on_all_10']['auroc']:.4f}  "
          f"Brier {ref_metrics['predictor_v3_on_all_10']['brier']:.4f}")
    print(f"   raw item count         : AUROC "
          f"{ref_metrics['raw_item_count']['auroc']:.4f}")

    # The item-count reference is a mandatory baseline, so it enters the reported
    # table as a first-class arm at the full-question budget. Ranking uses the raw
    # count; the probability scale (count/10) is used for Brier and threshold
    # metrics, which is monotone in the count and so leaves AUROC unchanged.
    count_prob = ref_item_count / 10.0
    cm = core_metrics(y, count_prob, tau)
    rows.append({
        "arm": "item_count", "B": max(budgets),
        "auroc": fast_auroc(y, ref_item_count),
        "brier": cm["brier"], "ece_10bin": cm["ece_10bin"],
        "sensitivity": cm["sensitivity"], "specificity": cm["specificity"],
        "ppv": cm["ppv"], "npv": cm["npv"],
        "questions_used": float(N_ITEMS),
        "cumulative_reward": None,
        "auroc_tau_secondary": core_metrics(y, count_prob, tau_secondary)["auroc"],
        "item_count_asked_subset_auroc": None,
        "selection_criterion": "no selection: every one of the 10 items is asked",
        "trained_on": "none (label-free definition)",
        "polish_used_in_training": False,
        "predictor_version": "n/a (not a learned scorer)",
        "_p": ref_item_count,
    })
    print(f"   item_count arm recorded at B={max(budgets)}: AUROC "
          f"{rows[-1]['auroc']:.4f}")

    print("\n[5/6] mandatory-baseline check and paired comparisons")
    reported = sorted({r["arm"] for r in rows})
    try:
        assert_all_baselines(reported, context="step14 external evaluation")
        print(f"   all mandatory baselines present: {list(REQUIRED_BASELINES)}")
    except Exception as exc:
        print(f"   -> FAIL: {exc}")
        return 2
    missing = missing_baselines(reported)
    assert not missing, missing

    reference = ref_metrics["predictor_v3_on_all_10"]["auroc"]
    comparisons = []
    for r in rows:
        ci = paired_ci(y, r["_p"], ref_p, n_boot, seed)
        ci_raw_count = paired_ci(y, r["_p"], ref_item_count, n_boot, seed)
        comparisons.append({
            "arm": r["arm"], "B": r["B"],
            "auroc": r["auroc"],
            "questions_used": r["questions_used"],
            "vs_full_v3_predictor": ci,
            "vs_full_item_count": ci_raw_count,
            "equivalence_declared": "NOT DECLARED",
            "equivalence_note": (
                "acceptable_AUROC_margin is OPEN; no margin was chosen, so no "
                "non-inferiority or equivalence claim is made in either direction."),
        })

    for c in comparisons:
        v = c["vs_full_item_count"]
        if v.get("status") == "UNDEFINED":
            continue
        print(f"   {c['arm']:<26} B={c['B']:<3} AUROC {c['auroc']:.4f}  "
              f"diff vs item-count {v['point_estimate']:+.4f} "
              f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")

    print("\n[6/6] write artifact")
    for r in rows:
        r.pop("_p", None)
    artifact = {
        "artifact": "polish_rl_external_evaluation",
        "tag": TAG,
        "git_sha": git_sha(REPO),
        "protocol_artifact": "results/polish_rl_external_protocol.json",
        "protocol_sha_verified": sha_ok,
        "predictor_version": PREDICTOR_VERSION,
        "predictor_sha256": actual_sha,
        "protocol_frozen_before_polish": True,
        "research_question": protocol["research_question"],
        "cohort": {
            "n": int(len(y)),
            "asd": int((y == 1).sum()),
            "control": int((y == 0).sum()),
            "label_source": "clinician-established (clinical GROUP)",
            "instrument": "Q-CHAT-25 projected to Q-CHAT-10",
        },
        "budgets": budgets,
        "full_question_reference": ref_metrics,
        "results": rows,
        "comparisons_vs_full_question": comparisons,
        "equivalence": protocol["equivalence"],
        "mandatory_baselines": {
            "required": list(REQUIRED_BASELINES), "reported": reported,
            "missing": missing, "all_present": not missing},
        "threshold": {
            "primary_tau": tau, "secondary_tau_sensitivity_only": tau_secondary,
            "tuned_on_polish": False},
        "no_tuning_statement": (
            "Every hyperparameter, seed, budget, baseline and metric was read from "
            "the frozen protocol. This script fits nothing and selects nothing."),
        "limitations": [
            "One Polish external cohort only; no replication.",
            "Label-definition shift: Saudi questionnaire-derived vs Polish "
            "clinician-established.",
            "Ordinal-to-binary projection discards ordinal information by design.",
            "n = 252 gives finite precision; intervals are wide.",
            "Cross-country and cross-instrument domain shift is unquantified.",
            "Screening prediction is not clinical diagnosis.",
            "Saudi training labels are circular, which flatters policies during "
            "training.",
        ],
        "contains_participant_rows": False,
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_artifact(REPO, artifact, "polish_rl_external_evaluation.json")
    print(f"   -> {path.relative_to(REPO)}")
    print("\nEvaluation complete. Actual outcomes are reported as measured; no arm "
          "was favoured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())