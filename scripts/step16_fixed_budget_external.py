"""Step 16 — corrected fixed-budget external evaluation on the sealed Polish cohort.

Runs ONCE, after scripts/step15_corrected_design.py froze the corrected design.
Nothing here fits, selects or tunes anything; every choice is read from the frozen
protocol artifact.

Design corrections relative to the retained first evaluation:
  * every arm asks EXACTLY B unique questions (b_min = B), including random;
  * random is ``RandomFixedLengthPolicy``, which never stops;
  * TWO references are reported separately (v3 full-information, item count);
  * the prespecified confirmatory family F1 (8 comparisons) is Holm-corrected and
    is the ONLY confirmatory analysis; everything else is exploratory.

The equivalence / non-inferiority margin remains OPEN. No equivalence,
non-inferiority or "statistically indistinguishable" claim is made anywhere.

MANDATORY CAVEAT — this is external validation against clinician-established ASD
labels. It is NOT a diagnosis, NOT clinical validation, and NOT a basis for
deployment. The reward optimised by every policy is a SURROGATE defined against a
screening-derived label.
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
FIGS = REPO / "docs" / "figures"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.provenance import git_sha, write_artifact

TAG = ("corrected fixed-budget external evaluation; Polish clinician-established "
       "labels; research prototype, NOT a diagnostic device; design frozen before "
       "the cohort was opened; nothing tuned here")

ARMS = ("random_fixed", "greedy_information_gain", "beta_greedy_evoi", "dqn")
F1_ARMS = (("greedy_information_gain", "beta_greedy_evoi"),)
F1_BUDGETS = (2, 3, 4, 5)

CRITERION = {
    "random_fixed":
        "an item drawn uniformly at random from the unasked legal items; STOP is "
        "never legal, so exactly B unique questions are asked",
    "greedy_information_gain":
        "the item with the highest empirical information gain "
        "IG(s,j) = H(Y|s) - sum_v P(v|s,j) H(Y|s,j,v) on the Saudi training support",
    "beta_greedy_evoi":
        "the item with the highest expected value of information under the "
        "Beta(1,1)-smoothed Saudi training posterior",
    "dqn":
        "the item with the highest learned Q-value for the current state; STOP is "
        "not legal before B questions have been asked",
}


def evaluate_arm(records, policy, predictor, budget: int, tau: float,
                 lambda_cost: float) -> Dict[str, Any]:
    from src.env.environment import run_episode
    from src.eval.fast_stats import auroc

    eps = [run_episode(rec, question_budget=budget, b_min=budget, policy=policy,
                        predictor=predictor.predict_state,
                        lambda_cost=lambda_cost, cost_mode="uniform", tau=tau)
           for rec in records]
    p = np.array([e["p_hat"] for e in eps])
    y = np.array([int(rec["label"]) for rec in records])
    asked = np.array([len(e["items_asked"]) for e in eps])
    return {
        "p": p, "y": y,
        "auroc": float(auroc(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_terminal_reward": float(np.mean([e["R"] for e in eps])),
        "questions_used": float(asked.mean()),
        "min_questions": int(asked.min()),
        "max_questions": int(asked.max()),
    }


def make_figure(budgets, series: Dict[str, Dict[int, float]],
                ref_v3: float, ref_count: float, out_path: Path) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "random_fixed": ("RandomFixed", "tab:gray", "o", "--"),
        "greedy_information_gain": ("Greedy IG", "tab:blue", "s", "-"),
        "beta_greedy_evoi": ("Beta-Greedy (EVOI)", "tab:green", "^", "-"),
        "dqn": ("DQN (selected seed)", "tab:red", "D", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    for arm, (label, colour, marker, ls) in style.items():
        xs = [b for b in budgets if b in series.get(arm, {})]
        ys = [series[arm][b] for b in xs]
        ax.plot(xs, ys, label=label, color=colour, marker=marker, linestyle=ls,
                linewidth=1.8, markersize=6)
    ax.axhline(ref_v3, color="tab:purple", linestyle="-", linewidth=2,
               label="Reference A: v3 full information (10 items)")
    ax.axhline(ref_count, color="tab:brown", linestyle=":", linewidth=2,
               label="Reference B: item count (10 items)")
    ax.set_xlabel("questions asked (fixed budget B)")
    ax.set_ylabel("AUROC on Polish (clinician-established labels)")
    ax.set_title("Question selection vs budget")
    ax.set_xticks(budgets)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for arm, (label, colour, marker, ls) in style.items():
        xs = [b for b in budgets if b in series.get(arm, {})]
        ys = [ref_v3 - series[arm][b] for b in xs]
        ax.plot(xs, ys, label=label, color=colour, marker=marker, linestyle=ls,
                linewidth=1.8, markersize=6)
    ax.set_xlabel("questions asked (fixed budget B)")
    ax.set_ylabel("AUROC loss vs v3 full information")
    ax.set_title("Performance loss from full information (lower is better)")
    ax.set_xticks(budgets)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Polish external validation — corrected fixed-budget design "
                 "(research prototype, not a diagnostic device)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"path": str(out_path.relative_to(REPO))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("Step 16 - corrected fixed-budget evaluation on sealed Polish")
    print(f"  {TAG}")
    print("=" * 78)

    from src.data.ingest import POLISH_CSV, load_dataset
    from src.data.qchat10_contract import MODEL_FEATURE_ORDER, build_feature_matrix
    from src.data.splits import stratified_split
    from src.eval.baselines import REQUIRED_BASELINES, assert_all_baselines
    from src.eval.external_metrics import core_metrics
    from src.eval.fast_stats import holm_adjust, paired_auroc_comparison
    from src.models.logistic_predictor import (N_ITEMS, PREDICTOR_VERSION,
                                               LogisticPredictor, sha256_of)
    from src.policies.beta_greedy import BetaGreedyPolicy
    from src.policies.dqn import DQNPolicy
    from src.policies.greedy import GreedyIGPolicy
    from src.policies.random_fixed import RandomFixedLengthPolicy

    protocol = json.loads(
        (RESULTS / "polish_fixed_budget_protocol.json").read_text(encoding="utf-8"))
    budgets = list(protocol["budgets"])
    tau = float(protocol["threshold"]["primary_tau"])
    lambda_cost = 0.0
    n_boot = args.bootstrap or int(protocol["bootstrap"]["n_resamples"])
    seed = int(protocol["seed"])

    print("\n[1/6] verify the frozen corrected design")
    predictor_path = RESULTS / "predictor_logistic_saudi_v3.pkl"
    sha_ok = sha256_of(predictor_path) == protocol["predictor"]["sha256"]
    print(f"   design frozen before Polish : {protocol['stage']}")
    print(f"   predictor sha matches       : {sha_ok}")
    print(f"   budgets                     : {budgets}")
    print(f"   b_min                       : B for every arm")
    print(f"   equivalence margin          : "
          f"{protocol['equivalence']['acceptable_AUROC_margin']}")
    print(f"   confirmatory family F1 size : "
          f"{protocol['confirmatory_family_F1']['family_size']} (Holm, alpha 0.05)")
    if not sha_ok:
        print("   -> BLOCKED: predictor does not match the frozen design")
        return 2

    predictor = LogisticPredictor.load(predictor_path)

    print("\n[2/6] sealed cohort + Saudi reference split")
    import pandas as pd
    df = pd.read_csv(POLISH_CSV, encoding="utf-8")
    features = np.array(build_feature_matrix(
        {e["polish_var"]: df[e["polish_var"]].tolist()
         for e in MODEL_FEATURE_ORDER}, allow_missing=False))
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]])
    polish = [{"item_responses": features[i].astype(float), "label": int(y[i]),
               "missing_mask": np.zeros(N_ITEMS, dtype=bool)}
              for i in range(len(y))]
    saudi = load_dataset("saudi", synthetic=False)
    saudi_train, saudi_val, _ = stratified_split(saudi, seed=seed)
    print(f"   Polish {len(y)} (ASD {int((y == 1).sum())} / "
          f"control {int((y == 0).sum())}); Saudi val {len(saudi_val)}")

    def dqn_for(B: int, sd: int) -> DQNPolicy:
        import torch
        blob = torch.load(MULTI / f"dqn_v3_B{B}_seed{sd}.pt",
                          map_location="cpu", weights_only=False)
        pol = DQNPolicy(n_items=blob["n_items"], m_list=blob["m_list"],
                        seed=blob["seed"])
        pol.q.load_state_dict(blob["q_state_dict"])
        pol.target.load_state_dict(blob["target_state_dict"])
        pol.q.eval()
        pol.target.eval()
        return pol

    print("\n[3/6] evaluate every arm at every budget (b_min = B)")
    results: List[Dict[str, Any]] = []
    store: Dict[tuple, np.ndarray] = {}
    for B in budgets:
        sel = protocol["dqn_multi_seed"]["per_budget"][f"B{B}"]["selected_seed"]
        arms = {
            "random_fixed": RandomFixedLengthPolicy(seed=seed),
            "greedy_information_gain": GreedyIGPolicy(saudi_train,
                                                      n_items=N_ITEMS),
            "beta_greedy_evoi": BetaGreedyPolicy(saudi_train, n_items=N_ITEMS,
                                                 lambda_cost=lambda_cost,
                                                 select_by="evoi"),
            "dqn": dqn_for(B, sel),
        }
        for name, pol in arms.items():
            pol_res = evaluate_arm(polish, pol, predictor, B, tau, lambda_cost)
            saudi_res = evaluate_arm(saudi_val, pol, predictor, B, tau, lambda_cost)
            m = core_metrics(pol_res["y"], pol_res["p"], tau)
            results.append({
                "arm": name, "B": B,
                "auroc": pol_res["auroc"], "brier": pol_res["brier"],
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "ppv": m["ppv"], "npv": m["npv"],
                "questions_used": pol_res["questions_used"],
                "min_questions": pol_res["min_questions"],
                "max_questions": pol_res["max_questions"],
                "cumulative_reward": pol_res["mean_terminal_reward"],
                "saudi_validation_reward": saudi_res["mean_terminal_reward"],
                "saudi_validation_auroc": saudi_res["auroc"],
                "selection_criterion": CRITERION[name],
                "polish_used_in_training": False,
                "predictor_version": PREDICTOR_VERSION,
            })
            store[(name, B)] = pol_res["p"]
            flag = "" if pol_res["min_questions"] == B == pol_res["max_questions"] \
                else "  <-- NOT EXACTLY B"
            print(f"   B={B:<3} {name:<26} AUROC {pol_res['auroc']:.4f}  "
                  f"q {pol_res['questions_used']:.2f} "
                  f"[{pol_res['min_questions']}-{pol_res['max_questions']}]  "
                  f"R {pol_res['mean_terminal_reward']:.4f}{flag}")

    bad = [r for r in results
           if r["min_questions"] != r["B"] or r["max_questions"] != r["B"]]
    assert not bad, f"arms not budget-matched: {[(r['arm'], r['B']) for r in bad]}"

    print("\n[4/6] references A and B, reported separately")
    ref_v3_p = np.array([predictor.predict_features(row) for row in features])
    ref_count = features.sum(axis=1).astype(float)
    from src.eval.fast_stats import auroc
    refA = {"name": "A_v3_full_information",
            "auroc": float(auroc(y, ref_v3_p)),
            "brier": float(np.mean((ref_v3_p - y) ** 2))}
    refB = {"name": "B_item_count",
            "auroc": float(auroc(y, ref_count))}
    store[("reference_A_v3_full", 10)] = ref_v3_p
    store[("reference_B_item_count", 10)] = ref_count

    # Reference B is the mandatory item-count baseline, so it enters the reported
    # set as a first-class entry at the full-question budget. Ranking uses the raw
    # count; count/10 is used for probability-scale metrics (monotone, so AUROC is
    # unchanged).
    count_prob = ref_count / 10.0
    m_count = core_metrics(y, count_prob, tau)
    results.append({
        "arm": "item_count", "B": max(budgets),
        "auroc": refB["auroc"], "brier": m_count["brier"],
        "sensitivity": m_count["sensitivity"],
        "specificity": m_count["specificity"],
        "ppv": m_count["ppv"], "npv": m_count["npv"],
        "questions_used": float(N_ITEMS),
        "min_questions": N_ITEMS, "max_questions": N_ITEMS,
        "cumulative_reward": None,
        "saudi_validation_reward": None,
        "saudi_validation_auroc": None,
        "selection_criterion": "no selection: all 10 items contribute to the count",
        "polish_used_in_training": False,
        "predictor_version": "n/a (not a learned scorer; Reference B)",
        "delta_vs_v3_full": refB["auroc"] - refA["auroc"],
        "delta_vs_item_count": 0.0,
    })

    for r in results:
        if r["arm"] == "item_count":
            continue
        r["delta_vs_v3_full"] = r["auroc"] - refA["auroc"]
        r["delta_vs_item_count"] = r["auroc"] - refB["auroc"]
    print(f"   A v3 full information (10 items) : AUROC {refA['auroc']:.4f}  "
          f"Brier {refA['brier']:.4f}")
    print(f"   B item count (10 items)          : AUROC {refB['auroc']:.4f}")

    print("\n[5/6] confirmatory family F1 (Holm, alpha=0.05) — the ONLY "
          "confirmatory analysis")
    f1_rows, f1_p = [], {}
    for B in F1_BUDGETS:
        for arm in ("greedy_information_gain", "beta_greedy_evoi"):
            key = f"{arm} - random_fixed @ B={B}"
            cmp_ = paired_auroc_comparison(y, store[(arm, B)],
                                           store[("random_fixed", B)],
                                           n_resamples=n_boot, seed=seed)
            cmp_["comparison"] = key
            cmp_["family"] = "F1_confirmatory"
            f1_rows.append(cmp_)
            f1_p[key] = 0.5 * (1.0 + cmp_["bootstrap_p_a_greater"]) \
                if cmp_.get("status") == "OK" else 1.0
    holm = holm_adjust(f1_p, alpha=0.05)
    for row in f1_rows:
        row["holm_adjusted_p"] = holm["adjusted_p"].get(row["comparison"])
        row["holm_reject_at_0.05"] = holm["reject"].get(row["comparison"])
        row["correction"] = "holm-bonferroni (family F1, 8 comparisons)"
    for row in f1_rows:
        print(f"   {row['comparison']:<44} diff {row['difference']:+.4f} "
              f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
              f"P(A>B) {row['bootstrap_p_a_greater']:.3f}  "
              f"holm p {row['holm_adjusted_p']:.4f}  "
              f"reject {row['holm_reject_at_0.05']}")

    print("\n      exploratory (NOT confirmatory, uncorrected, CIs only)")
    exploratory = []
    pairs = [("greedy_information_gain", "beta_greedy_evoi"),
             ("greedy_information_gain", "dqn"),
             ("beta_greedy_evoi", "dqn"),
             ("random_fixed", "dqn")]
    for B in budgets:
        for a, b in pairs:
            cmp_ = paired_auroc_comparison(y, store[(a, B)], store[(b, B)],
                                           n_resamples=n_boot, seed=seed)
            cmp_["comparison"] = f"{a} - {b} @ B={B}"
            cmp_["family"] = "exploratory"
            cmp_["correction"] = "none; exploratory, not part of F1"
            exploratory.append(cmp_)
        for arm in ARMS:
            cmp_ = paired_auroc_comparison(y, store[(arm, B)], ref_v3_p,
                                           n_resamples=n_boot, seed=seed)
            cmp_["comparison"] = f"{arm} - reference_A_v3_full @ B={B}"
            cmp_["family"] = "exploratory_vs_reference"
            cmp_["correction"] = "none; exploratory"
            exploratory.append(cmp_)

    print("\n      Exact DP optimality gap under the simulator")
    dp = {r["B"]: r for r in protocol["exact_dp"]["results"]}
    gaps = []
    for B in budgets:
        j_exact = dp.get(B, {}).get("J_exact")
        for r in [x for x in results if x["B"] == B]:
            if j_exact is None or r["cumulative_reward"] is None:
                continue
            gaps.append({"arm": r["arm"], "B": B, "J_exact": j_exact,
                         "J_policy": r["cumulative_reward"],
                         "optimality_gap": float(j_exact - r["cumulative_reward"])})
    for arm in ARMS:
        g5 = next((g for g in gaps if g["arm"] == arm and g["B"] == 5), None)
        ge = next((g for g in gaps if g["arm"] == arm and g["B"] == 10), None)
        if g5:
            print(f"   {arm:<26} gap@B=5 {g5['optimality_gap']:+.4f}"
                  + (f"   gap@B=10 {ge['optimality_gap']:+.4f}" if ge else ""))

    print("\n      reward/surrogate diagnostic (Saudi simulator vs external AUROC)")
    from scipy.stats import spearmanr
    # item_count is a reference, not a simulator-based arm, so it has no simulator
    # reward and is excluded from this diagnostic rather than imputed.
    cells = [(r["saudi_validation_reward"], r["auroc"]) for r in results
             if r["saudi_validation_reward"] is not None]
    rr = [c[0] for c in cells]
    aa = [c[1] for c in cells]
    reward_audit = {
        "status": "SURROGATE OBJECTIVE",
        "formula": "R = 1 - (p_hat - y)^2, lambda = 0",
        "development_label": "Saudi Q-CHAT-derived questionnaire label",
        "external_target": "Polish clinician-established label",
        "prohibited": ("describing this reward as optimising clinical diagnosis "
                       "or any clinical outcome"),
        "pearson": float(np.corrcoef(rr, aa)[0, 1]),
        "spearman": float(__import__("scipy.stats", fromlist=["x"])
                          .spearmanr(rr, aa).statistic),
        "n_points": len(rr),
        "points": "one per (arm, budget) cell",
        "interpretation_limit": ("A correlation across arm/budget cells is a "
                                 "diagnostic, not a causal claim, and is not used "
                                 "to tune the reward."),
    }
    print(f"   Saudi simulator reward vs Polish AUROC: pearson "
          f"{reward_audit['pearson']:.3f}  spearman "
          f"{reward_audit['spearman']:.3f}  (n={reward_audit['n_points']} cells)")

    reported = sorted({r["arm"] for r in results})
    assert_all_baselines(reported, context="step16")
    print(f"\n      mandatory baselines present: {sorted(REQUIRED_BASELINES)}")

    print("\n[6/6] figure + artifact")
    series = {arm: {r["B"]: r["auroc"] for r in results if r["arm"] == arm}
              for arm in ARMS}
    figure = make_figure(budgets, series, refA["auroc"], refB["auroc"],
                         FIGS / "polish_fixed_budget_auroc_vs_questions.png")
    print(f"   -> {figure['path']}")

    spread = {}
    for B in budgets:
        rec = protocol["dqn_multi_seed"]["per_budget"][f"B{B}"]
        seeds = protocol["dqn_multi_seed"]["per_seed_records"][f"B{B}"]
        spread[f"B{B}"] = {
            "selected_seed": rec["selected_seed"],
            "saudi_val_reward_mean": rec["validation_reward_mean"],
            "saudi_val_reward_sd": rec["validation_reward_sd"],
            "saudi_val_auroc_mean": rec["validation_auroc_mean"],
            "saudi_val_auroc_sd": rec["validation_auroc_sd"],
            "per_seed": [{"seed": s["seed"],
                          "training_reward": s["training"]["training_reward_mean"],
                          "saudi_validation_reward":
                              s["saudi_validation"]["mean_terminal_reward"],
                          "saudi_validation_auroc": s["sauri_validation"]["auroc"]
                          if "sauri_validation" in s else
                          s["saudi_validation"]["auroc"],
                          "sha256": s["sha256"]} for s in seeds],
        }

    artifact = {
        "artifact": "polish_fixed_budget_external_evaluation",
        "tag": TAG,
        "git_sha": git_sha(REPO),
        "design_artifact": "results/polish_fixed_budget_protocol.json",
        "design_sha_verified": sha_ok,
        "retains_first_evaluation": "results/polish_rl_external_evaluation.json",
        "predictor_version": PREDICTOR_VERSION,
        "predictor_sha256": sha256_of(predictor_path),
        "cohort": {"n": int(len(y)), "asd": int((y == 1).sum()),
                   "control": int((y == 0).sum()),
                   "label_source": "clinician-established (clinical GROUP)"},
        "budgets": budgets,
        "fixed_budget_semantics": protocol["fixed_budget_semantics"],
        "references": {"A_v3_full_information": refA, "B_item_count": refB,
                       "note": protocol["references"]["note"]},
        "b10_interpretation": {
            "statement": (
                "Under the current frozen v3 scorer, asking all 10 questions "
                "reaches the scorer's full-information performance. The separate "
                "item-count reference is slightly stronger externally."),
            "why_all_arms_coincide": (
                "At B=10 the fixed-budget semantics require all ten items to be "
                "asked, so no selection freedom remains and every arm returns an "
                "identical result. B=10 is therefore a reference point, not a "
                "comparison between policies."),
        },
        "results": results,
        "confirmatory_family_F1": {
            "question": protocol["confirmatory_family_F1"]["question"],
            "correction": "holm-bonferroni", "alpha": 0.05,
            "family_size": protocol["confirmatory_family_F1"]["family_size"],
            "exclusive": True, "comparisons": f1_rows,
            "multiple_comparison_note": (
                "F1 is the only confirmatory family. The exploratory set is "
                "uncorrected and must not be read as independent evidence."),
        },
        "exploratory_comparisons": exploratory,
        "exact_dp": {"results": protocol["exact_dp"]["results"],
                     "optimality_gaps": gaps,
                     "interpretation_limit":
                         protocol["exact_dp"]["interpretation_limit"]},
        "reward_surrogate_audit": reward_audit,
        "dqn_seed_spread": spread,
        "equivalence": protocol["equivalence"],
        "bayesian_prior_summary": {
            "primary_variant": "C_factorized_item",
            "support": protocol["bayesian_prior"]["audit"]["support"],
            "variant_A_failed": True,
            "alpha_prespecified": protocol["bayesian_prior"][
                "prespecified_alpha_variant_B"],
            "selected_using_polish": False,
        },
        "mandatory_baselines": {"required": list(REQUIRED_BASELINES),
                                "reported": reported, "all_present": True},
        "threshold": {"primary_tau": tau, "tuned_on_polish": False},
        "nz_cross_dataset": protocol["nz_cross_dataset"],
        "ordinal_future_work": protocol["ordinal_future_work"],
        "no_tuning_statement": (
            "Every hyperparameter, seed, budget, baseline, prior and metric was "
            "read from the frozen design artifact. This script fits and selects "
            "nothing, and inspects no Polish result during selection."),
        "contains_participant_rows": False,
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_artifact(REPO, artifact,
                          "polish_fixed_budget_external_evaluation.json")
    print(f"   -> {path.relative_to(REPO)}")
    return 0


MULTI = RESULTS / "policies_v3_multiseed"

if __name__ == "__main__":
    raise SystemExit(main())