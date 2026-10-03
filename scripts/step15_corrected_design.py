"""Step 15 — corrected experimental design, Saudi-only stage.

Runs entirely on Saudi development data. The Polish cohort is NOT read here, so
everything the corrected external evaluation depends on is fixed before the sealed
cohort is opened.

Three deliverables:
  1. Bayesian prior audit with three prespecified variants (A empirical joint,
     B add-alpha with alpha=0.5 fixed in advance, C factorised = PRIMARY).
  2. DQN trained with 5 independent seeds per budget, each scored on the Saudi
     VALIDATION split, one seed frozen per budget by a prespecified rule.
  3. Exact DP under the environment's own reward, giving J_exact(B).

Selection rule for DQN (prespecified, Saudi-only, Polish never inspected):
    highest mean terminal reward on the Saudi validation split; ties -> lowest seed.

Nothing frozen is overwritten: v2 and v3 predictors, the single-seed
results/policies_v3/dqn_v3_B*.pt files, and the Step 10 primary result are all left
byte-identical. New artefacts go to results/policies_v3_multiseed/.
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
MULTISEED = RESULTS / "policies_v3_multiseed"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.provenance import git_sha, write_artifact

TAG = ("corrected design, Saudi-only stage; research prototype, NOT a diagnostic "
       "device; Polish cohort NOT read in this script")

BUDGETS = (2, 3, 4, 5, 10)
DQN_SEEDS = (0, 1, 2, 3, 4)
DQN_EPISODES = 2000
SEED = 0
PRIMARY_TAU = 0.5
LAMBDA_COST = 0.0
PRIOR_ALPHA = 0.5


def train_dqn(train_recs, predictor, n_items: int, budget: int, episodes: int,
              seed: int):
    import torch

    from scripts.step4_train_policies import collect_episode
    from src.policies.dqn import DQNPolicy
    from src.policies.replay import ReplayBuffer

    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = ReplayBuffer(capacity=100_000, seed=seed)
    pol = DQNPolicy(n_items=n_items, seed=seed)
    losses: List[float] = []
    terminal_rewards: List[float] = []
    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        eps = 1.0 + (0.05 - 1.0) * frac
        rec = train_recs[ep % len(train_recs)]
        r, _n = collect_episode(rec, budget, budget, pol, predictor.predict_state,
                                buf, n_items, LAMBDA_COST, PRIMARY_TAU, epsilon=eps)
        terminal_rewards.append(float(r))
        if len(buf) >= 32:
            for _ in range(4):
                losses.append(pol.train_step(buf.sample(32), gamma=0.99))
        if (ep + 1) % 200 == 0:
            pol.update_target()
    return pol, {
        "training_reward_mean": float(np.mean(terminal_rewards)),
        "training_reward_sd": float(np.std(terminal_rewards)),
        "loss_first_50": float(np.mean(losses[:50])) if losses else None,
        "loss_final_50": float(np.mean(losses[-50:])) if losses else None,
        "n_updates": len(losses),
        "episodes": int(episodes),
        "trained_at_budget": int(budget),
    }


def score_policy_on(records, policy, predictor, budget: int) -> Dict[str, float]:
    """Mean terminal reward and AUROC on a held-out Saudi split."""
    from src.env.environment import run_episode
    from src.eval.fast_stats import auroc

    eps, y = [], []
    for rec in records:
        eps.append(run_episode(rec, question_budget=budget, b_min=budget,
                               policy=policy, predictor=predictor.predict_state,
                               lambda_cost=LAMBDA_COST, cost_mode="uniform",
                               tau=PRIMARY_TAU))
        y.append(int(rec["label"]))
    p = np.array([e["p_hat"] for e in eps])
    return {
        "mean_terminal_reward": float(np.mean([e["R"] for e in eps])),
        "reward_sd": float(np.std([e["R"] for e in eps])),
        "auroc": float(auroc(y, p)),
        "brier": float(np.mean((p - np.array(y, dtype=float)) ** 2)),
        "mean_items_asked": float(np.mean([len(e["items_asked"]) for e in eps])),
        "n": len(eps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=DQN_EPISODES)
    args = ap.parse_args()

    print("=" * 78)
    print("Step 15 - corrected design: Saudi-only stage")
    print(f"  {TAG}")
    print("=" * 78)

    from src.data.ingest import load_dataset
    from src.data.splits import SCHEME, split_fingerprint, stratified_split
    from src.models.logistic_predictor import (N_ITEMS, PREDICTOR_VERSION,
                                               LogisticPredictor, sha256_of)
    from src.models.prior_audit import PRIOR_VARIANTS, audit as prior_audit

    MULTISEED.mkdir(parents=True, exist_ok=True)

    predictor_path = RESULTS / "predictor_logistic_saudi_v3.pkl"
    predictor = LogisticPredictor.load(predictor_path)
    print(f"\n[0/4] predictor {PREDICTOR_VERSION}  sha256 {sha256_of(predictor_path)[:16]}...")

    saudi = load_dataset("saudi", synthetic=False)
    train, val, test = stratified_split(saudi, seed=SEED)
    X = lambda recs: np.array([np.asarray(r["item_responses"], dtype=int)
                               for r in recs])
    print(f"      Saudi splits train {len(train)} / val {len(val)} / test {len(test)}")

    # ---------------------------------------------------------------- 1
    print("\n[1/4] Bayesian prior audit (Saudi only)")
    audit_result = prior_audit(predictor, X(train), {"saudi_val": X(val),
                                                      "saudi_test": X(test)})
    s = audit_result["support"]
    print(f"   possible configurations   : {s['n_possible_configurations']}")
    print(f"   train observed            : {s['train_observed_configurations']}")
    print(f"   train zero-mass           : {s['train_zero_mass_configurations']}")
    print(f"   train cells with count 1  : {s['train_cells_with_count_one']}")
    print(f"   effective support         : {s['effective_support_participation_ratio']:.1f}")
    print(f"   val configs absent in train: {s['saudi_val_configs_absent_from_train']}")
    for v in audit_result["sensitivity"]:
        if "error" in v:
            print(f"   {v['variant']:<22} ERROR {v['error'][:50]}")
            continue
        print(f"   {v['variant']:<22} zero-mass {v['prior_zero_mass_cells']:>4}  "
              f"undefined {v['n_states_with_zero_prior_mass']:>3}  "
              f"meets_definition {v['meets_definition_everywhere']}")

    # ---------------------------------------------------------------- 2
    print(f"\n[2/4] DQN x {len(DQN_SEEDS)} seeds per budget (Saudi train only)")
    import torch
    seed_records: Dict[str, List[Dict[str, Any]]] = {}
    selection: Dict[str, Any] = {}
    for B in BUDGETS:
        rows = []
        for sd in DQN_SEEDS:
            t0 = time.time()
            pol, info = train_dqn(train, predictor, N_ITEMS, B, args.episodes, sd)
            path = MULTISEED / f"dqn_v3_B{B}_seed{sd}.pt"
            torch.save({"q_state_dict": pol.q.state_dict(),
                        "target_state_dict": pol.target.state_dict(),
                        "n_items": N_ITEMS, "m_list": None, "lr": 1e-3,
                        "seed": sd, "trained_at_budget": B,
                        "predictor_version": PREDICTOR_VERSION}, path)
            val_score = score_policy_on(val, pol, predictor, B)
            rows.append({"seed": sd, "path": str(path.relative_to(REPO)),
                         "sha256": sha256_of(path),
                         "training": info,
                         "saudi_validation": val_score})
            print(f"   B={B:<3} seed {sd}  trainR {info['training_reward_mean']:.4f}"
                  f"  valR {val_score['mean_terminal_reward']:.4f}"
                  f"  valAUROC {val_score['auroc']:.4f}  {time.time() - t0:.0f}s")
        rewards = np.array([r["saudi_validation"]["mean_terminal_reward"]
                            for r in rows])
        aurocs = np.array([r["saudi_validation"]["auroc"] for r in rows])
        # prespecified rule: highest mean terminal reward on Saudi validation;
        # ties broken by lowest seed.
        best = max(rows, key=lambda r: (r["saudi_validation"]["mean_terminal_reward"],
                                        -r["seed"]))
        seed_records[f"B{B}"] = rows
        selection[f"B{B}"] = {
            "selected_seed": best["seed"],
            "selected_sha256": best["sha256"],
            "selected_path": best["path"],
            "rule": "highest mean terminal reward on Saudi validation; ties -> lowest seed",
            "validation_reward_mean": float(rewards.mean()),
            "validation_reward_sd": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
            "validation_auroc_mean": float(aurocs.mean()),
            "validation_auroc_sd": float(aurocs.std(ddof=1)) if len(aurocs) > 1 else 0.0,
            "all_seeds_reported": True,
            "polish_inspected_during_selection": False,
        }
        print(f"   B={B:<3} -> selected seed {best['seed']}  "
              f"valR {rewards.mean():.4f} +/- {selection[f'B{B}']['validation_reward_sd']:.4f}"
              f"  valAUROC {aurocs.mean():.4f} +/- "
              f"{selection[f'B{B}']['validation_auroc_sd']:.4f}")

    # ---------------------------------------------------------------- 3
    print("\n[3/4] Exact DP under the environment reward (oracle)")
    from src.solvers.exact_custom import ExactDP
    dp_rows = []
    for B in BUDGETS:
        t0 = time.time()
        try:
            dp = ExactDP(train, n_items=N_ITEMS, budget=B, b_min=B,
                         lambda_cost=LAMBDA_COST, cost_mode="uniform",
                         predictor=predictor.predict_state)
            sol = dp.solve()
            dp_rows.append({"B": B, "J_exact": float(sol["V_star"]),
                            "status": sol["status"], "n_states": sol["n_states"],
                            "seconds": round(time.time() - t0, 2),
                            "objective": "1 - E[(p_hat - y)^2] over the empirical "
                                         "support, the environment's own reward"})
            print(f"   B={B:<3} J_exact {sol['V_star']:.6f}  states {sol['n_states']}  "
                  f"{time.time() - t0:.1f}s")
        except Exception as exc:
            dp_rows.append({"B": B, "J_exact": None,
                            "status": f"FAILED {type(exc).__name__}: {exc}"})
            print(f"   B={B:<3} FAILED {exc}")

    # ---------------------------------------------------------------- 4
    print("\n[4/4] freeze the corrected design protocol")
    artifact = {
        "artifact": "polish_fixed_budget_protocol",
        "tag": TAG,
        "git_sha": git_sha(REPO),
        "stage": "saudi_only; Polish not read",
        "corrects": "results/polish_rl_external_evaluation.json (retained, not deleted)",
        "predictor": {
            "version": PREDICTOR_VERSION,
            "sha256": sha256_of(predictor_path),
            "retrained": False,
            "polish_used": False,
        },
        "fixed_budget_semantics": {
            "b_min": "B (equal to question_budget) for EVERY policy",
            "guarantee": "every episode asks exactly B unique questions",
            "random_arm": "RandomFixedLengthPolicy; STOP never legal; uniform over "
                          "unasked legal items",
            "random_policy_with_stop": "RandomPolicy retained UNTOUCHED for a "
                                       "separate future stopping experiment; not "
                                       "mixed into this comparison",
            "learned_stop_activated": False,
            "rationale": "isolates QUESTION SELECTION from STOPPING",
        },
        "references": {
            "A_v3_full_information": "v3 predictor using all 10 items",
            "B_item_count": "count of atypical answers over all 10 items",
            "note": "Both are reported. Neither is called the sole ceiling.",
            "deltas_reported": ["delta_vs_v3_full", "delta_vs_item_count"],
        },
        "confirmatory_family_F1": {
            "question": ("Do adaptive heuristic question-selection policies "
                         "outperform fixed-length random question selection at "
                         "reduced question budgets?"),
            "comparisons": [f"{a} - {b} at B={B}"
                            for B in (2, 3, 4, 5)
                            for a in ("greedy_information_gain",
                                      "beta_greedy_evoi")
                            for b in ("random_fixed",)],
            "family_size": 8,
            "correction": "holm-bonferroni",
            "alpha": 0.05,
            "exclusive": ("These are the ONLY confirmatory policy comparisons. "
                          "Everything else is exploratory/descriptive with effect "
                          "estimates and 95% CIs."),
        },
        "equivalence": {
            "acceptable_AUROC_margin": "OPEN",
            "state": "OPEN",
            "claims_forbidden": ["equivalence", "non-inferiority",
                                 "statistically indistinguishable"],
            "note": ("No margin is chosen. Formal equivalence is future work "
                     "requiring a justified margin."),
        },
        "dqn_multi_seed": {
            "seeds": list(DQN_SEEDS),
            "n_seeds": len(DQN_SEEDS),
            "episodes_per_seed": int(args.episodes),
            "selection_rule": ("highest mean terminal reward on the SAUDI "
                               "VALIDATION split; ties broken by lowest seed"),
            "polish_inspected_during_selection": False,
            "all_seeds_reported": True,
            "per_budget": selection,
            "per_seed_records": seed_records,
        },
        "exact_dp": {
            "solver": "src/solvers/exact_custom.py ExactDP",
            "predictor_supplied": True,
            "why": ("The solver originally optimised 1-(p_emp-y)^2, a different "
                    "objective from the environment's 1-(p_hat-y)^2. An optional "
                    "predictor argument was added so the oracle optimises the "
                    "environment's own reward; the default is unchanged and "
                    "backward compatible."),
            "interpretation_limit": ("ExactDP is an oracle UNDER THE SIMULATOR. It "
                                     "is NOT clinically optimal and carries no "
                                     "clinical interpretation."),
            "results": dp_rows,
        },
        "bayesian_prior": {
            "primary_variant": "C_factorized_item",
            "prespecified_alpha_variant_B": PRIOR_ALPHA,
            "alpha_tuned": False,
            "selected_using_polish": False,
            "audit": audit_result,
        },
        "reward_audit": {
            "formula": "R = 1 - (p_hat - y)^2, lambda = 0",
            "development_label": "Saudi Q-CHAT-derived questionnaire label",
            "external_target": "Polish clinician-established label",
            "status": "SURROGATE OBJECTIVE",
            "prohibited_description": ("optimising clinical diagnosis, or any "
                                       "clinical outcome"),
            "note": ("The reward is defined against a screening-derived label. "
                     "Whether simulator reward tracks external clinical "
                     "discrimination is an empirical question, audited separately."),
        },
        "bootstrap": {
            "method": "paired percentile bootstrap, vectorised tie-aware AUROC",
            "n_resamples": 5000,
            "identical_resamples_within_comparison": True,
            "holm_applied_to": "F1 only",
        },
        "budgets": list(BUDGETS),
        "seed": SEED,
        "threshold": {"primary_tau": PRIMARY_TAU, "tuned_on_polish": False},
        "training_split": {"scheme": SCHEME,
                           "fingerprint": split_fingerprint(train, val, test),
                           "n_train": len(train), "n_val": len(val)},
        "nz_cross_dataset": {
            "status": "DEFERRED",
            "reason": ("The primary 1054-row NZ cohort is absent. The pooled "
                       "6075-row Autism_Screening_Data_Combined.csv is NOT a "
                       "substitute and will not be used for the primary experiment."),
            "future_work": ("Cross-country development/validation using the "
                            "primary NZ 1054-row cohort requires acquisition or "
                            "verification of the specified source file."),
            "nz_loader_modified": False,
        },
        "ordinal_future_work": {
            "status": "DOCUMENTED_NOT_IMPLEMENTED",
            "reason": ("The binary projection discards ordinal information. An "
                       "ordinal Q-CHAT-10 predictor is a future experiment."),
            "binary_contract_changed": False,
        },
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_artifact(REPO, artifact, "polish_fixed_budget_protocol.json")
    print(f"   -> {path.relative_to(REPO)}")
    print("\nSaudi-only stage complete. Polish may now be opened exactly once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())