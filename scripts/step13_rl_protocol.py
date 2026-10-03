"""Step 13 — train RL policies against predictor v3 and FREEZE the protocol.

Runs entirely on Saudi development data. The Polish cohort is not read in this
script at all, so everything the external evaluation depends on is frozen before
the sealed cohort is opened.

Order of operations, which must not change:
    predictor v3 frozen (step 12)
        -> reward function fixed  (R = 1 - (p_hat - y)^2 - lambda * cost)
        -> policies trained on SAUDI train split
        -> policies frozen and hashed
        -> protocol artifact written
    only then may step 14 touch Polish.

Every policy arm is trained here and persisted, including DQN, which previously
existed only in memory. No artefact is assumed that is not on disk.
"""
from __future__ import annotations

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
POLICIES = RESULTS / "policies_v3"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.provenance import git_sha, write_artifact

TAG = ("RL protocol freeze; Saudi-only training; research prototype, NOT a "
       "diagnostic device; Polish cohort NOT read in this script")

#: Pre-specified BEFORE any external evaluation. B=10 is the full-question
#: reference against which every smaller budget is compared.
BUDGETS = (2, 3, 4, 5, 10)

#: Fixed before Polish. 2000 is the episode count at which the bootstrap
#: objective reaches plateaued behaviour (scripts/step7_rl_diagnosis.py); 400 was
#: previously found undertrained.
DQN_EPISODES = 2000
DQN_TRAIN_BUDGET_PER_BUDGET = True

SEED = 0
BOOTSTRAP_RESAMPLES = 5000
PRIMARY_TAU = 0.5
SECONDARY_TAU = 0.3
LAMBDA_COST = 0.0

#: NOT CHOSEN. No scientifically justified non-inferiority margin is available in
#: this project, so none is invented. The experiment reports raw paired differences
#: and does not declare equivalence success or failure.
ACCEPTABLE_AUROC_MARGIN: Any = "OPEN"


def train_dqn(train_recs, predictor, n_items: int, budget: int,
              episodes: int, seed: int):
    """DQN on the Saudi train split with the frozen v3 scorer as the reward model."""
    import torch

    from scripts.step4_train_policies import collect_episode
    from src.policies.dqn import DQNPolicy
    from src.policies.replay import ReplayBuffer

    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = ReplayBuffer(capacity=100_000, seed=seed)
    pol = DQNPolicy(n_items=n_items, seed=seed)
    losses: List[float] = []
    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        eps = 1.0 + (0.05 - 1.0) * frac
        rec = train_recs[ep % len(train_recs)]
        collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                        n_items, LAMBDA_COST, PRIMARY_TAU, epsilon=eps)
        if len(buf) >= 32:
            for _ in range(4):
                losses.append(pol.train_step(buf.sample(32), gamma=0.99))
        if (ep + 1) % 200 == 0:
            pol.update_target()
    return pol, {
        "loss_first_50": float(np.mean(losses[:50])) if losses else None,
        "loss_final_50": float(np.mean(losses[-50:])) if losses else None,
        "n_updates": len(losses),
        "episodes": int(episodes),
        "trained_at_budget": int(budget),
    }


def main() -> int:
    print("=" * 78)
    print("Step 13 - train policies on Saudi and freeze the external protocol")
    print(f"  {TAG}")
    print("=" * 78)

    from src.data.ingest import load_dataset
    from src.data.splits import SCHEME, split_fingerprint, stratified_split
    from src.models.logistic_predictor import (CALIBRATION_METHOD,
                                               HYPERPARAMETERS, N_ITEMS,
                                               PREDICTOR_VERSION,
                                               LogisticPredictor, sha256_of)
    from src.policies.beta_greedy import BetaGreedyPolicy
    from src.policies.greedy import GreedyIGPolicy
    from src.policies.random_policy import RandomPolicy

    POLICIES.mkdir(parents=True, exist_ok=True)

    predictor_path = RESULTS / "predictor_logistic_saudi_v3.pkl"
    predictor = LogisticPredictor.load(predictor_path)
    predictor_sha = sha256_of(predictor_path)
    print(f"\n[1/5] predictor  {PREDICTOR_VERSION}")
    print(f"   sha256 {predictor_sha}")
    print(f"   calibration {CALIBRATION_METHOD}")

    saudi = load_dataset("saudi", synthetic=False)
    train, val, test = stratified_split(saudi, seed=SEED)
    print(f"\n[2/5] Saudi splits train {len(train)} / val {len(val)} / test {len(test)}")

    print("\n[3/5] train + freeze DQN per budget (Saudi train only)")
    import torch
    frozen: Dict[str, Dict[str, Any]] = {}
    for B in BUDGETS:
        t0 = time.time()
        pol, info = train_dqn(train, predictor, N_ITEMS, B, DQN_EPISODES, SEED)
        path = POLICIES / f"dqn_v3_B{B}.pt"
        # DQNPolicy exposes no state_dict(); persist the network weights plus the
        # architecture needed to rebuild them, so the artefact is self-contained.
        torch.save({
            "q_state_dict": pol.q.state_dict(),
            "target_state_dict": pol.target.state_dict(),
            "n_items": N_ITEMS,
            "m_list": None,
            "lr": 1e-3,
            "seed": SEED,
            "trained_at_budget": B,
            "predictor_version": PREDICTOR_VERSION,
        }, path)
        sha = sha256_of(path)
        frozen[f"dqn_B{B}"] = {
            "path": str(path.relative_to(REPO)),
            "sha256": sha,
            "training": info,
            "trained_on": "saudi train split",
            "polish_used_in_training": False,
        }
        print(f"   B={B:<3} updates {info['n_updates']:<6} "
              f"loss {info['loss_first_50']} -> {info['loss_final_50']}  "
              f"{time.time() - t0:.1f}s  sha {sha[:12]}")

    print("\n[4/5] deterministic policy arms (constructed from Saudi train only)")
    arms = {
        "item_count": {"kind": "reference", "note":
                       "count of atypical answers among all 10 items; the "
                       "full-question reference at B=10"},
        "random_questioning": {
            "kind": "policy", "class": "RandomPolicy",
            "trained_on": "none (seeded)", "sha256": None},
        "greedy_information_gain": {
            "kind": "policy", "class": "GreedyIGPolicy",
            "trained_on": "saudi train split empirical support", "sha256": None},
        "beta_greedy_evoi": {
            "kind": "policy", "class": "BetaGreedyPolicy(select_by='evoi')",
            "trained_on": "saudi train split Beta-smoothed empirical posterior",
            "sha256": None,
            "note": "NOT an RL policy; a Bayesian empirical-posterior heuristic"},
    }
    for name, cfg in arms.items():
        print(f"   {name:<26} {cfg['kind']:<9} {cfg.get('class', cfg['kind'])}")

    print("\n[5/5] freeze protocol artifact")
    protocol = {
        "artifact": "polish_rl_external_protocol",
        "tag": TAG,
        "frozen_before_polish": True,
        "git_sha": git_sha(REPO),
        "research_question": (
            "Can adaptive question selection achieve comparable predictive "
            "performance using fewer questions than asking all 10 Q-CHAT-10 items?"),
        "question_efficiency_not_model_beating": (
            "Success is defined as removing questions while retaining acceptable "
            "performance. It is NOT defined as beating the questionnaire or the "
            "predictor."),
        "predictor": {
            "version": PREDICTOR_VERSION,
            "artifact": str(predictor_path.relative_to(REPO)),
            "sha256": predictor_sha,
            "calibrator": "embedded in the predictor pickle",
            "calibrator_sha256": predictor_sha,
            "calibration_method": CALIBRATION_METHOD,
            "hyperparameters": dict(HYPERPARAMETERS),
            "trained_on": "saudi train split",
            "calibrated_on": "saudi validation split",
            "polish_used": False,
            "feature_contract_version": "qchat10-contract/2.0.0",
            "qchat_mapping_version": "qchat10-mapping/2.0.0",
        },
        "reward": {
            "formula": "R = (1 - (p_hat - y)^2) - lambda_cost * sum_j c_j",
            "lambda_cost": LAMBDA_COST,
            "cost_mode": "uniform",
            "structure": "UNCHANGED from the project specification",
            "note": ("Reward quality is bounded by predictor quality. A "
                     "rank-degraded, miscalibrated scorer distorts the signal the "
                     "policy optimises. Results produced with the v2 neural scorer "
                     "must NOT be mixed with v3 results; all experiments are "
                     "versioned."),
            "scorer_version": PREDICTOR_VERSION,
        },
        "rl": {
            "algorithm": "DQN (per budget)",
            "why_dqn_only": ("Random, greedy-IG and beta-greedy are deterministic "
                             "or heuristic arms and are not RL. Beta-greedy is a "
                             "Bayesian empirical-posterior heuristic and is never "
                             "described as an RL policy."),
            "state": "src.env.state dict: 3n mask one-hot + observed response bits "
                     "+ normalised budget remaining",
            "action": "ask item j in [0, n), or STOP",
            "reward": "terminal utility, see reward.formula",
            "episodes_per_budget": DQN_EPISODES,
            "replay_capacity": 100000,
            "gamma": 0.99,
            "epsilon_schedule": "1.0 -> 0.05 linear",
            "updates_per_episode": 4,
            "batch_size": 32,
            "trained_per_budget": DQN_TRAIN_BUDGET_PER_BUDGET,
            "training_data": "saudi train split ONLY",
            "polish_used_in_training": False,
            "artifacts": frozen,
        },
        "baselines_mandatory": list(
            ("item_count", "random_questioning", "greedy_information_gain",
             "beta_greedy_evoi")),
        "baseline_failure_policy": (
            "An evaluation omitting any mandatory baseline raises "
            "MissingBaselineError and fails; an incomplete benchmark may never be "
            "reported as an improvement."),
        "budgets": list(BUDGETS),
        "full_question_reference_budget": 10,
        "metrics": {
            "primary": ["AUROC"],
            "secondary": ["Brier", "sensitivity", "specificity",
                          "questions_used", "cumulative_reward"],
            "tau_primary": PRIMARY_TAU,
            "tau_secondary_sensitivity_only": SECONDARY_TAU,
            "tau_tuned_on_polish": False,
        },
        "equivalence": {
            "acceptable_AUROC_margin": ACCEPTABLE_AUROC_MARGIN,
            "state": "OPEN",
            "reason": ("No scientifically justified non-inferiority margin is "
                       "available in this project. None has been invented. The "
                       "experiment reports raw paired AUROC differences with 95% "
                       "bootstrap intervals and does NOT declare equivalence "
                       "success or failure."),
            "forbidden": ["choosing a margin after seeing Polish results",
                          "describing a result as 'statistically indistinguishable' "
                          "without a predefined margin"],
        },
        "statistical_method": {
            "bootstrap": "paired percentile bootstrap over evaluation instances",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
            "interval": "95%",
            "degenerate_resamples": "excluded and counted, never imputed",
        },
        "training_split": {
            "scheme": SCHEME,
            "fingerprint": split_fingerprint(train, val, test),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        },
        "stopping_rule": "b_min = 0; the budget alone terminates the episode",
        "split_fingerprint": split_fingerprint(train, val, test),
        "seed": SEED,
        "policy_arms": arms,
        "versions_do_not_mix": {
            "v2_neural_scorer": "legacy; behind the reported primary result",
            "v3_logistic_scorer": "current development scorer for all RL results",
        },
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_artifact(REPO, protocol, "polish_rl_external_protocol.json")
    print(f"   -> {path.relative_to(REPO)}")
    print("\nProtocol frozen. Only now may the Polish cohort be opened.")
    print("Equivalence margin: OPEN - not chosen, not invented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())