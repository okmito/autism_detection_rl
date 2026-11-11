from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import RESULTS_DIR


def _load_json(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_csv(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def research_dashboard() -> dict:
    artifacts = {}
    for fname in [
        "step5_policy_benchmark_saudi.json", "perf_vs_budget_saudi.json", "lambda_sweep_saudi.json",
        "evoi_scale_saudi.json", "predictor_saudi_metrics.json", "predictor_uci_child_metrics.json",
        "benchmark_multiseed_saudi.json", "outcome_explainability_saudi.json", "dp_tractability_sweep.json",
        "action_agreement_val.json", "ablation_af_val.json", "hybrid_stopping_audit.json",
        "uncertainty_support_val.json", "v4_v7_validation_status.json", "polish_external_validation.json",
    ]:
        artifacts[fname] = _load_json(fname)

    perf_csv = _load_csv("perf_vs_budget_saudi.csv")
    step5_csv = _load_csv("step5_policy_benchmark_saudi.csv")
    bench_csv = _load_csv("benchmark_multiseed_saudi.csv")

    limitations = [
        "This dashboard reports available offline research artifacts only; it does not fabricate missing experiments.",
        "Saudi/UCI questionnaire-derived labels are circular in parts of the dataset (see audits/reports); results are research measurements, not clinical accuracy.",
        "External validation is blocked/incompatible in current artifacts where indicated; open verification gates (e.g. V-4/V-6/V-7) remain unless evidence shows otherwise.",
        "Live application inference uses the configured demo/research predictor and is separate from offline benchmark aggregates.",
    ]
    return {
        "artifacts": artifacts,
        "tables": {
            "perf_vs_budget_saudi.csv": perf_csv,
            "step5_policy_benchmark_saudi.csv": step5_csv,
            "benchmark_multiseed_saudi.csv": bench_csv,
        },
        "limitations": limitations,
    }


def available_policies() -> list:
    # Honest availability based on runtime-loadable implementations in src/policies.
    policies = []
    def add(pid, name, kind, status, reason=""):
        policies.append({"id": pid, "name": name, "kind": kind, "status": status, "reason": reason})
    add("greedy", "Greedy Information Gain", "heuristic", "available", "Empirical IG over training support; used by demo/app.")
    add("random", "Random", "baseline", "available", "Uniform random over legal actions (may stop early).")
    add("beta_greedy", "Beta-Greedy / EVOI (smoothed)", "heuristic", "available", "Beta-prior smoothed IG/EVOI variant.")
    add("random_fixed", "Random Fixed (asks budget)", "baseline", "available", "Random item selection without voluntary stop (runs to budget unless forced).")
    add("static_fixed", "Exact Fixed Subset", "baseline", "available_if_data", "Best fixed subset baseline when computable for given budget/data.")
    add("static_rfe", "Static RFE Subset", "baseline", "available_if_data", "RFE-selected fixed subset baseline.")
    add("irt_cat", "IRT-CAT", "model-based", "available_if_fitted", "Requires fitted IRT/selection context; available where implemented.")
    add("exact", "Exact DP Reference", "reference", "available_limited", "Backward-induction reference on tractable synthetic/small cases; not a general runtime policy.")
    add("dqn", "DQN", "learned", "unavailable", "No pre-trained demo checkpoint found in results; training artifacts/scripts exist but model not loaded for live inference.")
    add("ppo", "PPO", "learned", "unavailable", "No pre-trained demo checkpoint found in results; training artifacts/scripts exist but model not loaded for live inference.")
    return policies
