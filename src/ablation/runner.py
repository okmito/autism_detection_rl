"""Ablation suite — §20 (7 ablations)"""
from __future__ import annotations
from typing import Dict, Any, List

ABLATIONS = {
    "AB-1": "Reward scoring rule: Brier vs 0/1 correctness",
    "AB-2": "State encoding: three-state vs two-state missing-collapsed",
    "AB-3": "lambda grid granularity: coarse vs fine",
    "AB-4": "Policy algorithm: DQN vs PPO",
    "AB-5": "Predictor training: frozen vs jointly trained",
    "AB-6": "Cost model: uniform vs non-uniform",
    "AB-7": "Stopping: fixed-length vs adaptive stopping under same max B",
}

def ablation_report(results: Dict[str, Any]) -> str:
    lines=["Ablation results:"]
    for k in sorted(ABLATIONS):
        v = results.get(k, "NOT RUN")
        lines.append(f"  {k} ({ABLATIONS[k]}): {v}")
    return "\n".join(lines)
