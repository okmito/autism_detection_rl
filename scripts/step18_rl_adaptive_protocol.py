"""Step 18 — freeze the adaptive-RL questionnaire protocol BEFORE the Polish cohort is analysed with it.

Writes ``results/rl_adaptive/protocol.json``: every hyper-parameter grid, the selection rule, the non-inferiority
margin, the integration gate, and a SHA-256 fingerprint of the code that will run. ``step19`` refuses to run if the
code no longer matches the fingerprint.

Development and debugging of ``src/rl/`` used the Saudi data only (circular labels; see the report). The Polish
cohort had been opened by earlier steps (14, 16) for fixed-budget evaluations with a Saudi-trained scorer; this
protocol re-uses it as a *development* cohort under nested cross-validation, which is a change of role recorded
here and in RL_ADAPTIVE_REPORT.md as a decision requiring supervisor ratification.

Research prototype. Not a diagnostic device.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.provenance import git_sha  # noqa: E402
from src.rl.experiment import DEFAULT_PROTOCOL, code_fingerprint  # noqa: E402

OUT = REPO / "results" / "rl_adaptive"
CODE_FILES = ["belief.py", "mdp.py", "planner.py", "dqn.py", "baselines.py", "evaluate.py", "experiment.py", "data.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing protocol (records the overwrite)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists() and not args.force:
        print(f"protocol already frozen at {path}; refusing to overwrite (use --force and document why)")
        return 1
    protocol = dict(DEFAULT_PROTOCOL)
    protocol.update({
        "artifact": "rl_adaptive_protocol",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
        "code_files": [f"src/rl/{f}" for f in CODE_FILES],
        "code_sha256": code_fingerprint([REPO / "src" / "rl" / f for f in CODE_FILES]),
        "cohort": {"name": "Polish Q-CHAT (projected to Q-CHAT-10 binary, contract qchat10-contract/2.0.0)",
                   "label": "clinician-established ASD vs control (case-control sampling)",
                   "role": "development cohort under nested cross-validation (changed from sealed external test)"},
        "primary_question": ("Can an RL agent that chooses ASK(item)/STOP from the answers so far use fewer "
                             "questions than the full 10-item Q-CHAT-10 while keeping balanced accuracy within "
                             "delta of it?"),
        "agents": {
            "rl_value_iteration": "model-based RL: exact value iteration on the learned naive-Bayes belief MDP",
            "rl_dqn": "model-free RL: Double DQN on replayed real histories, cross-fitted terminal rewards, "
                      "3-seed mean-Q ensemble",
        },
        "comparators": ["full_questionnaire (same scorer, all 10 items)", "official Q-CHAT-10 cutoff (sum >= 4)",
                        "fixed_official_order_k", "fixed_best_subset_k (forward selection on training fold)",
                        "random_fixed_k", "belief_greedy_ig_k", "greedy_ig_k (existing GreedyIGPolicy)",
                        "beta_greedy_k (existing BetaGreedyPolicy)", "legacy_dqn_v3_k (Saudi-trained artifacts)",
                        "threshold_stopping (non-RL adaptive stopping, same selection rule)"],
        "primary_test": ("For each RL agent: paired bootstrap 95% CI of BA(agent) - BA(full_questionnaire); "
                         "non-inferior iff the lower bound > -delta. One-sided bootstrap p-values, Holm across the "
                         "two agents."),
        "integration_gate": {
            "G1": "BA non-inferiority vs full questionnaire holds after Holm (delta = 0.05)",
            "G2": "mean questions < 10 with the bootstrap 95% CI upper bound below 10",
            "G3": "sensitivity difference vs full questionnaire: 95% CI lower bound > -0.10",
            "G4": "engineering: no illegal action, deterministic inference, cap respected (tests)",
            "preference": "if both agents pass, serve rl_value_iteration (exact decomposable explanations, no "
                          "seed ensemble); otherwise serve the agent that passes; otherwise serve none",
        },
        "exploratory": ["Brier-terminal reward variant (value iteration)", "cost sweep / Pareto frontier",
                        "matched-burden comparison against fixed-length arms",
                        "transfer: trained on Saudi (circular labels), tested on Polish"],
        "claim_boundary": ("Retrospective replay of recorded answers within one case-control cohort (n=252). "
                           "Not prospective validation; not a diagnosis."),
    })
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"protocol frozen -> {path}")
    print(f"code sha256      : {protocol['code_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
