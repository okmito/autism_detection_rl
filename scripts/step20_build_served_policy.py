"""Step 20 — build the policy the application serves, if (and only if) an RL agent passed the integration gate.

Reads the gate from ``results/rl_adaptive/experiment_results.json`` (step 19). For the selected agent it applies the
*same* constrained cost-selection procedure the nested cross-validation evaluated (inner 3-fold CV, fewest mean
questions with balanced accuracy within delta_select of the full questionnaire) to the whole development cohort,
refits, and writes ``results/rl_adaptive/served_policy.json`` (+ DQN weights when the DQN agent is served).

``--inner-repeats R`` (value iteration only; default 1 = the pre-registered single split) averages that inner-CV
estimate over R fold assignments before applying the same rule (src/rl/selection.py). Step 21 showed the single-split
estimate accepted a cost (0.05) that violates the rule on average; step 23 validates the averaged estimate by nested CV.

When no agent passed, the artifact is still written, recording the failed gate and its reason, so the application
can tell the user why the RL policy is unavailable instead of failing silently.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.provenance import git_sha  # noqa: E402
from src.rl.data import load_polish_qchat10  # noqa: E402
from src.rl.experiment import Study, run_dqn_jobs, stratified_folds  # noqa: E402
from src.rl.served import SERVED_NAME, write_served_policy  # noqa: E402

OUT = REPO / "results" / "rl_adaptive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-repeats", type=int, default=1,
                    help="average the inner-CV selection estimate over this many fold assignments (VI only)")
    ap.add_argument("--backup-as", default=None,
                    help="copy the existing served_policy.json to this file name (in results/rl_adaptive) first")
    args = ap.parse_args()
    P = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    R = json.loads((OUT / "experiment_results.json").read_text(encoding="utf-8"))
    gate = R["integration_gate"]
    agent = gate.get("selected_for_application")
    base = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(REPO),
            "protocol_code_sha256": P["code_sha256"], "experiment_generated_utc": R["generated_utc"]}
    if agent is None:
        reasons = []
        for a in ("rl_value_iteration", "rl_dqn"):
            g = gate[a]
            failed = [k for k in ("G1_ba_non_inferior_holm", "G2_fewer_questions", "G3_sensitivity_ok") if not g[k]]
            reasons.append(f"{a} failed {', '.join(failed)}")
        payload = {**base, "agent": None,
                   "gate": {"passed": False, "reason": "no RL agent passed the integration gate (" +
                            "; ".join(reasons) + ")", "details": gate}}
        write_served_policy(OUT / SERVED_NAME, payload)
        print("gate failed; recorded:", payload["gate"]["reason"])
        return 0

    X, y = load_polish_qchat10()
    st = Study(X, y, P, workers=14, progress=print)
    sds = None
    if agent == "rl_dqn":
        inner = stratified_folds(y, P["inner_folds"], P["seed"] + 1)
        jobs = []
        for f in range(P["inner_folds"]):
            jobs += st._dqn_jobs_for((("served", "inner"), f), X[inner != f], y[inner != f])
        sds = run_dqn_jobs(jobs, 14, print)
    key = "vi" if agent == "rl_value_iteration" else "dqn"
    if args.inner_repeats > 1:
        if agent != "rl_value_iteration":
            raise SystemExit("--inner-repeats is implemented for the value-iteration agent only")
        from src.rl.selection import repeated_inner_table
        rep = repeated_inner_table(st, X, y, args.inner_repeats)
        tables = {"full_ba": rep["full_ba"], "tables": {key: rep["table"]}, "selected": {key: rep["selected"]}}
        method = {"method": f"inner 3-fold CV averaged over {args.inner_repeats} fold assignments (src/rl/selection.py)",
                  "meets_rule_in_n_repeats": {str(c): v for c, v in rep["meets_rule_in_n_repeats"].items()},
                  "single_split_selection": list(st.select(rep["per_repeat"][0]["table"],
                                                           rep["per_repeat"][0]["full_ba"]))}
    else:
        tables = st.inner_tables(X, y, sds, tag=("served", "inner") if sds is not None else None)
        method = {"method": "inner 3-fold CV, single split (pre-registered)"}
    cost, rule = tables["selected"][key]
    belief = st.belief(X, y)
    belief.trained_on = "Polish Q-CHAT cohort (Q-CHAT-10 projection), clinician-established labels, n=252"
    cfg = st.cfg.__class__(**{**st.cfg.to_dict(), "cost": float(cost)})
    summ = R["summary"][agent]
    cmp_ = R["comparisons"][f"{agent}_vs_full_questionnaire"]
    payload = {**base, "agent": agent, "belief": belief.to_dict(), "mdp": cfg.to_dict(),
               "cost_selection": {"rule": rule, "inner_full_ba": tables["full_ba"], **method,
                                  "table": {str(k): v for k, v in tables["tables"][key].items()}},
               "gate": {"passed": True, "reason": None, "details": gate[agent]},
               "expected_performance": {
                   "source": "nested cross-validation (step 19), Polish cohort, retrospective replay",
                   "balanced_accuracy": summ["balanced_accuracy"], "sensitivity": summ["sensitivity"],
                   "specificity": summ["specificity"], "auroc": summ["auroc"],
                   "mean_questions": summ["mean_questions"], "median_questions": summ["median_questions"],
                   "ba_difference_vs_full_questionnaire": cmp_["balanced_accuracy"]["difference"],
                   "ba_difference_ci95": cmp_["balanced_accuracy"]["ci95"]}}
    if agent == "rl_dqn":
        import torch
        from src.models.artifact_integrity import sha256_file
        from src.rl.dqn import DQNAgent
        ag = DQNAgent.train(X, y, cfg, st.dcfg, seeds=P["dqn"]["seeds"], belief=belief,
                            belief_kwargs=P["belief"])
        wpath = OUT / "served_dqn_weights.pt"
        torch.save(ag.state_dicts(), wpath)
        payload["dqn_weights"] = {"file": wpath.name, "sha256": sha256_file(wpath), "seeds": P["dqn"]["seeds"]}
        payload["dqn_config"] = st.dcfg.to_dict()
    val = OUT / "selection_correction_validation.json"
    if args.inner_repeats > 1 and val.exists():
        V = json.loads(val.read_text(encoding="utf-8"))
        rows = [r["after"] for r in V["per_seed"]]
        payload["expected_performance_corrected_selection"] = {
            "source": f"post-hoc nested CV of the averaged selection (step 23), {len(rows)} outer seeds",
            **{k: float(sum(r[k] for r in rows) / len(rows))
               for k in ("ba", "sens", "spec", "auroc", "brier", "mean_q", "delta_ba", "delta_auroc")}}
    if args.backup_as and (OUT / SERVED_NAME).exists():
        shutil.copy2(OUT / SERVED_NAME, OUT / args.backup_as)
    write_served_policy(OUT / SERVED_NAME, payload)
    print(f"served policy: {agent}, cost {cost} ({rule}) -> {OUT / SERVED_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
