"""Step 19 — run the frozen adaptive-RL protocol (nested CV on the Polish cohort) and write the results.

Outputs (``results/`` is git-ignored; nothing here contains participant identifiers):
  results/rl_adaptive/experiment_results.json   aggregated metrics, CIs, tests, gate, selection log
  results/rl_adaptive/predictions.npz           per-row (index-only) predictions and question counts

Research prototype. Retrospective replay of recorded answers; not prospective validation; not a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.provenance import git_sha  # noqa: E402
from src.rl.data import QCHAT10_OFFICIAL_CUTOFF, load_polish_qchat10, load_saudi_qchat10  # noqa: E402
from src.rl.evaluate import adaptivity_profile, rollout, screening_metrics  # noqa: E402
from src.rl.experiment import (Study, code_fingerprint, holm, non_inferiority, paired_bootstrap,  # noqa: E402
                               run_dqn_jobs, strip_draws)

OUT = REPO / "results" / "rl_adaptive"
MULTI = REPO / "results" / "policies_v3_multiseed"


def legacy_dqn_loader_factory():
    """Saudi-trained DQN v3 artifacts (the existing RL policies), selected seed per budget from step 16."""
    art = REPO / "results" / "polish_fixed_budget_external_evaluation.json"
    if not art.exists() or not MULTI.exists():
        return None
    spread = json.loads(art.read_text(encoding="utf-8")).get("dqn_seed_spread", {})
    cache = {}

    def load(k: int):
        if k in cache:
            return cache[k]
        sel = (spread.get(f"B{k}") or {}).get("selected_seed")
        path = MULTI / f"dqn_v3_B{k}_seed{sel}.pt"
        if sel is None or not path.exists():
            cache[k] = None
            return None
        import torch
        from src.policies.dqn import DQNPolicy
        blob = torch.load(path, map_location="cpu", weights_only=False)
        pol = DQNPolicy(n_items=blob["n_items"], m_list=blob["m_list"], seed=blob["seed"])
        pol.q.load_state_dict(blob["q_state_dict"])
        pol.q.eval()
        cache[k] = pol
        return pol
    return load


def official_cutoff(X: np.ndarray) -> dict:
    """Q-CHAT-10 sum score on all 10 items mapped monotonically so that p >= 0.5 <=> sum >= 4 (not a probability)."""
    s = np.nansum(X, axis=1)
    p = 0.5 + (s - (QCHAT10_OFFICIAL_CUTOFF - 0.5)) / 20.0
    return {"p": p, "n_q": np.full(len(s), X.shape[1]), "forced": np.zeros(len(s), bool)}


def summarise(y, preds, cfg_max, tau):
    out = {}
    for name, d in sorted(preds.items()):
        m = screening_metrics(y, d["p"], d["n_q"], tau, max_questions=cfg_max, forced=d.get("forced"))
        if "seq" in d:
            m["adaptivity"] = adaptivity_profile(d["seq"], d["codes"])
        out[name] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    proto_path = OUT / "protocol.json"
    if not proto_path.exists():
        print("BLOCKED: run scripts/step18_rl_adaptive_protocol.py first")
        return 2
    P = json.loads(proto_path.read_text(encoding="utf-8"))
    sha = code_fingerprint([REPO / f for f in P["code_files"]])
    if sha != P["code_sha256"]:
        print("BLOCKED: src/rl code changed after the protocol was frozen")
        return 2
    print("[1/5] protocol verified", P["frozen_at_utc"], sha[:12])

    X, y = load_polish_qchat10()
    tau = P["mdp"]["tau"]
    print(f"[2/5] Polish cohort n={len(y)} (ASD {int(y.sum())} / control {int((1 - y).sum())})")
    study = Study(X, y, P, workers=args.workers, progress=print, legacy_dqn_loader=legacy_dqn_loader_factory())
    res = study.run()
    preds = res["preds"]
    preds["official_cutoff_full10"] = official_cutoff(X)

    print("[3/5] statistics")
    summary = summarise(y, preds, P["mdp"]["max_questions"], tau)
    nb, bseed = P["bootstrap"]["n_resamples"], P["bootstrap"]["seed"]
    full = preds["full_questionnaire"]
    comparisons, ni, pvals = {}, {}, {}
    for agent in ("rl_value_iteration", "rl_dqn", "threshold_stopping"):
        bt = paired_bootstrap(y, preds[agent], full, tau, nb, bseed)
        comparisons[f"{agent}_vs_full_questionnaire"] = bt
        ni[agent] = {d: non_inferiority(bt, "balanced_accuracy", d)
                     for d in [P["non_inferiority"]["delta"]] + P["non_inferiority"]["sensitivity_deltas"]}
        if agent.startswith("rl_"):
            pvals[agent] = ni[agent][P["non_inferiority"]["delta"]]["p_one_sided"]
        comparisons[f"{agent}_vs_official_cutoff"] = paired_bootstrap(y, preds[agent], preds["official_cutoff_full10"],
                                                                      tau, nb, bseed)
    holm_res = holm(pvals, 0.05)
    comparisons["rl_value_iteration_vs_rl_dqn"] = paired_bootstrap(y, preds["rl_value_iteration"], preds["rl_dqn"],
                                                                   tau, nb, bseed)
    for agent in ("rl_value_iteration", "rl_dqn"):
        comparisons[f"{agent}_vs_threshold_stopping"] = paired_bootstrap(y, preds[agent], preds["threshold_stopping"],
                                                                         tau, nb, bseed)
    # matched burden: the fixed-length arm with k = round(mean questions of the agent)
    matched = {}
    for agent in ("rl_value_iteration", "rl_dqn"):
        k = int(np.clip(round(summary[agent]["mean_questions"]), 1, 9))
        matched[agent] = {"k": k}
        for arm in ("fixed_best_subset", "fixed_official_order", "random_fixed", "belief_greedy_ig", "greedy_ig",
                    "beta_greedy"):
            matched[agent][arm] = paired_bootstrap(y, preds[agent], preds[f"{arm}_k{k}"], tau, nb, bseed)

    # gate
    gate = {}
    delta = P["non_inferiority"]["delta"]
    for agent in ("rl_value_iteration", "rl_dqn"):
        bt = comparisons[f"{agent}_vs_full_questionnaire"]
        q_hi = bt["mean_questions"]["ci95"][1] + bt["mean_questions"]["b"]
        g = {"G1_ba_non_inferior_holm": bool(holm_res[agent]["reject_h0"]),
             "G2_fewer_questions": bool(q_hi < P["mdp"]["max_questions"]),
             "G2_mean_questions_ci95_upper": float(q_hi),
             "G3_sensitivity_ci95_lower": bt["sensitivity"]["ci95"][0],
             "G3_sensitivity_ok": bool(bt["sensitivity"]["ci95"][0] > -0.10),
             "G4_engineering": "verified by tests/test_rl_*.py (not by this script)"}
        g["passes_research_gate"] = g["G1_ba_non_inferior_holm"] and g["G2_fewer_questions"] and g["G3_sensitivity_ok"]
        gate[agent] = g
    eligible = [a for a in ("rl_value_iteration", "rl_dqn") if gate[a]["passes_research_gate"]]
    gate["selected_for_application"] = eligible[0] if eligible else None

    # stability: value iteration with three other outer-fold seeds (cheap; DQN excluded for cost)
    stability = []
    for s in (1, 2, 3):
        P2 = json.loads(json.dumps(P))
        P2["seed"] = P["seed"] + 1000 * s
        r2 = Study(X, y, P2, workers=1, include_dqn=False).run()["preds"]
        m_vi = screening_metrics(y, r2["rl_value_iteration"]["p"], r2["rl_value_iteration"]["n_q"], tau)
        m_full = screening_metrics(y, r2["full_questionnaire"]["p"], r2["full_questionnaire"]["n_q"], tau)
        m_th = screening_metrics(y, r2["threshold_stopping"]["p"], r2["threshold_stopping"]["n_q"], tau)
        stability.append({"outer_seed": P2["seed"],
                          "rl_value_iteration": {k: m_vi[k] for k in ("balanced_accuracy", "auroc", "mean_questions")},
                          "threshold_stopping": {k: m_th[k] for k in ("balanced_accuracy", "auroc", "mean_questions")},
                          "full_questionnaire": {k: m_full[k] for k in ("balanced_accuracy", "auroc")}})

    print("[4/5] exploratory transfer: trained on Saudi (circular labels), tested on Polish")
    transfer = None
    if not args.skip_transfer:
        Xs, ys = load_saudi_qchat10()
        st = Study(Xs, ys, P, workers=args.workers, progress=print)
        b = st.belief(Xs, ys)
        jobs = []
        for f in range(P["inner_folds"]):
            from src.rl.experiment import stratified_folds
            inner = stratified_folds(ys, P["inner_folds"], P["seed"] + 1)
            jobs += st._dqn_jobs_for((("saudi", "inner"), f), Xs[inner != f], ys[inner != f])
        jobs += st._dqn_jobs_for(("saudi", "refit"), Xs, ys)
        sds = run_dqn_jobs(jobs, args.workers, print)
        inner = st.inner_tables(Xs, ys, sds, tag=("saudi", "inner"))
        c_vi, c_dqn = inner["selected"]["vi"][0], inner["selected"]["dqn"][0]
        from src.rl.baselines import FixedOrder
        tp = {"full_questionnaire": rollout(FixedOrder(range(10), 10), X, b, st.bcfg),
              "rl_value_iteration": rollout(st.vi(b, c_vi), X, b, st.cfg),
              "rl_dqn": rollout(st.dqn_agent([sds[(("saudi", "refit"), c_dqn, s)] for s in P["dqn"]["seeds"]], b,
                                             c_dqn), X, b, st.cfg)}
        transfer = {"selected_cost": {"rl_value_iteration": c_vi, "rl_dqn": c_dqn},
                    "saudi_inner_full_ba": inner["full_ba"],
                    "metrics_on_polish": summarise(y, tp, P["mdp"]["max_questions"], tau),
                    "vs_full": {a: paired_bootstrap(y, tp[a], tp["full_questionnaire"], tau, nb, bseed)
                                for a in ("rl_value_iteration", "rl_dqn")}}

    print("[5/5] write")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predictions.npz", y=y, outer_folds=res["outer_folds"],
                        **{f"{k}__p": v["p"] for k, v in preds.items()},
                        **{f"{k}__n_q": v["n_q"] for k, v in preds.items()})
    artifact = {
        "artifact": "rl_adaptive_experiment_results",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO), "protocol_code_sha256": P["code_sha256"], "protocol_frozen_at": P["frozen_at_utc"],
        "cohort": {"n": int(len(y)), "asd": int(y.sum()), "control": int((1 - y).sum())},
        "claim_boundary": P["claim_boundary"],
        "summary": summary, "comparisons": comparisons, "non_inferiority": ni, "holm": holm_res,
        "matched_burden": matched, "integration_gate": gate, "selection_log": res["selection_log"],
        "stability_value_iteration": stability, "transfer_saudi_to_polish": transfer,
        "contains_participant_rows": False, "runtime_seconds": round(time.time() - t0, 1),
    }
    (OUT / "experiment_results.json").write_text(json.dumps(strip_draws(artifact), indent=2), encoding="utf-8")
    for a in ("full_questionnaire", "official_cutoff_full10", "rl_value_iteration", "rl_dqn", "threshold_stopping"):
        m = summary[a]
        print(f"   {a:26s} BA {m['balanced_accuracy']:.3f}  sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}"
              f"  AUROC {m['auroc']:.3f}  meanQ {m['mean_questions']:.2f}")
    print("   gate:", {a: gate[a]["passes_research_gate"] for a in ("rl_value_iteration", "rl_dqn")},
          "->", gate["selected_for_application"])
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
