"""Step 21 — diagnose why the served adaptive-RL policy stops after two questions (read-only diagnosis).

Changes nothing in ``src/rl`` (the frozen protocol's code hash stays valid) and rewrites no existing artifact. Writes
``results/rl_adaptive/stopping_diagnosis.json``. Sections:

A. served policy: Q(STOP) vs best Q(ASK) at every node of its decision tree; which constraint binds
   (cost vs the 2-question minimum); reward breakdown of a two-question episode; remaining value of information.
B. out-of-sample stopping behaviour over several outer-fold seeds (nested CV, value iteration at every cost):
   stopping-point distribution, and screening quality *at each stopping point* against the full questionnaire on the
   same children (does stopping early lose decisions the full questionnaire would have got right?).
C. calibration of the belief at the states where the agent stops (is its confidence earned?).
D. model-free DQN: episode-length distribution during training (observed through a wrapper, not by editing the
   trainer) and of the trained greedy policy; Q(ASK) vs Q(STOP) at the same nodes.
E. the cost-selection step that put the served cost at the edge of the grid.

Research prototype; retrospective replay; not a diagnosis.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
np.seterr(over="ignore")

from src.rl.belief import NaiveBayesBelief  # noqa: E402
from src.rl.data import load_polish_qchat10  # noqa: E402
from src.rl.evaluate import rollout, screening_metrics  # noqa: E402
from src.rl.experiment import Study, stratified_folds  # noqa: E402
from src.rl.mdp import C_NEG, C_POS, MDPConfig, obs_val  # noqa: E402
from src.rl.planner import BeliefValueIteration  # noqa: E402
from src.rl.served import load_served_agent  # noqa: E402

OUT = REPO / "results" / "rl_adaptive"
NODES = [("root", []), ("A2=typ", [(1, C_NEG)]), ("A2=atyp", [(1, C_POS)]),
         ("A2=typ,A3=typ", [(1, C_NEG), (2, C_NEG)]), ("A2=typ,A3=atyp", [(1, C_NEG), (2, C_POS)]),
         ("A2=atyp,A3=typ", [(1, C_POS), (2, C_NEG)]), ("A2=atyp,A3=atyp", [(1, C_POS), (2, C_POS)]),
         ("A2,A3,A4=typ", [(1, C_NEG), (2, C_NEG), (3, C_NEG)])]


def codes_of(spec):
    c = np.zeros(10, dtype=np.int64)
    for j, v in spec:
        c[j] = v
    return c


def hist(n_q, upto=10):
    n_q = np.asarray(n_q)
    return {str(k): float(np.mean(n_q == k)) for k in range(1, upto + 1)}


# --------------------------------------------------------------------------------------------- A
def served_section(agent, X, y):
    b, cfg = agent.belief, agent.cfg
    full_info = BeliefValueIteration(b, replace(cfg, cost=0.0))          # value of asking everything, under the model
    nodes = []
    for name, spec in NODES:
        c = codes_of(spec)
        Q = agent.q_values(c[None])[0]
        ask = np.where(np.isfinite(Q[:-1]), Q[:-1], -np.inf)
        j = int(np.argmax(ask))
        o, v = obs_val(c)
        p = float(b.prob_arrays(o, v))
        stop_q = float(Q[-1]) if np.isfinite(Q[-1]) else None
        # expected accuracy if every remaining question could be asked for free, minus accuracy if stopping now
        v_free = float(full_info.V[full_info.index_of(c)])
        stop_now = float(max(p, 1 - p))
        nodes.append({"node": name, "p_positive": p, "q_stop": stop_q, "best_ask": f"A{j + 1}", "q_best_ask": float(ask[j]),
                      "ask_minus_stop": (float(ask[j]) - stop_q) if stop_q is not None else None,
                      "action": "STOP" if int(agent.act_batch(c[None])[0]) == 10 else f"A{int(agent.act_batch(c[None])[0]) + 1}",
                      "expected_accuracy_if_stopped": stop_now,
                      "expected_accuracy_if_all_remaining_asked_free": v_free,
                      "total_remaining_value_of_information": v_free - stop_now})
    # which constraint binds: the same served belief, cost and cap, with the minimum relaxed
    floors = {}
    for mq in (0, 1, 2):
        cfg2 = replace(cfg, min_questions=mq)
        r = rollout(BeliefValueIteration(b, cfg2), X, b, cfg2)
        floors[f"min_questions={mq}"] = {"mean": float(r["n_q"].mean()), "distribution": hist(r["n_q"])}
    # two-question episode reward breakdown (the A2=atyp,A3=atyp branch; one real child of each label if present)
    c = codes_of(NODES[6][1])
    o, v = obs_val(c)
    p = float(b.prob_arrays(o, v))
    ex = {"history": "A2 atypical, A3 atypical, then STOP", "p_at_stop": p, "decision": "REFER" if p >= cfg.tau else "NO_REFERRAL",
          "cost_per_question": cfg.cost, "question_cost_total": -2 * cfg.cost,
          "terminal_reward_if_child_is_ASD": 1.0, "terminal_reward_if_child_is_control": 0.0,
          "return_if_ASD": 1.0 - 2 * cfg.cost, "return_if_control": -2 * cfg.cost,
          "expected_return_under_model": p - 2 * cfg.cost,
          "information_gain_reward": "none (not part of the reward)",
          "incorrect_stop_penalty": "implicit: terminal reward 0 instead of 1",
          "children_in_cohort_with_this_history": int(((X[:, 1] == 1) & (X[:, 2] == 1)).sum()),
          "empirical_asd_rate_for_this_history": float(y[(X[:, 1] == 1) & (X[:, 2] == 1)].mean())}
    emp = {}
    for a2 in (0, 1):
        for a3 in (0, 1):
            m = (X[:, 1] == a2) & (X[:, 2] == a3)
            cc = np.zeros(10, int)
            cc[1], cc[2] = 1 + a2, 1 + a3
            o, v = obs_val(cc)
            emp[f"A2={a2},A3={a3}"] = {"n": int(m.sum()), "empirical_asd_rate": float(y[m].mean()),
                                        "model_p": float(b.prob_arrays(o, v))}
    return {"config": cfg.to_dict(), "nodes": nodes, "minimum_floor_counterfactual": floors,
            "two_question_episode": ex, "two_question_states_in_sample": emp,
            "note": "in-sample description of the served model (it was fitted on all 252 children)"}


# --------------------------------------------------------------------------------------------- B, C
def per_stop_quality(y, p_agent, nq, p_full, tau=0.5):
    out = {}
    da, df = p_agent >= tau, p_full >= tau
    for k in sorted(set(int(x) for x in nq)):
        m = nq == k
        out[str(k)] = {"n": int(m.sum()), "frac": float(m.mean()), "asd_rate": float(y[m].mean()),
                       "agent_accuracy": float(np.mean(da[m] == y[m])),
                       "full_questionnaire_accuracy_same_children": float(np.mean(df[m] == y[m])),
                       "decision_agreement_with_full": float(np.mean(da[m] == df[m])),
                       "full_would_fix": int(np.sum((da[m] != y[m]) & (df[m] == y[m]))),
                       "full_would_break": int(np.sum((da[m] == y[m]) & (df[m] != y[m])))}
    return out


def calibration_at_stop(y, p, nq, bins=(0, .2, .4, .5, .6, .8, 1.0001)):
    rows = []
    for k_lo, k_hi, lab in ((2, 2, "stopped at 2"), (3, 3, "stopped at 3"), (4, 10, "stopped at 4+")):
        m = (nq >= k_lo) & (nq <= k_hi)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = m & (p >= lo) & (p < hi)
            if mm.sum():
                rows.append({"group": lab, "p_bin": [lo, min(hi, 1.0)], "n": int(mm.sum()),
                             "mean_predicted": float(p[mm].mean()), "observed_asd_rate": float(y[mm].mean())})
    return rows


def nested_section(X, y, P, seeds, progress):
    per_seed = []
    for s in seeds:
        P2 = copy.deepcopy(P)
        P2["seed"] = P["seed"] + 1000 * s
        t = time.time()
        res = Study(X, y, P2, workers=1, include_dqn=False).run()
        pr = res["preds"]
        full = pr["full_questionnaire"]
        arms = {}
        for name in ["rl_value_iteration"] + [f"vi_c{c:g}" for c in P["cost_grid"]] + ["threshold_stopping"]:
            d = pr[name]
            m = screening_metrics(y, d["p"], d["n_q"])
            mf = screening_metrics(y, full["p"], full["n_q"])
            arms[name] = {"mean_q": m["mean_questions"], "median_q": m["median_questions"],
                          "stop_distribution": hist(d["n_q"]), "frac_at_cap": float(np.mean(d["n_q"] >= 10)),
                          "ba": m["balanced_accuracy"], "sens": m["sensitivity"], "spec": m["specificity"],
                          "auroc": m["auroc"], "brier": m["brier"],
                          "delta_vs_full": {k: m[k] - mf[k] for k in ("balanced_accuracy", "sensitivity", "specificity",
                                                                     "auroc", "brier")},
                          "per_stopping_point": per_stop_quality(y, d["p"], d["n_q"], full["p"]),
                          "calibration_at_stop": calibration_at_stop(y, d["p"], d["n_q"])}
        base = {}
        for k in (2, 3, 4):
            for fam in ("fixed_official_order", "fixed_best_subset", "random_fixed", "greedy_ig", "beta_greedy",
                        "belief_greedy_ig"):
                d = pr[f"{fam}_k{k}"]
                m = screening_metrics(y, d["p"], d["n_q"])
                base[f"{fam}_k{k}"] = {"ba": m["balanced_accuracy"], "auroc": m["auroc"], "sens": m["sensitivity"],
                                       "spec": m["specificity"]}
        mf = screening_metrics(y, full["p"], full["n_q"])
        per_seed.append({"outer_seed": P2["seed"], "selected_cost_per_fold": [f["selected"]["vi"]["value"]
                                                                               for f in res["selection_log"]],
                         "full_questionnaire": {"ba": mf["balanced_accuracy"], "auroc": mf["auroc"],
                                                "sens": mf["sensitivity"], "spec": mf["specificity"]},
                         "arms": arms, "fixed_length_baselines": base,
                         "inner_selection_tables": [f["tables"]["vi"] for f in res["selection_log"]],
                         "inner_full_ba": [f["inner_full_ba"] for f in res["selection_log"]]})
        progress(f"   seed {s}: {time.time() - t:.0f}s, VI selected costs {per_seed[-1]['selected_cost_per_fold']}")
    return per_seed


# --------------------------------------------------------------------------------------------- D
def dqn_section(X, y, P, seeds, costs, progress):
    import src.rl.dqn as dqn_mod
    from src.rl.dqn import DQNAgent, DQNConfig, train_dqn

    dcfg = DQNConfig(**P["dqn"]["config"])
    orig = dqn_mod._features
    out = {}
    for c in costs:
        cfg = MDPConfig(**{**P["mdp"], "cost": c})
        runs = []
        for s in seeds:
            # Observe training episodes without editing the trainer: the trainer calls _features twice per step,
            # first on the active rows' current histories (all have used == t), then on their next histories
            # (used == t for rows that took STOP, t + 1 for rows that asked). Counting the rows still at t in the
            # second call gives the number of episodes that ended with exactly t questions.
            lengths, state = [], {"second": False}

            def spy(codes, log_odds, cfg_, _state=state, _lengths=lengths):
                used = np.count_nonzero(codes, axis=1)
                if _state["second"]:
                    t = _state["t"]
                    _lengths.extend([t] * int(np.sum(used == t)))
                else:
                    _state["t"] = int(used.min()) if len(used) else 0
                _state["second"] = not _state["second"]
                return orig(codes, log_odds, cfg_)
            dqn_mod._features = spy
            try:
                net = train_dqn(X, y, cfg, dcfg, seed=s)
            finally:
                dqn_mod._features = orig
            L = np.array(lengths)
            n_ep = len(X) * dcfg.epochs
            assert len(L) == n_ep, (len(L), n_ep)
            per_epoch = L.reshape(dcfg.epochs, len(X))
            ag = DQNAgent([net], NaiveBayesBelief.fit(X, y, seed=P["seed"], **P["belief"]), cfg, dcfg)
            r = rollout(ag, X, ag.belief, cfg)
            nodes = []
            for name, spec in NODES:
                cc = codes_of(spec)
                Q = ag.q_values(cc[None])[0]
                ask = np.where(np.isfinite(Q[:-1]), Q[:-1], -np.inf)
                nodes.append({"node": name, "q_stop": float(Q[-1]) if np.isfinite(Q[-1]) else None,
                              "q_best_ask": float(ask.max()), "best_ask": f"A{int(np.argmax(ask)) + 1}"})
            third = dcfg.epochs // 3
            runs.append({"seed": s, "training_episodes": int(n_ep),
                         "training_mean_length": float(L.mean()), "training_median_length": float(np.median(L)),
                         "training_distribution": hist(L),
                         "training_mean_length_by_phase": {"first_third(high_eps)": float(per_epoch[:third].mean()),
                                                           "middle_third": float(per_epoch[third:2 * third].mean()),
                                                           "last_third(eps=0.05)": float(per_epoch[2 * third:].mean())},
                         "training_stop_action_frequency": float(len(L) / (L.sum() + len(L))),
                         "greedy_policy_in_sample_mean_length": float(r["n_q"].mean()),
                         "greedy_policy_in_sample_distribution": hist(r["n_q"]), "nodes": nodes})
            progress(f"   DQN c={c} seed {s}: train mean len {L.mean():.2f}, greedy {r['n_q'].mean():.2f}")
        out[f"c={c}"] = runs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dqn-seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    P = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    R = json.loads((OUT / "experiment_results.json").read_text(encoding="utf-8"))
    served = json.loads((OUT / "served_policy.json").read_text(encoding="utf-8"))
    X, y = load_polish_qchat10()
    agent, info = load_served_agent(REPO / "results")
    print("[A] served policy trace")
    A = served_section(agent, X, y)
    print("[E] cost selection")
    E = {"served_selection": served["cost_selection"], "grid": P["cost_grid"],
         "selected_is_grid_maximum": float(served["mdp"]["cost"]) == max(P["cost_grid"]),
         "delta_select": P["selection"]["delta_select"],
         "nested_cv_selected_costs_primary_seed": [f["selected"]["vi"]["value"] for f in R["selection_log"]]}
    print("[B,C] nested CV over outer seeds (value iteration, all costs)")
    B = nested_section(X, y, P, list(range(args.seeds)), print)
    print("[D] DQN training episodes")
    D = dqn_section(X, y, P, list(range(args.dqn_seeds)), [0.01, 0.02, 0.05], print)
    artifact = {"artifact": "rl_adaptive_stopping_diagnosis", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "served_artifact_sha256": info["artifact_sha256"], "A_served_policy": A, "B_C_nested_cv": B,
                "D_dqn": D, "E_cost_selection": E, "runtime_seconds": round(time.time() - t0, 1),
                "contains_participant_rows": False}
    (OUT / "stopping_diagnosis.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {OUT / 'stopping_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
