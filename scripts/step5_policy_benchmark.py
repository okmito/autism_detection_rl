"""Step 5 — benchmark the trained RL policies against the baselines.

Why this script exists
----------------------
Step 4 made DQN and PPO *trainable*. Until 2026-10-01 nothing in this repository
had ever trained a learned policy, so the project's central methodological claim
(§35: learned vs heuristic vs exactly-solved finite-sample optimum at matched
budgets) had never been measured. This script performs that measurement.

Design
------
A NEW script rather than an edit to ``step3_preliminary_reports.py``, so the
existing Step-3 artifacts stay byte-identical and the baseline remains
comparable across the document history.

Policies compared at matched question budget B:
  - ``greedy``  GreedyIGPolicy  — one-step information gain (§17 #7)
  - ``random``  RandomPolicy    — uniform over legal actions (§17 #6)
  - ``dqn``     DQNPolicy       — Double DQN, trained in Step 4
  - ``ppo``     PPOPolicy       — PPO, trained in Step 4
  - ``exact``   ExactDP         — exactly-solved finite-sample optimum (§14.1)

All policies are evaluated with the SAME ``run_episode`` evaluator and the SAME
predictor on the SAME held-out test split, so the comparison isolates the
acquisition policy.

MANDATORY CAVEAT — label circularity
-----------------------------------
Evaluated on the Saudi cohort, whose labels are a deterministic sum-threshold
over the questionnaire items themselves (§16.1). A learned policy can exploit
that rule perfectly. **Results here measure policy behaviour against the
questionnaire-defined construct. They are NOT evidence that any policy detects
autism.** See ``diagnosisReady.md`` §4.

Polish stays sealed (V-4 / V-7).

Outputs (under results/):
  - results/step5_policy_benchmark_saudi.{json,csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"

from src.data.ingest import load_dataset
from src.env.environment import STOP, run_episode
from src.eval.metrics import compute_metrics, acquisition_burden
from src.models.masked_predictor import MaskedPredictor
from src.policies.dqn import DQNPolicy
from src.policies.greedy import GreedyIGPolicy
from src.policies.ppo import PPOPolicy
from src.policies.random_policy import RandomPolicy
from src.solvers.exact_custom import ExactDP
from scripts.step4_train_policies import collect_episode, split_records
from src.policies.replay import ReplayBuffer

TAG = ("preliminary — RL policies benchmarked 2026-10-01 (Step 5); "
       "CIRCULAR questionnaire labels (Saudi §16.1); NOT clinical evidence; "
       "V-4 / V-6 / V-7 pending supervisor sign-off")

CIRCULARITY_WARNING = (
    "Every number here is measured against a deterministic sum-threshold oracle "
    "over the questionnaire items themselves (label == f(item_responses)). A "
    "learned policy can learn that rule perfectly; a low Brier or high UAR "
    "reflects fitting the label rule, not autism screening skill. These "
    "results answer the METHODOLOGICAL question (learned vs heuristic vs "
    "exact optimum at matched budgets) and say nothing clinical."
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def split(records, seed=0):
    """4-fold stratified train/val/test — identical to Step 3 so results align."""
    y = np.array([r["label"] for r in records])
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    tr, te = next(skf.split(np.arange(len(records)), y))
    tr, va = next(skf.split(tr, y[tr]))
    return ([records[i] for i in tr], [records[i] for i in va],
            [records[i] for i in te])


def train_dqn(train_recs, predictor, n_items, budget, episodes, seed):
    """Train DQN on the train split (Step 4 logic, local copy to avoid churn)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = ReplayBuffer(capacity=100_000, seed=seed)
    pol = DQNPolicy(n_items=n_items)
    losses = []
    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        eps = 1.0 + (0.05 - 1.0) * frac
        rec = train_recs[ep % len(train_recs)]
        collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                        n_items, 0.0, 0.5, epsilon=eps)
        if len(buf) >= 32:
            for _ in range(4):
                losses.append(pol.train_step(buf.sample(32), gamma=0.99))
        if (ep + 1) % 200 == 0:
            pol.update_target()
    return pol, {"loss_first_50": float(np.mean(losses[:50])) if losses else None,
                 "loss_final_50": float(np.mean(losses[-50:])) if losses else None,
                 "n_updates": len(losses)}


def train_ppo(train_recs, predictor, n_items, budget, episodes, seed):
    """Train PPO on the train split (Step 4 logic, local copy to avoid churn)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = PPOPolicy(n_items=n_items)
    for ep in range(episodes):
        buf = ReplayBuffer(capacity=10_000, seed=seed + ep)
        for k in range(8):
            rec = train_recs[(ep * 8 + k) % len(train_recs)]
            collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                            n_items, 0.0, 0.5, stochastic=True)
        data = buf.all()
        if not data:
            continue
        old_logp = []
        for s, a, r, s_next, done, legal in data:
            with torch.no_grad():
                logits = pol.actor(pol._encode(s)).cpu().numpy()[0]
            masked = np.full(n_items + 1, float("-inf"))
            for aa in legal:
                j = n_items if aa == STOP else aa
                masked[j] = logits[j]
            m = np.max(masked[np.isfinite(masked)])
            e = np.exp(masked - m); e[~np.isfinite(masked)] = 0
            p = e / e.sum()
            j = n_items if a == STOP else a
            old_logp.append(float(np.log(max(p[j], 1e-12))))
        for _ in range(4):
            pol.train_step(data, old_logp=old_logp)
    return pol, {"n_updates": episodes}


def exact_as_policy(dp):
    """Adapt ExactDP to the ``(state, legal) -> action`` policy protocol.

    ``ExactDP`` exposes ``get_action(state)`` rather than ``__call__``, so it
    cannot be handed to ``run_episode`` directly. The exact policy is a lookup
    keyed on (mask, value, budget remaining); ``get_action`` already handles
    unseen states with a safe fallback, so legality is preserved.
    """
    def policy(state, legal):
        action = dp.get_action(state)
        # Defensive: never emit an action the environment considers illegal.
        if action not in legal:
            items = [a for a in legal if a != STOP]
            return items[0] if items else STOP
        return action
    return policy


def evaluate(records, policy, predictor, B, n_items, lambda_cost=0.0, tau=0.5):
    """Run every record at budget B with the shared §10 evaluator."""
    eps = []
    for rec in records:
        eps.append(run_episode(rec, question_budget=B, b_min=0, policy=policy,
                               predictor=predictor, lambda_cost=lambda_cost,
                               cost_mode="uniform", tau=tau))
    y = np.array([r["label"] for r in records])
    p = np.array([e["p_hat"] for e in eps])
    m = compute_metrics(y, p, tau=tau)
    ab = acquisition_burden([e["items_asked"] for e in eps])
    rewards = [e["R"] for e in eps]
    m["items_asked_mean"] = ab["mean"]
    m["mean_reward"] = float(np.mean(rewards))
    m["stopped_early_frac"] = float(np.mean(
        [1.0 if e["stop_reason"] == "policy_stop" else 0.0 for e in eps]))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saudi")
    ap.add_argument("--budgets", default="1,2,3,4,5,6")
    ap.add_argument("--train-budget", type=int, default=6,
                    help="budget the RL policies are TRAINED at")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-exact", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("Step 5 — policy benchmark at matched budgets")
    print(f"  {TAG}")
    print("=" * 72)

    records = load_dataset(args.dataset, synthetic=args.synthetic)
    train, val, test = split(records, seed=args.seed)
    n_items = len(records[0]["item_responses"])
    print(f"[{args.dataset}] {len(records)} rows | train={len(train)} "
          f"val={len(val)} test={len(test)} | n_items={n_items}")

    predictor = MaskedPredictor(n_items=n_items, hidden=[128, 64],
                                calibration="isotonic")
    predictor.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=args.seed)
    predictor.fit_calibrator(val, method="isotonic")
    print("[predictor] trained + calibrated")

    print(f"\n[train] DQN + PPO at B={args.train_budget}, {args.episodes} episodes")
    dqn, dqn_stats = train_dqn(train, predictor, n_items, args.train_budget,
                               args.episodes, args.seed)
    print(f"   DQN {dqn_stats}")
    ppo, ppo_stats = train_ppo(train, predictor, n_items, args.train_budget,
                               args.episodes, args.seed)
    print(f"   PPO {ppo_stats}")

    budgets = [int(b) for b in args.budgets.split(",")]
    rows: List[Dict[str, Any]] = []

    for B in budgets:
        pols: Dict[str, Any] = {
            "greedy": GreedyIGPolicy(train, n_items=n_items),
            "random": RandomPolicy(seed=args.seed),
            "dqn": dqn,
            "ppo": ppo,
        }
        if not args.skip_exact:
            try:
                t0 = time.time()
                dp = ExactDP(train, n_items=n_items, budget=B, b_min=0,
                             lambda_cost=0.0, cost_mode="uniform")
                sol = dp.solve()
                if sol["status"] == "optimal":
                    pols["exact"] = exact_as_policy(dp)
                    print(f"[B={B}] ExactDP optimal V*={sol['V_star']:.6f} "
                          f"states={sol['n_states']} {time.time()-t0:.1f}s")
                else:
                    print(f"[B={B}] ExactDP {sol['status']} — omitted")
            except Exception as e:
                print(f"[B={B}] ExactDP failed: {type(e).__name__}: {e}")

        print(f"\n  B={B}:")
        for name, pol in pols.items():
            m = evaluate(test, pol, predictor, B, n_items)
            row = {"B": B, "policy": name,
                   "brier": round(m["brier"], 6),
                   "uar": round(m["uar"], 6),
                   "auroc": round(m["auroc"], 6) if m["auroc"] == m["auroc"] else None,
                   "ece": round(m["ece"], 6),
                   "items_asked_mean": round(m["items_asked_mean"], 4),
                   "mean_reward": round(m["mean_reward"], 6),
                   "stopped_early_frac": round(m["stopped_early_frac"], 4)}
            rows.append(row)
            print(f"    {name:<7} brier={m['brier']:.4f} uar={m['uar']:.4f} "
                  f"items={m['items_asked_mean']:.2f} "
                  f"stop_early={m['stopped_early_frac']:.2f}")

    # ---- V* - V_emp optimality gap (§14.3), on the TRAIN split ----
    gaps: List[Dict[str, Any]] = []
    if not args.skip_exact:
        print("\n[gap] V* - V_emp on the train split (empirical objective, λ=0)")
        for B in budgets:
            try:
                dp = ExactDP(train, n_items=n_items, budget=B, b_min=0,
                             lambda_cost=0.0, cost_mode="uniform")
                sol = dp.solve()
                if sol["status"] != "optimal":
                    continue
                v_star = sol["V_star"]
                row = {"B": B, "V_star": round(v_star, 8)}
                for name, pol in [("greedy", GreedyIGPolicy(train, n_items=n_items)),
                                  ("random", RandomPolicy(seed=args.seed)),
                                  ("dqn", dqn), ("ppo", ppo)]:
                    v_emp = dp.evaluate_policy(pol, train)
                    row[f"V_emp_{name}"] = round(v_emp, 8)
                    row[f"gap_{name}"] = round(v_star - v_emp, 8)
                gaps.append(row)
                gap_str = "  ".join(
                    f"{nm}={v_star - row[f'V_emp_{nm}']:+.6f}"
                    for nm in ["greedy", "random", "dqn", "ppo"]
                )
                print(f"    B={B} V*={v_star:.6f}  {gap_str}")
            except Exception as e:
                print(f"    B={B} gap failed: {type(e).__name__}: {e}")

    RESULTS.mkdir(exist_ok=True)
    artifact = {
        "tag": TAG,
        "step": "5 — policy benchmark at matched budgets",
        "dataset": args.dataset,
        "source": "synthetic" if args.synthetic else "real",
        "label_source": "questionnaire (CIRCULAR — §16.1)",
        "circularity_warning": CIRCULARITY_WARNING,
        "n_items": n_items,
        "train_budget": args.train_budget,
        "train_episodes": args.episodes,
        "budgets_evaluated": budgets,
        "seed": args.seed,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "training": {"dqn": dqn_stats, "ppo": ppo_stats},
        "per_budget": rows,
        "optimality_gap": gaps,
        "evaluator": "src.env.environment.run_episode (identical for all policies)",
        "polish_status": "SEALED — not opened (V-4 / V-7)",
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    j = RESULTS / f"step5_policy_benchmark_{args.dataset}.json"
    j.write_text(json.dumps(artifact, indent=2))
    c = RESULTS / f"step5_policy_benchmark_{args.dataset}.csv"
    with open(c, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n→ {j.relative_to(REPO)}")
    print(f"→ {c.relative_to(REPO)}")
    print("\nREMINDER: trained and evaluated on CIRCULAR labels. These results")
    print("are not clinical evidence. See diagnosisReady.md §4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
