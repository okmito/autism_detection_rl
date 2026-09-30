"""Step 4 — train the RL policies (DQN, PPO) on real data.

Why this script exists
----------------------
``src/policies/dqn.py`` and ``src/policies/ppo.py`` were shipped with training
code that had never been executed: nothing in the repository called
``train_step``, no replay buffer existed, and ``configs/config.yaml`` declared
``policy.type: dqn`` while every actual run used Greedy-IG or Random. This is
the first entry point that trains a learned policy.

What it does
------------
Collects episodes against the existing §10 environment, fills a replay buffer,
and runs DQN / PPO updates. Writes a training artifact to ``results/``.

IMPORTANT — label circularity
-----------------------------
Trained on the Saudi cohort, whose labels are a deterministic sum-threshold over
the questionnaire items themselves (§16.1). A policy can learn that rule
perfectly and learn nothing about autism. These artifacts demonstrate that the
training pipeline runs and produces finite losses; they are NOT evidence that
the policies work clinically. See ``diagnosisReady.md`` §4.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.ingest import load_dataset
from src.env.environment import STOP, run_episode
from src.env.state import get_legal_items, init_state, update_state
from src.models.masked_predictor import MaskedPredictor
from src.policies.dqn import DQNPolicy
from src.policies.ppo import PPOPolicy
from src.policies.replay import ReplayBuffer

TAG = ("preliminary — RL policies newly trained 2026-10-01; trained on "
       "CIRCULAR questionnaire labels (Saudi §16.1); NOT clinical evidence; "
       "V-4 / V-6 / V-7 pending supervisor sign-off")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def split_records(records: List[Dict[str, Any]], seed: int = 0):
    """Stratified 60/20/20 train/val/test split — mirrors Step 2/3."""
    from sklearn.model_selection import train_test_split
    labels = np.array([r["label"] for r in records])
    idx = np.arange(len(records))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=labels)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=labels[tmp])
    pick = lambda ix: [records[i] for i in ix]
    return pick(tr), pick(va), pick(te)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_episode(record, budget, b_min, policy, predictor, buf, n_items,
                    lambda_cost, tau, epsilon=0.0, stochastic=False):
    """Run one episode, recording transitions with per-state legal-action sets.

    The legal set stored on each transition is the set the bootstrap target must
    be masked to — dropping it is what made the original DQN target optimisable
    toward illegal actions.
    """
    state = init_state(record)
    state["questions_remaining"] = budget
    state["budget"] = budget
    remaining = budget
    raw = record["item_responses"]

    steps: List[Dict[str, Any]] = []
    while True:
        legal_items = get_legal_items(state)
        stop_legal = len(steps) >= b_min or not legal_items
        legal = list(legal_items) + ([STOP] if stop_legal else [])

        if remaining == 0 or not legal:
            # Episode terminates here. Terminal reward comes from the §10
            # utility of stopping at this state.
            p_hat = float(np.clip(predictor(state), 0.0, 1.0))
            y = int(record["label"])
            from src.env.costs import get_costs
            costs = get_costs(n_items, mode="uniform")
            asked = [s["a"] for s in steps]
            asked_cost = sum(costs[j] for j in asked if j != STOP)
            R = (1 - (p_hat - y) ** 2) - lambda_cost * asked_cost
            for st in steps:
                buf.add(st["s"], st["a"], st["r"], st["s_next"], False, st["legal"])
            return R, len(steps)

        if epsilon > 0 and hasattr(policy, "select_action"):
            action = policy.select_action(state, legal, epsilon=epsilon)
        elif stochastic and hasattr(policy, "select_action"):
            action = policy.select_action(state, legal, deterministic=False)
        else:
            action = policy(state, legal)

        if action == STOP:
            p_hat = float(np.clip(predictor(state), 0.0, 1.0))
            y = int(record["label"])
            from src.env.costs import get_costs
            costs = get_costs(n_items, mode="uniform")
            asked = [s["a"] for s in steps if s["a"] != STOP]
            asked_cost = sum(costs[j] for j in asked)
            R = (1 - (p_hat - y) ** 2) - lambda_cost * asked_cost
            buf.add_step(state, action, 0.0, None, True, legal)
            return R, len(steps)

        v_raw = raw[action]
        if np.isnan(v_raw):
            raise ValueError(f"Observed NaN for legal item {action}")
        v = int(v_raw)
        nxt = update_state(state, action, v, remaining - 1)
        nxt["questions_remaining"] = remaining - 1
        nxt["budget"] = budget
        steps.append({"s": state, "a": action, "r": 0.0,
                      "s_next": nxt, "legal": legal})
        state = nxt
        remaining -= 1


def train_dqn(train_records, predictor, n_items, budget, episodes, seed,
              batch_size=32, gamma=0.99, lr=1e-3, target_every=200,
              epsilon_start=1.0, epsilon_end=0.05, lambda_cost=0.0, tau=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = ReplayBuffer(capacity=100_000, seed=seed)
    pol = DQNPolicy(n_items=n_items, lr=lr)
    losses: List[float] = []
    total_steps = 0
    t0 = time.time()

    for ep in range(episodes):
        frac = ep / max(episodes - 1, 1)
        eps = epsilon_start + (epsilon_end - epsilon_start) * frac
        rec = train_records[ep % len(train_records)]
        collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                        n_items, lambda_cost, tau, epsilon=eps)
        total_steps += len(buf)

        if len(buf) >= batch_size:
            for _ in range(4):  # a few updates per episode
                losses.append(pol.train_step(buf.sample(batch_size), gamma=gamma))
        if (ep + 1) % target_every == 0:
            pol.update_target()

    return {
        "algo": "dqn",
        "final_loss_mean_50": float(np.mean(losses[-50:])) if losses else None,
        "loss_first_50": float(np.mean(losses[:50])) if losses else None,
        "n_updates": len(losses),
        "buffer": buf.stats(),
        "wall_seconds": round(time.time() - t0, 2),
    }


def train_ppo(train_records, predictor, n_items, budget, episodes, seed,
              lr=3e-4, epochs=4, lambda_cost=0.0, tau=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = PPOPolicy(n_items=n_items)
    stats: List[Dict[str, float]] = []
    t0 = time.time()

    for ep in range(episodes):
        buf = ReplayBuffer(capacity=10_000, seed=seed + ep)
        for k in range(8):  # batch of trajectories per update
            rec = train_records[(ep * 8 + k) % len(train_records)]
            collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                            n_items, lambda_cost, tau, stochastic=True)
        data = buf.all()
        if not data:
            continue
        old_logp = []
        for s, a, r, s_next, done, legal in data:
            # log-prob under the CURRENT policy, before the update
            with torch.no_grad():
                logits = pol.actor(pol._encode(s)).cpu().numpy()[0]
            import numpy as _np
            masked = _np.full(n_items + 1, float("-inf"))
            for aa in legal:
                masked[n_items if aa == STOP else aa] = logits[n_items if aa == STOP else aa]
            m = _np.max(masked[_np.isfinite(masked)])
            e = _np.exp(masked - m); e[~_np.isfinite(masked)] = 0
            p = e / e.sum()
            idx = n_items if a == STOP else a
            old_logp.append(float(_np.log(max(p[idx], 1e-12))))
        for _ in range(epochs):
            stats.append(pol.train_step(data, old_logp=old_logp))

    ent = [s["entropy"] for s in stats]
    vl = [s["value_loss"] for s in stats]
    return {
        "algo": "ppo",
        "entropy_first_20": float(np.mean(ent[:20])) if ent else None,
        "entropy_last_20": float(np.mean(ent[-20:])) if ent else None,
        "value_loss_first": vl[0] if vl else None,
        "value_loss_last": vl[-1] if vl else None,
        "n_updates": len(stats),
        "wall_seconds": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saudi")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algos", default="dqn,ppo")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = load_dataset(args.dataset, synthetic=args.synthetic)
    train_recs, val_recs, test_recs = split_records(records, seed=args.seed)
    n_items = len(records[0]["item_responses"])
    print(f"{args.dataset}: {len(records)} records, n_items={n_items}, "
          f"train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    predictor = MaskedPredictor(n_items=n_items)
    predictor.fit(train_recs, epochs=20, lr=1e-3, batch_size=32, seed=args.seed)
    predictor.fit_calibrator(val_recs)
    print("predictor trained + calibrated")

    out: Dict[str, Any] = {}
    for algo in [a.strip() for a in args.algos.split(",") if a.strip()]:
        print(f"\n=== training {algo} ({args.episodes} episodes, B={args.budget}) ===")
        if algo == "dqn":
            out[algo] = train_dqn(train_recs, predictor, n_items, args.budget,
                                  args.episodes, args.seed)
        elif algo == "ppo":
            out[algo] = train_ppo(train_recs, predictor, n_items, args.budget,
                                  args.episodes, args.seed)
        else:
            raise ValueError(f"unknown algo {algo}")
        for k, v in out[algo].items():
            print(f"  {k}: {v}")

    artifact = {
        "tag": TAG,
        "step": "4 — policy training",
        "dataset": args.dataset,
        "source": "synthetic" if args.synthetic else "real",
        "label_source": "questionnaire (CIRCULAR — see §16.1)",
        "circularity_warning": (
            "Trained against a deterministic sum-threshold oracle over the "
            "questionnaire items. Low loss here reflects learning the label rule, "
            "not autism screening skill. Not clinical evidence."
        ),
        "n_items": n_items,
        "budget": args.budget,
        "episodes": args.episodes,
        "seed": args.seed,
        "splits": {"train": len(train_recs), "val": len(val_recs),
                   "test": len(test_recs)},
        "results": out,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (REPO / "results").mkdir(exist_ok=True)
    dest = REPO / f"results/step4_policy_training_{args.dataset}.json"
    dest.write_text(json.dumps(artifact, indent=2))
    print(f"\n→ {dest.relative_to(REPO)}")
    print("\nNOTE: Part 2 (benchmarking these policies against Greedy/Random/ExactDP "
          "at matched budgets) has NOT been run.")


if __name__ == "__main__":
    main()
