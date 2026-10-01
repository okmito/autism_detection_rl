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

# A Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used in this script's progress output. Without this the script raised
# UnicodeEncodeError on its final print *after* every artifact had already been
# written, so it exited non-zero and looked like it had failed. Set once, here.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.ingest import load_dataset
from src.data.splits import stratified_split, split_fingerprint, SCHEME
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
    """Canonical stratified train/val/test split — shared with every other script.

    This previously used a local 60/20/20 ``train_test_split`` (303/101/102 on
    Saudi) while step2/step3/step5 and both demos used the 4-fold scheme
    (284/95/127), so the policies trained here saw a different record set from
    every artifact they were later compared against. Spec §17 requires identical
    splits across runnable baselines. ``seed`` is retained for call compatibility.
    """
    return stratified_split(records, seed=seed)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _legal_for(state, asked, b_min):
    """Legal action set for ``state``, mirroring the §10 rule in ``run_episode``.

    ``STOP`` is legal once ``asked >= b_min``, and is forced into the set when
    no unasked item remains.
    """
    legal_items = get_legal_items(state)
    stop_legal = (asked >= b_min) or (not legal_items)
    return list(legal_items) + ([STOP] if stop_legal else [])


def collect_episode(record, budget, b_min, policy, predictor, buf, n_items,
                    lambda_cost, tau, epsilon=0.0, stochastic=False, gamma=None):
    """Run one episode and store complete transitions in ``buf``.

    Returns ``(terminal_reward, n_items_asked)``.

    Transition bookkeeping
    ----------------------
    Three defects made the RL training a no-op before 2026-10-01. All three are
    fixed here and regression-tested in ``tests/test_rl_training.py``:

    1. **The terminal reward was discarded.** Intermediate transitions were
       stored with ``r = 0.0`` and the terminal transition was written with
       ``r = 0.0`` as well, so DQN optimised ``y = gamma * max_a Q(s', a)`` with
       no reward anywhere and PPO's advantage was identically zero. The §10
       utility is now attached to the terminal transition.
    2. **The legal set was off by one.** Each transition stored the legal set of
       the state it *left* rather than the state it *entered*, so the bootstrap
       mask still marked already-asked items legal. The set stored is now the
       legal set of ``s_next``.
    3. **Terminal transitions were never committed.** ``ReplayBuffer.add_step``
       appends to an internal ``_pending`` list that only
       ``ReplayBuffer.end_episode`` drains; nothing called it, so every STOP
       transition was silently dropped and ``_pending`` grew without bound.
       Transitions are now built explicitly and written with ``add``.

    ``gamma`` selects the reward semantics, which differ by algorithm because
    §11.1 emits only a terminal reward:

    * ``gamma=None`` (default, used by DQN) stores the **immediate** reward, so
      the Q-target ``r + gamma * max_a Q(s', a)`` bootstraps correctly.
    * ``gamma=<float>`` (used by PPO) stores the **discounted return-to-go**,
      ``G_t = r_t + gamma * G_{t+1}`` with ``G_T = r_T``. Storing the immediate
      reward for PPO would leave the advantage zero at every step except STOP,
      which trains the policy to stop immediately; returns-to-go propagate the
      terminal utility back to the decisions that produced it.
    """
    state = init_state(record)
    state["questions_remaining"] = budget
    state["budget"] = budget
    remaining = budget
    raw = record["item_responses"]
    from src.env.costs import get_costs
    costs = get_costs(n_items, mode="uniform")

    transitions: List[Any] = []
    asked: List[int] = []

    def _finish(terminal_state) -> Any:
        p_hat = float(np.clip(predictor(terminal_state), 0.0, 1.0))
        y = int(record["label"])
        asked_cost = sum(float(costs[j]) for j in asked)
        R = (1 - (p_hat - y) ** 2) - lambda_cost * asked_cost
        transitions.append((terminal_state, STOP, R, None, True, None))
        batch = _to_returns(transitions, gamma) if gamma is not None else transitions
        for t in batch:
            buf.add(*t)
        return R

    while True:
        legal_items = get_legal_items(state)
        stop_legal = (len(asked) >= b_min) or (not legal_items)
        legal = list(legal_items) + ([STOP] if stop_legal else [])

        # Terminal state reached without the policy choosing to stop: the budget
        # ran out, or every item has been observed.
        if remaining == 0 or not legal_items:
            return _finish(state), len(asked)

        if epsilon > 0 and hasattr(policy, "select_action"):
            action = policy.select_action(state, legal, epsilon=epsilon)
        elif stochastic and hasattr(policy, "select_action"):
            action = policy.select_action(state, legal, deterministic=False)
        else:
            action = policy(state, legal)

        if action == STOP:
            return _finish(state), len(asked)

        if action not in legal_items:
            raise ValueError(
                f"Illegal action {action} in collect_episode; legal={legal_items}"
            )

        v_raw = raw[action]
        if np.isnan(v_raw):
            raise ValueError(f"Observed NaN for legal item {action}")
        v = int(v_raw)
        nxt = update_state(state, action, v, remaining - 1)
        nxt["questions_remaining"] = remaining - 1
        nxt["budget"] = budget
        asked.append(action)
        # Legal set of the state this transition ENTERS, which is the set the
        # bootstrap target must be masked to.
        transitions.append((state, action, 0.0, nxt, False,
                            _legal_for(nxt, len(asked), b_min)))
        state = nxt
        remaining -= 1

    # Unreachable: the loop only exits through a terminal return.
    raise RuntimeError("collect_episode fell through the episode loop")


def _to_returns(transitions: List[Any], gamma: float) -> List[Any]:
    """Replace each transition's immediate reward with its discounted return.

    Backward sweep, terminal reward last, matching §11.1's terminal-only
    emission.
    """
    out = list(transitions)
    g = 0.0
    for i in range(len(out) - 1, -1, -1):
        s, a, r, s_next, done, legal = out[i]
        g = float(r) + (0.0 if done else gamma * g)
        out[i] = (s, a, g, s_next, done, legal)
    return out


def train_dqn(train_records, predictor, n_items, budget, episodes, seed,
              batch_size=32, gamma=0.99, lr=1e-3, target_every=200,
              epsilon_start=1.0, epsilon_end=0.05, lambda_cost=0.0, tau=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = ReplayBuffer(capacity=100_000, seed=seed)
    pol = DQNPolicy(n_items=n_items, lr=lr, seed=seed)
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
              lr=3e-4, epochs=4, lambda_cost=0.0, tau=0.5, gamma=0.99):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = PPOPolicy(n_items=n_items)
    stats: List[Dict[str, float]] = []
    t0 = time.time()

    for ep in range(episodes):
        buf = ReplayBuffer(capacity=10_000, seed=seed + ep)
        for k in range(8):  # batch of trajectories per update
            rec = train_records[(ep * 8 + k) % len(train_records)]
            # gamma is passed so collect_episode stores discounted returns-to-go
            # rather than the sparse terminal reward; see its docstring.
            collect_episode(rec, budget, 0, pol, predictor.predict_state, buf,
                            n_items, lambda_cost, tau, stochastic=True,
                            gamma=gamma)
        data = buf.all()
        if not data:
            continue
        old_logp = []
        for s, a, r, s_next, done, legal in data:
            # log-prob of the taken action under the CURRENT policy, before the
            # update. Legal-action masking is delegated to the policy so every
            # script derives the ratio identically.
            lp = pol.masked_log_probs(s, legal)
            old_logp.append(float(lp[n_items if a == STOP else a]))
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
    ap.add_argument("--episodes", type=int, default=2000,
                    help="training episodes; 2000 is the budget at which the "
                         "bootstrap objective reaches plateaued behaviour "
                         "(scripts/step7_rl_diagnosis.py). 400 was used before "
                         "that sweep and produced an undertrained policy.")
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

    predictor = MaskedPredictor(n_items=n_items, seed=args.seed)
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
        "predictor_version": predictor.version,
        "split_scheme": SCHEME,
        "split_fingerprint": split_fingerprint(train_recs, val_recs, test_recs),
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
