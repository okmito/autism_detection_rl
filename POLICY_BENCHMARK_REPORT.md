# Step 5 — Policy Benchmark Results (Part 2, 2026-10-01)

**What this measures:** the project's central methodological claim (§35) — learned vs heuristic vs exactly-solved finite-sample optimum, at **matched question-count budgets**.

**Status:** complete. This is the first time any learned policy in this repository has ever been *evaluated*, not merely defined.

**Reproduce:** `.venv/bin/python scripts/step5_policy_benchmark.py --episodes 400 --budgets 1,2,3,4,5,6`
**Artifact:** `results/step5_policy_benchmark_saudi.{json,csv}`

---

## 1. Setup

| | |
|---|---|
| Cohort | Saudi 506, **circular questionnaire labels** (§16.1) |
| Split | 4-fold stratified, identical to Step 3 (284 train / 95 val / 127 test) |
| Predictor | `MaskedMLP[128,64]` + isotonic, trained on train, calibrated on val |
| Evaluator | `src.env.environment.run_episode` — **identical for every policy**, so only the acquisition policy varies |
| Training | DQN and PPO trained at B=6, 400 episodes, seed 0 |
| λ | 0.0 (V-6 pending) |

Policies: `greedy` (information gain), `random`, `dqn` (Double DQN), `ppo`, `exact` (ExactDP backward induction, §14.1).

---

## 2. The headline result: greedy is near-optimal, RL is not

Optimality gap **V\* − V_emp** on the train split under the exact empirical objective (λ=0). Lower is better; 0 means optimal.

| B | V\* | greedy | dqn | ppo | random |
|---|---|---|---|---|---|
| 1 | 0.886826 | **−0.000000** | +0.044531 | +0.040883 | +0.055131 |
| 2 | 0.927780 | **+0.000000** | +0.055214 | +0.015399 | +0.059250 |
| 3 | 0.949005 | **+0.001595** | +0.032676 | +0.035361 | +0.060355 |
| 4 | 0.973594 | **+0.010196** | +0.046746 | +0.057203 | +0.078433 |
| 5 | 0.997653 | **+0.012969** | +0.048338 | +0.105890 | +0.090981 |
| 6 | 1.000000 | **+0.006749** | +0.038949 | +0.138141 | +0.090097 |

### What this says

1. **Greedy-IG is essentially optimal.** Gap ≤ 0.013 at every budget, and exactly 0 at B=1 and B=2. The one-step information-gain heuristic *is* the right policy for this problem.

2. **Both learned policies are worse than greedy.** DQN sits at ~0.04–0.055; PPO at 0.015–0.138. Neither approaches V\*.

3. **PPO gets monotonically worse as budget grows** — gap rises from +0.0154 (B=2) to +0.1381 (B=6). DQN is roughly flat around ~0.045.

4. **Random is the worst baseline** at low budget, but at B=5 and B=6 DQN and PPO are *worse than random*. At B=6, PPO's gap (0.138) is larger than random's (0.090).

**This is a negative result for reinforcement learning on this task, and it is the honest one.** The training did work — DQN loss fell 0.0068 → 0.0002 and PPO's value loss fell — but the policies converged to something worse than a one-step lookahead heuristic.

---

## 3. Test-set performance (Brier / UAR / mean items)

Held-out test split, same predictor, same evaluator:

| B | policy | Brier | UAR | items |
|---|---|---|---|---|
| 3 | greedy | 0.1181 | 0.8335 | 3.00 |
| 3 | **exact** | **0.1181** | 0.8335 | 3.00 |
| 3 | dqn | 0.1260 | 0.8457 | 3.00 |
| 3 | ppo | 0.1181 | 0.8214 | 2.87 |
| 5 | greedy | 0.0709 | 0.8989 | 5.00 |
| 5 | **exact** | **0.0394** | 0.9525 | 3.85 |
| 5 | dqn | 0.0982 | 0.8693 | 5.00 |
| 5 | ppo | 0.1024 | 0.8934 | 3.83 |
| 6 | greedy | 0.0614 | 0.9108 | 6.00 |
| 6 | **exact** | 0.0728 | 0.9050 | 4.50 |
| 6 | dqn | 0.0551 | 0.9167 | 6.00 |
| 6 | ppo | 0.1102 | 0.8996 | 1.84 |

At **B=3 greedy and the exact optimum produce identical test metrics** (Brier 0.1181, UAR 0.8335) — direct confirmation that greedy attains the optimum in that regime.

Note the train/test split in behaviour: on the **train** split exact beats greedy at B=5 and B=6, but on the **test** split greedy beats exact at B=6. The exact DP optimises the *empirical train* objective; greedy generalises better here. That gap between V_emp and test performance is the point of having a held-out split.

---

## 4. Behavioural observations

- **DQN never stops early** — `stopped_early_frac = 1.00` everywhere, and `items_asked_mean = B` exactly. It learned to always consume the budget. With λ=0 there is no cost penalty, so using every question is a locally sensible strategy, but it means DQN has no adaptive *stopping* behaviour at all.
- **PPO does stop early** — at B=6 it asks only 1.84 items on average, versus 6.00 for DQN. PPO learned a much more aggressive stopping policy, which is partly why its gap grows with budget: it terminates before exploiting the information it is paying to collect.
- Random asks 4.02 items at B=6, incidentally close to the optimum's 4.50.

---

## 5. Interpretation — and the mandatory caveat

### Why might RL lose to greedy here?

Not a bug, and not damning of RL in general. Plausible contributors, in rough order of confidence:

1. **The state space is small and the objective is nearly myopic.** With 10 binary items, one-step lookahead on empirical label statistics is very nearly the complete answer. There is little sequential structure for RL to exploit that a one-step calculation misses.
2. **Circular labels make the reward trivially exploitable.** The label is a deterministic function of the items, so reward shaping drives the policy toward "ask the items that determine the sum" — a solution to the *label rule*, not to autism screening. A policy that does this well can still look bad on a held-out split.
3. **Single seed, 400 episodes.** No seed variance is reported. The DQN/PPO numbers could move materially with seeds. `config.eval.seeds = 10` is specified but not yet used.
4. **λ=0 removes the cost pressure** that would make adaptive stopping matter. V-6 must be resolved before the λ-dependent claims are testable.

### The caveat that overrides all of the above

**These labels are circular.** `Qchat-10-Score == sum(A)`, verified 506/506. Every Brier, UAR, AUROC and gap above measures performance against a **questionnaire-defined construct**, not against autism. A policy can learn the scoring rule perfectly and detect nothing.

The artifact carries this in `circularity_warning` and in `tag`, and `tests/test_step5_benchmark.py` asserts both survive.

---

## 6. What this does and does not establish

**Establishes:**
- The training and benchmarking pipeline works end to end, and is now reproducible.
- RQ1/RQ2 now have an initial measurement: at matched budgets, greedy-IG attains the exact optimum and both learned policies do not.
- Greedy's near-optimality (§17 #7) is reproduced against an independently computed V\*.

**Does not establish:**
- Anything clinical. See §5.
- Anything about Polish. The cohort remains sealed behind V-4/V-7.
- Seed robustness. One seed, no variance estimates.
- Anything about λ. λ=0 only; V-6 pending.

---

## 7. Recommended next steps

1. **Multi-seed sweep** — run 10 seeds (`config.eval.seeds`) and report mean ± spread. The single-seed gap differences are currently not interpretable.
2. **Resolve V-6 (λ grid)** — with λ>0 the cost term makes adaptive stopping matter, which is the regime where a sequential policy has any chance of beating greedy.
3. **Diagnose the PPO early-stopping** — asking 1.84 items at B=6 suggests an exploration/credit-assignment issue worth isolating before any further tuning.
4. **Only then consider opening Polish** (V-4/V-7) for the transfer test (RQ3).

**Do not present the RL-vs-greedy comparison as a clinical finding under any circumstances.**
