# Step 5 — Policy Benchmark Results

> ## ⚠️ Read this before any number below
>
> **1. The §2 conclusion is RETRACTED.** The original claim — greedy-IG attains
> the exact optimum, the learned policies do not, presented as a negative result
> for RL — was measured with a rollout collector that never delivered a terminal
> reward, so no learned policy had ever received a learning signal. It was a
> measurement defect, not a finding. Current numbers are in **§A**.
>
> **2. The objective saturates, and that reframes everything.** At λ = 0 the
> empirical support becomes *pure* after roughly four questions (0.000 of states
> at depth 1 → 0.820 at depth 4 → 1.000 at depth 7), so `u_stop = 1 − (p_emp − y)²`
> reaches exactly 1.0, `V*` is exactly **1.000000**, and no further question can
> raise the reward. The exact reference's `pure_support_frac_at_stop` is **1.00**:
> it stops early when the objective saturates, not when the evidence is
> sufficient. The measured value of adaptive stopping at λ = 0 is a property of
> the label rule, not evidence about diagnostic information content. See
> `V6_LAMBDA_DECISION.md` §4 and `diagnosisReady.md` §4A.
>
> **3. H1 is not supported at B = 5 and B = 6.** The exact best fixed subset
> attains better held-out Brier, even though greedy is the policy closest to `V*`
> on the empirical objective at every budget. This holds for **every** adaptive
> arm tried (greedy, beta_greedy, DQN), so it is not an artefact of the greedy
> heuristic. See §A.4.
>
> **4. All labels are circular** (§16.1). Nothing here is clinical evidence.

**Original document preserved below. Read §A for current numbers.**

---

# ORIGINAL (2026-10-01, commit `b9f9049`) — RETRACTED, retained for the record

**What this measures:** the project's central methodological claim (§35) — learned vs heuristic vs exactly-solved finite-sample optimum, at **matched question-count budgets**.

**Status:** superseded — see the correction notice above and §A.

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

---
# §A — RE-MEASURED RESULTS (2026-10-01, after five corrections: collector, split, predictor, seeding, episode budget)

**Reproduce:** `.venv-win/Scripts/python scripts/step5_policy_benchmark.py` (default `--episodes 2000`)
**Artifact:** `results/step5_policy_benchmark_saudi.{json,csv}`
**Setup:** Saudi 506, circular questionnaire labels (§16.1) · canonical 4-fold
split 284/95/127, fingerprint `b2021998a83e7224` · `MaskedMLP[128,64]` +
isotonic, `predictor_version: 2` · seed 0 · trained at B=6 for **2,000 episodes** ·
λ=0 · identical `run_episode` evaluator for every policy · **reproducible from the recorded seed**.

Three corrections landed between the retracted run and this one, and each one
invalidated the previous numbers. All are recorded rather than overwritten:

| # | Correction | See |
|---|---|---|
| P0-1/2/3 | `collect_episode` never delivered a terminal reward, stored the wrong legal set, and dropped terminal rows | `RL_TRAINING_REPORT.md §2` |
| P0-9 | The canonical split was contaminated: `skf.split(train_idx, ...)` returns indices relative to `train_idx`, but four copies of the split passed them straight into `records[i]`. On the 506-row cohort **71 of 95 validation records were also in train and 24 were in test**. Row counts came out right (284/95/127), which is why it survived review | `AUDIT_REPORT.md` |
| P1-b | `MaskedPredictor` v1 fitted on **one** masked view per record (284 states), trained the `budget_norm` feature on the `budget=10` grid while inference runs at `budget=B`, and fitted the calibrator on **fully observed** states only — extrapolating on every partial state it was asked to score | A.2 |
| P0-10 | `nn.Linear` drew from torch's global RNG before the caller seeded it, so **every artifact was non-reproducible regardless of its recorded seed** | A.2, `AUDIT_REPORT.md` |
| P1-e | The 400-episode DQN was an **undertrained network**; the bootstrap objective needs ~2,000 episodes | **A.5** |

## A.1 The learning signal now exists

Before the P0 collector fix the replay buffer contained `rewards = {0.0}` and
`dones = {0.0}`. After it:

| | DQN loss (first 50 → final 50) | PPO value loss | PPO entropy (first 20 → last 20) | terminal rows in buffer |
|---|---|---|---|---|
| before | 0.0068 → 0.0002 | — | — | **0** |
| after (400 ep) | 0.0364 → 0.1696 | 0.3539 → 0.0604 | 2.1033 → 1.8715 | 1,580 |
| after (2,000 ep) | 0.0366 → **0.0042** | 0.3539 → 0.0776 | 2.1033 → **1.5679** | 7,980 |

Three things to read here.

1. **DQN loss rises at first, then falls.** The early rise is expected and not a
   regression: previously it was fitting an all-zero target, which it could drive
   to zero by shrinking every Q-value. It then fits a real high-variance Monte
   Carlo target. A monotonically falling loss is not a correctness property of
   DQN, so `tests/test_step5_benchmark.py::test_training_converged` no longer
   asserts it; it asserts the presence of a learning signal instead, and the
   delivery of the terminal reward is pinned directly in
   `tests/test_rl_training.py::test_collect_episode_stores_the_terminal_reward`.
2. **PPO entropy now falls.** Under the defect the advantage was identically zero
   and the entropy bonus was the sole active term, driving the actor *toward*
   uniform. A falling entropy is the signature of a policy gradient that is
   actually optimising something.
3. **Only the 2,000-episode run is a trained policy.** A.5 shows the earlier
   400-episode figure was an undertrained network.

## A.2 Predictor v1 → v2

Same architecture, same hyperparameters, same split; the change is the training
and calibration *distribution*. v1 is reproduced exactly via
`masks_per_record=1, eval_budgets=[n_items]`.

| | v1 (1 mask/record, `budget=10` grid, calibrator on full states) | v2 (16 masks/record, eval-budget grid, calibrator on inference grid) |
|---|---|---|
| training states | 284 | 4,848 |
| calibration points | 95 | 1,520 |
| test Brier | 0.0630 | **0.0512** |
| test UAR | 0.8513 | **0.8930** |
| test AUROC | 0.9627 | **0.9882** |
| test ECE | 0.0560 | **0.0475** |

(B=6, greedy policy, held-out test split of 127. An earlier attempt at v2 fitted
the calibrator on one state per validation record; that gave only 95 calibration
points, and isotonic made Brier *worse* (0.0982 → 0.1254). Sampling
`masks_per_record` states per validation record fixed it. The numbers above are
measured on identical inference-grid states for both versions.)

**Residual issue, stated rather than hidden:** on a *broad* sample of states
drawn uniformly over (budget, depth, subset), isotonic calibration still moves
ECE the wrong way for v2 (0.0487 → 0.0729), while improving it along the actual
B=6 episode path. Calibration quality is therefore distribution-dependent and
`isotonic` is a poor fit for partial states on circular labels. `platt` is what
the browser demo uses for this reason. This is open, not resolved.

## A.3 Optimality gap V\* − V_emp (train split, λ=0)

Trained for **2,000** episodes, not 400. See A.4 for why that matters.

| B | V\* | greedy | exact_fixed_subset | irt_cat | dqn | ppo | random |
|---|---|---|---|---|---|---|---|
| 1 | 0.889171 | **0.000000** | 0.000000 | 0.000000 | +0.035385 | +0.057106 | +0.049067 |
| 2 | 0.938629 | **0.000000** | 0.000000 | +0.000356 | +0.057551 | +0.076244 | +0.061845 |
| 3 | 0.961628 | **+0.001643** | +0.003052 | +0.014975 | +0.050277 | +0.099243 | +0.065956 |
| 4 | 0.978821 | **+0.007205** | +0.011126 | +0.018065 | +0.052995 | +0.116436 | +0.073449 |
| 5 | 0.998239 | **+0.013263** | +0.019995 | +0.027008 | +0.054706 | +0.135854 | +0.088180 |
| 6 | 1.000000 | **+0.004225** | +0.014085 | +0.026892 | +0.034918 | +0.137615 | +0.091082 |

**Greedy-IG's near-optimality survives every correction**, and no policy exceeds
`V*` at any budget
(`tests/test_step5_benchmark.py::test_no_policy_beats_exact_optimum`).

`static_rfe` is absent from this table only because it selects the *identical*
subset to `exact_fixed_subset` at every budget on this cohort, so its gap column
is the same to eight decimal places.

## A.4 Held-out test performance, and the H1 result

| B | policy | Brier | UAR | items | stop_early | train gap |
|---|---|---|---|---|---|---|
| 3 | greedy | **0.0876** | 0.8335 | 3.00 | 0.00 | +0.001643 |
| 3 | exact_fixed_subset | 0.1034 | 0.8335 | 3.00 | 0.00 | +0.003052 |
| 4 | greedy | **0.0691** | 0.7976 | 4.00 | 0.00 | +0.007205 |
| 4 | exact_fixed_subset | 0.0850 | 0.8335 | 4.00 | 0.00 | +0.011126 |
| 5 | greedy | 0.0655 | 0.8633 | 5.00 | 0.00 | +0.013263 |
| 5 | exact_fixed_subset | **0.0448** | **0.9227** | 5.00 | 0.00 | +0.019995 |
| 6 | exact_fixed_subset | **0.0479** | 0.9167 | 6.00 | 0.00 | +0.014085 |
| 6 | beta_greedy | 0.0482 | 0.9049 | 6.00 | 0.00 | **+0.004225** |
| 6 | greedy | 0.0512 | 0.8930 | 6.00 | 0.00 | **+0.004225** |
| 6 | dqn | 0.0640 | **0.9112** | 5.98 | 0.02 | +0.034918 |
| 6 | exact (DP) | 0.0644 | 0.8812 | 4.45 | 0.77 | 0 by definition |
| 6 | irt_cat | 0.0686 | 0.8993 | 6.00 | 0.00 | +0.026892 |
| 6 | random | 0.0918 | 0.8220 | 4.02 | 0.55 | +0.091082 |
| 6 | ppo | 0.1162 | 0.8398 | 1.43 | 1.00 | +0.137615 |

`beta_greedy` is the P1-f arm: a Beta(1,1)-smoothed posterior with EVOI selection
and cost-based stopping. It is **bit-identical to `greedy` on the empirical train
objective** (both +0.004225 at B=6) yet generalises better on held-out Brier
(0.0482 vs 0.0512). That is a direct measurement of the saturation problem: on a
saturated support the empirical objective is constant and cannot distinguish the
two policies, while the held-out data can. `GreedyIGPolicy` itself is untouched
— the new policy is an additional arm.

### H1 is testable, and is not supported at the upper budgets

H1 (spec §4): *"At each pre-declared question-count budget of 3–6 items … the
primary adaptive policy will achieve lower held-out Brier loss than the exact
best fixed-subset policy … with both methods evaluated at exactly `B` questions."*

| B | exact best fixed subset | beta_greedy | greedy | dqn | H1 (any adaptive arm) |
|---|---|---|---|---|---|
| 3 | 0.1034 | **0.0876** | **0.0876** | **0.0864** | supported |
| 4 | 0.0850 | **0.0735** | **0.0691** | **0.0748** | supported |
| 5 | **0.0448** | 0.0709 | 0.0655 | 0.0660 | **not supported** |
| 6 | **0.0479** | 0.0482 | 0.0512 | 0.0640 | **not supported** |

**The adaptive advantage shrinks and then reverses as the budget grows, for every
adaptive arm tried** — greedy, the Beta-prior EVOI policy, and DQN all fail at
B = 5 and B = 6. So the reversal is not an artefact of the greedy heuristic.

It also holds *even though* greedy and beta_greedy are the policies closest to
`V*` on the empirical objective at every budget (+0.0042 against the fixed
subset's +0.0141 at B=6). The advantage lives inside the training support; the
per-episode item ordering is the part that fails to transfer. A fixed subset
chosen to minimise training Brier is a lower-variance object, and once the budget
is large enough that *which* six matters less than *a good* six, that wins.

The **exact** arm shows the same pattern (0.0644 vs 0.0479 at B=6) despite being
the true optimum of the empirical objective — it stops early on 77% of episodes,
and A.6 explains why that early stopping does not transfer.

Spec §4 pre-registers that *"failure to outperform the exact best fixed subset or
generic CAT is a valid result"*, so this is reportable. It could not have been
obtained before P1-d, when the comparator did not exist.

## A.5 The DQN collapse was an undertrained network, not a property of RL

`scripts/step7_rl_diagnosis.py`, 5 seeds × {200, 400, 1000, 2000} episodes,
60 held-out records, trained predictor. Full table in
`results/rl_diagnosis_saudi.{json,csv}`.

| objective | episodes | items (mean ± sd) | UAR (mean ± sd) | distinct states |
|---|---|---|---|---|
| bootstrap | 200 | 0.00 ± 0.00 | 0.5000 ± 0.0000 | 167 |
| bootstrap | 400 | 0.60 ± 0.55 | 0.6615 ± 0.1485 | 352 |
| bootstrap | 1,000 | 4.30 ± 1.56 | 0.8881 ± 0.0180 | 1,347 |
| **bootstrap** | **2,000** | **5.94 ± 0.11** | **0.8870 ± 0.0184** | 3,462 |
| returns | **200** | **5.13 ± 0.66** | **0.8851 ± 0.0362** | 639 |
| returns | 400 | 4.97 ± 1.12 | 0.8775 ± 0.0441 | 1,123 |
| returns | 1,000 | 5.01 ± 0.42 | 0.8984 ± 0.0131 | 2,283 |
| returns | 2,000 | 5.59 ± 0.28 | 0.8840 ± 0.0162 | 3,770 |

Reference arms, same slice: greedy 6.00 items / UAR 0.9021 / Brier 0.0571;
degenerate lowest-index 6.00 / 0.8838 / 0.0798.

**The 400-episode figure reported earlier was an undertrained network.** At 400
episodes the bootstrapped objective is still at 0.60 items; it needs ~2,000 to
reach 5.94. The benchmark default has been raised from 400 to 2,000 accordingly,
and `tests/test_step7_rl_diagnosis.py::test_benchmark_uses_the_diagnosed_episode_budget`
fails if Step 5 is ever run at a smaller budget again.

**Why the bootstrap target is slow.** §11.1 emits a reward only at the end of an
episode, so a `B`-step episode with immediate rewards needs a `B`-step bootstrap
chain. The value of the *root* action is then a product of `B` noisy estimates
and the argmax is decided by estimation noise. The `returns` arm stores
discounted return-to-go and disables the bootstrap term, removing the chain
entirely — and it reaches plateaued behaviour by **200 episodes**, an order of
magnitude sooner.

**Neither arm beats greedy** (UAR 0.887/0.884 vs 0.9021) **or the exact best
fixed subset** (Brier). So the diagnosis improves the *credibility* of the RL
numbers without changing the conclusion.

### A pre-committed criterion that was the wrong question

The stop rule written before the sweep was *"state-coverage growth in the second
half < 10% and items-mean flat across seeds"*, and it returned
`still_improving = True` for both arms — because coverage genuinely keeps rising
(+156% and +134% in the second half). But coverage is not a convergence test for
a *policy*: it kept growing long after behaviour plateaued (UAR delta of −0.0011
and −0.0144 between the last two budgets). Both metrics are now recorded, and
the behaviour delta is the one that answers "has it learned". Recorded rather
than quietly replaced.

**PPO gets worse with more training** (1.43 items, Brier 0.1162 at 2,000
episodes, against 0.0953 at 400). With λ=0 nothing penalises stopping early, so
extra training drives the STOP action's advantage upward. That is the same
missing-cost-pressure the λ sweep is built to address, and it is the strongest
argument yet for signing off V-6.

## A.6 The finding that reframes all of the above

At λ = 0, `V* = 1.000000` **exactly**, because the empirical support becomes
*pure* after ~4 questions (fraction of states with a pure support: 0.000 at depth
1 → 0.820 at depth 4 → 1.000 at depth 7). Once pure, `u_stop = 1 - (p_emp - y)^2`
is exactly 1.0 and no further question can raise the reward. The exact policy's
early stopping therefore tracks objective saturation with
`pure_support_frac_at_stop = 1.00`, not evidence sufficiency.

Full analysis, the λ sweep, and the proposed grid are in
**`V6_LAMBDA_DECISION.md`**, which is decision input for the V-6 sign-off and
does not itself constitute that sign-off.

### The P1-f arm across the same sweep

`beta_greedy` (Beta(1,1) posterior, EVOI selection, cost-based stopping) is
swept at every cell of the Step-6 grid. **Its λ is on the support-posterior EVOI
scale, not the predictor scale** — see `V6_LAMBDA_DECISION.md` §2.1 — so this
table is a record of the arm, not a comparison against the columns above.

| λ | exact items | exact reward | beta items | beta reward | greedy reward |
|---|---|---|---|---|---|
| 0.000 | 4.45 | 0.89764 | 6.00 | 0.94276 | 0.94079 |
| 0.001 | 3.37 | 0.93364 | 3.31 | 0.93748 | 0.93479 |
| 0.005 | 3.37 | 0.92016 | 2.13 | 0.90120 | 0.91079 |
| 0.010 | 3.37 | 0.90331 | 2.13 | 0.89053 | 0.88079 |
| 0.020 | 2.17 | 0.86478 | 2.09 | 0.87736 | 0.82079 |
| 0.050 | 2.17 | 0.79982 | 1.43 | 0.81058 | 0.64079 |
| 0.100 | 1.44 | 0.74049 | 1.43 | 0.73933 | 0.34079 |
| 0.200 | 0.00 | 0.77861 | 0.00 | 0.77861 | −0.25921 |

It tracks the exact reference's question count across the sweep and beats
`greedy` at every λ ≥ 0.005 — greedy structurally cannot stop, so it pays for all
six questions. At λ = 0 it spends the full 6.00 items by construction, since the
stop test `max_j gain_j < 0` cannot fire; that is what keeps fixed-length (AB-7)
separable from adaptive stopping. Pinned by
`tests/test_step6_lambda_sweep.py::test_beta_greedy_never_stops_early_at_zero_lambda`.

The degeneracy boundary agrees: at λ = 0.2 both the exact and the EVOI policy
decide to ask nothing.

## A.7 What this establishes, and what it does not

**Establishes:**
- The training and benchmarking pipeline works end to end and is **reproducible
  from its recorded seed**. Verified by rerunning all six producing scripts and
  diffing: every substantive number is identical, and the only fields that change
  are the recorded wall-clock ones (`generated`, `wall_seconds`, `time_sec`,
  `dp_seconds`). Five defects that made earlier measurements
  meaningless are fixed and regression-tested: the rewardless collector, the
  contaminated split, the mis-trained and unseeded predictor, the constant
  `stop_reason`, and the undertrained episode budget.
- Greedy-IG is near-optimal against an independently computed `V*` — gap ≤ 0.0133
  at every budget — and that survived all five corrections.
- **H1 is now testable and is not supported at B=5 and B=6.** The exact best fixed
  subset attains better held-out Brier, even though greedy is closer to `V*` on the
  empirical objective. This is the project's most important current result and it
  runs *against* the adaptive-value hypothesis.
- Adaptive stopping is expressible and the exact reference uses it; greedy
  structurally cannot, and at λ ≥ 0.005 the entire exact−greedy gap is
  acquisition cost.
- The DQN collapse was an undertrained network, not a property of RL. The
  bootstrapped objective needs ~2,000 episodes; return-to-go needs ~200.

**Does not establish:**
- Anything clinical (§16.1 circularity; the saturation in A.6 is a direct
  consequence of it).
- Anything about Polish — sealed behind V-4/V-7, and now the load-bearing
  experiment for A.6.
- Seed robustness for the *benchmark* table. The diagnosis sweep uses 5 seeds
  (A.5); the Step-5 benchmark table is still single-seed, so no gap difference in
  A.3 is interpretable as an effect.
- Anything cost-dependent. λ = 0 throughout; V-6 unsigned.
- That RL beats greedy. DQN now trains to a sane policy and still does not.

**Do not present the RL-vs-greedy comparison as a clinical finding under any circumstances.**
