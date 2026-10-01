# RL Training Report — Part 1 (2026-10-01)

**Purpose:** document the defects found in RL code that had never executed, the fixes applied, and the initial real training runs.
**Companion:** `diagnosisReady.md` §3 records the audit that prompted this work; `POLICY_BENCHMARK_REPORT.md` reports how the trained policies actually perform.
**Status:** Part 1 complete. **Part 2 (benchmarking) complete 2026-10-01** — and it found that greedy-IG attains the exact optimum while these policies do not.

---

## 1. What was wrong

`src/policies/dqn.py` and `src/policies/ppo.py` shipped with training code that had never been run. Nothing in the repository called `train_step`; no replay buffer existed; every actual run used Greedy-IG or Random despite `configs/config.yaml` declaring `policy.type: dqn`.

Two genuine defects, neither of which had been caught because no test executed this code:

| # | Location | Defect | Consequence |
|---|---|---|---|
| 1 | `DQNPolicy.train_step` | Legal-action mask built into a local `mask`, then discarded (`skip for brevity` in the source) | Bootstrap target was a max over **all** actions, including already-asked items and premature STOP — precisely the behaviour a budgeted questionnaire policy must not learn |
| 2 | `PPOPolicy` | **No training method existed.** Actor and critic constructed; critic never received a gradient | `PPOPolicy` was an untrained sampler, not a learner |

**Retracted claim.** An earlier draft of this report also listed a third defect: that terminal next-states were built as `torch.zeros(1, D)` and would make `torch.cat` raise on mixed batches. **That was wrong.** `torch.cat(..., dim=0)` concatenates *along* dimension 0, and every row contributes exactly one `(1, D)` tensor, so the result is always `(batch, D)`. Verified empirically. The rewrite does construct terminal rows uniformly (clearer, and it removed the duplicated `torch.cat`), but it did not fix a real defect. Two genuine defects, not three.

## 2. What was changed

### `src/policies/dqn.py`
`train_step` rewritten: the legal-action mask is now built as a `(batch, n_actions)` tensor and applied via `masked_fill` before the `argmax` that selects the Double-DQN action. This is the substantive fix. Terminal rows are zero-filled uniformly and the duplicated `torch.cat` (the code built the next-state tensor twice) is collapsed into one; the dead masking block is removed, with comments recording why it existed.

### `src/policies/ppo.py`
Added `train_step` (clipped surrogate + value loss + entropy bonus), `select_action` (sampling or greedy), and `legal_mask`. Illegal actions are masked to `-inf` before the softmax, so they receive zero probability. Advantage normalisation skips single-sample batches (unbiased std is undefined there, which was emitting a torch warning).

Batch format aligned with DQN: `(s, a, r, s_next, done, legal_next)`, matching `ReplayBuffer`.

### `src/policies/replay.py` (new)
`ReplayBuffer` with FIFO capacity, `add_step` / `end_episode` stitching, seeded `sample`. **Stores the per-transition legal-action set** — not optional bookkeeping, since the bootstrap max must be masked to the next state's legal set, and that set has to survive storage.

### `scripts/step4_train_policies.py` (new)
First entry point that trains a learned policy. Collects episodes against the existing §10 environment, fills a buffer, runs DQN/PPO updates, writes `results/step4_policy_training_<dataset>.json`.

### `tests/test_rl_training.py` (new, 13 tests)
Each defect gets a regression test, plus replay-buffer coverage.

---

## 3. Testing approach

Two of the tests assert on the **loss returned before the optimiser step**, which pins the regression target exactly. Asserting on post-step Q values would have been wrong: one Adam step moves Q by about `lr`, not to the target — an earlier draft of these tests made that mistake and failed for the right reason.

- **Terminal target has no bootstrap** — with `done=1`, `y` must equal `r` exactly. Verified via `loss == MSE(Q(s,a), 0.75)` to float32 tolerance. If a terminal row leaked into the bootstrap, `y` would include `gamma * max Q`.
- **Illegal actions excluded from bootstrap** — both nets are made constant functions of their last-layer bias, with a huge value placed on an *illegal* action. Correct (masked) target uses the best legal value; the unmasked target would use the illegal one. The two are distinguishable.
- **PPO critic receives gradient** — critic parameters must change after `train_step`. Directly tests defect 3.
- **PPO assigns zero probability to illegal actions** — deterministic and sampled selection never return an illegal action.

**Result at the time of Part 1: 67 passed** (54 pre-existing + 13 new), no warnings.
The suite is now **79** (12 Step-5 benchmark tests added later the same day — see
`POLICY_BENCHMARK_REPORT.md`).

### A correction made during this work
While debugging the buffer, I initially "fixed" `end_episode` on the belief that it stored the wrong legal set. That was my error, not a bug in the original logic — the test data was wrong. Reverted; the buffer's indexing was correct. Recorded here because the intermediate commit would otherwise look like an intentional design change.

---

## 4. Initial real training runs (Saudi 506, B=6, seed 0)

Trained on the Saudi test-relevant train split (303 records), predictor `MaskedMLP[128,64]` + isotonic.

| Algorithm | Metric | Initial | Final | Updates | Wall |
|---|---|---|---|---|---|
| DQN | loss (mean 50) | 0.006889 | 0.000235 | 1,556 | 4.9 s |
| PPO | value loss | 0.066314 | 5.9e-05 | 1,584 | 6.6 s |
| PPO | entropy | 2.1146 | 2.1191 | — | — |

Both losses decrease, so the training loop genuinely optimises. PPO entropy is flat rather than collapsing, which indicates no deterministic collapse onto one action.

**These numbers demonstrate the pipeline runs. They are not evidence the policies are good.** Loss on this task is largely a measure of how well a network fits the label rule.

---

## 5. Label circularity — mandatory caveat

Trained on the Saudi cohort, whose labels are a **deterministic sum-threshold over the questionnaire items themselves** (§16.1: `Qchat-10-Score == sum(A)`, verified 506/506).

A DQN or PPO policy can learn that rule perfectly. Doing so would tell us nothing about autism. The artifact therefore carries:

```json
"circularity_warning": "Trained against a deterministic sum-threshold oracle over the
 questionnaire items. Low loss here reflects learning the label rule, not autism
 screening skill. Not clinical evidence."
```

and a `tag` naming the open gates. Every future RL artifact must carry the same warning.

---

## 6. Part 2 outcome — the trained policies lose to greedy

**The benchmark has been run** (`scripts/step5_policy_benchmark.py`, 2026-10-01). DQN and PPO were evaluated at matched budgets B ∈ {1..6} against Greedy-IG, Random, and the exact DP, using the same `run_episode` evaluator and the same held-out split.

**Result: the trained policies lose to greedy.** Optimality gap V\* − V_emp at B=6 is **+0.007 for greedy, +0.039 for DQN, +0.138 for PPO**. PPO is worse than uniform-random at B=5 and B=6. Full analysis in `POLICY_BENCHMARK_REPORT.md`.

So the training pipeline works and the measurement is sound — but converging to a low loss did not translate into a better policy. That is a finding about this task, not a bug in the code.

Remaining gaps: single seed (no variance), λ=0 only (V-6 pending). Nothing here changes the pre-existing result that the sealed Polish cohort (252 rows, clinician diagnoses) is the single cohort able to support any clinical claim.

---

## 7. Reproduce

```bash
.venv/bin/python -m pytest tests -q                                    # 96 passed (was 79 collected / 67 passing before §2)
.venv/bin/python scripts/step4_train_policies.py --episodes 400 --budget 6 --seed 0
.venv/bin/python scripts/step5_policy_benchmark.py --episodes 400 --budgets 1,2,3,4,5,6
```

Writes `results/step4_policy_training_saudi.json` and `results/step5_policy_benchmark_saudi.{json,csv}`. Requires the Saudi CSV under `data/raw/`; `--synthetic` runs without data.

---

# §2 — SECOND AUDIT (2026-10-01, later the same day): the collector was also broken

Part 1 above fixed `DQNPolicy.train_step` and added `PPOPolicy.train_step`. It
missed a third, larger defect: **`collect_episode` in
`scripts/step4_train_policies.py` never delivered a learning signal at all.** The
Part 1 fixes were correct and necessary, but they were operating on data that
carried no reward.

## 2.1 Evidence (measured on the real Saudi cohort, before the fix)

```
buffer rewards unique: {0.0}      dones unique: {0.0}
s_next None?: [False, False, False, False, False, False]
PPO train_step on all-zero rewards:
    {'loss': -0.0132, 'policy_loss': -0.0, 'value_loss': 0.0214, 'entropy': 2.390}
```

`policy_loss` was exactly `-0.0`: `adv = rewards = 0` (`src/policies/ppo.py:102`),
so the surrogate gradient was identically zero and the entropy bonus was the
sole active term. DQN was fitting `y = γ·max_a Q(s',a)` against an all-zero
target, which converges by shrinking every Q-value toward 0 — which is why the
learned Q-values were structureless and `argmax` always resolved to the lowest
legal item index.

## 2.2 The four defects

| # | Defect | Location | Effect |
|---|---|---|---|
| **P0-1** | Terminal reward discarded. Every transition stored `r = 0.0`, including the terminal one. | `step4:109,127` | No reward anywhere; DQN and PPO both blind |
| **P0-2** | The stored legal set was the legal set of the state **left**, not the state **entered**. An item asked on the way into `s_next` was still marked legal. | `step4:109` | Re-opened the very defect `ReplayBuffer.end_episode` was written to fix in Part 1 |
| **P0-3** | `ReplayBuffer.add_step` stages into `_pending`; nothing ever called `end_episode`. | `step4:109,127` | Every STOP transition silently discarded; `_pending` grew unbounded (measured 19 after 40 episodes) |
| **P0-4** | `"policy_stop" if stop_legal else "budget_exhausted"` — with `b_min == 0`, `stop_legal` is unconditionally `True`. | `src/env/environment.py:56` | `"budget_exhausted"` was dead code; `stopped_early_frac` was 1.0 for every policy |

Verified off-by-one for P0-2:

```
transition 0: a = A10
  stored legal_next : [0,1,2,3,4,5,6,7,8,9,-1]   <- A10 still legal
  TRUE legal of s_next: [0,1,2,3,4,5,6,7,8,-1]
```

## 2.3 Two latent crashes the collector bug had been hiding

These only became reachable once terminal transitions were actually written.

- **`PPOPolicy.legal_mask(None)` raised `TypeError`.** Terminal rows carry
  `legal_next is None` by the shared batch contract, and `legal_mask` iterated it
  unconditionally. Every PPO batch contains terminal rows, so PPO could not have
  run at all. Fixed by treating `None` as "terminal, unconstrained", matching how
  `DQNPolicy.train_step` already skipped it.
- **The importance ratio evaluated `nan`.** `masked_log_probs` returned `-inf` for
  a legal action whose logit was far below the maximum (`exp` underflow to
  exactly 0), so `exp(logp - old_logp)` computed `-inf - -inf`. Fixed with a
  finite `LOG_FLOOR` and by flooring the probability before `log`.

## 2.4 PPO needed a different reward, not just a delivered one

§11.1 emits a reward at the terminal transition only. Storing the *immediate*
reward therefore leaves PPO's advantage `r_t` zero at every step except STOP,
which trains the policy to stop immediately — a worse outcome than the original
bug, produced by a correct fix. `collect_episode` now takes `gamma`:

- `gamma=None` (DQN) stores the **immediate** reward, so the Q-target bootstraps.
- `gamma=<float>` (PPO) stores the **discounted return-to-go**
  `G_t = r_t + γ·G_{t+1}`, `G_T = r_T`, via the new `_to_returns` helper.

Verified: for a 4-step episode with `R = 0.8236` and `γ = 0.9`, the stored
rewards are `R·γ³ … R·γ⁰ = 0.6013 … 0.8236`.

The duplicated legal-masking block that computed `old_logp` was replaced by
`PPOPolicy.masked_log_probs`, so both trainer copies derive the importance ratio
identically.

## 2.5 A console-encoding crash that suppressed the artifacts

Both `step4` and `step5` ended with `print(f"\n→ {dest...}")`. A Windows console
defaults to cp1252, which cannot encode `→`, so both scripts raised
`UnicodeEncodeError` on their **final** line — after the artifacts had been
written. The scripts exited non-zero and looked like they had failed. This is why
`results/step4_policy_training_saudi.json` and
`results/step5_policy_benchmark_saudi.{json,csv}` were absent from the checkout,
and why the 12 Step-5 contract tests were skipping. Both scripts now reconfigure
stdout/stderr to UTF-8 with `errors="replace"`.

## 2.6 Reproducibility (P0-7)

`DQNPolicy.select_action` used the module-level `random` generator for
epsilon-greedy, which `torch.manual_seed` / `np.random.seed` do not control. Two
runs at the same seed produced different rollouts, so no DQN number was
regenerable. The policy now owns a seeded `random.Random(seed)`, and both trainer
copies pass the run seed through.

## 2.7 Verification after the fixes

```
transitions per episode : 6 items + 1 terminal STOP = 7
rewards                 : [0, 0, 0, 0, 0, 0, 0.8236]
dones                   : [0, 0, 0, 0, 0, 0, 1]
pending left behind     : 0
legal_next excludes already-asked items on every transition: True
DQN loss                : 0.0489 -> 0.1965   (real target now)
PPO value loss          : 0.4170 -> 0.0447
PPO entropy             : 2.0898 -> 1.6783   (was pinned upward by the bug)
stop_reason             : "budget_exhausted" / "policy_stop" now both reachable
epsilon reproducibility  : seed 7 == seed 7, seed 7 != seed 8
```

## 2.8 Tests added

16 new tests. 12 in `tests/test_rl_training.py` (one per defect plus end-to-end
DQN/PPO updates), 2 in `tests/test_legal_actions.py`
(`test_stop_forced_when_no_legal` updated to the stronger guarantee,
`test_stop_reason_distinguishes_voluntary_from_exhausted` added), and 2 in
`tests/test_step5_benchmark.py` (`test_training_converged` rescoped,
`test_stopped_early_frac_is_not_a_constant` added).

Suite: **96 passed, 0 skipped** (was 67 passed / 12 skipped).

## 2.9 Consequence for the reported result

`POLICY_BENCHMARK_REPORT.md` §2 has been retracted. The correct current reading is
in that document's §A. In short: greedy-IG's near-optimality is unchanged and
reproduced, while both learned policies now collapse to a degenerate always-stop
policy — a training-budget and λ=0 artefact, not yet a result about the task.

---

# §3 — THIRD AUDIT (2026-10-01, later still): the split was contaminated

The §2 collector fix made the training signal real, and the benchmark then
produced numbers. Two further defects were then found, the second of which
invalidates every number produced before it.

## 3.1 P0-9 — the canonical split leaked (found by a test written for P1-d)

`src/data/splits.py` was created so that every artifact would land on one
partition. The first version of it reproduced a pattern that was already present
in **four** places — `scripts/step2_train_and_sweep.py`,
`scripts/step3_preliminary_reports.py`, `scripts/step5_policy_benchmark.py` and
`scripts/demo_app.py`:

```python
train_idx, test_idx = next(skf.split(idx, y))
train_idx, val_idx = next(skf.split(train_idx, y[train_idx]))   # <-- BUG
val = [records[i] for i in val_idx]                             # <-- BUG
```

`skf.split(train_idx, ...)` returns indices **relative to `train_idx`**, not to
the original record list. Passing them straight into `records[i]` selects the
wrong records.

Measured on the 506-row Saudi cohort:

| | |
|---|---|
| validation records | 95 |
| …also in the **training** set | **71** |
| …also in the **test** set | **24** |

So the calibrator was fitted on a set that was 75% training data — which the
network had already memorised — and 25% test records. The row counts came out at
exactly the documented 284/95/127, which is why this survived review: **the sizes
were right and the membership was wrong.** `split_fingerprint` was added for
precisely this reason; equal row counts prove nothing.

`scripts/step4_train_policies.py` had a second, independent split defect: a local
60/20/20 `train_test_split` giving 303/101/102, so the policies trained there saw
a different record set from every artifact they were benchmarked against. Spec §17
requires identical splits across runnable baselines.

Both are fixed. All five consumers now resolve to fingerprint
`b2021998a83e7224`, and
`tests/test_step6_lambda_sweep.py::test_every_consumer_agrees_on_one_partition`
fails if any of them drifts. `src/audits/leakage.py` could not have caught this:
`LeakageTracker.record_fit` is never called from any production code, so the
§16.2 "blocking" audit has been recording nothing about the real pipeline.

## 3.2 P1-b — `MaskedPredictor` v1 → v2

Three defects, all in code that had been running since the scaffold was written:

| Defect | Location (v1) | Consequence |
|---|---|---|
| **One masked view per record per `fit()`** | `masked_predictor.py:83-105` | 284 training states for a 41-dim input, each visited once per epoch |
| **`budget_norm` trained on the wrong grid** | `:95-103` | Training always set `budget = n_items = 10` (grid `1.0 … 0.0`); the environment runs at `budget = B` (grid `1.0, 0.833, 0.667, 0.5, 0.333, 0.167, 0.0`). Four of seven inference values were never seen in training |
| **Calibrator fitted on fully observed states** | `:130-141` | The calibrator only ever saw raw outputs from fully revealed records (`[0.001, 1.0]`) and had to extrapolate on every partial state it was asked to score |

v2 samples `(budget, depth, revealed subset)` from the inference grid, draws
`masks_per_record = 16` views per record, and fits the calibrator on states drawn
from that same grid.

| | v1 | v2 |
|---|---|---|
| training states / calibration points | 284 / 95 | 4,848 / 1,520 |
| test Brier | 0.0630 | **0.0512** |
| test UAR | 0.8513 | **0.8930** |
| test AUROC | 0.9627 | **0.9882** |
| test ECE | 0.0560 | **0.0475** |

Both measured on identical inference-grid states; v1 reproduced exactly via
`masks_per_record=1, eval_budgets=[n_items]`.

**A mistake worth recording.** The first attempt at v2 fitted the calibrator on
one state per validation record, giving 95 calibration points — too thin to
constrain a monotone map across the range partial states occupy. Isotonic then made
Brier *worse*: 0.0982 → 0.1254, versus 0.0780 → 0.0800 for v1. The "fix" was
initially measured as a regression, and the cause was the sample size, not the
idea. `fit_calibrator` now draws `masks_per_record` states per record.

**Residual, not resolved:** on a broad uniform sample of (budget, depth, subset)
states, isotonic still moves ECE the wrong way for v2 (0.0487 → 0.0729) while
improving it along the B=6 episode path. Calibration quality is
distribution-dependent and `isotonic` is a poor fit for partial states on
circular labels — which is why the browser demo uses `platt`.

## 3.3 Artefact versioning

Every artifact now records `predictor_version` and `split_fingerprint` alongside
`git_sha` and `seed`, so a pre-fix and post-fix number can never be silently
mixed. The demo's model cache filename also encodes the version
(`demo_model_saudi_seed0_platt_v2.pt`); it previously did not, so a change to
training semantics left a stale cache on disk and the demo kept serving old
weights.

`tests/test_step6_lambda_sweep.py::test_artifact_split_fingerprint_is_current`
reproduces the canonical fingerprint and fails if a stored artifact was generated
on a different partition — which is what caught the stale λ artifact immediately
after the split fix.

## 3.4 Current state

`tests/`: **126 passed, 0 skipped** (was 67 passed / 12 skipped before §2).
All of steps 2–6 regenerated on real data with `predictor_version: 2` and split
fingerprint `b2021998a83e7224`.

DQN still collapses (0.00 items at B=6, UAR exactly 0.5000) and PPO is no longer
degenerate (2.43 items, best UAR of any policy). Greedy-IG's near-optimality
survived all three corrections (gap ≤ 0.0133). See `POLICY_BENCHMARK_REPORT.md`
§A and `V6_LAMBDA_DECISION.md`.
