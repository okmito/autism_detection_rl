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

**Result: 67 passed** (54 pre-existing + 13 new), no warnings.

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
.venv/bin/python -m pytest tests -q                                    # 79 passed
.venv/bin/python scripts/step4_train_policies.py --episodes 400 --budget 6 --seed 0
.venv/bin/python scripts/step5_policy_benchmark.py --episodes 400 --budgets 1,2,3,4,5,6
```

Writes `results/step4_policy_training_saudi.json` and `results/step5_policy_benchmark_saudi.{json,csv}`. Requires the Saudi CSV under `data/raw/`; `--synthetic` runs without data.
