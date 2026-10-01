# Audit Report — Real Data (2026-09-04; updated 2026-10-01)

All audits use real-data loaders from `src/data/ingest.py` with `synthetic=False`. No Polish outcomes used for tuning. NZ primary cohort: source located 2026-09-04 (see `V1_NZ_DATASET_RESOLUTION.md`); content validated on a public mirror; **licence = "Unknown"** is the only remaining V-1 blocker (human action required).

> **Updated 2026-10-01.** All four rows below were **re-verified against freshly re-fetched data** and reproduce exactly. See §"2026-10-01 re-verification" for what changed and the provenance of the two derived data files.

## Circularity Audit — `src/audits/circularity.py:audit_circularity` (§16.1)

Procedure: sum-threshold oracle over item responses for thresholds 0..n, report exact_match_rate, UAR, sens/spec.

| Dataset | n | Rows | Label source | Best thr | Exact match | UAR | Classification | Interpretation |
|---|---|---|---|---|---|---|---|---|
| Saudi | 506 | 506 | questionnaire | 4 | 1.0000 | 1.0000 | **Deterministic** | `Class == (Screening Score >=4)` 506/506; `Screening Score == sum(A1:A10)` 506/506 — questionnaire-derived circular labels (§15) |
| UCI Child | 292 | 292 | questionnaire | 7 | 1.0000 | 1.0000 | **Deterministic** | `Class` deterministic from sum(A) threshold 7 — also circular; must not support claim of independent clinical screening validity (§16.1 gate: display circularity status next to every result) |
| Polish | 252 | 252 | clinical | 0 | 0.5357 | 0.5000 | **Not circular** | `group` (ASD/control) clinician-established; Sum_QCHAT not deterministic — material for transfer analysis (RQ3) |
| **NZ (toddler target, content-validated on Kaggle mirror — V-1 licence gate pending)** | 1,054 | 1,054 | questionnaire | **4** | **1.0000** | 1.0000 | **Deterministic** (predicted) | `Class/ASD Traits == (Qchat-10-Score >= 4)` for 1,054/1,054 rows (recomputed from public mirror; see `V1_NZ_DATASET_RESOLUTION.md` §4). When the real file is placed at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`, `audit_circularity(load_dataset('nz'))` is expected to return the same row. |

Commands:
```
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('saudi')))"
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('uci_child')))"
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('polish')))"
# When NZ file is present:
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('nz')))"
```

Gate behaviour (§16.1): circular datasets (Saudi, UCI Child, NZ toddler target) may be used for policy-optimization/algorithm-comparison experiments **with circularity status displayed**; not for independent clinical screening validity claims. Polish (clinical) is the only dataset that supports an independent validity discussion.

## Leakage Audit — `src/audits/leakage.py:LeakageTracker` (§16.2)

Instrument every fitting operation with fold provenance; assert no fitted object applied to held-out rows before model-selection closed.

- Clean split (train indices 0..299, held-out 300..505): violations = 0 → **PASS**
- Poisoned control (fitted on 0..505 including held-out): violations >0 → **PASS (audit correctly fails)**
- `poisoned_control_demo()` → True → **PASS**

Commands:
```
python3 -c "from src.audits.leakage import LeakageTracker; t=LeakageTracker(); t.record_fit('scaler_on_train',list(range(300))); print(t.check_no_leakage(list(range(300,506))))"  → []
python3 -c "from src.audits.leakage import LeakageTracker; t=LeakageTracker(); t.record_fit('poisoned_full',list(range(506))); print(t.check_no_leakage(list(range(300,506))))" → non-empty
```

## Invalid-Value Handling — Polish qchat4 `11.0`

- Row 60, child_id `bdbp0221`, col `qchat4recode`, value `11.0` — **logged explicitly** to `src/data/ingest.py:_POLISH_INVALID_LOG`, treated as `MISSING` (`encoded=NaN`, `missing_mask True`), UserWarning emitted, raw string preserved in `raw_qchat`. Not silently converted per instruction item 7.

## Polish Isolation

Polish cohort (252) has **not** been opened for:
- calibration fitting (calibrators fit only on Saudi/UCI train/val folds)
- predictor hyperparameter selection
- threshold tuning (τ=0.5 frozen per `configs/config.yaml:eval.operating_threshold`)
- λ selection

V-4 (MDE/comparison family) and V-7 (Polish denominator freeze) remain **PENDING** — no confirmatory Polish evaluation run. The Step 3 preliminary reports (`results/perf_vs_budget_saudi.*`, `results/faithfulness_saudi.json`, `results/subgroup_saudi.json`) are Saudi-only.

## Step 2 / Step 3 Audit Trail (2026-09-04)

| Stage | Test file | Invariants |
|---|---|---|
| DP tractability sweep | `tests/test_dp_tractability_sweep.py` (10 tests) | all runs `status=optimal`; n_states monotone in B and in N; n_evals ≥ n_states; time < 24h; n_states < 50M; budgets {3,4,5,6} all present; source labelled `real` or `synthetic` |
| Step 3 artifacts | `tests/test_step3_artifacts.py` (4 tests) | CSV has 13 rows; JSON metadata carries preliminary tag, dataset=saudi, source, circularity_status, config, git_sha, terminal reference; faithfulness B=6, τ=0.5, 10 SHAP features; subgroup has sex_/age_ keys; differences-logged note present |
| V-2 no-claim rule | `tests/test_v2_no_claim_rule.py` (3 tests) | no forbidden phrases in non-allowlisted markdown; PRISMA template + worksheet present; all 8 required sections present |

## RL Policy Audit Trail (2026-10-01, new)

**Precondition.** Until 2026-10-01 no RL policy had ever been trained or evaluated in this repository. RQ1/RQ2 were unevidenced; every reported comparison was Greedy-IG vs Random. See `RL_TRAINING_REPORT.md` and `diagnosisReady.md` §3.

| Stage | Test file | Invariants |
|---|---|---|
| RL training | `tests/test_rl_training.py` (13 tests) | DQN bootstrap max excludes illegal actions; terminal transitions contribute no bootstrap (`y == r`); PPO critic receives a gradient; PPO assigns zero probability to illegal actions; replay buffer stitches episodes and preserves per-transition legal sets; round-trip buffer → DQN update |
| Step 5 benchmark | `tests/test_step5_benchmark.py` (12 tests) | **no policy beats V\*** (gap ≥ −1e-6, so the exact reference is sound); random strictly sub-optimal at every B; greedy gap < 0.02 (reproduces §17 #7); `items_asked_mean ≤ B`; Brier/UAR in [0,1]; artifact carries the circularity warning; Polish recorded SEALED; DQN loss decreases over training; CSV/JSON agree |

**Known implementation gap (2026-10-01):** `DQNPolicy.select_action` uses unseeded `numpy.random` for ε-greedy exploration, so DQN training is not bit-reproducible across runs. Greedy/PPO/random gaps reproduce exactly; DQN's vary by ~0.01–0.02 in the optimality gap. This is the direct evidence for the multi-seed recommendation — treat single-seed DQN numbers as indicative only.

## Step 5 Result — negative for RL (2026-10-01)

Optimality gap V\* − V_emp (Saudi, train split, λ=0, lower is better). Full analysis in `POLICY_BENCHMARK_REPORT.md`.

| B | greedy | dqn | ppo | random |
|---|---|---|---|---|
| 1 | −0.000000 | +0.044531 | +0.040883 | +0.055131 |
| 3 | +0.001595 | +0.032676 | +0.035361 | +0.060355 |
| 5 | +0.012969 | +0.048338 | +0.105890 | +0.090981 |
| 6 | +0.006749 | +0.038949 | +0.138141 | +0.090097 |

**Greedy-IG attains the exactly-solved optimum; neither learned policy does.** PPO is worse than uniform-random at B=5 and B=6. This is reported as a negative result for RL on this task, not a defect in the pipeline.

§16.1 gate applies to every one of these numbers: the labels are a deterministic sum-threshold over the items, so this measures fit to the questionnaire-defined construct and **is not clinical evidence**.

## 2026-10-01 re-verification

All four circularity rows re-verified after the datasets were re-fetched from scratch (the repo had lost `data/raw/` entirely — it is gitignored).

| Dataset | Re-verified | Result |
|---|---|---|
| Saudi 506 | ✅ | 506 rows, 341/165 labels, circularity Deterministic at thr 4, `Screening Score == sum(A)` 506/506 |
| UCI Child 292 | ✅ | 292 rows, 151/141 labels, 90 `?` markers preserved |
| Polish 252 | ✅ | 252 rows, 135 ASD / 117 control, `label_source=clinical`, Not circular, invalid `qchat4=11.0` at row 60 / `bdbp0221` |
| NZ 1,054 | ⛔ | still absent — V-1 licence gate unchanged, no file downloaded |

State counts recomputed from the restored Polish file: observed **2,667,729,775** and canonical **3,081,146,397** — both match `STATE_COUNT_VERIFICATION.md` exactly.

### ⚠️ Two data files are derived conversions, not originals
- **`data/raw/UCI/Autism-Child-Data.arff`** — UCI no longer serves an ARFF for dataset 419 (only `data.csv`). Generated from that official CSV: same 292 rows, same 21 columns, `NaN`→`?`; 90 missing markers (43 `ethnicity`, 43 `relation`, 4 `age`) match the documented count.
- **`data/raw/Q-CHAT Polish/polish_qchat.csv`** — generated from the original Mendeley Data `tmpkt2mfkg` SPSS `.sav` (sha256 `7fed516f…` verified against Mendeley's published hash), using the file's **own SPSS value labels** for `group` (1=ASD, 7=control) and `sex` (1=Male, 2=Female).

Saudi is a genuine original CSV. Full provenance: `AGENT_PROGRESS.md` §Data provenance.

## Test suite status

**79 passed** as of 2026-10-01 (54 pre-existing + 13 RL training + 12 Step-5 benchmark). All artifact-contract tests require their artifacts to exist; regenerating `results/` (gitignored) is required after a fresh clone or the DP-sweep and Step-3 tests skip.

> **Superseded later the same day — now 96 passed, 0 skipped.** "79" was the
> *collected* count, not the passing count: 12 of those tests skip when
> `results/step5_policy_benchmark_saudi.json` is absent, and that artifact was
> missing because `step5_policy_benchmark.py` crashed on its final `print` (P0-8
> below). 16 tests were added with the second audit. See the section below.


---

## 2026-10-01 (later) — Second RL audit: the rollout collector delivered no reward

`diagnosisReady.md §3` recorded that no RL policy had ever been trained, and
`RL_TRAINING_REPORT.md §1` recorded the Part 1 fixes to `train_step`. Neither
covered `collect_episode`, which is where the actual defect was. Full write-up in
`RL_TRAINING_REPORT.md §2`; the affected conclusion is retracted in
`POLICY_BENCHMARK_REPORT.md` and the corrected numbers are in its §A.

### Evidence (real Saudi cohort, before the fix)

```
buffer rewards unique: {0.0}      dones unique: {0.0}      terminal rows: 0
PPO train_step -> {'policy_loss': -0.0, 'value_loss': 0.0214, 'entropy': 2.390}
```

| # | Defect | Location | Consequence |
|---|---|---|---|
| P0-1 | Terminal reward never written; every row stored `r = 0.0` | `step4_train_policies.py:109,127` | DQN fitted `y = γ·max Q(s',a)` against an all-zero target; PPO advantage identically zero |
| P0-2 | Stored legal set belonged to the state **left**, not **entered** | `step4_train_policies.py:109` | Re-opened the masking defect `ReplayBuffer.end_episode` was written to close |
| P0-3 | `ReplayBuffer.end_episode` never called; terminal rows staged in `_pending` | `step4_train_policies.py:109,127` | Every STOP transition discarded; `_pending` leaked (measured 19 after 40 episodes) |
| P0-4 | `"policy_stop" if stop_legal else "budget_exhausted"` with `b_min == 0` | `src/env/environment.py:56` | `"budget_exhausted"` unreachable ⇒ `stopped_early_frac` ≡ 1.0 for every policy |
| P0-5 | `PPOPolicy.legal_mask(None)` raised `TypeError` on terminal rows | `src/policies/ppo.py:52` | PPO could not consume any batch with a terminal row; latent, hidden by P0-3 |
| P0-6 | `masked_log_probs` returned `-inf`, so the ratio computed `-inf − -inf` | new helper | `nan` policy loss |
| P0-7 | epsilon-greedy used the unseeded module-level `random` | `src/policies/dqn.py:42-44` | No DQN number was regenerable from its seed |
| P0-8 | Both step scripts ended with `print("→ …")`; a cp1252 console cannot encode `→` | `step4:349`, `step5:327` | Scripts exited non-zero **after** writing artifacts. This is why `results/step4_*` and `results/step5_*` were absent and 12 Step-5 contract tests skipped. Same class hit `step2` (λ) and `step3` (λ) mid-run. |

### All fixed, with regression tests

16 new tests. Suite is now **96 passed, 0 skipped** (was 67 passed / 12 skipped).
The "79 passed" figure recorded above was the *collected* count, not the passing
count, and is corrected in `README.md` and `AGENT_PROGRESS.md`.

`tests/test_step5_benchmark.py::test_training_converged` was rescoped: it required
`loss_final_50 < loss_first_50`, which was satisfiable only while the loop was
rewardless. It now asserts the presence of a learning signal, and the actual
delivery of the terminal reward is pinned directly in
`tests/test_rl_training.py::test_collect_episode_stores_the_terminal_reward`.
`test_stopped_early_frac_is_not_a_constant` was added to stop P0-4 regressing.

### What survives correction

Greedy-IG's optimality gap is **bit-identical** before and after the fix
(≤ 0.013 at every budget B ∈ {1..6}), because greedy and the exact reference
never read the replay buffer. The near-optimality of one-step information gain on
this instrument stands. What does not survive is any claim about the learned
policies: with a real reward they now collapse to a degenerate always-stop policy
(1.00 items at every B ≥ 2; DQN UAR exactly 0.5000), which is a training-budget
and λ=0 artefact rather than a result.

### Circularity and leakage status unchanged

No new circularity or leakage finding. All four rows above still hold, and Polish
remains sealed. Every RL number in `results/` continues to be measured against
the deterministic sum-threshold oracle, so none of it is clinical evidence.

---

## 2026-10-01 (third pass) — P0-9: the train/val/test split was contaminated

Found by a test written for an unrelated phase. **This invalidates every number
produced before it**, including the corrected figures in
`POLICY_BENCHMARK_REPORT.md §A` before the predictor work.

### The defect

Four copies of the split (`step2_train_and_sweep.py`, `step3_preliminary_reports.py`,
`step5_policy_benchmark.py`, `demo_app.py`) shared this pattern:

```python
train_idx, test_idx = next(skf.split(idx, y))
train_idx, val_idx = next(skf.split(train_idx, y[train_idx]))   # BUG
val = [records[i] for i in val_idx]                             # BUG
```

`skf.split(train_idx, ...)` returns indices **relative to `train_idx`**. Passing
them straight into `records[i]` selects the wrong records.

### Measured impact — Saudi 506

| | |
|---|---|
| validation records | 95 |
| …also in the **training** set | **71** |
| …also in the **test** set | **24** |

The calibrator was fitted on a set that was 75% training data (already memorised
by the network) and 25% test records. `src/audits/leakage.py` could not catch
this: `LeakageTracker.record_fit` is never called from any production code, so the
§16.2 "blocking" audit records nothing about the real pipeline.

**Why it survived review:** the row counts were exactly the documented
284/95/127. The sizes were right; the membership was wrong. `split_fingerprint`
was added to detect precisely this class of error, since equal counts prove
nothing.

### Second, independent split defect

`scripts/step4_train_policies.py` used a local 60/20/20 `train_test_split`
(303/101/102), so the policies trained there saw a different record set from every
artifact they were benchmarked against. Spec §17 requires identical splits.

### Fix

New `src/data/splits.py` with the canonical scheme, a corrected inner-fold
index remap, and `split_fingerprint`. All five consumers delegate to it and now
resolve to fingerprint `b2021998a83e7224`. Guarded by
`tests/test_step6_lambda_sweep.py::test_every_consumer_agrees_on_one_partition`
and `::test_artifact_split_fingerprint_is_current`.

### Also in this pass

**P1-b — `MaskedPredictor` v1 → v2.** Three defects: one masked training view per
record (284 states for a 41-dim input); the `budget_norm` feature trained on the
`budget=10` grid while inference runs at `budget=B`, so four of seven inference
values were never seen in training; and a calibrator fitted on fully observed
states that then had to extrapolate on every partial state. Test Brier
0.0630 → 0.0512, UAR 0.8513 → 0.8930, AUROC 0.9627 → 0.9882, ECE 0.0560 →
0.0475. Full detail, including a regression that was initially measured and then
diagnosed as a calibration sample-size problem, in `RL_TRAINING_REPORT.md §3.2`.

**New artifact:** `results/lambda_sweep_saudi.{json,csv}` from
`scripts/step6_lambda_sweep.py` — the V-6 evidence pack. Analysis and the
proposed grid in `V6_LAMBDA_DECISION.md`. **This does not sign off V-6.**

### The finding that reframes the results

At λ = 0 the empirical support becomes *pure* after ~4 questions
(0.000 of states at depth 1 → 0.820 at depth 4 → 1.000 at depth 7), so
`u_stop = 1 - (p_emp - y)^2` is exactly 1.0, `V*` is exactly 1.000000, and no
further question can raise the reward. The exact policy's
`pure_support_frac_at_stop` is **1.00**: it stops early precisely when the
objective has saturated, not when the evidence is sufficient. The measured value
of adaptive stopping at λ = 0 is therefore a property of the label rule.

This makes the Polish transfer test (RQ3/H2) the load-bearing experiment rather
than a secondary one, and it is why V-6 is proposed at λ ≥ 0.005.

### State

`tests/`: **126 passed, 0 skipped** (was 67 passed / 12 skipped).
Steps 2–6 regenerated on real data, every artifact carrying
`predictor_version: 2` and `split_fingerprint: b2021998a83e7224`.

### Circularity and leakage status

No new circularity finding. All four dataset rows still hold and Polish remains
sealed. The leakage audit is, if anything, in a worse position than reported
above: it did not detect a real train/test contamination, and
`record_fit` is still never called from production code (now tracked as an open
item in `AGENT_PROGRESS.md`).

---

## 2026-10-01 (fourth pass) — P0-10: the predictor was not reproducible, and H1 finally became testable

### P0-10 — model initialisation was unseeded

Found by accident: while adding a third baseline arm to Step 5, the greedy Brier
moved between two runs of the *same* command (`0.0490` → `0.0532` → `0.0518`).
Greedy-IG depends on no training whatsoever, so something upstream was
nondeterministic.

`MaskedMLP` builds `nn.Linear` layers, which draw from torch's **global** RNG.
`MaskedPredictor.__init__` constructed them, and `fit()` called
`torch.manual_seed(...)` *afterwards* — too late. `step3` and `step5` never seeded
torch in `main()` at all. So the initial weights were drawn from an unseeded
stream and **every run produced a different network, regardless of `--seed`**.

| Script | construct | seed in `main()` | affected |
|---|---|---|---|
| `step2_train_and_sweep.py` | `MaskedPredictor(..., seed=seed)` | via constructor | fixed |
| `step3_preliminary_reports.py` | `..., seed=0` | none | fixed |
| `step4_train_policies.py` | `..., seed=args.seed` | before construct | was OK |
| `step5_policy_benchmark.py` | `..., seed=args.seed` | **none** | fixed |
| `demo_app.py` | `..., seed=SEED` | none | fixed |

The fix is in the predictor rather than in each script: `MaskedPredictor.__init__`
now takes `seed` (default 0), seeds torch only around the layer construction, and
**restores the prior global RNG state** afterwards — so the predictor is
reproducible without perturbing the caller's stream. `seed=None` opts out and
honours the caller's state.

Verified reproducible from the recorded seed after the fix:

```
step5_policy_benchmark_saudi.csv  run1 == run2   SHA 397C6A53FA859FDA...
predictor_saudi_metrics.json      run1 == run2
```

Guarded by `tests/test_p1d_baselines.py::test_predictor_construction_is_seed_deterministic`
and `::test_predictor_construction_does_not_disturb_global_torch_rng`.

**Consequence: every artifact produced before this fix was non-reproducible
regardless of its recorded seed.** That includes the step4 and step5 artifacts
generated earlier the same day. They have all been regenerated.

### P1-d — the §17 baselines are now wired, so H1 is finally testable

`ExactFixedSubsetPolicy` (§17 #4), `StaticRFEPolicy` (#5) and `IRTCATPolicy` (#8)
existed but were instantiated **nowhere** in `scripts/`, `src/` or `tests/`. H1 is
written as "adaptive beats the exact best fixed subset at matched budget", so H1
had no runnable comparator at all. All three are now arms in Step 5.

Defects fixed in each while wiring them:

| File | Defect | Status |
|---|---|---|
| `static_rfe.py:28-30` | bare `except Exception:` fell back to index order with no record, so a degraded selection was indistinguishable from a converged one | now catches specific exceptions, warns, and sets `fallback_used` |
| `static_rfe.py` | missingness silently coerced to a real `0`, differing from the exhaustive baseline's NA-grouping, with the divergence undocumented | `missing_mode` recorded on both objects and asserted to differ |
| `static_fixed.py:28` | `lambda_cost` added to the subset comparison, where it silently did nothing — every candidate has size B so a uniform cost is a constant offset | no longer used for ranking; still applied to `best_value`; asserted to leave `best_subset` unchanged |
| `static_fixed.py:31` | `budget > n_items` fell back silently | now warns and sets `fallback_used` |
| `irt_cat.py:27-28` | difficulty `b <= 0` for **10/10** items on Saudi, putting every `p` at 0.92–0.97 and flattening Fisher information, so selection degenerated to the `a^2` proxy | latent trait estimated first, then 1-D ridge logistic per item; `b` now spans [-0.83, +0.06] and Fisher spread rose from ~0.05 to 0.44–1.56 |
| `irt_cat.py:20-22` | a wholly-unobserved column kept `a=1.0, b=0.0`, i.e. `I(0) = 0.25` from no data. **Latent, not active** — Saudi has 0/5060 missing cells — but it would bite on any cohort with an unobserved item | uninformative items get `a = 0`, so their Fisher information is exactly 0 |
| `irt_cat.py` | item direction undocumented: `1` is an *atypical* response, so the latent is symptom propensity, not ability | stated in the module docstring |

`IRTCATPolicy.information(state, legal)` was added so a benchmark or explanation
layer can report *why* an item was chosen, not only which.

### The H1 result

| B | greedy Brier | exact best fixed subset | H1 |
|---|---|---|---|
| 3 | **0.0876** | 0.1034 | supported |
| 4 | **0.0691** | 0.0850 | supported |
| 5 | 0.0655 | **0.0448** | **not supported** |
| 6 | 0.0512 | **0.0479** | **not supported** |

The adaptive advantage **shrinks and reverses as the budget grows**, even though
greedy remains strictly closer to `V*` on the empirical objective at every budget
(+0.0042 vs +0.0141 at B=6). The advantage lives in the training support; the
per-episode information-gain ordering is the part that does not transfer.

Spec §4 pre-registers that *"failure to outperform the exact best fixed subset or
generic CAT is a valid result"*, so this is reportable. It could not have been
obtained before P1-d.

`exact_fixed_subset` and `static_rfe` select **identical** subsets at every
budget on this cohort — a consistency signal between a cheap and an exhaustive
selector, now asserted in the test suite.

### State

`tests/`: **150 passed, 0 skipped** (was 126). 23 new tests in
`tests/test_p1d_baselines.py`. Steps 2–6 regenerated; every artifact now
reproducible from its recorded seed (recorded wall-clock fields excepted).

### Circularity and leakage status

Unchanged. Polish remains sealed. The saturation finding
(`V* = 1.000000`, `pure_support_frac_at_stop = 1.00`) is unaffected by the
predictor and the new baselines — it is a property of the label rule. See
`V6_LAMBDA_DECISION.md`.

---

## 2026-10-01 (fifth pass) — P1-e: the DQN collapse was an undertrained network

`scripts/step7_rl_diagnosis.py` → `results/rl_diagnosis_saudi.{json,csv}`.
Five seeds × {200, 400, 1000, 2000} episodes, 60 held-out records, trained
predictor. Two arms differing **only** in the learning target:

| objective | episodes | items (mean ± sd) | UAR (mean ± sd) | distinct states |
|---|---|---|---|---|
| bootstrap (spec §11.2) | 200 | 0.00 ± 0.00 | 0.5000 ± 0.0000 | 167 |
| bootstrap | 400 | 0.60 ± 0.55 | 0.6615 ± 0.1485 | 352 |
| bootstrap | 1,000 | 4.30 ± 1.56 | 0.8881 ± 0.0180 | 1,347 |
| **bootstrap** | **2,000** | **5.94 ± 0.11** | **0.8870 ± 0.0184** | 3,462 |
| **returns** | **200** | **5.13 ± 0.66** | **0.8851 ± 0.0362** | 639 |
| returns | 400 | 4.97 ± 1.12 | 0.8775 ± 0.0441 | 1,123 |
| returns | 1,000 | 5.01 ± 0.42 | 0.8984 ± 0.0131 | 2,283 |
| returns | 2,000 | 5.59 ± 0.28 | 0.8840 ± 0.0162 | 3,770 |

Reference: greedy 6.00 items / UAR 0.9021 / Brier 0.0571.

### What this establishes

**The 400-episode DQN figure was an undertrained network, not a property of
reinforcement learning.** At 400 episodes the bootstrapped objective is still at
0.60 items; it needs ~2,000 to reach 5.94. The Step-4/Step-5 default has been
raised from 400 to 2,000 episodes, and
`tests/test_step7_rl_diagnosis.py::test_benchmark_uses_the_diagnosed_episode_budget`
fails if Step 5 is ever run at a smaller budget again.

**Why the bootstrap target is slow.** §11.1 emits a reward only at the end of an
episode, so a `B`-step episode with immediate rewards needs a `B`-step bootstrap
chain; the root action's value is then a product of `B` noisy estimates and the
argmax is decided by estimation noise. The `returns` arm stores discounted
return-to-go and disables the bootstrap term (`DQNPolicy.train_step(bootstrap=False)`),
removing the chain, and reaches plateaued behaviour by **200 episodes** — an order
of magnitude sooner.

**And what it does not establish.** Neither arm beats greedy (UAR 0.887 / 0.884
vs 0.9021) or the exact best fixed subset on Brier. The diagnosis makes the RL
numbers *credible*; it does not make them favourable.

**PPO degrades with more training** (1.43 items, Brier 0.1162 at 2,000 episodes,
against 0.0953 at 400). With λ=0 nothing penalises stopping early, so extra
training drives the STOP action's advantage upward. This is the strongest
argument yet for signing off V-6, and it is a cost of the *missing* gate rather
than of the algorithm.

### A pre-committed criterion that asked the wrong question

The stop rule written before the sweep was *"state-coverage growth in the second
half < 10% and items-mean flat across seeds"*. It returned
`still_improving = True` for both arms — because coverage genuinely keeps rising
(+155.7% and +133.9%). But coverage is not a convergence test for a *policy*: it
kept growing long after behaviour plateaued (UAR delta of −0.0011 and −0.0144
across the last two budgets), and `items_mean` was not flat (5.94 ± 0.11 and
5.59 ± 0.28). Both metrics are now recorded in the artifact, and the
mis-specification is documented rather than quietly replaced.

### Corrected Step-5 numbers (2,000 episodes, B=6, held-out test split)

| policy | Brier | UAR | items | train gap |
|---|---|---|---|---|
| exact_fixed_subset | **0.0479** | 0.9167 | 6.00 | +0.014085 |
| greedy | 0.0512 | 0.8930 | 6.00 | **+0.004225** |
| dqn | 0.0640 | **0.9112** | 5.98 | +0.034918 |
| exact | 0.0644 | 0.8812 | 4.45 | 0 by definition |
| irt_cat | 0.0686 | 0.8993 | 6.00 | +0.026892 |
| random | 0.0918 | 0.8220 | 4.02 | +0.091082 |
| ppo | 0.1162 | 0.8398 | 1.43 | +0.137615 |

DQN's train gap fell from +0.167935 (400 episodes) to +0.034918. It is now a
trained policy, and it still does not beat greedy on the empirical objective.

### State

`tests/`: **163 passed, 0 skipped** (was 150). 13 new tests in
`tests/test_step7_rl_diagnosis.py`, including a small-scale reproduction of the
central claim so it is not taken on trust.

### Circularity and leakage status

Unchanged. Polish remains sealed. The saturation finding is unaffected by any of
this — it is a property of the label rule, not of the training. See
`V6_LAMBDA_DECISION.md`.

---

## 2026-10-01 (sixth pass) — P1-f: a Beta-prior / EVOI arm, and what it revealed

New file: `src/policies/beta_greedy.py`. **`GreedyIGPolicy` was deliberately left
untouched** — its near-optimality result is the project's strongest empirical
claim, and the new policy is an additional arm rather than a replacement.

### The degeneracy it fixes, measured

`GreedyIGPolicy` selects by `IG = H(Y|s) − E_v[H(Y|s,j,v)]` on the **raw**
empirical support mean. On a pure support `H(Y|s) = 0`, every child support is
pure too, so **every legal item's IG is exactly 0** and `argmax` falls through to
`greedy.py`'s `best_j = items_only[0]` initialisation. The "policy" becomes "ask
the lowest legal index".

Measured on the held-out split at B=6:

```
decision states sampled        : 480
  with a pure support          : 195 (40.6% of those reached on a greedy path;
                                61.3% of all decision states)
  greedy criterion identically zero on those : 195/195
  beta_greedy EVOI identically zero on those: 8/195
  median EVOI spread across legal items there: 0.000984
```

The 8 remaining vacuous states are ones where an item is unobserved for every
consistent record, so asking it genuinely reveals nothing.

### The two changes

1. **Beta(1,1) posterior** — `p = (n_pos + 1)/(n + 2)`. Interior on a pure
   support, so entropy is non-zero and the criterion is graded again. It also
   does the statistically right thing for free: a *small* pure support is weak
   evidence and keeps more entropy (n=1 → 0.918 bits, n=100 → 0.080 bits).
2. **EVOI stopping** — selection uses the expected improvement in the *actual*
   terminal utility rather than an entropy proxy:
   `u(s) = E_y[1−(p_s−y)²]`, `gain_j = u(s,j) − u(s)`, and `STOP` is returned
   when `max_j gain_j < λ·c_j`. At λ=0 the condition is never met, so the policy
   never stops early — which is what makes AB-7 (fixed-length vs adaptive
   stopping) separable rather than confounded with item selection.

`explain()` returns the posterior, a 95% credible interval, the per-item EVOI, the
entropy IG for comparison against greedy, the runner-up margin, and the stop
margin. Every field is a number the policy actually used. This is the P3-1
explainability hook designed in rather than retrofitted.

### Results (B=6, held-out test split, 127 records)

| λ | exact items | beta_greedy items | bg early-stop | exact reward | bg reward | greedy reward |
|---|---|---|---|---|---|---|
| 0.000 | 4.45 | 6.00 | 0.00 | 0.89764 | **0.94276** | 0.94079 |
| 0.001 | 3.37 | 3.31 | 0.92 | 0.93364 | **0.93748** | 0.93479 |
| 0.005 | 3.37 | 2.13 | 0.95 | **0.92016** | 0.90120 | 0.91079 |
| 0.010 | 3.37 | 2.13 | 0.95 | **0.90331** | 0.89053 | 0.88079 |
| 0.020 | 2.17 | 2.09 | 0.96 | 0.86478 | **0.87736** | 0.82079 |
| 0.050 | 2.17 | 1.43 | 1.00 | 0.79982 | **0.81058** | 0.64079 |
| 0.100 | 1.44 | 1.43 | 1.00 | 0.74049 | 0.73933 | 0.34079 |

`beta_greedy` tracks the exact reference's question count across the whole sweep,
beats it on held-out reward at 4 of 6 λ values, and beats `greedy` at every
λ ≥ 0.005 — because greedy structurally cannot stop and therefore pays for all
six questions.

At λ=0 the two are **bit-identical on the empirical train objective** (both
+0.004225 at B=6) yet `beta_greedy` generalises better on held-out Brier (0.0482
vs 0.0512). That is a direct measurement of the saturation problem: on a
saturated support the empirical objective is constant and cannot distinguish the
policies, while the held-out data can.

### A units finding that belongs in the V-6 pack

**One reward, two scales.** The λ grid proposed in `V6_LAMBDA_DECISION.md` is
justified against a marginal utility of 0.05–0.15 for the neural-predictor
reward. The support-posterior EVOI that `beta_greedy` uses has a median max-EVOI
of **0.0015 at depth 1** and **0.108 at depth 0** — roughly two orders of
magnitude smaller, and **negative** at depths 3–5 (asking more genuinely cannot
help a saturated support).

So a λ grid **does not transfer** between the two reward definitions without
recalibration. `beta_greedy`'s own useful range is 0.001–0.01 and should be swept
separately. Recorded as `lambda_units_caveat` in the Step-6 artifact.

### H1 still fails, now for three independent adaptive arms

| B | exact best fixed subset | beta_greedy | greedy | dqn | H1 |
|---|---|---|---|---|---|
| 3 | 0.1034 | **0.0876** | **0.0876** | **0.0864** | supported |
| 4 | 0.0850 | **0.0735** | **0.0691** | **0.0748** | supported |
| 5 | **0.0448** | 0.0709 | 0.0655 | 0.0660 | **not supported** |
| 6 | **0.0479** | 0.0482 | 0.0512 | 0.0640 | **not supported** |

The reversal is not an artefact of the greedy heuristic. Adding a principled
non-degenerate criterion did not rescue it.

### State

`tests/`: **187 passed, 0 skipped** (was 163). 24 new tests in
`tests/test_p1f_beta_greedy.py`, including a real-data regression that asserts
greedy's criterion is vacuous on all pure-support states while the new arm's is
not, and a test that `GreedyIGPolicy` still cannot select STOP.

### Circularity and leakage status

Unchanged. Polish remains sealed. The saturation finding is a property of the
label rule and is not affected by any of this.

### RETRACTED: the "one reward, two scales" claim, and the unsourced 0.05-0.15 figure

The sixth pass recorded that the V-6 λ grid "does not transfer between" the
neural-predictor reward and the support-posterior EVOI, because the latter's scale
is "roughly two orders of magnitude smaller". **That claim is false and is
withdrawn.** It was also resting on a number no code had ever produced.

`scripts/step8_evoi_scale_analysis.py` (new, P1-h) measured both quantities on the
same states, with the same support labels and the same functional form
`mean_y[1-(p-y)^2]`, substituting only `p`:

| decision statistic `max_j gain_j` (test, 762 states) | support posterior | neural predictor | ratio |
|---|---|---|---|
| min | −0.038704 | 0.000000 | |
| median | +0.001507 | +0.003529 | **0.427** |
| mean | +0.033739 | +0.039972 | **0.844** |
| p75 | +0.105179 | +0.103396 | **1.017** |
| p90 / p95 | +0.108152 | +0.108065 | **1.001** |
| max | +0.187500 | +0.321148 | 0.584 |

Both are expected Brier improvement — the same units as the reward term — with
ceiling `max_p[1-p(1-p)] = 0.25`. The mirrored implementation reproduces
`BetaGreedyPolicy.scores()` to **0.0** over 1,800 item-states, so the comparison is
like-for-like. The error that produced the false claim: comparing the
support-EVOI median **at depth 1** (0.0015) against a quoted predictor figure of
0.05–0.15. Measured, the predictor's own median is 0.0035 and its own depth-1
median is 0.0022 — the same order — and 0.05–0.15 is roughly its **p75–p95**
band, not a typical value.

**What actually makes V-6 hard, in place of the unit story.** On a *pure* support
the support-posterior EVOI is analytically non-positive:

```
gain_j = 1/(n+2)^2 − Σ_v w_v · 1/(n_v+2)^2  ≤ 0
```

because each child is pure and smaller. Verified: n=10→5/5 → −0.013464;
n=65→30/35 → −0.000621. The only reason the statistic is non-zero on a pure
support is the Beta(1,1) pseudo-count being diluted as the support shrinks — a
support-**size** artifact, not information about Y. Consequently the statistic is
**bimodal**, and the modes are the two support regimes: **44.9%** of decision
states at ≤ 0 (support pure) and **41.5%** at ≥ 0.01 (support still mixed).
Measured median by depth: +0.1082, +0.0015, +0.0212, −0.0003, −0.0004,
−0.0006.

So the threshold is in practice a **support-purity threshold**, and it does not
behave like a smoothly tunable cost parameter: thresholds 1e-05 through 0.001
produce **bit-identical** episodes, and the whole grid spans only **20 distinct
behaviours**, with the real transition a cliff between 0.001 and 0.003.

Corrected in: `V6_LAMBDA_DECISION.md` §2/§2.1, the `lambda_units_caveat` field of
`results/lambda_sweep_saudi.json`, the `step6_lambda_sweep.py` docstring, and
`AGENT_PROGRESS.md`. Pinned by
`tests/test_step6_lambda_sweep.py::test_lambda_units_caveat_is_declared` and
`::test_unsourced_marginal_utility_claim_is_not_asserted_in_the_script`, which fail
if the retracted framing is reinstated as an assertion.

**V-6 remains open.** The decision note is `V6_STOPPING_THRESHOLD_DECISION.md`; it
selects no threshold and recommends running the discriminating experiment (repeat
on Polish, V-4/V-7) before any threshold is signed.

### A correction found while verifying the reproducibility claim

The prior passes asserted "every artifact is byte-reproducible". Checking that
claim properly — snapshot `results/`, rerun all six producing scripts, diff —
showed it was **too strong**. After a full rerun:

| artifact | verdict |
|---|---|
| `predictor_saudi_metrics.json`, `perf_vs_budget_saudi.json` | byte-identical |
| `subgroup_saudi.json` | byte-identical |
| `rl_diagnosis_saudi.{json,csv}` | byte-identical apart from `generated` |
| `step5_policy_benchmark_saudi.{json,csv}` | byte-identical apart from `generated` |
| `step4_policy_training_saudi.json` | byte-identical apart from `generated`, `wall_seconds` |
| `lambda_sweep_saudi.{json,csv}` | all 64 rows identical apart from `dp_seconds` |
| `dp_tractability_sweep.{json,csv}` | all 48 rows identical apart from `time_sec` |

So the guarantee is real but is: **reproducible from the recorded seed, except
for recorded wall-clock fields** (`generated`, `wall_seconds`, `time_sec`,
`dp_seconds`). Every substantive number is identical. The wording has been
corrected in `README.md`, `AGENT_PROGRESS.md`, `POLICY_BENCHMARK_REPORT.md`,
`diagnosisReady.md` and `AUDIT_REPORT.md` rather than left as an overstatement.

Two process notes from the same check:

- `results/` also contains artifacts from a step-3 **synthetic** run alongside
  the real-data ones; the snapshot covered all of them and all matched.
- The first attempt at this check used wrong script filenames
  (`step2_train_predictor.py`, `step3_evaluate.py`); the real names are
  `step2_train_and_sweep.py` and `step3_preliminary_reports.py`. A failed rerun
  in a verification script must not be read as a reproducibility failure — the
  exit code needs checking before the diff is believed.

Test suite: **191 passed, 0 skipped** (was 187; the 4 new tests are the Step-6
`beta_greedy` and `lambda_units_caveat` contracts).


---

## 2026-10-01 (seventh pass) — P1-h/P1-i: the demo was claiming things it had not measured

Updating the frontend against the Step-8 findings surfaced three defects in what the
demo told a viewer. All three are *presentation* defects: the research code was
correct, but the UI overclaimed. Recorded because a demo that overstates its own
result is worse than no demo.

### F1 — the header advertised reinforcement learning; no RL policy was served

`index.html` was titled "Adaptive Autism Screening — **RL** Question Selection"
with the subtitle "**Reinforcement-learning policy** picks which Q-CHAT-10 questions
to ask". The demo only ever instantiates `GreedyIGPolicy` (information gain) and
`RandomPolicy`. The DQN and PPO arms exist in
`scripts/step5_policy_benchmark.py` but were never wired into the session layer.

Corrected: the header now says "Question Selection" and names the policy as
rule-based information-gain, with an explicit note that no RL policy is served
here and where the RL arms are actually benchmarked. `meta()` now publishes
`policies_available: ["greedy", "random"]` so the claim is checkable from the API
rather than only from prose.

### F2 — the greedy criterion is vacuous on a label-pure support, and the UI did not say so

This is the user-visible consequence of §0A.1. A viewer watches the policy ask
A1, A2, A3 ... in order and reasonably concludes the policy is broken. It is not:
on a label-pure support `H(Y|s) = 0` and every child support is pure too, so every
legal item's information gain is **exactly 0**, `argmax` has nothing to choose
between, and `greedy.py`'s `best_j = items_only[0]` initialisation takes over.

`scripts/demo_app.py` now computes a per-decision `selection_diagnostic` and the UI
renders it live. Measured on a real held-out episode through the running server:

```
  first: A9   support=284  n_pos=192  pure=False  vacuous=False  spread=0.387367
  chose A6    support=153  n_pos=150  pure=False  vacuous=False  spread=0.047761
  chose A1    support=124  n_pos=124  pure=True   vacuous=True   spread=0.000000
  chose A2    support=96   n_pos=96   pure=True   vacuous=True   spread=0.000000
  chose A3    support=86   n_pos=86   pure=True   vacuous=True   spread=0.000000
  chose A4    support=71   n_pos=71   pure=True   vacuous=True   spread=0.000000
  summary: 4 of 6 steps vacuous, support label-pure at stop
```

At step 3 the support becomes 124/124 positive, the IG spread collapses to exactly
0, all 8 remaining items tie, and the order becomes the fallback rule. The UI now
says so instead of leaving the viewer to guess.

`GreedyIGPolicy` is **not modified** — it is the file whose near-optimality result
is load-bearing. The diagnostic recomputes its formula in `demo_app.py`, and
`tests/test_demo_behavior_audit.py::test_selection_diagnostic_agrees_with_policy_choice`
asserts the recomputation and the policy pick the same item at every step, which is
what stops the diagnostic from drifting away from the real policy.

### F3 — a bug in the frontend's H1 reader made every budget look unsupported

`read_offline()` looked for `r["test_brier"]` in `step5_policy_benchmark_saudi.json`.
That artifact stores the field as `brier`. Every cell therefore came back `null`,
and `"supported"` evaluated to `False` for all six budgets — so the UI would have
reported H1 as **not supported at B = 3 and B = 4**, where it *is* supported.

Caught by running the server rather than only reading the code. Corrected, and
pinned by `test_meta_h1_is_populated_from_the_real_artifact`, which asserts the
audited supported-set exactly (B=3,4 supported; B=5,6 not). Reading H1 from the
artifact also means the UI cannot claim a win the comparator did not give.

### F4 — "same budget" was false for the random arm, in the browser and the console

`RandomPolicy` draws uniformly from `legal`, which **includes STOP**, so it ends
some episodes early. Measured over the 127 demo test records:

| | mean questions | asked 0 | reached 6 |
|---|---|---|---|
| greedy | 6.00 | 0% | 100% |
| random | 3.93 | 9.4% | 44.9% |

`demo_live.py` printed "adaptive question selection beats random selection at the
same budget" directly above a table showing 6.00 vs 4.02 questions. Part of that
gap is simply that random answered fewer questions. Both surfaces now state the
budgets side by side and point to the matched-budget comparison in
`POLICY_BENCHMARK_REPORT.md` §A.4. The browser shows the caveat only when the
random policy is selected, so it does not clutter the greedy path.

This is a property of the baseline as specified, not a coding error, so
`RandomPolicy` is unchanged. The consequence for the research record is noted in
§A.4: the random arm's item count is not an adaptive-policy comparison.

### Also corrected in passing

- "asked 3 of 10" conflated the 10-item pool with the budget of 6. Now "asked 3 of
  6 allowed".
- The status panel now renders H1 (per-budget, with the loss direction stated: a
  lower Brier is better, so the fixed subset *wins* at B=5,6) and the V-6 gate
  status, both read from artifacts so the page cannot drift from them.
- The footer states that no question-cost threshold is in force and that stopping
  is manual, because V-6 is unsigned.

### State

`tests/`: **204 passed, 0 skipped** (was 192; +12 covering the diagnostic's
agreement with the policy, the vacuity detection, the API contract, the H1 reader
regression, and four frontend-honesty assertions). The demo server was exercised
end to end on ports 8123-8127 for both policies and both modes; stderr clean.

Circularity and leakage status unchanged. Polish remains sealed. **V-6 remains
open with no threshold selected.**
