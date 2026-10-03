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

### ⚠️ RETRACTED — "A units finding that belongs in the V-6 pack"

**This subsection was wrong and is withdrawn.** It originally claimed *"One reward,
two scales"*: that the support-posterior EVOI is ~100× smaller than the
neural-predictor marginal utility, so a λ grid could not transfer between them, and
that `beta_greedy` therefore needs its own calibration range of 0.001–0.01.

Two separate errors, both now measured:

1. **The comparison premise was never measured.** The "0.05–0.15 marginal utility for
   the neural-predictor reward" this rested on appeared in prose only — no code
   produced it. Measured, 0.05–0.15 is approximately the **p75–p95 band** of the
   predictor's gain distribution, not a typical value.
2. **There is no unit mismatch to bridge.** Both estimators score the same functional
   form `mean_y[1 − (p − y)²]` over the *same* support labels, differing only in
   whether `p` is the Beta-smoothed support posterior or the calibrated predictor, so
   both are expected Brier improvement with ceiling `max_p[1 − p(1−p)] = 0.25`.
   Measured ratio (support ÷ predictor): **0.427 at the median, 0.844 at the mean,
   1.017 at p75** — a factor of ~1, not ~100. The original number came from comparing
   the support-EVOI median *at depth 1* (0.0015) against the unsourced figure, when
   the predictor's own depth-1 median is 0.0022.

`beta_greedy`'s "own useful range of 0.001–0.01" is withdrawn with the premise. What
survives is a different and sharper problem: on a **pure** support the
support-posterior EVOI is analytically **non-positive**
(`gain = 1/(n+2)² − Σ_v w_v·1/(n_v+2)² ≤ 0`, because every child support is pure and
smaller), so the statistic behaves as a support-purity indicator rather than a graded
information measure — and the threshold grid spans only ~20 distinct behaviours with a
cliff between 0.001 and 0.003.

Evidence, and the decision this feeds: `V6_STOPPING_THRESHOLD_DECISION.md` §E3–§E4 and
`scripts/step8_evoi_scale_analysis.py`. **V-6 remains open with no threshold
selected.**

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

---

## 2026-10-01 (eighth pass) — V-4 / V-7 external-validation phase opened, and blocked correctly

This pass built the V-4 / V-7 validation phase. It did **not** unseal the Polish
cohort, did not train on it, and did not remove a gate.

### What was added

| Path | Purpose |
|---|---|
| `src/data/provenance.py` | Record-level cohort-identity verification (schema, participant keys, value-label-decoded categoricals, per-participant numerics). Metadata-only output, with a payload guard that rejects participant-shaped fields. |
| `src/eval/gates.py` | V-4 MDE + Holm-Bonferroni confirmatory family; V-7 denominator reconciliation; frozen-vs-cohort feature compatibility. |
| `src/eval/external_validation.py` | `FrozenPredictor` (prediction-only, immutable), `assert_no_fitting`, `preflight`, `group_to_label`, external metrics, bootstrap CIs. |
| `scripts/step9_v4_v7_gates.py` | Writes `results/polish_provenance_verification.json` and `results/v4_v7_validation_status.json`. |
| `scripts/step10_external_validation.py` | Runs the real circularity + leakage audits, then external validation — currently stops with exit code 2. |
| `tests/test_v4_v7_validation.py` | 36 regression tests. |
| `.gitignore` | `data/*.sav` added, so the SPSS export can never be committed. |

### Gate states — all honest

| Gate | Automated evidence | Human sign-off | Overall |
|---|---|---|---|
| **V-4** MDE + family freeze | **PASS** | **OPEN** | **OPEN** |
| **V-7** Polish denominator | **PASS** | **OPEN** | **OPEN** |
| **V-7** Baseline 10 recomputation | **OPEN** | **OPEN** | **OPEN** |
| **External validation** | — | — | **BLOCKED** |

V-4's automated evidence passes: the minimum detectable effect is computed for all
five pre-declared confirmatory comparisons at n = 252, with Holm-Bonferroni
adjustment, across a sensitivity grid of assumed paired-difference SDs
(0.50 / 1.00 / 2.00). Every MDE figure is explicitly labelled assumption-dependent —
the SD is not measurable before the cohort is opened, so a **sensitivity grid**
rather than a single number is the honest output. The gate stays OPEN because only
a supervisor can ratify the assumed effect size.

V-7's denominator discrepancy is resolved from the data: the cohort holds
**135 ASD + 117 control = 252**, matching the published total and the published ASD
count, while the publication's own text states 135 + 118 = 253. The discrepancy is
in the publication's arithmetic, and the project denominator is 252. Baseline 10
remains **OPEN** because faithfully reimplementing Sollis et al. requires that
publication's model specification, which this repository does not contain;
implementing it from a paraphrase would fabricate a comparator.

### Why external validation is BLOCKED (two independent reasons)

1. **V-4/V-7 human sign-off is OPEN.** The sealed cohort may not be opened.
2. **The frozen predictor cannot consume this cohort.** Measured, not asserted:

   | | frozen (Saudi) | sealed (Polish) |
   |---|---|---|
   | items | 10 | 25 |
   | response scale | binary | ordinal, 4–6 levels |
   | encoder width | 41 (`4n+1`) | 199 (`3n + Σm + 1`) |

   `encode_state` already supports `m_list`, and `MaskedPredictor`/`DQNPolicy`/
   `PPOPolicy` already accept it — the plumbing exists but `load_polish` never
   populates it. **No coercion is offered.** Squeezing a 4–6 level ordinal response
   into a binary slot would silently destroy information, so `preflight` raises
   instead. A compatible predictor must be trained under an explicit, reviewed
   decision before this phase can proceed; that decision has not been taken.

### Circularity / leakage — measured with the project's own audits

| Cohort | best threshold | exact match | classification |
|---|---|---|---|
| Polish (`GROUP`, clinical) | — | **0.5437** | **Not circular** |
| Saudi (`Class`, questionnaire) | 4 | **1.0000** | Deterministic |

That contrast is precisely why the sealed cohort is informative, and why it must
never be used to tune anything. Leakage status **PASS**: the development cohort
carries `label_source = questionnaire`, the external cohort `label_source = clinical`,
the instruments are disjoint (10 binary items vs 25 ordinal), and the development
records expose no participant key column at all.

Also recorded: the Step 2/3 research metrics were produced with `isotonic`
calibration, but the only frozen artefact persisted on disk uses `platt` — the
isotonic artefact was never saved. Recorded rather than papered over; it means the
research configuration cannot be reproduced exactly from disk.

### Scope respected

No change to the RL environment. No change to `GreedyIGPolicy`. No Saudi
retraining. No cohort merge. No threshold invented. No gate removed. No sign-off
fabricated. No new dataset created from the `.sav`. **V-6 untouched.**

Tests: **240 passed, 0 skipped** (was 204; +36 new).

---

## 2026-10-01 (ninth pass) — calibration reproducibility closed; Polish compatibility answered OPEN

### 1. Calibration root cause

`results/predictor_saudi_metrics.json` recorded `calibration: isotonic`, while the
only persisted predictor on disk used Platt. The cause was **not** a wrong
calibration choice. It was that **`scripts/step2_train_and_sweep.py` never saved
the model it trained.**

* `_train_one` built a `MaskedPredictor(calibration="isotonic")`, fitted it, fitted
  the isotonic calibrator on the validation split, computed test metrics, and
  returned a metrics dict. The trained object went out of scope and was discarded.
* Its own docstring claimed it returned a *"serializable model"* — it did not.
* No `torch.save` existed anywhere in step2. The only persisted predictor was the
  browser demo's Platt cache (`demo_app.py`), which is a **different artefact with a
  different calibration** and a documented reason (isotonic collapses to 3
  breakpoints on 95 validation records, unusable for a live counter).

So the research metrics described a model that could not be reloaded, re-hashed, or
verified from anywhere in the repository. The metrics were never *wrong* — they were
**unreproducible**.

Spec §12 permits either method (*"isotonic or Platt calibration fitted only on
training/validation data"*) and the config offered both, so neither was uniquely
authoritative. The defect was the missing artefact, not the choice.

### 2. Calibration method selected

**isotonic remains authoritative for research metrics** — it is what produced every
research number in the repository (Steps 2–8). The Platt cache stays **demo-only**
and is now recorded as such in the gate artifact.

### 3. Fix and verification

Step 2 now persists the fitted predictor before returning metrics:

* `results/predictor_saudi_v2_isotonic.pt` — weights
* `results/predictor_saudi_v2_isotonic.pkl` — calibrator
* `results/predictor_uci_child_v2_isotonic.{pt,pkl}` — same for the second cohort

The filename carries `MaskedPredictor.VERSION` so a semantics change cannot reuse
stale weights, and the metrics artifact records both SHA-256 digests.

**The decisive check — are the old metrics reproducible?** Re-running step 2 after
the fix regenerated every metric **bit-identically**:

| field | before | after | match |
|---|---|---|---|
| `test_brier` | 0.009122572011964558 | 0.009122572011964558 | yes |
| `test_ece` | 0.021937832758047126 | 0.021937832758047126 | yes |
| `test_auroc` | 1.0 | 1.0 | yes |
| `test_logloss` | 0.030677343667907076 | 0.030677343667907076 | yes |
| `val_brier` | 0.015866726086649366 | 0.015866726086649366 | yes |

| | |
|---|---|
| weights sha256 | `2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4` |
| calibrator sha256 | `a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863` |
| artifact version | 2 |

No stale artefact was deleted and no recorded value was edited to make it agree —
the metrics were *regenerated* and matched on their own.

### 4. Polish feature compatibility — answered OPEN, not guessed

`src/eval/qchat10_subset.py` asks whether a defensible Q-CHAT-10 subset can be
extracted so the frozen model runs unmodified. Three requirements, all unmet:

| requirement | state | why |
|---|---|---|
| scale compatibility | **OPEN** | 25 of 25 items ordinal, **0 natively binary**; no stated ordinal → binary rule |
| item identity | **OPEN** | cohort carries **no question wording** — only codes and SPSS value labels |
| provenance | **OPEN** | no canonical Q-CHAT-10 / Q-CHAT-25 item definition in this repository |

A partial mapping is refused, not accepted: the analysis requires exactly 10
identified items before it will call a subset defensible. Nothing is truncated,
thresholded, or coerced.

### 5. Governance — all human items remain OPEN

| Gate / item | Automated | Human |
|---|---|---|
| V-4 MDE + family freeze | PASS | **OPEN** |
| V-7 denominator (252) | PASS | **OPEN** |
| V-7 Baseline 10 | **OPEN** | **OPEN** |
| Q-CHAT-10 subset feasibility | **OPEN** | **OPEN** |
| External validation | **BLOCKED** | — |

`EVIDENCE_REQUESTS.md` states exactly what a supervisor must supply for each.
`POLISH_VALIDATION_DECISION.md` compares the three options and recommends
**Option A** — verified 10-item subset — on the grounds that it is the only option
that answers the question the gates were raised to ask and the only one that
validates the system actually built, while explicitly recording that the choice is
a research-priority judgement, not an engineering one, and belongs to the
supervisor.

### 6. Scope respected

No change to Saudi RL training, reward, state representation, policies, benchmark
artifacts, or V-6. The Polish cohort remains sealed. No gate removed, no sign-off
fabricated, no metric invented.

Tests: **258 passed, 0 skipped** (was 240; +18).

---

## 2026-10-01 (tenth pass) — Q-CHAT-10 feature contract: 9 of 10 verified, Q10 OPEN

The external-source mapping was supplied for Q1–Q9. The contract was then derived
from **primary sources already in the repository**, not from a secondary summary.

### Sources used

| source | what it establishes |
|---|---|
| `data/raw/Q-CHAT Saudi Arabia/ASD Screening Data for Toddlers in Saudi Arabia Data Set Description.pdf` | the instrument is *"an Arabic translation of Q-CHAT-10"*; `A1`…`A10` are **natively Binary (0,1)**, with `A{i}` = Q-CHAT-10 item {i}; scoring is *C/D/E → 1* for items 1–9 and *A/B/C → 1* for item 10 |
| `data/raw/Q-CHAT Polish/QCHAT.pdf` | full Q-CHAT-25 wording and printed option order for all 25 items |
| SPSS value labels of `QCHAT_dataset2_mendeley.sav` | each Polish response **code** → its label, and therefore the code ordering |

### The frozen model's feature contract

`src/data/ingest.py` builds the ten features as

```python
vals = np.array([float(row[f"A{j}"]) for j in range(1, 11)])
```

— read directly, with **no thresholding, no collapsing, no inversion, no
reordering**. Model feature *i* is column `A{i}` verbatim. Per the Saudi
description those columns already *are* the instrument's 0/1 recoding.

### Verified mapping (9 of 10)

| model feature | Saudi column | Q-CHAT-10 item | Q-CHAT-25 item | Polish variable |
|---|---|---|---|---|
| feature_1 | A1 | Q1 | Q1 | `qchat1recode` |
| feature_2 | A2 | Q2 | Q2 | `qchat2recode` |
| feature_3 | A3 | Q3 | Q6 | `qchat6recode` |
| feature_4 | A4 | Q4 | Q9 | `qchat9recode` |
| feature_5 | A5 | Q5 | Q10 | `qchat10recode` |
| feature_6 | A6 | Q6 | Q15 | `qchat15recode` |
| feature_7 | A7 | Q7 | Q17 | `qchat17recode` |
| feature_8 | A8 | Q8 | Q19 | `qchat19recode` |
| feature_9 | A9 | Q9 | Q25 | `qchat25recode` |
| **feature_10** | A10 | Q10 | **OPEN** | **OPEN** |

### Q10_MAPPING = OPEN

No verified Q-CHAT-25 variable corresponds to Q-CHAT-10 item 10. The remaining
unused Polish items were examined (Q3, Q4, Q5, Q7, Q8, Q11, Q12, Q13, Q14, Q16,
Q18, Q20, Q21, Q22, Q23, Q24) and none is *established* as item 10 — and item 10
has the opposite scoring direction (A/B/C → 1), so it cannot inherit the rule
used for items 1–9. Nothing was substituted.

`build_feature_matrix()` raises rather than returning a 9-vector, and
`external_validation.build_qchat10_features()` converts that into a precise
`ExternalValidationBlocked`. Step 10 therefore still exits **2** with
`metrics: null`.

### Ordinal → binary, and why it is a collapse not a cast

Every mapped item's SPSS codes run **typical → atypical** (verified against the
construct wording item by item), so the instrument's "C/D/E scores 1" reduces to
`code >= 2`:

```
feature_i = 1 if polish_code >= 2 else 0
```

Two details the tests now pin:

* **`qchat2recode` is not contiguously coded** — it is `0,1,2,3,**5**`, with code 4
  never observed. The split is applied to the code *value*, and an unobserved code
  raises rather than being scored. (An earlier version of the contract indexed
  labels positionally and would have accepted a code 4 that does not exist.)
* **`qchat25recode` is reverse-coded** — code 0 is `nigdy` (never) and code 4 is
  `wiele razy/dzień` (many times a day), i.e. the opposite of the printed option
  order. Applying the *printed* C/D/E letters would mark "never stares" as
  atypical, which contradicts the construct; the code-wise rule is the coherent
  reading and the reasoning is recorded in the contract for review.

Information loss is inherent and stated: 5 categories collapse to 2. The
documented invalid sentinel `11.0` is never scored, out-of-vocabulary codes raise,
and missing values return `NaN` rather than being imputed as 0.

### The published cut-off was not used

The instrument's *"more than 3 points → potential ASD traits"* is a rule for the
raw questionnaire. It is **not** the model's decision threshold; that stays the
frozen τ = 0.5 on a calibrated probability, asserted by test.

### Where 199 comes from

Measured, not assumed: `encode_state` returns **199** for the Polish cohort, and
`3n + Σm_list + 1 = 3·25 + 123 + 1 = 199`. The layout is
`[3n = 75 mask one-hot] [Σm = 123 response one-hot] [1 normalised budget]`. The
response block is concatenated **per item with that item's own cardinality**, so it
is 123 rather than 25×5 = 125 — two items have 4 levels, not 5. It is not a
one-hot over the full grid, and not an ordinal expansion.

### Frozen model untouched

`predictor_saudi_v2_isotonic.pt` sha256 `2366a283…` and `.pkl` sha256 `a1978f27…`
both verified **unchanged** after this pass. No retraining. No change to the RL
environment, reward, policies, benchmark artefacts, or V-6.

Tests: **309 passed, 0 skipped** (was 258; +51).

---

## 2026-10-02 (eleventh pass) — CORRECTED Q-CHAT-10 contract: mapping fixed, all 10 verified

The previously recorded Q1–Q9 mapping was **wrong** from Q3 onward. The corrected,
authoritative mapping (verified against the Autism Research Centre Q-CHAT-10
instrument and the original 25-item Q-CHAT source) is now the single canonical
definition in `src/data/qchat10_contract.py`.

| model feature | Saudi col | Q-CHAT-10 | Q-CHAT-25 | Polish variable |
|---|---|---|---|---|
| feature_1 | A1 | Q1 | Q1 | `qchat1recode` |
| feature_2 | A2 | Q2 | Q2 | `qchat2recode` |
| feature_3 | A3 | Q3 | **Q5** | `qchat5recode` |
| feature_4 | A4 | Q4 | **Q6** | `qchat6recode` |
| feature_5 | A5 | Q5 | **Q9** | `qchat9recode` |
| feature_6 | A6 | Q6 | **Q10** | `qchat10recode` |
| feature_7 | A7 | Q7 | **Q15** | `qchat15recode` |
| feature_8 | A8 | Q8 | **Q17** | `qchat17recode` |
| feature_9 | A9 | Q9 | **Q19** | `qchat19recode` |
| feature_10 | A10 | **Q10** | **Q25** | `qchat25recode` |

Q1 and Q2 were already correct; everything after Q2 was shifted. The Polish variable
set changed by exactly one addition: `qchat5recode`. Fifteen Q-CHAT-25 items remain
unused.

### Scoring is derived from the value labels, not from code magnitude

For every code the contract resolves SPSS code → Polish label → the English wording
printed in the Q-CHAT-25 instrument → the printed letter A–E, then applies the
official Q-CHAT-10 rule for that item (C/D/E = 1 for items 1–9; A/B/C = 1 for item
10). That derivation is the answer.

The simpler `code >= 2` happens to agree for all ten items, but it is a *consequence*,
not the premise. `verify_split_rule()` proves the two agree and raises on any
disagreement, so the shortcut can never silently diverge.

This matters concretely for **Q10 → Q25**, the only reverse-coded item:

```
qchat25recode:  0 = nigdy (never)   -> letter E -> 0
                1 = mniej niż raz/tydzień -> letter D -> 0
                2 = kilka razy/tydzień    -> letter C -> 1
                3 = kilka razy/dzień       -> letter B -> 1
                4 = wiele razy/dzień       -> letter A -> 1
```

More frequent staring carries a **higher** code and scores **1**. The printed order
(A = many times a day) runs opposite to the SPSS codes, so a magnitude assumption
would have inverted `feature_10`.

### Two latent defects found and fixed

1. **A duplicate scoring rule lived in `src/data/ingest.py`.** The dead function
   `qchat10_binary_map` encoded item 10 as `1 if v <= 2 else 0`, which assumes the
   10th item's codes run in *printed* order. Applied to reverse-coded
   `qchat25recode` that rule inverts the feature. It had no callers, so nothing was
   affected, but it was a live hazard and a second hard-coded mapping. Removed; the
   contract module is now the only definition, and a test enforces that.

2. **`_VALID_QCHAT_VALUES["qchat2recode"]` omitted `niemożliwe`.** That label is the
   valid code-5 response. The constant is unreferenced (dead), so no data was
   affected, but it was factually wrong and would have rejected valid responses if
   ever wired up. Corrected, and a test now cross-checks it against the contract.

### Polish label strings are accepted natively

The integrated Polish CSV stores the **label text**, not numeric codes. The contract
resolves either representation, so no lossy pre-conversion is needed.

### Shape and interface verified

* `build_feature_matrix` returns `(252, 10)` for the real cohort — exactly 10 binary
  positions, zero NaN, all columns non-degenerate (atypical rates 0.27–0.65).
* No padding, no missing position, no unrelated item; a short or long vector raises.
* The 10 binary features encode to the frozen **41-dimensional** input via the
  project's own `init_state`/`update_state` convention (`4n + 1`, `m_list=None`),
  and both frozen artefacts accept it in a forward pass. **No metric computed.**
* `predictor_saudi_v2_isotonic.{pt,pkl}` and `demo_model_saudi_seed0_platt_v2.{pt,pkl}`
  sha256 all **unchanged**. No retraining. RL environment, reward, policies,
  benchmark artefacts and V-6 untouched.

### External validation remains BLOCKED

`Q10_MAPPING` is no longer OPEN, but that is **not** a validation pass. Step 10 exits
**2** with `metrics: null`. Two blockers stand:

1. **V-4 and V-7 human sign-off are OPEN** — the sealed cohort may not be opened.
2. **The 25-item raw representation still does not feed the 41-dimensional frozen
   encoder directly.** The verified projection is a legitimate route, but it is
   authorised separately and deliberately not performed here.

The instrument's published ">3 points" rule was **not** used as the model threshold;
τ remains the frozen calibrated 0.5.

Tests: **345 passed, 0 skipped** (was 309; +36).

---

## 2026-10-02 (twelfth pass) — V-4 / V-7 / projection sign-off package prepared

Three supervisor decision packages produced. **No gate was closed. No approval was
fabricated or inferred.**

### New files

| file | purpose |
|---|---|
| `docs/V4_SIGNOFF_PACKAGE.md` | confirmatory family, n, MDE grid, assumed SD, Holm family, decision block |
| `docs/V7_SIGNOFF_PACKAGE.md` | denominator evidence vs published inconsistency, Baseline 10 specification request |
| `docs/QCHAT10_PROJECTION_APPROVAL.md` | per-item code/label/letter/binary evidence, information-loss statement, decision block |
| `docs/POLISH_EXTERNAL_VALIDATION_SIGNOFF.md` | mentor-facing summary of all eleven required points |
| `results/pre_validation_reproducibility.json` | machine-readable reproducibility record |

### Third gate condition added

The Q-CHAT-10 projection is now a **named, machine-readable gate**
(`Q-CHAT-10-PROJECTION`) rather than a note inside another gate. Step 10 requires
**all three** conditions and names each unmet one separately:

```json
"required_conditions": {
  "v4_human_sign_off": "OPEN", "v7_human_sign_off": "OPEN",
  "qchat10_projection_approval": "OPEN" }, "all_conditions_closed": false
```

### Evidence vs approval kept separate

`analyse_subset_feasibility` previously hard-coded `provenance` as unsatisfiable. Now
the three **evidence** requirements (`scale_compatibility`, `item_identity`,
`provenance`) can reach PASS from facts, while a **fourth** requirement,
`supervisor_approval`, is always OPEN and can never be closed by code. A new test
pins exactly this: with complete verified evidence the report reads
`automated_evidence = PASS`, `supervisor_approval = OPEN`,
`approved_as_external_representation = False`, `state = OPEN`.

### Reproducibility record

`results/pre_validation_reproducibility.json` pins predictor weights and calibrator
SHA256, calibration method (`isotonic`), git SHA, integrated CSV and SPSS source
SHA256, split fingerprint, feature-contract version `qchat10-contract/2.0.0` and
mapping version `qchat10-mapping/2.0.0`. It re-hashes the artefacts at write time and
records `modified_since_training: false` and `hashes_unchanged: true`.

### State after this pass

* Step 9 exit 0. Step 10 **exit 2**, `metrics: null`, three gate blockers plus the
  representation note.
* Tests **349 passed, 0 skipped** (was 345; +4).
* All four frozen/data SHA256s unchanged. No retraining. RL environment, reward,
  solvers, models, configs and V-6 untouched. No commit, no push.

Two of my own tests failed on the first run because they asserted the previous
blocker wording and the previous subset blocking set; both were updated to assert the
new, stricter behaviour rather than relaxed.

---

## 2026-10-02 (thirteenth pass) — supervisor decisions applied; PRIMARY external validation executed

Supervisor research decisions of 2026-10-02 (DECISIONS 1-9) implemented exactly. No
approval was inferred by any automated process; each is recorded with its reference.

### Result — reported as measured

Discrimination transfers: **AUROC 0.8959 (95% CI 0.8560-0.9314)** against
clinician-established Polish labels, Brier 0.1398, at the frozen tau = 0.5
(sensitivity 0.800, specificity 0.880, PPV 0.885, NPV 0.792).

**Two findings that materially qualify that result:**

1. **The model does not beat a trivial baseline.** Counting atypical answers gives
   AUROC **0.9369**; the frozen model gives 0.8959. The network is **0.041 worse**
   than simply summing the instrument. The instrument carries the signal, not the
   model.
2. **Calibration is poor.** Calibration slope **0.167** against an ideal 1.0,
   calibration-in-the-large 0.501, ECE 0.137. The isotonic calibrator has collapsed
   to a near-bimodal output (120 predictions ~0, 107 ~1). The ranking transfers; the
   probabilities do not and must not be read as calibrated risks.

Both are recorded in `results/polish_external_validation.json` under
`headline_finding`, `internal_item_count_reference` and `limitations` rather than
smoothed over.

### Gate logic

Seven conditions now gate the primary run, all required:
projection evidence, provenance, leakage, circularity, frozen artefact, calibration
reproducibility, denominator. `primary_gate_open()` fails if any one fails or is
unevaluated; a test drives all eight single-failure cases and the missing case.

Removed from the gate, per decision: V-4 SD ratification, Baseline-10
specification. Both are recorded in `NON_BLOCKING_FOR_PRIMARY` with reasons.

The MDE grid is **retained, not deleted**, and relabelled
`SENSITIVITY_ANALYSIS_REPORTED`.

### Honesty corrections made along the way

* Step 9's artifact tag claimed "NO human sign-off has been given; this artifact does
  NOT open the Polish cohort". That became false once decisions were recorded, so the
  tag was rewritten to state what is actually approved and that Step 10 alone opens
  the cohort after all seven conditions pass.
* `encode_qchat10_features` had been setting `budget = 0` while the Saudi metrics were
  computed with `budget = n`. Both yield 0.0 because `questions_remaining = 0`, so no
  result changed, but the convention now matches and a test asserts byte-equality
  against the exact Saudi terminal scoring path.

### Unchanged

All four frozen/data SHA256s verified unchanged. No retraining. RL environment,
reward, solvers, policies, configs and V-6 untouched (`git diff --stat` empty over
those paths). No commit, no push.

Tests: **368 passed, 0 skipped** (was 349; +19).

Full write-up: `docs/PRIMARY_EXTERNAL_VALIDATION_RESULTS.md`.

---

## 2026-10-02 (fourteenth pass) — diagnosis: WHY the frozen predictor loses to item count

Diagnostic pass. No retraining, no frozen artefact modified, no model selected on
Polish, and the Step 10 primary result left exactly as recorded.

### The gap is statistically resolved

Paired percentile bootstrap, identical resamples, 5000 resamples:

```
AUROC(frozen) - AUROC(item_count) = -0.0410
95% CI                            = [-0.0636, -0.0186]   entirely below zero
P(model better) = 0.0000    P(model worse) = 1.0000
```

The model is **significantly worse**, not indeterminate. Brier, by contrast, is
indeterminate: -0.0052, CI [-0.0319, +0.0224].

### Two separable causes, both identified

1. **Network saturation.** ~41% of Saudi terminal states score exactly 1.0
   pre-sigmoid versus ~21% on Polish. The Saudi target is a deterministic
   sum-threshold, so the loss is minimised by an overconfident mapping. Raw output
   sd is 0.476 — already near-bimodal before calibration.
2. **The isotonic calibrator destroys ranking.** It maps 103 distinct raw scores
   onto 16 levels, creating ties. Raw AUROC 0.9292 -> isotonic 0.8959. Isotonic
   alone costs **0.033 AUROC**.

The ranking was largely intact before calibration. The authoritative calibrator is
what broke it.

### Ablations (all Saudi-only, nothing fitted on Polish)

| predictor | Brier | AUROC | ECE | slope |
|---|---:|---:|---:|---:|
| raw uncalibrated | 0.1421 | 0.9292 | 0.1365 | 0.271 |
| Platt (existing Saudi artefact) | 0.1384 | 0.9296 | 0.1367 | 0.310 |
| isotonic (PRIMARY) | 0.1398 | 0.8959 | 0.1370 | 0.167 |
| item count | 0.1450 | 0.9369 | 0.1690 | 1.246 |
| **logistic L2 C=0.1 (Saudi train)** | **0.1097** | 0.9349 | **0.0775** | 1.370 |

Isotonic is the worst calibrator of the three on BOTH Brier and AUROC. A plain
logistic regression trained on 284 Saudi records beats the frozen network on every
metric.

### Incremental information

Best Saudi logistic minus item count = **-0.0021**. Logistic can use *which* items
are atypical, so it is the fair test of incremental information; it also fails to
beat the count. The item count captures essentially all available discrimination
signal at n = 252.

Precise wording adopted: "The current frozen predictor did not demonstrate
incremental discrimination over the item-count reference on this external cohort."

### Classification: C (confirmed)

### RL component kept separate

The predictor result does not invalidate adaptive question selection, but it bounds
it: every policy reward is scored through this predictor, so a rank-degraded,
miscalibrated scorer distorts the reward. Existing Saudi benchmark retained with an
explicit warning that those labels are circular and flatter every policy. Two defects
flagged, not hidden: beta_greedy is numerically identical to greedy (the known H1
saturation), and ppo asks only 1.43 items and matches nothing (looks like premature
stopping, not efficiency). No trained RL policy artefact exists on disk, so a fresh
external RL evaluation was deliberately not run.

### New permanent baselines

`src/eval/baselines.py` makes `item_count`, `random_questioning` and
`greedy_information_gain` mandatory references for any future improvement claim, with
`missing_baselines()` reporting omissions. Documented motivation: the 0.041 deficit.

### Unchanged

Primary result frozen: tau 0.5, AUROC 0.8959, Brier 0.1398, ECE 0.1370, item-count
reference 0.9369. All six artefact/data SHA256s verified unchanged. RL environment,
reward, policies, configs and V-6 untouched. No commit, no push.

Tests: **389 passed, 0 skipped** (was 368; +21).

Write-up: `docs/PREDICTOR_DIAGNOSIS.md`.

---

## 2026-10-02 (fifteenth pass) — predictor v3, frozen RL protocol, adaptive-selection external evaluation

Supervisor decisions implemented. Predictor v2 retained unmodified as a frozen legacy
artifact; it is NOT deleted.

### Predictor v3 (`logistic-saudi-v3`)

L2 logistic C=0.1 on the Saudi train split, Platt calibration on the Saudi validation
split. Artifact `results/predictor_logistic_saudi_v3.pkl`
sha256 `a6f28b85f2932151b745696f0bb24943024c31ac23be41fae91c4f6c483643a2`.
Saudi held-out test: Brier 0.0458, AUROC 1.0000 (circular labels), ECE 0.1777.

Partial states are scored by exact Bayesian marginalisation over all 2^10
configurations with Saudi-training priors, rather than imputing unobserved items.

**Two real bugs found and fixed during construction** (both documented):
1. Platt fitted on the *logit* diverged — the near-separable Saudi target saturates
   raw probabilities, logits reach +-40, and the unpenalised fit produced
   coefficient 60.9, collapsing every prediction to ~0.001. Fixed by regressing on
   the bounded raw probability (standard Platt form, monotone).
2. The 2^10 lookup table used a reversed bit convention relative to
   `itertools.product`, and normalised by the prior mass of ALL configurations
   instead of the CONSISTENT subset. Together these returned ~0.001 everywhere.
   Both fixed; `test_full_vector_and_state_scoring_agree_exactly` locks it.

### Protocol frozen before Polish

`results/polish_rl_external_protocol.json`. Budgets {2,3,4,5,10}, seed 0,
paired percentile bootstrap 5000 resamples, primary tau 0.5 (never tuned on
Polish), 0.3 recorded as secondary sensitivity only. Reward structure unchanged.

**`acceptable_AUROC_margin = "OPEN"`.** No margin was chosen and none invented; no
equivalence or non-inferiority claim is made in either direction.

DQN policies were **trained and persisted** per budget (`results/policies_v3/`),
which they previously were not. All five sha256-verified.

### Mandatory baselines

`REQUIRED_BASELINES` now includes `beta_greedy_evoi`. `assert_all_baselines()` fires
`MissingBaselineError` — it genuinely blocked the run during development until
`item_count` was recorded as a first-class arm.

### Polish results (once, after the freeze)

Full-question reference: v3 predictor AUROC 0.9349, raw item count 0.9369.

| B | greedy IG | beta-greedy | DQN | random |
|---|---|---|---|---|
| 2 | 0.7879 | 0.7879 | 0.7964 | 0.7877 |
| 3 | 0.8495 | 0.8483 | 0.8679 | 0.8233 |
| 4 | 0.8904 | 0.8915 | 0.8676 | 0.8247 |
| 5 | 0.9047 | 0.9081 | 0.8760 | 0.7821 |
| 10 | 0.9349 | 0.9349 | 0.9328 | 0.8070 |

### Conclusion: negative, reported as measured

**Adaptive questioning did NOT reduce questions while retaining acceptable
performance.** Every arm at every budget below 10 has a paired AUROC difference
whose 95% interval lies entirely below zero. Best sub-full result, beta-greedy at
B=5: -0.0288 [-0.0507, -0.0091].

Three further findings reported rather than smoothed over:
- **RL did not beat the heuristics.** DQN won at B=3 but lost at B=4 and B=5.
- **Beta-greedy equals greedy** to within 0.002 everywhere.
- Performance tracks questions asked almost as if selection barely mattered.

### Unchanged

v2 artifacts sha256-verified unchanged. RL environment, reward structure, existing
policy implementations, configs and V-6 untouched (`git status` clean over those
paths). Primary Step 10 result untouched. No commit, no push.

Tests: **414 passed, 0 skipped** (was 389; +25).

Write-up: `docs/PREDICTOR_V3_AND_RL_RESULTS.md`.

---

## 2026-10-02 (sixteenth pass) — corrected experimental design; confirmatory result is NULL

All approved decisions implemented. Existing results retained, not deleted. No frozen
material overwritten. Polish never used to tune.

### Design corrections

1. **Random baseline**: new `RandomFixedLengthPolicy` (STOP never legal).
   `b_min = B` for every arm, so every episode asks exactly B unique questions.
   Verified `min == max == B` for all arms and budgets. The old `RandomPolicy` asked
   only 1.73/2, 2.60/3, 3.12/4, 3.52/5, 4.96/10 — the defect that invalidated the
   first "adaptive beats random" comparison. `RandomPolicy` is untouched and
   reserved for a future stopping experiment.
2. **Two references**, reported separately: A v3 full-information 0.9349, B item
   count 0.9369. `delta_vs_v3_full` and `delta_vs_item_count` for every row. Neither
   called a sole ceiling.
3. **B=10 wording corrected.** All four arms return exactly 0.9349 because the
   fixed-budget semantics force all ten items; B=10 is a reference, not a comparison.

### Confirmatory family F1 — NULL

8 comparisons, Holm-Bonferroni alpha 0.05, 5000 paired bootstrap resamples with
identical resamples within each comparison. **All 8 intervals include zero; all
Holm-adjusted p = 1.0.**

Adaptive selection did NOT demonstrate superiority over budget-matched random at any
reduced budget. This **contradicts** the reframing proposed before execution, exactly
as the instruction to let the corrected experiment decide anticipated. The earlier
apparent adaptive advantage was an artefact of the random arm being under-asked.

Exploratory, uncorrected: random-fixed is AHEAD at B=4 (0.9144 vs greedy 0.8904);
DQN best at B=2/B=3, worst at B=4; beta-greedy and greedy near-identical.

### ExactDP objective inconsistency found and fixed

`ExactDP` optimised `1-(p_emp-y)^2` using the empirical label mean, NOT the
environment's `1-(p_hat-y)^2`. An optimality gap between the solver and
environment-trained policies would have been meaningless. An OPTIONAL `predictor`
argument was added; default `None` preserves prior behaviour bit-for-bit, verified:
B=2 still returns V*=0.93862870, matching the pre-existing artifact.

Optimality gap at B=5: greedy +0.0727, beta +0.0737, random +0.0964, **DQN +0.1003**
(worst). Reported as an oracle under the simulator, never as clinically optimal.

### Bayesian marginalisation audit

Saudi joint prior: 155/1024 observed, 869 zero-mass, 121 cells with count 1,
effective support 35.6. Variant A (empirical joint) FAILS with 35 undefined states —
exactly the Saudi-val configurations absent from train. Variant C (factorised)
PRIMARY; variant B add-alpha with **prespecified** alpha=0.5. No variant selected
using Polish.

### Reward/surrogate audit

`R = 1-(p_hat-y)^2`, lambda=0, unchanged, but the development label is
questionnaire-derived and the external target is clinician-established, so the
reward is a SURROGATE. Across 20 cells, reward vs external AUROC gives Pearson 0.845
/ Spearman 0.836, driven by budget level. **At matched budget the reward does not
rank-order external AUROC**: at B=4 the lowest-reward arm has the highest AUROC.

### DQN reproducibility

5 seeds per budget, 2000 episodes each, selection by highest mean terminal reward on
Saudi validation (ties -> lowest seed). Polish not inspected. All five seeds reported
per budget. Validation AUROC SD 0.013-0.027 at reduced budgets and exactly 0 at B=10
where all items are forced — an independent sanity check. New artifacts in
`results/policies_v3_multiseed/`; original single-seed files retained.

### Deferred / future work

NZ cross-dataset **DEFERRED**: the primary 1054-row file is absent, the pooled
6075-row file was NOT used as a substitute, and the NZ loader was NOT modified.
Ordinal Q-CHAT-10 predictor documented as `DOCUMENTED_NOT_IMPLEMENTED`; binary
contract unchanged. Equivalence margin remains OPEN; no equivalence,
non-inferiority or "statistically indistinguishable" claim anywhere.

### Unchanged

v2 and v3 predictors, all calibration artifacts, Step 10 primary result (tau 0.5,
AUROC 0.8959, Brier 0.1398), Saudi and Polish source data, RL environment contract,
reward definition, V-6 decision — all byte-identical. First evaluation retained.
No commit, no push.

Tests: **453 passed, 0 skipped** (was 414; +39).

Write-up: `docs/CORRECTED_DESIGN_RESULTS.md`.
Figure: `docs/figures/polish_fixed_budget_auroc_vs_questions.png`.

---

## 2026-10-02 (seventeenth pass) — frontend / demo redesign (presentation only)

Scope was strictly the demo front end. No research logic was touched.

### The one functional defect fixed

`scripts/demo_static/index.html` contained a **hard-coded frontend budget
constant** (`const BUDGET = 6`). The budget, instrument size, decision threshold and
the selectable policy list are now all read from `/api/meta`, so the UI cannot drift
from the backend configuration. Progress renders as `Question n of <budget>` computed
from `meta.budget`, and the policy selector is generated from
`meta.policies_available`.

A related inaccuracy was corrected rather than carried over. The old UI warned that the
random baseline "asks fewer questions". Measured behaviour: in **Interactive** mode the
random arm always asks the full budget (STOP is never offered to the policy), whereas
in **Auto episode** mode `run_episode` runs with `b_min=0`, so STOP *is* legal and
random stops early on some records. The disclosure is retained but scoped to the mode
where it is actually true.

### Design

Dark neon multi-gradient theme replaced with a calm, minimal, light interface: one
accent colour, neutral background, restrained borders, subtle shadows, generous
whitespace, no decorative animation. Repositioned as **"Adaptive Autism Screening —
Research Prototype"** with the required tagline and a subtle non-diagnosis disclaimer.
Hierarchy: header -> session/policy -> question -> answers -> progress -> explanation
-> result. Deep technical material moved under collapsible sections. Large gauge
replaced by a large continuous value plus a small explanatory label and a thin scale
marker, so no gauge implies clinical certainty.

Explainability is separated into a collapsible "Why this question?" driven entirely by
backend `selection` fields (`criterion_vacuous`, `ig_spread`, `support_size`,
`support_is_pure`, `tie_at_max`, `n_legal`, `posterior`, `top_items`). No explanation
is invented client-side; when `applicable === false` the UI says the policy has no
scoring criterion rather than fabricating one.

Added: policy details panel, skeleton/loading states ("Selecting next question…",
"Updating screening estimate…"), double-submit prevention, friendly errors with raw
backend text behind a "Technical details" disclosure, "Start new screening" reset,
responsive breakpoints, focus-visible styles, aria-pressed/labelledby/live-region, and
`prefers-reduced-motion`.

Architecture preserved: one self-contained HTML file, no framework, no CDN, no build
step.

### Existing honesty guards honoured

Four existing demo-audit tests asserted disclosures the rewrite had dropped; all four
were restored rather than deleted: the "no reinforcement-learning policy is served"
notice, the label-pure degeneracy explainer, the random early-stop disclosure (wired to
the policy toggle), and the V-6-unsigned / H1 "not supported at B" wording.

Two tests asserted the *old* hard-coded-budget pattern that this task was required to
remove, so they were updated to assert the stronger property instead: no budget
constant in the front end, budget read from `/api/meta`, and the screening estimate
rendered continuously with no bucketing or whole-percent rounding.

### Unchanged

`scripts/demo_app.py` and the whole API surface are byte-clean. RL environment, reward,
predictors, model weights, calibration artifacts, policies, configs, Q-CHAT mapping,
thresholds, V-4/V-7 validation and the V-6 decision are untouched. All six frozen
artifact/data SHA256s verified unchanged. No retraining. No commit, no push.

Tests: **487 passed, 0 skipped** (was 453; +34, of which 34 are new frontend guards in
`tests/test_demo_frontend_redesign.py`).

Live smoke test: both exposed policies (Greedy Information Gain, Random Fixed-Length)
in both modes — session start, question render, answer submit, next question, no
repeated items, progress 6/6, selection diagnostics, result render, counterfactual,
reset. Degenerate label-pure path exercised (support 63, IG spread 0.0, 6 items tied).
