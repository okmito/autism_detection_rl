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

