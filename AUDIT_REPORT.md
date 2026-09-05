# Audit Report — Real Data (2026-09-04)

All audits use real-data loaders from `src/data/ingest.py` with `synthetic=False`. No Polish outcomes used for tuning. NZ primary cohort: source located 2026-09-04 (see `V1_NZ_DATASET_RESOLUTION.md`); content validated on a public mirror; **licence = "Unknown"** is the only remaining V-1 blocker (human action required).

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

All 17 new tests pass (43 total in `tests/`, 1.78s on a single CPU thread).

