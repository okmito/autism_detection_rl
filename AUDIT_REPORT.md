# Audit Report — Real Data (2026-08-30)

All audits use real-data loaders from `src/data/ingest.py` with `synthetic=False`. No Polish outcomes used for tuning. NZ primary cohort NOT included (blocked).

## Circularity Audit — `src/audits/circularity.py:audit_circularity` (§16.1)

Procedure: sum-threshold oracle over item responses for thresholds 0..n, report exact_match_rate, UAR, sens/spec.

| Dataset | n | Rows | Label source | Best thr | Exact match | UAR | Classification | Interpretation |
|---|---|---|---|---|---|---|---|---|
| Saudi | 506 | 506 | questionnaire | 4 | 1.0000 | 1.0000 | **Deterministic** | `Class == (Screening Score >=4)` 506/506; `Screening Score == sum(A1:A10)` 506/506 — questionnaire-derived circular labels (§15) |
| UCI Child | 292 | 292 | questionnaire | 7 | 1.0000 | 1.0000 | **Deterministic** | `Class` deterministic from sum(A) threshold 7 — also circular; must not support claim of independent clinical screening validity (§16.1 gate: display circularity status next to every result) |
| Polish | 252 | 252 | clinical | 0 | 0.5357 | 0.5000 | **Not circular** | `group` (ASD/control) clinician-established; Sum_QCHAT not deterministic — material for transfer analysis (RQ3) |

Commands:
```
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('saudi')))"
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('uci_child')))"
python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('polish')))"
```

Gate behaviour (§16.1): circular datasets (Saudi, UCI Child) may be used for policy-optimization/algorithm-comparison experiments **with circularity status displayed**; not for independent clinical screening validity claims.

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

V-4 (MDE/comparison family) and V-7 (Polish denominator freeze) remain **PENDING** — no confirmatory Polish evaluation run.

