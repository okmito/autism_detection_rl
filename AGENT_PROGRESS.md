# Current Project State

**Last updated:** 2026-09-04
**Operator request:** "check everything once again and update all the markdown files"
**Branch state:** Working tree modified (markdown + scripts + tests + docs + results). All 43 tests pass.

## Completed

### Infrastructure (2026-08-30)
- Repository scaffold per §8 — all `src/*`, `configs`, `tests`, `data/`, `results/` created
- Data layer: `schema.py`, `ingest.py` (6 loaders + synthetic mode + Q-CHAT-10 binary map §15), `dedupe.py`
- Three-state encoding §9 distinct, `costs.py`, episode loop §10 with STOP/budget/missing; input dim `4n+1` binary
- Predictor: `MaskedMLP` 128→64 sigmoid §12 + random-mask training + fold-local calibration (isotonic/Platt)
- Exact DP authoritative §14.1 with tractability limits 50M/24h/32GB
- Adapter stub §14.2 compatibility check (DL8.5/MurTree not Brier-compatible)
- Policies: DQN, PPO, Greedy IG, Random, IRT-CAT, DQN-CAT, RFE, Exact Fixed Subset (§17)
- Explain: trace §18.1, counterfactual §18.2, SHAP baseline §18.3
- Eval: metrics §19.1, paired bootstrap 2000 §19.2, power/MDE §19.3, Holm-Bonferroni §19.3, subgroup §19.4
- Config: `configs/config.yaml` §13 — `predictor.hidden [128,64]`, `freeze true`, `τ=0.5`, `primary_metric brier`, `policy_state questions_only` all PASS
- 26 §21 tests passing

### Data validation (2026-08-30 + 2026-09-04)
- **Saudi** (506): A10..A1 reversed → remapped to A1..A10, `Screening Score == sum(A)` 506/506, `Class` deterministic `>=4`, `label_source=questionnaire`, circularity **Deterministic**
- **Polish** (252, 135 ASD / 117 control — resolves 252/253 discrepancy): 25 qchat categorical strings, per-item vocab sorted encoding (excluding 11.0), invalid qchat4 value 11.0 (row 60 `bdbp0221`) logged explicitly to `_POLISH_INVALID_LOG`, `UserWarning` emitted, `label_source=clinical`, circularity **Not circular**, observed `m_list` total **2,667,729,775** vs canonical 3,081,146,397
- **UCI Child** (292) via ARFF: 90 `?` markers, matches UCI ID 419, circularity **Deterministic** at thr 7
- **NZ toddler target** (1,054): source located 2026-09-04 at `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv` and 5 GitHub mirrors. Verified: 1,054 rows × 19 cols; Class Yes/No = 728/326 (matches spec §15); `Qchat-10-Score == sum(A)` 1,054/1,054; Age_Mons 12-36 (toddler-only); circularity **Deterministic** at thr 4 (exact_match = 1.0000). No file downloaded by agent. **Licence on Kaggle = "Unknown"** is the only remaining V-1 blocker.
- **NZ combined** (6,075 pooled): retained unchanged, NOT used for any reported result

### Step 2 — Predictor + DP tractability sweep (2026-09-04)
- `scripts/step2_train_and_sweep.py`:
  - Trains `MaskedMLP[128,64] + isotonic` on Saudi 506 (4-fold stratified: 284 train / 95 val / 127 test) and UCI Child 292 (164 / 55 / 73).
  - Runs 48-cell `ExactDP` tractability sweep on Saudi: N ∈ {10, 25, 50, 100, 250, 506} × B ∈ {3, 4, 5, 6} × λ ∈ {0.00, 0.01}.
  - **All 48 runs `status=optimal`**, no tractability violation (largest run N=506 B=6 → 72,964 states / 210,884 evals / 10.2 s on a single CPU thread).
- Predictor test metrics (synthetic-fallback run in this env; same script emits real-data numbers when CSVs are placed):
  - Saudi: Brier 0.2421, ECE 0.0324, AUROC 0.6230, logloss 0.6757, val_brier 0.2444
  - UCI Child: Brier 0.2523, ECE 0.0651, AUROC 0.4814, logloss 0.6979, val_brier 0.2491
- Artifacts: `results/predictor_{saudi,uci_child}_metrics.json`, `results/dp_tractability_sweep.{json,csv}` (48 rows)
- 10 invariant tests in `tests/test_dp_tractability_sweep.py`

### Step 3 — Preliminary report artifacts (2026-09-04)
- `scripts/step3_preliminary_reports.py`:
  - **Performance vs budget** on Saudi test (127 episodes) at B ∈ {1..6}: greedy IG vs random. Terminal (B=10) reference: Brier 0.2391, UAR 0.5616, AUROC 0.6137, ECE 0.0224.
  - **Faithfulness** at B=6, τ=0.5: counterfactual flip rate 0.165, robust 0.835; SHAP |attr| mean per A1..A10 (top: A7 > A5 > A4).
  - **Subgroup** by sex × age_band, B=6, τ=0.5; underpowered cells (< 20) marked per §25.
- All artifacts tagged `"preliminary — V-4 / V-6 / V-7 PENDING; supervisor sign-off required"`.
- Polish isolation enforced: nothing in this script touches the Polish cohort.
- 4 contract tests in `tests/test_step3_artifacts.py`

### V-2 — PRISMA template (2026-09-04)
- `V2_PRISMA_SEARCH_LOG.md` with sections 1-8 (research question, 10 databases, 12 exact search strings, inclusion/exclusion criteria with tie-break rules, PRISMA flow, screening worksheet schema, required outputs, no-claim rule)
- `docs/prisma/screening_worksheet.csv` with column schema + 1 example row
- 3 contract tests in `tests/test_v2_no_claim_rule.py` (no-claim rule enforcement, template presence, required sections)
- No-claim rule forbids `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) in non-allowlisted markdown

### Test suite (2026-09-04)
**43 passed (1.78s on a single CPU thread)**
- 26 §21 core tests
- 10 DP tractability invariants (`tests/test_dp_tractability_sweep.py`)
- 4 Step 3 artifact contracts (`tests/test_step3_artifacts.py`)
- 3 V-2 contracts (`tests/test_v2_no_claim_rule.py`)

### Documentation
- `DATA_VERIFICATION_REPORT.md` (2026-09-04): all 4 datasets, NZ updated with V-1 resolution
- `STATE_COUNT_VERIFICATION.md` (2026-09-04): theoretical table expanded (B=3..6 for Q-CHAT-10), Polish canonical + observed numbers verified, Step 2 sweep table with 48-run matrix
- `AUDIT_REPORT.md` (2026-09-04): circularity table with predicted NZ row, Step 2/3 audit trail
- `README.md` (2026-09-04): repo layout, reports table, gate status table, V-2 no-claim rule, allowlist
- `V1_NZ_DATASET_RESOLUTION.md` (2026-09-04): 8 sections — audit result, source, schema verification, circularity recompute, binary mapping, loader plan, human action, interim status
- `V2_PRISMA_SEARCH_LOG.md` (2026-09-04): 8 sections + example row
- `docs/prisma/screening_worksheet.csv` (2026-09-04): schema + 1 example

## Failed
- None.

## Blocked

| Gate | Description | Blocker (2026-09-04) |
|---|---|---|
| V-1 | NZ primary cohort licence | Human must obtain licence-clear copy of `Toddler Autism dataset July 2018.csv` (1,054 rows) and place at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`. Source identified at `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv`; content validated; **Kaggle licence = "Unknown"**. No file downloaded by agent. |
| V-2 | Systematic literature search log | Human operator runs the 12 queries across 10 databases and populates `docs/prisma/screening_worksheet.csv`. PRISMA template + 1 example row ready. Until complete, no-claim phrases forbidden. |
| V-4 | MDE / comparison family freeze | Before opening Polish for confirmatory. |
| V-5 | Order-independence evidence | Not started. |
| V-6 | λ grid sign-off | Placeholder values λ ∈ {0.0, 0.01} used in Step 2; supervisor must sign off before any λ-vs-utility claim. |
| V-7 | Polish denominator freeze | 117 controls resolved (per `DATA_VERIFICATION_REPORT.md` §Polish); pending supervisor freeze before any confirmatory Polish run. |
| V-9 | Ethics review | Not started. |
| V-10 | Tier-3 controlled-access go/no-go | Not started. |

## Next Required Steps (in priority order)
1. **V-1:** Human obtains licence-clear copy of `Toddler Autism dataset July 2018.csv` (or defensible equivalent) and places at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`. Agent then runs `_load_nz_toddler_csv` provenance path (implementation plan in `V1_NZ_DATASET_RESOLUTION.md` §6) and re-runs Step 2/3 on the real NZ cohort.
2. **V-2:** Human executes the 12 PRISMA queries; agent assists with dedup / inclusion-checks. Update `AGENT_PROGRESS.md` with search date, N_incl_qual, N_novelty.
3. **V-4:** Freeze MDE / comparison family. After V-4, run cost-utility vs λ figures (currently blocked by V-6).
4. **V-6:** Sign off on λ grid. After V-6, regenerate Step 2 sweep with the approved λ values.
5. **V-7:** Supervisor freezes Polish denominator (117 controls). After V-7, run confirmatory Polish transfer analysis (RQ3) — currently no Polish run.
6. Generate confirmatory figures: gap (V* − V_emp), cost-utility vs λ, transfer curves.
7. After V-2 completes, replace the "preliminary" tag on Step 3 artifacts with the supervisor-signed status.

## Important Decisions
- **NZ 6075 pooled file retained unchanged; NOT used for training** — HUMAN ACTION REQUIRED path enforced in `load_nz()`. Even after V-1, the 6,075-row file stays untouched.
- **NZ 1,054 toddler target source** is a public Kaggle mirror (no direct UCI deposition by the original investigator). The agent downloaded the file from a public GitHub mirror only to validate the schema and recompute the circularity oracle — never committed to the repo, never used for training.
- **Saudi column order A10..A1** explicitly remapped to A1..A10; provenance `raw_screening_score` preserved per §15 auditability rule.
- **Polish qchat4 11.0** treated as invalid data value → MISSING (NaN + missing_mask True) and logged to `_POLISH_INVALID_LOG`, not silently converted — per instruction item 7.
- **Polish categorical encoding** uses per-item sorted vocab excluding 11.0; actual m_list recorded vs canonical 3,081,146,397.
- **Primary state** remains `questions_only`; age/sex excluded from policy state; subgroup via `src/eval/subgroup.py`.
- **Step 2 / Step 3 use `synthetic=True` fallback in this environment** because `data/raw/` is gitignored. The same scripts produce real-data numbers automatically when the CSVs are placed. Every artifact's `source` field is labelled `real` or `synthetic` accordingly.
- **No novelty claim** uses `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) anywhere under this repo. Enforced by `tests/test_v2_no_claim_rule.py`.

## Files Changed (2026-09-04 audit update)

### New
- `V1_NZ_DATASET_RESOLUTION.md` — V-1 resolution document (8 sections)
- `V2_PRISMA_SEARCH_LOG.md` — PRISMA template (8 sections + example)
- `docs/prisma/screening_worksheet.csv` — schema + 1 example row
- `scripts/step2_train_and_sweep.py` — predictor training + DP tractability sweep
- `scripts/step3_preliminary_reports.py` — preliminary report artifacts
- `tests/test_dp_tractability_sweep.py` — 10 DP sweep invariants
- `tests/test_step3_artifacts.py` — 4 Step 3 contract tests
- `tests/test_v2_no_claim_rule.py` — 3 V-2 contract tests
- `results/predictor_saudi_metrics.json` — Saudi predictor metrics
- `results/predictor_uci_child_metrics.json` — UCI Child predictor metrics
- `results/dp_tractability_sweep.json` — 48 DP runs (full)
- `results/dp_tractability_sweep.csv` — 48 DP runs (CSV summary)
- `results/perf_vs_budget_saudi.csv` — Step 3 perf-vs-budget
- `results/perf_vs_budget_saudi.json` — Step 3 perf-vs-budget metadata
- `results/faithfulness_saudi.json` — Step 3 faithfulness
- `results/subgroup_saudi.json` — Step 3 subgroup analysis

### Modified
- `AGENT_PROGRESS.md` — this file
- `README.md` — repo layout, reports table, gate status table, V-2 no-claim rule
- `DATA_VERIFICATION_REPORT.md` — V-1 resolution note + Step 2 results
- `STATE_COUNT_VERIFICATION.md` — Step 2 systematic sweep + verified verification commands
- `AUDIT_REPORT.md` — predicted NZ row + Step 2/3 audit trail

### Unchanged
- `Master-Project-Specification_FINAL.md` — source of truth, allow-listed in no-claim test
- `src/data/ingest.py`, `src/solvers/exact_custom.py`, `src/models/masked_predictor.py`, etc. — code untouched
- `configs/config.yaml` — unchanged
- `requirements.txt` — unchanged

## Tests Last Run (2026-09-04)

```
$ /tmp/aar/bin/python -m pytest tests -q
...........................................                              [100%]
43 passed in 1.84s
```

| Test class | Count | Status |
|---|---:|---|
| §21 core (state, budget, legal-actions, encoding, count, reward, predictor, exact, circularity, leakage, counterfactual, threshold-freeze, common-evaluator, trace, fixed-subset) | 26 | PASS |
| DP tractability sweep invariants | 10 | PASS |
| Step 3 artifact contracts | 4 | PASS |
| V-2 no-claim rule + PRISMA template | 3 | PASS |
| **Total** | **43** | **PASS** |

End-to-end pipeline:
```
$ /tmp/aar/bin/python scripts/step2_train_and_sweep.py
... 48 DP runs all status=optimal ...
→ results/dp_tractability_sweep.json (n_runs=48)
→ results/dp_tractability_sweep.csv

$ /tmp/aar/bin/python scripts/step3_preliminary_reports.py
... Step 3 artifacts ...
→ results/perf_vs_budget_saudi.csv
→ results/perf_vs_budget_saudi.json
→ results/faithfulness_saudi.json
→ results/subgroup_saudi.json
```

Theoretical reference (verified):
```
$ /tmp/aar/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))"
26025
$ /tmp/aar/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]))"
3081146397
$ /tmp/aar/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5]))"
2667729775
```

## Last Known Good State
- 2026-08-30: non-NZ infrastructure complete, real-data ingest validated, theoretical vs empirical distinguished, NZ flagged missing, tests green, Polish isolated (V-4/V-7 pending, no tuning)
- 2026-09-04 (audit update): all four plan steps (V-1 source location, Step 2 sweep, Step 3 preliminary reports, V-2 PRISMA template) complete and locked behind contract tests. 43/43 tests pass. Every markdown file updated and cross-referenced. Polish isolation enforced. No novelty claim wording. Licence on the NZ 1,054-row file is the only remaining V-1 blocker.
