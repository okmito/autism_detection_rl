# Current Project State

**Last updated:** 2026-10-01 (Parts 0+1+2+3 complete)
**Operator request:** "make the project ready for diagnosis" → audit produced `diagnosisReady.md`; Parts 0–3 implemented.
**Branch state:** see git log. **79 tests pass** (54 pre-existing + 13 RL + 12 Step-5 benchmark).
**READ FIRST, in this order:**
1. `diagnosisReady.md` — the audit: the RL half had **never been trained** before 2026-10-01, and what clinical readiness would actually require.
2. `POLICY_BENCHMARK_REPORT.md` — **the headline result**: greedy-IG attains the exact optimum; both learned policies do not.

## 2026-10-01 — Part 2: policy benchmark ✅ (first real RL measurement)

`scripts/step5_policy_benchmark.py` evaluates DQN, PPO, Greedy-IG, Random, and ExactDP at matched budgets B ∈ {1..6} on the same held-out test split, same predictor, same `run_episode` evaluator. Artifact: `results/step5_policy_benchmark_saudi.{json,csv}`. Tests: `tests/test_step5_benchmark.py` (12).

**Optimality gap V\* − V_emp (train split, λ=0, lower is better):**

| B | greedy | dqn | ppo | random |
|---|---|---|---|---|
| 1 | −0.000000 | +0.044531 | +0.040883 | +0.055131 |
| 3 | +0.001595 | +0.032676 | +0.035361 | +0.060355 |
| 5 | +0.012969 | +0.048338 | +0.105890 | +0.090981 |
| 6 | +0.006749 | +0.038949 | **+0.138141** | +0.090097 |

**This is a negative result for RL on this task, and it is the honest one.** Training worked (DQN loss 0.0068→0.0002, PPO value loss 0.066→6e-5) but the policies converge to something worse than a one-step lookahead heuristic. PPO degrades monotonically with budget and is worse than random at B=5/B=6.

Behavioural notes: DQN **never stops early** (asks exactly B every time — no adaptive stopping under λ=0); PPO stops very early (1.84 items at B=6), which likely explains its worsening gap. At B=3 greedy and exact produce **identical** test metrics, reproducing the §17 #7 near-optimality claim.

Caveats: single seed (no variance), λ=0 only (V-6 pending), and **all labels are circular** — these numbers measure fit to the questionnaire's own scoring rule, not autism. Full analysis in `POLICY_BENCHMARK_REPORT.md`.

## 2026-10-01 — Part 0 (environment) + Part 1 (RL code)

### Part 0 — environment restored ✅
The repo had **no `.venv`, no `data/raw/`, no `results/`** (all gitignored), so no documented number was reproducible.
- Rebuilt venv on Python 3.14.7. **Note:** `requirements.txt` resolves `torch` to the CUDA build, which exhausts `/tmp` (3.7G tmpfs). Installed CPU-only torch via `--extra-index-url https://download.pytorch.org/whl/cpu` with `TMPDIR` pointed off the tmpfs. Same disk failure class as 2026-09-30.
- Restored all three datasets — see "Data provenance" below; **the UCI ARFF and the Polish CSV are derived conversions, not originals**.
- Re-ran Step 2 + Step 3: all 8 artifacts regenerated with `source: real`. All 48 DP runs `status=optimal`. State counts match `STATE_COUNT_VERIFICATION.md` exactly.
- Test suite: 40 passed / 14 skipped → after regeneration **54 passed, 0 skipped**.

### Part 1 — RL code fixed and trained ✅
**Two** real defects in code that had never executed (full detail in `RL_TRAINING_REPORT.md`):
1. `DQNPolicy.train_step` built a legal-action mask and discarded it → bootstrap optimised toward illegal actions.
2. `PPOPolicy` had **no training method at all** — the critic never received a gradient.

A third alleged defect (terminal next-state tensor shapes breaking `torch.cat`) was investigated on 2026-10-01 and **retracted** — `torch.cat(dim=0)` concatenates along dim 0, so the original code did not raise. Do not repeat that claim.

Fixed both; added `src/policies/replay.py` (`ReplayBuffer`, stores per-transition legal sets), `scripts/step4_train_policies.py`, and `tests/test_rl_training.py` (13 tests). Aligned PPO's batch format with DQN's.

First real runs (Saudi 506, B=6, seed 0): DQN loss 0.006889 → 0.000235; PPO value loss 0.066314 → 5.9e-05; PPO entropy flat at ~2.11 (no collapse). Artifact: `results/step4_policy_training_saudi.json`.

### Part 2 — policy benchmark ✅ DONE 2026-10-01
DQN, PPO, Greedy-IG, Random and ExactDP are now evaluated at matched budgets B ∈ {1..6} on the same held-out test split with the same evaluator. See the Part 2 section at the top of this file and `POLICY_BENCHMARK_REPORT.md`.

**Finding: greedy-IG attains the exact optimum (gap ≤ 0.013); both learned policies do not, and PPO is worse than random at B=5/6.** Next: multi-seed variance, then resolve V-6 (λ) before any λ-dependent claim.

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

### ⚠️ Superseded note (2026-10-01)
The "Policies: DQN, PPO, …" line above records the **scaffold** only. Through 2026-09-30 those policy classes existed but **had never been trained or evaluated** — no training call existed anywhere in the repo, and every run used Greedy-IG or Random despite `configs/config.yaml` declaring `policy.type: dqn`. Both were fixed and both were first trained/benchmarked on 2026-10-01. See `RL_TRAINING_REPORT.md` and `POLICY_BENCHMARK_REPORT.md`.

### Data provenance (2026-10-01) — ⚠️ two files are DERIVED, not originals
All three CSVs were re-fetched from scratch on 2026-10-01. Any agent re-doing this must know two files are conversions:

| Dataset | Source | Status |
|---|---|---|
| **Saudi 506** | `github.com/Sugandaram/Autism-Spectrum-Disorder-Screening-Data-for-Toddlers-in-Saudi-Arabia-Data-Set` (only repo found hosting the file) | **Original CSV.** 506 rows, `A10..A1` order, Class 341/165 — matches §15 |
| **UCI Child 292** | `archive.ics.uci.edu/static/public/419/data.csv` | ⚠️ **DERIVED.** UCI **no longer serves an ARFF for id 419** — only CSV. `Autism-Child-Data.arff` was generated from the official CSV: same 292 rows, same 21 columns, `NaN`→`?`. Verified: 292 records, 151/141 labels, 90 `?` markers (43 in `ethnicity`, 43 in `relation`, 4 in `age`) — the 90 matches the documented count. **Not the original ARFF file.** |
| **Polish 252** | Mendeley Data `tmpkt2mfkg` (`QCHAT_dataset1.sav`, sha256 `7fed516f…` verified against Mendeley's published hash) | ⚠️ **DERIVED CSV from SPSS `.sav`.** The `.sav` is the original; `polish_qchat.csv` was generated via pyreadstat. `group`/`sex` written using the file's **own SPSS value labels** (group 1=ASD, 7=control; sex 1=Male, 2=Female), not guessed. Verified: 252 records, 135 ASD / 117 control, `label_source=clinical`, invalid `qchat4=11.0` at row 60 / `bdbp0221` — matches `DATA_VERIFICATION_REPORT.md` exactly. |
| **NZ 1054** | — | Still **absent**, V-1 blocked (licence "Unknown"). Never downloaded. |

**Verification after restore:** Polish state count recomputed from the restored file → `2,667,729,775`, exactly matching `STATE_COUNT_VERIFICATION.md`. Canonical → `3,081,146,397`. Both confirmed.

### Data validation (2026-08-30 + 2026-09-04)
- **Saudi** (506): A10..A1 reversed → remapped to A1..A10, `Screening Score == sum(A)` 506/506, `Class` deterministic `>=4`, `label_source=questionnaire`, circularity **Deterministic**
- **Polish** (252, 135 ASD / 117 control — resolves 252/253 discrepancy): 25 qchat categorical strings, per-item vocab sorted encoding (excluding 11.0), invalid qchat4 value 11.0 (row 60 `bdbp0221`) logged explicitly to `_POLISH_INVALID_LOG`, `UserWarning` emitted, `label_source=clinical`, circularity **Not circular**, observed `m_list` total **2,667,729,775** vs canonical 3,081,146,397
- **UCI Child** (292) via ARFF: 90 `?` markers, matches UCI ID 419, circularity **Deterministic** at thr 7
- **NZ toddler target** (1,054): source located 2026-09-04 at `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv` and 5 GitHub mirrors. Verified: 1,054 rows × 19 cols; Class Yes/No = 728/326 (matches spec §15); `Qchat-10-Score == sum(A)` 1,054/1,054; Age_Mons 12-36 (toddler-only); circularity **Deterministic** at thr 4 (exact_match = 1.0000). No file downloaded by agent. **Licence on Kaggle = "Unknown"** is the only remaining V-1 blocker.
- **NZ combined** (6,075 pooled): retained unchanged, NOT used for any reported result

### Step 2 — Predictor + DP tractability sweep (2026-09-04, re-run on real data 2026-09-30)
- `scripts/step2_train_and_sweep.py`:
  - Trains `MaskedMLP[128,64] + isotonic` on Saudi 506 (4-fold stratified: 284 train / 95 val / 127 test) and UCI Child 292 (164 / 55 / 73).
  - Runs 48-cell `ExactDP` tractability sweep on Saudi: N ∈ {10, 25, 50, 100, 250, 506} × B ∈ {3, 4, 5, 6} × λ ∈ {0.00, 0.01}.
  - **All 48 runs `status=optimal`**, no tractability violation (largest run N=506 B=6 → 25,023 states / 68,408 evals / 2.2 s on a single CPU thread).
- Predictor test metrics (**real CSVs**, 2026-09-30 run):
  - Saudi: Brier 0.0134, ECE 0.0168, AUROC 0.9877, logloss 0.1234 (near-ceiling AUROC expected — labels deterministically circular, §16.1 gate applies)
  - UCI Child: Brier 0.0713, ECE 0.0765, AUROC 0.9301, logloss 0.9562 (same circularity caveat)
- Artifacts: `results/predictor_{saudi,uci_child}_metrics.json`, `results/dp_tractability_sweep.{json,csv}` (48 rows, all `source=real`)
- 10 invariant tests in `tests/test_dp_tractability_sweep.py`
- (Prior 2026-09-04 synthetic-fallback numbers — Saudi Brier 0.2421/AUROC 0.6230; sweep largest 72,964 states — preserved in `AUDIT_UPDATE_2026-09-04.md` as historical record; superseded.)

### Step 3 — Preliminary report artifacts (2026-09-04, re-run on real data 2026-09-30)
- `scripts/step3_preliminary_reports.py`:
  - **Performance vs budget** on Saudi test (127 episodes) at B ∈ {1..6}: greedy IG vs random. Terminal (B=10) reference: Brier 0.0053, UAR 0.9881, AUROC 0.9997, ECE 0.0049 (real data; ceiling reflects label circularity).
  - **Faithfulness** at B=6, τ=0.5: counterfactual flip rate 0.417, robust 0.583; SHAP |attr| mean per A1..A10 (top: A8 > A6 > A2).
  - **Subgroup** by sex × age_band, B=6, τ=0.5 (sex_F n=86 UAR 0.935; sex_M n=41 UAR 0.892); underpowered cells (< 20) marked per §25.
- All artifacts tagged `"preliminary — V-4 / V-6 / V-7 PENDING; supervisor sign-off required"` and carry `source=real`.
- Polish isolation enforced: nothing in this script touches the Polish cohort.
- 4 contract tests in `tests/test_step3_artifacts.py`

### V-2 — PRISMA template (2026-09-04)
- `V2_PRISMA_SEARCH_LOG.md` with sections 1-8 (research question, 10 databases, 12 exact search strings, inclusion/exclusion criteria with tie-break rules, PRISMA flow, screening worksheet schema, required outputs, no-claim rule)
- `docs/prisma/screening_worksheet.csv` with column schema + 1 example row
- 3 contract tests in `tests/test_v2_no_claim_rule.py` (no-claim rule enforcement, template presence, required sections)
- No-claim rule forbids `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) in non-allowlisted markdown

### Environment rebuild + real-data re-run (2026-09-30)
- Rebuilt venv (`.venv/`, Python 3.14) with full `requirements.txt` — prior `/tmp/aar` venv was lost with `/tmp`.
- Found `results/` empty (gitignored; prior artifacts never survived the environment) — regenerated all 8 artifacts from the real CSVs in `data/raw/` (Saudi 506, Polish 252, UCI Child 292 all present locally; only NZ 1,054 missing pending V-1).
- Every artifact now carries `"source": "real"` (previously `synthetic`). Numbers changed accordingly — see Step 2/3 sections and `STATE_COUNT_VERIFICATION.md`.
- Fixed `tests/test_v2_no_claim_rule.py` scan scope: it was walking `REPO.rglob("*.md")` and failed once `.venv/site-packages/**/*.md` appeared inside the repo. Now skips hidden directories (environment artifacts are not project markdown; the rule's intent is unchanged).

### Test suite (superseded — see 2026-10-01)
The 2026-09-30 entry said 43 tests. That figure predated the 11 demo-behaviour tests in `56a1e3d`; 54 was correct then, and 67 is correct now. Breakdown in the 2026-10-01 test table below.

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
0. ~~Part 2 (benchmark the trained policies)~~ ✅ **DONE 2026-10-01** — see top of file.
1. **Multi-seed variance (NEW — now the top methodological gap).** The Step-5 gaps come from a single seed. Run `config.eval.seeds = 10` seeds and report mean ± spread; the current DQN/PPO-vs-greedy differences are not interpretable without it. Add `--seeds` to `scripts/step5_policy_benchmark.py`.
2. **V-6 (λ grid sign-off).** Every current number is λ=0, so cost is unpriced and adaptive stopping is unexcused — which is exactly the regime where a sequential policy could beat greedy. Resolving λ is the highest-value change to the experiment.
3. **Diagnose PPO's early stopping** (1.84 items at B=6). Likely exploration/credit-assignment; isolate before further tuning.
4. **V-1:** Human obtains licence-clear copy of `Toddler Autism dataset July 2018.csv` (or defensible equivalent) and places at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`. Agent then runs the `_load_nz_toddler_csv` provenance path (plan in `V1_NZ_DATASET_RESOLUTION.md` §6) and re-runs Step 2/3 on the real NZ cohort.
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
- **Step 2 / Step 3 run on REAL Saudi / UCI Child data as of 2026-09-30** (`source: real` in every artifact). NZ remains synthetic-fallback pending V-1 and is never reported. Prior synthetic-fallback numbers are retained only as historical record in `AUDIT_UPDATE_2026-09-04.md`.
- **No novelty claim** uses `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) anywhere under this repo. Enforced by `tests/test_v2_no_claim_rule.py`.

## Files Changed (2026-10-01)

### New
- `diagnosisReady.md` — diagnostic-readiness audit + roadmap (read this first)
- `RL_TRAINING_REPORT.md` — Part 1 defects, fixes, first real training runs
- `POLICY_BENCHMARK_REPORT.md` — **Part 2 results: greedy attains the exact optimum, RL does not**
- `src/policies/replay.py` — `ReplayBuffer` (stores per-transition legal-action sets)
- `scripts/step4_train_policies.py` — first entry point that trains DQN/PPO
- `scripts/step5_policy_benchmark.py` — matched-budget benchmark incl. ExactDP reference + optimality gap
- `tests/test_rl_training.py` — 13 regression tests for the RL defects
- `tests/test_step5_benchmark.py` — 12 benchmark contract tests
- `results/step4_policy_training_saudi.json` — first real training run (gitignored)
- `results/step5_policy_benchmark_saudi.{json,csv}` — first real benchmark (gitignored)

### Modified
- `src/policies/dqn.py` — `train_step` rewritten: legal mask applied, duplicate `torch.cat` collapsed
- `src/policies/ppo.py` — **added** `train_step` / `select_action` / `legal_mask`; batch format aligned with DQN
- `AGENT_PROGRESS.md` — this file
- `README.md` — test counts, RL section, layout, reports table, invariants

### Derived data files (⚠️ not originals — see Data provenance)
- `data/raw/UCI/Autism-Child-Data.arff` — generated from UCI's official CSV; UCI no longer serves ARFF for id 419
- `data/raw/Q-CHAT Polish/polish_qchat.csv` — generated from the original Mendeley `.sav` via pyreadstat

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

## Tests Last Run (2026-10-01)

```
$ .venv/bin/python -m pytest tests -q
........................................................................ [ 91%]
.......                                                                  [100%]
79 passed in 3.29s
```

| Test class | Count | Status |
|---|---:|---|
| §21 core (state, budget, legal-actions, encoding, count, reward, predictor, exact, circularity, leakage, counterfactual, threshold-freeze, common-evaluator, trace, fixed-subset) | 26 | PASS |
| DP tractability sweep invariants | 10 | PASS |
| Step 3 artifact contracts | 4 | PASS |
| V-2 no-claim rule + PRISMA template | 3 | PASS |
| Demo behaviour audit regressions | 11 | PASS |
| RL training regressions (`tests/test_rl_training.py`) | 13 | PASS (2026-10-01) |
| Step 5 benchmark contracts (`tests/test_step5_benchmark.py`) | 12 | PASS (2026-10-01) |
| **Total** | **79** | **PASS** |

**Correction to the record:** this file previously said "43 tests" and a "working tree modified" state; the 43 figure predated the 11 demo-behaviour tests added in commit `56a1e3d`. README's "54" was the correct pre-2026-10-01 count. 13 RL tests → 67, then 12 Step-5 tests → 79.

| Test class | Count | Status |
|---|---:|---|
| (see the 2026-10-01 test table above — 67 total) | 67 | PASS |

End-to-end pipeline:
```
$ .venv/bin/python scripts/step2_train_and_sweep.py
... 48 DP runs all status=optimal ...
→ results/dp_tractability_sweep.json (n_runs=48)
→ results/dp_tractability_sweep.csv

$ .venv/bin/python scripts/step3_preliminary_reports.py
... Step 3 artifacts ...
→ results/perf_vs_budget_saudi.csv
→ results/perf_vs_budget_saudi.json
→ results/faithfulness_saudi.json
→ results/subgroup_saudi.json
```

Theoretical reference (verified):
```
$ .venv/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))"
26025
$ .venv/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]))"
3081146397
$ .venv/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5]))"
2667729775
```

## Last Known Good State
- 2026-08-30: non-NZ infrastructure complete, real-data ingest validated, theoretical vs empirical distinguished, NZ flagged missing, tests green, Polish isolated (V-4/V-7 pending, no tuning)
- 2026-09-04 (audit update): all four plan steps (V-1 source location, Step 2 sweep, Step 3 preliminary reports, V-2 PRISMA template) complete and locked behind contract tests. 43/43 tests pass. Every markdown file updated and cross-referenced. Polish isolation enforced. No novelty claim wording. Licence on the NZ 1,054-row file is the only remaining V-1 blocker.
- 2026-09-30 (env rebuild + real-data run): venv rebuilt, results/ regenerated from the real Saudi/UCI CSVs (`source: real` everywhere), no-claim test scope fixed to skip hidden dirs, all four living docs updated with real numbers. 43/43 tests pass (superseded count — see 2026-10-01). The 8 verification gates are unchanged — all remaining substantive work is gated on human/supervisor actions (V-1 licence, V-2 searches, V-4/V-6/V-7 sign-offs, V-5/V-9/V-10 not started).
- 2026-10-01 (Parts 0+1): repo audit found **no RL policy had ever been trained** — `DQNPolicy.train_step` was never called and discarded its legal-action mask (bootstrap optimised toward illegal actions); `PPOPolicy` had no training method at all. Both fixed, replay buffer + training script added, 13 regression tests written. A third alleged defect (terminal next-state shapes breaking `torch.cat`) was investigated and **retracted** — `torch.cat(dim=0)` concatenates along dim 0, so the original code did not raise; do not repeat that claim. Environment rebuilt (CPU-only torch; `/tmp` tmpfs is too small for the CUDA build) and all three datasets restored — two of them as documented conversions. **67/67 tests pass**.
- 2026-10-01 (Parts 2+3): **first real RL measurement.** `scripts/step5_policy_benchmark.py` benchmarks DQN/PPO/Greedy/Random/ExactDP at matched budgets B∈{1..6} with a shared evaluator, plus the V\*−V_emp optimality gap. **Result: greedy-IG attains the exact optimum (gap ≤0.013); DQN (~0.04–0.055) and PPO (0.015–0.138) do not — PPO is worse than random at B=5/B=6.** Negative result for RL on this task, reported as such. 12 benchmark tests added → **79/79 pass**. `POLICY_BENCHMARK_REPORT.md` written. Polish still sealed; 8 verification gates unchanged. **Top next steps: multi-seed variance, then V-6 (λ) — every current number is λ=0, so cost is unpriced and adaptive stopping is unexcused.**
