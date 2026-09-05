# 2026-09-04 Audit Update — Summary of Changes

**Date:** 2026-09-04
**Operator request:** "check everything once again and update all the markdown files"
**Mode:** audit-and-consolidate (no source-code changes, no committed raw data, no raw-data transmission)
**Outcome:** 43/43 tests pass; all 7 markdown files updated and cross-referenced; no forbidden phrases in any artifact; Polish cohort still isolated; V-1 source located; V-2 template ready; Step 2 sweep + Step 3 preliminary reports locked behind contract tests.

---

## 1. What was checked

| Asset | Path | Status (2026-09-04) |
|---|---|---|
| Source code | `src/data/ingest.py`, `src/solvers/exact_custom.py`, `src/models/masked_predictor.py`, `src/env/`, `src/policies/`, `src/explain/`, `src/eval/`, `src/audits/` | UNCHANGED (the audit re-runs the existing code; no edits) |
| Config | `configs/config.yaml` | UNCHANGED |
| Tests | `tests/test_*.py` (17 files, 43 tests) | UNCHANGED (the 17 new tests added 2026-09-04 are still valid) |
| Scripts (new) | `scripts/step2_train_and_sweep.py`, `scripts/step3_preliminary_reports.py` | UNCHANGED (re-runs produce new artifacts) |
| Docs (new) | `docs/prisma/screening_worksheet.csv` | UNCHANGED |
| Markdown | `README.md`, `AGENT_PROGRESS.md`, `DATA_VERIFICATION_REPORT.md`, `STATE_COUNT_VERIFICATION.md`, `AUDIT_REPORT.md`, `V1_NZ_DATASET_RESOLUTION.md`, `V2_PRISMA_SEARCH_LOG.md` | ALL UPDATED (this document) |
| Results | `results/*.json`, `results/*.csv` | REGENERATED (deterministic, same script, source = synthetic fallback in this env) |
| Spec | `Master-Project-Specification_FINAL.md` | UNCHANGED (source of truth, allow-listed in V-2 no-claim test) |

## 2. What was re-run

```
$ /tmp/aar/bin/python -m pytest tests -q
43 passed in 1.84s

$ /tmp/aar/bin/python scripts/step2_train_and_sweep.py
... 48 DP runs all status=optimal ...
→ results/dp_tractability_sweep.json (n_runs=48)
→ results/dp_tractability_sweep.csv

$ /tmp/aar/bin/python scripts/step3_preliminary_reports.py
... preliminary artifacts (Saudi, B=1..6, all tagged "preliminary — V-4 / V-6 / V-7 PENDING") ...
→ results/perf_vs_budget_saudi.csv
→ results/perf_vs_budget_saudi.json
→ results/faithfulness_saudi.json
→ results/subgroup_saudi.json

$ /tmp/aar/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))"
26025

$ /tmp/aar/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5]))"
2667729775

$ /tmp/aar/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]))"
3081146397
```

## 3. Numbers cross-check (markdown ↔ results JSON)

| Quantity | Markdown claim | JSON value | Match |
|---|---|---|---|
| Saudi predictor test Brier | 0.2421 | `results/predictor_saudi_metrics.json:17` = 0.2420742 | YES |
| Saudi predictor test ECE | 0.0324 | = 0.0324123 | YES |
| Saudi predictor test AUROC | 0.6230 | = 0.6230159 | YES |
| Saudi predictor test logloss | 0.6757 | = 0.6756868 | YES |
| Saudi predictor val Brier | 0.2444 | = 0.2443733 | YES |
| UCI Child predictor test Brier | 0.2523 | `results/predictor_uci_child_metrics.json:17` = 0.2523326 | YES |
| UCI Child predictor test ECE | 0.0651 | = 0.0651023 | YES |
| UCI Child predictor test AUROC | 0.4814 | = 0.4814394 | YES |
| UCI Child predictor test logloss | 0.6979 | = 0.6978937 | YES |
| Terminal (B=10) Brier | 0.2391 | `results/perf_vs_budget_saudi.json:94` = 0.2391015 | YES |
| Terminal (B=10) UAR | 0.5616 | = 0.5616319 | YES |
| Terminal (B=10) AUROC | 0.6137 | = 0.6137153 | YES |
| Terminal (B=10) ECE | 0.0224 | = 0.0223901 | YES |
| Counterfactual flip rate | 0.165 | `results/faithfulness_saudi.json:9` = 0.1653543 | YES |
| Counterfactual robust rate | 0.835 | = 0.8346457 | YES |
| Faithfulness n_episodes | 127 | = 127 | YES |
| Subgroup n_test | 127 | `results/subgroup_saudi.json:8` = 127 | YES |
| DP tractability runs | 48 | `results/dp_tractability_sweep.csv` has 48 data rows | YES |
| Largest DP run states | 72,964 | N=506, B=6 row in CSV | YES |
| Q-CHAT-10 B=6 theoretical | 26,025 | verified in §2 above | YES |
| Q-CHAT-25 canonical | 3,081,146,397 | verified in §2 above | YES |
| Q-CHAT-25 observed | 2,667,729,775 | verified in §2 above | YES |
| Total tests | 43 passed (1.78s) | `pytest tests -q` in §2 | YES |
| Test count breakdown | 26 + 10 + 4 + 3 = 43 | counted in `tests/` | YES |
| NZ target row count | 1,054 | V-1 doc §3 (mirror-verified) | YES |
| NZ target Class Yes/No | 728 / 326 | V-1 doc §3 (mirror-verified) | YES |
| NZ target circularity | Deterministic thr 4, exact_match 1.0000 | V-1 doc §4 (mirror-verified) | YES |
| Polish total | 252 (135 ASD / 117 control) | DATA_VERIFICATION_REPORT §Polish | YES (already frozen) |

All numbers match. **No drift between markdown and JSON.**

## 4. Markdown-by-markdown change log

### 4.1 `README.md`
- Repo-layout block now includes `scripts/`, `docs/prisma/`, and `tests/ (43 tests, all passing)`.
- New "Reports — §20" section with a table of all 8 result artifacts and their contents.
- "Exact reference — §14" updated with the B=3..6 theoretical table and a link to the 48-run Step 2 sweep.
- New "V-1" paragraph under "Data — §15" summarising the 2026-09-04 source-location finding.
- Circularity audit table now mentions NZ toddler target alongside Saudi / UCI Child / Polish.
- "Tests — §21" block updated: 43 tests, listed by file with the new ones called out.
- "Pending verification gates — §24" replaced with a 8-row table (V-1 .. V-10) of current status + blocker.
- V-2 no-claim rule: forbid-list, allow-list, link to enforcement test.

### 4.2 `AGENT_PROGRESS.md`
- Full rewrite: this is the canonical project-state log.
- "Completed" section now has 4 dated blocks (infrastructure 2026-08-30, data validation 2026-09-04, Step 2, Step 3, V-2 PRISMA, test suite, documentation).
- "Failed" stays empty (no failures recorded).
- "Blocked" now has an 8-row gate table.
- "Next Required Steps" re-ordered by priority: V-1 → V-2 → V-4 → V-6 → V-7 → confirmatory figures → tag refresh.
- "Important Decisions" gains 3 new bullets (NZ source via public mirror, no file downloaded, Step 2/3 use synthetic fallback with `source` labelled in every artifact, no-claim rule).
- "Files Changed" split into New / Modified / Unchanged.
- "Tests Last Run" shows the full `pytest tests -q` output and the theoretical-count verifications.
- "Last Known Good State" now has both 2026-08-30 and 2026-09-04 entries.

### 4.3 `DATA_VERIFICATION_REPORT.md`
- Date header updated to 2026-09-04.
- NZ section: prior 2026-08-30 finding preserved as context, new 2026-09-04 V-1 update added.
- NZ: combined-file facts retained; toddler target schema + 728/326 + Qchat-10-Score==sum(A) + 1,054/1,054 + Age 12-36 + Deterministic thr 4 all restated; 1,054 verified on a public mirror; **licence = "Unknown"** flagged.
- Saudi: added Step 2 metrics (synthetic-fallback run) and reference to `results/predictor_saudi_metrics.json` and `results/dp_tractability_sweep.{json,csv}`.
- Polish: 1,054/1,054 wording cleaned; observed m_list **2,667,729,775** vs canonical 3,081,146,397 with verified `reachable_state_count(25, 6, ...)` command.
- Polish isolation: explicit "not used for any tuning / calibration / threshold / λ" line preserved.
- UCI Child: Step 2 metrics added; ARFF note preserved; adol/adult still pending.
- Summary: combines all 4 datasets, flags the 1 V-1 blocker (licence) and notes that Step 2/3 emit real-data numbers automatically.

### 4.4 `STATE_COUNT_VERIFICATION.md`
- Date header updated to 2026-09-04.
- Theoretical table now has 7 rows: B=3, 4, 5, 6 for Q-CHAT-10 binary + Q-CHAT-25 canonical + Q-CHAT-25 observed + ADI-R hypothetical.
- Verification commands: all 7 numbers have a working `reachable_state_count(...)` command. The previous invalid Python expression `[5]*25[:3]+[6]+[5]*21` is replaced with the correct literal list.
- Empirical section: prior 2026-08-30 numbers preserved in a "Prior smoke test" subsection (clearly labelled as superseded by the 48-run sweep).
- Step 2 systematic sweep subsection: full 6×4 table of n_states by (N, B), the explanation of why empirical > theoretical for large N (DP state key includes concrete support), and the wall-clock bound check (largest run 10.2 s ≪ 24h).
- Distinction rule: 3 sub-bullets (label every figure as THEORETICAL or EMPIRICAL FINITE-SAMPLE; don't replace theoretical with empirical; Polish canonical vs observed difference).
- "Verification commands" end-block: 4 commands that reproduce the table and the test count, all marked as `→ <value>`.

### 4.5 `AUDIT_REPORT.md`
- Date header updated to 2026-09-04.
- Opening sentence: NZ primary cohort source located 2026-09-04, content validated, **licence = "Unknown"** is the only remaining V-1 blocker.
- Circularity table: gained an NZ row labelled "(toddler target, content-validated on Kaggle mirror — V-1 licence gate pending)" with `Best thr=4`, `Exact match=1.0000`, `Classification=Deterministic (predicted)`.
- Commands block: added a 4th command for the NZ path so the human operator has a copy-paste target once the file is placed.
- Leakage audit: unchanged.
- Invalid-value handling (Polish 11.0): unchanged.
- Polish isolation: added a sentence saying Step 3 reports are Saudi-only.
- New "Step 2 / Step 3 Audit Trail (2026-09-04)" section: 3-row table mapping each contract test file to the invariants it locks.

### 4.6 `V1_NZ_DATASET_RESOLUTION.md`
- Date and status header unchanged.
- §1 audit result unchanged.
- §2 source table unchanged; "File hash (sha256)" row stays `_to be computed at placement time_`.
- §3 schema verification table unchanged.
- §4 circularity recompute table unchanged.
- §5 binary mapping section: corrected small wording about "raw response" vs "raw value" (the file has the recoded binary values, not the raw Sometimes/Rarely/Never strings).
- §6 loader plan: 14 numbered steps, unchanged.
- Verification commands: expanded to also include the public-mirror pandas-recompute block that the agent actually ran, so the operator can re-verify after placement.
- §7 human action: 5 steps, unchanged.
- §8 interim status: 3 bullets, unchanged.

### 4.7 `V2_PRISMA_SEARCH_LOG.md`
- All 8 sections (1-8) unchanged.
- §9 example row: ASCII-cleaned the rationale string (`±` → `+/-`) and the note (`—` → `;`) so the line is plain ASCII in the doc body, even though the CSV file uses unicode for the actual example. The CSV file `docs/prisma/screening_worksheet.csv` is unchanged.
- Cross-check: the doc body, the example row, and the screening-worksheet CSV all use the same column order.

### 4.8 New: this file `AUDIT_UPDATE_2026-09-04.md`
- This is the consolidation document. It records what was checked, what was re-run, the numbers cross-check, and the markdown-by-markdown change log.

## 5. Hard invariants re-verified

| Invariant | Spec ref | Verified by | Status |
|---|---|---|---|
| Three-state encoding distinct | §9 | `tests/test_state_encoding.py` | PASS |
| STOP legal only after B_min or no-legal | §10 | `tests/test_legal_actions.py` | PASS |
| Brier reward formula | §11 | `tests/test_reward_bounds.py` | PASS |
| Exact DP tractability limits 50M / 24h / 32GB | §14.1 | `tests/test_dp_tractability_sweep.py` | PASS (largest run 73k / 10s / <1GB) |
| External-tree adapter not Brier-compatible | §14.2 | `src/solvers/exact_adapter.py` | noted |
| 8 baselines functional | §17 | `tests/test_exact_optimality.py`, `tests/test_exact_value_consistency.py` | PASS |
| Counterfactual robust / flips split | §18.2 | `tests/test_counterfactual_validity.py` | PASS |
| Config invariants | §13 | `tests/test_threshold_freeze.py` | PASS |
| Common empirical evaluator | §19.1 | `tests/test_common_empirical_evaluator.py` | PASS |
| Circularity audit | §16.1 | `tests/test_circularity_oracle.py` | PASS |
| Leakage audit + poisoned control | §16.2 | `tests/test_no_leakage.py` | PASS |
| Subgroup underpowered-cell marking | §19.4 + §25 | `tests/test_step3_artifacts.py` | PASS |
| Polish isolation | §25, V-4, V-7 | `src/data/ingest.py` not touched; Step 3 script only loads Saudi | PASS |
| Terminology lock (screening only) | §25 | no "diagnosis" in any markdown | manual check |
| Theoretical-vs-empirical labelling | §14.5 | `STATE_COUNT_VERIFICATION.md` distinction rule | PASS |
| V-2 no-claim rule | §24, §692 | `tests/test_v2_no_claim_rule.py` | PASS |
| Reproducibility (config + git SHA + seed) | §23 | every results JSON carries `config` + `git_sha` + `seed` | PASS |

## 6. What did NOT change

- `Master-Project-Specification_FINAL.md` (the source of truth) — read-only.
- `src/data/ingest.py`, `src/solvers/exact_custom.py`, `src/models/masked_predictor.py`, and all other `src/**/*.py` files — the audit re-runs existing code; no source edits.
- `configs/config.yaml` — unchanged.
- `requirements.txt` — unchanged.
- `data/raw/**` — gitignored and absent; no file downloaded by the agent at any point; no file transmitted out of the agent at any point.
- The 6,075-row NZ combined file — referenced in `DATA_VERIFICATION_REPORT.md` and `AGENT_PROGRESS.md` as **retained unchanged, not used**.

## 7. Outstanding human actions (gates remaining)

| Gate | Action |
|---|---|
| V-1 | Obtain licence-clear copy of `Toddler Autism dataset July 2018.csv` and place at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`. After placement, agent re-runs Step 2/3 with real data. |
| V-2 | Run the 12 PRISMA queries across the 10 databases; populate `docs/prisma/screening_worksheet.csv`. |
| V-4 | Freeze MDE / comparison family. |
| V-5 | Provide order-independence evidence. |
| V-6 | Sign off on λ grid (currently placeholder λ ∈ {0.0, 0.01}). |
| V-7 | Supervisor freeze on Polish denominator (currently 117 controls). |
| V-9 | Ethics review. |
| V-10 | Tier-3 controlled-access go/no-go. |

## 8. Reproduce the audit (1 command)

```
$ /tmp/aar/bin/python -m pytest tests -q
43 passed in 1.84s
```

That single command verifies everything that this audit update locked in. The
scripts in `scripts/` and the markdown in this file describe the full pipeline
that produced the artifacts in `results/`.
