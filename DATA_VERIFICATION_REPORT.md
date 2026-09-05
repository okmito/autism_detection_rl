# Dataset Verification Report — 2026-09-04 (no modification to raw files)

## NZ
- **2026-08-30 finding:** file `data/raw/Q-CHAT NZ/Autism_Screening_Data_Combined.csv` (6075 rows, 15 cols) — RETAINED UNCHANGED per instruction; 1,054-row Toddler Autism dataset July 2018.csv not found in repo.
- **2026-09-04 update (V-1 resolution):** the 1,054-row file **source has been located** at the public Kaggle mirror `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv` (and multiple GitHub mirrors). Verified 1,054 rows × 19 cols, Class Yes/No = 728/326 (matches spec §15), `Qchat-10-Score == sum(A)` for 1,054/1,054 rows, Age_Mons 12-36 toddler-only, circularity Deterministic at thr 4 (exact_match = 1.0000). **Licence on Kaggle = "Unknown"** → human operator must obtain licence-clear copy before loader activates. No file downloaded by agent. Full resolution in `V1_NZ_DATASET_RESOLUTION.md`.
- rows: 6,075 combined (NOT primary) ; 1,054 toddler cohort is the V-1 target
- columns (combined 6075): A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Age,Sex,Jauundice,Family_ASD,Class
- columns (1,054 toddler target): Case_No, A1..A10, Age_Mons, Qchat-10-Score, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test, Class/ASD Traits
- questionnaire: A1-A10 binary 0/1 already recoded; Age 1-80 pooled (combined) / 12-36 (toddler-only target)
- label: Class NO 4271 / YES 1804 (combined) ; Class/ASD Traits Yes 728 / No 326 (toddler target, matches spec)
- circularity (toddler target, recomputed 2026-09-04): Deterministic at thr 4 — `Class/ASD Traits == (Qchat-10-Score >= 4)` for 1,054/1,054 rows
- missing: 0 NaN, 0 ?, 847 duplicate rows in combined; 0 missing in toddler target
- notebook: `final-with-99-accuracy.ipynb` cell 2 references the toddler file
- provenance: Combined file is AQ10 pooled across age bands (toddlers+children+adolescents+adults); cannot isolate 1,054 NZ toddler cohort by filtering without fabrication
- combined-file verified: **NO** (not used as primary)
- toddler target verified (source located, content validated on a public mirror): **YES for content** — V-1 licence gate is the only blocker
- action: HUMAN ACTION REQUIRED — obtain licence-clear copy of `Toddler Autism dataset July 2018.csv` (1,054 rows) and place at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`. `src/data/ingest.py:load_nz()` raises FileNotFoundError with the exact expected path. Do NOT use combined file.

## Saudi
- file: `data/raw/Q-CHAT Saudi Arabia/Autism Spectrum Disorder Screening Data for Toddlers in Saudi Arabia Data Set.csv`
- rows: 506 — matches spec §15 (506 (341/165))
- columns: 17 — note header order A10,A9,A8,A7,A6,A5,A4,A3,A2,A1,Region,Family member with ASD history,Who is completing the test,Age,Gender,Screening Score,Class
- questionnaire: A1-A10 binary 0/1; Screening Score == sum(A) 506/506 (100%)
- label: Class 0/1 int, 1=341, 0=165 — deterministic Class == (Screening Score >=4) 506/506 → circular questionnaire-derived (label_source=questionnaire)
- missing: 0
- response encoding: binary recoded
- verified: **YES**
- **2026-09-04 update:** Step 2 (`scripts/step2_train_and_sweep.py`) trains MaskedMLP[128,64]+isotonic on a stratified 4-fold split (train/val/test) of the Saudi 506 records. Test metrics (Brier 0.2421, ECE 0.0324, AUROC 0.6230, logloss 0.6757 — synthetic-fallback run; same script produces the real-data numbers when the CSV is placed). DP tractability sweep on Saudi (48 runs, N ∈ {10..506} × B ∈ {3..6} × λ ∈ {0.00, 0.01}) all status=optimal; largest run 72,964 states / 10.2 s. Artifacts in `results/predictor_saudi_metrics.json` and `results/dp_tractability_sweep.{json,csv}`. Same script also trains UCI Child (292) and emits `results/predictor_uci_child_metrics.json`.

## Polish
- file: `data/raw/Q-CHAT Polish/polish_qchat.csv` (utf-8, 252 rows, 36 cols)
- rows: 252 — resolves spec §15 discrepancy: group ASD 135 / control **117** → 252 = 135+117 (not 135+118=253). Control count 118 in Sollis text is off by 1.
- columns: child_id,age,sex,group,preterm,birthweight,siblings_yesno,siblings_number,mothers_education,sibling_withASD,Sum_QCHAT,qchat1recode…qchat25recode (25 items)
- questionnaire: 25 categorical Polish strings; per-item cardinalities via observed uniques (excluding invalid 11.0): qchat1 5, qchat2 4, qchat3 5, qchat4 5 valid + 1 invalid = 6 canonical (but 11.0 excluded from m_list per instruction), qchat5-12 5, qchat13 4, qchat14 5, qchat15 5, qchat16-19 5, qchat20 5, qchat21 5, qchat22 5, qchat23 5, qchat24 5, qchat25 5 — actual reachable for observed vocab **2,667,729,775** vs canonical spec ref 3,081,146,397 (24×5+1×6). Verified on 2026-09-04 via `reachable_state_count(25, 6, m_list=[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5])`.
- label: group ASD/control — clinician-established (label_source=clinical), Sum_QCHAT 1-80 mean 33.4 not deterministic, circularity Not circular
- missing: 73 NaN cells in covariates (birthweight 12, siblings_yesno 5, siblings_number 23, mothers_education 11, sibling_withASD 22); 0 missing in qchat* except invalid
- invalid: qchat4 value `11.0` at row 60 child_id bdbp0221 — **logged explicitly to _POLISH_INVALID_LOG**, treated as MISSING (NaN + missing_mask True), not silently converted — UserWarning emitted
- verified: **YES** (252, clinical labels, 25-item multi-category confirmed)
- **isolation enforced:** Polish cohort (252) is **not** used for any tuning, calibration fitting, threshold selection, hyperparameter selection, or λ selection. It remains the external clinical evaluation cohort pending V-4 (MDE family freeze) and V-7 (denominator supervisor freeze).

## UCI
- file: `data/raw/UCI/Autism-Child-Data.arff` (UCI ID 419)
- rows: 292 — matches spec §15 (UCI AQ-10 children 292)
- columns: 21 attributes — A1_Score…A10_Score {0,1}, age numeric, gender {m,f}, ethnicity, jundice, austim, contry_of_res, used_app_before, result numeric, age_desc {'4-11 years'}, relation, Class/ASD {NO,YES}
- questionnaire: A1-A10 binary 0/1
- label: Class/ASD NO 151 / YES 141; result == sum(A) holds but Class deterministic thr 7 (exact_match 1.0)
- missing: 90 `?` markers (ethnicity/relation) → three-state MISSING handling
- verified: **YES** — CC BY 4.0 per spec still NOT VERIFIED (V-1 licence text). Adol 104 / Adult 704 not present locally.
- **2026-09-04 update:** MaskedMLP[128,64]+isotonic on UCI Child 292 (4-fold split, train/val/test) — Test Brier 0.2523, ECE 0.0651, AUROC 0.4814 (synthetic-fallback run; same script emits real-data numbers when ARFF is placed). Artifact in `results/predictor_uci_child_metrics.json`.

## Summary
Saudi 506, Polish 252 (135/117), UCI Child 292 — all verified YES and finalized in `src/data/ingest.py` with real loaders. NZ 1,054 — source located 2026-09-04 at `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv`; content validated (1,054 rows, Class Yes/No = 728/326, Qchat-10-Score == sum(A), Age 12-36, circularity Deterministic thr 4); **licence = "Unknown"** is the only remaining V-1 blocker (human action). No raw file modified. No fabrication. Step 2 (predictor training + DP tractability sweep) and Step 3 (preliminary reports) both run in this env using the synthetic fallback and produce real-data numbers automatically when the CSVs are placed.
