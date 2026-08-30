# Dataset Verification Report — 2026-08-30 (no modification to raw files)

## NZ
- file: `data/raw/Q-CHAT NZ/Autism_Screening_Data_Combined.csv` (6075 rows, 15 cols) — **RETAINED UNCHANGED per instruction**
- rows: 6075 (NOT 1054)
- columns: A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Age,Sex,Jauundice,Family_ASD,Class
- questionnaire: A1-A10 binary 0/1 already recoded; Age 1-80 pooled (not toddler-only); sex m/f; no Qchat-10-Score/Ethnicity/Who completed test
- label: Class NO 4271 / YES 1804; not deterministically sum>=threshold (thr6 0.823, thr7 0.934)
- missing: 0 NaN, 0 ?, 847 duplicate rows
- notebook: `final-with-99-accuracy.ipynb` cell 2 loads `Toddler Autism dataset July 2018.csv` with columns Case_No/A1-A10/Age_Mons/Qchat-10-Score/Sex/Ethnicity/Jaundice/Family_mem_with_ASD/Who completed the test/Class/ASD Traits — **file not present** under `data/raw/Q-CHAT NZ/` nor anywhere in repo (searched `Toddler Autism dataset July 2018.csv`, `data_csv.csv`, `autism_screening.csv` — 0 hits)
- provenance: Combined file is AQ10 pooled across age bands (toddlers+children+adolescents+adults); cannot isolate 1054 NZ toddler cohort by filtering without fabrication
- verified: **NO**
- action: HUMAN ACTION REQUIRED — NZ primary cohort missing. `src/data/ingest.py:load_nz()` raises FileNotFoundError with expected path `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv` (1054 rows). Do NOT use combined file.

## Saudi
- file: `data/raw/Q-CHAT Saudi Arabia/Autism Spectrum Disorder Screening Data for Toddlers in Saudi Arabia Data Set.csv`
- rows: 506 — matches spec §15 (506 (341/165))
- columns: 17 — note header order A10,A9,A8,A7,A6,A5,A4,A3,A2,A1,Region,Family member with ASD history,Who is completing the test,Age,Gender,Screening Score,Class
- questionnaire: A1-A10 binary 0/1; Screening Score == sum(A) 506/506 (100%)
- label: Class 0/1 int, 1=341, 0=165 — deterministic Class == (Screening Score >=4) 506/506 → circular questionnaire-derived (label_source=questionnaire)
- missing: 0
- response encoding: binary recoded
- verified: **YES**

## Polish
- file: `data/raw/Q-CHAT Polish/polish_qchat.csv` (utf-8, 252 rows, 36 cols)
- rows: 252 — resolves spec §15 discrepancy: group ASD 135 / control **117** → 252 = 135+117 (not 135+118=253). Control count 118 in Sollis text is off by 1.
- columns: child_id,age,sex,group,preterm,birthweight,siblings_yesno,siblings_number,mothers_education,sibling_withASD,Sum_QCHAT,qchat1recode…qchat25recode (25 items)
- questionnaire: 25 categorical Polish strings; per-item cardinalities via observed uniques (excluding invalid 11.0): qchat1 5, qchat2 4, qchat3 5, qchat4 5 valid + 1 invalid =6 canonical, qchat5-12 5, qchat13 4, qchat14 5, qchat15 5, qchat16-19 5, qchat20 5, qchat21 5, qchat22 5, qchat23 5, qchat24 5, qchat25 5 — actual reachable for observed vocab 2667729775 vs canonical spec ref 3081146397 (24×5+1×6)
- label: group ASD/control — clinician-established (label_source=clinical), Sum_QCHAT 1-80 mean 33.4 not deterministic, circularity Not circular
- missing: 73 NaN cells in covariates (birthweight 12, siblings_yesno 5, siblings_number 23, mothers_education 11, sibling_withASD 22); 0 missing in qchat* except invalid
- invalid: qchat4 value `11.0` at row 60 child_id bdbp0221 — **logged explicitly to _POLISH_INVALID_LOG**, treated as MISSING (NaN + missing_mask True), not silently converted — UserWarning emitted
- verified: **YES** (252, clinical labels, 25-item multi-category confirmed)

## UCI
- file: `data/raw/UCI/Autism-Child-Data.arff` (UCI ID 419)
- rows: 292 — matches spec §15 (UCI AQ-10 children 292)
- columns: 21 attributes — A1_Score…A10_Score {0,1}, age numeric, gender {m,f}, ethnicity, jundice, austim, contry_of_res, used_app_before, result numeric, age_desc {'4-11 years'}, relation, Class/ASD {NO,YES}
- questionnaire: A1-A10 binary 0/1
- label: Class/ASD NO 151 / YES 141; result == sum(A) holds but Class deterministic thr 7 (exact_match 1.0)
- missing: 90 `?` markers (ethnicity/relation) → three-state MISSING handling
- verified: **YES** — CC BY 4.0 per spec still NOT VERIFIED (V-1 licence text). Adol 104 / Adult 704 not present locally.

## Summary
Saudi 506, Polish 252 (135/117), UCI Child 292 — all verified YES and finalized in src/data/ingest.py with real loaders. NZ 1,054 — HUMAN ACTION REQUIRED (Toddler Autism dataset July 2018.csv missing; combined 6075 retained but not used). No raw file modified. No fabrication.
