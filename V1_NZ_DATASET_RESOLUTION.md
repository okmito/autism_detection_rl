# V-1 — NZ 1,054-row Toddler Autism Dataset Resolution

**Date:** 2026-09-04
**Status:** FILE LOCATED — pending human action for licence verification and formal placement
**Gate:** Spec §15, §24, §25 (V-1 blocks Checkpoint 1)

## 1. Audit result

`data/raw/` is gitignored (correct per spec §8) and **not present** in this working tree.

> **Re-checked 2026-10-01:** V-1 remains **unresolved**. The other three datasets were
> re-fetched and verified on that date (see `DATA_VERIFICATION_REPORT.md`), but the NZ
> 1,054-row toddler file was **still not obtained** — the Kaggle licence remains "Unknown"
> and no file was downloaded by the agent. Nothing in the §3–§8 analysis below has changed.
> Once a licence-clear copy is placed at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`,
> Step 2/3/5 can be re-run on the real NZ cohort.
Prior searches in `AGENT_PROGRESS.md` and `DATA_VERIFICATION_REPORT.md` concluded the
1,054-row file is missing. That conclusion was incomplete — see §2.

A re-search across public mirrors **does** identify the file.

## 2. Source identified

| Field | Value |
|---|---|
| Filename | `Toddler Autism dataset July 2018.csv` |
| Row count | **1,054** (verified) |
| Source mirror (primary, human-action) | `https://www.kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv` (Kaggle dataset id 5499687, uploader Mamoona Mushtaq, modified 2024-08-05, 10,287 bytes) |
| Alternative public mirrors | `github.com/NiharikaRajaram/Autism-detection`, `github.com/MarieNav/Data-Analytics_Toddler-Autism`, `github.com/Priyal2807/Classification-of-Autism-Data`, `github.com/SavarToteja/Autism-in-Toddlers`, `github.com/whytheevanssoftware/Machine-Learning-Compendium` |
| Original research source | Fadi Thabtah (same author as UCI IDs 419, 420, 426); the toddler cohort was not directly deposited to UCI but is the "Toddler" arm of Thabtah's AQ-10 deployment |
| Licence (per Kaggle metadata) | **"Unknown"** — declared as `{"@type":"CreativeWork","name":"Unknown","url":""}` in the JSON-LD |
| File hash (sha256) | _to be computed at placement time and recorded here_ |

### Licence risk — explicit flag

V-1 explicitly requires recording the **actual access/licence terms** for the data
artifact. The Kaggle metadata declares the licence as `Unknown`. This is **not
satisfying V-1** on its own. Two paths to resolve:

1. **Best path (preferred).** Human contacts the original investigator (Fadi Thabtah)
   or the kaggle uploader to obtain written permission / a CC licence, OR obtains the
   file from a mirror that carries a clearer licence statement, OR
2. **Document the limitation** in `DATA_VERIFICATION_REPORT.md` and continue with the
   same synthetic-mode + Saudi/UCI + Polish strategy already in place, with NZ
   listed as `PENDING V-1` until a licence-clear source is obtained.

No file has been downloaded by the agent. The agent will not download, commit, or
otherwise transmit the file. The file must be placed by the human operator at
`data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv` and the loader will
only then activate.

## 3. Schema verification (1,054 rows, programmatic)

Headers (19 columns, exactly as expected):

```
Case_No, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
Age_Mons, Qchat-10-Score, Sex, Ethnicity, Jaundice,
Family_mem_with_ASD, Who completed the test, Class/ASD Traits
```

Note the trailing space in `Class/ASD Traits ` — loader must strip before grouping.

| Property | Value | Notes |
|---|---|---|
| Rows | 1,054 | matches spec §15 |
| Class Yes / No | 728 / 326 | matches spec §15 (728/326) |
| `Qchat-10-Score == sum(A1..A10)` | True (1,054/1,054) | per-row recomputed |
| A1..A10 values | binary {0,1} only | no `?` / NaN |
| Missing values per column | 0 everywhere | |
| Age_Mons range | 12..36 | toddler-only, no out-of-range |
| Sex | m=735, f=319 | |
| Ethnicity | 11 categories (White European, asian, middle eastern, south asian, black, Hispanic, Others, Latino, mixed, Pacifica, Native Indian) | |
| Jaundice | no=766, yes=288 | |
| Family_mem_with_ASD | no=884, yes=170 | |
| Who completed the test | family member=1018, Health Care Professional=24, **Health care professional=5**, Self=4, Others=3 | **case-inconsistency to normalize** |
| Class/ASD Traits | Yes=728, No=326 (after strip) | matches spec exactly |

## 4. Circularity audit (already known to be deterministic; recomputed)

Sum-threshold oracle over `Qchat-10-Score` against `Class/ASD Traits` Yes/No:

| Threshold | Exact match |
|---:|---:|
| 0 | 0.6907 |
| 1 | 0.7419 |
| 2 | 0.8254 |
| 3 | 0.9089 |
| **4** | **1.0000** |
| 5 | 0.8956 |
| 6 | 0.7818 |
| 7 | 0.6907 |
| 8 | 0.5626 |
| 9 | 0.4706 |
| 10 | 0.3805 |

→ **Deterministic** at threshold 4. `Class/ASD Traits == (Qchat-10-Score >= 4)` for
1,054/1,054 rows. Matches the prior progress doc expectation. NZ cohort is
**questionnaire-derived**, label_source = `questionnaire`, and §16.1 gate applies
(deterministically circular, may be used for policy/algorithm experiments with
circularity status displayed).

## 5. Q-CHAT-10 binary mapping — declared primary representation

Spec §15 defines a primary binary Q-CHAT-10 representation with the explicit
rule: Q1–9 Sometimes/Rarely/Never→1, Always/Usually→0 ; Q10 Always/Usually/Sometimes→1,
Rarely/Never→0.

This file already contains the **recoded binary** values (A1..A10 ∈ {0,1}). The
raw "Sometimes/Rarely/Never/Always/Usually" string form is **not present** in the
downloaded CSV (it is only on the Saudi instrument).

Implication for the loader:

* The published `A1..A10` columns are already in the binary representation that
  `qchat10_binary_map()` would emit for the Saudi raw form. Loader must pass them
  through as-is and **preserve the original integers** under a `raw_qchat` /
  `raw_screening_score` field for audit (per spec §15 auditability rule).
* `Qchat-10-Score` is the sum that would define the questionnaire label — it must
  **not** be passed to the model as a feature (per spec §15 "Primary Q-CHAT-10
  representation lock").
* The 1,054-row cohort has `Sex` not in `M`/`F` form but in `m`/`f` (lowercase);
  loader must normalize to `M`/`F` for the common `covariates.sex` schema field.

## 6. Loader implementation plan (ready, not yet activated)

Path: `src/data/ingest.py:_load_nz_toddler_csv`

```text
1. Read CSV (UTF-8) with header; columns as in §3.
2. Strip whitespace from Class/ASD Traits; map {Yes→1, No→0}.
3. Normalize Sex: {m→M, f→F}.
4. Normalize Who completed the test:
     "Health care professional" and "Health Care Professional" → "Health Care Professional"
     (other categories left as-is; "Others" / "Self" preserved).
5. item_responses := A1..A10 as float64 (no NaN — file is complete).
6. missing_mask := all-False length-10 array.
7. label_source := "questionnaire"   # deterministic circular per §4
8. label := 1 if Class/ASD Traits == "Yes" else 0
9. covariates := {"age_band": "<toddler>", "sex": normalized, "age_months": int(Age_Mons)}
10. raw_screening_score := int(Qchat-10-Score)         # for audit, not model feature
11. raw_qchat := A1..A10.tolist()                      # already binary
12. provenance := "kaggle:mamoonamushtaq/toddler-autism-dataset-july-2018-csv:1054"
13. Emit UserWarning if file-row count != 1054 (defensive).
14. Do NOT load Qchat-10-Score into any model feature; gate by inspection in
    `__init__`/`__repr__` if a debug-only export is added.
```

Verification commands (planned, not run by agent on the live repo since the file is gitignored and absent):

```
# After V-1 licence gate clears and the file is placed at the expected path:
python3 -c "from src.data.ingest import load_dataset; r=load_dataset('nz'); \
  print(len(r), r[0]['item_responses'], r[0]['label'], r[0]['label_source'])"
python3 -c "from src.audits.circularity import audit_circularity; \
  from src.data.ingest import load_dataset; print(audit_circularity(load_dataset('nz')))"
# expect: 1054, [0,0,0,0,0,0,1,1,0,1], 0, 'questionnaire'  # row 1
# expect:  Deterministic thr 4  # exact_match = 1.0000

# Recompute (already done by agent on a public mirror — for audit, not training):
python3 -c "
import pandas as pd
df = pd.read_csv('Toddler Autism dataset July 2018.csv')
print('rows:', len(df), 'cols:', list(df.columns))
print('Class counts:', df['Class/ASD Traits '].str.strip().value_counts().to_dict())
print('Qchat-10-Score == sum(A)?', (df['Qchat-10-Score'] == df[[f'A{i}' for i in range(1,11)]].sum(axis=1)).all())
print('Age range:', df['Age_Mons'].min(), '..', df['Age_Mons'].max())
"
# expect: rows: 1054, cols: 19
# expect: Class counts: {'Yes': 728, 'No': 326}
# expect: True
# expect: 12 .. 36
```

## 7. Required human action

1. **Obtain the file** with documented licence (preferred paths in §2).
2. **Place it** at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv` (the
   exact expected path per `src/data/ingest.py:NZ_TODDLER_CSV`).
3. **Record licence and SHA-256** in this file (§2 row updated) and in
   `DATA_VERIFICATION_REPORT.md` §NZ section.
4. **Confirm the human-action task** in the project log: AGENT_PROGRESS.md
   "Blocked" list — V-1 entry moved to "Completed" with a one-line commit.
5. After (1)–(4), agent runs the verification commands in §6 and updates
   `AUDIT_REPORT.md` to add an NZ row to the circularity table.

## 8. Interim status (no file present)

* `load_nz(synthetic=True)` continues to return the 1,054-record synthetic
  fallback for tests (reproducible, seed=1). It is not used for any reported
  number in `DATA_VERIFICATION_REPORT.md`, `AUDIT_REPORT.md`, or
  `STATE_COUNT_VERIFICATION.md`.
* `load_nz(synthetic=False)` raises `FileNotFoundError` with the exact
  expected-path string and the 6,075-pooled-file warning. This is the
  behaviour verified in `AGENT_PROGRESS.md` and is the **correct, locked
  default** until §7 step (2) completes.
* The 6,075-row pooled `Autism_Screening_Data_Combined.csv` is **not** used
  for any reported result and is **not** modified.
