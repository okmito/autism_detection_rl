# POLISH EXTERNAL VALIDATION — SIGN-OFF PACKAGE

**For:** project supervisor / mentor
**Date:** 2026-10-02
**Overall status: BLOCKED — awaiting three supervisor decisions.**
**Claim boundary: research prototype. Not a diagnostic device. Not clinical validation.**

---

## 1. Research question

Does an adaptive-questionnaire-selection model trained on one cohort and frozen
retain its discrimination on an independently labelled cohort from a different
country and a different (longer) instrument?

Specifically: does the model, selecting up to a small budget of Q-CHAT questions,
achieve held-out predictive performance on the Polish cohort that supports H1
(adaptive selection matches exhaustive subset search at a given budget) and H2
(no material transfer gap between development and external cohorts)?

---

## 2. Why Saudi is the development data

- It is the only cohort whose labels are **questionnaire-derived**, which is exactly
  the regime the model was designed for.
- The model was trained, calibrated and thresholded **only** on Saudi. The split is
  stratified 4-fold with a second fold held for validation: 284 train / 95 val /
  127 test from 506 records.
- The label source is recorded in the artifact as `questionnaire`.

**Known and deliberate weakness.** A circularity audit confirms the Saudi
questionnaire label is **exactly reproducible** from the items
(`exact_match_rate = 1.0000`, deterministic). The development labels are therefore
**not independent** of the questionnaire. This is a property of the source dataset,
not a modelling choice, and it is why Saudi metrics alone cannot support any claim
of generalisation.

---

## 3. Why Polish is the external validation cohort

- Its labels are **clinician-established**, not questionnaire-derived.
- The circularity audit gives `exact_match_rate = 0.5437` — the clinical `GROUP` is
  **not** reproducible as a threshold over the questionnaire items. That contrast
  against Saudi is the whole point: Polish is a genuinely independent target.
- It comes from a different country and a different instrument edition
  (Q-CHAT-25, 25 items, ordinal), which tests transportability rather than
  memorisation.

---

## 4. Why the datasets are not merged

1. **Label semantics differ.** Saudi labels are questionnaire-derived; Polish labels
   are clinical. Pooling them would create a single target that is neither, and the
   model could learn the *source* rather than the condition.
2. **A Polish contribution to training would destroy the evaluation.** The external
   value of this cohort is that nothing was fitted on it. Any use in fitting,
   calibration, threshold tuning or RL training spends it permanently.
3. **Cohort size.** 252 Polish rows against 506 Saudi rows would dominate a naive
   concatenation and quietly reweight every metric.
4. **Circularity leakage.** Saudi labels being questionnaire-derived means a joint
   fit rewards questionnaire reconstruction — the opposite of the intended test.
5. **Governance.** The Polish cohort is **sealed**. It was not used for any fitting
   decision in this project and must not be.

---

## 5. Exact Q-CHAT-10 mapping

| model feature | Saudi column | Q-CHAT-10 | Q-CHAT-25 | Polish variable |
|---|---|---|---|---|
| feature_1 | A1 | Q1 | Q1 | `qchat1recode` |
| feature_2 | A2 | Q2 | Q2 | `qchat2recode` |
| feature_3 | A3 | Q3 | Q5 | `qchat5recode` |
| feature_4 | A4 | Q4 | Q6 | `qchat6recode` |
| feature_5 | A5 | Q5 | Q9 | `qchat9recode` |
| feature_6 | A6 | Q6 | Q10 | `qchat10recode` |
| feature_7 | A7 | Q7 | Q15 | `qchat15recode` |
| feature_8 | A8 | Q8 | Q17 | `qchat17recode` |
| feature_9 | A9 | Q9 | Q19 | `qchat19recode` |
| feature_10 | A10 | Q10 | Q25 | `qchat25recode` |

Versions: `qchat10-mapping/2.0.0`, `qchat10-contract/2.0.0`.
Fifteen Q-CHAT-25 items are unused.

---

## 6. Exact binary projection

Each response resolves through its **own** SPSS value label, not by code magnitude:

```
code -> Polish label -> English wording printed in the instrument
     -> printed letter A-E -> official Q-CHAT-10 direction
```

- Items Q1-Q9: letters **C/D/E** score 1.
- Item Q10: letters **A/B/C** score 1.

Result: `code >= 2` for all ten items — **verified as a consequence**, not assumed.
`verify_split_rule()` recomputes both paths and raises on disagreement.

Two items required specific care:

- **`qchat2recode`** is coded `0,1,2,3,5`; code 4 never occurs and is refused.
- **`qchat25recode`** is reverse-coded (`0 = never`, `4 = many times a day`), so
  more frequent staring carries a higher code and correctly scores 1.

**Information loss: five ordinal levels collapse to two; at least three levels per
item are discarded, irreversibly.**

---

## 7. 41-dimensional compatibility

The cohort projects to `(252, 10)` — exactly 10 binary positions, values `{0.0, 1.0}`,
**zero** NaN, no padding, no unrelated item, atypical rates 0.27-0.65.

Those 10 features encode to the frozen interface as:

```
3n mask one-hot (30) + n response bits (10) + 1 normalised budget term  =  41
```

using the project's own `init_state`/`update_state` convention with binary
`m_list=None` — the same `4n + 1` form used at training time. Both frozen artifacts
accept the encoding in a forward pass. **No metric was computed.**

The model's architecture is untouched: `hidden = [128, 64]`, no retraining, no
new layer, no threshold change.

---

## 8. V-4 status

```
AUTOMATED EVIDENCE = PASS
HUMAN SIGN-OFF     = OPEN
```

Five confirmatory comparisons frozen (H1 at budgets 3-6, plus H2.transfer), primary
endpoint held-out Brier, Holm-Bonferroni at family alpha 0.05, n = 252, power 0.80.
MDE ranges from about **0.09** (SD 0.50) to **0.43** (SD 2.00) Brier units depending
on the assumed paired-difference SD, which **cannot be measured** because it depends
on the effect being sought.

**Decision needed:** ratify which assumed SD the MDE is conditional on.
Full detail: `docs/V4_SIGNOFF_PACKAGE.md`.

---

## 9. V-7 status

```
AUTOMATED DENOMINATOR = PASS
AUTOMATED BASELINE 10 = OPEN
HUMAN SIGN-OFF        = OPEN
```

Observed in the data: **135 ASD + 117 control = 252**, corroborated at 252/252
participant identifiers with 0 duplicates.
Published in the source text: **135 + 118 = 253**, contradicting its own stated total
of 252. Recorded as an unresolved textual inconsistency; **no correction to the
publication is asserted.**

Baseline 10 requires reimplementing the **Sollis et al.** model. That publication's
specification is not in this repository. **No substitute comparator was created** —
implementing from a paraphrase would fabricate a baseline indistinguishable from a
real one.

**Decisions needed:** ratify n = 252; confirm the 118 figure is recorded, not
corrected; direct whether to request the Sollis specification or publish with
Baseline 10 marked unreproduced.
Full detail: `docs/V7_SIGNOFF_PACKAGE.md`.

---

## 10. Remaining decisions

| # | Gate | Automated | Human | Package |
|---|---|---|---|---|
| 1 | V-4 — MDE assumption | PASS | **OPEN** | `docs/V4_SIGNOFF_PACKAGE.md` |
| 2 | V-7 — denominator + Baseline 10 | PASS / OPEN | **OPEN** | `docs/V7_SIGNOFF_PACKAGE.md` |
| 3 | Q-CHAT-10 projection | PASS | **OPEN** | `docs/QCHAT10_PROJECTION_APPROVAL.md` |

Machine-readable status lives at
`results/v4_v7_validation_status.json` -> `gates.external_validation`:

```json
"required_conditions": {
  "v4_human_sign_off":           "OPEN",
  "v7_human_sign_off":           "OPEN",
  "qchat10_projection_approval": "OPEN"
},
"all_conditions_closed": false
```

**All three must be closed before the cohort may be opened.** `OPEN` is never
converted to `PASS` by any automated process.

---

## 11. What happens after approval

Once **all three** conditions read `PASS`:

1. `scripts/step9_v4_v7_gates.py` re-runs and records the signed states. The gate
   artifact is **not** edited by hand; the decision is entered in the gate source and
   the script regenerates it.
2. `scripts/step10_external_validation.py` re-runs. Its gate check passes all three
   conditions, then: load frozen artefacts → assert no fitting method is exposed →
   validate cohort schema and target → leakage audit → project `(252, 10)` → encode
   to 41 dimensions → **predict once** → report metrics with bootstrap CIs.
3. Metrics land in `results/polish_external_validation.json`, metadata only, **no
   participant rows**.
4. The frozen artefacts are verified unchanged by hash before and after.
5. No parameter is ever fitted on Polish data. Calibration stays the frozen
   isotonic artefact.

**Until then: Step 10 exits 2 with `metrics: null`, and no prediction exists.**

---

## Reproducibility

| item | value |
|---|---|
| git SHA | `8f0fb1059dab77bc94128729ba8cc73be5095dc7` (branch `main`) |
| predictor weights | `results/predictor_saudi_v2_isotonic.pt` |
| weights sha256 | `2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4` |
| calibrator | `results/predictor_saudi_v2_isotonic.pkl` |
| calibrator sha256 | `a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863` |
| calibration method | **isotonic** (authoritative for research) |
| demo cache | `demo_model_saudi_seed0_platt_v2.pt` — **DEMO ONLY**, Platt, never a substitute |
| split fingerprint | `b2021998a83e7224` (`stratified-4fold-2nd-fold-val`) |
| integrated Polish CSV sha256 | `6c08970053e63ad9748527ef7fca904b360a068d71be7725651aa46fc6b3dafd` |
| SPSS source sha256 | `7fed516fc4e615f4750d0e44c1eea538bbdcdee989e8c2ff8de719976fa08a41` |
| cohort identity | `same_cohort` — SAV and CSV are the same 252 participants |
| feature-contract version | `qchat10-contract/2.0.0` |
| Q-CHAT mapping version | `qchat10-mapping/2.0.0` |

Full machine-readable record:
`results/pre_validation_reproducibility.json`.

---

MANDATORY CAVEAT — this is a research prototype. Agreement between a research
prototype and an independently labelled cohort is **not** a diagnosis, **not**
clinical validation, and **not** a basis for clinical deployment. No result from
this work should be used to assess an individual child.