# PRIMARY EXTERNAL VALIDATION — RESULTS

**Run date:** 2026-10-02
**Decision basis:** supervisor research decisions of 2026-10-02 (DECISIONS 1–9)
**Claim boundary: research prototype. External validation against
clinician-established ASD labels. NOT a diagnosis. NOT a diagnostic device.**

---

## Headline finding — read this first

**Discrimination transfers, but the model adds no value and is miscalibrated.**

| finding | value |
|---|---|
| Frozen model AUROC | **0.896** (95% CI 0.856–0.931) |
| Trivial item-count reference AUROC | **0.937** |
| Model minus reference | **−0.041** |
| Calibration slope | **0.167** (ideal 1.0) |
| ECE (10 bin) | **0.137** |

The frozen model **does not outperform simply counting atypical answers.** It is
worse by 0.041 AUROC. Both are far above chance, so the *instrument* carries real
signal — but that signal is the questionnaire, not the trained network.

Calibration is poor. The isotonic calibrator has collapsed to a near-bimodal output
(120 of 252 predictions ≈ 0, 107 ≈ 1), so **the predicted probabilities must not be
read as calibrated risks on this cohort.** The ranking transfers; the probabilities
do not.

---

## Dataset

```
N = 252
ASD      = 135
Control  = 117
```

Source: `data/raw/Q-CHAT Polish/polish_qchat.csv`, sha256 `6c089700…`. Independently
verified against `data/QCHAT_dataset2 mendeley.sav` at 252/252 participant
identifiers, 0 duplicates, 0 mismatches.

**Published discrepancy, retained not corrected.** The source publication states
135 + 118 = 253 in one location, contradicting its own stated total of 252. The
observed data (135 + 117 = 252) is used. The dataset was **not** altered.

---

## Frozen model

| property | value |
|---|---|
| predictor version | 2 |
| calibration | **isotonic** (authoritative for research) |
| weights | `results/predictor_saudi_v2_isotonic.pt` |
| weights sha256 | `2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4` |
| calibrator | `results/predictor_saudi_v2_isotonic.pkl` |
| calibrator sha256 | `a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863` |
| split fingerprint | `b2021998a83e7224` |
| development label source | `questionnaire` (circular) |
| retrained | **no** |

Verified unchanged before and after the run.

---

## Feature contract

```
Q1  → Q-CHAT-25 Q1     → qchat1recode  → A1  → feature_1
Q2  → Q-CHAT-25 Q2     → qchat2recode  → A2  → feature_2
Q3  → Q-CHAT-25 Q5     → qchat5recode  → A3  → feature_3
Q4  → Q-CHAT-25 Q6     → qchat6recode  → A4  → feature_4
Q5  → Q-CHAT-25 Q9     → qchat9recode  → A5  → feature_5
Q6  → Q-CHAT-25 Q10    → qchat10recode → A6  → feature_6
Q7  → Q-CHAT-25 Q15    → qchat15recode → A7  → feature_7
Q8  → Q-CHAT-25 Q17    → qchat17recode → A8  → feature_8
Q9  → Q-CHAT-25 Q19    → qchat19recode → A9  → feature_9
Q10 → Q-CHAT-25 Q25    → qchat25recode → A10 → feature_10
```

Projected shape `(252, 10)`, values `{0.0, 1.0}`, zero missing. Encoded to the
frozen **41-dimensional** interface (`4n + 1`, binary `m_list=None`), verified
byte-identical to the terminal scoring path used for the Saudi metrics.

Scoring direction is derived per item from the printed option letters, not from code
magnitude: items Q1–Q9 score C/D/E; item Q10 scores A/B/C.

**The projection intentionally discards ordinal information.** Each five-level Polish
response collapses to two levels, so at least three levels per item are lost
irreversibly. It is **not** information-preserving.

---

## Validation — primary threshold τ = 0.5

| metric | point | 95% CI |
|---|---|---|
| N | 252 | — |
| Brier | 0.1398 | 0.1012 – 0.1803 |
| AUROC | 0.8959 | 0.8560 – 0.9314 |
| AUPRC | 0.8860 | 0.8351 – 0.9316 |
| Sensitivity | 0.8000 | 0.7368 – 0.8633 |
| Specificity | 0.8803 | 0.8200 – 0.9346 |
| PPV | 0.8852 | 0.8240 – 0.9380 |
| NPV | 0.7923 | 0.7244 – 0.8571 |
| ECE (10 bin) | 0.1370 | 0.1008 – 0.1832 |

AUPRC baseline (prevalence) = 0.5357.

**Confusion matrix at τ = 0.5** — rows actual `[control, ASD]`, cols predicted:

```
[[103,  14],
 [ 27, 108]]
```

TP 108, TN 103, FP 14, FN 27.

Intervals are a **paired percentile bootstrap** over evaluation instances, 2000
resamples, seed 0, 0 resamples excluded as degenerate. Every reported point estimate
lies inside its own interval; nothing is imputed.

### Calibration diagnostics (descriptive only)

| quantity | value | ideal |
|---|---|---|
| calibration-in-the-large (intercept) | 0.501 | 0.0 |
| calibration slope | 0.167 | 1.0 |
| mean predicted risk | 0.477 | 0.536 (prevalence) |

These are **diagnostics describing the frozen model's behaviour.** They were not
used to recalibrate, adjust any prediction, or select the threshold.

---

## Secondary sensitivity — τ = 0.3

Reported separately. **Never used to tune the frozen model or select the primary
threshold.**

| metric | point | 95% CI |
|---|---|---|
| Sensitivity | 0.8148 | 0.7517 – 0.8768 |
| Specificity | 0.8547 | 0.7890 – 0.9143 |
| PPV | 0.8661 | 0.8017 – 0.9219 |
| NPV | 0.8000 | 0.7333 – 0.8647 |

Moving the threshold 0.5 → 0.3 buys ~1.5 points of sensitivity and costs ~2.6 points
of specificity. It does not change the qualitative conclusion.

---

## V-4

```
automated evidence ............... PASS
MDE analysis role ................ SENSITIVITY_ANALYSIS_REPORTED
gates primary external validation  NO
human SD ratification required ... NO
```

The MDE grid is **retained, not deleted** (SD = 0.50 / 1.00 / 2.00 → MDE ≈
0.088–0.431 Brier). It is labelled sensitivity/planning analysis because an assumed
paired-difference SD is arbitrary unless independently justified. Precision is
conveyed by the bootstrap intervals instead.

---

## V-7

```
denominator ....................... 252  (PASS)
baseline 10 ...................... OPEN_UNREPRODUCED
baseline 10 blocks primary ....... no
```

The Sollis et al. specification was **not** fabricated and **no substitute comparator
was constructed**. Any later comparative claim needing Baseline 10 stays blocked
until the exact citation and model specification are obtained.

The item-count figure reported above is a **trivial internal reference for
interpretability**, explicitly *not* Baseline 10.

---

## Leakage — PASS

| check | result |
|---|---|
| development label source | `questionnaire` |
| external label source | `clinical` |
| label sources disjoint | **yes** |
| participant key intersection | **0** |
| development carries a participant key column | no |
| instruments disjoint | yes |

No fitting, calibration, threshold tuning, feature selection, hyperparameter
selection, RL training or reward tuning touched Polish. The loaded predictor was
asserted to expose no fitting method and to be a `FrozenPredictor`.

---

## Circularity — PASS

| cohort | exact match | classification |
|---|---|---|
| Polish (clinical target) | 0.5437 | not circular |
| Saudi (questionnaire target) | 1.0000 | deterministic |

This contrast is what makes the external cohort informative. It is also the central
interpretive caveat: **the development target was questionnaire-derived and the
external target is clinician-established**, so the transfer being measured is
between two different label definitions.

---

## Limitations

1. **The model does not beat a trivial baseline.** AUROC 0.896 vs 0.937 for counting
   atypical answers. The network contributes no value on this cohort.
2. **Poorly calibrated.** Slope 0.167 vs an ideal 1.0; ECE 0.137; isotonic output
   saturated to near-bimodal. Probabilities are not calibrated risks here.
3. **Different label definitions.** Saudi labels are questionnaire-derived and
   exactly reproducible from the items; Polish labels are clinician-established.
4. **Ordinal information discarded by design.** Five levels collapse to two; at
   least three levels per item lost irreversibly.
5. **Cross-country and cross-instrument shift.** Saudi Arabic Q-CHAT-10 → Polish
   Q-CHAT-25. Unquantified.
6. **Finite sample.** n = 252. Intervals are wide; not powered for small effects.
7. **Screening prediction is not clinical diagnosis.** Nothing here diagnoses autism
   in any individual, and nothing here establishes clinical validity or a basis for
   deployment.
8. **Single artefact, single split.** No replication across seeds or cohorts.

---

## Interpretation

| level | statement |
|---|---|
| 1. Screening-model transfer | The predictor trained on Saudi screening-derived labels reaches AUROC 0.896 on an independent Polish cohort. |
| 2. Prediction of clinician labels | Agreement is against clinician-established labels that the audit confirms are not reproducible from the questionnaire. |
| 3. Screening / referral use | Conceivable but **NOT established**; would need prospective clinical evaluation. |
| 4. Medical diagnosis | **NOT CLAIMED.** This work does not diagnose autism. |

The result is **external validation against clinician-established ASD labels** — not
"the model diagnosed autism."

---

## Reproduction

```
python -m pytest tests -q                     # 368 passed
python scripts/step9_v4_v7_gates.py           # exit 0
python scripts/step10_external_validation.py  # exit 0
```

Machine-readable: `results/polish_external_validation.json` (metadata only, no
participant rows), `results/v4_v7_validation_status.json`,
`results/pre_validation_reproducibility.json`.