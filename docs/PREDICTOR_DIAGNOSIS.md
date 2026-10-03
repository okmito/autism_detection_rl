# WHAT THE POLISH VALIDATION ACTUALLY SHOWS

**Diagnostic pass, 2026-10-02.** No retraining. No frozen artefact modified. No
model selected using Polish. The Step 10 primary result is unchanged.

Machine-readable: `results/predictor_diagnosis.json`, `results/predictor_ablation.csv`.

---

## 1. Frozen model (PRIMARY — unchanged)

| metric | value |
|---|---|
| threshold τ | 0.5 (frozen) |
| AUROC | **0.8959** |
| Brier | 0.1398 |
| ECE (10 bin) | 0.1370 |
| calibration slope | 0.167 (ideal 1.0) |
| n distinct predictions | 16 |
| sensitivity / specificity | 0.800 / 0.880 |

Reproduced exactly from the frozen artefacts (AUROC and Brier match the recorded
values to <1e-12). Weights `2366a283…`, calibrator `a1978f27…`, both verified
unchanged.

---

## 2. Item-count baseline

| metric | value |
|---|---|
| AUROC | **0.9369** |
| Brier | 0.1450 |
| ECE | 0.1690 |
| calibration slope | **1.246** (near-ideal) |

Definition audit: count of atypical responses among the **same 10 projected items**,
**same 252 participants**, **same binary encodings** as the frozen model. The
construction takes only a feature matrix — no label argument exists in the
signature, and no threshold search was performed. Raw count and count/10 give
identical AUROC (monotone scaling).

---

## 3. Paired AUROC difference

Paired percentile bootstrap, **identical resamples** for both predictors, 5000
resamples, 0 degenerate excluded.

```
AUROC(model) - AUROC(item_count) = -0.0410
95% CI                          = [-0.0636, -0.0186]
P(model better)                 = 0.0000
P(model worse)                  = 1.0000
```

**The entire interval lies below zero**, so the frozen predictor is
**significantly worse** than simply counting atypical answers. This is not an
overlapping-CI indeterminate result.

Brier, by contrast, is **indeterminate**: difference −0.0052, 95% CI
[−0.0319, +0.0224] spans zero. The two predictors are not distinguishable on Brier.

---

## 4. Calibration diagnosis

The collapse has **two independent causes**, and the calibrator is the worse of them.

**(a) The network itself saturates.** On Saudi terminal states ~41% of raw outputs
are exactly 1.0 pre-sigmoid; on Polish only ~21%. The Saudi target is a
deterministic sum-threshold, so the loss is minimised by an extremely confident
mapping. Raw network output has sd 0.476 with a median of 0.23 — it is already
near-bimodal before any calibration.

**(b) The isotonic calibrator destroys ranking.** It maps 103 distinct raw scores
onto only **16** output levels, creating ties. Raw AUROC is **0.9292**; after
isotonic it is **0.8959**. **Isotonic costs 0.033 AUROC.**

Mean predicted risk on Polish is 0.464 (raw) / 0.477 (isotonic) against an observed
prevalence of 0.536 — systematic under-prediction of risk, concentrated in the low
bin where 120 of 252 records sit near 0 but 17.5% of them are ASD.

So the ranking was largely intact before calibration, and **the authoritative
calibrator is what broke it.**

---

## 5. Calibration ablation (Saudi-trained calibrators only)

Pre-specified secondary diagnostic. Nothing fitted on Polish.

| predictor / calibration | Brier | AUROC | ECE | slope |
|---|---:|---:|---:|---:|
| raw, uncalibrated | 0.1421 | **0.9292** | 0.1365 | 0.271 |
| Platt (existing Saudi artefact) | **0.1384** | **0.9296** | 0.1367 | 0.310 |
| Platt applied to isotonic net | 0.1384 | 0.9296 | 0.1367 | 0.310 |
| **isotonic (PRIMARY)** | 0.1398 | **0.8959** | 0.1370 | 0.167 |

**Isotonic is the worst of the three on both Brier and AUROC.** Platt is better on
both. This does **not** make Platt the primary result — the primary remains frozen
isotonic per the protocol. It is a diagnostic finding that isotonic was the wrong
choice for this deployment.

---

## 6. Predictor ablation (every model fitted on Saudi only)

| model | Brier | AUROC | ECE | slope | trained on |
|---|---:|---:|---:|---:|---|
| frozen isotonic (PRIMARY) | 0.1398 | 0.8959 | 0.1370 | 0.167 | Saudi, frozen |
| item count | 0.1450 | **0.9369** | 0.1690 | 1.246 | none (label-free) |
| **logistic L2, C=0.1** | **0.1097** | 0.9349 | **0.0775** | 1.370 | Saudi train (284) |
| logistic L2, C=1 | 0.1178 | 0.9340 | 0.0965 | 0.638 | Saudi train (284) |
| logistic L1, C=1 | 0.1199 | 0.9309 | 0.1016 | 0.582 | Saudi train (284) |
| Saudi prevalence constant | 0.2684 | 0.5000 | 0.1403 | 0.068 | Saudi train (284) |

**A plain L2 logistic regression trained on 284 Saudi records beats the frozen
neural network on every metric** — Brier 0.110 vs 0.140, ECE 0.078 vs 0.137, AUROC
0.935 vs 0.896. Regularisation strengths were pre-specified, not selected on Polish.

---

## 7. Does the learned predictor add anything beyond the item count?

| comparison | AUROC difference |
|---|---|
| frozen network − item count | **−0.0410** (CI [−0.064, −0.019]) |
| best Saudi logistic − item count | **−0.0021** |

Logistic regression can use *which* items are atypical, not only how many, so it is
the fair test of incremental information. It also fails to beat the count (−0.002,
indistinguishable). **The item count captures essentially all the available
discrimination signal in this instrument.**

Precise wording, as required:

> The current frozen predictor did not demonstrate incremental discrimination over
> the item-count reference on this external cohort. A plain logistic regression
> trained on Saudi development data also did not exceed the item count, indicating
> the count captures essentially all available signal at n = 252.

The count is *worse* on calibration (ECE 0.169, though slope 1.25); logistic is the
best-calibrated option found (ECE 0.078).

---

## 8. RL contribution (COMPONENT B, assessed separately)

The predictor underperforming the item count does **not** invalidate the RL
question-selection component. But it bounds it: every policy's reward is scored
*through this predictor*, so a miscalibrated, rank-degraded predictor distorts the
reward signal the policy optimises.

Existing Saudi benchmark (all policies, B = 3 and 6):

| B | policy | Brier | AUROC | items asked |
|---|---|---:|---:|---:|
| 3 | random | 0.1176 | 0.9056 | 2.54 |
| 3 | greedy | 0.0876 | 0.9528 | 3.00 |
| 3 | beta_greedy | 0.0876 | 0.9528 | 3.00 |
| 3 | dqn | 0.0864 | 0.9359 | 3.00 |
| 3 | ppo | 0.1162 | 0.8913 | 1.43 |
| 6 | random | 0.0918 | 0.9422 | 4.02 |
| 6 | greedy | 0.0512 | 0.9882 | 6.00 |
| 6 | dqn | 0.0640 | 0.9692 | 5.98 |
| 6 | ppo | 0.1162 | 0.8913 | 1.43 |

**These Saudi numbers are not evidence.** They are on circular questionnaire-derived
labels, where an item-count-equivalent signal is almost perfectly recoverable
(Saudi circularity = 1.0000), which flatters every policy.

Two honest caveats:
- **beta_greedy is numerically identical to greedy** at both budgets. That is the
  previously documented H1 saturation (V\* = 1.000000 at λ=0), not a finding.
- **ppo asks only 1.43 items and matches none of the others.** That looks like
  premature stopping rather than efficiency, and should be treated as a defect to
  investigate, not a win.

**No trained RL policy artefact exists on disk** (step 4 trains in memory, step 5
benchmarks immediately). A fresh external RL evaluation would require re-training on
Saudi, deliberately not done in this diagnostic pass.

The real question for COMPONENT B — "can an adaptive policy reach comparable
predictive performance using fewer questions?" — is **still unanswered on external
data.**

---

## 9. Research conclusion

**Classification: C — the frozen predictor underperforms the simple baseline,
statistically confirmed.**

```
AUROC difference  -0.0410,  95% CI [-0.0636, -0.0186]  (entirely below zero)
```

Precise statement:

> On this external cohort the current frozen predictor did not demonstrate
> incremental discrimination over the item-count reference; its AUROC was
> significantly lower. The gap is attributable to two identified, separable defects —
> network saturation induced by a deterministic training target, and tie-creating
> behaviour in the isotonic calibrator that costs 0.033 AUROC on its own. A plain
> logistic regression trained on Saudi development data outperformed the frozen
> network on every metric.

**Where the contribution should focus: adaptive question selection, not predictive modelling.**

Predictive modelling on this instrument is close to solved by counting: the ceiling
is the questionnaire's own information, and a linear count already reaches AUROC
0.937. Effort spent improving the classifier has low expected return.

Adaptive selection is the genuinely open question and the only component where
fewer questions could plausibly buy comparable accuracy. But it must be re-grounded
on a predictor that is at least as good as a logistic regression, and evaluated
against a floor that now includes item count, random questioning, and greedy
information gain.

---

## 10. Recommended next experiment

**One experiment, no Polish leakage.**

**Question.** On an independent external cohort, can an adaptive policy reach the
predictive accuracy of *reading all ten items* while asking significantly fewer?

**Pre-registered design.**

1. **Replace the predictor.** Train a plain L2 logistic regression (C = 0.1,
   pre-specified) on the Saudi train split; calibrate on the Saudi validation split
   with **Platt**, which outperformed isotonic on both Brier and AUROC. Freeze and
   hash. New predictor version — the existing frozen artefacts and the Step 10
   primary result remain untouched and reported as-is.
2. **Establish the floor.** Report item count, random questioning, and greedy
   information gain at every budget, per `src/eval/baselines.py`. No claim of
   improvement without beating all three.
3. **Train policies on Saudi only.** DQN and beta-greedy against the new
   predictor, Polish untouched throughout.
4. **Evaluate once on Polish.** For each budget B ∈ {2, 3, 4, 5}: Brier, AUROC,
   items asked, and the **smallest B at which the policy's AUROC is
   statistically indistinguishable from the 10-item item-count AUROC** (paired
   bootstrap). That B is the headline efficiency result.
5. **Pre-register the primary endpoint** — items-asked at AUROC parity — and the
   analysis before seeing Polish.

**Why this is the right next step.** It targets the only component with open
scientific value, uses the strongest predictor found rather than the broken one,
and measures against a floor that this diagnostic established is high.

---

## 11. Baseline policy going forward

Per this investigation, `src/eval/baselines.py` makes three references mandatory:

1. `item_count`
2. `random_questioning`
3. `greedy_information_gain`

`missing_baselines()` reports any result set that omits them. Tests:
`tests/test_baselines_and_diagnosis.py`.

---

## Reproduction

```
python -m pytest tests -q                                  # 389 passed
python scripts/step9_v4_v7_gates.py                        # exit 0
python scripts/step10_external_validation.py               # exit 0, result unchanged
python scripts/step11_predictor_diagnosis.py               # exit 0
```

No commit. No push. No retraining. No frozen artefact modified.