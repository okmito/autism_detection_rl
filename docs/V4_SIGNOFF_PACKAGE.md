# V-4 SIGN-OFF PACKAGE

**Gate:** Pre-compute MDE and freeze the confirmatory comparison family
**Spec:** `Master-Project-Specification_FINAL.md` §19.3, §24 (V-4)
**Status:**

```
AUTOMATED EVIDENCE = PASS
HUMAN SIGN-OFF     = OPEN
OVERALL            = OPEN
```

`OPEN` is not converted to `PASS` anywhere in this package. The supervisor decides.

---

## 1. Confirmatory comparisons (frozen)

Five comparisons, fixed before the Polish cohort was opened. Nothing is added,
removed, or re-weighted after the fact.

| id | hypothesis | budget | arm A | arm B | metric |
|---|---|---|---|---|---|
| H1.B3 | H1 | 3 | greedy | exact_fixed_subset | Brier |
| H1.B4 | H1 | 4 | greedy | exact_fixed_subset | Brier |
| H1.B5 | H1 | 5 | greedy | exact_fixed_subset | Brier |
| H1.B6 | H1 | 6 | greedy | exact_fixed_subset | Brier |
| H2.transfer | H2 | — | polish_minus_saudi_gap | 0 | Brier gap |

**Primary endpoint:** held-out Brier loss.
**Multiple-comparison correction:** Holm–Bonferroni.
**Family alpha:** 0.05. **Family size:** 5.

### Explicitly excluded from the family

- every other budget (1, 2, 8, 10)
- every policy other than the primary adaptive policy
- exploratory subgroup cells below the pre-declared minimum size
- UQ/AUPRC and any threshold-dependent secondary metric

Rationale: spec §19.3 restricts the confirmatory family to comparisons that
directly test H1–H2, and forbids multiplying every metric, budget, tier and
subgroup into one undifferentiated p-value family.

---

## 2. Sample size

| quantity | value |
|---|---|
| External evaluation cohort *n* | **252** |
| Development cohort *n* | 506 (284 train / 95 val / 127 test) |
| Polish composition | 135 ASD + 117 control |
| Target power | 0.80 |

The evaluation *n* is 252 — the row count of the sealed cohort, confirmed against
the SPSS source at 252/252 identifier matches with 0 duplicates. See the V-7
package for the 252 vs 253 discrepancy, which does **not** affect this figure.

---

## 3. MDE calculation and assumed SD

Every MDE below is **conditional on an assumed paired-difference standard
deviation.** That quantity is *not measured* in this project and cannot be,
because it depends on the effect being detected. The figures are therefore a
**sensitivity grid over plausible assumptions, not a measurement.**

```
MDE = (z_{1-α_adj/2} + z_{1-β}) · SD_diff / sqrt(n)

n = 252,  power = 0.80  →  z_{1-β} = 0.8416
Holm-adjusted α per comparison as tabulated
```

### Sensitivity grid — MDE by assumed SD of the paired difference

| comparison | Holm α_adj | SD = 2.00 (conservative) | SD = 1.00 (moderate) | SD = 0.50 (optimistic) |
|---|---|---|---|---|
| H1.B3 | 0.010000 | 0.430558 | 0.215279 | 0.107640 |
| H1.B4 | 0.012500 | 0.420716 | 0.210358 | 0.105179 |
| H1.B5 | 0.016667 | 0.407647 | 0.203824 | 0.101912 |
| H1.B6 | 0.025000 | 0.388425 | 0.194212 | 0.097106 |
| H2.transfer | 0.050000 | 0.352967 | 0.176483 | 0.088242 |

All values are Brier-loss units. Brier loss is bounded in [0, 1].

**Reading the grid.** Under the moderate assumption (SD_diff = 1.00) the study can
detect a paired Brier difference of roughly **0.18–0.22**. Under the optimistic
assumption (SD_diff = 0.50) roughly **0.09–0.11**. Under the conservative
assumption (SD_diff = 2.00) roughly **0.35–0.43**.

**Honest caveat.** A conservative MDE of ~0.39 Brier units against a metric bounded
at 1.0 is a *weak* design — it would only detect a very large effect. The
optimistic assumption produces an implausibly sensitive design. The moderate
assumption is the most defensible of the three, but selecting it is a
**supervisory judgement**, not an automated one. This package does not select it.

---

## 4. Why automated evidence cannot close this gate

The MDE grid is computed and reproducible. What is missing is a ratified
assumption. Code cannot decide which effect size the study should be powered for;
that choice determines whether a null result will be interpreted as "no effect" or
"underpowered". Only a supervisor can make it.

---

## 5. Decision requiring supervisor approval

> **Ratify the assumed standard deviation of the paired Brier difference used for
> the V-4 MDE computation, and record that ratification.**

Specifically, approve **one** of:

- [ ] **SD_diff = 1.00 (moderate)** → MDE ≈ 0.176–0.215 Brier
- [ ] **SD_diff = 2.00 (conservative)** → MDE ≈ 0.353–0.431 Brier
- [ ] **SD_diff = 0.50 (optimistic)** → MDE ≈ 0.088–0.108 Brier
- [ ] **Other value, stated explicitly:** ______________________

And confirm:

- [ ] The five-comparison confirmatory family above is correct and final.
- [ ] Holm–Bonferroni at family alpha 0.05 across 5 comparisons is correct.
- [ ] *n* = 252 is the correct evaluation sample size.
- [ ] The published Q-CHAT-10 "more than 3 points" screening rule is **not** the
      model decision threshold, and is not used as one.

**Signature:** ________________  **Date:** ____________

Until this is signed, `human_sign_off` remains `OPEN` in
`results/v4_v7_validation_status.json`, and Step 10 remains blocked.

---

MANDATORY CAVEAT — this is a research prototype. Nothing in this package is a
clinical claim, a diagnostic device, or evidence of medical validity.