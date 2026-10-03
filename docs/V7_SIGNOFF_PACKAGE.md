# V-7 SIGN-OFF PACKAGE

**Gate:** Recompute Baseline 10 and resolve the denominator
**Spec:** `Master-Project-Specification_FINAL.md` §15, §24 (V-7)
**Status:**

```
AUTOMATED DENOMINATOR = PASS
AUTOMATED BASELINE 10 = OPEN
HUMAN SIGN-OFF        = OPEN
OVERALL               = OPEN
```

---

## 1. Observed dataset fact

Measured from `data/raw/Q-CHAT Polish/polish_qchat.csv` and independently
re-verified against the SPSS source `data/QCHAT_dataset2 mendeley.sav`
(252/252 participant identifiers matched, 0 duplicates, 0 mismatches, 252/252
categorical agreement after SPSS value-label decoding).

| class | observed count |
|---|---|
| ASD | **135** |
| control | **117** |
| **total rows** | **252** |

**135 ASD + 117 control = 252.** This is a fact about the delivered data and is
independently corroborated at the participant level.

---

## 2. Published textual inconsistency

As stated in the source publication's description:

| class | published count |
|---|---|
| ASD | 135 |
| control | **118** |
| published stated total | 252 |
| **135 + 118 (arithmetic sum)** | **253** |

The publication's own class counts sum to **253**, which contradicts its own
stated total of **252**.

### Finding-by-finding

| check | published | observed | agree |
|---|---|---|---|
| total matches row count | 252 | 252 | yes |
| ASD count matches | 135 | 135 | yes |
| control count matches | 118 | 117 | **no** |
| published ASD+control internally consistent | 253 | — | **no** |

### Resolution — and its limits

The denominator used throughout this project is **252**. This is resolved *from
the data*, not by correcting the publication.

**No correction to the publication is asserted or invented.** The discrepancy is
recorded as an unresolved textual inconsistency in the source. Two readings remain
possible and this project does not choose between them:

1. the published control count of 118 is a typographical error, or
2. one control participant was excluded from the released data.

Distinguishing these would require the authors or the study protocol. Neither is
available here. Because the observed row count (252) matches the published total
(252) and the observed ASD count (135) matches the published ASD count (135), the
**analysis denominator of 252 is not in doubt** — only the explanation of the
discrepancy is.

---

## 3. Baseline 10 — OPEN

**Status: OPEN. No substitute comparator has been created.**

### Why it cannot be closed here

Spec §19.5 requires Baseline 10 to **reimplement the Sollis et al. model on the
same public data**. Doing so faithfully requires that publication's model
specification, which is **not present in this repository**. Reimplementing from a
paraphrase or from memory would fabricate a comparator — producing a number that
looks like a published baseline but is actually an invention. That is worse than
having no baseline, because it would be indistinguishable from a real one in any
downstream table.

The gate is therefore left `OPEN` rather than closed with a guess.

### Exactly what must be supplied to close it

1. **Full citation and DOI** for the Sollis et al. publication.
2. **The paper's stated model specification**, specifically:
   - the feature set used (which variables, which encoding)
   - the estimator/algorithm
   - the hyperparameters and any tuning protocol
   - the preprocessing steps (imputation, scaling, class balancing)
   - the train/test split protocol, if the paper reports one
3. **The paper's own class counts**, which would independently settle the §2
   discrepancy above.

### Explicitly prohibited to close this gate

- Substituting a different published classifier and labelling it "Baseline 10".
- Implementing a plausible-looking model and attributing it to Sollis et al.
- Reporting any comparator number without a traceable specification.
- Renaming a locally trained model as a reproduction of a published result.

If the specification cannot be obtained, the correct outcome is to **report
Baseline 10 as unreproduced**, not to approximate it.

---

## 4. Decision requiring supervisor approval

> **Ratify the denominator of 252 for all external-validation reporting, and
> record the published 135 + 118 = 253 figure as an unresolved textual
> inconsistency in the source publication without amending it.**

- [ ] Confirm **n = 252** (135 ASD + 117 control) as the analysis denominator.
- [ ] Confirm the published 118-control figure is recorded as an inconsistency,
      **not** corrected.
- [ ] Acknowledge Baseline 10 is **OPEN** and unreproduced.
- [ ] Direct whether to request the Sollis et al. specification from the authors
      (recommended) or to publish with Baseline 10 explicitly marked unreproduced.
- [ ] Confirm Baseline 10 is **not** blocking for the confirmatory H1/H2 family,
      which does not depend on it. *(If you consider it blocking, say so — this
      package does not decide it for you.)*

**Signature:** ________________  **Date:** ____________

Until signed, `human_sign_off` remains `OPEN` and Step 10 remains blocked.

---

MANDATORY CAVEAT — this is a research prototype. Nothing in this package is a
clinical claim, a diagnostic device, or evidence of medical validity.