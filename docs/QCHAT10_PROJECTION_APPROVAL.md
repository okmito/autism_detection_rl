# Q-CHAT-10 PROJECTION APPROVAL PACKAGE

**Purpose:** obtain a supervisor decision on whether the verified Q-CHAT-10 ->
Q-CHAT-25 binary projection may serve as the **external-validation
representation** for the frozen Saudi predictor.

**Status:**

```
AUTOMATED EVIDENCE = PASS
HUMAN APPROVAL     = OPEN
OVERALL            = OPEN
```

Machine-readable gate id: **`Q-CHAT-10-PROJECTION`**, at
`results/v4_v7_validation_status.json` -> `gates.external_validation.qchat10_projection_approval`.

---

## 1. The pipeline being approved

```
Q-CHAT-25 Polish variables          25 ordinal items, 4-6 response levels
        |
        |  verified item identity (instrument sources, section 2)
        v
Q-CHAT-10 item identity             10 items
        |
        |  item-specific scoring direction (official rule, section 3)
        v
10 binary model features            shape (252, 10), values {0.0, 1.0}
        |
        |  binary state encoding, m_list=None  ->  4n + 1
        v
41-dimensional frozen input         the exact interface the model was trained on
```

No model is retrained, no layer is changed, and no architecture is touched. This is
a **representation decision only**.

---

## 2. Canonical item mapping (verified)

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

Mapping version: `qchat10-mapping/2.0.0`. Contract version:
`qchat10-contract/2.0.0`.

Fifteen Q-CHAT-25 items are **not** used: Q3, Q4, Q7, Q8, Q11, Q12, Q13, Q14,
Q16, Q18, Q20, Q21, Q22, Q23, Q24.

### Sources

| source | establishes |
|---|---|
| Autism Research Centre Q-CHAT-10 instrument | Q-CHAT-10 item order; official scoring (C/D/E = 1 for items 1-9, A/B/C = 1 for item 10) |
| original 25-item Q-CHAT source | which 25-item corresponds to each 10-item |
| `data/raw/Q-CHAT Saudi Arabia/...Data Set Description.pdf` | `A{i}` is item {i} and is natively binary |
| `data/raw/Q-CHAT Polish/QCHAT.pdf` | printed option order A-E and wording per item |
| `data/QCHAT_dataset2 mendeley.sav` (SPSS value labels) | the numeric code behind each option |

### Superseded mapping

An earlier draft recorded Q3->Q6, Q4->Q9, Q5->Q10, Q6->Q15, Q7->Q17, Q8->Q19,
Q9->Q25 with Q10 unresolved. That was **wrong** from Q3 onward. It is superseded by
the table above and by regression tests that assert each old value is *not* used.

---

## 3. Scoring direction

**Derived, not assumed.** For each response the contract resolves:

```
SPSS code -> Polish value label -> English wording printed in the instrument
          -> printed letter A-E -> official Q-CHAT-10 direction for that item
```

- Items Q1-Q9: printed letters **C, D, E** score **1**; A, B score 0.
- Item Q10: printed letters **A, B, C** score **1**; D, E score 0.

The compact rule `code >= 2` holds for all ten items, but it is a **proven
consequence** of the derivation, not its premise. `verify_split_rule()` recomputes
both and raises if they ever disagree, so the shortcut cannot silently drift.

This distinction is load-bearing in two places:

- **`qchat2recode` is not contiguously coded** (`0,1,2,3,5`; code 4 never
  observed). A code-magnitude assumption would accept a nonexistent code 4. It is
  refused.
- **`qchat25recode` is reverse-coded**: printed order runs many-times-a-day (A) down
  to never (E), but SPSS codes run `0 = never` up to `4 = many times a day`. Higher
  code = more frequent staring. The derived rule correctly scores frequent staring
  as **1**.

---

## 4. Per-item evidence

#### Q1  ->  Q-CHAT-25 Q1  ->  qchat1recode  ->  A1

*Construct (from the instrument):* looks at you when you call his/her name

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | zawsze | A | **0** | typical |
| 1 | zazwyczaj | B | **0** | typical |
| 2 | czasami | C | **1** | atypical / risk side |
| 3 | rzadko | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q2  ->  Q-CHAT-25 Q2  ->  qchat2recode  ->  A2

*Construct (from the instrument):* how easy it is to get eye contact

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | b.łatwo | A | **0** | typical |
| 1 | dość łatwo | B | **0** | typical |
| 2 | dość trudno | C | **1** | atypical / risk side |
| 3 | b. trudno | D | **1** | atypical / risk side |
| 5 | niemożliwe | E | **1** | atypical / risk side |

#### Q3  ->  Q-CHAT-25 Q5  ->  qchat5recode  ->  A3

*Construct (from the instrument):* points to indicate that s/he wants something

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | wiele razy/dzień | A | **0** | typical |
| 1 | kilka razy/dzień | B | **0** | typical |
| 2 | kilka razy/tydzień | C | **1** | atypical / risk side |
| 3 | mniej niż raz/tydzień | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q4  ->  Q-CHAT-25 Q6  ->  qchat6recode  ->  A4

*Construct (from the instrument):* points to share interest with you

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | wiele razy/dzień | A | **0** | typical |
| 1 | kilka razy/dzień | B | **0** | typical |
| 2 | kilka razy/tydzień | C | **1** | atypical / risk side |
| 3 | mniej niż raz/tydzień | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q5  ->  Q-CHAT-25 Q9  ->  qchat9recode  ->  A5

*Construct (from the instrument):* pretends (e.g. cares for dolls, talks on a toy phone)

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | wiele razy/dzień | A | **0** | typical |
| 1 | kilka razy/dzień | B | **0** | typical |
| 2 | kilka razy/tydzień | C | **1** | atypical / risk side |
| 3 | mniej niż raz/tydzień | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q6  ->  Q-CHAT-25 Q10  ->  qchat10recode  ->  A6

*Construct (from the instrument):* follows where you're looking

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | wiele razy/dzień | A | **0** | typical |
| 1 | kilka razy/dzień | B | **0** | typical |
| 2 | kilka razy/tydzień | C | **1** | atypical / risk side |
| 3 | mniej niż raz/tydzień | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q7  ->  Q-CHAT-25 Q15  ->  qchat15recode  ->  A7

*Construct (from the instrument):* shows signs of wanting to comfort a visibly upset person

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | zawsze | A | **0** | typical |
| 1 | zazwyczaj | B | **0** | typical |
| 2 | czasami | C | **1** | atypical / risk side |
| 3 | rzadko | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q8  ->  Q-CHAT-25 Q17  ->  qchat17recode  ->  A8

*Construct (from the instrument):* how typical the child's first words were

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | b.typowe | A | **0** | typical |
| 1 | całkiem typowe | B | **0** | typical |
| 2 | trochę niezwykłe | C | **1** | atypical / risk side |
| 3 | b.niezwykłe | D | **1** | atypical / risk side |
| 4 | nie mówi | E | **1** | atypical / risk side |

#### Q9  ->  Q-CHAT-25 Q19  ->  qchat19recode  ->  A9

*Construct (from the instrument):* uses simple gestures

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | wiele razy/dzień | A | **0** | typical |
| 1 | kilka razy/dzień | B | **0** | typical |
| 2 | kilka razy/tydzień | C | **1** | atypical / risk side |
| 3 | mniej niż raz/tydzień | D | **1** | atypical / risk side |
| 4 | nigdy | E | **1** | atypical / risk side |

#### Q10  ->  Q-CHAT-25 Q25  ->  qchat25recode  ->  A10

*Construct (from the instrument):* stares at nothing with no apparent purpose

| code | SPSS value label | printed letter | binary | decoded meaning |
|---|---|---|---|---|
| 0 | nigdy | E | **0** | typical |
| 1 | mniej niż raz/tydzień | D | **0** | typical |
| 2 | kilka razy/tydzień | C | **1** | atypical / risk side |
| 3 | kilka razy/dzień | B | **1** | atypical / risk side |
| 4 | wiele razy/dzień | A | **1** | atypical / risk side |
---

## 5. Information loss — stated explicitly

**The binary projection is lossy and irreversible.** Each Polish item carries 4-6
ordinal response levels. Each is collapsed to 2. **At least three response levels
per item are discarded**, and no transformation recovers them.

Consequences the supervisor must weigh:

1. **Information is destroyed before the model ever sees it.** A participant who
   answers "a few times a week" and one who answers "many times a day" become
   indistinguishable to the frozen predictor.
2. **The collapse is instrument-justified but not information-preserving.** It
   follows the official Q-CHAT-10 scoring rule exactly. That rule was designed for
   screening totals, not for preserving ordinal signal in a trained network.
3. **Ordinal signal is discarded asymmetrically.** The middle categories are the
   ones lost; the extremes survive. For Q-CHAT items the extremes are the
   clinically salient ones, which is the strongest argument in favour of the
   projection — but it is an argument, not a guarantee.
4. **Any validation result will describe the projected representation**, not the
   full Polish responses. The claim must be phrased accordingly.
5. **The alternative is not free.** Refusing the projection leaves no 10-item binary
   view of this cohort at all; the alternative would be training a predictor over
   the native 25-item ordinal representation, which is a different model and a
   larger decision.

**This loss is a reason to require approval, not a reason to assume consent.**

---

## 6. Verification performed

| check | result |
|---|---|
| feature matrix shape on the real cohort | **(252, 10)** |
| distinct values present | `{0.0, 1.0}` only |
| missing / NaN cells | **0** |
| short or long vector | refused |
| unrelated Q-CHAT-25 items used | none |
| short / long feature vector into encoder | refused |
| encoding of 10 binary features | **(41,)** `float32` |
| forward pass on both frozen artifacts | accepted (no metric computed) |
| `predictor_saudi_v2_isotonic.pt` sha256 | `2366a283…` **unchanged** |
| `predictor_saudi_v2_isotonic.pkl` sha256 | `a1978f27…` **unchanged** |
| `demo_model_saudi_seed0_platt_v2.pt/.pkl` sha256 | **unchanged** |
| full test suite | **349 passed, 0 skipped** |

Per-column atypical rates span 0.27-0.65, so no column is degenerate or constant.

---

## 7. Decision requiring supervisor approval

> **Approve, or reject, the verified Q-CHAT-10 binary projection as the
> external-validation representation for the frozen predictor.**

- [ ] **APPROVE** the item mapping in section 2 as the Q-CHAT-10 identity mapping.
- [ ] **APPROVE** the item-specific scoring derivation in section 3.
- [ ] **ACCEPT** the information loss in section 5, understanding that validation
      metrics will describe the projected representation and not the native
      ordinal responses.
- [ ] **APPROVE** that the cohort is projected to `(252, 10)` and encoded to the
      existing 41-dimensional frozen interface, with no retraining and no
      architecture change.
- [ ] **REJECT** — record the reason; the gate then remains OPEN and Step 10 stays
      blocked.

**Signature:** ________________  **Date:** ____________

Until signed, `qchat10_projection_approval.human_approval` remains `OPEN`, and
Step 10 exits 2 with `metrics: null`.

---

MANDATORY CAVEAT — this is a research prototype. Nothing in this package is a
clinical claim, a diagnostic device, or evidence of medical validity. Agreement
with an independently labelled cohort is not a diagnosis.
