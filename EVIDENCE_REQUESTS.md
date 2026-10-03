# Evidence Request — items that require a person, not a script

**Status: OPEN. Nothing here has been resolved programmatically, and nothing
here may be marked resolved by an automated run.**

Three V-4/V-7 requirements cannot be closed from the repository's own contents.
This document states exactly what is missing, who must supply it, and what is
blocked until they do. It exists so the blockers are visible and actionable
rather than quietly deferred.

---

## E-1 — Baseline 10: Sollis et al. model specification

**Gate:** V-7 · **State:** OPEN · **Blocks:** any published-benchmark comparison

Spec §19.5 requires:

> *"The published Sollis et al. numbers are not treated as a standalone
> statistical comparator. Baseline 10 reimplements their model using the same
> public data and produces predictions..."*

To reimplement that faithfully I need, and the repository does **not** contain:

| # | Missing item | Why it is required |
|---|---|---|
| 1 | Full citation + DOI for the Sollis et al. publication | Baseline 10 must be attributed to a specific method |
| 2 | The paper's stated feature set | Determines which Q-CHAT items feed the baseline |
| 3 | The paper's estimator and hyperparameters | A reimplementation with different settings is a different model |
| 4 | The paper's stated class counts | This is the same 135/118/252 discrepancy V-7 exists to settle |
| 5 | Preprocessing rules (missing values, encoding, scaling) | Determines whether predictions are comparable at all |

**Why it is left OPEN rather than implemented:** the Polish items carry **no
question wording** — coded columns and SPSS value labels, nothing more — and no canonical
Q-CHAT-25 item definition exists in this repository. Writing a Baseline 10 from a
paraphrase of a paper would produce a model that is not the paper's model while
appearing to be. That is a fabricated comparator, and it is exactly the failure
mode the spec's V-7 gate exists to prevent.

**Resolved so far:** the denominator half of V-7 *is* closed by data inspection
(135 ASD + 117 control = 252). Of the two halves, the Baseline 10 half remains open.

---

## E-2 — V-4 MDE sign-off: assumed effect size

**Gate:** V-4 · **State:** automated PASS, human sign-off OPEN · **Blocks:** confirmatory Polish evaluation

The MDE for all five pre-declared confirmatory comparisons has been computed at
n = 252 with Holm-Bonferroni adjustment. Every figure is conditional on an
**assumed** paired-difference standard deviation, which is not measurable before
the cohort is opened:

| comparison | α (Holm) | MDE @ sd 0.50 | MDE @ sd 1.00 | MDE @ sd 2.00 |
|---|---|---|---|---|
| H1.B3 | 0.01000 | 0.1076 | 0.2153 | 0.4306 |
| H1.B4 | 0.01250 | 0.1052 | 0.2104 | 0.4207 |
| H1.B5 | 0.01667 | 0.1019 | 0.2038 | 0.4076 |
| H1.B6 | 0.02500 | 0.0971 | 0.1942 | 0.3884 |
| H2.transfer | 0.05000 | 0.0882 | 0.1765 | 0.3530 |

**Decision needed:** which assumed SD (or another basis) is pre-declared as
authoritative for the confirmatory family. A supervisor must choose *before* the
cohort is opened; choosing afterwards would be an unblinded power calculation.

---

## E-3 — Q-CHAT-10 ↔ Q-CHAT-25 item correspondence

**Gate:** external validation · **State:** OPEN · **Blocks:** any 10-item subset path

Measured, not assumed:

* Polish cohort: **25 items, 0 natively binary, all ordinal with 4–6 levels.**
* Development model: **10 items, binary 0/1.**
* The cohort contains **no question wording** — it carries codes and SPSS value labels.

Three requirements for a defensible subset, all currently unmet:

| Requirement | State | Missing |
|---|---|---|
| scale compatibility | OPEN | no stated ordinal → binary rule; no item is natively binary |
| item identity | OPEN | no item text in the dataset; a shared 5-point scale does **not** imply shared construct |
| provenance | OPEN | no canonical Q-CHAT-10 or Q-CHAT-25 item definition in this repository |

**Needed:** an authoritative Q-CHAT-10 and Q-CHAT-25 item definition (the
published instrument, with wording), so that item correspondence can be
*established* rather than guessed.

---

## What is **not** requested

These were resolved from the data and need no human input:

* **V-7 denominator.** Resolved: the cohort holds 135 ASD + 117 control = 252,
  matching the published total and published ASD count. The publication's own
  text states 135 + 118 = 253, i.e. its arithmetic is internally inconsistent.
  The project denominator is **252**.
* **Dataset identity.** Resolved: the local `QCHAT_dataset2 mendeley.sav` is the
  same 252-participant cohort as the integrated Polish CSV (252/252 keys,
  sha256 matches the verified `QCHAT_dataset1.sav`).
* **Calibration authority.** Resolved: **isotonic** is authoritative for research
  metrics; the Platt cache serves the demo rather than the research record. Research metrics now regenerate
  bit-identically from `results/predictor_saudi_v2_isotonic.pt`.

---

## How to close each item

| Item | Owner | Action | Result |
|---|---|---|---|
| E-1 | supervisor / mentor | supply the Sollis et al. citation + model spec | Baseline 10 implemented; V-7 closes |
| E-2 | supervisor | ratify an assumed SD from the table above | V-4 closes |
| E-3 | supervisor / mentor | supply the published instrument wording for both forms | subset path becomes assessable |

Until E-1, E-2 and E-3 are closed, `scripts/step10_external_validation.py`
exits 2 and reports no metrics. That is the correct behaviour, not a bug to be
worked around.