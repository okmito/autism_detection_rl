# CORRECTED ADAPTIVE-SELECTION EXPERIMENT — RESULTS

**Date:** 2026-10-02 · **Claim boundary: research prototype. External validation
against clinician-established ASD labels. NOT a diagnosis. NOT a diagnostic device.**

Retains `results/polish_rl_external_evaluation.json` unchanged. Adds
`results/polish_fixed_budget_protocol.json` and
`results/polish_fixed_budget_external_evaluation.json`.

---

## 1. RANDOM BASELINE — fixed-length

`RandomFixedLengthPolicy` (`src/policies/random_fixed.py`): STOP is never legal, so
every episode asks **exactly B unique questions**. `b_min = B` for **every** arm.
`RandomPolicy` is untouched and reserved for a separate stopping experiment; it is
not mixed into this comparison.

**Confirmed:** every arm shows `min_questions == max_questions == B` for all
B ∈ {2,3,4,5,10}, asserted in tests.

## 2. FULL-INFORMATION REFERENCES — both reported

| reference | AUROC | Brier |
|---|---:|---:|
| **A.** v3 predictor, all 10 items | 0.9349 | 0.1104 |
| **B.** item count, all 10 items | 0.9369 | — |

`delta_vs_v3_full` isolates **question-selection loss**; `delta_vs_item_count`
compares against the strongest simple questionnaire reference. Neither is called a
sole ceiling.

**B=10 wording:** under the frozen v3 scorer, asking all 10 questions reaches the
scorer's full-information performance; the separate item-count reference is slightly
stronger externally. At B=10 the fixed-budget semantics force all ten items, so no
selection freedom remains and **all four arms return exactly 0.9349** — B=10 is a
reference point, not a policy comparison.

## 3. POLICY COMPARISON — AUROC by budget

| B | random_fixed | greedy IG | beta-greedy | DQN | reward: rand / greedy / beta / DQN |
|---|---|---|---|---|---|
| 2 | 0.7630 | 0.7879 | 0.7879 | 0.7935 | .7362 / .7612 / .7612 / .7284 |
| 3 | 0.8566 | 0.8495 | 0.8483 | 0.8774 | .7646 / .7990 / .7987 / .7694 |
| 4 | **0.9144** | 0.8904 | 0.8915 | 0.8644 | .8054 / .8246 / .8248 / .8031 |
| 5 | 0.8907 | 0.9047 | **0.9081** | 0.8859 | .8205 / .8442 / .8431 / .8166 |
| 10 | 0.9349 | 0.9349 | 0.9349 | 0.9349 | .8896 (all identical) |

Loss vs reference A: B=2 −0.141 to −0.172 · B=3 −0.058 to −0.087 · B=4 −0.020 to
−0.071 · B=5 −0.027 to −0.049 · B=10 0.0000.

## 4. PAIRWISE COMPARISONS — confirmatory family F1

Holm–Bonferroni, α = 0.05, 8 comparisons, 5000 paired bootstrap resamples with
identical resamples within each comparison. **This is the only confirmatory
analysis.**

| comparison | difference | 95% CI | P(A>B) | Holm p | reject |
|---|---:|---|---:|---:|---|
| greedy − random @ B=2 | +0.0250 | [−0.0316, +0.0838] | 0.811 | 1.0000 | no |
| beta − random @ B=2 | +0.0250 | [−0.0316, +0.0838] | 0.811 | 1.0000 | no |
| greedy − random @ B=3 | −0.0072 | [−0.0465, +0.0327] | 0.373 | 1.0000 | no |
| beta − random @ B=3 | −0.0083 | [−0.0479, +0.0315] | 0.352 | 1.0000 | no |
| greedy − random @ B=4 | −0.0240 | [−0.0622, +0.0133] | 0.108 | 1.0000 | no |
| beta − random @ B=4 | −0.0229 | [−0.0612, +0.0148] | 0.118 | 1.0000 | no |
| greedy − random @ B=5 | +0.0140 | [−0.0119, +0.0402] | 0.857 | 1.0000 | no |
| beta − random @ B=5 | +0.0174 | [−0.0103, +0.0448] | 0.895 | 1.0000 | no |

**All 8 intervals include zero; all Holm-adjusted p = 1.0.** Everything else
(policy-vs-policy at B=10, DQN comparisons, reference comparisons) is exploratory,
uncorrected, and reported with intervals only.

## 5. DQN REPRODUCIBILITY — 5 seeds per budget

Selection rule prespecified: highest mean terminal reward on the **Saudi validation**
split, ties → lowest seed. **Polish was not inspected during selection.** All five
seeds reported per budget.

| B | selected seed | val reward (mean ± SD) | val AUROC (mean ± SD) |
|---|---|---|---|
| 2 | 1 | 0.8028 ± 0.0076 | 0.8473 ± 0.0272 |
| 3 | 1 | 0.8185 ± 0.0125 | 0.9292 ± 0.0207 |
| 4 | 4 | 0.8450 ± 0.0106 | 0.9583 ± 0.0129 |
| 5 | 2 | 0.8766 ± 0.0244 | 0.9733 ± 0.0195 |
| 10 | 0 | 0.9552 ± 0.0000 | 1.0000 ± 0.0000 |

Per-seed reward, validation reward, validation AUROC and policy hash are recorded in
the artifact. Seed spread is material at reduced budgets (AUROC SD up to 0.027) and
**zero at B=10**, where every item is forced and all seeds behave identically — an
independent sanity check on the fixed-budget design. New artifacts in
`results/policies_v3_multiseed/`; the original single-seed files are retained.

## 6. EXACT DP — optimality gap under the simulator

**`ExactDP` originally optimised a different objective** (`1−(p_emp−y)²`, the
empirical label mean) from the environment's `1−(p̂−y)²`. An optional `predictor`
argument was added so the oracle optimises the environment's own reward. Default is
`None`, and backward compatibility is verified: without a predictor, B=2 returns
V\*=0.93862870, identical to the pre-existing artifact.

| B | J_exact | gap: random | gap: greedy | gap: beta | gap: DQN |
|---|---|---|---|---|---|
| 2 | 0.843300 | — | — | — | — |
| 3 | 0.873258 | — | — | — | — |
| 4 | 0.895485 | — | — | — | — |
| 5 | 0.916867 | +0.0964 | +0.0727 | +0.0737 | +0.1003 |
| 10 | 0.954033 | +0.0644 | +0.0644 | +0.0644 | +0.0644 |

**DQN has the largest optimality gap at B=5 (+0.1003), worse than random-fixed
(+0.0964) and well worse than greedy (+0.0727).** ExactDP is an oracle **under the
simulator** and carries no clinical interpretation.

## 7. REWARD AUDIT — surrogate objective

`R = 1 − (p̂ − y)²`, λ = 0 — unchanged. But the **development label is the Saudi
Q-CHAT-derived questionnaire label** while the **external target is the Polish
clinician-established label**. The reward is therefore a **surrogate** and is never
described as optimising clinical diagnosis.

Diagnostic across 20 (arm, budget) cells: Saudi simulator reward vs Polish AUROC
gives Pearson 0.845, Spearman 0.836. That correlation is driven largely by budget
level. **At matched budget the reward does not rank-order external AUROC:**

| B | highest simulator reward | highest external AUROC |
|---|---|---|
| 2 | greedy (0.7879) | DQN (0.7935) |
| 3 | greedy (0.8495) | DQN (0.8774) |
| 4 | beta (0.8915) | **random_fixed (0.9144)** |
| 5 | greedy (0.9047) | beta (0.9081) |

At B=4 the *lowest*-reward arm has the *highest* external AUROC. This is a
correlation diagnostic, not a causal claim, and was not used to tune anything.

## 8. BAYESIAN MARGINALISATION — support and sensitivity

| quantity | value |
|---|---|
| possible configurations | **1024** |
| observed in Saudi train | **155** |
| zero-mass | **869** |
| cells with count 1 | **121** |
| effective support (participation ratio) | **35.6** |
| Saudi val configs absent from train | 35 |

| variant | zero-mass cells | undefined states | meets definition |
|---|---:|---:|---|
| **A** empirical joint | 869 | 35 | **no** |
| **B** add-α joint (α=0.5, prespecified) | 0 | 0 | yes |
| **C** factorised item — **PRIMARY** | 0 | 0 | yes |

Variant A fails exactly as predicted: the 35 undefined states are precisely the
Saudi-val configurations absent from train. **C is primary**, justified by this
support measurement, not by any external result. α = 0.5 was fixed before execution
and never tuned. No variant was selected using Polish.

## 9. MAIN CONCLUSION

**On the budget-matched comparison, adaptive question selection did NOT demonstrate
superiority over fixed-length random selection at any reduced budget.**

This is a **confirmed null** on the prespecified confirmatory family F1: all 8
Holm-corrected intervals include zero (all adjusted p = 1.0). It **contradicts** the
reframing proposed before this experiment was run. The earlier apparent advantage of
adaptive policies was an artefact of the random arm asking **fewer** questions
(1.73/2, 2.60/3, 3.12/4, 3.52/5, 4.96/10); once budget-matched, random is
competitive at every budget and *ahead* at B=4.

Reported as measured, without causal language:

- Random-fixed is not worse than the heuristics at any budget (F1 null).
- DQN did not approach heuristic performance reliably: best at B=2 and B=3, worst at
  B=4, and carrying the largest optimality gap at B=5.
- Beta-greedy and greedy are near-indistinguishable (identical at B=2 and B=10).
- The surrogate reward does not rank-order external AUROC at matched budget.
- Performance tracks **how many** questions are asked far more than **which** ones.
  Under the fixed v3 scorer, discriminative information appears broadly distributed
  across items rather than concentrated in a few identifiable ones.

**No equivalence or non-inferiority claim is made.** The margin remains OPEN.

## 10. LIMITATIONS

- **One Polish external cohort**; no replication.
- **Label-definition shift**: Saudi questionnaire-derived (circular, 1.0000) vs
  Polish clinician-established (0.5437). Policies were trained only against the
  former.
- **Ordinal-to-binary projection** discards ≥3 response levels per item, irreversibly.
  Not information-preserving.
- **n = 252**; intervals are wide. F1 nulls are absence of evidence, not evidence of
  equivalence.
- **Cross-country / cross-instrument domain shift** unquantified.
- **Screening prediction is not clinical diagnosis.**
- **No equivalence margin** exists, so no non-inferiority claim is possible.
- **DQN seed variability** is material (AUROC SD up to 0.027); single-seed results
  would be fragile.
- **B=10 is degenerate** under fixed-budget semantics; it is a reference, not a
  comparison.
- The **reward is a surrogate**; its within-budget ranking does not track external
  discrimination.
- Polish training labels are circular, which flatters policies during training.

## 11. FUTURE WORK (documented, not implemented)

- **Ordinal Q-CHAT-10 predictor** — the binary projection discards information.
  Recorded as `DOCUMENTED_NOT_IMPLEMENTED`; the binary contract is unchanged.
- **Cross-country development/validation** using the primary NZ 1054-row cohort
  requires acquisition or verification of the specified source file. The pooled
  6075-row file was **not** used as a substitute and the NZ loader was **not**
  modified.
- **Equivalence / non-inferiority** requires a justified margin.
- **Learned stopping** is a separate experiment; `RandomPolicy` with STOP is reserved
  for it.

## 12. PROTOCOL DECISIONS (distinct from results)

b_min = B for all arms · budgets {2,3,4,5,10} · F1 = 8 comparisons, Holm α=0.05,
the only confirmatory analysis · equivalence margin OPEN · α=0.5 for prior variant B
prespecified · DQN 5 seeds, Saudi-validation selection rule · paired percentile
bootstrap 5000, identical resamples within comparison · primary τ=0.5, never tuned on
Polish · ExactDP optional predictor for objective consistency.

## 13. MAIN FIGURE

`docs/figures/polish_fixed_budget_auroc_vs_questions.png` — AUROC vs questions
asked (RandomFixed, Greedy IG, Beta-Greedy, DQN, reference A, reference B) plus
performance loss relative to v3 full information.

## Reproduction

```
python -m pytest tests -q                                   # 453 passed
python scripts/step15_corrected_design.py                   # Saudi-only; design frozen
python scripts/step16_fixed_budget_external.py              # Polish, once
```

v2 and v3 predictors, all calibration artifacts, the Step 10 primary result (τ 0.5,
AUROC 0.8959, Brier 0.1398), Saudi and Polish source data, the RL environment
contract, the reward definition and the V-6 decision are all byte-identical. **No
commit. No push.**