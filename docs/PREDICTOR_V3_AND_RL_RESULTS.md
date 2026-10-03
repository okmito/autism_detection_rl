# PREDICTOR V3 AND ADAPTIVE QUESTION-SELECTION EXTERNAL EVALUATION

**Date:** 2026-10-02 · **Claim boundary: research prototype. External validation
against clinician-established ASD labels. NOT a diagnosis. NOT a diagnostic device.**

---

## 1. NEW PREDICTOR

| property | value |
|---|---|
| model | `sklearn` L2 Logistic Regression, **C = 0.1** |
| calibration | **Platt sigmoid on the raw probability**, fitted on the Saudi **validation** split |
| intercept / basis | `log-odds = β₀ + Σ βᵢxᵢ` over the ten Q-CHAT-10 binary items |
| artifact | `results/predictor_logistic_saudi_v3.pkl` |
| sha256 | `a6f28b85f2932151b745696f0bb24943024c31ac23be41fae91c4f6c483643a2` |
| training data | Saudi **train** split, n = 284 |
| validation data | Saudi **validation** split, n = 95 — calibration only, never coefficients |
| feature contract | `qchat10-contract/2.0.0` (unchanged) |
| split fingerprint | frozen in `results/polish_rl_external_protocol.json` |
| git SHA | recorded in the protocol artifact |
| Polish used | **no** |

### Why this predictor

1. The frozen neural predictor scored AUROC 0.8959 externally against 0.9369 for item count; paired difference −0.0410, 95% CI [−0.0636, −0.0186].
2. **Isotonic** calibration collapsed 103 distinct raw scores onto 16 levels, creating ties that cost 0.033 AUROC alone.
3. In a Saudi-only ablation, L2 logistic C=0.1 beat the frozen network on **every** metric (Brier 0.1097 vs 0.1398; ECE 0.0775 vs 0.1370; AUROC 0.9349 vs 0.8959).
4. Ten inspectable coefficients instead of a hidden representation.
5. Smoother probabilities give the RL reward a better-conditioned signal.

**This is not a claim of clinical superiority.** No clinical claim is made anywhere. It is the new development scorer.

### Two bugs found and fixed while building it

- **Platt on the logit diverged.** The Saudi target is a deterministic sum-threshold, so the model is near-separable: raw probabilities saturate, logits reach ±40, and an unpenalised logit-domain fit produced coefficient **60.9**, collapsing every prediction to ~0.001. Fixed by regressing on the bounded raw probability — the standard Platt form, monotone, so discrimination is unaffected by construction.
- **Reversed config-index convention.** `itertools.product` varies the *last* element fastest; the table key assumed bit *j* varies fastest, so lookups returned arbitrary configurations. Combined with normalising by the prior mass of *all* 1024 configurations instead of the *consistent* subset, v3 briefly reported ~0.001 everywhere. Both fixed, and `test_full_vector_and_state_scoring_agree_exactly` locks it.

### Explainability

| item | β (log-odds) | odds × when atypical | p (others typical) → p (this item atypical) |
|---|---|---|---|
| Q1 | +0.529 | ×1.70 | 0.091 → 0.146 |
| Q2 | +0.806 | ×2.24 | 0.091 → 0.184 |
| Q3 | +0.705 | ×2.02 | 0.091 → 0.169 |
| Q4 | +0.559 | ×1.75 | 0.091 → 0.149 |
| Q5 | +0.629 | ×1.88 | 0.091 → 0.159 |
| Q6 | +0.978 | ×2.66 | 0.091 → 0.211 |
| Q7 | +0.556 | ×1.74 | 0.091 → 0.149 |
| Q8 | +0.623 | ×1.87 | 0.091 → 0.158 |
| Q9 | +0.962 | ×2.62 | 0.091 → 0.208 |
| Q10 | +0.447 | ×1.56 | 0.091 → 0.136 |

Every coefficient is positive: atypical responses raise the log-odds. Q6 and Q9 carry the most weight. Reported exactly as fitted — no post-hoc interpretation added.

### Partial states

The environment presents partial states, so unobserved items are handled by **exact Bayesian marginalisation** over the 2¹⁰ configurations, with per-item priors from the **Saudi training** prevalence. Exact under a stated independence assumption; strictly better than imputing unobserved items as typical.

---

## 2. BASELINES

All four are **mandatory**. `assert_all_baselines()` raises `MissingBaselineError` if any is absent — this fired during development and blocked the run until `item_count` was reported as a first-class arm.

| baseline | role |
|---|---|
| **A. Item count** | count of atypical answers; the full-question reference |
| **B. Random questioning** | uniform draw from the legal set; lower bound |
| **C. Greedy information gain** | highest empirical `IG(s,j) = H(Y\|s) − Σ_v P(v\|s,j)H(Y\|s,j,v)` |
| **D. Beta-greedy / EVOI** | highest expected value of information under a Beta(1,1)-smoothed Saudi posterior. **Not an RL policy** — a Bayesian empirical-posterior heuristic |

---

## 3. RL

| property | value |
|---|---|
| algorithm | **DQN, one policy per budget** (previously trained only in memory; now trained **and persisted**) |
| artifacts | `results/policies_v3/dqn_v3_B{2,3,4,5,10}.pt`, each sha256-recorded in the protocol |
| state | 3n mask one-hot + observed response bits + normalised budget remaining |
| action | ask item *j* ∈ [0, n), or STOP |
| reward | `R = (1 − (p̂ − y)²) − λ·Σcⱼ` with λ = 0 — **structure unchanged** |
| training | Saudi **train** split only, 2000 episodes/budget, γ = 0.99, ε 1.0 → 0.05, 4 updates/episode, batch 32 |
| Polish used in training | **no** |

**Reward quality is bounded by predictor quality.** A rank-degraded, miscalibrated scorer distorts the signal a policy optimises — which is precisely why v2 was replaced. Results from the v2 and v3 scorers are **never mixed**; all experiments are versioned.

---

## 4. PROTOCOL

`results/polish_rl_external_protocol.json` — frozen **before** the sealed cohort was read.

| item | value |
|---|---|
| budgets | **2, 3, 4, 5, 10** (10 = full-question reference) |
| primary metric | AUROC |
| secondary | Brier, sensitivity, specificity, questions_used, cumulative_reward |
| primary τ | **0.5**, not tuned on Polish |
| secondary τ | 0.3, **sensitivity analysis only** |
| seed | 0 |
| bootstrap | paired percentile, 5000 resamples, 95%, degenerate resamples excluded and counted |
| stopping rule | `b_min = 0`; the budget alone terminates the episode |

### Equivalence margin: **OPEN**

```
acceptable_AUROC_margin : "OPEN"
state                   : "OPEN"
```

**No margin was chosen and none was invented.** No scientifically justified non-inferiority margin is available in this project. The experiment therefore reports raw paired AUROC differences with 95% intervals and **declares no equivalence in either direction**. The phrase "statistically indistinguishable" is not used as a conclusion anywhere.

---

## 5. POLISH RESULTS

Cohort n = 252 (ASD 135 / control 117), clinician-established labels, protocol hash verified.

### Full-question reference (B = 10)

| reference | AUROC | Brier |
|---|---:|---:|
| v3 predictor on all 10 items | **0.9349** | 0.1104 |
| raw item count | **0.9369** | — |

### Every arm at every budget

| B | arm | AUROC | Brier | sens | spec | questions | reward |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | random | 0.7877 | 0.2536 | | | 1.73 | 0.7464 |
| 2 | greedy IG | 0.7879 | 0.2388 | | | 2.00 | 0.7612 |
| 2 | beta-greedy | 0.7879 | 0.2388 | | | 2.00 | 0.7612 |
| 2 | DQN | 0.7964 | 0.2430 | | | 2.00 | 0.7570 |
| 3 | random | 0.8233 | 0.2355 | | | 2.60 | 0.7645 |
| 3 | greedy IG | 0.8495 | 0.2010 | | | 3.00 | 0.7990 |
| 3 | beta-greedy | 0.8483 | 0.2013 | | | 3.00 | 0.7987 |
| 3 | DQN | **0.8679** | 0.2463 | | | 3.00 | 0.7537 |
| 4 | random | 0.8247 | 0.2248 | | | 3.12 | 0.7752 |
| 4 | greedy IG | 0.8904 | 0.1754 | | | 4.00 | 0.8246 |
| 4 | beta-greedy | **0.8915** | 0.1752 | | | 4.00 | 0.8248 |
| 4 | DQN | 0.8676 | 0.1885 | | | 4.00 | 0.8115 |
| 5 | random | 0.7821 | 0.2279 | | | 3.52 | 0.7721 |
| 5 | greedy IG | 0.9047 | 0.1558 | | | 5.00 | 0.8442 |
| 5 | beta-greedy | **0.9081** | 0.1569 | | | 5.00 | 0.8431 |
| 5 | DQN | 0.8760 | 0.1714 | | | 4.92 | 0.8286 |
| 10 | random | 0.8070 | 0.2010 | | | 4.96 | 0.7990 |
| 10 | greedy IG | 0.9349 | 0.1104 | | | 10.00 | 0.8896 |
| 10 | beta-greedy | 0.9349 | 0.1104 | | | 10.00 | 0.8896 |
| 10 | DQN | 0.9328 | 0.1108 | | | 9.75 | 0.8892 |

### The comparison that matters — paired AUROC difference vs the full-question item count

| B | arm | difference | 95% CI | includes 0 |
|---|---|---:|---|---|
| 2 | random | −0.1492 | [−0.2081, −0.0932] | no |
| 2 | greedy IG | −0.1490 | [−0.1923, −0.1073] | no |
| 2 | beta-greedy | −0.1490 | [−0.1923, −0.1073] | no |
| 2 | DQN | −0.1405 | [−0.1831, −0.0988] | no |
| 3 | random | −0.1136 | [−0.1595, −0.0704] | no |
| 3 | greedy IG | −0.0874 | [−0.1217, −0.0554] | no |
| 3 | beta-greedy | −0.0886 | [−0.1228, −0.0566] | no |
| 3 | DQN | −0.0690 | [−0.1013, −0.0398] | no |
| 4 | random | −0.1122 | [−0.1593, −0.0674] | no |
| 4 | greedy IG | −0.0465 | [−0.0727, −0.0225] | no |
| 4 | beta-greedy | −0.0454 | [−0.0710, −0.0218] | no |
| 4 | DQN | −0.0693 | [−0.0999, −0.0400] | no |
| 5 | random | −0.1548 | [−0.2139, −0.1010] | no |
| 5 | greedy IG | −0.0322 | [−0.0551, −0.0121] | no |
| 5 | beta-greedy | **−0.0288** | [−0.0507, −0.0091] | no |
| 5 | DQN | −0.0609 | [−0.0892, −0.0345] | no |
| 10 | greedy IG | −0.0021 | [−0.0078, +0.0033] | yes |
| 10 | beta-greedy | −0.0021 | [−0.0078, +0.0033] | yes |
| 10 | DQN | −0.0041 | [−0.0106, +0.0021] | yes |

**Every arm at every budget below 10 has a 95% interval entirely below zero.**

---

## 6. PRIMARY CONCLUSION

**No — adaptive questioning did not reduce the number of questions while maintaining pre-specified performance.**

At **B = 5**, the best arm (beta-greedy) reaches AUROC 0.9081 against the full-question item count's 0.9369: a paired difference of **−0.0288, 95% CI [−0.0507, −0.0091]**, entirely below zero. Asking half the questions costs a measurable and statistically resolvable amount of discrimination.

The pattern is monotone and consistent across every arm: B = 2 → ≈0.79, B = 3 → ≈0.85, B = 4 → ≈0.89, B = 5 → ≈0.91, B = 10 → 0.935. Performance tracks the number of questions asked almost exactly as if selection barely mattered.

### Three further findings, reported as measured

1. **RL did not beat the heuristics.** DQN won at B = 3 (0.8679 vs greedy 0.8495) but **lost** at B = 4 (0.8676 vs 0.8904) and B = 5 (0.8760 vs 0.9047). Mixed and inconsistent.
2. **Beta-greedy ≈ greedy.** Identical at B = 2 and B = 10; within 0.002 elsewhere. The EVOI criterion did not add anything over plain information gain.
3. **The full-question interval includes zero for the informed arms** (−0.0021 [−0.0078, +0.0033]). Since the margin is **OPEN**, this is stated as a fact about the interval and **not** converted into an equivalence claim.

### Why the ceiling is the instrument

At B = 10 greedy asks all ten items and reaches AUROC 0.9349 — statistically the same as raw item count (0.9369). The instrument's own information is the ceiling. No selection policy can exceed it, and every question removed costs signal that the count was already capturing.

---

## 7. LIMITATIONS

- **One Polish external cohort only.** No replication; no second external dataset.
- **Label-definition shift.** Saudi training labels are questionnaire-derived and exactly circular (1.0000); Polish labels are clinician-established (0.5437). The transfer measured is between two different label definitions, and circular training labels flatter every policy.
- **Ordinal-to-binary projection.** Five Polish levels collapse to two; at least three levels per item are discarded irreversibly. Not information-preserving.
- **Finite sample.** n = 252. Intervals are wide; the study is not powered for small effects. Several B = 10 comparisons sit within ±0.01 of zero.
- **Cross-country and cross-instrument domain shift** is unquantified.
- **Screening prediction is not clinical diagnosis.** Nothing here diagnoses autism or establishes clinical validity.
- **No equivalence margin** was available, so no non-inferiority or equivalence claim is made in either direction.
- **`random_questioning` stops early** (`b_min = 0` allows STOP), so it asked 4.96 items even at B = 10. It is a lower bound, not a fixed-budget arm.
- **Saudi-only policy training** means the policies were never optimised against clinician-established labels.

---

## Reproduction

```
python -m pytest tests -q                                  # 414 passed
python scripts/step12_new_predictor.py                    # freeze predictor v3
python scripts/step13_rl_protocol.py                      # train + freeze protocol
python scripts/step14_rl_external_eval.py                 # Polish, once
```

v2 artefacts (`predictor_saudi_v2_isotonic.pt/.pkl`) verified **unchanged** and retained as legacy. RL environment, reward structure, existing policy implementations, configs and V-6 untouched. **No commit. No push.**