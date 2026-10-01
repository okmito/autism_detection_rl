# V-6 — Question-Cost (λ) Grid: Evidence Pack and Recommendation

**Status:** DECISION INPUT. This document does **not** sign off V-6. §24 requires a
supervisor sign-off on the λ grid before any cost-dependent claim is made, and
that sign-off has not happened. Everything below is evidence assembled so the
sign-off can be made on measurements rather than on a placeholder.

| | |
|---|---|
| **Gate** | V-6 (§24) — λ grid sign-off |
| **Script** | `scripts/step6_lambda_sweep.py` |
| **Artifact** | `results/lambda_sweep_saudi.{json,csv}` (64 cells = 8 λ × 8 B) |
| **Cohort** | Saudi 506, **circular questionnaire labels** (§16.1) |
| **Split** | canonical 4-fold, 284/95/127, fingerprint `b2021998a83e7224` |
| **Seed** | 0 |
| **Predictor** | **not used** — see §1 |
| **Reproduce** | `.venv-win/Scripts/python scripts/step6_lambda_sweep.py` |
| **Tests** | `tests/test_step6_lambda_sweep.py` |

---

## 1. Why this evidence survives a predictor change

`ExactDP` reads the empirical training support directly and never calls the
masked neural predictor, so `V*` and `V_emp` are functions of the training labels
alone. The held-out policy comparison likewise uses the empirical support
posterior rather than the neural predictor, so every number below is
**predictor-independent** and remains valid after the predictor retrain
(`MaskedPredictor.VERSION = 2`). The artifact records
`"predictor_independent": true`.

> An earlier revision of `step6` passed a constant `lambda s: 0.5` predictor for
> the held-out comparison. That makes `1 - (p_hat - y)^2` identically 0.75 for
> every record, so `mean_reward` collapsed to `0.75 - lambda * n_items` and
> measured nothing beyond acquisition cost. Corrected to the support posterior,
> which is the same quantity `ExactDP` optimises and therefore directly
> comparable to `V*`.

## 2. Units, and why the grid must straddle them

The reward is

```
R = (1 - (p_hat - y)^2) - lambda * sum_j c_j ,   c_j = 1 (uniform)
```

so **λ is in Brier units per question** and the utility term is bounded by 1. The
largest gain any question can produce is `max_p [1 − p(1−p)] = 0.25`, attained at
`p = 0.5`.

> ### ⚠️ RETRACTED — the previous justification of this section
>
> This section previously read: *"the measured marginal utility of an informative
> question on this instrument is roughly 0.05–0.15 Brier."*
>
> **That figure was never measured by any code in this repository.** It appeared
> only in prose — in the `step6_lambda_sweep.py` docstring and in earlier revisions
> of this file — and `tests/test_step6_lambda_sweep.py::test_informative_band_is_present`
> asserted only that the grid contained ≥ 3 values in [0.001, 0.05]; the number
> itself was in the docstring, never in an assertion.
>
> It has now been measured by `scripts/step8_evoi_scale_analysis.py`, and as
> stated it was wrong. See `V6_STOPPING_THRESHOLD_DECISION.md` for the full
> analysis and the open decision.

### Measured replacement

Best remaining question, expressed as expected Brier improvement (test split, 762
decision states, budget 6):

| statistic | support posterior | neural predictor |
|---|---|---|
| min | −0.038704 | 0.000000 |
| **median** | **+0.001507** | **+0.003529** |
| mean | +0.033739 | +0.039972 |
| p75 | +0.105179 | +0.103396 |
| p90 | +0.108152 | +0.108065 |
| p95 | +0.108152 | +0.108065 |
| max | +0.187500 | +0.321148 |

0.05–0.15 is approximately the **p75–p95 band**, not the typical value. The
median is an order of magnitude below it. A grid that must "straddle the
marginal utility" is therefore straddling the wrong part of the distribution.

## 2.1 Both EVOI scales are the SAME units — the earlier claim was false

An earlier revision of this file claimed *"one reward, two scales"*: that the
support-posterior EVOI was ~100× smaller than the neural-predictor marginal
utility, so a λ grid could not transfer between them. **Measured, this is false.**

Both quantities are computed with the *same functional form on the same support
labels*, differing only in which `p` is substituted:

```
u_support(S) = mean_{y in S} [ 1 - (p_beta(S) - y)^2 ]
u_pred(S)    = mean_{y in S} [ 1 - (p_hat(S) - y)^2 ]
```

Both are expected Brier improvement, dimensionless, ceiling 0.25. The mirrored
implementation reproduces `BetaGreedyPolicy.scores()` to **0.0** absolute error
over 1,800 item-states, so the two sides are on identical footing. Ratios
(support ÷ predictor): **0.43** at the median, **0.84** at the mean, **1.02** at
p75, **1.00** at p90 and p95. That is a factor of ~1, not ~100.

The original claim arose from comparing the support-EVOI *median at depth 1*
(0.0015) against the unsourced 0.05–0.15 figure. The predictor's own depth-1
median is 0.0022 — the same order.

### What does survive, and it is a different problem

On a **pure** support the support-posterior EVOI is *analytically non-positive*.
With `p = (n_pos+1)/(n+2)`, a pure support has `u = 1 − 1/(n+2)²`, and every child
is pure and smaller, so

```
gain_j = 1/(n+2)² − Σ_v w_v · 1/(n_v+2)²  ≤  0     for every split
```

Verified numerically (n=10 split 5/5 → −0.013464; n=65 split 30/35 → −0.000621).
The only reason the statistic is non-zero on a pure support is the Beta(1,1)
pseudo-count being *diluted when the support shrinks* — a support-**size**
artifact, not information about Y.

Measured consequence — the statistic is bimodal, and the modes are the two
support regimes:

| share of decision states | value of `max_j EVOI` | regime |
|---|---|---|
| 44.9% | ≤ 0 | support pure — no information left |
| 13.6% | (0, 0.01) | transition |
| 41.5% | ≥ 0.01 | support still label-mixed — real information |

So the "EVOI threshold" is in practice a **support-purity threshold**, and it
does not behave like a smoothly tunable cost parameter. This, not a unit
mismatch, is what makes V-6 hard. The decision is open and is written up in
`V6_STOPPING_THRESHOLD_DECISION.md`.

## 3. Results — B=6 (headline), B=3 and B=10 in the artifact

| λ | V* | exact items | exact reward | beta items | beta reward | greedy reward | random reward | exact − greedy |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 1.000000 | 4.45 | 0.89764 | 6.00 | 0.94276 | 0.94079 | 0.89113 | **−0.04315** |
| 0.001 | 0.996757 | 3.37 | 0.93364 | 3.31 | 0.93748 | 0.93479 | 0.88712 | −0.00115 |
| 0.005 | 0.983785 | 3.37 | **0.92016** | 2.13 | 0.90120 | 0.91079 | 0.87105 | +0.00937 |
| 0.010 | 0.967570 | 3.37 | **0.90331** | 2.13 | 0.89053 | 0.88079 | 0.85097 | +0.02252 |
| 0.020 | 0.938834 | 2.17 | 0.86478 | 2.09 | **0.87736** | 0.82079 | 0.81082 | +0.04399 |
| 0.050 | 0.872918 | 2.17 | 0.79982 | 1.43 | **0.81058** | 0.64079 | 0.69034 | +0.15903 |
| 0.100 | 0.792680 | 1.44 | **0.74049** | 1.43 | 0.73933 | 0.34079 | 0.48956 | +0.39970 |
| 0.200 | 0.780996 | 0.00 | 0.77861 | 0.00 | 0.77861 | −0.25921 | 0.08798 | +1.03783 |

The `beta items` and `beta reward` columns are in the **same units** as the
others (§2.1) and are directly comparable. It beats `greedy` at every
λ ≥ 0.005, because greedy structurally cannot stop and therefore pays for all
six questions, and it tracks the exact reference's question count across the
sweep. At λ = 0 it spends the full 6.00 items by construction (§2.1, and
`tests/test_p1f_beta_greedy.py`), which is why its question count does not track
there.

> **Caveat on reading this table.** `beta_greedy`'s stopping statistic is
> `max_j` support-posterior EVOI, and §2.1 shows that statistic is *analytically
> non-positive on a pure support*. On this cohort that makes the policy, at any
> λ > 0, close to a **support-purity detector**: it asks until the empirical
> support is label-pure, then stops. Its question counts should therefore **not**
> be read as a cost–benefit trade-off until V-6 settles what the threshold is
> supposed to mean. The open decision is in
> `V6_STOPPING_THRESHOLD_DECISION.md`.

The informative band is **λ ∈ [0.005, 0.05]**. At λ = 0.2 the exact policy asks
**zero** questions — the cost of any question exceeds the maximum available
information — and the grid is degenerate (flagged automatically in the artifact
as `degenerate_lambdas: [0.2]`).

## 4. The finding that reframes every result in this repository

At λ = 0, `V* = 1.000000` **exactly**. The empirical support becomes *pure* —
every training record still consistent with the observed answers carries the same
label — very quickly:

| depth (questions asked) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|
| fraction of states with a pure support | 0.000 | 0.437 | 0.567 | 0.820 | 0.933 | 0.982 | 1.000 | 1.000 |

Once the support is pure, `p_emp ∈ {0,1}`, so `u_stop = 1 - (p_emp - y)^2`
equals **exactly 1.0** and no further question can raise the reward. Two
consequences, both measured:

1. **`pure_support_frac_at_stop = 1.00` at λ = 0, B ≥ 6.** The exact policy does
   not stop early because a short interview is sufficient — it stops early
   *precisely when the objective has saturated*. The measured value of adaptive
   stopping at λ = 0 is a property of the label rule, not evidence about
   diagnostic information content.
2. **The λ = 0 optimum is not the best held-out policy.** At B = 6, greedy beats
   the exact reference on held-out reward by 0.043, because the λ = 0 policy
   overfits a saturated training support. On the train split `V* = 1.000000` is
   genuinely optimal, so this is a generalisation result, not a solver error.
   `tests/test_step5_benchmark.py::test_no_policy_beats_exact_optimum` remains
   satisfied — nothing exceeds `V*` on the objective it optimises.

`GreedyIGPolicy` cannot be blamed for the λ = 0 tie: it structurally strips
`STOP` from the legal set (`src/policies/greedy.py:37-39`) and therefore spends
the full budget at every λ. That is why the exact − greedy gap here is **entirely
acquisition cost** and stays near zero until questions are priced at
λ ≳ 0.005. `tests/test_step6_lambda_sweep.py::test_greedy_always_spends_the_full_budget`
pins this so a future change to greedy cannot pass unnoticed.

## 5. Recommendation for the supervisor

**Proposed grid: λ ∈ {0, 0.005, 0.01, 0.02, 0.05}**, reported as a
cost–utility frontier (§11.1 "cost-utility frontier", kept separate from the
budget–performance curve and never relabelled as one).

Rationale, in the order that matters:

| λ | why include it |
|---|---|
| 0 | The unpriced reference. Retains continuity with every number produced to date, and §4 shows it is *not* the generalising optimum — that is itself reportable. |
| 0.005 | The smallest value at which the cost term separates policies (gap +0.009 at B=6). Below this, cost is numerically irrelevant. |
| 0.01 | The "half a question" operating point. Gap +0.023, exact spends 3.37 items. |
| 0.02 | Gap +0.044. Cost becomes a material fraction of the utility term. |
| 0.05 | Upper end of the informative band. Gap +0.159; exact spends 2.17 items. Approaching the region where λ-dependent claims are most informative. |

Explicitly **excluded**, with reasons:

- **λ = 0.1** — already degenerate in effect: exact spends 1.44 items and the
  greedy reward collapses to 0.34. Reports cost, not screening.
- **λ = 0.2** — degenerate; the optimum is to ask nothing (0.00 items). The
  artifact flags this automatically.
- **λ = 0.001** — retained in the artifact for continuity but below the
  discrimination threshold; it separates nothing (gap −0.001 at B=6).

## 6. What this does not settle

- **Nothing clinical.** Every number is measured against a deterministic
  sum-threshold oracle (`label = 1[sum(A) ≥ 4]`, verified 506/506). The
  saturation in §4 is a direct consequence.
- **Nothing about Polish.** The cohort remains sealed behind V-4/V-7. It is the
  load-bearing experiment for §4: on clinician-established labels the support
  would not be expected to become pure, so both the objective and the measured
  value of stopping would change. This is RQ3/H2 and it is the single most
  informative experiment the project has available.
- **Nothing about the learned policies.** DQN and PPO were trained at λ = 0 in
  `scripts/step4_train_policies.py`; re-training them across the signed-off grid
  is the next step once the sign-off exists.

## 7. Invariants checked automatically

`scripts/step6_lambda_sweep.py` asserts these rather than assuming them, and
`tests/test_step6_lambda_sweep.py` pins them:

| Invariant | Result |
|---|---|
| `V*` non-increasing in λ, at every budget | **holds at all 8 budgets** |
| Degeneracy boundary detected, excluding B=1 | λ = 0.2 flagged; B=1 never flagged |
| Item counts within budget for all policies | holds, 64/64 cells |
| `V*` saturates at λ = 0 | `1.000000`, flag set, B ≥ 6 |
| Exact stops only on a pure support at λ = 0 | `1.00`, and falls once priced |
| Support purity non-decreasing in depth | holds, 0.000 → 1.000 |
| Artifact split fingerprint matches the canonical split | `b2021998a83e7224` |

## 8. Related documents

- `RL_TRAINING_REPORT.md` §2 — the collector defects that made the first RL
  measurement meaningless
- `POLICY_BENCHMARK_REPORT.md` §A — the corrected matched-budget benchmark
- `AUDIT_REPORT.md` — the split-leak fix and the full P0 register
- `AGENT_PROGRESS.md` — re-prioritised next steps
- `Master-Project-Specification_FINAL.md` §11.1, §20 (AB-3), §24 (V-6)
