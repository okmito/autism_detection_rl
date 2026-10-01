# V-6 — Stopping-Threshold Decision Note

**Status: DECISION REQUIRED. V-6 is NOT signed off and this note does not sign it off.**
No threshold is selected anywhere in this document. Every candidate is reported
with its measured consequences so the decision can be made on evidence.

| | |
|---|---|
| **Gate** | V-6 (§24) — question-cost / stopping rule |
| **Analysis script** | `scripts/step8_evoi_scale_analysis.py` |
| **Artifacts** | `results/evoi_scale_saudi.{json,csv}` |
| **Cohort** | Saudi 506, **circular questionnaire labels** (§16.1) |
| **Split** | canonical, 284/95/127, fingerprint `b2021998a83e7224` |
| **Seed** | 0 · **Budget** 6 · **Test episodes** 127 · **Decision states** 762 |
| **Neural predictor** | `MaskedMLP[128,64]` + Platt, `predictor_version: 2` |
| **Existing code** | unchanged (`src/policies/beta_greedy.py` untouched; rules are subclasses in the analysis script) |
| **Tests** | 192 pass, 0 skip |

---

## DECISION REQUIRED

> **Which stopping statistic should define the V-6 cost parameter λ, and on what
> basis should its value be set — given that the current support-posterior EVOI
> is not a graded information measure on this cohort?**

This decomposes into two questions the supervisor must answer:

1. **Statistic.** Should λ be defined against
   (a) the raw support-posterior EVOI `max_j gain_j`,
   (b) the neural-predictor EVOI,
   (c) a *normalised* quantity (relative EVOI, normalised IG, uncertainty per
   question, or a reward/EVOI ratio), or
   (d) an explicitly **fixed-length** rule with no stopping at all?
2. **Calibration basis.** If a stopping rule is kept, should its threshold be set
   by
   (i) an **absolute** value in Brier units per question,
   (ii) a **percentile** of the statistic's own distribution, calibrated on the
   validation split, or
   (iii) a **utility target** (e.g. "stop when the remaining attainable utility
   is below *r*" — note this is the *opposite* inequality to the one currently
   implemented)?

A related question that must be answered alongside: **is adaptive stopping
in scope for V-6 at all**, given §0A.1 of `diagnosisReady.md` — that on this
cohort the support becomes label-pure after ~4 questions, so stopping early is
measuring label saturation rather than diagnostic sufficiency?

---

## EVIDENCE

All values below are measured, not estimated. Source: `results/evoi_scale_saudi.json`.

### E1. How support-posterior EVOI is computed (as implemented)

In `src/policies/beta_greedy.py`, for a state `s` with training support
`S = {i : record i is consistent with the answers so far}`:

```
p_beta(S) = (n_pos + 1) / (|S| + 2)                 # Beta(1,1) pseudo-counts
u(S)      = mean_{y in S} [ 1 - (p_beta(S) - y)^2 ] # expected terminal utility

for each legal item j, split S on item j into branches (observed value v, and
"item j was not answered"), with weights w_v = |S_v| / |S|:

    u(S_v)   = mean_{y in S_v} [ 1 - (p_beta(S_v) - y)^2 ]
    gain_j   = sum_v w_v * u(S_v)  -  u(S)

STOP  is returned when   max_j gain_j  <  lambda * c_j        (c_j = 1)
```

Note `u` is the *realised* utility averaged over the support's own label
distribution, **not** `1 − p(1−p)`. `argmax_j` also uses `gain_j` to pick the item.

**Fidelity check.** The analysis script reimplements this and compares against
`BetaGreedyPolicy.scores()`: max absolute difference **0.0** over **1,800**
item-states. The comparison below is therefore like-for-like.

### E2. Empirical distribution of the decision statistic `max_j gain_j`

Test split, 762 decision states, budget 6, greedy-IG visit order:

| | min | **median** | mean | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|---|
| support-posterior EVOI | −0.038704 | **+0.001507** | +0.033739 | +0.105179 | +0.108152 | +0.108152 | +0.187500 |
| neural-predictor gain | 0.000000 | **+0.003529** | +0.039972 | +0.103396 | +0.108065 | +0.108065 | +0.321148 |

Validation split (570 states) agrees: support median +0.001507, mean +0.033926;
predictor median +0.003529, mean +0.038591.

Pooled over individual legal items (5,715 observations) rather than the per-state
maximum: support median +0.001121, mean +0.020529, p95 +0.105179; predictor
median +0.001910, mean +0.020810, p95 +0.101142.

### E3. Scale mismatch, quantified — **there is none**

Both quantities use the *same functional form on the same support labels* and
differ only in which `p` is substituted, so both are expected Brier improvement:
dimensionless, ceiling `max_p [1 − p(1−p)] = 0.25`.

| ratio support ÷ predictor | median | mean | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| | **0.427** | **0.844** | **1.017** | **1.001** | **1.001** | 0.584 |

As a fraction of the 0.25 ceiling: support median 0.006, predictor median 0.014.

> **This retracts a claim made in an earlier revision of this repository.** The
> V-6 pack previously stated *"one reward, two scales"*, that the support EVOI was
> ~100× smaller than the predictor's marginal utility so a λ grid could not
> transfer. That was based on comparing the support-EVOI median **at depth 1**
> (0.0015) against a quoted predictor figure of **0.05–0.15 that no code ever
> measured**. Measured properly, the predictor's own median is 0.0035 and its
> own depth-1 median is 0.0022 — the same order. The corrected text is now in
> `V6_LAMBDA_DECISION.md` §2/§2.1, in the `lambda_units_caveat` field of
> `results/lambda_sweep_saudi.json`, and pinned by
> `tests/test_step6_lambda_sweep.py::test_lambda_units_caveat_is_declared`.

### E4. What the statistic actually measures: a purity detector

On a **pure** support (`n_pos ∈ {0, |S|}`), `u = 1 − 1/(|S|+2)²` and every child is
pure and smaller, so

```
gain_j = 1/(|S|+2)² − Σ_v w_v · 1/(|S_v|+2)²  ≤  0      for every possible split
```

Verified: n=10 → 5/5 gives −0.013464; n=20 → 10/10 gives −0.004878; n=65 → 30/35
gives −0.000621; n=100 → 1/99 gives −0.001112. The statistic is non-zero on a pure
support *only* because the Beta(1,1) pseudo-count is diluted when the support
shrinks — a support-**size** artifact, not information about Y.

Measured, by depth and purity (test split, `max_j` support EVOI):

| depth | pure-support frac | PURE states: median | IMPURE states: median |
|---|---|---|---|
| 0 | 0.000 | — | +0.108152 |
| 1 | 0.000 | — | +0.001507 |
| 2 | 0.425 | **−0.000193** | +0.021201 |
| 3 | 0.543 | **−0.000322** | +0.001920 |
| 4 | 0.803 | **−0.000405** | +0.017245 |
| 5 | 0.921 | **−0.000566** | +0.080499 |

The distribution is bimodal, and the modes *are* the two support regimes:

| share of the 762 states | `max_j EVOI` | regime |
|---|---|---|
| **44.9%** | ≤ 0 | support pure — nothing left to learn |
| 13.6% | (0, 0.01) | transition |
| **41.5%** | ≥ 0.01 | support still label-mixed — real information |

### E5. Stop rate at candidate thresholds — the grid is not a continuum

A state stops if `max_j gain_j < threshold`. No value is recommended.

| threshold | val stop rate | test stop rate | resulting mean questions |
|---|---|---|---|
| 0 | 0.402 | 0.408 | — |
| 1e-05 | 0.444 | 0.449 | 3.31 |
| 3e-05 | 0.444 | 0.449 | 3.31 |
| 0.0001 | 0.444 | 0.449 | 3.31 |
| 0.0003 | 0.444 | 0.449 | 3.31 |
| 0.001 | 0.444 | 0.449 | 3.31 |
| **0.003** | **0.586** | **0.585** | **2.13** |
| 0.01 | 0.586 | 0.585 | 2.13 |
| 0.03 | 0.632 | 0.638 | 1.78 |
| 0.05 | 0.721 | 0.717 | 1.43 |
| 0.1 | 0.732 | 0.740 | 1.43 |
| 0.15 | 0.995 | 0.995 | 0.00 |
| 0.2 | 1.000 | 1.000 | 0.00 |
| 0.25 | 1.000 | 1.000 | 0.00 |

**The region 1e-05 … 0.001 is flat**: five distinct threshold values produce
*bit-identical* episodes. The choice among them is not a decision. The entire
grid spans **20 distinct behaviours**, and the real transition is a cliff between
0.001 and 0.003.

### E6. Offline sensitivity of every candidate rule

Test split, budget 6, λ = 0, reward = support posterior. `premature` = stopped
while the support was still label-mixed; `unnec` = mean questions asked *after*
the support first became pure; `regret` = full-length reward − this rule's reward.

| rule | τ | meanQ | full-len | premature | unnec | reward | Brier | ECE₁₀ | regret |
|---|---|---|---|---|---|---|---|---|---|
| *reference: ask all 6* | — | 6.00 | 1.000 | 0.0000 | 2.69 | 0.9428 | 0.0572 | 0.0449 | 0 |
| support_evoi | 1e-05 … 0.001 | 3.31 | 0.079 | **0.0000** | 0.00 | 0.9408 | 0.0592 | 0.0567 | +0.0020 |
| support_evoi | 0.003 – 0.01 | 2.13 | 0.047 | 0.8189 | 0.00 | 0.9119 | 0.0881 | 0.0708 | +0.0309 |
| support_evoi | 0.03 | 1.78 | 0.000 | 0.9843 | 0.00 | 0.9064 | 0.0936 | 0.0593 | +0.0363 |
| support_evoi | 0.05 – 0.1 | 1.43 | 0.000 | 1.0000 | 0.00 | 0.8818 | 0.1182 | 0.0668 | +0.0609 |
| support_evoi | 0.15 – 0.25 | 0.00 | 0.000 | 1.0000 | 0.00 | 0.7786 | 0.2214 | 0.0068 | +0.1641 |
| **predictor_evoi** | 1e-05 | 4.98 | 0.449 | 0.0315 | 1.67 | **0.9467** | 0.0533 | 0.0449 | **−0.0039** |
| **predictor_evoi** | 3e-05 – 3e-04 | 4.04 | 0.433 | 0.0315 | 0.73 | **0.9467** | 0.0533 | 0.0449 | **−0.0039** |
| predictor_evoi | 0.001 | 3.87 | 0.339 | 0.0000 | 0.56 | 0.9388 | 0.0612 | 0.0606 | +0.0039 |
| predictor_evoi | 0.003 | 2.50 | 0.126 | 0.5748 | 0.06 | 0.9191 | 0.0809 | 0.0690 | +0.0237 |
| predictor_evoi | 0.01 | 2.17 | 0.079 | 0.8189 | 0.03 | 0.9119 | 0.0881 | 0.0708 | +0.0309 |
| predictor_evoi | 0.03 | 1.80 | 0.016 | 0.9843 | 0.02 | 0.9064 | 0.0936 | 0.0593 | +0.0363 |
| predictor_evoi | 0.05 – 0.1 | 1.43 | 0.000 | 1.0000 | 0.00 | 0.8818 | 0.1182 | 0.0668 | +0.0609 |
| predictor_evoi | ≥ 0.15 | 0.00 | 0.000 | 1.0000 | 0.00 | 0.7786 | 0.2214 | 0.0068 | +0.1641 |
| relative_evoi | 0.0001 – 0.001 | 3.31 | 0.079 | 0.0000 | 0.00 | 0.9408 | 0.0592 | 0.0567 | +0.0020 |
| relative_evoi | 0.003 – 0.01 | 2.13 | 0.047 | 0.8189 | 0.00 | 0.9119 | 0.0881 | 0.0708 | +0.0309 |
| relative_evoi | 0.03 | 1.78 | 0.000 | 0.9843 | 0.00 | 0.9064 | 0.0936 | 0.0593 | +0.0363 |
| relative_evoi | 0.1 | 1.43 | 0.000 | 1.0000 | 0.00 | 0.8818 | 0.1182 | 0.0668 | +0.0609 |
| **normalised_ig** | 0.001 – 0.01 | 3.27 | 0.071 | 0.0315 | 0.00 | **0.9482** | **0.0518** | 0.0528 | **−0.0054** |
| normalised_ig | 0.05 | 2.96 | 0.055 | 0.2756 | 0.00 | 0.9409 | 0.0591 | 0.0545 | +0.0018 |
| normalised_ig | 0.1 | 1.78 | 0.000 | 0.9843 | 0.00 | 0.9064 | 0.0936 | 0.0593 | +0.0363 |
| normalised_ig | 0.2 | 1.72 | 0.000 | 1.0000 | 0.00 | 0.8964 | 0.1036 | 0.0644 | +0.0464 |
| normalised_ig | 0.5 | 0.00 | 0.000 | 1.0000 | 0.00 | 0.7786 | 0.2214 | 0.0068 | +0.1641 |
| **uncertainty_per_question** | 0.001 | 3.27 | 0.071 | 0.0315 | 0.00 | **0.9482** | **0.0518** | 0.0528 | **−0.0054** |
| uncertainty_per_question | 0.01 | 2.96 | 0.055 | 0.2756 | 0.00 | 0.9409 | 0.0591 | 0.0545 | +0.0018 |
| uncertainty_per_question | 0.05 | 2.09 | 0.039 | 0.8504 | 0.00 | 0.9192 | 0.0808 | 0.0669 | +0.0235 |
| uncertainty_per_question | 0.1 | 1.78 | 0.000 | 0.9843 | 0.00 | 0.9064 | 0.0936 | 0.0593 | +0.0363 |
| reward_per_evoi | 3 | 6.00 | 1.000 | 0.0000 | 2.69 | 0.9428 | 0.0572 | 0.0449 | 0 |
| reward_per_evoi | 10 … 1e4 | 0.00 | 0.000 | 1.0000 | 0.00 | 0.7786 | 0.2214 | 0.0068 | +0.1641 |
| percentile (val-calibrated) | q=25 → −0.000566 | 3.81 | 0.126 | 0.0000 | 0.50 | 0.9388 | 0.0612 | 0.0606 | +0.0039 |
| percentile (val-calibrated) | q=50 → 0.001507 | 3.31 | 0.079 | 0.0000 | 0.00 | 0.9408 | 0.0592 | 0.0567 | +0.0020 |
| percentile (val-calibrated) | q=75 → 0.105179 | 1.43 | 0.000 | 1.0000 | 0.00 | 0.8818 | 0.1182 | 0.0668 | +0.0609 |
| percentile (val-calibrated) | q=90 → 0.108152 | 1.00 | 0.000 | 1.0000 | 0.00 | 0.8494 | 0.1506 | 0.0571 | +0.0933 |

Three results stand out:

1. **Three rule families beat asking all six questions on reward** while asking
   fewer questions (negative regret): `predictor_evoi` at τ ≤ 3e-4,
   `normalised_ig` at τ ∈ [0.001, 0.01], and `uncertainty_per_question` at
   τ = 0.001. Best measured reward is **0.9482 at 3.27 questions** against
   **0.9428 at 6.00** — a reward *gain* of +0.0054 for 2.73 fewer questions.
2. **`predictor_evoi` dominates raw `support_evoi` on every axis at once**: at
   4.04 questions it attains 0.9467 reward and 3.2% premature stops, where
   `support_evoi` needs to drop to 3.31 questions for 0.9408 and its next step
   down (τ = 0.003) collapses to 81.9% premature stops. It is also the estimator
   already in the same units as the reward (§E3).
3. **Calibration never degrades** for any candidate (ECE₁₀ 0.0449–0.0708, against
   0.0449 for the full-length reference). This is expected and is *not*
   reassuring: the labels are a deterministic function of the items, so any
   policy that reads enough items is perfectly calibrated by construction.
   Calibration therefore has **no discriminating power** on this cohort and
   should not be used as a selection criterion here.

### E7. What the "premature stop" metric does and does not mean

`premature_stop_frac` counts episodes that stopped while the empirical support was
still **label-mixed**. It is a statement about the *training support*, not about
clinical sufficiency: on this cohort a mixed support means the training evidence
does not yet determine the label. Because the label is `1[sum(A) ≥ 4]`
(506/506 verified), a support-pure state is reached by construction and a
premature stop is, on this cohort, close to "the interview was shorter than the
label rule needs". It is the right metric for a *purity-detector* reading of the
policy and the wrong metric for a *sufficiency* reading. Both readings are open.

### E8. Is normalising the EVOI mathematically justified?

**Dimensional consistency: already satisfied, so normalisation is not required
for it.** Both `u_support` and `u_pred` are expected Brier improvement — the same
units as the reward term `(1 − (p − y)²)`, dimensionless, ceiling 0.25. A raw
absolute threshold is therefore already commensurate with λ in the reward. §E3
confirms this empirically (ratio ≈ 1 at p75–p95). Any earlier claim that
normalisation was needed to fix a unit mismatch was wrong.

**What normalisation does change: the decision problem, not the units.** It
replaces an absolute criterion with a relative one. That is justified only if the
absolute scale is unstable across the states the policy visits. Measured, it *is*
unstable — but for a reason normalisation cannot fix:

| depth | support-EVOI median |
|---|---|
| 0 | +0.108152 |
| 1 | +0.001507 |
| 2 | +0.021201 |
| 3 | −0.000322 |
| 4 | −0.000405 |
| 5 | −0.000566 |

The statistic moves by **two orders of magnitude across depth** *because the
support collapses*, not because of a scale convention. A scale-free rule does not
recover a graded information measure here; it re-expresses a purity indicator.
This is why §E6 shows the normalised variants landing on the same ~20 operating
points as the raw rule.

**One normalisation is actively pathological.** `normalised_ig = IG / H(p_s)`
divides by an entropy that goes to zero as the support becomes pure — precisely
where the decision matters. When `H → 0` with `IG > 0` the ratio diverges to
+∞, which reads as "infinitely worth asking" and *inverts* the intended
behaviour at exactly the states where stopping should be easiest. In the
implementation the divergence is real: at τ = 0.5 the rule asks **zero**
questions on every episode, the worst outcome in the whole grid. The good
`normalised_ig` numbers in E6 (τ ∈ [0.001, 0.01]) are obtained in the region
where the pathology does not fire, so they should not be read as evidence that
the normalisation is sound.

**`reward_per_evoi = 1/EVOI` is not a distinct rule at all.**
`1/EVOI < τ ⟺ EVOI > 1/τ`, so it is a raw threshold in a reparameterised constant.
E6 confirms it: τ = 3 reproduces the full-length reference exactly and τ ≥ 10
collapses to zero questions, with nothing in between. It adds no information and
should not be presented as an alternative.

**`percentile` calibration buys nothing on this data, for a diagnosable reason.**
Because the distribution is bimodal (§E4), a quantile of it jumps between modes
rather than trimming a tail. The val-calibrated q=50 lands at 0.001507 — inside
the flat region — and reproduces the raw τ=0.001 behaviour *exactly* (3.31
questions, 0.9408 reward). q=75 jumps straight past the cliff to 1.43 questions
and 100% premature stops. Percentile thresholding is therefore **unstable here**,
and its apparent advantage (scale-free, no units) is not realised.

### E9. The grid is not a continuum, so "pick a λ" is the wrong question

Across all rules and all thresholds tested, the grid realises **20 distinct
behaviours**. The 1e-05 … 0.001 region is five thresholds producing identical
episodes. Framing V-6 as "which value in [0, 0.05]" invites a false choice between
numbers that mean the same thing, while the actual decision — stay in the flat
region or cross the cliff — is a single binary judgement.

---

## OPTIONS

**Option 1 — Fixed-length only; no stopping rule in V-6.**
V-6 covers the cost parameter for *item selection* at a pre-declared budget B, and
adaptive stopping is deferred. AB-7 (fixed-length vs adaptive) is then reported
with the stopping arm explicitly out of scope.

**Option 2 — Raw absolute threshold on the raw support-posterior EVOI.**
Keep `beta_greedy` as implemented: `STOP iff max_j gain_j < λ`. λ is in Brier
units per question, commensurate with the reward, and needs no normalisation.

**Option 3 — Raw absolute threshold on the neural-predictor EVOI.**
Same rule, but the stopping statistic is computed from the calibrated predictor
instead of the training support. Dominates Option 2 on every measured axis in E6
and is the estimator already in the reward's units.

**Option 4 — Normalised / scale-free statistic** (relative EVOI, normalised IG, or
uncertainty-reduced-per-question), with the threshold set by percentile or by a
utility target.

**Option 5 — Defer: change the objective before choosing a threshold.**
Recognise that every option above is choosing a *purity-detection operating
point* on a cohort whose labels make the objective saturate, and first fix the
objective (or adjudicate on Polish, V-4/V-7) so that "worth asking" means
something other than "the support is not yet pure".

---

## TRADE-OFFS

| | advantages | disadvantages |
|---|---|---|
| **1 · fixed-length** | No threshold to calibrate, so no unvalidated normalisation. Cleanest separation of item selection from stopping (AB-7). Removes the decision from V-6 entirely. | Forgoes a measured gain: E6 shows rules reaching 0.9482 reward at 3.27 questions against 0.9428 at 6.00. No cost–utility frontier, so §11.1's frontier requirement stays unmet. Cannot answer "how long is the interview?" at all. |
| **2 · raw support EVOI** | Already implemented, tested (24 tests), and shipped in the λ sweep. Dimensionless and commensurate with the reward, so no normalisation is needed. Sits inside the flat region for any λ ≤ 0.001, so the choice is forgiving. | The statistic is **analytically non-positive on a pure support** (E4), so the policy is a purity detector, not a cost–benefit rule. Zero premature stops are obtained only by dropping 2.7 questions. λ > 0.001 is catastrophic (81.9% premature stops). Sits in the support-posterior regime, which §4 of `diagnosisReady.md` shows collapses on this cohort. |
| **3 · predictor EVOI** | Best measured trade-off in E6: 0.9467 reward at 4.04 questions, 3.2% premature stops, Brier 0.053, ECE 0.045 — it *beats* the full-length reference. Non-degenerate on pure supports, so the criterion stays informative where the support one goes to zero. Same units as the reward. | Introduces a dependency on the trained predictor, so V-6 would no longer be predictor-independent — the property that made `results/lambda_sweep_saudi.json` valid across the P1-b retrain. Ties the stopping rule to predictor quality, and the predictor is itself trained on circular labels. Adds a model to the critical path of a gate decision. |
| **4 · normalised** | Scale-free, so in principle portable across cohorts and instruments — the property that would matter on Polish. Percentile calibration needs no unit argument at all. | **Not justified by the measurement here**: §E8 shows the raw scale is already commensurate, and the instability normalisation targets is caused by support collapse, not by units. `normalised_ig` is *pathological* where `H → 0` (τ = 0.5 → zero questions, worst result in the grid). `reward_per_evoi` is algebraically a raw threshold, not a new rule. Percentile calibration is unstable on a bimodal distribution. Selecting among these on this cohort would be choosing an unvalidated normalisation. |
| **5 · defer** | Addresses the root cause: the objective saturates because labels are a sum-threshold, so *no* threshold on this cohort can be validated as a sufficiency criterion. Polish is the only cohort that could adjudicate it. Prevents a premature, unfalsifiable sign-off. | Blocks V-6 indefinitely behind V-4/V-7. Leaves the project unable to state any interview-length or cost claim. Discards a measured, reproducible improvement (Option 3) that is at least internally consistent. |

Cross-cutting: Options 1 and 5 are the only ones that avoid putting an
unvalidated operating point into a signed artifact. Options 2 and 3 require
accepting that the resulting "stopping policy" encodes label saturation.
Option 4 additionally requires accepting a normalisation that §E8 shows is
unnecessary for units and unstable in practice.

---

## RECOMMENDED EXPERIMENT

**Do not choose a threshold yet. Run the discriminating experiment first.** The
question "which λ?" is unanswerable on this cohort because the statistic is a
purity indicator (E4) and the grid is a cliff, not a continuum (E9). The
experiment that *would* discriminate the options is the one that breaks the
circularity:

> **E-X. Repeat this entire analysis on the Polish cohort (V-4 / V-7), whose
> labels are clinician-established and therefore do not produce a pure support
> by construction.**

Specifically, on Polish, with the split and predictor re-derived:

1. Recompute the purity profile by depth. **Predicted under the saturation
   account:** the support stays label-mixed far longer and the depth-4/5 pure
   fraction falls well below the 0.803 / 0.921 measured on Saudi. **If it does
   not fall, the saturation account is wrong** and the Saudi result needs a
   different explanation.
2. Recompute the `max_j gain_j` distribution. **Predicted:** unimodal and
   positive, with no 44.9% mass at ≤ 0, so the statistic becomes a genuine
   graded information measure and an absolute threshold becomes meaningful.
3. Re-run the E5 threshold grid and the E6 sensitivity table. **Predicted:** a
   smooth stop-rate curve with many distinct operating points instead of 20, so
   that "which λ" becomes a real question. **If the curve is still a cliff, the
   stopping-rule framing is wrong regardless of cohort.**
4. Re-check whether Options 2 and 3 still rank as in E6. **If the predictor-EVOI
   advantage disappears once the support no longer collapses, then Option 3's
   advantage was an artefact of the saturated regime**, and the V-6 pack should
   say so.

E-X is blocked on V-4/V-7 and **cannot be run on the current data.** Two
interim steps are available now and do not require the Polish cohort:

- **E-Y (runnable now, ~1 hour).** Five-seed repeat of the E5/E6 analysis. Every
  number here is a **single seed (0) on 127 episodes**, and the differences
  between neighbouring operating points are ~0.002 reward. This establishes
  whether the ranking of the options is stable at all, and produces the error
  bars the V-6 note currently lacks. If the ranking is not stable, no threshold
  should be signed on this evidence regardless of which option is chosen.
- **E-Z (runnable now).** Ablate the Beta prior. The entire positive signal at
  shallow depth comes from the `1/(n+2)²` dilution term (E4). Re-running with
  `BETA_A0 = BETA_B0 → 0` should drive the pure-support mass from 44.9% to
  exactly 50% (pure supports contribute exactly 0) and confirm the mechanism
  directly. This is a mechanism check, not a candidate policy.

**V-6 remains open. No threshold is selected in this note, and none should be
inferred from the ordering of the rows in E6** — those rows are ordered by rule
family, not by recommendation.

---

## Related documents

- `scripts/step8_evoi_scale_analysis.py` — the analysis; selects no threshold
- `results/evoi_scale_saudi.{json,csv}` — measured distributions and sensitivity
- `V6_LAMBDA_DECISION.md` §2, §2.1 — corrected units discussion and retraction
- `diagnosisReady.md` §0A.1 — why the objective saturates at λ = 0
- `POLICY_BENCHMARK_REPORT.md` §A.6 — the saturation finding
- `Master-Project-Specification_FINAL.md` §11.1, §24 (V-6)
