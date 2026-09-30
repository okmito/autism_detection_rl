---
title: "Master Project Specification — Build From Scratch"
project: "Adaptive RL for Explainable Autism Detection"
status: "Research specification — implementation-ready after blocking methodological gates"
version: "v5 — final anomaly cleanup of the August 2026 master specification"
prepared: "29 August 2026"
---

# Adaptive RL for Explainable Autism Detection
## Master Project Specification

**What this document is.** A single, self-contained specification sufficient to build this entire project from an empty repository — no other document needs to be consulted. It contains the research framing (why, what question, what's novel), the literature grounding, and the implementation-level detail (repository layout, state encoding, episode protocol, reward formula, predictor, audits, baselines, explanation module, evaluation, config schema, test suite) in one place. Hand this file to a coding agent or a developer and nothing else should be required to start.

**System function:** adaptive screening and specialist-referral recommendation. **Not a diagnostic tool.** The registered project title uses "Detection"; the system itself performs screening only, and every output, log line, and interface label says so.

**Team**

| Name | Registration Number |
|---|---|
| Mitesh Sai Devar | 23BCE8242 |
| Y. Darahaas Reddy | 23BCE8629 |
| Rupak Vivek Sai Oleti | 23BCE8279 |
| Srinidhi Sanka | 23BCE8773 |

**Guide:** Dr. Allapati Rajya Lakshmi

**Document map.** Part I (§1–4) — what and why. Part II (§5–6) — literature and novelty. Part III (§7–22) — the build. Part IV (§23–27) — timeline, gates, limitations.

---

# Part I — What We're Building and Why

## 1. One-paragraph summary

Build a system that, given a child's answers to a screening questionnaire revealed one at a time, decides which question to ask next and when to stop, then issues a specialist-referral recommendation with a calibrated risk score and a complete decision trace. The primary methodological experiment compares learned and heuristic sequential policies with a **finite-sample empirical optimum** whenever that optimum can be computed exactly for the same state representation, question-count budget, acquisition costs, and scoring objective. The project measures the computational tractability boundary, evaluates policies at matched question-count budgets, tests transfer from questionnaire-derived labels to clinician-established diagnoses, and evaluates the faithfulness of intrinsic decision traces against a clearly separated post-hoc prediction-explanation baseline. The project is a research benchmark, not a diagnostic or clinical-deployment system.

## 2. The problem

Autism screening instruments are commonly administered item by item, even when earlier responses already make some later questions less informative. The Q-CHAT-10 has ten items and the full Q-CHAT has 25; larger autism assessment batteries are substantially longer. The practical motivation here is therefore not to claim that adaptive assessment itself is new, but to study **how much adaptive value is available under a fixed question budget and how closely learned policies approach an exactly solved reference**.

**The bottleneck is assessment burden and decision efficiency, not simply classifier accuracy.** A classifier that is more accurate on a completed instrument does not by itself reduce the number of questions presented. Sequential item acquisition explicitly optimizes both information gained and the cost of asking additional questions.

Computerized adaptive testing is already established in autism assessment. The 2026 CAT-Autism study developed a bifactor multidimensional-IRT CAT from 490 candidate caregiver-report items, retained 424 informative items, and reported a mean of 13 administered items with AUC 0.95 for ages 1–5 and 0.94 for ages 6–18 against clinician-assigned diagnostic status. This removes any defensible claim that adaptive item selection for autism is absent from the literature. citeturn589590search0

A second directly relevant precedent is Ardulov et al. (2021), *Scientific Reports*, which formulated diagnostic classification as a sequential decision process and used Q-learning to recommend which clinical item to query next while jointly seeking diagnostic accuracy and fewer queried items in an ASD-versus-ADHD setting. The present project therefore makes no novelty claim for RL-based sequential autism assessment itself. Its proposed distinction is the exact finite-sample reference, matched-budget benchmark, label-source/construct-shift evaluation, and explicit audit framework. citeturn786433search0turn786433search2

The open question pursued here is narrower: **under a small, explicit item budget, what is the gap between learned/heuristic sequential acquisition policies and an exactly solved finite-sample optimum, and does any advantage survive transfer from questionnaire-defined labels to clinician-established diagnoses?**

### 2.1 Formal scope and assumptions

The exact reference is defined over a **binary acquisition representation**, a finite **question-count** budget B, known per-item acquisition costs that enter the utility penalty, each item asked at most once, a deterministic acquisition policy at evaluation time, and a terminal utility based on a proper scoring rule. Raw multi-level questionnaire responses are preserved, but the primary Q-CHAT-10 experiment uses the explicit binary mapping locked in §15. The main tractability calculation assumes complete records and counts only item-observation states; covariates are excluded from that closed-form count. Real structural missingness is handled separately through the three-state representation in §9. The exact reference is **empirically optimal for the declared training sample, state representation, question budget, costs, and objective**; it is not claimed to be clinically or population optimal.

## 3. Research questions

- **RQ1 (tractability — primary).** For each instrument size and **question-count budget**, when does exact optimization of the declared sequential objective exceed the pre-declared resource limit, and how closely do learned and heuristic policies approach the exact finite-sample reference where it is computable?
- **RQ2 (adaptivity value).** At the same question-count budget, does adaptive acquisition improve held-out screening utility relative to the **exact best fixed subset**, RFE, and a generic IRT-based CAT implemented on the same instrument?
- **RQ3 (transfer).** How does any adaptive advantage change when moving from questionnaire-derived labels to clinician-established diagnoses in the Polish cohort?
- **RQ4 (explanation faithfulness).** How faithfully does the intrinsic sequential decision trace explain the policy's acquisition/stopping behaviour, and how does that differ from what a matched post-hoc SHAP explanation can explain about the final prediction?
- **RQ5 (subgroup).** Does item-selection behaviour differ by sex, age band, or their interaction, and are those differences accompanied by materially different performance or assessment burden?

## 4. Hypotheses

**Validity check (not a hypothesis).** When the learned policy is evaluated against the exact reference using the **same empirical value function**, its empirical value cannot exceed the exact optimum. A reported violation indicates an implementation or evaluation error. This statement applies only to the finite-sample reference problem, not to held-out clinical performance.

> **Implementation status (checked 2026-10-01):** the validity check above was executed as an automated regression test in `tests/test_step5_benchmark.py::test_no_policy_beats_exact_optimum`. Across budgets B∈{1..6} and all four policies (greedy, random, DQN, PPO) on the Saudi train split, **no violations were found** — the exact reference is behaving as specified. Test-count and per-run detail in `POLICY_BENCHMARK_REPORT.md`.

- **H1.** At each pre-declared question-count budget of 3–6 items included in the confirmatory family, the primary adaptive policy will achieve lower held-out Brier loss than the exact best fixed-subset policy on the primary Tier-1 dataset, with both methods evaluated at exactly `B` questions. Results on additional Tier-1 datasets are secondary unless included in the pre-registered comparison family.
- **H2.** The adaptive-versus-fixed difference in held-out Brier loss observed on questionnaire-derived labels will be smaller when the resulting policies are evaluated on the clinician-established Polish cohort; the transfer gap is reported with uncertainty.

**Publishable null result:** failure to outperform the exact best fixed subset or generic CAT is a valid result; so is exact optimization remaining computationally feasible throughout the tested range. Explanation results are treated as exploratory and do not determine whether the main research question succeeds.

# Part II — Literature Grounding

## 5. Prior work, by cluster

**Item reduction for autism instruments.** Earlier work reported very small fixed subsets for ADI-R and ADOS, but later replication work raised concerns about generalisability and class imbalance. Sollis, Wall & Washington (2025, *Scientific Reports* 15:39091) studied compact Q-CHAT-10 subsets, training on New Zealand and Saudi Arabian datasets with questionnaire-derived labels and testing on a Polish cohort with clinician-established diagnoses. Their reported Saudi-trained four-item model reached AUROC 87% ± 11 at the 0.3 threshold, with sensitivity 84% and specificity 80%. The study therefore provides a directly relevant **static feature-selection reference**, not a sequential adaptive-policy benchmark. citeturn589590search1

**Closest domain-specific prior art — CAT-Autism.** Tseng et al. (2026), *JAMA Network Open*, developed and validated CAT-Autism using a bifactor MIRT model. The study considered 490 caregiver-report items, retained 424 items satisfying the model criterion, and reported mean administration of 13 items, with AUC 0.95 for ages 1–5 and 0.94 for ages 6–18 against clinician-assigned diagnosis. The method demonstrates that adaptive autism assessment is already a real and clinically oriented research direction. It is therefore treated here as the **closest domain-specific prior art**, while the project differs in objective, instrument size, exact-reference methodology, and label-source-shift analysis. citeturn589590search0

**Sequential feature acquisition and RL.** Prior work frames sequential cost-sensitive feature acquisition as a reinforcement-learning problem, including policy-gradient and DQN-style approaches. In autism-related work specifically, Ardulov et al. (2021) used Q-learning to learn which clinical items to query next in an ASD-versus-ADHD setting while minimizing queried information. This confirms that RL-based adaptive clinical item selection is established prior art. citeturn786433search0 The present project therefore treats RL as a method component, not as the novelty claim.

**Optimal decision trees / exact sequential policies.** Under the assumptions in §2.1, a deterministic adaptive binary-question policy with a fixed maximum depth is representable as a binary decision tree. DL8.5 and MurTree are optimal decision-tree methods, but their standard objectives are centered on classification-tree criteria such as misclassification/accuracy rather than automatically matching this project's exact Brier-plus-cost objective. STreeD and related dynamic-programming formulations broaden the class of objectives that can be optimized when the objective satisfies the required separability conditions. Consequently, the project will **not assume in advance** that an off-the-shelf tree solver constitutes the exact reference; V-2 requires a formal compatibility check. citeturn554941search0turn554941search3turn554941search6

**Explainability in autism ML.** Prior autism-ML literature contains extensive post-hoc feature-attribution work, including SHAP-based studies, but the project treats explainability as a comparison of **explanation targets** rather than as a claim that SHAP and a sequential trace are identical objects. The intrinsic trace explains which question the policy selected and when it stopped; SHAP explains the contribution of features to a matched static prediction. RQ4 therefore evaluates acquisition/stopping faithfulness specifically.

**Dataset validity.** Some public questionnaire datasets use labels derived from the same questionnaire responses used as predictive features. This creates **label circularity**, which is distinct from ordinary preprocessing leakage. Both are therefore audited separately. The Polish Q-CHAT dataset is materially different: its labels are clinically established and are separate from the Q-CHAT responses used as features. The published 2025 study describes the Polish dataset as 252 toddlers and explicitly distinguishes its clinical labels from the Q-CHAT-derived labels in NZ and Saudi data. citeturn589590search1turn168169search1

**Ethics and construct validity.** The project does not infer that questionnaire items are neutral or universally valid measures of autistic traits. Circularity findings are accompanied by a construct-validity discussion, and the absence of participatory validation with autistic people is retained as an explicit limitation rather than treated as unavoidable.

## 6. Novelty position — provisional, gated on a systematic search

**No final novelty claim is permitted before V-2 (§24) is completed and logged.** The project uses the following pre-registered novelty position:

- **Primary methodological contribution:** a reproducible benchmark of learned, heuristic, and classical adaptive policies against an **exact finite-sample optimum under the same declared sequential objective**, wherever that optimum is computationally tractable.
- **Secondary contribution:** a measured **tractability frontier** showing where exact optimization becomes infeasible under pre-declared state, memory, and wall-clock limits.
- **Secondary contribution:** explicit **label-source shift analysis**, separating questionnaire-derived training labels from clinician-established evaluation labels.
- **Secondary contribution:** a dual audit for **label circularity and preprocessing leakage** plus an intrinsic-vs-post-hoc explanation-faithfulness analysis.

The project makes **no novelty claim** that adaptive item selection for autism or RL-based sequential clinical item selection is new. CAT-Autism establishes modern autism-specific CAT, and Ardulov et al. establish an earlier Q-learning formulation for sequential autism-related clinical item selection. citeturn589590search0turn786433search0

The project's novelty is also **not** an original RL algorithm. DQN/PPO, sequential feature acquisition, IRT-CAT, and optimal decision-tree methods are used as established methodological components.

The provisional literature claim is deliberately limited and concerns the **combination of experimental elements**, not the individual methods:

> **Preliminary-search claim:** the reviewed literature did not identify, in the sources examined so far, an autism-screening study that combines (i) an exactly solved finite-sample sequential reference under the same declared objective, (ii) matched comparisons with learned, heuristic, fixed-subset, and CAT policies, (iii) an explicit tractability boundary, and (iv) questionnaire-to-clinical label/construct-shift evaluation. This is a provisional claim and is not finalized until V-2 is completed.

| Dimension | Rating | Basis |
|---|---|---|
| Algorithmic originality | Low | Uses established RL, AFA, IRT-CAT, and exact-tree/DP methods |
| Problem formulation originality | Moderate | Exact finite-budget policy benchmark + explicit label/construct shift |
| Domain novelty | Low–Moderate | Adaptive autism assessment already exists; the novelty is in the benchmark/formulation |
| Evaluation rigour | High | Exact reference, matched budgets, external clinical-label cohort, circularity/leakage audits, power/FWER plan, subgroup and faithfulness analyses |
| Practical impact | Low–Moderate | Research benchmark; no clinical deployment |

# Part III — Complete Build Specification

## 7. Component build order

| # | Component | Spec | Depends on |
|---|---|---|---|
| 1 | Data layer (ingest, validate, dedupe) | §15 | Nothing — build first |
| 2 | Circularity audit (blocking) | §16.1 | Data layer |
| 3 | Leakage audit (blocking) | §16.2 | Data layer |
| 4 | Environment (state, actions, reward, episode loop) | §9–11 | Data layer |
| 5 | Predictor (scores partial observations) | §12 | Environment |
| 6 | Exact solver | §14 | Environment, Data layer |
| 7 | Learned policy (DQN / PPO) | §11.2 | Environment, Predictor |
| 8 | Baselines 1–10 | §17 | Environment, Predictor |
| 9 | Explanation module | §18 | Predictor, Policy |
| 10 | Evaluation harness | §19 | All of the above |
| 11 | Ablation suite | §20 | Environment, Policy |
| 12 | Reporting | §19.6 | Evaluation harness |

Nothing in row *n* starts before its dependencies are built and tested.

## 8. Repository layout

```
project-root/
├── src/
│   ├── data/
│   │   ├── ingest.py           # one loader per dataset tier (§15)
│   │   ├── schema.py           # validates the common schema in §15
│   │   └── dedupe.py           # cross-source duplicate detection (§15)
│   ├── audits/
│   │   ├── circularity.py      # §16.1 — blocking
│   │   └── leakage.py          # §16.2 — blocking
│   ├── env/
│   │   ├── state.py            # three-state item encoding (§9)
│   │   ├── environment.py      # episode loop (§10), legal actions, termination
│   │   └── costs.py            # per-item cost model (§11.1)
│   ├── solvers/
│   │   ├── exact_adapter.py    # optional STreeD / DL8.5 / MurTree cross-checks (§14)
│   │   └── exact_custom.py     # authoritative backward-induction DP (§14)
│   ├── models/
│   │   └── masked_predictor.py # §12
│   ├── policies/
│   │   ├── dqn.py              # §11.2
│   │   ├── ppo.py
│   │   ├── greedy.py
│   │   ├── random_policy.py
│   │   ├── irt_cat.py
│   │   ├── dqn_cat.py
│   │   └── static_rfe.py
│   ├── explain/
│   │   ├── trace.py            # §18.1
│   │   ├── counterfactual.py   # §18.2
│   │   └── shap_baseline.py    # §18.3
│   ├── eval/
│   │   ├── metrics.py          # §19.1
│   │   ├── bootstrap.py        # §19.2
│   │   ├── power.py            # §19.3
│   │   ├── fwer.py             # §19.3
│   │   └── subgroup.py         # §19.4
│   └── ablation/
│       └── runner.py           # §20
├── configs/                    # Hydra YAML (§13)
├── tests/                      # mirrors src/ 1:1 (§21)
├── results/                    # MLflow store — git-ignored
├── data/                       # raw + processed — git-ignored
└── README.md
```

Every file exists before Checkpoint 1 closes, even as a stub with `NotImplementedError` and a docstring, so people can build in parallel without merge conflicts on shape.

## 9. Item state encoding — the single most important implementation detail

Every item, at every point in an episode, is in exactly one of three conditions:

| Condition | Meaning | Legal action? |
|---|---|---|
| `UNASKED` | Present in the source record but not yet revealed | Yes |
| `OBSERVED(v)` | Revealed response category/value `v` | No — already asked |
| `MISSING` | Structurally absent from the source record | **No, never** |

```text
mask[j]  ∈ {0, 1, 2}     # 0=UNASKED, 1=OBSERVED, 2=MISSING
value[j] = canonical response representation when mask[j] == 1
```

For the **primary Q-CHAT-10 benchmark**, the response representation is binary and follows the published mapping in §15. For Q-CHAT-25 and any other multi-category instrument, `value[j]` is a categorical response representation with item-specific cardinality `m_j`; the raw response is preserved separately.

The **primary exact and learned benchmark excludes age/sex from the policy state** so that exact and learned policies use the same state representation. Age and sex remain in the dataset schema for subgroup analysis. A covariate-conditioned policy, if later studied, is a separate sensitivity experiment and is not mixed into the primary optimality-gap analysis.

**Network input:** one-hot `mask` (3×*n*) ⊕ one-hot/categorical response representation ⊕ normalised question budget remaining. For binary items this is `4*n + 1`; for heterogeneous categorical items, the response portion has `Σ_j m_j` dimensions.

**Subgroup note:** age and sex are not used to define the primary exact-reference value.

**Hard rule:** `UNASKED`, `MISSING`, and every observed response category have distinct representations and distinct legal-action semantics. An observed zero/category index must never be used to encode absence.

**State-count qualification:** for a complete instrument with item-specific response cardinalities `m_j`, the theoretical number of observation states reachable with question budget B is:

```text
Σ_{S ⊆ {1..n}, |S| ≤ B}  Π_{j ∈ S} m_j
```

If every item has the same `m` response categories, this reduces to `Σ_{k=0..B} C(n,k) m^k`. The binary Q-CHAT-10 case therefore has `m=2`; the multi-category Q-CHAT-25 calculation must use its actual item cardinalities rather than the binary formula. Structural missingness can only reduce the reachable action/state space relative to the complete-record reference.

## 10. Episode protocol

```text
function RUN_EPISODE(record, question_budget B, B_min, policy, predictor):
    state ← INIT_STATE(record)
    questions_remaining ← B
    items_asked ← 0

    loop:
        legal_items ← { j : mask[j] == UNASKED }
        stop_legal ← (items_asked ≥ B_min)

        if legal_items is empty:
            stop_legal ← true

        legal ← legal_items ∪ ({STOP} if stop_legal else {})
        if questions_remaining == 0:
            action ← STOP
        else:
            action ← policy(state, legal)

        if action == STOP:
            p_hat ← predictor(state)
            y ← record.true_label
            R ← (1 − (p_hat − y)^2) − λ · Σ c_j for j asked
            EMIT decision = (p_hat ≥ τ ? "REFER" : "NO_REFERRAL_INDICATED")
            EMIT trace = ordered list of (item, value, belief_before, belief_after)
            EMIT counterfactual = CF(state, p_hat)
            return (decision, p_hat, trace, counterfactual, R)

        belief_before ← predictor(state)
        v ← record[action]
        mask[action] ← OBSERVED
        value[action] ← v
        questions_remaining -= 1
        items_asked += 1
        state ← UPDATE_STATE(mask, value, questions_remaining)
        belief_after ← predictor(state)

`STOP` is legal only after `B_min` items have been asked, except when no legal unasked item remains. `STOP` is forced when the question-count budget reaches zero. Item costs affect the terminal utility but do not change the primary question-count budget.

**Operating threshold:** the primary internal operating point is pre-declared as `τ = 0.5`; threshold-independent metrics (AUROC/AUPRC) are primary for cross-dataset ranking. Any secondary threshold is fixed before the external Polish outcome is viewed and is never tuned on Polish test labels.

**Exact-reference note:** the learned episode uses the masked predictor in §12. The exact policy does not depend on that neural predictor. Exact and learned policies are compared for optimality-gap purposes under the common empirical evaluator defined in §14.

### 11.1 Reward

```text
R = (1 − (p_hat − y)^2) − λ · Σ_{j ∈ asked} c_j
```

- **First term** — one minus Brier loss for a binary referral probability. The underlying Brier score is strictly proper for probabilistic forecasts; the acquisition-cost penalty makes the overall objective a sequential cost-sensitive utility.
- **`c_j`** — per-item acquisition cost. Uniform cost is the primary setting (`c_j = 1`); non-uniform costs are the predefined cost ablation.
- **`λ`** — trade-off coefficient. Its pre-declared grid is specified at implementation start (V-6) and is not tuned using the external Polish labels.
- **`B_min`** — minimum number of asked items before `STOP` becomes legal.
- Intermediate rewards are zero; only the terminal reward is emitted.

Two evaluation frontiers are kept separate:

1. **Budget-performance curve:** fix each budget `B` and compare policies at matched `B`.
2. **Cost-utility frontier:** vary `λ` and report the resulting utility/acquisition-cost trade-off.

The second is not relabelled as a “budget curve.”

### 11.2 Learned policy architecture

Double DQN is the primary learned policy; PPO is the pre-registered algorithmic ablation.

```text
Input:  4n + 1  (§9, primary benchmark)
   ↓
Hidden: 256 units, ReLU
   ↓
Hidden: 128 units, ReLU
   ↓
Output: n+1 Q-values (n item-actions + STOP)
```

Illegal actions are masked to `−∞` before selection. The predictor is frozen by default. No larger model is treated as a source of novelty.

## 12. Predictor (masked classifier)

The predictor scores a partially observed state and is used by learned policies and held-out evaluation. It is **not** the source of the exact reference value.

- **Input:** the same primary `4*n + 1` vector as §9.
- **Architecture:** MLP, 128 → 64, ReLU, single sigmoid output.
- **Training:** on training records with randomly masked subsets sampled across budgets 0…*n*, subject to the dataset's actual missingness pattern.
- **Output:** calibrated probability `p̂ ∈ [0,1]` for the referral-positive class.
- **Calibration:** isotonic or Platt calibration fitted only on training/validation data; report ECE and Brier before and after calibration.
- **Freeze:** frozen by default before policy training; joint training is AB-5.
- **All model fitting is fold-local:** no calibration, scaling, masking-derived statistics, or other fitted transform may use validation or external-test rows.

## 13. Config schema (Hydra)

```yaml
dataset:
  tier: T1 | T2
  name: nz | saudi | uci_child | uci_adol | uci_adult | polish

env:
  question_budget: int
  b_min: int
  cost_mode: uniform | weighted

reward:
  lambda_grid: [...]      # PENDING V-6 (§24)
  scoring: brier

predictor:
  hidden: [128, 64]
  calibration: isotonic | platt
  freeze: true

policy:
  type: exact_dp | dqn | ppo | greedy | random | irt_cat | dqn_cat | static_rfe

eval:
  seeds: 10
  bootstrap_resamples: 2000
  operating_threshold: 0.5
  policy_state: questions_only
  primary_metric: brier
  fwer: holm_bonferroni

exact:
  authoritative_solver: custom_dp
  objective: empirical_brier_minus_cost
  max_states: 50000000
  max_hours: 24
  max_ram_gb: 32
```

Every run writes config hash, git commit SHA, and random seed to MLflow. No run is reported without this triple.

## 14. Exact solver strategy

### 14.1 Authoritative exact reference

The authoritative exact reference is a **custom backward-induction dynamic program** over the finite training sample and the declared budget. The exact state consists of the observation pattern, its consistent training-support set `D_s`, and the remaining **question-count budget**. Accumulated acquisition cost is derived from the asked-item set and enters utility; it is not a second budget in the primary experiment. At a terminal state `s` with support count `N_s > 0`, define the empirical positive-class probability:

```text
p_emp(s) = mean(y_i for i in D_s)
```

and the terminal empirical Brier utility:

```text
U_stop(s) = 1 − mean((p_emp(s) − y_i)^2 for i in D_s)
```

The exact Bellman recursion is then:

```text
V*(s, b) = max(
    U_stop(s)                  if STOP is legal at this state,
    max_{j in legal(s)} [ -λ c_j + Σ_v P_emp(v | s,j) V*(T(s,j,v), b-1) ]
)
```

where `b` is the remaining question count, `STOP` is legal only when the number of observed items in `s` is at least `B_min`, and stopping is forced when `b = 0` or no legal item remains. Zero-probability response branches are skipped. Any state with `N_s = 0` is unreachable under the empirical transition model. This recursion is the **definition of the exact finite-sample reference**.

### 14.2 Role of external exact-tree solvers

STreeD, DL8.5, and MurTree are used as **cross-checks or accelerators only when their supported objective/constraint formulation is formally equivalent to §14.1**. They are never silently substituted for the authoritative exact objective.

In particular:

- If a solver optimizes misclassification or accuracy rather than the declared Brier-plus-cost objective, its result is **not** called the exact reference for this project.
- A solver may be reported as an **exact surrogate** when it is itself exact for a different objective.
- A heuristic or anytime result is reported as an **approximation**, not as an exact optimum.
- No fallback changes the definition of “exact.” If the authoritative DP cannot finish within the declared resource limits, that run supplies a **tractability-boundary observation**.

This distinction is required because DL8.5 and MurTree are established optimal classification-tree algorithms, but their standard formulations do not automatically imply exact optimization of this project's custom sequential Brier-plus-cost objective. citeturn554941search0turn554941search3turn554941search6

### 14.3 Optimality-gap definition

For a learned deterministic policy `π`, calculate:

```text
Gap(π) = V*(s0, B) − V_emp(π; D_train, B, λ)
```

where both terms use **the same empirical transition probabilities, terminal empirical Brier utility, budget, cost model, and λ**. The learned predictor is used to train/select actions but is not used to redefine the exact reference value. `Gap(π)` therefore measures approximation to the finite-sample reference problem.

Separately report held-out value and Polish clinical performance. Those are generalisation measures, not optimality-gap measures.

### 14.4 Tractability measurement

At every exact run record wall-clock time, peak memory, number of unique reachable states, and the number of memoized state-action evaluations. Resource limits are:

- 5×10⁷ unique states,
- 24 hours wall-clock,
- 32 GB RAM.

Exceeding a limit is not treated as a software failure if the run is cleanly terminated and the boundary is logged; it is tractability data for RQ1.

### 14.5 Reference state counts

For a complete instrument with item-specific response cardinalities `m_j`, the number of observation states reachable with question budget `B` is:

`Σ_{S ⊆ {1..n}, |S| ≤ B} Π_{j ∈ S} m_j`.

For the primary **binary Q-CHAT-10** representation (`n=10`, `m=2`) and `B=6`, this is exactly `26,025`.

For the canonical Q-CHAT-25 response structure, 24 items have five response categories and item 4 has a sixth “my child does not speak” category. Under that representation, the exact theoretical count at `B=6` is `3,081,146,397`. The implementation must use the actual source encoding rather than assume homogeneous item cardinality. citeturn537865search3turn537865search12 citeturn389770search13turn389770search14

| Instrument / representation | Items | Budget | Reference state count | Interpretation |
|---|---:|---:|---:|---|
| Q-CHAT-10 binary | 10 | 6 | 26,025 | Complete-record theoretical reference |
| Q-CHAT-25 canonical response cardinalities | 25 | 6 | 3,081,146,397 | Exact theoretical count for 24 five-category items plus item 4 with six response categories |
| ADI-R binary hypothetical | 93 | 6 | 50,494,563,219 | Theoretical extrapolation only; no direct item-level run planned |

**Arithmetic is generated programmatically.** These values must be emitted by a test/calculation script at runtime rather than treated as hand-entered constants. The 93-item figure is intentionally theoretical and must not be presented as an empirical ADI-R result.

## 15. Datasets — exact build requirements

| Tier | Dataset | Size | Labels | Access / status | Role |
|---|---|---:|---|---|---|
| 1 | Q-CHAT-10, New Zealand | 1,054 (728/326) | Questionnaire-derived, threshold-based | **[PENDING V-1]** | Primary T1 training cohort |
| 1 | Q-CHAT-10, Saudi Arabia | 506 (341/165) | Questionnaire-derived, threshold-based | **[PENDING V-1]** | Second training source; supports static benchmark reproduction |
| 1 | UCI AQ-10 children | 292 | Questionnaire-derived | CC BY 4.0 (verified in current spec) | Supplementary; exercises missingness if confirmed in loader |
| 1 | UCI AQ-10 adolescent | 104 | Questionnaire-derived | CC BY 4.0 (verified in current spec) | Descriptive / supplementary |
| 1 | UCI AQ-10 adult | 704 | Questionnaire-derived | CC BY 4.0 (verified in current spec) | Supplementary; not pooled with toddler data |
| 2 | Q-CHAT-25, Poland | 252 in the published evaluation; the publication contains a conflicting 135+118 class-count statement that sums to 253 | Clinician-established diagnosis | **[PENDING V-1]** | Primary external clinical evaluation cohort |
| 3 | ADI-R item-level (AGRE/NDA/SFARI) | Varies | Clinician-established | Controlled access | Stretch only — go/no-go at Checkpoint 4 |

**Important source clarification:** the Polish source is a 25-item Q-CHAT dataset. The 2025 publication reports 252 cases and describes 135 diagnosed with autism; its text contains a denominator discrepancy between 118 and 117 typically developing cases in different places. This project must resolve the actual row count during V-7 by loading the public data and locking the final denominator before any result is reported. citeturn168169search0turn168169search1

**Common internal schema.** `src/data/ingest.py` implements one loader per row, each returning:

```python
{
  "item_responses": np.ndarray,     # shape (n,), values in the dataset's canonical item scale or NaN
  "label": int,                     # {0,1}
  "label_source": str,              # "questionnaire" | "clinical"
  "covariates": {"age_band": str, "sex": str},
  "missing_mask": np.ndarray        # shape (n,), True where genuinely absent
}
```

For the common sequential environment, questionnaire responses are transformed into a **documented binary acquisition representation** only when the source item itself is binary or when a pre-registered binarisation rule is justified and applied consistently. For the primary Q-CHAT-10 experiment, use the binary response mapping explicitly reported by Sollis et al.: for questions 1–9, Sometimes/Rarely/Never → 1 and Always/Usually → 0; for question 10, Always/Usually/Sometimes → 1 and Rarely/Never → 0. The raw responses remain preserved for auditability, while the exact-state benchmark operates on this declared binary representation. Any alternate ordinal representation is a sensitivity analysis and cannot silently replace the primary representation. citeturn552351view0

No downstream code touches dataset-specific formats directly.

**Primary Q-CHAT-10 representation lock.** The loader must verify the declared binary mapping before modeling. The overall Q-CHAT score used to define questionnaire-derived labels is not passed to the model as a feature. citeturn552351view0

**Instrument-role lock.** Q-CHAT-10 is the primary fully specified binary benchmark. Q-CHAT-25 is the multi-category extension used for tractability and, where data permit, sequential-policy evaluation with its canonical item-response encoding. Results from the two instruments are never pooled as if they had the same response representation.

Permanently excluded: facial-image datasets and neuroimaging data. This is a structural research-scope decision, not a claim that those modalities are universally invalid for autism research.

## 16. Audits — both blocking, both before any modelling

### 16.1 Circularity audit

**What it tests:** whether the dataset's label is deterministically or near-deterministically generated from the same questionnaire items supplied to the model.

**Procedure:**

1. Reconstruct the documented score/threshold rule where known.
2. Fit a sum-threshold oracle over the same item responses for plausible thresholds.
3. Report exact-match rate, UAR, and sensitivity/specificity where defined.
4. Classify circularity as:
   - **Deterministic:** label exactly generated by the same features and rule.
   - **Near-deterministic:** exact-match ≥ 0.99 but not demonstrably the published deterministic rule.

**Gate behaviour:** a circular dataset is not used to support a claim of independent clinical screening validity. It may still be used for policy-optimization and algorithm-comparison experiments, with the circularity status displayed next to every relevant result.

### 16.2 Leakage audit

**What it tests:** whether any fitted transform, calibration object, feature-selection step, hyperparameter selection, or learned model has used information from outside its permitted training fold.

**Procedure:**

1. Instrument every fitting operation with fold provenance.
2. Assert that no fitted object is applied to a held-out row before the model-selection stage is closed.
3. Run a deliberately poisoned control in which a transform is fitted on the full dataset and verify that the audit fails.

**Gate behaviour:** hard stop. Any confirmed leakage violation invalidates the affected experiment until rerun.

### 16.3 Construct-validity note

Where labels are circular and questionnaire items themselves define the label, results are explicitly described as performance against the **questionnaire-defined construct**, not independent evidence of clinical autism diagnosis. The discussion should avoid treating individual behavioural items as universally valid or culturally neutral indicators.

## 17. Baselines — ten, with build notes

All runnable baselines use identical train/validation/test splits, matched **question-count budgets**, missingness rules, seeds, and metric definitions unless a method intrinsically requires a different representation. Baselines are grouped by objective so unlike objectives are not conflated.

| # | Baseline | Build note |
|---|---|---|
| 1 | Full-observation classifier | Standard supervised classifier using the complete available instrument. It is a full-information reference, **not** an information-theoretic upper bound. |
| 2 | Sum-threshold oracle | Circularity diagnostic only; never optimized against and never treated as clinical evidence on circular datasets. |
| 3 | Exact finite-sample optimal policy | Authoritative reference defined in §14. |
| 4 | Exact best fixed subset | Exhaustive subset search over all subsets of size exactly B on the training data under the same empirical utility as §14. The primary fixed-subset comparison therefore isolates item-selection value at a common full length; adaptive-stopping value is tested separately in AB-7. |
| 5 | Static RFE subset | Fixed subset per budget, selected only on training data, then evaluated without further feature selection. |
| 6 | Random selection | Uniform random choice among legal items; repeat over the same seed set. |
| 7 | Greedy information gain | One-step lookahead using `IG(s,j)=H(Y|s)-Σ_v P(v|s,j)H(Y|s,j,v)` under the empirical training state distribution. This is a heuristic, not an exact policy. |
| 8 | Generic IRT-CAT on the project instrument | Unidimensional 2PL CAT fitted only on training data; initialise ability with a pre-declared EAP prior, select the unasked item with maximum expected Fisher information, and stop only at the matched question-count budget. This is not labelled as a reproduction of CAT-Autism. |
| 9 | RL-based CAT (DQN) | Same action/state machinery as the primary DQN, but with a pre-declared psychometric reward: terminal negative posterior ability-estimation variance, with no Brier term. The reward definition and IRT estimator are frozen before training. |
| 10 | Published four-item Q-CHAT benchmark, reproduced | Reproduce Sollis et al.'s published static four-item pipeline on the public NZ/Saudi/Polish data. Report discrepancies between paper and reimplementation rather than substituting the published headline value. citeturn589590search1 |

### 17.1 Q-learning autism precedent (context, not matched baseline)

Ardulov et al. (2021) is a contextual prior-art reference for RL-based sequential item selection in autism-related clinical classification. It is not a matched baseline because its task, labels, instrument, and robustness objective differ from this project. Its inclusion prevents the project from overstating novelty in RL-based adaptive questioning. citeturn786433search0

### 17.2 Published CAT-Autism reference (context, not matched baseline)

CAT-Autism is reported as a **published contextual reference**, not a matched runnable baseline, because its 490-item bank, MIRT calibration, age-specific item sets, and termination procedure are materially different from the small public instruments used in the main experiment. Its reported performance should therefore be cited for literature positioning, not statistically pooled with the project's matched-budget Q-CHAT experiments. citeturn589590search0

## 18. Explanation module

### 18.1 Decision trace (primary, intrinsic)

Every episode emits:

```python
[
  {"step": 1, "item": "A7", "value": 1, "belief_before": 0.42, "belief_after": 0.71},
  {"step": 2, "item": "A3", "value": 0, "belief_before": 0.71, "belief_after": 0.58},
  ...,
  {"stop_reason": "policy_stop" | "budget_exhausted", "final": 0.58}
]
```

This is intrinsic: it records the policy's actual acquisition and stopping behaviour. Belief-before is computed from the current predictor state before the selected action; belief-after is computed from the new state after the observed response.

### 18.2 Counterfactual

For the terminal state, search observed items for the minimum single-response change that flips the **declared operating decision threshold `τ`**. Report the minimum-cost or minimum-count counterfactual according to the pre-declared tie-break rule. If none exists, report that the decision is robust to every single observed-item flip.

### 18.3 SHAP comparison baseline

SHAP is applied only to a matched static classifier using the **same observed item subset** available to the policy at termination. It is a comparison explanation, not the system's intrinsic explanation.

### 18.4 Explanation target alignment

RQ4 distinguishes two targets:

- **Acquisition/stopping faithfulness:** does the explanation identify the items that actually drove which question was asked next and whether the policy stopped?
- **Prediction faithfulness:** does the explanation identify features whose perturbation materially changes the final risk estimate?

The primary comparison is the first target because it matches what the decision trace natively represents. SHAP is evaluated on the same terminal observed-feature set, but it is not treated as a direct explanation of the sequential action-selection mechanism.

## 19. Evaluation and statistics

### 19.1 Metrics

| Metric | Implementation | Why |
|---|---|---|
| **Brier / terminal utility** | `brier_score_loss` and declared terminal utility | **Primary confirmatory endpoint aligned with the optimization objective** |
| Sensitivity / specificity | Confusion-matrix derivation at pre-declared threshold | Screening relevance |
| AUROC / AUPRC | `roc_auc_score` / `average_precision_score` | Threshold-independent ranking quality |
| UAR | `balanced_accuracy_score` | Primary screening operating-point metric |
| ECE | documented calibration bins | Probability calibration |
| Acquisition burden | Mean / median items asked and stopping distribution | Direct adaptive-assessment objective |
| **Empirical optimality gap** | `V*(s0,B) − V_emp(π;D_train,B,λ)` | Approximation to the finite-sample exact reference |
| Explanation faithfulness | Pre-specified deletion/insertion and action-prediction metrics | RQ4 |

Every reported performance figure states its question budget and carries a 95% CI.
The **primary confirmatory endpoint for H1 is held-out Brier loss at a matched maximum question-count budget**. For the confirmatory H1 comparison, both adaptive and fixed-subset arms use the same fixed-length evaluation mode (exactly `B` questions); adaptive stopping is evaluated separately in AB-7. Terminal utility (`1 − Brier − λ·acquisition_cost`) is reported separately as the decision-objective measure. UAR is the primary threshold-dependent screening metric; AUROC/AUPRC and ECE are secondary endpoints.

### 19.2 Confidence intervals

Use paired bootstrap with 2,000 resamples on matched evaluation instances to quantify **evaluation-sample uncertainty**. Across the 10 pre-declared random seeds, report mean and seed dispersion separately; do not select a best seed. Do not call seed dispersion a 95% CI unless a specific hierarchical procedure is implemented and documented.

### 19.3 Power and multiple comparisons

- **MDE:** compute the minimum detectable effect for each confirmatory comparison **before** the Polish cohort is opened (V-4).
- **Primary confirmatory family:** pre-declare the exact budget(s), dataset(s), policy comparison(s), and endpoints that directly test H1–H2. Apply Holm–Bonferroni to that family. Do not make an existential claim such as “at least one dataset” unless the associated multiplicity is included in that confirmatory family.
- **Secondary/exploratory metrics:** still report 95% CIs, but do not create an enormous undifferentiated p-value family by multiplying every metric, budget, tier, and exploratory subgroup into one omnibus correction.
- The exact family and hypotheses are frozen at Checkpoint 3 before the external cohort is accessed.

### 19.4 Subgroup and faithfulness analyses

- **Subgroup (RQ5):** UAR, acquisition burden, and stopping distribution by sex and age band. The sex×age interaction is exploratory and is reported only when subgroup cell sizes meet the pre-declared minimum; otherwise it is described as underpowered rather than interpreted.
- **Faithfulness (RQ4):** primary endpoint is action/stopping faithfulness. Any exploratory trace-vs-SHAP difference is accompanied by a paired CI and an explicit statement that the explanation targets differ. No clinical superiority claim is made from this comparison.

### 19.5 Published benchmark comparison

The published Sollis et al. numbers are **not** treated as a standalone statistical comparator. Baseline 10 reimplements their model using the same public data and produces predictions on the same Polish individuals as the project's models. It first reproduces the published feature set and preprocessing, including the demographic variables used by the paper. Because that feature set is not identical to the primary question-only benchmark, the original reproduction is a contextual reference; a question-only harmonized reproduction must be used for any direct matched comparison. Any model-comparison test is performed on paired subject-level predictions or paired loss contributions where the statistical procedure is valid. It is never a comparison of one AUROC with a published mean ± SD. Threshold-dependent metrics are compared only when the threshold-selection protocol is equivalent. The published 0.3 and 0.5 operating points are reported for reproduction where applicable. AUROC is threshold-independent; thresholds matter for sensitivity, specificity, UAR, and binary decisions. The project's primary operating threshold is fixed independently at 0.5 unless an alternative is pre-registered. citeturn589590search1

### 19.6 Reporting outputs

Produce:

- performance-vs-budget curves;
- cost-utility vs `λ` frontiers;
- exact-solver tractability curves (time/memory/reachable states vs `n` and `B`);
- empirical optimality-gap plots with support counts;
- ten-baseline comparison tables;
- subgroup tables;
- explanation-faithfulness curves;
- circularity and leakage audit reports;
- a reproduction table for the published Q-CHAT benchmark.

## 20. Ablation suite

| ID | Ablation | Compares |
|---|---|---|
| AB-1 | Reward scoring rule | Negated Brier vs plain 0/1 correctness |
| AB-2 | State encoding | Three-state missing-aware vs two-state missing-collapsed |
| AB-3 | λ grid granularity | Coarse vs fine cost-utility sweep |
| AB-4 | Policy algorithm | Double DQN vs PPO |
| AB-5 | Predictor training | Frozen vs jointly trained |
| AB-6 | Cost model | Uniform vs pre-declared non-uniform costs |
| AB-7 | Stopping | Fixed-length adaptive selection vs adaptive selection plus stopping under the same maximum B |

Each ablation reports its exact question budget, λ setting, seeds, and whether the change affects the training objective or only the evaluation protocol. AB-6 changes acquisition cost only; it does not redefine the primary question-count budget. AB-7 isolates the value of adaptive stopping from the value of adaptive item ordering. In the fixed-length arm, the policy must select exactly B items and cannot stop early. The ablation suite therefore contains seven pre-declared ablations.

## 21. Test suite — minimum required

| Test | Asserts |
|---|---|
| `test_state_encoding` | `UNASKED ≠ MISSING ≠ OBSERVED(0)` and round-trip correctness |
| `test_legal_actions` | `MISSING` never legal; `STOP` illegal below `B_min` **unless no legal unasked item remains**, in which case `STOP` is forced |
| `test_budget_exhaustion` | No policy can exceed the declared budget; stop is forced when no legal question remains |
| `test_reachable_state_count` | For the complete binary reference, `n=10, B=6` gives exactly 26,025; the multi-category Q-CHAT-25 cardinality formula is also checked independently |
| `test_reward_bounds` | `R ∈ [−λ·Σ_{j asked} c_j, 1]` for each valid episode under the declared cost model |
| `test_predictor_partial_input` | Predictor returns a valid probability for every valid mask configuration, including all-UNASKED |
| `test_exact_optimality` | On a synthetic toy problem, the custom DP value equals or exceeds every enumerated legal deterministic policy value under the same empirical objective |
| `test_exact_value_consistency` | Exact-policy value and independently re-evaluated policy value match under the same empirical state evaluator |
| `test_no_leakage` | A fold-fitted transformer never sees held-out rows; poisoned control is caught |
| `test_circularity_oracle` | Known synthetic deterministic threshold labels are detected as deterministic circularity |
| `test_counterfactual_validity` | Every returned counterfactual flips the operating decision when applied, or is explicitly marked robust/no-flip |
| `test_threshold_freeze` | External test thresholds cannot be optimized after the Polish labels are accessed |
| `test_common_empirical_evaluator` | Exact and learned policies use the same empirical evaluator for optimality-gap calculations |
| `test_trace_belief_update` | Recorded belief-before/after values equal predictor outputs on the corresponding states |
| `test_exact_fixed_subset` | Exhaustive fixed-subset oracle matches the best enumerated fixed subset on a toy problem |

## 22. Tech stack

```
Python 3.11+
PyTorch                          # predictor (§12) + DQN/PPO (§11.2)
Stable-Baselines3 or CleanRL     # RL algorithm implementations
scikit-learn                     # classical baselines, metrics, RFE
catsim / girth / mirtCAT         # Baseline 8 (§17)
SHAP                             # explanation comparison baseline (§18.3)
STreeD / DL8.5 / MurTree         # optional exact-tree cross-checks only (§14)
NumPy / pandas
Hydra                            # config management (§13)
MLflow                           # run tracking
Docker                           # reproducibility
pytest                           # test suite (§21)
```

No dependency outside this list without updating this document first. Solver-specific dependencies may be installed in isolated containers only if the reproducibility manifest records their exact version and command-line interface.

---

# Part IV — Timeline, Gates, Limitations

## 23. Ten-week roadmap, four checkpoints

| Checkpoint | Week | Gate | Ships |
|---|---:|---|---|
| **1** | 1 | **Blocking.** Circularity + leakage audits run; V-1 licensing and V-2 literature/systematic-search protocol completed | Repo scaffold, state encoding, tests, random/greedy baselines, evaluation skeleton |
| **2** | 3 | Exact empirical reference attempted at 10 and 25 items; if a resource limit is hit, the boundary is logged rather than replacing exactness with a different objective | Exact-DP implementation, predictor trained/frozen, baseline suite |
| **3** | 6 | **Blocking.** MDE + confirmatory comparison family filed before Polish data are opened | Learned policy, Tier-1 ablations, explanation module |
| **4** | 9 | External cohort evaluated under frozen protocol; Tier-3 go/no-go decided | Full comparison, subgroup/faithfulness analyses, reproducibility report |

The Polish external cohort is opened exactly once for confirmatory evaluation. Developmental choices that require seeing its outcomes are forbidden after the gate is closed.

## 24. Outstanding verification — blocking items

| Item | Blocks | What's needed |
|---|---|---|
| V-1 | Checkpoint 1 | Read and record the actual access/licence terms for NZ, Saudi, and Polish data and freeze the exact data artifacts used |
| **V-2** | **Checkpoint 1; all novelty claims** | Complete a systematic literature search; record every database/search engine, exact query string, search date, screening criteria, duplicate handling, and included/excluded studies; verify whether prior work contains the same exact finite-budget objective/reference formulation; formally position the project against CAT-Autism, the 2025 Q-CHAT compact-subset work, sequential feature-acquisition RL, and exact decision-tree methods |
| V-4 | Checkpoint 3 | Pre-compute MDE for each confirmatory endpoint and finalize the confirmatory comparison family |
| V-5 | Checkpoint 2 | Verify literature/evidence concerning item-order effects; until then, treat order-independence as an explicit modelling assumption and interpret adaptivity results as an upper bound under that assumption |
| V-6 | Start of implementation | Supervisor sign-off on the λ grid and threshold policy |
| V-7 | Checkpoint 3 | Recompute Baseline 10 from public data and resolve the Polish denominator discrepancy before locking results |
| V-9 | Checkpoint 1 | Confirm institutional ethics-review requirement for retrospective secondary analysis |
| V-10 | Checkpoint 4 | Tier-3 controlled-access go/no-go decision |

No novelty statement should use the phrases **“first,” “only,” “no prior work,” or “absent from the literature”** until V-2 is completed. After V-2, any “first” claim must be tied to a documented search scope and dated search log.

## 25. Ethical and scope constraints — hard rules, checked before merge

- No facial-image data in this project.
- No neuroimaging data; this is a structural scope decision because the main episode abstraction requires sequentially revealable questionnaire items.
- No real respondent interaction at any stage; training and evaluation are retrospective and de-identified.
- No clinical deployment code, patient-data upload, or live diagnostic/reporting interface.
- Subgroup differences are logged and reported, never suppressed.
- Terminology lock: **screening**, **referral recommendation**, and **risk estimate**; never represent the system output as a diagnosis.
- The Polish cohort is treated as an external clinical evaluation cohort, not as a resource for tuning thresholds or model selection.

## 26. Deliverables checklist

- [ ] Source repository, containerised, matching §8 layout
- [ ] Circularity audit report (§16.1)
- [ ] Leakage audit report (§16.2)
- [ ] Ingestion pipeline for all approved datasets in §15
- [ ] Exact-DP implementation and tractability measurements (§14)
- [ ] All ten runnable baselines, with CAT-Autism and Ardulov et al. treated as contextual prior art rather than falsely matched baselines
- [ ] Trained policy checkpoints + configs + seeds in MLflow
- [ ] Performance-vs-budget and cost-utility-vs-λ curves with 95% CIs
- [ ] Empirical optimality-gap figures with per-state support counts and common empirical evaluator
- [ ] Subgroup audit — sex, age band, interaction
- [ ] Explanation-faithfulness comparison, aligned to the acquisition/stopping target
- [ ] Seven-ablation results table
- [ ] Reproducibility check: fresh clone → one command → matching results under the declared environment
- [ ] PRISMA-style literature search log — blocks final novelty wording (V-2)
- [ ] Reproduction report for Sollis et al., including the paper's reported-vs-recomputed discrepancy table

## 27. Known limitations — state these, don't hide them

- Primary training datasets use questionnaire-derived labels; only the Polish cohort supplies clinician-established labels in the current public-data plan.
- The strongest exact-state regime is small (10 items) under the primary binary representation; retaining the multi-category Q-CHAT-25 response scale produces a much larger theoretical state space and may make exact optimization infeasible; the computational regime where adaptive selection is potentially most valuable may be unreachable with public item-level data.
- The order-independence assumption is not yet established for these instruments. If question order changes responses or reporting behaviour, retrospective recorded answers cannot fully reproduce the causal effect of administering questions adaptively. Until V-5 is resolved, the results should be interpreted as an **order-invariant counterfactual simulation** rather than evidence of causal benefit from changing administration order.
- The exact reference is optimal only for the declared finite training sample, state representation, costs, and objective. It is not a claim of clinical optimality or population optimality.
- The exact reference and learned policies use different training mechanisms: the exact arm uses the empirical state evaluator, while the learned arm uses a masked predictor. This is intentional. The optimality gap uses the common empirical evaluator; predictor-versus-empirical reward discrepancy is reported separately to identify possible surrogate exploitation.
- The Polish sample is small and drawn from specialist clinical settings; external performance estimates will therefore have wide uncertainty and may not generalise to population screening. Differences among NZ, Saudi, and Polish data also include recruitment, geography, language, and construct/label-source changes, so performance differences cannot be attributed to label source alone.
- The published 2025 Q-CHAT study contains a denominator/reporting discrepancy for the Polish control count and different uncertainty figures in different sections. This project must report that discrepancy rather than silently selecting a preferred value. citeturn589590search1
- No clinician-in-the-loop validation of the explanation interface is included.
- No participatory design with autistic people is included; this is an explicit limitation.
- CAT-Autism uses a much larger and psychometrically different item bank. Its strong results limit claims about domain novelty but do not constitute a matched-head-to-head comparison with this small-instrument benchmark. citeturn589590search0

---

## External verification notes used for this revision

- Tseng et al. (2026), *JAMA Network Open*: CAT-Autism uses a 490-item caregiver-report bank, retains 424 items under the bifactor MIRT model, and reports mean 13-item administration with AUC 0.95 (ages 1–5) and 0.94 (ages 6–18). citeturn589590search0
- Ardulov et al. (2021), *Scientific Reports*: Q-learning was used for sequential clinical item selection in an ASD-versus-ADHD task, with the stated goal of maximizing diagnostic accuracy while minimizing queried information. citeturn786433search0
- Sollis, Wall & Washington (2025), *Scientific Reports*: four-item Q-CHAT-10 models trained on NZ/Saudi data were evaluated on 252 Polish toddlers with clinician-established labels; the paper reports Saudi AUROC 87 ± 11 and contains an internal discrepancy in the Polish control count and in reported AUROC dispersion. citeturn589590search1turn168169search0
- DL8.5, MurTree, and dynamic-programming optimal-tree work establish exact decision-tree optimization methods, but do not justify assuming exactness for this project's custom sequential Brier-plus-cost objective without a formal compatibility proof. citeturn554941search0turn554941search3turn554941search6

*This is the canonical, self-contained specification. A coding agent or developer should be able to build the project from this document alone once blocking gates V-1, V-2, V-4, V-5, V-6, V-7, V-9, and V-10 are resolved. If any build detail changes during implementation — especially the reward function, exact-value definition, solver, dataset schema, threshold protocol, or metric definitions — update this document in the same sitting and regenerate any derived configuration, test, and reporting artifacts from it.*

- Allison et al. (2008) defines Q-CHAT as a 25-item questionnaire with ordered response options; the primary Q-CHAT-25 tractability calculation therefore does not assume binary items. citeturn389770search13turn389770search14
