# RL ADAPTIVE QUESTIONNAIRE — RESEARCH AND IMPLEMENTATION REPORT

**Date:** 2026-11-13 · Branch `neurobridge-development` · nothing committed.
**Claim boundary:** retrospective replay of recorded Q-CHAT answers in one case-control cohort. Research prototype;
not a diagnosis, not a diagnostic device, not prospective validation.

---

## 1. Current RL architecture and its limitations (audit, verified against the code on this branch)

| audit finding | status on this branch | evidence |
|---|---|---|
| The served app uses greedy information gain, not a trained RL policy | **confirmed** (before this work) | `app/services/screening.py::make_policy` served `GreedyIGPolicy` / `RandomPolicy` / `BetaGreedyPolicy`; no learned checkpoint was loaded |
| DQN and PPO exist as offline research implementations | **confirmed** | `src/policies/dqn.py`, `src/policies/ppo.py`; artifacts in `results/policies_v3*` trained at fixed budgets on Saudi data |
| Greedy often has no meaningful information-gain choice | **confirmed** | `src/screening/diagnostics.py`: on the circular Saudi label the empirical support becomes pure; IG is identically 0 at 195 of 195 pure-support states (test split, B=6), so `argmax` returns the lowest unasked index |
| The app asks a fixed six questions | **confirmed** | `configs/config.yaml env.question_budget: 6`; `engine.next_question` never offers STOP |
| Adaptive stopping is disabled | **confirmed** | `config.yaml` policy comment; `is_exhausted` is the stopping rule in the app, apart from the user's manual stop |
| Saudi labels are derived from the questionnaire answers | **confirmed, exactly** | `Class == 1[A1+...+A10 >= 4]` holds for 506 of 506 rows (re-checked) |
| External evaluation showed no reliable advantage of adaptive over fixed-length selection | **confirmed** | `docs/CORRECTED_DESIGN_RESULTS.md`: at matched budgets all 8 Holm-corrected adaptive-vs-random intervals include 0 |

Further limitations of the existing RL code found in this audit:

* **State:** `encode_state` (mask one-hot + answer bits + budget fraction) has no belief / uncertainty feature.
* **STOP was never learnable:** every arm ran with `b_min = B`, so the episode ended at the budget.
* **Reward against a circular label:** `R = 1 - (p_hat - y)^2` with the Saudi `y`, which is a deterministic function of
  the answers. A policy optimising it can at most learn to reproduce the scoring rule.
* **PPO is not PPO:** `PPOPolicy.train_step` uses the normalised raw reward as the advantage (no return, no critic
  baseline), i.e. a REINFORCE variant, and on-policy learning is the least sample-efficient choice for a few hundred
  records.
* **DQN (existing)**: Double-DQN update rule is correct (masked next actions, terminal rows not bootstrapped), but the
  per-sample Python encoding is too slow for nested cross-validation, and the persisted v3 artifacts have the largest
  optimality gap of all arms at B=5 (`docs/CORRECTED_DESIGN_RESULTS.md` §6).

## 2. The MDP

Implemented in `src/rl/mdp.py`.

**State.** A per-item code vector `h ∈ {0,1,2,3}^10`: 0 not asked, 1 answered typical, 2 answered atypical, 3 asked
but skipped ("prefer not to answer"). From `h` the agent sees: which items were asked, their answers, questions used
(skips count, they cost the respondent a question), remaining budget, and the belief `P(screen-positive | h)` from the
belief model. The DQN input is the one-hot code (40) + belief log-odds/4 + used/B + remaining/B (43 features). Nothing
known solely after the episode (label, unasked answers) is in the state.

**Actions.** `ASK j` for any item with code 0, and `STOP`. An asked item is never legal (enforced by `legal_mask`, by
`ReplayBatch.step`, by `rollout`, and in the app by `rl_apply`). STOP is legal from `min_questions = 2` on and forced at
`max_questions = 10` (safety cap) or when no item is left.

**Transitions.** In training and evaluation the answer to `ASK j` is the participant's recorded answer (replay of real
data). Under the belief model, `P(x_j = 1 | h) = θ_j1·P(y=1|h) + θ_j0·P(y=0|h)`.

**Belief model** (`src/rl/belief.py`). Naive Bayes with Beta(1,1) smoothing gives closed-form posteriors on any
partial history and closed-form answer predictions. Q-CHAT items are correlated, so the raw evidence sum is
over-confident; one evidence weight `w` (log-odds = logit(π) + w·ΣLLR) is fitted by logistic regression on
**cross-fitted** partial histories of the training data (5 folds, 20 random histories per record, sizes 1..10).
Skipped and unasked items contribute nothing (exact marginalisation under the model).

## 3. Reward formulation and justification

Three formulations were compared on their merits:

| approach | what it optimises | assessment |
|---|---|---|
| **A. Scalar** `q(final) - c·#questions` | one trade-off point per `c` | simple and standard, but `c` is arbitrary and has no clinical meaning by itself |
| **B. Constrained** min E[#questions] s.t. quality ≥ full − δ | the actual research question | needs a quality metric and a margin; solved by Lagrangian relaxation |
| **C. Pareto** sweep of `c` | the whole trade-off curve | descriptive; does not pick an operating point |

**Recommendation and implementation: B as the primary formulation, solved through A, reported with C.**
For a fixed `c`, maximising `E[q] − c·E[#questions]` is the Lagrangian of problem B, so the agent is trained with the
scalar reward (A) for every `c` on a grid, and `c` is then chosen by the constraint (B): the fewest mean questions
whose inner-cross-validated balanced accuracy is within `delta_select = 0.02` of the full questionnaire's. The whole
grid is reported as a Pareto frontier (C). `c` is therefore never "chosen and assumed optimal"; it is the multiplier
that makes the constraint bind, selected without touching test data.

Terminal quality `q`:

* **primary: decision accuracy** `1[decision(p_hat, τ=0.5) == y]`. Its expectation is the probability of a correct
  screening decision, so `E[q]` is the quantity the constraint is stated in (on this near-balanced cohort, 54% ASD,
  accuracy ≈ balanced accuracy). Its consequence is the behaviour we want from a screener: a question is worth asking
  when it can change the referral decision, which is stochastic curtailment learned rather than hand-coded.
* **sensitivity analysis: Brier** `1 − (p_hat − y)^2`, a proper scoring rule that also rewards calibration and ranking.

Per-question cost `−c` is the burden term. "Penalties for poor stopping decisions" are built into `q`: stopping on the
wrong side of the threshold earns 0 instead of 1. An asymmetric false-negative cost is not implemented;
the reward is symmetric pending a supervisor decision (§12).

**Why this does not just reproduce the scoring formula:** `y` in the primary experiment is the clinician-established
diagnosis group, not a function of the answers. On the Saudi data (label = `1[sum ≥ 4]`), the same reward would make the
optimal policy deterministic curtailment of the sum score, which is exactly what development runs on Saudi showed
(value iteration kept BA 0.994 with 4.3 questions on average); that is a property of the label, not of RL, and Saudi
results are therefore used for debugging and as an exploratory transfer arm, nothing more. A reward cannot repair circular
ground truth.

## 4. Algorithms selected and why

Two RL agents, chosen to answer different questions:

1. **Model-based RL — exact value iteration on the learned belief MDP** (`src/rl/planner.py`). The environment model
   (belief + answer predictions) is learned from training data; with 10 binary items every history is one of 4^10 =
   1,048,576 codes, so the Bellman equation is solved exactly by backward induction in about 0.4 s. It is Bayes-optimal
   *under the learned model*, deterministic, and its value function decomposes exactly into answer branches, which is
   what makes its explanations exact. Weakness: inherits the naive-Bayes independence assumption.
2. **Model-free RL — Double DQN on replayed real histories** (`src/rl/dqn.py`). Learns from actual answer sequences and
   actual labels, so it does not share the model's assumptions. Same update rule as the existing `DQNPolicy`, applied to
   the new MDP, vectorised, with **cross-fitted** terminal rewards (a training record's belief comes from a model that
   did not see it) and a 3-seed mean-Q ensemble for deterministic, lower-variance inference. Hyper-parameters were set
   on Saudi development data before the protocol was frozen (16 updates/epoch, 400 epochs, 64-64 MLP).

The existing PPO was not adapted (see §1); an on-policy method adds variance without a reason on this data size.

## 5. Training and evaluation methodology

* **Data audit.** Saudi (n=506): labels circular (above), used for development/debugging and an exploratory transfer
  arm. Polish (n=252, 135 ASD / 117 control, toddlers 18-24 months): labels clinician-established and independent of
  the answers (best sum-score threshold agrees with the label for 86.9% of children — the circularity audit classifies it "Not circular"; the official cutoff sum ≥ 4 agrees for 84.5%; AUROC of the sum 0.937); Q-CHAT-25 projected to Q-CHAT-10 binary
  with the verified contract; one missing value in an item not used by Q-CHAT-10. It is a **case-control** sample, so
  prevalence (54%) is far from a screening population's, PPV/NPV do not transfer, and n=252 limits precision. No
  participant overlap between cohorts (existing leakage audit).
* **Role change of the Polish cohort.** Earlier steps used it as a sealed external test set for Saudi-trained scorers.
  Here it is the development cohort under nested cross-validation — the one cohort with independent labels. This is
  recorded in the protocol as a decision requiring supervisor ratification (§12).
* **Pre-registration.** `scripts/step18_rl_adaptive_protocol.py` froze grids, selection rule, margin, tests and the
  integration gate, with a SHA-256 of the `src/rl` code, before Polish was analysed with this code;
  `scripts/step19_rl_adaptive_experiment.py` refuses to run if the code changed.
* **Nested CV.** Outer stratified 5-fold (each participant tested once by models that never saw them); inner stratified
  3-fold inside each outer-training set selects `c` (RL) or the confidence threshold (heuristic) by the constrained rule;
  refit on the outer-training set; evaluate on the outer fold. Every arm is scored by the same outer-training belief
  model, so differences come from which and how many questions are asked.
* **Primary test.** Non-inferiority of balanced accuracy vs the full 10-item questionnaire (same scorer): paired
  bootstrap (5000 resamples) 95% CI lower bound > −δ, δ = 0.05; one-sided bootstrap p-values, Holm across the two RL
  agents. Sensitivity analyses at δ = 0.02, 0.03, 0.10.
* **Integration gate (pre-registered).** G1 non-inferiority after Holm; G2 mean questions CI upper bound < 10; G3
  sensitivity-difference CI lower bound > −0.10; G4 engineering tests. If both agents pass, value iteration is
  preferred (exact explanations, no seed ensemble).

## 6. Baseline comparison (nested CV, Polish n=252, each child tested once)

Source: `results/rl_adaptive/experiment_results.json` (protocol frozen 2026-11-13T17:16Z, code sha `d101e770…`,
runtime 19 min). All arms scored at STOP by the same outer-training belief model; τ = 0.5.

### 6.1 Primary arms

| arm | BA | sens | spec | AUROC | Brier | mean Q | median Q | reached cap |
|---|---|---|---|---|---|---|---|---|
| full questionnaire (10 items, same scorer) | 0.864 | 0.822 | 0.906 | 0.937 | 0.101 | 10.00 | 10 | 100% |
| official Q-CHAT-10 cutoff (sum ≥ 4, 10 items) | 0.850 | 0.785 | 0.915 | 0.937 | — | 10.00 | 10 | 100% |
| **RL value iteration** (model-based, selected c) | **0.851** | 0.822 | 0.880 | 0.900 | 0.120 | **4.20** | 3 | 6% |
| **RL DQN** (model-free, selected c) | 0.838 | 0.830 | 0.846 | 0.885 | 0.125 | 3.50 | 3 | 0% |
| threshold stopping (non-RL adaptive, selected hi) | 0.860 | 0.822 | 0.897 | 0.907 | 0.110 | 5.37 | 4 | 24% |

Stopping points (number of children): value iteration 2:63, 3:80, 4:42, 5:7, 6:13, 7:12, 8:4, 9:15, 10:16;
DQN 2:91, 3:53, 4:38, 5:34, 6:33, 7:3; threshold stopping 2:41, 3:25, 4:94, 5:8, 6:4, 7:12, 9:7, 10:61.

### 6.2 Paired differences vs the full questionnaire (bootstrap 95% CI, 5000 resamples)

| agent | ΔBA | Δsens | Δspec | ΔAUROC | Δquestions |
|---|---|---|---|---|---|
| RL value iteration | −0.013 [−0.036, +0.010] | 0.000 [−0.029, +0.030] | −0.026 [−0.065, +0.009] | **−0.038 [−0.064, −0.015]** | **−5.80 [−6.10, −5.50]** |
| RL DQN | −0.026 [−0.059, +0.007] | +0.007 [−0.031, +0.046] | **−0.060 [−0.114, −0.008]** | **−0.053 [−0.082, −0.025]** | −6.50 [−6.68, −6.32] |
| threshold stopping | −0.004 [−0.019, +0.009] | 0.000 [0.000, 0.000] | −0.009 [−0.038, +0.018] | **−0.031 [−0.056, −0.008]** | −4.63 [−4.98, −4.26] |

Against the official sum ≥ 4 rule, value iteration: ΔBA +0.001 [−0.028, +0.032], Δsens +0.037 [−0.007, +0.082],
Δspec −0.034 [−0.076, 0.000], with 5.8 fewer questions.

### 6.3 Primary test (non-inferiority of balanced accuracy, δ = 0.05, Holm over the two RL agents)

| agent | CI lower bound | one-sided p | Holm p | non-inferior at δ=0.05 | at δ=0.03 | at δ=0.02 |
|---|---|---|---|---|---|---|
| RL value iteration | −0.036 | 0.0018 | **0.0036** | **yes** | no | no |
| RL DQN | −0.059 | 0.083 | 0.083 | no | no | no |
| (threshold stopping, not in the family) | −0.019 | — | — | yes | yes | yes |

### 6.4 Comparisons that isolate what RL adds

| comparison | ΔBA | Δquestions | reading |
|---|---|---|---|
| value iteration − threshold stopping | −0.009 [−0.028, +0.010] | **−1.17 [−1.39, −0.98]** | VI asks 1.2 fewer questions; BA difference not resolved |
| value iteration − DQN | +0.013 [−0.017, +0.044] | +0.70 [+0.42, +0.97] | model-based not resolvably better on BA; asks 0.7 more |
| DQN − threshold stopping | −0.022 [−0.053, +0.008] | −1.87 [−2.21, −1.54] | DQN loses specificity (−0.051 [−0.099, −0.008]) |

**Matched burden.** Against fixed-length arms at k = 4 (≈ value iteration's mean of 4.2), ΔBA was +0.007 (best fixed
subset), +0.021 (official order), +0.018 (random), +0.021 (belief-greedy IG), +0.003 (existing greedy IG), −0.001
(existing beta-greedy); **every 95% interval includes 0**. Adaptivity at the same average length did not resolvably
beat fixed-length questionnaires on balanced accuracy.

### 6.5 Fixed-length baselines (BA / AUROC at k questions; all from the same nested CV)

| k | official order | best subset (fwd sel.) | random | belief-greedy IG | greedy IG (existing) | beta-greedy (existing) | DQN v3 (existing, Saudi-trained) |
|---|---|---|---|---|---|---|---|
| 2 | 0.785 / 0.789 | 0.819 / 0.842 | 0.795 / 0.844 | 0.823 / 0.822 | 0.831 / 0.830 | 0.829 / 0.839 | 0.780 / 0.812 |
| 3 | 0.844 / 0.856 | 0.822 / 0.879 | 0.818 / 0.870 | 0.861 / 0.883 | 0.816 / 0.861 | 0.815 / 0.865 | 0.812 / 0.848 |
| 4 | 0.830 / 0.896 | 0.844 / 0.906 | 0.833 / 0.886 | 0.830 / 0.900 | 0.848 / 0.902 | 0.852 / 0.909 | 0.793 / 0.857 |
| 5 | 0.862 / 0.907 | 0.831 / 0.911 | 0.842 / 0.913 | 0.856 / 0.905 | 0.852 / 0.906 | 0.860 / 0.915 | 0.828 / 0.897 |
| 6 | 0.866 / 0.913 | 0.824 / 0.923 | 0.833 / 0.914 | 0.856 / 0.915 | 0.857 / 0.920 | 0.857 / 0.931 | — |
| 8 | 0.863 / 0.928 | 0.854 / 0.933 | 0.852 / 0.921 | 0.844 / 0.925 | 0.891 / 0.935 | 0.865 / 0.933 | — |
| 10 | 0.864 / 0.937 | | | | | | |

### 6.6 Pareto frontier (descriptive: every cost / threshold evaluated on the outer folds, not selected)

| value iteration c | BA | AUROC | mean Q | | threshold hi | BA | AUROC | mean Q | | DQN c | BA | AUROC | mean Q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.864 | 0.937 | 10.00 | | 0.99 | 0.864 | 0.938 | 8.97 | | 0 | 0.843 | 0.899 | 4.40 |
| 0.0025 | 0.864 | 0.934 | 8.31 | | 0.975 | 0.864 | 0.938 | 8.47 | | 0.0025 | 0.830 | 0.896 | 4.40 |
| 0.005 | 0.864 | 0.926 | 7.15 | | 0.95 | 0.864 | 0.935 | 7.39 | | 0.005 | 0.853 | 0.901 | 4.46 |
| 0.01 | 0.864 | 0.913 | 5.63 | | 0.9 | 0.864 | 0.920 | 6.10 | | 0.01 | 0.814 | 0.871 | 3.19 |
| 0.02 | 0.856 | 0.894 | 4.17 | | 0.85 | 0.860 | 0.913 | 5.15 | | 0.02 | 0.842 | 0.880 | 2.98 |
| 0.03 | 0.860 | 0.897 | 3.61 | | 0.8 | 0.860 | 0.898 | 4.38 | | 0.03 | 0.832 | 0.859 | 2.36 |
| 0.05 | 0.859 | 0.884 | 2.75 | | | | | | | 0.05 | 0.832 | 0.850 | 2.02 |

Brier-terminal value iteration (sensitivity analysis): identical to the accuracy reward up to c = 0.005, then worse on
BA (c = 0.02: 0.831 at 3.20 questions; c = 0.05: 0.823 at 2.00) — the accuracy reward is the better fit for the
stated constraint, as argued in §3.

Readings, as measured:

* On balanced accuracy the value-iteration frontier lies on or above the threshold heuristic's at every length
  (e.g. 0.860 at 3.6 questions vs 0.860 at 4.4); at c = 0.01 it matches the full questionnaire's BA (0.864) with 5.6
  questions on average.
* **AUROC and Brier fall steadily as questions are removed, for every arm.** Thresholded BA plateaus after about three
  questions on this cohort; ranking does not. Shortening the questionnaire costs ranking information even when the
  referral decision is preserved.
* The model-free DQN is not monotone in c and is below value iteration at every comparable length.

### 6.7 Stability and transfer

**Fold-assignment stability (value iteration and threshold stopping re-run with three further outer-fold seeds;
DQN excluded for compute):**

| outer seed | VI BA (Δ vs full) | VI mean Q | threshold BA (Δ vs full) | threshold mean Q | full BA |
|---|---|---|---|---|---|
| primary | 0.851 (−0.013) | 4.20 | 0.860 (−0.004) | 5.37 | 0.864 |
| +1000 | 0.844 (−0.017) | 3.20 | 0.843 (−0.017) | 4.25 | 0.860 |
| +2000 | 0.850 (−0.009) | 3.16 | 0.864 (+0.004) | 4.72 | 0.859 |
| +3000 | 0.832 (−0.032) | 3.70 | 0.864 (0.000) | 5.02 | 0.864 |

The value-iteration BA loss ranges from −0.009 to −0.032 depending only on how participants were split. With the
primary split's interval width (±0.023), the +3000 split would put the lower bound near −0.055, i.e. **the δ = 0.05
non-inferiority conclusion is not robust to the fold assignment.** The threshold heuristic is more stable on BA but
uses about 1–1.5 more questions.

**Exploratory transfer (trained on Saudi circular labels, tested on all 252 Polish children):** value iteration
(c = 0.03 selected on Saudi) BA 0.839 with 3.68 questions vs 0.839 for the full questionnaire under the same
Saudi-trained scorer (ΔBA 0.000 [−0.024, +0.024]); AUROC 0.866 vs 0.933. DQN (c = 0) BA 0.822, 4.62 questions. On Saudi
the optimal policy is curtailment of the sum score, and curtailing at the official cutoff transfers reasonably for the
thresholded decision — but this is a curtailment result, not evidence that RL learned clinical information.

## 7. Adaptive question-selection explanations

`src/rl/explain.py::explain_decision` returns, for every ASK decision:

* the agent's own **Q-values** for every legal action (the chosen action is their argmax under the agent's tie rule —
  asserted by `tests/test_rl_explain.py` at every state the agent visits);
* for value iteration, the **exact Bellman decomposition** of the chosen value: P(atypical answer), belief after each
  answer, value after each answer, question cost — reconstructing Q to machine precision (tested);
* what the question can tell us (belief model): belief after each answer, whether either answer could change the
  referral decision, probability that it does, expected entropy reduction;
* margin over STOP (or "STOP not allowed before the 2-question minimum") and the runner-up question with its margin;
* a template narrative filled from those numbers (no free-form generation).

For the DQN the Q-values are the network's; the answer-branch numbers come from the belief model and the explanation
says `faithful_to_agent_internals: false`.

Example (served policy, opening decision): "A2 was chosen because it had the highest value of all questions the agent
could ask … If answered atypical the estimate would move to 84%; if typical, to 45% … Either answer could change the
screening outcome. Stopping is not allowed before 2 questions (safety minimum). The next-best question, A3, was worth
0.04 points less."

**Response dependence, measured.** Value iteration produced 53 distinct question sequences over 252 children, 9
distinct lengths, 2 distinct opening items across folds, and its second question depended on the opening answer in
the pooled outer-fold runs. The served policy (one model) always opens A2 → A3 and branches from the third question on
(A2, A3 typical → A4; then A5 or STOP …). Its adaptivity is mostly *when to stop* and *what to ask third and later*.

## 8. Stopping mechanism and justification

STOP is an action with its own value, `Q(h, STOP) = E[decision correct | h]`; the agent stops when no question's
expected improvement in decision accuracy exceeds its cost `c`. With the accuracy reward this is learned stochastic
curtailment: questions are asked while they can still change the referral decision with enough probability. Safety
constraints that are not learned: minimum 2 questions, cap 10 (the full questionnaire), STOP forced at the cap and
labelled `budget_exhausted` in the session and the explanation.

STOP explanations (`explain_decision` at a STOP) contain: stop reason and whether it was forced; STOP value vs the
best remaining question (shortfall); that question's probability of changing the decision; final estimate and
decision; the **range of beliefs reachable within the remaining budget** (most extreme answer patterns); the answers
that moved the estimate most (log-odds shifts); remaining-uncertainty fields; limitations (not a diagnosis;
case-control cohort; naive-Bayes assumption; questions left unasked).

## 9. Application integration status

**Integrated, selectable, gated, not the default.**

* Pre-registered gate: value iteration **passed** G1 (Holm p = 0.0036), G2 (mean-questions CI upper 4.50 < 10), G3
  (sensitivity CI lower −0.029 > −0.10); DQN **failed** G1. G4 is the test suite (§10).
* `scripts/step20_build_served_policy.py` applied the constrained cost-selection to all 252 children. Served cost
  **c = 0.03** after the §13 correction (rule estimated over ten inner splits; the original single-split choice,
  c = 0.05, violated the rule on average and made the policy stop after two questions for 43% of children), belief evidence weight w = 0.524 (the naive
  model's raw evidence is roughly twice too confident). Artifact `results/rl_adaptive/served_policy.json`,
  SHA-256 self-hash verified at load.
* Architecture: user response → `engine.rl_apply` (state update; skip = MISSING) → `engine.rl_decide` (agent Q-values
  → ASK/STOP + explanation) → API → UI. Selection logic lives in `src/rl` and `src/screening/session.py`; the frontend
  renders what the API returns.
* App (`app/services/screening.py`): policy `rl_adaptive`; first question always ASK; after every answer the agent
  decides; STOP → `policy_stop` with `stop_explanation`; cap → `budget_exhausted`; user can still stop manually
  (`user_stop`); "prefer not to answer" (`skip`) for RL sessions; invalid / conflicting / stale answers → 400 / 409;
  each trace step stores the explanation of the decision that asked it, and the result stores `policy_info`
  (agent, artifact hash, MDP config), so a session can be reconstructed. If the artifact changes mid-session the next
  answer is a 409, never a silent policy switch.
* **Fallback:** missing, altered or gate-failed artifact → `rl_adaptive` is listed unavailable with the reason
  (`/api/meta.rl_policy`), and a request for it starts a greedy session labelled `policy_info.fallback` (shown as a
  banner in the UI and stored with the result).
* Auto replay is refused for `rl_adaptive` (every Polish record trained the served model, so a replay would not be
  held out).
* Preserved: Shapley attribution, supporting/opposing evidence, counterfactuals, support report, limitations,
  privacy endpoints (smoke-tested end to end on a temporary DB). The outcome explanation now names the
  naive-Bayes partial-state method and the Polish dataset for RL sessions.
* **Default stays `greedy`** (`configs/config.yaml`; one line to change) because the δ = 0.05 margin is not ratified,
  the result is not robust to the fold assignment (§6.7), and the quality requirement itself is under review (§13).

## 10. Test results

New tests: `tests/test_rl_mdp.py` (28: belief, legal actions, rewards, Bellman equation vs brute-force recursion,
determinism, cap, re-ask rejection, skips, DQN seed determinism, selection rule, Holm),
`tests/test_rl_explain.py` (7: explanation = action actually taken, exact decomposition, stop / forced-stop
explanations), `tests/test_app_rl_adaptive.py` (14: API flow, response-dependent sequences, early stop, determinism,
skip, invalid input, auto refused, resume, fallback, failed gate, tampered artifact, mid-session policy change),
3 jsdom tests appended to `tests/test_app_frontend.py`, and `tests/test_rl_selection.py` (3: the ten-split cost
selection of §13.4 — one split reproduces the protocol selection, the average and per-split rule counts are exact,
invalid repeat counts are rejected).

Full suite on the final tree (including the §13 correction): **1005 passed, 12 failed, 80 skipped** (before this
work: 950 passed, 12 failed, 80 skipped). The 12 failing tests are the same test IDs as in the run before this work,
unrelated to it: 8 frozen-hash checks on
`predictor_saudi_v2_isotonic.*` / retained-evaluation files, 3 V-4/V-7 `.sav` fixture tests, and the markdown
phrase rule (pre-existing hits in other reports; this report adds none). No existing test was modified or weakened;
one frontend change (send `skip` only when true) was made so an existing exact-payload test keeps passing unchanged.

## 11. Research limitations

1. **One case-control cohort, n = 252.** Cases are diagnosed children recruited as cases; controls are not a screening
   population. Spectrum bias makes short screens look better than they would be in general-population screening;
   prevalence 54% means the opening belief (54%) and every PPV/NPV are cohort artefacts.
2. **The non-inferiority margin (0.05 BA) is provisional** and was set by me, not by a clinical body or the supervisor.
   The result fails at 0.03 and 0.02, and is not robust to the fold assignment (§6.7).
3. **Ranking is lost.** AUROC falls by 0.038 [0.015, 0.064] with value iteration; any use that ranks children or
   reports calibrated risk is harmed even when the referral decision is preserved.
4. **Adaptivity vs short fixed forms:** at matched burden no significant BA advantage over fixed-length questionnaires;
   most of the benefit comes from variable-length *stopping*. A simple confidence-threshold rule reaches similar BA
   with ~1.2 more questions.
5. **Model-based agent rests on naive Bayes** with one evidence weight (w = 0.52 says the independence assumption is
   badly violated); the plan is optimal under that model and no further.
6. **Model-free DQN is data-limited:** ~130–170 training children per fit; high seed variance; not monotone in c.
7. **Short questionnaires by design of the quality requirement:** BA at τ saturates after ~3 questions on this
   cohort, so the selected policy is short (served c = 0.03; 27% of children stop at 2 in-sample). The originally
   served c = 0.05 violated the selection rule on average and was corrected (§13).
8. **Binary projection of Q-CHAT-25** discards ordinal information; the instrument itself was not changed.
9. **Retrospective replay:** answers were given to all items in fixed order; asking items adaptively, out of order,
   or fewer of them may change how caregivers answer. Not tested.
10. **Polish cohort role change** (sealed external → development) and repeated prior use of the cohort.
11. **Development on Saudi** informed DQN hyper-parameters; Saudi labels are circular.

## 12. Decisions requiring professor input

1. **Non-inferiority margin** for balanced accuracy (0.05 used provisionally; results differ at 0.03), and whether
   sensitivity should carry its own margin.
2. **Role of the Polish cohort**: ratify its use as the development cohort under nested CV, which ends its status as
   an untouched external test set.
3. **Asymmetric error costs:** should a missed referral cost more than a false referral? (Reward is symmetric now.)
4. **Minimum questions** (2) and **maximum** (10), and whether a 2-question stop is acceptable for a screening
   prototype; whether to extend the cost grid or cap c.
5. **Default app policy:** switch `policy.type` to `rl_adaptive` or keep it selectable.
6. **Primary metric:** BA at a threshold (what the gate uses) vs AUROC/Brier (which degrade).
7. **New data:** a prospective or population-based sample with independent diagnoses, ideally with adaptive
   administration, is what a defensible efficacy claim needs; also the 1,054-row NZ file (absent).
8. Whether ordinal Q-CHAT-25 responses (instrument change) should be explored as research.

## 13. Follow-up: why the served policy stopped after two questions (stopping diagnosis)

Reported symptom: the app's `rl_adaptive` policy stopped after two questions. Diagnosed before changing anything
(`scripts/step21_stopping_diagnosis.py` → `results/rl_adaptive/stopping_diagnosis.json`; repeated-split check and
quality-requirement sweep → `quality_requirement_sweep.json`; correction validated by
`scripts/step23_selection_correction_validation.py` → `selection_correction_validation.json`).

### 13.1 Ruled out (with evidence)

| suspect | finding |
|---|---|
| application stopping logic | an RL session ends when the agent's own action is STOP (`policy_stop`), at the cap (`budget_exhausted`) or on the user's stop button; every smoke-test session ended `policy_stop` |
| action selection / masks | STOP is masked before 2 questions, illegal and repeated items are masked; the Q-values of value iteration equal a brute-force Bellman recursion (tests) |
| reward calculation | reward = −c per question, 1[referral decision correct] at STOP; no information-gain term; no extra stop penalty; checked against `terminal_quality` |
| incomplete state | the state is the full answer history (unasked / typical / atypical / skipped per item) — complete for this MDP |
| over-confident belief at the stopping points | at the two-question stops the model is, if anything, *under*-confident: P(ASD) model vs observed — A2 & A3 atypical 0.945 vs 0.979 (n=48), A2 atypical & A3 typical 0.775 vs 0.947 (n=19), A2 typical & A3 atypical 0.724 vs 0.854 (n=41). Out of sample, children stopped at 2 had predicted 0.925 vs observed 0.942 (p ≥ 0.8 bin) |

### 13.2 What the served policy (c = 0.05) did

Every child with an atypical answer on A2 or A3 was stopped at 2 (108/252, in-sample); the rest continued. At those
nodes Q(STOP) beat the best question by 0.023–0.042. Two-question episode (A2, A3 atypical → STOP, refer): question
cost −0.10; terminal +1 if the child has ASD, 0 otherwise; return 0.90 / −0.10; expected 0.845 under the model.

**The 2-question minimum hid an even earlier stop:** without it the same policy stops 89 children (35%) after one
question. So "2" is the floor, not the policy's preference.

### 13.3 Root cause: question cost overpowers the referral-accuracy term, and the selection estimate let it

* **Scale of the trade-off.** On this cohort the whole difference between asking 2 and asking 10 questions is about
  3–4 points of referral accuracy (out-of-sample accuracy 0.825–0.829 for the best 2-question policies vs 0.861 for
  all ten). At c = 0.05 eight extra questions cost 0.40. Scoring all 82 nested-CV arms by their *out-of-sample*
  return (accuracy − c × questions), the best arm at c = 0.05 is "ask two, then stop" (DQN c=0.05: 2.02 questions,
  return 0.729; greedy IG k=2: 0.725). **Stopping at 2 is what the reward asks for at that cost.**
* **The model-free DQN found exactly that shortcut**: at c = 0.05 (all 3 seeds) and c = 0.02 (2 of 3) its greedy
  policy stops *every* child at 2, regardless of the answers; training episodes shrank from a mean of ~4.6 questions
  under heavy exploration to ~2.2 in the last third (median 2; STOP chosen in ~24% of steps). Some of its Q(STOP)
  estimates exceed 1, the largest possible return — function-approximation error. (The DQN is not served.)
* **Value iteration asked more than the reward strictly requires** because its belief over-estimates risk for
  children typical on A2 and A3 (0.345 vs 0.243 observed), so it keeps asking them.
* **Why c = 0.05 was served.** Step 20 chose the cost from one inner 3-fold split, where c = 0.05 lost 0.013 BA
  (inside the pre-registered 0.02 tolerance). Averaged over ten splits it loses 0.021 and meets the rule in 3/10;
  in nested CV over five outer seeds it lost 0.022 [range −0.055, −0.005]. c = 0.05 is also the largest value in the
  grid. **The served cost violated the pre-registered quality requirement on average.**
* **Were the 2-question stops sufficient?** At c = 0.05 (nested CV, 5 seeds), children stopped at 2 got 87.0% correct
  decisions vs 90.7% if the full questionnaire had been asked (full questionnaire corrects 39, breaks 15, of 645);
  at c = 0.01 the 2-question stops (11% of children) agree with the full questionnaire for 100%.

**Conclusion.** No bug. Stopping at two was the optimal policy for the objective it was given: a large cost per
question combined with a quality measure (thresholded referral accuracy) that a case-control cohort saturates after
two or three strongly discriminating items (A2 atypical: 48% of cases, 3% of controls). It is partly genuine
sufficiency — for most two-question stops the full questionnaire gives the same referral — and partly a
reward-maximising shortcut — at c = 0.05 the two-question stops lost about 4% of correct decisions net (39 the full questionnaire would correct vs 15 it would break, of 645), and ranking/calibration (which the
reward ignores) degrade (AUROC −0.070). The shortcut is also tied to the cohort's 54% prevalence: with a lower prior
the same reward stops *earlier* (prior 0.10: 81% of children at 2; prior 0.02: everyone stops at 2 with no referral),
because under an accuracy reward a question has no value unless its answer can flip the decision at τ = 0.5.

### 13.4 Correction applied (no new research decision)

Keep the objective, reward, cost grid, minimum, cap, belief model and the pre-registered rule ("fewest questions with
BA within 0.02 of the full questionnaire"); estimate that rule's BA on **ten** inner fold assignments instead of one
(`src/rl/selection.py`; split 0 is the protocol's split; `step20 --inner-repeats 10`). Selected cost: **0.03**
(meets the rule in 9/10 splits). Previous artifact kept as `served_policy.before_selection_fix.json`.

| | served cost | stop at 2 (in-sample) | mean Q (in-sample) | nested CV of the procedure (5 outer seeds): mean Q | stop at 2 | ΔBA vs full | ΔAUROC |
|---|---|---|---|---|---|---|---|
| before (single split) | 0.05 | 43% | 2.70 | 3.63 [3.16, 4.20] | 27% | −0.013 [−0.032, +0.004] | −0.053 |
| after (10 splits) | 0.03 | 27% | 3.21 | 3.30 [3.11, 3.54] | 28% | −0.015 [−0.028, −0.003] | −0.058 |

(Ranges are over the five outer seeds; full questionnaire BA 0.856–0.864.) The served policy now stops at 2 when A2 is
atypical; a child typical on A2 and atypical on A3 continues (A4, A1, A9 …), and a child answering everything typical
stops at 3. **At the procedure level the correction changes nothing material**: the cost choice becomes stable
(mostly 0.03 instead of 0.005–0.05), quality and length are the same. It fixes a served artifact that violated its own
rule; it does not make the questionnaire longer, and it should not be read as solving the trade-off.

### 13.5 What a more balanced trade-off requires — a research decision, not implemented

With BA at τ as the sole quality requirement, this cohort supports about three questions. Exploratory nested CV of
the same agent under alternative *pre-specified* requirements (margins illustrative; 5 outer seeds):

| quality requirement in the selection rule | mean Q | stop at 2 | ΔBA vs full | ΔAUROC | Brier (full 0.102) | same referral as full |
|---|---|---|---|---|---|---|
| R0 BA within 0.02 (pre-registered) | 3.63 | 27% | −0.013 | −0.053 | 0.124 | 95.6% |
| R1 R0 + sensitivity within 0.02 | 3.93 | 25% | −0.010 | −0.047 | 0.121 | 96.7% |
| R2 same referral as full for ≥ 95% | 3.65 | 23% | −0.002 | −0.043 | 0.118 | 96.7% |
| R3 R0 + AUROC within 0.02 | 7.23 | 1% | −0.001 | −0.012 | 0.106 | 99.9% |
| R4 R0 + Brier within 0.01 | 6.58 | 7% | −0.003 | −0.019 | 0.110 | 99.5% |

Fixed-cost reference points (nested CV, 5 seeds): c = 0.01 → 5.56 questions, ΔBA 0.000, ΔAUROC −0.025, 11% stop at 2
(all agreeing with the full questionnaire); c = 0.02 → 4.21, −0.008, −0.042.

If the intended use needs a calibrated risk or a ranking (beyond a refer / don't-refer decision), the requirement
must say so (R3/R4-type), and the agent then asks ~6.5–7 questions. If it needs only the referral decision, ~3–4
questions is what this data supports. Choosing between them, and the margins, is for the supervisor (§12.6).
Also for the supervisor: the 2-question minimum has no research justification (it binds for 35% of children at
c = 0.05); and an asymmetric error cost (§12.3) would change where the agent stops.

---

## Direct answers

**Does NeuroBridge now have a genuinely RL-driven adaptive questionnaire?** Yes, as a selectable app policy. The
agent is model-based RL (value iteration on a belief MDP learned from data); it chooses every question and the stopping
point from the Bellman values. The model-free DQN was built and evaluated but did not pass the gate. Greedy remains the
default pending your supervisor's decisions.

**Does the agent select questions based on previous answers?** Yes: the next question and whether to stop depend on
the answers (53 distinct sequences, 9 lengths in nested CV; tests assert different sequences for different answer
patterns). In the served model the opening pair is fixed (A2, A3) and branching starts at the third question.

**Can it make a meaningful STOP decision?** Yes. STOP is a valued action, chosen when no question repays its cost; 94%
of children stopped by the policy before the cap in nested CV; every STOP is explained with the shortfall of the best
remaining question and the reachable belief range. Caveat (§13): where it stops is set by the cost per question
and the quality requirement; with BA as the requirement it stops after 2–3 questions for most children, and the
originally served cost (0.05) over-shortened it until corrected.

**Does it balance quality and burden?** Yes by construction (constrained objective via its Lagrangian, Pareto sweep
reported): 4.2 vs 10 questions for −0.013 BA. The price is ranking (AUROC −0.038).

**Is there evidence it performs comparably to fixed-question screening with fewer questions?** Within the stated
limits: value iteration was non-inferior on balanced accuracy at δ = 0.05 (Holm p = 0.004) with 5.8 fewer questions
(primary pre-registered analysis). It is **not** non-inferior at δ = 0.03, the conclusion is **not robust** to the
fold assignment, AUROC is significantly lower, and at matched average length it did not beat fixed short forms. No
evidence of "better than" the full questionnaire.

**Are the results scientifically defensible?** As a retrospective, pre-registered, nested-cross-validated simulation
against independent clinician labels — yes, with the limitations above stated. As evidence of clinical screening
performance — no: one small case-control cohort, a provisional margin, and fold-sensitivity.

**What to discuss with your professor next:** the margin and primary metric (§12.1, §12.6); ratifying the Polish
cohort's role (§12.2); error-cost asymmetry and minimum length (§12.3–4); whether the ranking loss matters for the
intended use; and what data could support a confirmatory study (§12.7). A useful framing for the discussion: on this
cohort the benefit is learned *stopping*, not learned *selection*.

## Reproduction

```
python scripts/step18_rl_adaptive_protocol.py      # freezes the protocol (refuses to overwrite)
python scripts/step19_rl_adaptive_experiment.py    # ~20 min on 14 workers; refuses if src/rl changed
python scripts/step20_build_served_policy.py --inner-repeats 10   # served artifact iff the gate passed (§13.4)
python scripts/step21_stopping_diagnosis.py        # stopping diagnosis (§13)
python scripts/step22_quality_requirement_sweep.py # exploratory quality requirements (§13.5)
python scripts/step23_selection_correction_validation.py   # nested CV before/after the selection correction
python -m pytest tests/test_rl_mdp.py tests/test_rl_explain.py tests/test_app_rl_adaptive.py tests/test_app_frontend.py tests/test_rl_selection.py
```

Existing artifacts (`results/predictor_*`, `results/policies_v3*`, earlier protocol/evaluation JSONs) were read, never
rewritten; new outputs are under `results/rl_adaptive/` (git-ignored, no participant identifiers).
