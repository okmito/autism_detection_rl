# Current Project State

**Last updated:** 2026-10-09 (P4: the optional support layer after the result)
**Operator request:** "continue" — finish the P4 support layer begun in the
previous session (`src/support/` data modules + two test files existed, the
assembly module and every engine/report test did not).
**Branch state:** see git log (`feat/outcome-explainability`). **527 passed, 12 failed (all environmental), 78 skipped.**

## 2026-10-09 (tenth pass) — P4: the support layer is complete, wired and tested

The previous session had landed the support layer's *data* — `schemas.py`,
`domains.py`, `questions.py`, `strategies.py`, `engine.py` — plus two contract
test files. What was missing was the assembly module those files already
referenced (`src/support/report.py`), every rule test for the engine, every test
for the report, and any wiring into the demos. All four are now done.

**Built (all post-hoc; the screening path is untouched):**

| module | what it adds |
|---|---|
| `src/support/report.py` | `build_support_report()` — the single assembly point; adapters from `run_episode` output and from the `outcome-explanation/1.0` dict into the Pydantic contracts; evidence extraction with deterministic ids (`ev-obs-A3`); documented generation-status rule; report-level limitations (pending review, hypothesis-not-need, optionality) |
| `src/support/engine.py` | `questions_to_offer()` — the suggested/optional split, the only permitted direction from screening data to a support question; preference abstentions no longer read as preferences; documented why `HYPOTHESIS_OPTED_IN` is never emitted |
| `scripts/demo_app.py` | finished sessions retain their episode result + explanation; `POST /api/session/support`; the result payload carries the follow-up questionnaire |
| `scripts/demo_static/index.html` | an "Optional support ideas" screen rendering backend fields only (status wording is presentation metadata for backend enum values; no suggestion text is composed client-side) |
| `scripts/demo_live.py` | STEP 6c prints the questionnaire split, the assessment table, the offered suggestions with their reasons, the limitations and the disclaimer |
| `docs/SUPPORT_LAYER_DESIGN.md` | design record: the safety rule, the evidence-type table, the honest-status table, method choices and rejections, open items |

**The core property, enforced structurally and tested:** *a hypothesis never
fires a recommendation.* An atypical observed response can put a follow-up
question on the **suggested** list; only the person's own answer to it can
produce a suggestion, which then carries its triggering assessment id and a
plain-language `why_selected`. Domains with no Q-CHAT-10 linkage (sensory,
transitions, daily living, accessibility) disclose that in their wording, can
never be suggested from data, and can only be user-reported. Every strategy
ships `pending_expert_review` and the report says so.

**Two design corrections found while testing:**
* the accessibility domain's yes/no question could fire the format
  recommendation without a stated format — it is now a choice question, and a
  choice alongside an abstention is recorded as `unknown`, not as a preference;
* an unanswered question behaves exactly like a skipped questionnaire (locked by
  a test): a missing question id is never read as "no".

**Tests:** +90 in the support layer (schemas 28, registry 6, questions 8,
strategies 7, engine 26, report 20) and +13 demo-wiring tests
(`tests/test_support_demo.py`), all on synthetic fixtures. Suite:
**437 → 527 passed**, the same 12 environmental failures, 78 skipped.

**Verification performed:** the browser demo was driven end-to-end over HTTP
(suggested/optional split, confirmed need, stated preference, explicit "no",
abstention, skip, and a 400 on a choice question answered without a choice);
the terminal demo ran the full pipeline on the real Saudi cohort and printed
STEP 6c; `to_json_safe` round-trips through `json.dumps`; two builds of the same
inputs are identical apart from `created_at`.

**Deliberately not produced:** a batch `results/*.json` of assembled reports —
that would be a store of people's own answers, and the consent/storage story
does not exist yet (recorded in the design doc's open items).

## 2026-10-09 (ninth pass) — P3-outcome: explaining the final screening result

Built the explainability layer for the **outcome** (not the policy): why the
system produced a particular screening result. The policy-side question (which
question was asked) already had the selection diagnostic, belief trace, and
`beta_greedy.explain()`.

**Verified outcome path (the plan's central finding, confirmed in source):**
`run_episode` (`src/env/environment.py:10`) uses the RL policy only for
*acquisition*; on STOP the outcome is `p_hat = predictor(final_state)`
(`environment.py:56`), decision `REFER` iff `p_hat >= tau` (`environment.py:62`).
The reward is a training signal, not an outcome and not a confidence. Two
frozen predictors: v2 `MaskedPredictor` (MLP, legacy) and v3 `LogisticPredictor`
(L2 logistic + Platt + exact marginalisation over 2^10 configs, development
scorer). Also verified: **no recurrent policies, no Stable-Baselines3, no
curriculum learning exist in this repo** — DQN and PPO are custom MLP
implementations (`src/policies/{dqn,ppo}.py`).

**Built (all post-hoc; no retraining; environment/reward/τ/seeds/splits untouched):**

| module | what it adds |
|---|---|
| `src/explain/attribution.py` | **exact group-Shapley** over questionnaire items under the predictor's own partial-state value function (absent = UNASKED = marginalised, not zeroed). ≤ 2^k evaluations (k ≤ 6 observed items); deterministic; efficiency identity `Σφ = p_hat − v(∅)` asserted |
| `src/explain/counterfactual.py` | `find_counterfactual` **preserved unchanged**; new `find_counterfactual_rich`: verified single flips, prior-plausibility feasibility, minimal multi-flip sets, unask, ask-more |
| `src/explain/uncertainty.py` | v3 **prior-sensitivity band** (Dirichlet-multinomial resampling of the item prior at the training sample size). Reported *unavailable* for v2 — not fabricated |
| `src/explain/limitations.py` | centralised limitation statements + mandatory non-diagnostic disclaimer |
| `src/explain/outcome.py` | `explain_outcome()` → schema `outcome-explanation/1.0`; compares decisions semantically across the repo's two vocabularies (`REFER` in the environment, `REFERRAL_RECOMMENDED` in the demo) |
| `src/explain/render.py` | deterministic text rendering from evidence fields only |
| `src/models/logistic_predictor.py` | pure `probability_with_prior()` (prediction behaviour unchanged) |
| `scripts/step17_outcome_explainability.py` | `results/outcome_explainability_saudi.json` + faithfulness metrics |

**Measured (Saudi test, B=6, greedy, v3, 127 episodes — `results/outcome_explainability_saudi.json`):** additivity max error **1.11e-16**; counterfactual found 0.346 / robust 0.654; deletion AUC guided 0.1413 vs random 0.1274; insertion AUC 1.2799 vs 1.1832; **rank stability under one extra observation is low (Spearman mean 0.164)** — reported as measured, not smoothed; overhead 20.1 ms/episode. None of these is a clinical claim.

**Honesty debts closed** (from the open-items table): item 8 — the trace
sequential-consistency assertion is restored (`tests/test_trace_belief_update.py`
now independently replays states and asserts `belief_after[i] ==
belief_before[i+1]` and final consistency with `p_hat`); item 9 —
`src/explain/shap_baseline.py` now documents that it is **not SHAP**
(single-flip delta, non-additive), retained only for artifact compatibility,
with `tests/test_shap_baseline_honesty.py` locking the disclosure.

**Baseline restoration on this machine (verified, all deterministic):** ran
step2, step3-adjacent artifacts via step12, step9, step4, step5, step13, step15,
step16. Suite moved from **18 failed / 335 passed / 134 skipped** to **12 failed
/ 437 passed / 78 skipped**. The 12 remaining failures are all environmental or
pre-existing — **not regressions**:
* 6 × "frozen v2 artifact hash" tests — regenerated v2 bytes differ from the
  pinned hashes recorded on the authors' machine. Verified deterministic here
  (`b47b52c7…`/`8c8c5ed6…` across repeated runs; documented value
  `0.009122572011964558` vs this machine's `0.008569736301893147`). The pins
  belong to the environment that produced them; do not "fix" the test.
* 1 × `test_first_evaluation_results_are_retained_not_deleted` — the retained
  step-14 artifact never existed on this machine, and step14 now refuses to
  regenerate it by design: the retained run predates `random_fixed`, so it
  fails the current mandatory-baseline gate (`assert_all_baselines`). Obtain the
  artifact from the team or accept the failure.
* 1 × `test_v2_no_claim_rule` — **pre-existing doc violation on main**:
  non-novelty uses of "first"/"only" in `V6_LAMBDA_DECISION.md` and
  `POLICY_BENCHMARK_REPORT.md` (e.g. the "first 50 → final 50" table header).
  Left untouched deliberately; fixing it is a docs decision, not a code one.
* 3 × `test_v4_v7_validation` — need the gitignored
  `data/QCHAT_dataset2 mendeley.sav`, absent locally.

**Tests:** +41 new (attribution 8, counterfactual 10, explanation object 13,
shap-honesty 3, step17 7). All new tests pass; the 57 demo/frontend tests pass
unchanged. The demo now shows a "Why this result?" panel driven only by backend
fields (frontend rule preserved), and the terminal demo prints the rendered
explanation at STEP 6b.

**Manual verification performed:** browser demo driven end-to-end over HTTP
(interactive + auto): explanation present, JSON round-trips, v2 correctly
reports "no per-session interval available"; terminal demo smoke-tested.

**Known gaps carried forward:** the Step-17 artifact reports the v3 scorer
only; v2 group attribution is implemented and unit-tested but not batch-run;
an LLM verbaliser remains deferred (would need field-grounding tests).

## 2026-10-01 (sixth pass) — P1-f: a Beta-prior / EVOI arm; P1-g: saturation promoted to the front page

**`src/policies/beta_greedy.py`, a NEW arm. `GreedyIGPolicy` deliberately untouched.** It exists because greedy's criterion is *vacuously zero* on this data: on a pure support `H(Y|s) = 0` and every child support is pure too, so every legal item's IG is exactly 0 and `argmax` falls through to the lowest legal index.

```
decision states sampled (held-out, B=6)      : 480
  with a pure support                        : 195
  greedy criterion identically zero on those : 195/195
  beta_greedy EVOI identically zero on those: 8/195
  median EVOI spread across legal items there: 0.000984
```

Two changes: a **Beta(1,1) posterior** (interior on a pure support, and it correctly treats a 3-record agreement as weak evidence — 0.918 bits at n=1 vs 0.080 at n=100), and **EVOI stopping** on the actual terminal utility (`gain_j = u(s,j) − u(s)`, stop when `max_j gain_j < λ·c_j`). At λ=0 it never stops early, which makes AB-7 separable. `explain()` returns the posterior, 95% credible interval, per-item EVOI, entropy IG, runner-up margin and stop margin — the P3-1 hook designed in rather than retrofitted.

**Results (B=6):** `beta_greedy` tracks the exact reference's question count across the whole λ sweep, beats it on held-out reward at 4 of 6 λ values, and beats `greedy` at every λ ≥ 0.005 (greedy cannot stop, so it pays for all six). At λ=0 the two are **bit-identical on the train objective** (+0.004225) yet `beta_greedy` generalises better (0.0482 vs 0.0512 Brier) — a direct measurement of the saturation problem: the empirical objective cannot tell them apart, the held-out data can.

**⚠ RETRACTED — see Step 8.** This entry originally claimed *"one reward, two scales"*: that the support-posterior EVOI was ~100× smaller than the neural-predictor marginal utility, so a λ grid could not transfer. **That was wrong.** It rested on a 0.05–0.15 predictor figure that **no code ever measured** — prose only. `scripts/step8_evoi_scale_analysis.py` measured both sides with the same functional form and found them in the **same units** (ratio 0.43 at the median, 0.84 at the mean, 1.02 at p75). What survives is a *different* problem: the support-posterior EVOI is **analytically non-positive on a pure support**, so the statistic behaves as a purity indicator, and the λ grid spans only ~20 distinct behaviours with a cliff between 0.001 and 0.003. The open decision is `V6_STOPPING_THRESHOLD_DECISION.md`; **V-6 is not signed off and no threshold is selected.**

**H1 now fails for three independent adaptive arms** (greedy, beta_greedy, DQN) at B=5 and B=6 — the reversal is not an artefact of the greedy heuristic. Adding a principled non-degenerate criterion did not rescue it.

**P1-g:** the saturation finding and the H1 reversal are now on the front page of `diagnosisReady.md` (§0, §0A) and in the `POLICY_BENCHMARK_REPORT.md` banner, together with the corrected P0 register. V-6 and Polish are named as the two items that move to the front of the queue.

## 2026-10-01 (fifth pass) — P1-e: the DQN collapse was an undertrained network

`scripts/step7_rl_diagnosis.py`, 5 seeds × {200, 400, 1000, 2000} episodes, 60 held-out records, trained predictor. Two arms, differing **only** in the learning target:

| objective | episodes | items (mean ± sd) | UAR (mean ± sd) | distinct states |
|---|---|---|---|---|
| bootstrap (spec §11.2) | 200 | 0.00 ± 0.00 | 0.5000 ± 0.0000 | 167 |
| bootstrap | 400 | 0.60 ± 0.55 | 0.6615 ± 0.1485 | 352 |
| bootstrap | 1,000 | 4.30 ± 1.56 | 0.8881 ± 0.0180 | 1,347 |
| **bootstrap** | **2,000** | **5.94 ± 0.11** | **0.8870 ± 0.0184** | 3,462 |
| **returns** | **200** | **5.13 ± 0.66** | **0.8851 ± 0.0362** | 639 |
| returns | 400 | 4.97 ± 1.12 | 0.8775 ± 0.0441 | 1,123 |
| returns | 1,000 | 5.01 ± 0.42 | 0.8984 ± 0.0131 | 2,283 |
| returns | 2,000 | 5.59 ± 0.28 | 0.8840 ± 0.0162 | 3,770 |

Reference: greedy 6.00 items / UAR 0.9021 / Brier 0.0571.

**The 400-episode DQN figure was an undertrained network, not a property of RL.** At 400 episodes the bootstrapped objective is still at 0.60 items; it needs ~2,000 to reach 5.94. The Step-4/Step-5 default has been raised from 400 to 2,000, and `tests/test_step7_rl_diagnosis.py::test_benchmark_uses_the_diagnosed_episode_budget` fails if Step 5 is ever run at a smaller budget again.

**Why the bootstrap target is slow.** §11.1 emits a reward only at the end of an episode, so a `B`-step episode with immediate rewards needs a `B`-step bootstrap chain; the root action's value is then a product of `B` noisy estimates. Storing discounted return-to-go and disabling the bootstrap term removes the chain and reaches plateaued behaviour by **200 episodes** — an order of magnitude sooner.

**But the diagnosis does not rescue RL.** Neither arm beats greedy (UAR 0.887 / 0.884 vs 0.9021) or the exact best fixed subset on Brier. It makes the RL numbers *credible*, not favourable.

**PPO degrades with more training** (1.43 items, Brier 0.1162 at 2,000 vs 0.0953 at 400). With λ=0 nothing penalises stopping early, so extra training drives STOP's advantage upward. That is the strongest argument yet for signing off V-6.

**A pre-committed criterion that asked the wrong question.** The stop rule was "state-coverage growth < 10% and items-mean flat", and it returned `still_improving = True` for both arms. Coverage genuinely keeps rising (+156%, +134% in the second half) — but coverage is not a convergence test for a *policy*. Behaviour plateaued (UAR delta −0.0011 and −0.0144 across the last two budgets). Both metrics are now recorded, with the mis-specification documented rather than quietly replaced.

## 2026-10-01 (fourth pass) — P0-10: the predictor was unseeded; P1-d: H1 is finally testable

**P0-10 — every artifact was non-reproducible regardless of its recorded seed.** `MaskedMLP` builds `nn.Linear` layers, which draw from torch's *global* RNG, and `fit()` seeded torch *after* construction. `step3` and `step5` never seeded it in `main()` at all. Found by accident when greedy's Brier moved between two runs of the same command — greedy depends on no training, so something upstream was nondeterministic. Fixed inside `MaskedPredictor.__init__(seed=...)`, which seeds only around construction and then restores the caller's RNG state. `step2`/`step5` CSVs and `predictor_saudi_metrics.json` now hash identically across runs. Detail in `AUDIT_REPORT.md`.

**P1-d — the §17 baselines are wired.** `ExactFixedSubsetPolicy`, `StaticRFEPolicy` and `IRTCATPolicy` existed but were instantiated nowhere, so **H1 had no runnable comparator**. All three are now arms in Step 5, and each had a defect fixed first — most importantly `IRTCATPolicy`'s difficulty estimate gave `b <= 0` for 10/10 items, flattening Fisher information so CAT selection degenerated to a discrimination proxy. Detail in `AUDIT_REPORT.md`.

### H1 result — not supported at the upper budgets

| B | greedy Brier | exact best fixed subset | H1 |
|---|---|---|---|
| 3 | **0.0876** | 0.1034 | supported |
| 4 | **0.0691** | 0.0850 | supported |
| 5 | 0.0655 | **0.0448** | **not supported** |
| 6 | 0.0512 | **0.0479** | **not supported** |

The adaptive advantage shrinks and reverses as the budget grows, **even though greedy stays strictly closer to `V*` on the empirical objective at every budget** (+0.0042 vs +0.0141 at B=6). The advantage lives inside the training support; the per-episode information-gain ordering is what fails to transfer. Spec §4 pre-registers that failing to outperform the exact best fixed subset is a valid result, so this is reportable — but it is a real negative for the project's adaptive-value hypothesis and should be written up as such, not softened.

`exact_fixed_subset` and `static_rfe` select **identical** subsets at every budget on this cohort — a useful consistency signal between a cheap and an exhaustive selector, now asserted in the tests.

## 2026-10-01 (third pass) — P0-9: the split was contaminated, and the predictor was mis-trained

Found by a test written for an unrelated phase. **This invalidated every number produced before it.**

**P0-9 — train/val/test contamination.** Four copies of the split passed *relative* indices from a nested `skf.split(train_idx, ...)` straight into `records[i]`. On the 506-row Saudi cohort **71 of 95 validation records were also in train and 24 were in test**. The row counts came out at exactly the documented 284/95/127, which is why it survived review — the sizes were right, the membership was wrong. `src/data/splits.py` now owns the canonical scheme and `split_fingerprint` detects this class of error; all five consumers (step2/3/4/5, demo) agree on `b2021998a83e7224`. Separately, `step4` used a divergent 60/20/20 split (303/101/102), violating spec §17.

**P1-b — `MaskedPredictor` v1 → v2.** One masked training view per record (284 states for a 41-dim input); the `budget_norm` feature trained on the `budget=10` grid while inference runs at `budget=B` (four of seven inference values never seen); calibrator fitted on fully observed states only, so it extrapolated on every partial state. **Test Brier 0.0630 → 0.0512, UAR 0.8513 → 0.8930, AUROC 0.9627 → 0.9882, ECE 0.0560 → 0.0475.** v1 is exactly reproducible via `masks_per_record=1, eval_budgets=[n_items]`.

## 2026-10-01 — P1-a: V-6 evidence pack ✅ (does not sign off V-6)

`scripts/step6_lambda_sweep.py` → `results/lambda_sweep_saudi.{json,csv}` (64 cells, 8 λ × 8 B). **Predictor-independent**: `ExactDP` reads the training support directly, and the held-out comparison uses the empirical support posterior rather than the neural predictor — so the pack stays valid across the P1-b retrain. Analysis and proposed grid in `V6_LAMBDA_DECISION.md`.

**Proposed grid: λ ∈ {0, 0.005, 0.01, 0.02, 0.05}**, reported as a cost–utility frontier, never relabelled as a budget curve. Excluded: λ=0.1 (cost dominates) and λ=0.2 (degenerate — the optimum asks **zero** questions).

**The finding that reframes everything:** at λ=0 the empirical support becomes *pure* after ~4 questions (0.000 of states at depth 1 → 0.820 at depth 4 → 1.000 at depth 7), so `u_stop = 1 - (p_emp - y)^2` is exactly 1.0, `V*` is exactly `1.000000`, and no further question can raise the reward. The exact policy's `pure_support_frac_at_stop` is **1.00** — it stops early precisely when the objective has saturated, not when evidence is sufficient. The measured value of adaptive stopping at λ=0 is a property of the label rule. **This makes the Polish transfer test (RQ3/H2) the load-bearing experiment, not a secondary one.**

## 2026-10-01 (second pass) — the rollout collector delivered no reward ⚠️

Part 1 fixed `train_step` but missed `collect_episode`, which stored `r = 0.0` on every row, attached the legal set of the state each transition *left* rather than the one it *entered*, and staged terminal rows in a list nothing drained. DQN fitted `y = γ·max Q(s',a)` against an all-zero target; PPO's advantage was **identically zero**. The first Step-5 conclusion ("RL loses to greedy") was therefore a measurement defect. Full detail in `RL_TRAINING_REPORT.md §2`.

| # | Defect | Fix |
|---|---|---|
| P0-1 | Terminal reward discarded | §10 utility attached to the terminal transition |
| P0-2 | `legal_next` was the legal set of the state *left* | now the legal set of `s_next` |
| P0-3 | `end_episode` never called → STOP rows discarded, `_pending` leaked | transitions written with `add`, `_pending` stays 0 |
| P0-4 | `stop_reason` could never be `"budget_exhausted"` → `stopped_early_frac` ≡ 1.0 | termination cause recorded properly |
| P0-7 | epsilon-greedy used the unseeded module `random` | per-policy seeded `random.Random(seed)` |
| — | `PPOPolicy.legal_mask(None)` raised `TypeError` on terminal rows (latent, hidden by P0-3) | `None` ⇒ unconstrained |
| — | importance ratio evaluated `nan` (`-inf − -inf`) | finite `LOG_FLOOR` |
| — | step2/3/4/5 crashed on their final `print` (cp1252 cannot encode `→`/`λ`), exiting non-zero *after* writing artifacts — this is why the step4/step5 artifacts were missing and 12 tests skipped | stdout/stderr reconfigured to UTF-8 |

## 2026-10-01 — benchmark status after all five corrections

At B=6, Saudi test split, `predictor_version: 2`, split fingerprint `b2021998a83e7224`, seed 0, λ=0, **2,000 training episodes**, reproducible from seed. **All eight §17 arms present, plus eta_greedy (P1-f).**

| policy | Brier | UAR | items | stop_early | train gap V\*−V_emp |
|---|---|---|---|---|---|
| exact_fixed_subset | **0.0479** | 0.9167 | 6.00 | 0.00 | +0.014085 |
| static_rfe | **0.0479** | 0.9167 | 6.00 | 0.00 | +0.014085 |
| greedy | 0.0512 | 0.8930 | 6.00 | 0.00 | **+0.004225** |
| dqn | 0.0640 | **0.9112** | 5.98 | 0.02 | +0.034918 |
| exact | 0.0644 | 0.8812 | 4.45 | 0.77 | 0 by definition |
| irt_cat | 0.0686 | 0.8993 | 6.00 | 0.00 | +0.026892 |
| random | 0.0918 | 0.8220 | 4.02 | 0.55 | +0.091082 |
| ppo | 0.1162 | 0.8398 | 1.43 | 1.00 | +0.137615 |

- **Greedy-IG's near-optimality survived all five corrections** — gap ≤ 0.0133 at every budget, and it is the single closest policy to `V*` at every budget.
- **But the exact best fixed subset wins held-out Brier at B=5 and B=6.** See the H1 table above. This is the project's most important current result and it runs *against* the adaptive-value hypothesis.
- **DQN now trains to a sane policy** (5.98 items, UAR 0.9112, train gap +0.035) but still does not beat greedy on the empirical objective or the fixed subset on Brier.
- `static_rfe` is bit-identical to `exact_fixed_subset` at every budget.
- The Step-5 table is still single-seed — the 5-seed variance is in the Step-7 diagnosis, not here. No gap difference above is interpretable as an effect.

## 2026-10-01 — Part 2: policy benchmark ⚠️ SUPERSEDED — see the sections above

`scripts/step5_policy_benchmark.py` evaluates DQN, PPO, Greedy-IG, Random, and ExactDP at matched budgets B ∈ {1..6} on the same held-out test split,same predictor, same `run_episode` evaluator. Artifact: `results/step5_policy_benchmark_saudi.{json,csv}`. Tests: `tests/test_step5_benchmark.py` (12).

**Optimality gap V\* − V_emp (train split, λ=0, lower is better):**

| B | greedy | dqn | ppo | random |
|---|---|---|---|---|
| 1 | −0.000000 | +0.044531 | +0.040883 | +0.055131 |
| 3 | +0.001595 | +0.032676 | +0.035361 | +0.060355 |
| 5 | +0.012969 | +0.048338 | +0.105890 | +0.090981 |
| 6 | +0.006749 | +0.038949 | **+0.138141** | +0.090097 |

**This is a negative result for RL on this task, and it is the honest one.** Training worked (DQN loss 0.0068→0.0002, PPO value loss 0.066→6e-5) but the policies converge to something worse than a one-step lookahead heuristic. PPO degrades monotonically with budget and is worse than random at B=5/B=6.

Behavioural notes: DQN **never stops early** (asks exactly B every time — no adaptive stopping under λ=0); PPO stops very early (1.84 items at B=6), which likely explains its worsening gap. At B=3 greedy and exact produce **identical** test metrics, reproducing the §17 #7 near-optimality claim.

Caveats: single seed (no variance), λ=0 only (V-6 pending), and **all labels are circular** — these numbers measure fit to the questionnaire's own scoring rule, not autism. Full analysis in `POLICY_BENCHMARK_REPORT.md`.

> ⛔ **The Part 2 section above is retained verbatim and is VOID.** The DQN loss
> "0.0068→0.0002" and PPO "entropy flat at ~2.11" quoted here are signatures of a
> training loop that received no reward at all. Read the audit section at the top
> of this file, `RL_TRAINING_REPORT.md §2`, and `POLICY_BENCHMARK_REPORT.md §A`.

## 2026-10-01 — Part 0 (environment) + Part 1 (RL code)

### Part 0 — environment restored ✅
The repo had **no `.venv`, no `data/raw/`, no `results/`** (all gitignored), so no documented number was reproducible.
- Rebuilt venv on Python 3.14.7. **Note:** `requirements.txt` resolves `torch` to the CUDA build, which exhausts `/tmp` (3.7G tmpfs). Installed CPU-only torch via `--extra-index-url https://download.pytorch.org/whl/cpu` with `TMPDIR` pointed off the tmpfs. Same disk failure class as 2026-09-30.
- Restored all three datasets — see "Data provenance" below; **the UCI ARFF and the Polish CSV are derived conversions, not originals**.
- Re-ran Step 2 + Step 3: all 8 artifacts regenerated with `source: real`. All 48 DP runs `status=optimal`. State counts match `STATE_COUNT_VERIFICATION.md` exactly.
- Test suite: 40 passed / 14 skipped → after regeneration **54 passed, 0 skipped**.

### Part 1 — RL code fixed ⚠️ incomplete (collector not fixed)
**Two** real defects in code that had never executed (full detail in `RL_TRAINING_REPORT.md`):
1. `DQNPolicy.train_step` built a legal-action mask and discarded it → bootstrap optimised toward illegal actions.
2. `PPOPolicy` had **no training method at all** — the critic never received a gradient.

A third alleged defect (terminal next-state tensor shapes breaking `torch.cat`) was investigated on 2026-10-01 and **retracted** — `torch.cat(dim=0)` concatenates along dim 0, so the original code did not raise. Do not repeat that claim.

Fixed both; added `src/policies/replay.py` (`ReplayBuffer`, stores per-transition legal sets), `scripts/step4_train_policies.py`, and `tests/test_rl_training.py` (13 tests). Aligned PPO's batch format with DQN's.

First real runs (Saudi 506, B=6, seed 0): DQN loss 0.006889 → 0.000235; PPO value loss 0.066314 → 5.9e-05; PPO entropy flat at ~2.11 (no collapse). Artifact: `results/step4_policy_training_saudi.json`.

### Part 2 — policy benchmark ✅ DONE 2026-10-01
DQN, PPO, Greedy-IG, Random and ExactDP are now evaluated at matched budgets B ∈ {1..6} on the same held-out test split with the same evaluator. See the Part 2 section at the top of this file and `POLICY_BENCHMARK_REPORT.md`.

**Finding: greedy-IG attains the exact optimum (gap ≤ 0.013); both learned policies do not, and PPO is worse than random at B=5/6.** Next: multi-seed variance, then resolve V-6 (λ) before any λ-dependent claim.

## Completed

### Infrastructure (2026-08-30)
- Repository scaffold per §8 — all `src/*`, `configs`, `tests`, `data/`, `results/` created
- Data layer: `schema.py`, `ingest.py` (6 loaders + synthetic mode + Q-CHAT-10 binary map §15), `dedupe.py`
- Three-state encoding §9 distinct, `costs.py`, episode loop §10 with STOP/budget/missing; input dim `4n+1` binary
- Predictor: `MaskedMLP` 128→64 sigmoid §12 + random-mask training + fold-local calibration (isotonic/Platt)
- Exact DP authoritative §14.1 with tractability limits 50M/24h/32GB
- Adapter stub §14.2 compatibility check (DL8.5/MurTree not Brier-compatible)
- Policies: DQN, PPO, Greedy IG, Random, IRT-CAT, DQN-CAT, RFE, Exact Fixed Subset (§17)
- Explain: trace §18.1, counterfactual §18.2, SHAP baseline §18.3
- Eval: metrics §19.1, paired bootstrap 2000 §19.2, power/MDE §19.3, Holm-Bonferroni §19.3, subgroup §19.4
- Config: `configs/config.yaml` §13 — `predictor.hidden [128,64]`, `freeze true`, `τ=0.5`, `primary_metric brier`, `policy_state questions_only` all PASS
- 26 §21 tests passing

### ⚠️ Superseded note (2026-10-01)
The "Policies: DQN, PPO, …" line above records the **scaffold** only. Through 2026-09-30 those policy classes existed but **had never been trained or evaluated** — no training call existed anywhere in the repo, and every run used Greedy-IG or Random despite `configs/config.yaml` declaring `policy.type: dqn`. Both were fixed and both were first trained/benchmarked on 2026-10-01. See `RL_TRAINING_REPORT.md` and `POLICY_BENCHMARK_REPORT.md`.

### Data provenance (2026-10-01) — ⚠️ two files are DERIVED, not originals
All three CSVs were re-fetched from scratch on 2026-10-01. Any agent re-doing this must know two files are conversions:

| Dataset | Source | Status |
|---|---|---|
| **Saudi 506** | `github.com/Sugandaram/Autism-Spectrum-Disorder-Screening-Data-for-Toddlers-in-Saudi-Arabia-Data-Set` (only repo found hosting the file) | **Original CSV.** 506 rows, `A10..A1` order, Class 341/165 — matches §15 |
| **UCI Child 292** | `archive.ics.uci.edu/static/public/419/data.csv` | ⚠️ **DERIVED.** UCI **no longer serves an ARFF for id 419** — only CSV. `Autism-Child-Data.arff` was generated from the official CSV: same 292 rows, same 21 columns, `NaN`→`?`. Verified: 292 records, 151/141 labels, 90 `?` markers (43 in `ethnicity`, 43 in `relation`, 4 in `age`) — the 90 matches the documented count. **Not the original ARFF file.** |
| **Polish 252** | Mendeley Data `tmpkt2mfkg` (`QCHAT_dataset1.sav`, sha256 `7fed516f…` verified against Mendeley's published hash) | ⚠️ **DERIVED CSV from SPSS `.sav`.** The `.sav` is the original; `polish_qchat.csv` was generated via pyreadstat. `group`/`sex` written using the file's **own SPSS value labels** (group 1=ASD, 7=control; sex 1=Male, 2=Female), not guessed. Verified: 252 records, 135 ASD / 117 control, `label_source=clinical`, invalid `qchat4=11.0` at row 60 / `bdbp0221` — matches `DATA_VERIFICATION_REPORT.md` exactly. |
| **NZ 1054** | — | Still **absent**, V-1 blocked (licence "Unknown"). Never downloaded. |

**Verification after restore:** Polish state count recomputed from the restored file → `2,667,729,775`, exactly matching `STATE_COUNT_VERIFICATION.md`. Canonical → `3,081,146,397`. Both confirmed.

### Data validation (2026-08-30 + 2026-09-04)
- **Saudi** (506): A10..A1 reversed → remapped to A1..A10, `Screening Score == sum(A)` 506/506, `Class` deterministic `>=4`, `label_source=questionnaire`, circularity **Deterministic**
- **Polish** (252, 135 ASD / 117 control — resolves 252/253 discrepancy): 25 qchat categorical strings, per-item vocab sorted encoding (excluding 11.0), invalid qchat4 value 11.0 (row 60 `bdbp0221`) logged explicitly to `_POLISH_INVALID_LOG`, `UserWarning` emitted, `label_source=clinical`, circularity **Not circular**, observed `m_list` total **2,667,729,775** vs canonical 3,081,146,397
- **UCI Child** (292) via ARFF: 90 `?` markers, matches UCI ID 419, circularity **Deterministic** at thr 7
- **NZ toddler target** (1,054): source located 2026-09-04 at `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv` and 5 GitHub mirrors. Verified: 1,054 rows × 19 cols; Class Yes/No = 728/326 (matches spec §15); `Qchat-10-Score == sum(A)` 1,054/1,054; Age_Mons 12-36 (toddler-only); circularity **Deterministic** at thr 4 (exact_match = 1.0000). No file downloaded by agent. **Licence on Kaggle = "Unknown"** is the only remaining V-1 blocker.
- **NZ combined** (6,075 pooled): retained unchanged, NOT used for any reported result

### Step 2 — Predictor + DP tractability sweep (2026-09-04, re-run on real data 2026-09-30)
- `scripts/step2_train_and_sweep.py`:
  - Trains `MaskedMLP[128,64] + isotonic` on Saudi 506 (4-fold stratified: 284 train / 95 val / 127 test) and UCI Child 292 (164 / 55 / 73).
  - Runs 48-cell `ExactDP` tractability sweep on Saudi: N ∈ {10, 25, 50, 100, 250, 506} × B ∈ {3, 4, 5, 6} × λ ∈ {0.00, 0.01}.
  - **All 48 runs `status=optimal`**, no tractability violation (largest run N=506 B=6 → 25,023 states / 68,408 evals / 2.2 s on a single CPU thread).
- Predictor test metrics (**real CSVs**, 2026-09-30 run):
  - Saudi: Brier 0.0134, ECE 0.0168, AUROC 0.9877, logloss 0.1234 (near-ceiling AUROC expected — labels deterministically circular, §16.1 gate applies)
  - UCI Child: Brier 0.0713, ECE 0.0765, AUROC 0.9301, logloss 0.9562 (same circularity caveat)
- Artifacts: `results/predictor_{saudi,uci_child}_metrics.json`, `results/dp_tractability_sweep.{json,csv}` (48 rows, all `source=real`)
- 10 invariant tests in `tests/test_dp_tractability_sweep.py`
- (Prior 2026-09-04 synthetic-fallback numbers — Saudi Brier 0.2421/AUROC 0.6230; sweep largest 72,964 states — preserved in `AUDIT_UPDATE_2026-09-04.md` as historical record; superseded.)

### Step 3 — Preliminary report artifacts (2026-09-04, re-run on real data 2026-09-30)
- `scripts/step3_preliminary_reports.py`:
  - **Performance vs budget** on Saudi test (127 episodes) at B ∈ {1..6}: greedy IG vs random. Terminal (B=10) reference: Brier 0.0053, UAR 0.9881, AUROC 0.9997, ECE 0.0049 (real data; ceiling reflects label circularity).
  - **Faithfulness** at B=6, τ=0.5: counterfactual flip rate 0.417, robust 0.583; SHAP |attr| mean per A1..A10 (top: A8 > A6 > A2).
  - **Subgroup** by sex × age_band, B=6, τ=0.5 (sex_F n=86 UAR 0.935; sex_M n=41 UAR 0.892); underpowered cells (< 20) marked per §25.
- All artifacts tagged `"preliminary — V-4 / V-6 / V-7 PENDING; supervisor sign-off required"` and carry `source=real`.
- Polish isolation enforced: nothing in this script touches the Polish cohort.
- 4 contract tests in `tests/test_step3_artifacts.py`

### V-2 — PRISMA template (2026-09-04)
- `V2_PRISMA_SEARCH_LOG.md` with sections 1-8 (research question, 10 databases, 12 exact search strings, inclusion/exclusion criteria with tie-break rules, PRISMA flow, screening worksheet schema, required outputs, no-claim rule)
- `docs/prisma/screening_worksheet.csv` with column schema + 1 example row
- 3 contract tests in `tests/test_v2_no_claim_rule.py` (no-claim rule enforcement, template presence, required sections)
- No-claim rule forbids `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) in non-allowlisted markdown

### Environment rebuild + real-data re-run (2026-09-30)
- Rebuilt venv (`.venv/`, Python 3.14) with full `requirements.txt` — prior `/tmp/aar` venv was lost with `/tmp`.
- Found `results/` empty (gitignored; prior artifacts never survived the environment) — regenerated all 8 artifacts from the real CSVs in `data/raw/` (Saudi 506, Polish 252, UCI Child 292 all present locally; only NZ 1,054 missing pending V-1).
- Every artifact now carries `"source": "real"` (previously `synthetic`). Numbers changed accordingly — see Step 2/3 sections and `STATE_COUNT_VERIFICATION.md`.
- Fixed `tests/test_v2_no_claim_rule.py` scan scope: it was walking `REPO.rglob("*.md")` and failed once `.venv/site-packages/**/*.md` appeared inside the repo. Now skips hidden directories (environment artifacts are not project markdown; the rule's intent is unchanged).

### Test suite (superseded — see 2026-10-01)
The 2026-09-30 entry said 43 tests. That figure predated the 11 demo-behaviour tests in `56a1e3d`; 54 was correct then, and 67 is correct now. Breakdown in the 2026-10-01 test table below.

### Documentation
- `DATA_VERIFICATION_REPORT.md` (2026-09-04): all 4 datasets, NZ updated with V-1 resolution
- `STATE_COUNT_VERIFICATION.md` (2026-09-04): theoretical table expanded (B=3..6 for Q-CHAT-10), Polish canonical + observed numbers verified, Step 2 sweep table with 48-run matrix
- `AUDIT_REPORT.md` (2026-09-04): circularity table with predicted NZ row, Step 2/3 audit trail
- `README.md` (2026-09-04): repo layout, reports table, gate status table, V-2 no-claim rule, allowlist
- `V1_NZ_DATASET_RESOLUTION.md` (2026-09-04): 8 sections — audit result, source, schema verification, circularity recompute, binary mapping, loader plan, human action, interim status
- `V2_PRISMA_SEARCH_LOG.md` (2026-09-04): 8 sections + example row
- `docs/prisma/screening_worksheet.csv` (2026-09-04): schema + 1 example

## Failed
- None.

## Blocked

| Gate | Description | Blocker (2026-09-04) |
|---|---|---|
| V-1 | NZ primary cohort licence | Human must obtain licence-clear copy of `Toddler Autism dataset July 2018.csv` (1,054 rows) and place at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`. Source identified at `kaggle.com/datasets/mamoonamushtaq/toddler-autism-dataset-july-2018-csv`; content validated; **Kaggle licence = "Unknown"**. No file downloaded by agent. |
| V-2 | Systematic literature search log | Human operator runs the 12 queries across 10 databases and populates `docs/prisma/screening_worksheet.csv`. PRISMA template + 1 example row ready. Until complete, no-claim phrases forbidden. |
| V-4 | MDE / comparison family freeze | Before opening Polish for confirmatory. |
| V-5 | Order-independence evidence | Not started. |
| V-6 | λ grid sign-off | Placeholder values λ ∈ {0.0, 0.01} used in Step 2; supervisor must sign off before any λ-vs-utility claim. |
| V-7 | Polish denominator freeze | 117 controls resolved (per `DATA_VERIFICATION_REPORT.md` §Polish); pending supervisor freeze before any confirmatory Polish run. |
| V-9 | Ethics review | Not started. |
| V-10 | Tier-3 controlled-access go/no-go | Not started. |

## Next Required Steps (in priority order)

> **Re-prioritised 2026-10-01 after the fourth pass.** Items 0-4 below are done;
> what remains is listed from item 5. The P1-e, P1-f and P1-g phases have not
> been started.

0. ~~P0 collector defects~~ ✅ `RL_TRAINING_REPORT.md §2`
0. ~~P0 split contamination (P0-9)~~ ✅ `RL_TRAINING_REPORT.md §3.1`
0. ~~P0 unseeded predictor (P0-10)~~ ✅ artifacts now reproducible from their seed
0. ~~P1-b predictor v1 → v2~~ ✅ `RL_TRAINING_REPORT.md §3.2`
0. ~~P1-a V-6 evidence pack~~ ✅ `V6_LAMBDA_DECISION.md` — **decision input only; V-6 is NOT signed off**
0. ~~P1-d wire the §17 baselines~~ ✅ `AUDIT_REPORT.md` — **H1 now measurable, and not supported at B=5/6**
0. ~~P1-e bounded DQN diagnosis~~ ✅ `POLICY_BENCHMARK_REPORT.md §A.5` — the collapse was an undertrained network; benchmark default raised to 2,000 episodes

1. **Supervisor sign-off on the λ grid (V-6).** This is the single blocking gate for every cost-dependent claim, and the evidence pack is ready. Proposed: λ ∈ {0, 0.005, 0.01, 0.02, 0.05} as a cost–utility frontier. Nothing in the agent's scope can substitute for this. **Step 7 strengthens the case**: PPO's stopping behaviour *degrades* with more training precisely because λ=0 leaves stopping unpriced.
2. **Multi-seed the Step-5 table.** The 5-seed variance now exists only in the Step-7 diagnosis. Add `--seeds` to `step5_policy_benchmark.py` and report mean ± sd, so the A.3 gap differences become interpretable. Bounded scope — do not open-ended-tune the network.
3. **Consider promoting the return-to-go target to the primary DQN.** Step 7 shows it reaches plateaued behaviour ~10× sooner (200 vs 2,000 episodes) and is state-coverage-equivalent. That is a spec §11.2 deviation (Double DQN with a bootstrap target), so it is an owner decision, not an agent one. It is currently recorded as an ablation in Step 7.
4. **P1-f — a prior-based policy as a NEW arm** (`src/policies/beta_greedy.py`): Beta(1,1) posterior on the support so `H(Y|s)` is non-zero on a pure support, plus EVOI stopping (`stop when max_j E[Δutility] < λ·c_j`), and it should return the full IG vector as the P3-1 explainability hook. **Do not modify `GreedyIGPolicy`** — its near-optimality result is the project's strongest empirical claim, and adding an arm lets us measure how much of that claim depends on label circularity.
5. **P1-g — promote the saturation finding.** The `V* = 1.000000` / pure-support result belongs in `diagnosisReady.md` and on the benchmark report's front page, and it reframes the framing of RQ2, H1 and AB-7. Polish becomes the load-bearing experiment.
6. **V-1:** Human obtains licence-clear copy of `Toddler Autism dataset July 2018.csv` and places it at `data/raw/Q-CHAT NZ/`. Then run the `_load_nz_toddler_csv` provenance path (`V1_NZ_DATASET_RESOLUTION.md §6`).
7. **V-2:** Human executes the 12 PRISMA queries; agent assists with dedup / inclusion checks.
8. **V-4 / V-7:** Freeze MDE / comparison family; supervisor freezes the Polish denominator. Polish stays sealed until both.
9. **V-9 (IRB)** and **V-10 (controlled access):** not started, longest lead time of anything remaining, and they cost nothing to begin now.
10. After V-2, replace the "preliminary" tag on Step 3 artifacts with the supervisor-signed status.
11. **P4 follow-ups (agent scope, unblocked):** a human expert-review pass over the strategy library (the only thing standing between the support content and any use beyond a research demonstration); a data-protection decision before any follow-up answer is stored; and unit tests for `scripts/demo_live.py` itself (open item 10).

### Known open items (deliberately not fixed — recorded so they are not lost)

| # | Issue | Why deferred |
|---|---|---|
| 1 | **Two disjoint beliefs.** `p_hat` (neural, calibrated) is what the UI shows and the reward uses; `p_emp` (empirical support mean) drives all question selection. They disagree by up to 0.33 absolute. | Design limitation, but it undermines every explanation the demo can give. Address before P3. |
| 2 | **Support collapse.** 70.4% of reachable states have `p_emp ∈ {0,1}`, so `H(Y|s) = 0` and every item's IG is exactly 0; `greedy.py:74` then silently falls back to the lowest legal index. Median support is 3–5 records by depth 8. | Root cause is the circular label plus small N. Needs the Beta prior (item 5). |
| 3 | **Greedy cannot stop.** `GreedyIGPolicy` strips `STOP` from the legal set, so it spends the full budget at every λ. At λ ≥ 0.005 the exact − greedy cost-utility gap is **entirely acquisition cost**. | Preserved on purpose — see item 5. |
| 4 | `src/audits/leakage.py` — `record_fit` is never called from production code, so the §16.2 "blocking" audit records nothing. **It did not detect the P0-9 split leak.** | Now demonstrated to be a real gap, not a theoretical one. |
| 5 | `src/ablation/runner.py` is a string table; `ablation_report` is called from nowhere, so none of the seven §20 ablations has run. AB-3 is blocked by V-6. | Needs the λ sign-off. |
| 6 | `configs/config.yaml` is never parsed by any code — no `yaml.safe_load`, no OmegaConf, no hydra. Every setting is duplicated as a literal. | Tech debt; it is what let the P0-6-style drift recur. |
| 7 | `src/data/schema.py` validation and `dedupe.py` never invoked. `dedupe._record_hash` includes `label`, so a cross-source duplicate with a *disagreeing* label is not flagged. | Low urgency. |
| 8 | ~~`tests/test_trace_belief_update.py:27` — the sequential-consistency assertion was commented out and replaced with `pass`~~ ✅ closed in the ninth pass | — |
| 9 | ~~`src/explain/shap_baseline.py` is **not SHAP**~~ ✅ closed in the ninth pass — documented as a single-flip delta, retained only for artifact compatibility, with a contract test locking the disclosure | — |
| 10 | `scripts/demo_live.py` has zero test coverage. | P4. **Partly closed:** STEP 6c is reachable and was smoke-tested end-to-end on the real cohort, but the script itself still has no unit tests. |
| 11 | `BUDGET = 6` duplicated in `configs/config.yaml:6`, `demo_app.py:47`, `demo_static/index.html:187`; `Session._legal()` never offers STOP so the demo cannot stop early; the UI says "of 10 questions" while the budget is 6. | Demo/UX; P4. The STOP gap is now covered by the frontend's explicit "End session" control; the constant duplication remains. |
| 12 | Isotonic calibration moves ECE the wrong way on a broad uniform state sample (0.0487 → 0.0729) while improving it along the B=6 episode path. The demo uses Platt for this reason. | Open, not resolved. |
| 13 | The P4 strategy library is entirely `pending_expert_review`; no human has reviewed any of it. | A human review pass is required before any use beyond research demonstration; the assembled report discloses the gate. |
| 14 | No P4 batch results artifact is produced, by design: an assembled report contains the person's own answers, and there is no consent/retention story yet. | Deliberate — recorded in `docs/SUPPORT_LAYER_DESIGN.md` §8. |

## Important Decisions
- **NZ 6075 pooled file retained unchanged; NOT used for training** — HUMAN ACTION REQUIRED path enforced in `load_nz()`. Even after V-1, the 6,075-row file stays untouched.
- **NZ 1,054 toddler target source** is a public Kaggle mirror (no direct UCI deposition by the original investigator). The agent downloaded the file from a public GitHub mirror only to validate the schema and recompute the circularity oracle — never committed to the repo, never used for training.
- **Saudi column order A10..A1** explicitly remapped to A1..A10; provenance `raw_screening_score` preserved per §15 auditability rule.
- **Polish qchat4 11.0** treated as invalid data value → MISSING (NaN + missing_mask True) and logged to `_POLISH_INVALID_LOG`, not silently converted — per instruction item 7.
- **Polish categorical encoding** uses per-item sorted vocab excluding 11.0; actual m_list recorded vs canonical 3,081,146,397.
- **Primary state** remains `questions_only`; age/sex excluded from policy state; subgroup via `src/eval/subgroup.py`.
- **Step 2 / Step 3 run on REAL Saudi / UCI Child data as of 2026-09-30** (`source: real` in every artifact). NZ remains synthetic-fallback pending V-1 and is never reported. Prior synthetic-fallback numbers are retained only as historical record in `AUDIT_UPDATE_2026-09-04.md`.
- **No novelty claim** uses `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) anywhere under this repo. Enforced by `tests/test_v2_no_claim_rule.py`.

## Files Changed (2026-10-01)

### New
- `diagnosisReady.md` — diagnostic-readiness audit + roadmap (read this first)
- `RL_TRAINING_REPORT.md` — Part 1 defects, fixes, first real training runs
- `POLICY_BENCHMARK_REPORT.md` — **Part 2 results: greedy attains the exact optimum, RL does not**
- `src/policies/replay.py` — `ReplayBuffer` (stores per-transition legal-action sets)
- `scripts/step4_train_policies.py` — first entry point that trains DQN/PPO
- `scripts/step5_policy_benchmark.py` — matched-budget benchmark incl. ExactDP reference + optimality gap
- `tests/test_rl_training.py` — 13 regression tests for the RL defects
- `tests/test_step5_benchmark.py` — 12 benchmark contract tests
- `results/step4_policy_training_saudi.json` — first real training run (gitignored)
- `results/step5_policy_benchmark_saudi.{json,csv}` — first real benchmark (gitignored)

### Modified
- `src/policies/dqn.py` — `train_step` rewritten: legal mask applied, duplicate `torch.cat` collapsed
- `src/policies/ppo.py` — **added** `train_step` / `select_action` / `legal_mask`; batch format aligned with DQN
- `AGENT_PROGRESS.md` — this file
- `README.md` — test counts, RL section, layout, reports table, invariants

### Derived data files (⚠️ not originals — see Data provenance)
- `data/raw/UCI/Autism-Child-Data.arff` — generated from UCI's official CSV; UCI no longer serves ARFF for id 419
- `data/raw/Q-CHAT Polish/polish_qchat.csv` — generated from the original Mendeley `.sav` via pyreadstat

## Files Changed (2026-09-04 audit update)

### New
- `V1_NZ_DATASET_RESOLUTION.md` — V-1 resolution document (8 sections)
- `V2_PRISMA_SEARCH_LOG.md` — PRISMA template (8 sections + example)
- `docs/prisma/screening_worksheet.csv` — schema + 1 example row
- `scripts/step2_train_and_sweep.py` — predictor training + DP tractability sweep
- `scripts/step3_preliminary_reports.py` — preliminary report artifacts
- `tests/test_dp_tractability_sweep.py` — 10 DP sweep invariants
- `tests/test_step3_artifacts.py` — 4 Step 3 contract tests
- `tests/test_v2_no_claim_rule.py` — 3 V-2 contract tests
- `results/predictor_saudi_metrics.json` — Saudi predictor metrics
- `results/predictor_uci_child_metrics.json` — UCI Child predictor metrics
- `results/dp_tractability_sweep.json` — 48 DP runs (full)
- `results/dp_tractability_sweep.csv` — 48 DP runs (CSV summary)
- `results/perf_vs_budget_saudi.csv` — Step 3 perf-vs-budget
- `results/perf_vs_budget_saudi.json` — Step 3 perf-vs-budget metadata
- `results/faithfulness_saudi.json` — Step 3 faithfulness
- `results/subgroup_saudi.json` — Step 3 subgroup analysis

### Modified
- `AGENT_PROGRESS.md` — this file
- `README.md` — repo layout, reports table, gate status table, V-2 no-claim rule
- `DATA_VERIFICATION_REPORT.md` — V-1 resolution note + Step 2 results
- `STATE_COUNT_VERIFICATION.md` — Step 2 systematic sweep + verified verification commands
- `AUDIT_REPORT.md` — predicted NZ row + Step 2/3 audit trail

### Unchanged
- `Master-Project-Specification_FINAL.md` — source of truth, allow-listed in no-claim test
- `src/data/ingest.py`, `src/solvers/exact_custom.py`, `src/models/masked_predictor.py`, etc. — code untouched
- `configs/config.yaml` — unchanged
- `requirements.txt` — unchanged

## Tests Last Run (2026-10-01, after the fourth pass)

```
$ .venv-win/Scripts/python -m pytest tests -q
........................................................................ [ 48%]
........................................................................ [ 96%]
......                                                                   [100%]
150 passed in 25.02s   (intermediate: after the fourth pass)
```

Final state after the fifth pass: **187 passed, 0 skipped** — 13 new tests in
`tests/test_step7_rl_diagnosis.py`.

| Test class | Count | Status |
|---|---:|---|
| Step-7 RL-collapse diagnosis (`tests/test_step7_rl_diagnosis.py`) | 13 | PASS |
| Step-6 λ sweep + canonical split (`tests/test_step6_lambda_sweep.py`) | 31 | PASS |
| P1-d baselines + predictor reproducibility (`tests/test_p1d_baselines.py`) | 23 | PASS |
| RL training regressions (`tests/test_rl_training.py`) | 28 | PASS |
| Step 5 benchmark contracts (`tests/test_step5_benchmark.py`) | 14 | PASS |
| Demo behaviour audit regressions (`tests/test_demo_behavior_audit.py`) | 11 | PASS |
| DP tractability sweep invariants | 10 | PASS |
| §21 core (state, budget, legal-actions, encoding, count, reward, predictor, exact, circularity, leakage, counterfactual, threshold-freeze, common-evaluator, trace, fixed-subset) | 22 | PASS |
| Step 3 artifact contracts | 4 | PASS |
| V-2 no-claim rule + PRISMA template | 3 | PASS |
| Leakage audit | 3 | PASS |
| **Total** | **163** | **PASS, 0 skipped** |

**Correction to the record:** this file previously said "43 tests" and a "working tree modified" state; the 43 figure predated the 11 demo-behaviour tests added in commit `56a1e3d`. README's "54" was the correct pre-2026-10-01 count. 13 RL tests → 67, then 12 Step-5 tests → 79.

| Test class | Count | Status |
|---|---:|---|
| (see the 2026-10-01 test table above — 67 total) | 67 | PASS |

End-to-end pipeline:
```
$ .venv/bin/python scripts/step2_train_and_sweep.py
... 48 DP runs all status=optimal ...
→ results/dp_tractability_sweep.json (n_runs=48)
→ results/dp_tractability_sweep.csv

$ .venv/bin/python scripts/step3_preliminary_reports.py
... Step 3 artifacts ...
→ results/perf_vs_budget_saudi.csv
→ results/perf_vs_budget_saudi.json
→ results/faithfulness_saudi.json
→ results/subgroup_saudi.json
```

Theoretical reference (verified):
```
$ .venv/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))"
26025
$ .venv/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]))"
3081146397
$ .venv/bin/python -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5]))"
2667729775
```

## Last Known Good State
- 2026-08-30: non-NZ infrastructure complete, real-data ingest validated, theoretical vs empirical distinguished, NZ flagged missing, tests green, Polish isolated (V-4/V-7 pending, no tuning)
- 2026-09-04 (audit update): all four plan steps (V-1 source location, Step 2 sweep, Step 3 preliminary reports, V-2 PRISMA template) complete and locked behind contract tests. 43/43 tests pass. Every markdown file updated and cross-referenced. Polish isolation enforced. No novelty claim wording. Licence on the NZ 1,054-row file is the only remaining V-1 blocker.
- 2026-09-30 (env rebuild + real-data run): venv rebuilt, results/ regenerated from the real Saudi/UCI CSVs (`source: real` everywhere), no-claim test scope fixed to skip hidden dirs, all four living docs updated with real numbers. 43/43 tests pass (superseded count — see 2026-10-01). The 8 verification gates are unchanged — all remaining substantive work is gated on human/supervisor actions (V-1 licence, V-2 searches, V-4/V-6/V-7 sign-offs, V-5/V-9/V-10 not started).
- 2026-10-01 (Parts 0+1): repo audit found **no RL policy had ever been trained** — `DQNPolicy.train_step` was never called and discarded its legal-action mask (bootstrap optimised toward illegal actions); `PPOPolicy` had no training method at all. Both fixed, replay buffer + training script added, 13 regression tests written. A third alleged defect (terminal next-state shapes breaking `torch.cat`) was investigated and **retracted** — `torch.cat(dim=0)` concatenates along dim 0, so the original code did not raise; do not repeat that claim. Environment rebuilt (CPU-only torch; `/tmp` tmpfs is too small for the CUDA build) and all three datasets restored — two of them as documented conversions. **67/67 tests pass**.
- 2026-10-01 (Parts 2+3): **first real RL measurement.** `scripts/step5_policy_benchmark.py` benchmarks DQN/PPO/Greedy/Random/ExactDP at matched budgets B∈{1..6} with a shared evaluator, plus the V\*−V_emp optimality gap. **Result: greedy-IG attains the exact optimum (gap ≤0.013); DQN (~0.04–0.055) and PPO (0.015–0.138) do not — PPO is worse than random at B=5/B=6.** Negative result for RL on this task, reported as such. 12 benchmark tests added → **79/79 pass**. `POLICY_BENCHMARK_REPORT.md` written. Polish still sealed; 8 verification gates unchanged. **Top next steps: multi-seed variance, then V-6 (λ) — every current number is λ=0, so cost is unpriced and adaptive stopping is unexcused.**

---

## 2026-10-01 (eighth pass) — V-4 / V-7 validation phase

Opened the external-validation phase against the sealed Polish cohort. It is
correctly **blocked**, and no gate was removed to make it pass.

### Dataset question resolved first

`data/QCHAT_dataset2 mendeley.sav` was verified as the **same 252-participant
cohort** already integrated as `data/raw/Q-CHAT Polish/polish_qchat.csv` — 252/252
keys, 252/252 numeric agreement, identical schema and value-label vocabularies,
0 duplicates. Its sha256 `7fed516f…` is the hash this repository already recorded
for the verified `QCHAT_dataset1.sav` export, so the local `dataset2` filename is
misleading. **Disposition: provenance evidence only, never ingested.**

### Current gate states

| Gate | Automated | Human sign-off | Overall |
|---|---|---|---|
| V-4 MDE + confirmatory family freeze | PASS | **OPEN** | **OPEN** |
| V-7 Polish denominator (252 vs 253) | PASS | **OPEN** | **OPEN** |
| V-7 Baseline 10 recomputation | OPEN | **OPEN** | **OPEN** |
| External validation on the sealed cohort | — | — | **BLOCKED** |

### Two blockers, independently sufficient

1. V-4/V-7 human sign-off outstanding — the cohort stays sealed.
2. Feature-contract mismatch: the frozen Saudi predictor declares 10 binary items
   (`input_dim = 41`); the Polish cohort supplies 25 ordinal items
   (`input_dim = 199`). `encode_state` and the model classes already accept
   `m_list`, but `load_polish` never populates it. Coercion is refused by design.

### Circularity contrast that motivates the whole phase

| Cohort | exact match | classification |
|---|---|---|
| Polish `GROUP` (clinical) | 0.5437 | **Not circular** |
| Saudi `Class` (questionnaire) | 1.0000 | Deterministic |

### Commands

```powershell
.venv-win\Scripts\python scripts\step9_v4_v7_gates.py
.venv-win\Scripts\python scripts\step10_external_validation.py   # exits 2 while blocked
.venv-win\Scripts\python -m pytest tests -q
```

Tests: **487 passed, 0 skipped** (was 204; +283).
Detail in `AUDIT_REPORT.md` (eighth pass) and `DATA_VERIFICATION_REPORT.md`.
