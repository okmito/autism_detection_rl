# Outcome-Level Explainability — Design Record (P3-outcome)

**Date:** 2026-10-09 · **Claim boundary: research prototype. Screening support only.
NOT a diagnosis, NOT a diagnostic device, NOT clinically validated.**

This document records what was built, what it explains, and what it deliberately
does not explain. It is the companion to the code under `src/explain/`; the
implementation phases and acceptance criteria live in `AGENT_PROGRESS.md`
(P3-outcome entries).

---

## 1. What the system's outcome actually is (verified)

The final screening outcome is **not** produced by the RL policy. Verified in
source:

1. `src/env/environment.py::run_episode` runs the interview. The policy only
   decides *which questions are asked*.
2. On STOP, the outcome is computed by a **separate calibrated classifier**
   applied to the terminal state: `p_hat = predictor(final_state)`
   (`environment.py:56`), and `decision = "REFER" if p_hat >= tau(=0.5) else
   "NO_REFERRAL_INDICATED"` (`environment.py:62`).
3. The terminal reward `R = (1 - (p_hat - y)^2) - lambda * sum(c_j)` is scored
   against the record label; it is a training/evaluation signal, **not** the
   outcome and **not** a confidence.
4. Two predictor versions exist and are untouched by this work:
   * **v2** `src/models/masked_predictor.py` — MLP[128,64] + isotonic, frozen
     legacy (the reported external-validation artifact).
   * **v3** `src/models/logistic_predictor.py` — L2 logistic (C=0.1) + Platt on
     the raw probability, partial states scored by **exact marginalisation**
     over the 2^10 item configurations under a factorised Saudi-train prior.

So the explainability target is: **the calibrated probability and the
thresholded referral decision for one interview**, plus the observed responses
that produced them.

---

## 2. Architecture (post-hoc, no retraining, no inference change)

```
run_episode(...) ──► {p_hat, decision, final_state, trace, items_asked, stop_reason}
                          │
                          ▼
     src/explain/outcome.py :: explain_outcome(episode_result, predictor, ...)
        ├─ attribution.py     exact group-Shapley over questionnaire items
        ├─ counterfactual.py  verified flips + feasibility + minimal sets + unask/ask-more
        ├─ uncertainty.py     v3 prior-sensitivity band (unavailable for v2 — reported, not faked)
        ├─ limitations.py     limitation statements + mandatory disclaimer
        └─ render.py          deterministic text (terminal demo, tests)
                          │
                          ▼
        structured explanation (schema outcome-explanation/1.0)
             ├─► results/outcome_explainability_saudi.json   (scripts/step17)
             ├─► demo result screen                           (scripts/demo_app.py + demo_static/index.html)
             └─► terminal walkthrough                         (scripts/demo_live.py STEP 6b)
```

**Nothing is retrained.** The layer is a pure consumer of `run_episode` output
plus read-only predictor re-evaluation on perturbed states. `environment.py`,
the reward, τ, seeds, action/observation spaces, and both frozen predictors are
byte-for-byte unchanged; the benchmark artifacts are unaffected.

---

## 3. Method choices and why

| Question | Method | Why this one |
|---|---|---|
| Which responses mattered? | **Exact group-Shapley** over questionnaire items (`attribution.py`) | A Q-CHAT-10 interview observes ≤ 6 items, so all 2^k coalitions can be enumerated (≤ 1024 predictor evaluations) — exact, deterministic, no sampling, and the Shapley **efficiency identity** `Σφ = p_hat − v(∅)` is asserted (measured max error 1.1e-16 on the Saudi test split). Cross-checked against the `shap` package's Exact explainer in tests. |
| What could change the result? | Extended counterfactuals (`counterfactual.py`) | Every flip is verified by re-evaluation; feasibility is scored under the factorised training prior; multi-flip minimal sets, unask and ask-more variants added. The original §18.2 `find_counterfactual` is preserved unchanged. |
| How certain is the result? | Prior-sensitivity band (`uncertainty.py`) | The only uncertainty the architecture supports honestly: resampling the item prior at the training sample size (Dirichlet-multinomial). Reported as **model stability, not clinical uncertainty**; reported as *unavailable* for v2 rather than fabricated. |
| Why this sequence of questions? | Not rebuilt — already exists | The acquisition-side trace (`selection_diagnostic`, beta_greedy `explain()`, IRT-CAT `information()`) answers a different question (policy behaviour) and stays where it was. |

**Rejected:** LIME (sampling-based, adds a dependency outside spec §22's stack,
strictly dominated by exact enumeration at n=10); gradient saliency for v3 (a
table lookup has no meaningful gradient; for the v2 MLP the one-hot block inputs
make raw input gradients weak); KernelSHAP approximations (superseded by exact
enumeration). The deprecated `shap_baseline` module (single-flip deltas,
non-additive) is **not used**; it carries a deprecation notice and a contract
test.

---

## 4. The value function (what "contribution" means)

An item is **present** in a coalition when it is held at its observed response;
**absent** means UNASKED, and the predictor's own partial-state semantics then
define absent (v3: marginalised under the prior; v2: the network's learned
behaviour). Therefore:

* `v(∅)` is the model's **prior-only** estimate; attributions explain the
  movement from prior-only to the final estimate.
* `φ > 0` means "given this response, the estimate is higher than it would be
  with the other observed responses held and this one unasked".
* On these datasets responses are **correlated** (the Saudi label is a
  deterministic sum-threshold), so Shapley values split credit among correlated
  items. They are **not causal effects** and the output says so.

---

## 5. Measured faithfulness (Saudi test split, B=6, greedy, v3, 127 episodes)

From `results/outcome_explainability_saudi.json` (Step 17):

| Metric | Value | Reading |
|---|---|---|
| Additivity max error | 1.11e-16 | efficiency identity holds exactly |
| Counterfactual found / robust | 0.346 / 0.654 | share of sessions where one answer change moves the decision |
| Deletion AUC: guided vs random | 0.1413 vs 0.1274 | attribution ordering ranks responses by influence, modestly above chance |
| Insertion AUC: guided vs random | 1.2799 vs 1.1832 | same, stronger |
| Rank stability (one more question observed) | Spearman mean 0.164 | **low** — attributions are not rank-stable when the evidence set grows |
| Explanation overhead | 20.1 ms/episode (max 23.0) | negligible against episode cost |
| Counterfactuals plausible under prior | 0.181 of all sessions | 82% of flipping counterfactuals involve at least one response pattern that is at least as plausible as the original |
| Seen in training | 0.339 | roughly a third of flipping counterfactuals use only patterns present in training |

The stability result is reported as measured, not smoothed: on this predictor
and dataset, per-item attribution ranks move materially when an additional
question is observed, because the prior mass re-concentrates. This is a
limitation of the explanation, stated in the artifact's limitations block.

**None of these numbers is a clinical claim.** They describe the model.

---

## 6. What the explanation does not do

* It does not diagnose, and no wording anywhere presents it as a diagnosis.
* It does not turn an RL action probability, Q-value, or cumulative reward into
  a confidence.
* It does not claim attribution causality, and it does not claim that a
  "correct" explanation implies a clinically correct model.
* Counterfactuals are model-behaviour statements: changing an answer changes
  the model's output, not a person's underlying condition.
* For v2 (the demo's current predictor) no per-session interval is produced.

---

## 7. Files

| File | Role |
|---|---|
| `src/explain/attribution.py` | exact group-Shapley; `AttributionResult` |
| `src/explain/counterfactual.py` | `find_counterfactual` (unchanged §18.2) + `find_counterfactual_rich` |
| `src/explain/uncertainty.py` | `prior_sensitivity_band` |
| `src/explain/limitations.py` | limitation statements + `screening_disclaimer` |
| `src/explain/outcome.py` | `explain_outcome` orchestrator, schema `outcome-explanation/1.0` |
| `src/explain/render.py` | deterministic text rendering |
| `src/models/logistic_predictor.py` | added pure `probability_with_prior` (prediction behaviour unchanged) |
| `scripts/step17_outcome_explainability.py` | artifact + faithfulness metrics |
| `scripts/demo_app.py`, `scripts/demo_static/index.html`, `scripts/demo_live.py` | presentation of the same object |
| `tests/test_outcome_attribution.py`, `test_outcome_counterfactual.py`, `test_outcome_explain.py`, `test_shap_baseline_honesty.py`, `test_step17_outcome_explainability.py` | validation |

## 8. Open items for the next phase

* v2 group-Shapley is implemented and exercised in tests; the Step-17 artifact
  currently reports the v3 scorer only.
* A verbaliser (LLM) layer, if ever added, must be restricted to the verified
  fields of the explanation object with a field-grounding test; it is not part
  of the shipped MVP.
* Regret-style comparison of the explanation against the exact DP policy
  (policy-side analysis) remains out of scope here.
