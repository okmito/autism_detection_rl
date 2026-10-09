# Support Layer — Design Record (P4)

**Date:** 2026-10-09 · **Claim boundary: research prototype. Screening support only.
NOT a diagnosis, NOT a diagnostic device, NOT clinically validated.**

P4 adds what happens **after** the screening result and its explanation: an
optional, skippable follow-up questionnaire and a deterministic set of
*support ideas*. This document records what was built, the one safety property
the layer is designed around, and what it deliberately does not do. The
implementation lives under `src/support/`; the P3-outcome record for the
explanation it consumes is `docs/EXPLAINABILITY_OUTCOME_PLAN.md`.

---

## 1. The problem this layer has to not create

The screening result is a **calibrated model score plus a thresholded referral
recommendation**. It is not a finding of need, and on this project's datasets it
cannot be: the labels are questionnaire-derived (label circularity is measured
and disclosed, `DATA_VERIFICATION_REPORT.md`). A layer that turns a screening
score into "your child needs help with communication" would invent a need from a
correlated feature — the exact failure mode this project's audit trail exists to
prevent.

So the layer is built around one structural rule:

> **A hypothesis never fires a recommendation.**

Only the person's own follow-up answer (or their stated preference) can produce
a suggestion, and every suggestion carries the assessment id that triggered it
plus a plain-language `why_selected`.

## 2. Architecture

```
run_episode ──► {p_hat, decision, final_state, trace, items_asked, stop_reason}
                     │
                     ▼
     src/explain/outcome.py :: explain_outcome(...)   (schema outcome-explanation/1.0)
                     │
                     ▼
     src/support/report.py :: build_support_report(episode_result, explanation,
                                                   followup_answers)
        ├─ schemas.py      Pydantic contracts; every invariant is a validator
        ├─ domains.py      which observed items may justify ASKING about a domain
        ├─ questions.py    the optional follow-up questionnaire (neutral wording)
        ├─ strategies.py   curated strategies, all pending human expert review
        ├─ engine.py       deterministic rules; assessments → recommendations
        └─ report.py       assembly + adaptation from the explanation schema
                     │
                     ▼
        ScreeningReport (schema support-report/1.0)
             ├─► scripts/demo_app.py     POST /api/session/support  (browser demo)
             └─► scripts/demo_live.py    STEP 6c                    (terminal demo)
```

Nothing here trains, retrains or mutates anything. The support layer is a pure
consumer of the episode result, the explanation object, and the person's own
answers. The environment, the reward, τ, the seeds, the splits and both frozen
predictors are untouched; every benchmark artifact is unaffected.

## 3. Evidence types are a type-level distinction

`EvidenceItem.source_type` has exactly three members, and no field name or
convention can blur them:

| source type | meaning | who produces it |
|---|---|---|
| `observed_response` | a recorded screening answer (`item_code`, `observed_value` required) | `evidence_from_episode` |
| `model_derived` | produced by a fitted component (e.g. a Shapley contribution) | the explanation adapter |
| `user_reported` | the person's own follow-up answer or stated preference | the engine |

The only bridge from screening data to a support question is one-way and runs
through the person: an atypical observed response can put a domain's follow-up
question on the *suggested* list (`engine.questions_to_offer`), and only the
person's answer to that question can establish a need.

## 4. The honest-status table

Every domain gets exactly one assessment, and its status is one of six values
with defined consequences:

| status | set when | produces suggestions? |
|---|---|---|
| `user_confirmed` | the person answered "yes" to the domain's question | **yes** (basis `user_confirmed`) |
| `user_stated_preference` | the person named a preference (choice question, answered "yes") | yes, as `general_guidance` — a preference is not a need |
| `hypothesis_from_observed` | atypical observed response(s), no follow-up answer | **never** |
| `user_declined` | the person answered "no" | no |
| `unknown` | `unsure` / `not_applicable` / `prefer_not_to_answer` | no, and nothing is inferred either way |
| `insufficient_evidence` | no observed linkage and no answer | no |

`user_confirmed` requires `assessment_method = user_report` and
`user_confirmed = True`; a hypothesis can never be marked confirmed. Both are
Pydantic model validators, so a bug in the rules fails loudly at assembly time
rather than silently shipping a need the person never stated.

Domains whose `evidence_item_codes` are empty (sensory, transitions, daily
living, accessibility) have **no screening-data linkage**: their questions say so
in their helper text, they can never be suggested from data, and they can only
ever be user-reported.

## 5. Method choices and why

| Question | Choice | Why this one |
|---|---|---|
| How are needs established? | The person's own answer, recorded verbatim | The screening score is a model output on circular labels; it cannot establish a need, and the layer must not pretend it can. |
| What can screening data do? | Put a follow-up question on the *suggested* list | One-way linkage: data may invite the question, never answer it. |
| How are suggestions selected? | A deterministic rule table over assessments | No trained component, no demographic input, fully auditable, and testable in isolation. |
| Where does content come from? | A curated strategy library, all `pending_expert_review` | No external source is asserted without human verification (no URLs are recorded at all); the report discloses the review state. |
| How is honesty enforced? | Pydantic validators + report-level cross-reference checks | `extra="forbid"` everywhere; every `related_assessment_id` / `evidence_reference` must resolve inside the report. |

**Rejected:** deriving suggestions from the RL reward, the policy's internal
state, or any Q-value (they are training signals, not outcomes); an LLM
verbaliser (deferred — if ever added it must be restricted to verified fields
with a field-grounding test); any wording that frames a suggestion as
responding to a difficulty rather than to a request.

## 6. What the layer does not do

* It does not diagnose, and no wording anywhere presents it as a diagnosis.
* It does not turn the screening probability into a need, a severity, or a
  prediction about support effectiveness.
* It does not claim any suggestion is clinically validated: the entire library
  ships `pending_expert_review` and the report says so.
* It does not invent uncertainty, prioritisation, or cost information. There is
  no ranking by severity and no "best" suggestion.
* `RecommendationBasis.HYPOTHESIS_OPTED_IN` exists in the schema but is never
  emitted by the engine, by construction: opting in to a hypothesis *is*
  answering "yes" to its question, which is the stronger classification
  `USER_CONFIRMED`.

## 7. Files

| File | Role |
|---|---|
| `src/support/schemas.py` | Pydantic contracts (`support-report/1.0`), enums, `to_json_safe` |
| `src/support/domains.py` | domain registry and its (honest) evidence linkage |
| `src/support/questions.py` | the optional follow-up questionnaire |
| `src/support/strategies.py` | curated strategy library + guidance provenance |
| `src/support/engine.py` | assessments, recommendations, answer validation, question split |
| `src/support/report.py` | assembly from episode + explanation + answers; adapters |
| `scripts/demo_app.py` | `POST /api/session/support`, `followup` payload on the result |
| `scripts/demo_static/index.html` | optional support screen (renders backend fields only) |
| `scripts/demo_live.py` | STEP 6c terminal walkthrough |
| `tests/test_support_schemas.py`, `test_support_registry.py`, `test_support_questions.py`, `test_support_strategies.py`, `test_support_engine.py`, `test_support_report.py`, `test_support_demo.py` | validation (all synthetic fixtures) |

## 8. Open items for the next phase

* The strategy library is project-curated and unreviewed; a human expert review
  pass is required before any use beyond research demonstration, and the report
  discloses that gate as `review_status`.
* The follow-up questionnaire is offered but never persisted: there is no
  storage, retention or consent story yet, and none should be added without a
  data-protection review. For the same reason this phase deliberately produces
  **no batch results artifact** — a `results/*.json` of assembled reports would
  be a store of people's own answers. The per-session report is the artifact,
  and the tests exercise it on synthetic fixtures only.
* Subgroup fairness of the *suggested* question split has not been measured (the
  linkage is item-level and deterministic, so the exposure is the screening
  trajectory itself).
* The browser demo's support screen is functional but minimal; it has no
  accessibility audit beyond the patterns the rest of the demo already uses.
