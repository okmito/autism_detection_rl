# Support Layer — Design Record (P4, revised)

**Date:** 2026-10-09 · **Claim boundary: research prototype. Screening support only.
NOT a diagnosis, NOT a diagnostic device, NOT clinically validated.**

**Revision note.** The first version of this layer (schema `support-report/1.0`)
asked a *second* questionnaire after the screening result and required an
endorsement before any suggestion could be offered. That design was replaced:
the follow-up questionnaire, its answer vocabulary, its validation, its API
route and its screen are **removed**. The support report is now generated
automatically from the answers the person already gave. This record describes
the revised design (`support-report/2.0`); the P3-outcome record for the
explanation it consumes is `docs/EXPLAINABILITY_OUTCOME_PLAN.md`.

---

## 1. What the revision had to fix first

Before any mapping could be trusted, the questionnaire itself had to be pinned
down. `scripts/demo_app.py` carried a hand-written list of item wording that had
**drifted from the verified feature contract** (`src/data/qchat10_contract.py`,
signed off in `docs/QCHAT10_PROJECTION_APPROVAL.md`): from A3 onward, the demo
asked about one construct while the answer was recorded as the feature for a
*different* construct. No test compared the two, so eight questions were
mis-labelled and the support linkage written against them was wrong in three
places.

`src/support/questionnaire.py` now derives the instrument from the contract —
construct, feature id, Polish variable and scoring direction are all read from
it, never restated — and `tests/test_support_questionnaire.py` asserts that the
demo's displayed wording, the contract's feature order and the support layer's
item table agree. Correcting the wording is display-only: the stored 0/1 value
and its feature index are unchanged, so screening behaviour, the RL environment
and every benchmark artifact are byte-identical.

## 2. Architecture

```
run_episode ──► {p_hat, decision, final_state, trace, items_asked, stop_reason}
                     │
                     ▼
     src/explain/outcome.py :: explain_outcome(...)   (outcome-explanation/1.0)
                     │
                     ▼
     src/support/report.py :: build_support_report(episode_result, explanation)
        ├─ questionnaire.py   the instrument, derived from the verified contract
        ├─ schemas.py         Pydantic contracts (support-report/2.0)
        ├─ domains.py         which items may justify which area, + unmeasured areas
        ├─ strategies.py      curated content, provenance, review status
        ├─ engine.py          deterministic rules; assessments → suggestions
        └─ report.py          assembly + instrument-level validation
                     │
                     ▼
        ScreeningReport (support-report/2.0)
             ├─► scripts/demo_app.py    result payload (support_report)
             └─► scripts/demo_live.py   STEP 6c
```

Nothing here trains, retrains or mutates anything. The support layer is a pure
consumer of the episode result, the explanation object and the recorded
answers. The environment, the reward, τ, the seeds, the splits and both frozen
predictors are untouched.

## 3. The safety property, restated

The previous rule was *"a hypothesis never fires a recommendation"* (only a
user-confirmed need fired). With no second questionnaire that rule would mean no
suggestions at all, so it is replaced by:

> **A suggestion may only be offered because of an answer that was actually
> given, and it must name the items and the responses that triggered it.**

Enforced structurally, in three places:

1. `SupportRecommendation.triggering_question_ids` + `triggering_responses` are
   required, validated against the instrument, and their length must match
   (`src/support/schemas.py`);
2. `ScreeningReport` fails closed: a suggestion whose trigger ids are not part
   of the report's own recorded evidence is a validation error, not a dropped
   field;
3. the engine's rule table is per-strategy: a strategy fires only when *its own*
   trigger items were answered atypically.

Framing follows the same rule: a suggestion says it is offered *because of the
kind of answer recorded*, never that a difficulty exists, is likely, or is
measured. There is no scoring, ranking or weighting by severity, and no
probability of any kind.

## 4. Verified mapping — what Q-CHAT-10 measures, and what it does not

The instrument asks about: looking when called, ease of eye contact, pointing to
request, pointing to share interest, pretending, following gaze, wanting to
comfort an upset person, first words, simple gestures, and staring at nothing.

| Area (report label) | Trigger items (verified constructs) | Suggestion |
|---|---|---|
| Getting a message across | A3 points to request, A4 points to share interest, A8 first words, A9 simple gestures | `rec_comm_visual_supports` |
| Getting attention before you speak | A1 response to name, A2 ease of eye contact | `rec_comm_attention_cues` |
| Extra processing time | A1, A2, A6 follows your gaze | `rec_comm_processing_time` |
| Noticing and naming feelings | A7 wants to comfort an upset person | `rec_emotion_name_what_you_see` |

Items **asked but with no curated, reviewed suggestion**: A5 (pretending) and
A10 (staring at nothing). They are disclosed in the report instead of being
quietly dropped.

Areas the questionnaire **does not measure**, listed in `unassessed_areas` and
never offered: sensory comfort, predictability/routines/transitions, daily
organisation, and preferred information format. Eight previously curated
strategies address those areas and are therefore structurally unreachable; they
stay in the library as *reviewed-as data* in
`NON_TRIGGERABLE_STRATEGIES`, each with an explicit `no_link_reason`, and
`tests/test_support_engine.py` asserts exhaustively that no answer pattern can
produce one. Two more (about the child's own moments of distress) are parked for
the same reason: A7 measures responding to *another* person's distress, not the
child's own.

## 5. Explanation content (all verified fields)

`explain_outcome()` is reused unchanged — exact group-Shapley attribution over
observed items, verified counterfactuals, the v3 prior-sensitivity band
(reported unavailable for v2 rather than fabricated), limitations and the
deterministic renderer. The report adds the instrument's own wording for each
recorded answer (`questionnaire_evidence`), so a reader can see exactly which
question each contribution refers to. Generation status stays PARTIAL whenever
something the architecture could not supply is missing (no attribution, no
uncertainty band — every v2 session — or explanation warnings).

## 6. Pydantic contracts and what they reject

| Model | Role |
|---|---|
| `ScreeningResult` | the frozen outcome: decision, `p_hat`, τ, items asked, budget, model identity |
| `QuestionnaireEvidence` | one recorded answer — id, verified wording, binary response, feature id; ids, wording and feature id are all checked against the instrument |
| `FeatureContribution` | a computed attribution; direction must match the sign, values must be finite |
| `ExplanationResult` | the explanation, with evidence, contributions, uncertainty and limitations |
| `SupportNeedAssessment` | `EVIDENCE_SUGGESTED` / `NO_EVIDENCE` / `NOT_MEASURED`, with trigger ids that must appear in its own evidence |
| `SupportRecommendation` | a suggestion: title, description, triggering items + responses, basis (`OBSERVED_RESPONSE_PATTERN`), source, review status, applicability, limitations |
| `ScreeningReport` | the assembled report, with `unassessed_areas` and fail-closed evidence sufficiency |

Rejected: unknown or malformed question ids, non-binary responses, mismatched
feature ids, invented wording, non-finite numerics, unattributed suggestions,
suggestions justified by answers not in the report, and unknown fields
everywhere (`extra="forbid"`).

## 7. What this layer does not do

* It does not diagnose, and no wording presents it as a diagnosis.
* It does not establish a need, a difficulty, a severity or a probability.
* It does not invent uncertainty or prioritisation, and it does not claim any
  suggestion is clinically validated: the whole library ships
  `pending_expert_review` and the report says so.
* It does not use an LLM anywhere in the decision path. If a verbaliser is ever
  added it must be restricted to verified fields, with a field-grounding test.
* It does not ask anything. There is no second form, no optional questions, and
  no endpoint to submit answers to.

## 8. Files

| File | Role |
|---|---|
| `src/support/questionnaire.py` | the instrument, derived from the verified contract |
| `src/support/schemas.py` | Pydantic contracts (`support-report/2.0`) |
| `src/support/domains.py` | trigger mapping, unmeasured areas, asked-without-suggestion items |
| `src/support/questions.py` | **deleted** (the second questionnaire) |
| `src/support/strategies.py` | triggerable + non-triggerable libraries, provenance, review status |
| `src/support/engine.py` | assessments, suggestions, deterministic rules, bounded ids |
| `src/support/report.py` | assembly from the episode + explanation alone |
| `scripts/demo_app.py`, `demo_static/index.html`, `demo_live.py` | auto-generated report on the result screen / STEP 6c; no support form or endpoint |
| tests | `test_support_questionnaire`, `_schemas`, `_registry`, `_questions` (**deleted**), `_strategies`, `_engine`, `_report`, `_demo` |

## 9. Open items

* The strategy library is unreviewed; a human expert-review pass is required
  before any use beyond research demonstration, and the report discloses that
  gate as `review_status`.
* The follow-up answers that used to be collected no longer exist; nothing
  about a person is stored by this layer, and no batch artifact of reports is
  written for that reason.
* Coverage is deliberately narrow: four areas out of ten items. Widening it
  needs either new instrument items (a research-protocol change, not a code
  change) or human review of the parked content.
* The demo's support panel is functional but minimal; it has no accessibility
  audit beyond the patterns the rest of the demo already uses.
