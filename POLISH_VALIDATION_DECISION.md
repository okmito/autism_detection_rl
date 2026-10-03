# POLISH EXTERNAL VALIDATION DECISION

**Status: AWAITING SUPERVISOR DECISION. No option is selected here, and no
threshold, mapping, or baseline is invented.**

Prepared 2026-10-01 against `Master-Project-Specification_FINAL.md` §15, §19.3,
§19.5, §24. Options are compared on scientific grounds; implementation
convenience is explicitly *not* a criterion and appears in the last
column.

## The situation in one paragraph

The sealed Polish cohort is a Q-CHAT-25 instrument: 25 items, **0 natively
binary**, every item ordinal with 4–6 response levels, `input_dim = 199`. The
frozen development predictor consumes Q-CHAT-10: 10 items, binary 0/1,
`input_dim = 41`. No subset correspondence can be verified, because the cohort
carries **no question wording** and this repository contains no canonical
Q-CHAT-10 or Q-CHAT-25 item definition. External validation therefore cannot run
today, and three options exist.

---

## Option A — Use a validated Q-CHAT-10 subset

**Scientific question answered.** Does the *existing 10-item adaptive screening
system* transfer to an independently labelled population, holding the instrument
fixed?

**Compatibility.** Requires a verified mapping in hand. Today there is none: all
three requirements (scale compatibility, item identity, provenance) are OPEN.
`src/eval/qchat10_subset.py` records this explicitly.

**Leakage risk.** Low, and this is Option A's main virtue. The frozen predictor
is applied unchanged; Polish contributes no fitted parameter. Any ordinal → binary
rule must be declared from the instrument definition and applied *without*
reference to Polish outcomes.

**Impact on existing RL research.** None. The 10-item state representation,
policies, and every benchmark artifact stay exactly as they are.

**Retraining required.** No.

**Does it validate the current system?** **Yes — no other option does, and it
does.** It tests the deployed instrument and the adaptive protocol as specified.

**Required governance.** E-3 (published instrument wording for both forms), plus
V-4 sign-off (E-2) and V-7 sign-off. A reviewed ordinal → binary conversion rule
must be approved before use, and must not be tuned on Polish.

**Honest weakness.** Q-CHAT-10 and Q-CHAT-25 are different instruments. Even a
correctly verified subset is a *derived* variable set, so the validation applies to
a reconstructed Q-CHAT-10 rather than to the administered form. And the
information content differs: the 10-item form is shorter by construction, so a
validated subset would partly measure instrument reduction.

---

## Option B — Build a separate 25-item ordinal predictor

**Scientific question answered.** How well does a predictor generalise across
*instruments* (Q-CHAT-10 → Q-CHAT-25) at the level of raw screening ability?

**Compatibility.** Total. `encode_state` already implements the `m_list`
ordinal path (199 dims), and `MaskedPredictor`, `DQNPolicy` and `PPOPolicy`
already accept `m_list`. What is missing is that `load_polish` never populates
it. This is a wiring gap, not an architectural one.

**Leakage risk.** **Material and must be managed.** A 25-item predictor must be
trained on *some* data. Polish cannot be both the training set and the external
validation set. The realistic options are: train on a third cohort (none
available), or train on Polish itself — which destroys the external-validation
claim entirely. Under spec §19.3's fold-local fitting rule, a Polish-trained
model then validates on nothing.

**Impact on existing RL research.** Significant. A 25-item model needs its own
RL environment, its own reward evaluation, and re-derivation of every benchmark.
Q-CHAT-25 at B=6 has **3,081,146,397** states (already verified in
`STATE_COUNT_VERIFICATION.md`), so exact DP is intractable and the exact reference
that anchors the whole benchmark would be unavailable for the new instrument.

**Retraining required.** Yes — predictor, calibrator, and (for comparability) all
RL policies.

**Does it validate the current system?** **No.** It validates a *different* system
on a different instrument. It answers a real and interesting question, but not the
one V-4/V-7 was raised to answer.

**Required governance.** V-4, V-7, E-1, E-2, E-3, plus a decision on whether a
new instrument enters the research programme at all — which is a scope question
for the supervisor, not an engineering one.

---

## Option C — Defer Polish external validation

**Scientific question answered.** None now. Preserves the cohort intact for a
later, properly designed evaluation.

**Compatibility.** Not applicable.

**Leakage risk.** Zero. The cohort remains sealed.

**Impact on existing RL research.** None.

**Retraining required.** No.

**Does it validate the current system?** No, not yet.

**Required governance.** None beyond noting the deferral, which is itself a
supervisor decision that should be recorded.

**Honest weakness.** The circularity problem in the development cohort is real and
unresolved: Saudi labels are a deterministic sum-threshold over the items
(`exact_match = 1.0000`). A clinician-labelled cohort is the one thing that can test whether the
screening system means anything beyond fitting the label rule. Deferring leaves
that question open indefinitely, and it is the strongest reason the project has
for acquiring this cohort at all.

---

## Comparison

| | A — verified 10-item subset | B — 25-item ordinal predictor | C — defer |
|---|---|---|---|
| Answers the V-4/V-7 question | **Yes** | No | No |
| Validates the *current* system | **Yes** | No | No |
| Frozen predictor used unchanged | **Yes** | No | n/a |
| Retraining | No | **Predictor + calibrator + RL** | No |
| Leakage risk | Low | **Material** | None |
| Impact on RL research | None | **Large** (new instrument, intractable DP) | None |
| Item mapping needed | Yes (**OPEN**, E-3) | No | n/a |
| State today | Blocked on E-1/E-2/E-3 | Blocked on a scope decision + same gates | Blocked on a deferral decision |
| Implementation cost | Low | **High** | None |

## Recommendation

**Option A**, conditional on E-3 being satisfied, because it is the sole option
that answers the question the gates were raised to ask, and the sole one that
validates the system actually built. Option B is a legitimate *future* programme
of work on a second instrument, but it should not be launched as a route to
external validation, because doing so either leaks the cohort or validates
something other than the specified system. Option C is honest but leaves the
project's central weakness unaddressed.

**This is a recommendation, not a decision.** The choice between "validate the
current system on a reconstructed subset" and "invest in the 25-item instrument
properly" is a research-priority judgement, and it belongs to the supervisor.

---

## What is unchanged regardless of the choice

* Saudi RL training, reward, state representation, policies, and every benchmark
  artifact — **untouched**.
* The Polish cohort remains **sealed** and contributes no fitted parameter.
* V-6 (question-cost threshold) — **untouched and still unsigned**.
* No ordinal value is truncated, thresholded, or coerced anywhere.
* `scripts/step10_external_validation.py` exits **2** and reports **no metrics**
  until the applicable requirements are closed.
