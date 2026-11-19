# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

NeuroBridge is the owner's college research project. It is built to serve two real-world audiences and the people who evaluate the research:

- **Doctors / clinicians.** They run or review an autism screening with a family, read the specialist-referral recommendation and check *why* the system asked what it asked (decision trace, explanations).
- **Parents / caregivers.** They answer screening questions about their own child, see what the result does and does not mean, record everyday observations and get non-diagnostic support ideas.
- **Examiners, supervisors and researchers.** They judge the research: the RL policy compared with heuristic and exact-optimum baselines, the metrics, and the documented limitations (RL Intelligence Lab, Explainable AI, Research & Analytics views).

All three audiences are confirmed in scope. No audience is subordinate. Each surface should know which of them it mainly serves.

## Product Purpose

An adaptive, explainable autism **screening and specialist-referral recommendation** tool. It is **not a diagnostic tool**. Every screening output is labelled as screening.

Given a child's Q-CHAT answers revealed one at a time, a policy picks the next question and when to stop, then issues a referral recommendation with a calibrated risk score and a complete decision trace. Around that core sit contextual observations, observation-grounded support planning, a timeline, and privacy/consent controls.

Success means:
- a parent can finish a screening and understand its result without being frightened or falsely reassured;
- a clinician can trust the result because they can inspect it;
- an examiner can verify the research claims and see their limits.

## Positioning

The owner confirmed all four of these as the product's claim:

1. **A transparent adaptive RL questionnaire.** Every question choice and stopping decision is explained and benchmarked against a finite-sample exact optimum (custom dynamic program) for the same state, budget and actions.
2. **Fewer questions at kept quality**, reported against clinician-established labels (Polish cohort) rather than questionnaire-derived labels alone.
3. **Screening and support in one place.** Support ideas are generated from the screening answers and logged observations themselves.
4. **An honest research instrument.** Circular labels, open verification gates and blocked external validation are shown, never hidden.

Closest prior art: CAT-Autism (Tseng et al. 2026, MIRT-based adaptive testing) and compact Q-CHAT-10 subsets (Sollis, Wall & Washington 2025). Both are static or IRT-based. Neither uses a learned sequential policy with an exact-optimum reference and label-source-shift analysis.

## Operating Context

- Runs locally as a Flask app with SQLite (`data/neurobridge.db`, gitignored). No external services or keys.
- Main workflows: guided screening (start/resume, answer, adaptive skip/stop, result), Context Explorer (observations), Support Studio, Timeline, RL Intelligence Lab (replay/step), Explainable AI, Research & Analytics, Privacy & Consent.
- Used in a college research setting: live demos to examiners, plus use by or with doctors and parents.
- Two web UIs exist: the legacy vanilla-JS SPA at `/` (pinned by tests) and the React + TypeScript "SOFT / SYSTEM" UI at `/ui/` (`frontend/`). Whether `/ui/` replaces `/` is **undecided**.

## Capabilities and Constraints

- **Never diagnostic.** Outputs are screening/referral estimates. Support ideas are non-diagnostic.
- **Data sensitivity.** Free-text notes about a child are stored locally and must never be committed or logged. Consent is recorded, and privacy export and delete endpoints exist.
- **Server CSP**: `script-src 'self'`, `style-src 'self'`, `font-src 'self'`. No inline script or style, no remote fonts.
- Mutating API requests must be `application/json`.
- Instrument: Q-CHAT-10 (training) and Q-CHAT-25 (Polish clinical evaluation). Primary labels are questionnaire-derived, and the Polish cohort is the clinician-labelled one.
- Open research facts, which future work must report as open and never as settled: V-2 novelty claim (wording such as "first" is forbidden in docs until V-2 completes), external validation status, consent enforcement.
- Terminology: "screening", "referral recommendation", "risk score", "decision trace", "support ideas". Avoid "diagnosis" and "detects autism" for outputs.

## Brand Commitments

- Name: **NeuroBridge**. Page title: "NeuroBridge — Adaptive Screening & Support (Research Prototype)".
- "Research prototype" and "not a diagnostic tool" labelling is binding wherever results appear.

## Evidence on Hand

- Research artifacts in `results/` (gitignored, regenerable): policy benchmarks, λ sweep, RL diagnosis, EVOI analysis, gate evidence. See `README.md`, `RL_ADAPTIVE_REPORT.md` and `Master-Project-Specification_FINAL.md`.
- Datasets: Saudi Q-CHAT-10 (506), Polish Q-CHAT-25 (about 252, clinician-established, sealed).
- **Absent, and must not be fabricated:** clinician-in-the-loop validation of the interface, participatory design with parents, user testimonials, clinical deployment or approval claims.

## Product Principles

1. **Screening, never diagnosis.** Every result says what it cannot conclude.
2. **Show the reasoning.** Each question asked and each stop is explainable to a doctor and checkable by an examiner.
3. **Honesty over impressiveness.** Limitations, circular labels and blocked validation sit next to the numbers they qualify.
4. **Speak to the reader in front of you.** Parent-facing screens stay plain and calm. Clinician- and research-facing screens can be dense and precise.
5. **The child's data stays local and private.**

## Accessibility & Inclusion

Parents of children being screened may be anxious, non-technical or non-native English readers, so plain language and calm, non-alarming result framing are required. The UI should be keyboard-operable with visible focus and meet WCAG AA contrast in light and dark themes, which the React UI already targets. Rendered verification of this is still pending.
