# Readiness for Diagnostic Use — Assessment and Roadmap

**Status:** assessment + roadmap. Nothing in this document has been implemented except where explicitly marked.
**Date:** 2026-10-01
**Scope note:** this document records an audit of the repository as it stood at commit `94b313e`, plus the findings that led to the implementation work described in §6.
**Terminology:** the system performs **screening** and **referral recommendation**. It is not a diagnostic tool. That is a deliberate, spec-enforced constraint (§1, §35, §27 of `Master-Project-Specification_FINAL.md`), not an oversight. See §1 for why it matters to this roadmap.

---

## 1. The central finding: the framing is wrong before the code is

The request that prompted this audit was "make the project ready for diagnosis." Before any engineering work, that goal needs to be split, because it conflates two projects with very different economics.

### (A) Make it a defensible research result

The spec's stated contribution (§35) is a comparison of learned and heuristic sequential policies against a **finite-sample empirical optimum** computed exactly for the same state representation, budget, costs and objective. Every piece of machinery for that is written. The RL half of it has never been run.

This is achievable, in weeks, and it is what is actually blocking.

### (B) Make it clinically deployable

This requires IRB approval, a prospective cohort with independent ground truth, clinician-in-loop oversight, and medical-device regulatory classification. The realistic horizon is **2–4 years**, and it requires a clinical partner and a funded ethics process — not a code backlog.

**The distinction that matters:** (A) is blocked by ~a few weeks of engineering. (B) is blocked by institutions, money, and time. Optimising for (B) while (A) is unbuilt is how projects stall: the hard external dependencies have long lead times, so the right move is to clear the cheap internal ones in parallel and start the long-lead external ones early.

**Consequence for the spec:** if the project genuinely becomes a diagnostic tool, its stated purpose changes. §27 and the terminology lock (§25) would need rewriting, and the no-claim rule would need revisiting. That is a legitimate direction, but it is a decision for the project owner, not something to drift into through deployment work.

---

## 2. Repository audit — what is actually there

Commit `94b313e`, clean working tree, 6 commits total, ~4.1k LOC of Python, 54 tests.

### Present and working
- **§8 scaffold** — data layer (6 loaders, synthetic fallback, Q-CHAT-10 binary map), three-state encoding (`4n+1`), episode loop with STOP/budget/missing semantics, `MaskedMLP[128,64]` with isotonic/Platt calibration, authoritative exact backward-induction DP, 8 policy classes, explain (trace/counterfactual/SHAP), eval (paired bootstrap, power/MDE, Holm-Bonferroni, subgroup), audits (circularity, leakage), ablation registry.
- **54 tests passing** — 26 §21 core + 10 DP-tractability invariants + 4 Step-3 artifact contracts + 3 PRISMA no-claim + 11 demo-behaviour regressions.
- **Step 2** — predictor trained, 48-cell DP tractability sweep on real Saudi data, all `status=optimal`. Largest run N=506, B=6 → 25,023 states / 68,408 evals / 2.2s, well inside the 50M / 24h / 32GB bound.
- **Step 3** — preliminary perf-vs-budget (B∈{1..6}), faithfulness, subgroup artifacts, all tagged preliminary.
- **Demos** — browser demo (`scripts/demo_app.py` + static UI) and terminal demo (`scripts/demo_live.py`).

### Datasets
| Cohort | Rows | Label source | Status |
|---|---|---|---|
| Saudi | 506 | questionnaire (circular) | real, verified, `Deterministic` circularity |
| UCI Child | 292 | questionnaire (circular) | real, verified, `Deterministic` circularity |
| Polish | 252 (135/117) | **clinical** — not circular | real, verified, **deliberately sealed** behind V-4/V-7 |
| NZ toddler | 1,054 | questionnaire (circular) | **blocked on V-1** — source located and content validated, but Kaggle licence is "Unknown"; never downloaded by the agent |

---

## 3. The RL half has never been trained

This is the most important finding in the audit, and it was not visible from the progress documentation.

`src/policies/dqn.py` and `src/policies/ppo.py` exist. Nothing trains them.

Evidence:

1. **No training calls exist.** A grep for `.train(` / `fit(` across `scripts/` and `src/` returns *predictor* `.fit()` calls and nothing else. No policy training call exists anywhere in the repository.
2. **`DQNPolicy.train_step` is never called** by any script, test, or demo.
3. **No replay buffer exists** anywhere in the repository — there is no `replay.py`, no buffer class, no `ReplayBuffer` symbol.
4. **`configs/config.yaml` declares `policy.type: dqn`**, which implies the configuration expects DQN to be the operative policy.
5. **Every actual run uses Greedy-IG or Random.** `step3_preliminary_reports.py` and `demo_app.py` import `GreedyIGPolicy` and `RandomPolicy`, and import no learned policy.

So RQ1, RQ2, ablation AB-4 (DQN vs PPO) and AB-5 (frozen vs jointly trained) have **never been produced once**. What runs in the demo is a heuristic information-gain policy, not reinforcement learning. The repo's name and premise are not currently exercised by its own results.

### 3.1 Defects in the untested RL code

Found by reading the sources. These have never been caught because the code has never executed.

**`src/policies/dqn.py` — `train_step`, dead legal-action mask.** The method builds a legal-action mask, writes it into a local `mask` variable, and then does nothing with it. The comment in the source (`skip for brevity — use max over legal only`) records that the masking was knowingly omitted. The actual target computation then runs `next_q.argmax(dim=1)` over **unmasked** Q-values, so illegal actions leak into the bootstrap target. For a policy whose entire purpose is to respect a question budget and a `B_min` floor, silently optimising toward illegal actions is a correctness defect, not a cosmetic one.

**`src/policies/dqn.py` — `train_step`, terminal-transition handling (investigated; NOT a bug).** The code builds `torch.zeros(1, D)` for terminal next-states and concatenates. This was flagged during the 2026-10-01 audit as a shape error that would raise on mixed batches. **That claim was wrong and is retracted:** `torch.cat(..., dim=0)` concatenates *along* dimension 0, and each row contributes exactly one `(1, D)` tensor, so the result is always `(batch, D)`. Empirically confirmed. No fix was required; the rewrite simplified the construction but did not correct a defect.

The substantive terminal-handling point that remains valid: terminal rows are zero-filled, so their Q-values are meaningless. That is harmless because `done == 1` zeroes the bootstrap term. The masking defect (#1) is what made the non-terminal rows wrong.

**`src/policies/ppo.py` — no training method exists at all.** The class instantiates an actor and a critic and an optimiser over both parameter sets. It has `__call__` (which selects greedily by argmax over legal logits) and nothing else. There is no `train_step`. **The critic never receives a gradient**, because no loss is ever constructed. `PPOPolicy` is an untrained actor-critic used as a stochastic-but-untrained policy.

**Note on `PPOPolicy.__call__`:** it computes a softmax over legal actions and then discards it, returning the argmax. The probability computation is dead code. This is not a bug so much as unfinished intent — the docstring calls it a stub.

### 3.2 Consequence for every reported number

Because no RL policy is trained, the entire empirical comparison rests on **Greedy-IG vs Random** — a heuristic against a uniform-random baseline. The gap between those two is not a measurement of what reinforcement learning contributes here. Any reading of the current Step-3 artifacts as an RL result is unsupported by the artifacts themselves.

---

## 4. The circularity problem

Independently of the RL gap, three of four cohorts are graded against a formula over their own inputs.

Saudi, UCI Child and the NZ toddler target all derive their labels from a **sum-threshold oracle** over the questionnaire items themselves (`Qchat-10-Score == sum(A)` verified 1,054/1,054 on NZ and 506/506 on Saudi). The circularity audit classifies all three as `Deterministic`.

This matters for the roadmap because it constrains what can be concluded, no matter how good the engineering is:

- The reported Saudi predictor AUROC of 0.9877 is substantially an artefact of the labelling rule, not evidence of clinical discrimination. The spec's §16.1 gate already flags this; the README's own troubleshooting section concedes it.
- A trained RL policy could **exploit the circularity perfectly** and that would tell us nothing whatsoever about autism. The reward signal is contaminated at the source.
- Therefore the Polish cohort — 252 rows with genuine clinician-established diagnoses — is the single cohort on which any clinical claim could be based. It is 252 rows, and it is sealed.

**The design tension:** Polish is simultaneously the single clean dataset and the single valid held-out transfer evaluation. Unsealing it for training would destroy the transfer test. The project's pre-registration discipline says keep it sealed; that decision is correct and should be preserved.

**The real constraint:** there is currently no dataset large enough to both train an adaptive policy and provide an independent clinical test of it. That is a data-collection problem, not an engineering problem, and it bounds what Parts 1–3 can honestly claim.

---

## 5. Reproducibility is currently broken

This checkout has **no `.venv`, no `data/raw/`, and no `results/`**. All three are gitignored:

```
.venv/     data/raw/     results/
```

Consequences:

- **Every number in `README.md`, `AGENT_PROGRESS.md` and the audit reports is currently unverifiable in this environment.** The 8 result artifacts the documentation cites no longer exist on disk.
- The progress file records a prior incident of exactly this: on 2026-09-30, `results/` was found empty and all 8 artifacts had to be regenerated from scratch.
- Raw CSVs must be re-fetched by a human from their original sources. The agent did not download the NZ file and must not; the other three are needed.

This must be fixed before any other work, because without it no change can be validated.

---

## 6. Implementation work completed (Parts 0 and 1)

Parts 0 and 1 of the §7 plan were implemented on 2026-10-01. See `RL_TRAINING_REPORT.md` for the full write-up, and `AGENT_PROGRESS.md` for the current state log.

**Part 0 — environment restored.** Virtualenv rebuilt, raw CSVs restored, all 8 Step-2/3 artifacts regenerated with `source: real`, test suite verified at 54 passed.

**Part 1 — RL code fixed and training infrastructure built.** The two genuine defects in §3.1 corrected, a replay buffer added, and `scripts/step4_train_policies.py` written to actually train DQN and PPO. (A third alleged defect was investigated and retracted — §3.1.)

**Part 2 (policy benchmark) is NOT done.** DQN and PPO have been trained but have **not** been evaluated at matched budgets against Greedy-IG, Random and the exact DP. The RL claim remains unevidenced until that runs.

---

## 7. Roadmap

### Part 0 — Restore reproducibility ✅ complete
- Rebuild venv from `requirements.txt`
- Restore Saudi 506 / Polish 252 / UCI Child 292 CSVs to `data/raw/`
- Re-run Step 2 and Step 3; confirm all 8 artifacts carry `source: real`
- Confirm 54 tests pass
- **Done 2026-10-01.**

### Part 1 — Fix the RL code that has never run ✅ complete
- Correct `DQNPolicy.train_step`: apply the legal-action mask to the bootstrap target; remove the dead masking block
- Correct `PPOPolicy`: add `train_step` (clipped surrogate + value loss + entropy bonus) — it had none, so its critic never trained
- ~~Fix terminal-transition tensor shapes~~ — **retracted, see §3.1**
- Add `src/policies/replay.py`: transition buffer preserving per-transition legal-action sets (required — the mask must survive storage, otherwise targets are computed on illegal actions)
- Add `step4_train_policies.py`: episode collection against the §10 environment, buffer fill, update loop, seeding from `config.eval.seeds`
- Tests for each: DQN illegal-mask correctness, terminal-target no-bootstrap, PPO critic gradient, replay-buffer stitching
- **Done 2026-10-01.**

### Part 2 — Put the trained policies in the benchmark ⬜ not started
Extend the Step-3 comparison to include DQN, PPO and `ExactDP` at matched budget (B ∈ {1..6} plus terminal) on the same Saudi test split and seeds.

Recommended as a **new script** (`step5_policy_benchmark.py`) rather than an edit to `step3_preliminary_reports.py`, so existing artifacts stay byte-identical and the baseline remains comparable across the doc history.

**Mandatory caveat, to be carried in-band in every artifact:** DQN and PPO will be trained and evaluated against circular questionnaire labels. A policy can learn the circularity perfectly and learn nothing about autism. Part 2 produces the methodological result the spec asks for — learned vs heuristic vs exact-optimum at matched budget — and does **not** produce clinical evidence.

### Part 3 — Reconcile the documentation ⬜ partially done
- `AGENT_PROGRESS.md`: 43 → 54 tests; add demo work from `56a1e3d`; correct "working tree modified"; record Part 1
- `README.md`: remove duplicated "Covers …" paragraph; add RL artifacts to the reports table
- `RL_TRAINING_REPORT.md`: Part 1 defects, Part 2 results when available, explicit circularity caveat
- **Status: RL-specific docs written. Baseline doc drift (§3, §7 test counts and demo work) still outstanding.**

### Part 4 — Clinical readiness gap analysis ⬜ documented, not built
Documented here, not built. In rough dependency order:

| Gap | Why it blocks |
|---|---|
| **Clinical ground truth** | A mere 252 rows exist with clinician diagnoses. A diagnostic tool needs an independent cohort measured against ADOS-2 / ADI-R. Circular labels cannot serve. |
| **Prospective validation** | Everything is retrospective on public data. Diagnostic claims need prospective, pre-registered evaluation on a new cohort. |
| **IRB / ethics (V-9)** | Not started. Non-negotiable before any human data collection. |
| **Clinician-in-loop** | Listed as a limitation in §27. No clinician oversight, no override path, no escalation route. |
| **Calibration under shift** | Isotonic is fitted on Saudi. No evidence it transfers across population, language, or clinical setting. |
| **Fairness / subgroup floors** | Subgroup analysis exists but is underpowered (n=41, n=86) and marked as such. No minimum-performance floor is defined. |
| **Regulatory classification** | Likely software-as-a-medical-device in most jurisdictions. Classification, QMS, and post-market monitoring are organisational work. |
| **Terminology / spec revision** | Becoming a diagnostic tool changes §1, §25 and §27. Owner decision. |
| **Prospective data scale** | Per §4, no existing cohort is large enough to both train a policy and independently test it. |

**Horizon: 2–4 years, external dependencies, not a code backlog.**

---

## 8. Recommendation

**Do not pursue diagnostic readiness as the next milestone.** The repo has 6 commits, all from a single agent, no CI configuration, documentation that drifts from reality, and an RL half that has never executed. Building clinical claims on that foundation would put the credibility of the eventual result at risk regardless of how much clinical effort followed.

**Clear the cheap internal dependencies first:** no — begin with the cheap internal dependencies. Parts 0–2 make the project's actual premise testable and are a matter of weeks. **Start the long-lead external dependencies in parallel** — IRB and a clinical partner have multi-year lead times and cost nothing to begin while the engineering proceeds.

The honest summary: Part 2 will finally answer the question the whole project was designed around, and it will do so on contaminated labels. That answer is worth having. A diagnostic claim would need §7 Part 4, and no amount of Part 1 code substitutes for it.

---

## 9. Open decisions for the project owner

1. **Do Parts 2 and 3 next?** Recommended order: Part 2 (benchmark the now-trained policies), then Part 3 (doc reconciliation).
2. **Keep Polish sealed?** Recommended: yes. It is the single clean transfer evaluation and destroying it for training would be irreversible.
3. **Start the IRB conversation now?** Longest lead time of any remaining item. Costs nothing to initiate.
4. **Is diagnostic use actually the goal?** If yes, §1 above changes the project's stated purpose and the spec needs revising before — not after — clinical work begins.
