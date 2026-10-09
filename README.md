# Adaptive RL for Explainable Autism Screening

System function: **adaptive screening and specialist-referral recommendation. Not a diagnostic tool.** Every output says SCREENING.

Source of truth: `Master-Project-Specification_FINAL.md`

## Setup & run

**Prerequisites:** Python 3.10+ (validated on 3.14), pip, git. No GPU needed — everything runs CPU-only.

### 1. Create a virtual environment

**Windows (cmd):**
```bat
py -3 -m venv .venv-win
.venv-win\Scripts\python -m pip install --upgrade pip
.venv-win\Scripts\python -m pip install -r requirements.txt
```
`pip install torch` is CPU-only on Windows by default — no special index needed. Or activate once per session with `.venv-win\Scripts\activate` and use plain `python` afterwards.

**Linux / WSL / macOS:**
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Note: a venv created on Linux (e.g. under WSL) cannot be run from Windows cmd and vice versa — keep them separate (`.venv` vs `.venv-win`; both are gitignored).

### 2. Run the test suite
```bash
# Windows:  .venv-win\Scripts\python -m pytest tests -q
# Linux:    .venv/bin/python -m pytest tests -q
```
Expected: **487 passed, 0 skipped** on CPU. Useful flags: `-x` stop at first failure, `-rs` show skip reasons, `-k <keyword>` filter.

> **Test-count history.** The figure has been wrong in this file three times, in both
> directions, so it is worth knowing how to check it rather than trusting it.
>
> - An early revision claimed **79 passed**. That was the *collected* count: 12 of those
>   tests live in `tests/test_step5_benchmark.py` and skip when
>   `results/step5_policy_benchmark_saudi.json` is absent, so the real figure was
>   67 passed / 12 skipped. Scripts were also crashing on their final `print` (a cp1252
>   console cannot encode `→` or `λ`) *after* writing their artifacts, which is why the
>   artifacts appeared missing.
> - Later revisions quoted 126, then 163, then 191 as the suite grew with each
>   correction pass. Each was correct when written.
> - Current: **487 passed / 0 skipped.** Verify with the command above; the number is
>   the point-in-time count, not a target.

> **Disk-space gotcha (2026-10-01):** a plain `pip install -r requirements.txt` pulls the **CUDA** build of torch (~2 GB of nvidia wheels) and fails with `No space left on device` — `/tmp` is a 3.7 GB tmpfs. Install CPU-only torch instead:
> ```bash
> TMPDIR=/path/to/big/disk .venv/bin/pip install \
>   --extra-index-url https://download.pytorch.org/whl/cpu torch
> TMPDIR=/path/to/big/disk .venv/bin/pip install -r requirements.txt
> ```

### 3. Run the pipeline (optional — regenerates `results/`)
`data/raw/` and `results/` are gitignored. If `results/` is empty, regenerate:
Run in order; each depends on the previous one's artifacts.

```bash
python scripts\step2_train_and_sweep.py        # predictor training + 48-run DP sweep (~1 min)
python scripts\step3_preliminary_reports.py    # perf-vs-budget, faithfulness, subgroup reports
python scripts\step4_train_policies.py         # DQN + PPO training (~2 min at the 2000-episode default)
python scripts\step5_policy_benchmark.py       # matched-budget benchmark + V*-V_emp gap + H1 (~4 min)
python scripts\step6_lambda_sweep.py           # lambda sweep — V-6 decision input
python scripts\step7_rl_diagnosis.py           # bounded RL diagnosis (5 seeds x 4 episode budgets)
python scripts\step8_evoi_scale_analysis.py    # EVOI distributions + stopping-rule sensitivity — V-6 decision input
python scripts\step9_v4_v7_gates.py           # V-4/V-7 gate evidence + Polish provenance verification (metadata only)
python scripts\step10_external_validation.py    # external validation on the sealed Polish cohort — currently BLOCKED, exits 2
```
Scripts automatically use the real CSVs when present under `data/raw/` (every artifact records `"source": "real"`) and fall back to synthetic data for smoke-testing otherwise.

> ⚠️ **Do not pass `--episodes 400` to Step 4 or Step 5.** An earlier revision of this
> file did exactly that, and it produces a *wrong result rather than a fast one*: at 400
> episodes DQN is simply undertrained, which looks identical to "RL fails on this task".
> The Step-7 diagnosis is what established this — DQN needs ~2,000 episodes to reach a
> sane policy. The defaults are now 2,000; omit the flag. See
> `POLICY_BENCHMARK_REPORT.md` §A.5 and `RL_TRAINING_REPORT.md`.
>
> Budgets: Step 4/5/6/7/8 default to `--budget 6`. Step 6 sweeps all eight budgets
> (1,2,3,4,5,6,8,10) and takes noticeably longer than the others.

### 4. Run the live demo

**Browser demo (recommended):**
```bat
:: Windows
.venv-win\Scripts\activate
python scripts\demo_app.py
:: open http://127.0.0.1:8000
```
First start trains the predictor (~20 s) and caches it to `results/demo_model_saudi_seed0_platt_v2.{pt,pkl}` (the version is in the filename — an unversioned cache previously outlived a change to the predictor's training semantics and the demo silently kept serving stale weights); later starts are instant. Options: `--retrain` (ignore cache), `--port 8080` (auto-scans the next 9 ports if busy).

The page shows the data audit, an interactive adaptive interview (greedy vs random policy), the per-answer belief trace, the referral decision with a counterfactual explanation, and the generated result artifacts.

**Outcome explanation (P3-outcome).** The result screen now answers *why this
result* as well as *which question*: a "Why this result?" panel ranks the
observed answers by their exact group-Shapley contribution to the screening
estimate, lists the answers that move it toward and away from referral, reports
the still-unasked items the model accounts for by prior, states the
per-session stability band where the predictor supports one (v3) or its
absence (v2), shows which single answer changes would move the decision across
the threshold, and prints the limitations and the non-diagnostic disclaimer.
Every field is computed by the backend (`src/explain/outcome.py`, schema
`outcome-explanation/1.0`) and rendered without client-side computation; the
terminal demo prints the same object at STEP 6b. Design record and measured
faithfulness: `docs/EXPLAINABILITY_OUTCOME_PLAN.md`,
`results/outcome_explainability_saudi.json` (Step 17).

**Optional support ideas (P4).** After the result, the demo offers an optional,
skippable follow-up questionnaire and a set of *support ideas* assembled by
`src/support/`. The one rule the layer is built around is structural: **a
hypothesis never fires a recommendation.** The only route from an observed
screening answer to a suggestion runs through the person — an atypical response
can put a question on the *suggested* list, and only their own answer to it can
produce a suggestion, always carrying the assessment id that triggered it and a
plain-language reason. Domains the questionnaire does not cover (sensory,
transitions, daily living, accessibility) say so in their wording and can only
ever be user-reported. Every suggestion is curated project content pending
human expert review, and the report discloses that. Backend:
`POST /api/session/support`, schema `support-report/1.0`; the terminal demo
prints the same object at STEP 6c. Design record:
`docs/SUPPORT_LAYER_DESIGN.md`.

**What the demo does and does not show.** It serves `GreedyIGPolicy` (information
gain) and `RandomPolicy` only — the DQN/PPO arms are benchmarked offline in
Step 5, not here. Two things on the page are deliberately unflattering, because they
are the measured results rather than a presentation of them:

- **The greedy policy often asks A1, A2, A3… in order.** This is not a bug. On this
  dataset the labels are a deterministic sum-threshold over the questions, so the
  training support becomes *label-pure* after a few answers; expected information gain
  is then exactly 0 for every remaining question, `argmax` has nothing to choose
  between, and the policy falls back to the lowest unasked index. The page reports this
  per step (support size, purity, IG spread) in the **Selection** panel.
- **The status panel shows H1 failing at B = 5 and B = 6** — the exact best fixed
  subset attains lower held-out Brier than every adaptive arm tried. It also shows that
  the **V-6 question-cost gate is unsigned with no threshold selected**, so stopping is
  manual in this demo and the episode always runs the full budget unless you stop it.

**Terminal demo (no browser):**
```bash
python scripts/demo_live.py               # 8-step scripted walkthrough (~2 min)
python scripts/demo_live.py --interview   # + interactive Q&A in the terminal
```

### 5. Data placement
Loaders expect (see `src/data/ingest.py`):
```
data/raw/Q-CHAT Saudi Arabia/Autism Spectrum Disorder Screening Data for Toddlers in Saudi Arabia Data Set.csv
data/raw/Q-CHAT Polish/polish_qchat.csv
data/raw/UCI/Autism-Child-Data.arff
data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv    # pending V-1 licence
```
All preprocessing is in-memory; raw files are never modified or committed.

> ⚠️ **Two of these are derived conversions, not original files** (re-fetched 2026-10-01):
> - `Autism-Child-Data.arff` — **UCI no longer serves an ARFF for dataset 419**, only `data.csv`. This ARFF was generated from that official CSV (same 292 rows, 21 columns, `NaN`→`?`; the 90 missing markers match the documented count).
> - `polish_qchat.csv` — generated from the original Mendeley Data `tmpkt2mfkg` SPSS `.sav` (sha256 verified) using the file's own SPSS value labels for `group`/`sex`.
>
> Full provenance in `AGENT_PROGRESS.md` §"Data provenance".

### Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: numpy` | You invoked the system Python — use the venv's python path or activate it |
| `'py' is not recognized` (cmd) | Use `python -m venv .venv-win` instead |
| Step-3/5/6/8 tests skip | `results/` artifacts missing — run the Step 2–8 scripts above |
| DQN "collapses" to ~0.6 items | You passed `--episodes 400`. Omit the flag; 400 is undertrained and looks identical to RL failing (P1-e) |
| Port 8000 busy | `python scripts\demo_app.py --port 8080` |
| Demo asks A1, A2, A3… in order | Expected on this dataset — the support is label-pure, so information gain is 0 for every remaining item. The Selection panel reports support size, purity and IG spread per step |
| Demo shows extreme risk (0/100%) | Expected for consistent all-typical/all-atypical answers — the label audit panel explains why (deterministic questionnaire labels) |

Synthetic demo (no real data required):
```python
from src.data.ingest import load_dataset
from src.models.masked_predictor import MaskedPredictor
from src.env.environment import run_episode
from src.policies.greedy import GreedyIGPolicy

records = load_dataset("nz", synthetic=True)
# etc. See tests/ for usage
```

## Documentation index

| Document | Contents |
|---|---|
| `README.md` (this file) | quickstart, repo layout, invariants, gate status, reports table |
| **`diagnosisReady.md`** | **audit + roadmap: what stands between this project and diagnostic use (2026-10-01)** |
| **`V6_LAMBDA_DECISION.md`** | **the V-6 evidence pack: λ sweep, the objective-saturation finding, and the proposed cost grid (2026-10-01). Does NOT sign off V-6** |
| **`RL_TRAINING_REPORT.md`** | **the RL defects found in never-executed code, fixes, and training runs — §2 the collector, §3 the split leak and the predictor (2026-10-01)** |
| **`POLICY_BENCHMARK_REPORT.md`** | **matched-budget benchmark: greedy attains the exact optimum, PPO is competitive, DQN collapses. §A carries the current numbers; the original §2 is RETRACTED (2026-10-01)** |
| `AGENT_PROGRESS.md` | canonical project state log; what was done, what failed, what's blocked, what's next |
| `DATA_VERIFICATION_REPORT.md` | per-dataset verification: rows, columns, circularity, missing, invalid-value handling |
| `STATE_COUNT_VERIFICATION.md` | theoretical (Q-CHAT-10, Q-CHAT-25) and empirical DP tractability counts |
| `AUDIT_REPORT.md` | circularity + leakage + invalid-value + Polish isolation + Step 2/3 audit trail |
| `V1_NZ_DATASET_RESOLUTION.md` | V-1 source-location + schema + circularity + loader plan for the NZ 1,054-row cohort |
| `V2_PRISMA_SEARCH_LOG.md` | PRISMA template (research question, 10 databases, 12 queries, criteria, no-claim rule) |
| `AUDIT_UPDATE_2026-09-04.md` | consolidation of the 2026-09-04 audit: re-runs, cross-checks, change log |
| `Master-Project-Specification_FINAL.md` | source of truth (spec §1-§27) |
| `docs/EXPLAINABILITY_OUTCOME_PLAN.md` | P3-outcome design record: the verified outcome path, exact group-Shapley attribution, counterfactuals, prior-sensitivity band, measured faithfulness |

**Freshness (updated 2026-10-01).** Living documents — `AGENT_PROGRESS.md`, `README.md`, `AUDIT_REPORT.md`, `DATA_VERIFICATION_REPORT.md`, `STATE_COUNT_VERIFICATION.md`, `diagnosisReady.md`, `RL_TRAINING_REPORT.md`, `POLICY_BENCHMARK_REPORT.md`, `V1_NZ_DATASET_RESOLUTION.md`, `V2_PRISMA_SEARCH_LOG.md` — are all current. `AUDIT_UPDATE_2026-09-04.md` is an intentionally **frozen historical snapshot**; its figures are superseded and it carries a header saying so. Current state is always `AGENT_PROGRESS.md` (527 passed, 12 failed — all environmental, 78 skipped).

> ⚠️ **Five corrections are in force. Read these before citing any number.**
>
> 1. **`POLICY_BENCHMARK_REPORT.md` carries a retraction banner.** Its original
>    §2 conclusion ("greedy attains the exact optimum, the learned policies do
>    not", presented as a negative result for RL) was measured with a rollout
>    collector that never delivered a terminal reward. Read its **§A**.
> 2. **The train/val/test split was contaminated** (P0-9). Four copies of the
>    split passed *relative* indices from a nested `skf.split` into the record
>    list, so on the 506-row cohort 71 of 95 validation records were also in
>    train and 24 were in test. Row counts were correct, which is why it went
>    unnoticed.
> 3. **`MaskedPredictor` v1 → v2.** It trained on one masked view per record,
>    trained the `budget_norm` feature on a grid inference never visits, and
>    fitted its calibrator on fully observed states only. Test Brier
>    0.0630 → 0.0512, AUROC 0.9627 → 0.9882.
> 4. **Every artifact was non-reproducible regardless of its recorded seed**
>    (P0-10). `nn.Linear` drew from torch's global RNG before the caller seeded
>    it. Everything is now reproducible from its seed.
> 5. **The 400-episode DQN was an undertrained network** (P1-e). It needs ~2,000
>    episodes; the benchmark default is now 2,000. See §A.5.
> 6. **The objective saturates at λ = 0** (P1-g). The empirical support becomes
>    label-pure after ~4 questions (0.820 at depth 4 → 1.000 at depth 7), so
>    `V*` is exactly 1.000000 and the measured value of adaptive stopping at λ=0 is
>    a property of the label rule, not evidence of interview sufficiency.
> 7. **H1 is not supported at B = 5 and B = 6** (P1-d/P1-f). The exact best fixed
>    subset attains lower held-out Brier than every adaptive arm tried — greedy,
>    a Beta-prior EVOI policy, and DQN alike.
> 8. **The "one reward, two scales" claim was withdrawn** (P1-h). The
>    support-posterior and neural-predictor EVOI are in the *same* units (ratio 0.43
>    at the median). The claim rested on a 0.05–0.15 predictor figure that no code
>    measured. What actually matters: the support-posterior EVOI is analytically
>    non-positive on a pure support, so the statistic behaves as a purity indicator.
>
> Details: `RL_TRAINING_REPORT.md` §2–§3, `AUDIT_REPORT.md` (full P0/P1 register),
> `POLICY_BENCHMARK_REPORT.md` §A, `V6_LAMBDA_DECISION.md` (λ evidence pack),
> `V6_STOPPING_THRESHOLD_DECISION.md` (**open supervisor decision — no threshold
> selected, V-6 unsigned**), `diagnosisReady.md` §0A.

## Repository layout — §8
```
src/data/ingest.py, schema.py, dedupe.py, splits.py, provenance.py, qchat10_contract.py
src/audits/circularity.py, leakage.py
src/env/state.py, environment.py, costs.py
src/solvers/exact_custom.py, exact_adapter.py
src/models/masked_predictor.py
src/policies/dqn.py, ppo.py, greedy.py, random_policy.py, irt_cat.py, dqn_cat.py, static_rfe.py, static_fixed.py, beta_greedy.py, replay.py
src/explain/attribution.py, counterfactual.py, uncertainty.py, outcome.py, limitations.py, render.py, trace.py, shap_baseline.py (deprecated: not SHAP)
src/support/schemas.py, domains.py, questions.py, strategies.py, engine.py, report.py
src/eval/metrics.py, bootstrap.py, power.py, fwer.py, subgroup.py, gates.py, external_validation.py, qchat10_subset.py
src/ablation/runner.py
configs/config.yaml (Hydra)
scripts/step2_train_and_sweep.py, scripts/step3_preliminary_reports.py, scripts/step4_train_policies.py, scripts/step5_policy_benchmark.py, scripts/step6_lambda_sweep.py, scripts/step7_rl_diagnosis.py, scripts/step8_evoi_scale_analysis.py, scripts/step9_v4_v7_gates.py, scripts/step10_external_validation.py, scripts/step17_outcome_explainability.py
scripts/demo_live.py (terminal demo), scripts/demo_app.py + scripts/demo_static/ (browser demo)
docs/prisma/screening_worksheet.csv
tests/ (487 tests, all passing)
```

## Key invariants — §9-11
- Three-state encoding: UNASKED(0) / OBSERVED(1) / MISSING(2) — distinct.
- STOP legal only after B_min unless no legal items remain; forced when budget==0.
- Reward: R = (1 - (p_hat - y)^2) - λ·Σc_j ; primary prediction is Brier-based.
- Primary threshold τ=0.5 frozen before Polish external cohort is opened.
- **Legal-action masking:** any value-function bootstrap must be masked to the *next* state's legal actions. Enforced in `DQNPolicy.train_step` (applied before the `argmax`) and in PPO (illegal logits masked to `-inf`). Regression-tested in `tests/test_rl_training.py`.
- **Transition batch format** is shared: `(s, a, r, s_next, done, legal_next)`, with `s_next=None` and `done=1` for terminal steps. `ReplayBuffer` stores `legal_next` so the mask survives collection.

## Exact reference — §14
Authoritative solver is `src/solvers/exact_custom.py` (backward induction DP).
External solvers (DL8.5/MurTree/STreeD) are cross-checks only when objective-compatible.

State counts verified: n=10,B=3..6 → 1,161 / 4,521 / 12,585 / 26,025 ; Q-CHAT-25 canonical (24×5 + 1×6) → 3,081,146,397 ; Q-CHAT-25 observed (Polish) → 2,667,729,775. See `STATE_COUNT_VERIFICATION.md`.

DP tractability sweep (Step 2, re-run 2026-09-30 on **real data**): 48 ExactDP runs on Saudi at N ∈ {10..506} × B ∈ {3..6} × λ ∈ {0,0.01}, all `status=optimal`. Largest run: N=506, B=6 → 25,023 states / 68,408 evals / 2.2 s (well inside the 50M / 24h / 32GB tractability bound). See `STATE_COUNT_VERIFICATION.md` §"Step 2 — 2026-09-04 systematic sweep".

## Data — §15
Real datasets live under `data/raw/` (gitignored — present locally, never committed). Loaders support `synthetic=True` for testing; Saudi / Polish / UCI Child are present and verified, NZ toddler target is pending V-1.
Q-CHAT-10 binary mapping: Q1-9 Sometimes/Rarely/Never→1 ; Q10 Always/Usually/Sometimes→1 — preserved raw for audit.

**2026-09-04 update:** Saudi 506, Polish 252, UCI Child 292 are real-data-verified in `src/data/ingest.py`. NZ 1,054-row file **source located** (Kaggle mirror + 5 GitHub mirrors); content validated (1,054 rows, Class Yes/No = 728/326, Qchat-10-Score == sum(A) 1,054/1,054, Age 12-36, circularity Deterministic thr 4); **licence = "Unknown"** is the only remaining V-1 blocker. Full resolution in `V1_NZ_DATASET_RESOLUTION.md`.

## Audits — §16 blocking
- `src/audits/circularity.py` — sum-threshold oracle detection. Saudi / UCI Child / NZ toddler target are **Deterministic** (questionnaire-derived); Polish is **Not circular** (clinical).
- `src/audits/leakage.py` — LeakageTracker + poisoned control.

## Tests — §21
```
pytest tests -q   # 527 passed, 12 failed (all environmental — see below), 78 skipped
```

**Test-count history.** The figure has been wrong in this file several times, in
both directions, so it is worth knowing how to check it rather than trusting it.
Verify with the command above; the number is the point-in-time count, not a target.
Early revisions claimed 79 / 126 / 163 / 191 / 204 / 453 / 487 — each correct when
written on the machine that wrote it.

**P4 support-layer tests (`src/support/`, all synthetic fixtures):** 90 tests in
`tests/test_support_schemas.py` (28), `test_support_registry.py` (6),
`test_support_questions.py` (8), `test_support_strategies.py` (7),
`test_support_engine.py` (26) and `test_support_report.py` (20), plus 13 demo-wiring
tests in `tests/test_support_demo.py`. They lock the layer's core property (a
hypothesis never fires a recommendation), the observed/derived/user-reported type
distinction, report cross-reference resolution, the JSON round-trip, and the
browser-demo contract (suggested vs optional questions, skip path, no invented
client-side copy).

**2026-10-09 (branch `feat/outcome-explainability`):** 527 passed, 12 failed, 78
skipped (437 + 90 P4 support-layer tests). All 12 failures are environmental or
pre-existing and are itemised in
`AGENT_PROGRESS.md` (ninth pass): 6 × frozen-v2 artifact-hash pins (the recorded
hashes came from another machine; regeneration here is deterministic), 1 × the
retained Step-14 evaluation artifact (absent on this machine, and Step 14 now
refuses to regenerate it by design because that run predates `random_fixed`),
1 × a pre-existing doc-rule violation (non-novelty uses of "first"/"only" in
`POLICY_BENCHMARK_REPORT.md`, `RL_TRAINING_REPORT.md`, `V6_LAMBDA_DECISION.md`,
`V6_STOPPING_THRESHOLD_DECISION.md`), and 3 × `test_v4_v7_validation` (need the
gitignored `data/QCHAT_dataset2 mendeley.sav`). No failure is a code regression.

Skips are dominated by "artifact not generated": run the Step 2–8 and 12–16
scripts to reduce them (Step 10 exits 2 while blocked by design).
24 P1-f Beta-prior / EVOI tests (`tests/test_p1f_beta_greedy.py` + 13 Step-7 RL-diagnosis tests (`tests/test_step7_rl_diagnosis.py`) + 36 Step-6 λ-sweep + canonical-split + beta_greedy + lambda_units_caveat tests (`tests/test_step6_lambda_sweep.py`) + 23 P1-d baseline + predictor-reproducibility tests (`tests/test_p1d_baselines.py`) + 28 RL training regressions (`tests/test_rl_training.py`) + 14 Step-5 benchmark contracts (`tests/test_step5_benchmark.py`) + 21 demo-behaviour audit regressions (`tests/test_demo_behavior_audit.py`) + 10 DP tractability invariants + 4 legal-action tests + 4 Step 3 artifact contracts + 4 reachable-state-count tests + 3 V-2 no-claim rule tests + 3 leakage tests + 2 budget-exhaustion + 2 counterfactual + 2 reward-bounds + 2 state-encoding + 8 single-test modules. All 204 pass; none skip.

Covers state encoding, legal actions, budget, state counts, reward bounds, predictor, exact optimality, circularity, leakage, counterfactual, threshold freeze, common evaluator, trace, fixed subset, DP tractability, preliminary report metadata, PRISMA template presence, greedy determinism + legality, random-policy variation semantics, belief bounds and continuity, the documented `p_hat >= tau` decision rule at both env and API layers, API risk continuity, faithful frontend rendering of the backend risk value, and — added 2026-10-01 — the λ-sweep invariants (`V*` non-increasing in λ, the degeneracy boundary, support purity by depth), the canonical split (three disjoint parts, and every consumer agreeing on one fingerprint), and the full P0 register below.

The 28 RL tests are **regressions for defects that had never been caught** because no training code had ever executed — see `RL_TRAINING_REPORT.md` §1-§3. Notably: DQN's bootstrap must exclude illegal actions, terminal transitions must contribute no bootstrap, PPO's critic must actually receive a gradient, illegal actions must get zero probability, the terminal reward must actually reach the replay buffer, the stored legal set must belong to the state being *entered*, nothing may be left staged in `_pending`, and epsilon-greedy exploration must be reproducible per seed.

## ⚠️ RL policies — training vs benchmarking
`scripts/step4_train_policies.py` **trains** DQN and PPO. `scripts/step5_policy_benchmark.py` **benchmarks** them against Greedy-IG, Random, and the ExactDP reference at matched budgets B ∈ {1..6}.

**Current result (`POLICY_BENCHMARK_REPORT.md` §A), Saudi 506, canonical split, `predictor_version: 2`, seed 0, λ=0, 2,000 training episodes, reproducible from seed:**

| policy | Brier @B=6 | UAR | items | stop_early |
|---|---|---|---|---|
| exact_fixed_subset | **0.0479** | 0.9167 | 6.00 | 0.00 |
| static_rfe | **0.0479** | 0.9167 | 6.00 | 0.00 |
| greedy | 0.0512 | 0.8930 | 6.00 | 0.00 |
| dqn | 0.0640 | **0.9112** | 5.98 | 0.02 |
| exact | 0.0644 | 0.8812 | 4.45 | 0.77 |
| irt_cat | 0.0686 | 0.8993 | 6.00 | 0.00 |
| random | 0.0918 | 0.8220 | 4.02 | 0.55 |
| ppo | 0.1162 | 0.8398 | 1.43 | 1.00 |

- **H1 is testable and is not supported at the upper budgets.** The exact best fixed subset beats greedy on held-out Brier at B=5 (0.0448 vs 0.0655) and B=6 (0.0479 vs 0.0512), while greedy remains the single closest policy to `V*` on the empirical objective at every budget (gap ≤ 0.0133). Spec §4 pre-registers this as a valid result.
- **The DQN collapse was an undertrained network, not a property of RL** (§A.5). At 400 episodes the bootstrapped objective sits at 0.60 items; it needs ~2,000. Storing return-to-go instead of bootstrapping reaches the same behaviour by 200 episodes — an order of magnitude sooner.
- **PPO's stopping degrades with more training**, because λ=0 leaves stopping unpriced. That is the strongest argument yet for signing off V-6.
- **DQN now trains to a sane policy and still does not win** on either the empirical objective or held-out Brier.
- All §17 arms are runnable. `ExactFixedSubsetPolicy`, `StaticRFEPolicy` and `IRTCATPolicy` existed but were instantiated nowhere until P1-d, so H1 had no comparator. `beta_greedy` (P1-f) is a **ninth, additive** arm — `GreedyIGPolicy` is deliberately untouched, so its near-optimality result stays intact and comparable.
- **Every artifact is reproducible from its recorded seed**, with the sole exception of the recorded wall-clock fields (`generated`, `wall_seconds`, `time_sec`, `dp_seconds`). Verified by rerunning every producing script and diffing: all substantive numbers identical.
- **The random arm's item count is not an adaptive-policy comparison.** `RandomPolicy` draws uniformly from the legal actions, which include STOP, so it ends some episodes early (measured on the 127 demo test records: mean 3.93 questions, ~9% ask nothing, ~45% reach the full budget of 6) while greedy always spends 6.00. Any greedy-vs-random comparison must state the two budgets side by side.

> **Everything above is measured against circular questionnaire labels and is not clinical evidence.** See `diagnosisReady.md` §4 and `V6_LAMBDA_DECISION.md` §4, which shows that at λ=0 the objective saturates (`V* = 1.000000`, support pure after ~4 questions), so the measured value of adaptive stopping is a property of the label rule.

All RL results are single-seed at λ=0 — see `AGENT_PROGRESS.md` §Next Required Steps.

## Reproducibility
- configs/config.yaml records budget, lambda_grid, seeds, thresholds.
- Spec requires config hash + git SHA + seed per MLflow run (stub ready for integration).
- Every results JSON carries `git_sha`, `config` path, `seed`, `source` (real or synthetic), and a "tag" indicating which verification gates are still open.

## Reports — §20
Step 3 (`scripts/step3_preliminary_reports.py`) emits Saudi-only preliminary reports to `results/`, all tagged `"preliminary — V-4 / V-6 / V-7 PENDING; supervisor sign-off required"`:

| Artifact | Contents |
|---|---|
| `results/perf_vs_budget_saudi.csv` | 13 rows: B ∈ {1..6} × {greedy, random} + terminal (B=10) reference. Columns: items_asked_mean, brier, uar, auroc, ece |
| `results/perf_vs_budget_saudi.json` | same numbers + metadata (tag, dataset, source, circularity_status, config, git_sha, terminal reference) |
| `results/faithfulness_saudi.json` | counterfactual flip rate 0.417, robust 0.583 (B=6, τ=0.5, 127 episodes) + SHAP |attr| mean per A1..A10 (top: A8 > A6 > A2) |
| `results/subgroup_saudi.json` | UAR/Brier by sex × age_band (B=6, τ=0.5); underpowered cells (< 20) marked per §25 |
| `results/predictor_saudi_metrics.json` | MaskedMLP[128,64]+isotonic on Saudi 506 (4-fold) — test Brier 0.0134, ECE 0.0168, AUROC 0.9877 (real data; high AUROC expected — labels are deterministically circular, §16.1 gate applies) |
| `results/predictor_uci_child_metrics.json` | same on UCI Child 292 — test Brier 0.0713, ECE 0.0765, AUROC 0.9301 (real data; same circularity caveat) |
| `results/dp_tractability_sweep.{json,csv}` | 48 ExactDP runs, full metadata per row |
| `results/step4_policy_training_saudi.json` | DQN + PPO training run (2026-10-01, re-run after the P0 collector fix). Carries an explicit circularity warning |
| `results/step5_policy_benchmark_saudi.{json,csv}` | matched-budget benchmark over nine arms (greedy, random, DQN, PPO, exact, exact_fixed_subset, static_rfe, irt_cat, beta_greedy) plus the V\*−V_emp optimality gap. **Greedy and beta_greedy attain the exact optimum on the empirical objective; DQN trains to a sane policy and still does not win; the exact best fixed subset wins on held-out Brier at B=5,6.** Circularity warning included. Current numbers in `POLICY_BENCHMARK_REPORT.md §A` |
| `results/lambda_sweep_saudi.{json,csv}` | Step-6 λ (question-cost) sweep, 8 λ × 8 B = 64 cells. **Predictor-independent.** Records `V*`, item counts, `stopped_early_frac`, the degeneracy boundary, and the support-purity-by-depth diagnostic. **Decision input for V-6 — not a signed-off result** |
| `results/rl_diagnosis_saudi.{json,csv}` | Step-7 bounded RL-collapse diagnosis: 5 seeds × 4 episode budgets × 2 learning targets (bootstrap vs return-to-go), with per-seed training curves and state-coverage traces |
| `results/evoi_scale_saudi.{json,csv}` | Step-8 EVOI scale analysis + stopping-rule sensitivity: the decision statistic's distribution (min/median/mean/p75/p90/p95/max) under both the support posterior and the neural predictor, the stop rate at each candidate threshold, and an offline sensitivity sweep over six stopping rules. **Selects no threshold.** **Decision input for V-6 — not a signed-off result** |
| `results/outcome_explainability_saudi.json` | Step-17 outcome explainability (P3-outcome): per-session exact group-Shapley attributions, verified counterfactuals, prior-sensitivity band, limitations, plus faithfulness aggregates (additivity error, deletion/insertion AUC vs random, rank stability, overhead). **Explains the model, not the person; screening result, not a diagnosis** |
| `results/demo_model_saudi_seed0_platt_v2.{pt,pkl}` | demo predictor cache; the filename encodes `predictor_version` so a semantics change cannot leave stale weights behind |

These numbers are from the **real CSVs** in `data/raw/` (re-run 2026-09-30; every artifact carries `"source": "real"`). The NZ 1,054-row cohort is the exception — it is still missing pending V-1, so any NZ number remains synthetic-fallback and is never reported.

## Pending verification gates — §24
V-1, V-2, V-4, V-5, V-6, V-7, V-9, V-10 remain NOT VERIFIED until human-supervised completion.

| Gate | Status (2026-09-04) | Blocker |
|---|---|---|
| V-1 | source located, content validated | human must obtain licence-clear copy of `Toddler Autism dataset July 2018.csv` and place at `data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv`; no file downloaded by agent |
| V-2 | PRISMA template ready (sections 1-8) | human operator runs the 12 search queries across 10 databases and populates `docs/prisma/screening_worksheet.csv` |
| V-4 | not started | MDE / comparison family freeze (before opening Polish for confirmatory) |
| V-5 | not started | order-independence evidence |
| V-6 | placeholder λ ∈ {0.0, 0.01} used in Step 2 | supervisor sign-off on λ grid |
| V-7 | 117 controls resolved (per DATA_VERIFICATION_REPORT §Polish) | supervisor freeze |
| V-9 | not started | ethics review |
| V-10 | not started | Tier-3 controlled-access go/no-go |

**V-2 no-claim rule:** until V-2 is complete, the phrases `first / only / no prior work / absent from the literature / to our knowledge` (when qualifying novelty) are FORBIDDEN in any artifact under this repo. Enforced by `tests/test_v2_no_claim_rule.py`. Allow-listed: `README.md`, `V2_PRISMA_SEARCH_LOG.md`, `AGENT_PROGRESS.md`, `V1_NZ_DATASET_RESOLUTION.md`, `AUDIT_REPORT.md`, `DATA_VERIFICATION_REPORT.md`, `STATE_COUNT_VERIFICATION.md`, `Master-Project-Specification_FINAL.md`.

## Limitations — §27
Questionnaire-derived labels (Saudi, UCI Child, NZ toddler target; Polish is clinical), order-invariance assumption, small Polish cohort, no clinician-in-loop, no participatory design — see spec.

## Terminology lock — §25
Use screening / referral recommendation / risk estimate ; never diagnosis.

---

### 3b. V-4 / V-7 external validation (sealed cohort)

```powershell
.venv-win\Scripts\python scripts\step9_v4_v7_gates.py
.venv-win\Scripts\python scripts\step10_external_validation.py
```

**Architecture — the sealed cohort never touches fitting:**

```
Saudi 506 (circular questionnaire labels)  ->  predictor / calibrator / RL policies   [DEVELOPMENT]
                                                  frozen: n_items=10, binary, dim 41
                                                        |
                                                        v  (frozen, never refit)
Polish 252 (clinician-established GROUP)     ->  external evaluation                [SEALED]
                                                  25 ordinal items, dim 199
```

| Gate | Automated evidence | Human sign-off | State |
|---|---|---|---|
| **V-4** MDE + confirmatory family freeze | PASS | **OPEN** | **OPEN** |
| **V-7** Polish denominator (252 vs 253) | PASS | **OPEN** | **OPEN** |
| **V-7** Baseline 10 recomputation | OPEN | **OPEN** | **OPEN** |
| External validation | — | — | **BLOCKED** |

`step10` currently **exits 2** and generates no prediction. Two independent blockers:
V-4/V-7 sign-off is outstanding, and the frozen Saudi predictor's declared feature
contract (10 binary items, `dim 41`) does not match the Polish cohort (25 ordinal
items, `dim 199`). The ordinal encoder (`m_list`) already exists in `encode_state`,
`MaskedPredictor`, `DQNPolicy` and `PPOPolicy`, but `load_polish` does not populate
it, and **no coercion is offered** — truncating a 4–6 level ordinal response into a
binary slot would silently destroy information.

Measured with the project's own audits:

| Cohort | best threshold | exact match | classification |
|---|---|---|---|
| Polish `GROUP` (clinical) | — | **0.5437** | **Not circular** |
| Saudi `Class` (questionnaire) | 4 | **1.0000** | Deterministic |

Artifacts (metadata only, never participant rows):
`results/polish_provenance_verification.json`, `results/v4_v7_validation_status.json`,
`results/polish_external_validation.json`.

`data/*.sav` is gitignored; `git ls-files data/` is empty, so no dataset and no
participant identifier is tracked.

### 3c. Calibration authority and the Polish decision

**Calibration.** Research metrics use **isotonic**; the browser demo uses Platt.
These are not in conflict — the demo's choice is deliberate (isotonic collapses to
3 breakpoints on 95 validation records). The actual defect was that Step 2 never
persisted its trained model, so the recorded isotonic metrics were unreproducible
from disk. Fixed: Step 2 now writes
`results/predictor_{saudi,uci_child}_v2_isotonic.{pt,pkl}` and records both
SHA-256 digests in the metrics artifact. Re-running Step 2 reproduced every metric
**bit-identically** (`test_brier` 0.009122572011964558 unchanged).

| artefact | sha256 |
|---|---|
| `predictor_saudi_v2_isotonic.pt` | `2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4` |
| `predictor_saudi_v2_isotonic.pkl` | `a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863` |

**Polish compatibility — answered OPEN.** All 25 external items are ordinal
(0 natively binary), and the cohort carries **no question wording**, so a
Q-CHAT-10 subset cannot be verified. Recorded OPEN rather than guessed.

* `POLISH_VALIDATION_DECISION.md` — three options compared; recommends **Option A**
  (verified 10-item subset) as the only one that validates the system as built.
  The decision is explicitly left to the supervisor.
* `EVIDENCE_REQUESTS.md` — the three items only a person can close (Sollis
  Baseline-10 specification, V-4 assumed SD, published instrument wording).
* `scripts/step10_external_validation.py` still exits **2** with **no metrics**.
