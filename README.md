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
Expected: **67 passed** on CPU. Useful flags: `-x` stop at first failure, `-rs` show skip reasons, `-k <keyword>` filter.

> **Disk-space gotcha (2026-10-01):** a plain `pip install -r requirements.txt` pulls the **CUDA** build of torch (~2 GB of nvidia wheels) and fails with `No space left on device` — `/tmp` is a 3.7 GB tmpfs. Install CPU-only torch instead:
> ```bash
> TMPDIR=/path/to/big/disk .venv/bin/pip install \
>   --extra-index-url https://download.pytorch.org/whl/cpu torch
> TMPDIR=/path/to/big/disk .venv/bin/pip install -r requirements.txt
> ```

### 3. Run the pipeline (optional — regenerates `results/`)
`data/raw/` and `results/` are gitignored. If `results/` is empty, regenerate:
```bash
python scripts\step2_train_and_sweep.py        # predictor training + 48-run DP sweep (~1 min)
python scripts\step3_preliminary_reports.py    # perf-vs-budget, faithfulness, subgroup reports
python scripts\step4_train_policies.py --episodes 400 --budget 6   # DQN + PPO training
```
Scripts automatically use the real CSVs when present under `data/raw/` (every artifact records `"source": "real"`) and fall back to synthetic data for smoke-testing otherwise.

### 4. Run the live demo

**Browser demo (recommended):**
```bat
:: Windows
.venv-win\Scripts\activate
python scripts\demo_app.py
:: open http://127.0.0.1:8000
```
First start trains the predictor (~20 s) and caches it to `results/demo_model_saudi_seed0_platt.*`; later starts are instant. Options: `--retrain` (ignore cache), `--port 8080` (auto-scans the next 9 ports if busy). The page shows the data audit, an interactive adaptive interview (greedy vs random policy), the per-answer belief trace, the referral decision with a counterfactual explanation, and the project's generated result artifacts.

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
| Step-3 tests skip | `results/` artifacts missing — run the Step 2/3 scripts above |
| Port 8000 busy | `python scripts\demo_app.py --port 8080` |
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
| **`RL_TRAINING_REPORT.md`** | **the RL defects found in never-executed code, fixes, and first real training runs (2026-10-01)** |
| `AGENT_PROGRESS.md` | canonical project state log; what was done, what failed, what's blocked, what's next |
| `DATA_VERIFICATION_REPORT.md` | per-dataset verification: rows, columns, circularity, missing, invalid-value handling |
| `STATE_COUNT_VERIFICATION.md` | theoretical (Q-CHAT-10, Q-CHAT-25) and empirical DP tractability counts |
| `AUDIT_REPORT.md` | circularity + leakage + invalid-value + Polish isolation + Step 2/3 audit trail |
| `V1_NZ_DATASET_RESOLUTION.md` | V-1 source-location + schema + circularity + loader plan for the NZ 1,054-row cohort |
| `V2_PRISMA_SEARCH_LOG.md` | PRISMA template (research question, 10 databases, 12 queries, criteria, no-claim rule) |
| `AUDIT_UPDATE_2026-09-04.md` | consolidation of the 2026-09-04 audit: re-runs, cross-checks, change log |
| `Master-Project-Specification_FINAL.md` | source of truth (spec §1-§27) |

## Repository layout — §8
```
src/data/ingest.py, schema.py, dedupe.py
src/audits/circularity.py, leakage.py
src/env/state.py, environment.py, costs.py
src/solvers/exact_custom.py, exact_adapter.py
src/models/masked_predictor.py
src/policies/dqn.py, ppo.py, greedy.py, random_policy.py, irt_cat.py, dqn_cat.py, static_rfe.py, static_fixed.py, replay.py
src/explain/trace.py, counterfactual.py, shap_baseline.py
src/eval/metrics.py, bootstrap.py, power.py, fwer.py, subgroup.py
src/ablation/runner.py
configs/config.yaml (Hydra)
scripts/step2_train_and_sweep.py, scripts/step3_preliminary_reports.py, scripts/step4_train_policies.py
scripts/demo_live.py (terminal demo), scripts/demo_app.py + scripts/demo_static/ (browser demo)
docs/prisma/screening_worksheet.csv
tests/ (67 tests, all passing)
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
pytest tests -q   # 67 tests
```
26 original tests (§21 core) + 10 DP tractability invariants (`tests/test_dp_tractability_sweep.py`) + 4 Step 3 artifact contract tests (`tests/test_step3_artifacts.py`) + 3 V-2 no-claim rule tests (`tests/test_v2_no_claim_rule.py`) + 11 demo-behaviour audit regressions (`tests/test_demo_behavior_audit.py`) + **13 RL training regressions** (`tests/test_rl_training.py`). All passing (2026-10-01; the no-claim test skips hidden dirs such as `.venv/`, and the Step-3 path check accepts both Windows and POSIX separators).

Covers state encoding, legal actions, budget, state counts, reward bounds, predictor, exact optimality, circularity, leakage, counterfactual, threshold freeze, common evaluator, trace, fixed subset, DP tractability, preliminary report metadata, PRISMA template presence, plus: greedy determinism + legality, random-policy variation semantics, belief bounds and continuity (no isotonic step collapse), Platt partial-evidence posteriors staying interior, the documented `p_hat >= tau` decision rule at both env and API layers, API risk continuity, and faithful frontend rendering of the backend risk value.

The 13 RL tests are **regressions for defects that had never been caught** because no training code had ever executed — see `RL_TRAINING_REPORT.md`. Notably: DQN's bootstrap must exclude illegal actions, terminal transitions must contribute no bootstrap, PPO's critic must actually receive a gradient, and illegal actions must get zero probability.

## ⚠️ RL policies — training vs benchmarking
`scripts/step4_train_policies.py` **trains** DQN and PPO; it does not benchmark them. There is as yet **no evaluation of the trained policies** at matched budgets against Greedy-IG, Random, or the exact DP — the project's central claim (learned vs heuristic vs exactly-solved optimum) remains unevidenced. Adding that benchmark is the next unblocked step; see `diagnosisReady.md` §7 Part 2.

All RL results are trained against **circular questionnaire labels** and are therefore not clinical evidence. See `diagnosisReady.md` §4.

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
| `results/step4_policy_training_saudi.json` | first real DQN + PPO training run (2026-10-01). **Training only — no benchmark.** Carries an explicit circularity warning |

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
