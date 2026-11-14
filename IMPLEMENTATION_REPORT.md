# NeuroBridge: Implementation Report

Branch `neurobridge-development` (HEAD `c0a9d2b`). **Nothing was committed or pushed.** All changes are in the working tree for your review. The staged removals of `data/neurobridge.db` and the `autism_detection_rl` gitlink are preserved exactly as you left them.

---

## 1. Executive summary

Every item in the engineering audit that could be acted on safely has been implemented, tested, and mutation-checked. Items that were not done are listed with the reason in sections 8 and 10.

| | Before | After |
|---|---|---|
| Tests collected | 697 | **1042** (+345 new, all for the web app, shared engine, and metrics) |
| Full suite | The audit measured one extra failure in `test_security_audit` (the committed DB); README claimed 12 environmental failures | **950 passed, 12 failed, 80 skipped** |
| Web-app test coverage | none | API, security, data layer, privacy, validation, concurrency, screening, frontend DOM, support rules, config |

The 12 remaining failures are the same environmental set that existed before this work (frozen predictor hashes, one retained-results check, missing gitignored `.sav` exports, and the markdown phrase rule over older research reports). None is caused by or hidden by these changes. Details are in section 6.

**What changed for a user of the app**

- Any website the user visits could previously read and delete their locally stored observations. That is closed (no CORS, Origin/Host/Content-Type guard).
- "Delete all observations" now truly removes the text. Previously the text lived on in the timeline and in exports. Screening sessions can now be deleted, and the export covers every table.
- The Support Studio page works. It crashed on every load.
- The random policy can no longer open a session on a phantom question "A0", which made the session impossible to finish.
- A double-click on an answer button no longer answers the next question with the previous answer.
- The support report on the result page now shows the triggering answers, reasons and guidance, which were silently blank.
- Client mistakes are clean 4xx errors that name the field. Server faults are an opaque 500 with an `error_id` that also appears in the log.

**Two bugs the audit did not list were found and fixed along the way** (section 2, "Additional findings").

---

## 2. Every audit finding addressed

Status key: **Done** (implemented and tested), **Partial** (see note), **Skipped** (with reason), **Decision** (needs you).

### Critical

| ID | Status | Outcome |
|---|---|---|
| C1 | Done, plus **Decision** | `.gitignore` now covers `data/*.db`, `-wal`, `-shm`, `-journal`, and `node_modules/`. `test_no_committed_raw_data_or_sav` passes again. The DB's contents were never opened. Judged from structure and volume alone (5 sessions, 1 observation, 1 strategy, 2 timeline events) it is almost certainly developer test data, but that is an inference. The blob is still in history and on `origin`; purging it is your call (section 10). |
| C2 | Done | `flask_cors` removed. A `before_request` guard rejects non-loopback `Host` (DNS rebinding), non-JSON mutating requests (415), and foreign `Origin` (403). 19 tests, including forged forms and DNS-rebinding hosts. |
| C3 | Done | Four `$$('[data-...]', '#selector')` calls now pass elements. A static test forbids the pattern; DOM tests render all 9 routes and click through Support Studio. |
| C4 | Done | The opening question is chosen from legal items without STOP. A shared `require_legal_item` guard rejects STOP or any out-of-range index at the source. Tested with a policy that always returns STOP. |
| C5 | Done | Timeline rows keep a reference, not a copy. Text is read from the live record at query time, so deleting or editing the source removes or updates it everywhere. Migration `m3` blanks text that older versions had copied. New endpoints delete sessions and delete everything (consent log kept); export covers all tables. |

### Important

| ID | Status | Outcome |
|---|---|---|
| I1 | Done | Result page reads the real schema fields (`triggering_question_ids`, `triggering_responses`, `why_selected`, `description`, `source_attribution`, `review_status`). A contract test checks every field the renderer reads exists on the pydantic models. |
| I2 | Done | Answers carry `item_index`; the server answers 409 if it is not the pending question. Session writes use `WHERE id=? AND version=?`, so overlapping requests cannot both succeed. The client locks the buttons while a request is in flight and resyncs on 409. A 4-thread race test records exactly one answer. |
| I3 | Done | `app/validation.py` (strict pydantic): types, lengths, enums, binary values, bounded lists and query strings. `app/errors.py` maps client mistakes to 400/404/409/503; everything else is an opaque 500 with an error id. Unknown routes return JSON 404. Error text never echoes submitted values. |
| I4 | Done | `neurobridge.*` loggers cover predictor cache hit/miss/retrain/failure, artifact-load failures, guard rejections, and unhandled errors with ids. Request bodies are never logged. |
| I5 | Done | `requirements.txt` has pydantic and pyyaml and drops the never-imported mlflow and hydra, with a note on the CPU torch index. `requirements-dev.txt` adds pytest and documents optional extras. Verified by scanning every import in `app/` and `src/`. |
| I6 | Done, one deviation | Per-thread connections in WAL mode, `BEGIN IMMEDIATE` transactions (re-entrant), a read snapshot for export, versioned migrations (`PRAGMA user_version`), indexes for the real queries, and one UTC timestamp format. **Deviation:** no database-level foreign keys on `observations.session_id` or `triggered_observation_id`; SQLite cannot add them with `ALTER`. Instead the app validates references on write, clears them on delete, and migration `m3` clears existing dangling ones. |
| I7 | Done | Blank optional strings become NULL on create and update; required fields cannot be blanked on update. The edit form shows stored UTC times as local time. |
| I8 | Done | 299 new tests (section 5). Each fix was mutation-checked: revert it in a scratch copy and confirm the matching tests fail. |
| I9 | Done as relabel, **Decision** | The page now says consent is a record that does not switch features on or off. Enforcement is a product decision (section 10). |
| I10 | Done in code, **Decision** on artifacts | See "Metrics" below. |
| I11 | Done | `/api/health` reports `data_present` and `model_loaded` without loading the model. A missing dataset gives a 503 with instructions and no filesystem path. |

### Technical debt

| ID | Status | Outcome |
|---|---|---|
| T1 | Done for the app and the demo; **Partial** overall | New `src/screening/` package (`session.py`, `diagnostics.py`). The app and `scripts/demo_app.py` both call it for stepping, finishing, explanation, and report assembly, and the demo's documented diagnostic became the single copy. `rl_lab.replay` keeps its own instrumented loop (it records a snapshot per phase) but shares the decision rule and is pinned to `run_episode` by an 8-case parity test. Not merged: the two `build_predictor` functions (section 8). |
| T2 | Done for timeline and rules; **Partial** for routing | One `record_event` helper replaces 8 hand-written inserts. Suggestion rules moved out of the routing module. Routes were not split into Blueprints. |
| T3 | Done | Whole-word regexes, simple negation (a negator within two words in the same sentence), and every suggestion returns and displays the terms that fired and the field they came from. The limits of negation handling are stated in the module and pinned by a test. |
| T4 | Done, no values changed | The app reads budget, threshold and default policy from `configs/config.yaml` with `yaml.safe_load`; invalid values fail loudly. Entries that contradict the code (`dataset: nz`, `calibration: isotonic`) are annotated with comments, not edited. A test ties the app, `demo_app`, and `demo_live` constants to the file. |
| T5 | **Partial** | `os.chdir` removed from the app and the demo. `sys.path.insert` removed from the app. `ingest.py` reads are anchored to the repo root while its public path constants stay relative (scripts record them). `pyproject.toml` now has packaging metadata; a wheel builds cleanly and contains no data. Application tests pass when run from a different working directory. Remaining: section 8. |
| T6 | Done | Gitlink removed from the index (staged), as you approved. |
| T7 | **Skipped** | `tests/test_demo_behavior_audit.py:174` deliberately pins `run_episode`'s `"REFER"`, and frozen result artifacts contain that spelling, so the three compatibility shims must stay regardless. Changing the emitter adds risk and removes nothing. |
| T8 | Done | Interactive sessions start from a blank all-unasked state with no `record_index` and no `true_label`. Auto mode keeps record binding because it replays a held-out record. A test proves interactive start never reads the held-out records. |
| T9 | **Partial** | README test counts corrected (one line said 486 passed, others 558; now 950/12/80 throughout current-status lines). The 20 top-level report files were not moved. |
| T10 | Done (minimal) | The research card no longer prints "H1 supported: undefined" when an artifact has no verdict. |
| T11 | Not addressed | Low priority, marked "potential" in the audit. |
| T12 | **Partial** | Done: random `SECRET_KEY`, `localStorage` guarded, inline `onclick` removed (a test forbids inline handlers so a CSP is now possible), timeline kind filter complete. Not done: sha256 check before `pickle.load`, data fingerprint in the model cache key, expiry of abandoned sessions, the legacy demo's unbounded session dict. |

### Performance

| ID | Status | Outcome |
|---|---|---|
| P1 | **Skipped after measuring** | Implemented, then reverted. The audit scan costs 1.2 ms and the train matrix 0.06 ms per call. |
| P2 | **Skipped after measuring** | Implemented, then reverted. Building a greedy policy costs 0.18 ms. |
| P3 | **Skipped** | The extra counterfactual costs 1.4 ms of a 28 ms finish, and the UI reads `result.counterfactual`, so removing it would change behaviour for almost no gain. |
| P4 | **Skipped** | Single-user local app; nothing measurable. |
| P5 | Done | Timeline and observation filters run in SQL before the limit. This was also a correctness bug: matches older than the newest 500 events silently vanished. |
| P6 | **Skipped after measuring** | The dashboard call takes 1.9 ms. |
| P7 | Done | Vectorised bootstrap, **identical output to the old loop for the same seed, about 127x faster** (12.8 s to 0.10 s for 2000 resamples at n=506). |

### Metrics (I10 and P7 in detail)

- **I10.** `calibration_slope` was a regression on the raw probability with an L2 penalty, contrary to its own docstring. On a representative sample it reported **6.72**; the standard definition (unpenalised, on logit(p)) gives **2.28**. The new fit is a small Newton solver, checked against a tightly converged sklearn fit to about 1e-7. It returns NaN with a reason, never an invented number, for a single outcome class, constant predictions, or perfect separation. Saturated 0/1 predictions are clipped and stay finite. Results carry `calibration_scale: "logit"` so old and new values cannot be confused. `external_metrics.py` already used the correct definition and is unchanged.
- **P7.** Resamples that contain one outcome class have no AUROC or UAR. They are excluded and counted in `bootstrap_valid_resamples`, where the old code fed NaN to `np.percentile` and could turn the whole interval to NaN.

### Additional findings (not in the audit)

1. **RL Lab replay used half the displayed budget.** `rl_lab.replay` called `update_state(..., remaining - 1)` and then decremented again, so every answer cost two questions. Found by the new parity test and fixed.
2. **A reader on a new thread could block behind any writer.** My initial migration check took the writer lock before testing whether the database was already migrated. Flask's dev server uses a thread per request, so this would have stalled reads. Caught by a thread test I wrote and fixed with a lock-free fast path.

Also noted: a legitimate random-policy auto replay can stop with zero questions (about 1 in 11). A test I had written assumed at least one question and was intermittently failing. It is fixed, and a deterministic test now proves a zero-question report assembles correctly.

---

## 3. Changes per phase

**Phase 1: safety net and same-day fixes.** C1 (gitignore), C2, C3, C4, I1, I5, T6, the support-report limitations UI (per-recommendation limitations, report-level limitations, disclaimer), `tests/conftest.py`, the jsdom harness, and the initial 4 app test files. Full suite after: 680 passed.

**Phase 2: privacy, validation, data layer.** `app/db.py` rewrite with migrations `m1` to `m3`; `app/validation.py`; `app/errors.py`; `app/services/timeline.py`; I2 optimistic concurrency; C5 deletion and export; I4 logging; I9 relabel; I11 readiness; random `SECRET_KEY`. Full suite after: 783 passed.

**Phase 3: structure.** T1/T8 shared engine; T5 anchoring and packaging; T4 config loading; T3 suggestion rules; T2 helper extraction; T7 skipped. Full suite after: 868 passed.

**Phase 4: metrics and performance.** I10, P7, measurement-driven decisions on P1 to P6, a defect found in `rl_lab`, and the cheap optional items (`localStorage`, inline handler, H1 card). Full suite after: **950 passed.**

---

## 4. Files modified and why

**Modified**

| File | Why |
|---|---|
| `.gitignore` | Keep local databases and `node_modules/` out of git (C1). |
| `app/server.py` | Guard, error handlers, validation, timeline helper, privacy endpoints, full export, logging (C2, C5, I3, I4, T2). |
| `app/db.py` | Per-thread connections, transactions, migrations, UTC helper (I6, I7). |
| `app/config.py` | Read defaults from YAML; random secret (T4, T12). |
| `app/services/screening.py` | Shared engine, `item_index`/version concurrency, blank interactive state, readiness, logging (T1, T8, I2, I11, C4). |
| `app/services/rl_lab.py` | Budget double-decrement fix, shared decision rule, typed errors. |
| `app/services/research.py` | Log unreadable artifacts instead of ignoring them (I4). |
| `app/static/js/app.js` | C3 selectors, I1 fields, limitations UI, I2 lock/resync/`item_index`, privacy controls, local-time editing, matched-term display, H1 card, `localStorage` and inline-handler hardening. |
| `scripts/demo_app.py` | Session on the shared engine; no `os.chdir` (T1, T5). |
| `src/data/ingest.py` | Anchor reads to repo root (T5). |
| `src/eval/metrics.py`, `src/eval/fast_stats.py` | I10 and P7. |
| `configs/config.yaml` | Comments added; no value changed (T4). |
| `requirements.txt`, `README.md` | Manifest fix (I5); accurate counts and app documentation (T9). |
| `tests/test_v2_no_claim_rule.py` | The markdown scan now skips `node_modules` the same way it already skips `.venv`; third-party changelogs are not project claims. Project markdown is scanned exactly as before. |

**New**

| File | Purpose |
|---|---|
| `app/errors.py`, `app/validation.py` | Error types and request validation. |
| `app/services/timeline.py`, `app/services/support.py` | Timeline events; suggestion rules. |
| `src/screening/` (`session.py`, `diagnostics.py`) | The shared screening engine. |
| `pyproject.toml`, `requirements-dev.txt` | Packaging and pytest config; dev requirements. |
| `package.json`, `package-lock.json` | `jsdom` as a dev dependency for frontend tests. |
| `tests/conftest.py`, `tests/js/frontend_harness.js` | Temp-DB fixtures; jsdom runner. |
| 12 test files | See section 5. |

**Staged by you, preserved:** `D data/neurobridge.db`, `D autism_detection_rl`.

---

## 5. Tests added and their coverage

299 new tests in 12 files.

| File | Tests | Covers |
|---|---|---|
| `test_app_security.py` | 19 | C1 gitignore and untracked DB; C2 no CORS, forged forms and text/plain, foreign Origin, DNS rebinding, PUT/DELETE guarded, same-origin allowed. |
| `test_app_api.py` | 12 | CRUD contract for observations, strategies, attempts, suggestions, timeline, export, consent, privacy. |
| `test_app_screening.py` | 27 | C4 with a STOP-returning policy; every policy completes; auto mode; zero-question replay; stop; invalid answers; resume. |
| `test_app_concurrency.py` | 4 | Duplicate and stale answers (409), required `item_index`, version-conflict write refused, 4-thread race records exactly one answer. |
| `test_app_validation.py` | 53 | Wrong types, nulls, oversize, enums, binary values, references, query bounds, no value echo, opaque 500 with id, 503 without a path leak, health readiness. |
| `test_app_data_layer.py` | 24 | Migration of a legacy DB (copied text gone, live content untouched, blanks to NULL, UTC, dangling ids), transactions, nesting, thread isolation, no dirty reads, concurrent writers, timestamp normalisation and ordering. |
| `test_app_privacy.py` | 15 | Marker text planted in every field is absent from timeline, observations, strategies and export after each kind of delete; export completeness; SQL filters before limit. |
| `test_app_config.py` | 16 | YAML values, drift against demo constants, invalid values rejected. |
| `test_app_support_rules.py` | 39 | Whole-word matching, negation and its sentence boundary, the documented limit, field reporting, context rules, fallback. |
| `test_app_frontend.py` | 30 | All routes render; Support Studio; result-page fields; escaping; `item_index` sent; double-click lock; 409 resync; privacy buttons; local-time edit; no inline handlers; `localStorage` failure; H1 card. |
| `test_screening_engine.py` | 27 | Blank vs record state, STOP guard, answer application without mutating input, exhaustion, diagnostics, T8 on any dataset, RL-lab parity with `run_episode`. |
| `test_eval_metrics.py` | 33 | Slope vs reference solver, calibrated/over/under-confident/shifted simulations, undefined cases with reasons, saturation, invalid input, bootstrap equality with the old loop, chunk invariance, single-class resamples. |

**Mutation checking.** For each key fix I reverted or neutralised it in a scratch copy and confirmed the matching tests fail. Examples: version check removed, `item_index` check removed, reader fast-path removed (a thread hang), legacy text repair skipped, filter-before-limit disabled, substring matching restored, negation disabled, sentence boundary ignored, interactive start seeded from a record, probability-scale calibration fit, separation guard removed, NaNs passed to percentile, inline handler restored, unguarded `localStorage`, H1 card reverted. Two mutants initially survived (a sentence-boundary test whose second keyword masked the effect, and a T8 state comparison that cannot distinguish blank from record-seeded on this dataset); both tests were strengthened and re-checked. One mutant survives by design: the in-flight flag and the disabled buttons are redundant guards, either of which blocks the second request. One more (the `Z` timestamp suffix) matters solely on Python older than 3.11.

---

## 6. Test, build, and lint results

**Full suite:** `950 passed, 12 failed, 80 skipped` (1042 collected).

**The 12 failures (unchanged from before this work):**

| Count | Test | Cause |
|---|---|---|
| 7 | frozen predictor artifact hash checks (`test_baselines_and_diagnosis`, `test_corrected_design`, `test_qchat10_contract`, `test_predictor_v3_and_rl_protocol`) | The local `predictor_saudi_v2_isotonic` files do not match the recorded hashes. |
| 1 | `test_first_evaluation_results_are_retained_not_deleted` | Expected retained results are absent locally. |
| 3 | `test_v4_v7_validation` | The gitignored SPSS `.sav` export is missing. |
| 1 | `test_no_forbidden_phrases_in_markdown` | Older research report files contain the forbidden words. |

The skipped tests need data or artifacts that are not generated locally (for example `lambda_sweep_saudi.json`, the Polish run, the SPSS export). Frontend tests skip automatically when node or jsdom is absent.

**Build.** `python -m compileall app src scripts tests` exits 0. A wheel built into a scratch directory from a clean copy has the right metadata, dependencies, and static assets, and contains no database, data, results, or tests.

**JavaScript.** `node --check` passes for `app.js` and the test harness.

**Lint and type checking.** None is configured and none is installed (no ruff, flake8, pyflakes, or mypy, and no config files). Nothing was run and nothing is claimed. As a substitute I ran an AST scan for unused imports over every changed Python file and removed those I introduced. Pre-existing unused imports in `rl_lab.py`, `research.py` and `fast_stats.py` were left alone.

**Real database.** Verified byte-identical to the committed blob (same SHA-256), at schema version 0, with no WAL files. No test or tool opened it for writing; tests use temp files, and `conftest.py` asserts the path is not the real one.

**Migration rehearsal.** Run on a copy in the scratchpad (printing counts, never contents): schema version 0 to 3; row counts unchanged in all six tables; zero timeline rows still holding copied text; `integrity_check` ok; zero foreign-key violations.

---

## 7. Architectural changes

1. **`src/screening/`** is now the single implementation of a screening session: blank or record-seeded state, next question with the STOP guard, apply answer, exhaustion, finalise (counterfactual, explanation, support report, result), replayed diagnostics, and an auto run. It has no storage and no HTTP. The app persists its output in SQLite; the demo keeps it in memory.
2. **Data layer.** One connection per thread in WAL mode; explicit `BEGIN IMMEDIATE` writer transactions with nesting; read snapshots; versioned migrations applied under a write lock; UTC everywhere.
3. **Timeline holds references.** Events store `ref_type` and `ref_id`; text is joined from the live record on read, so privacy deletes need no extra step. Events with no source row carry constant text and nothing a person typed.
4. **Request boundary.** A guard hook for Host/Origin/Content-Type, strict pydantic models per route, and one JSON error envelope with a deliberate split between client errors (`ApiError` subclasses) and server faults (anything else, logged with an id).
5. **Optimistic concurrency** on screening sessions through a `version` column.
6. **Configuration.** Defaults come from `configs/config.yaml` with a validated fallback.
7. **Packaging.** `pyproject.toml` supports an editable install; the app no longer edits `sys.path` or the working directory on import.

---

## 8. Remaining technical debt

- **T5 remainder.** `os.chdir` still runs in `scripts/demo_live.py`, `step2_train_and_sweep.py`, and `step3_preliminary_reports.py`; 25 research scripts still insert the repo root into `sys.path` because they are run as plain files. Converting them is mechanical but touches many frozen-result producers, so it was not done blind.
- **T1 remainder.** `_build_predictor` (app) and `build_predictor` (demo) are still two copies with different logging. `rl_lab` keeps its own loop (pinned by a parity test).
- **T2.** Routes are one module; Blueprints would help but change nothing for users.
- **T4.** Research scripts still use literal constants rather than the YAML.
- **T7.** The `"REFER"` versus `"REFERRAL_RECOMMENDED"` split remains, with three shims.
- **T9.** About 20 report files sit at the repo root; README history sections keep older dated counts on purpose.
- **T11.** Dedupe hash collisions across sources are not reworded.
- **T12.** No sha256 check before `pickle.load`; the model cache key lacks a data fingerprint; abandoned sessions never expire; the legacy demo's session dict is unbounded.
- **No linter or type checker is configured.**
- **Python version.** `requires-python` is stated as `>=3.10`, but everything was run on 3.14 and no other version.
- **sklearn deprecation.** `src/eval/external_metrics.py` passes `penalty=None`, deprecated in scikit-learn 1.8 and slated for removal in 1.10. My new code avoids it; that older file still needs the change.
- **Optional items not done:** production WSGI server (E1), CI (E2), CSP and security headers (E3; the inline-handler blocker is now gone), wider accessibility work (E4), database encryption at rest (E5), "estimate" wording in the UI (E6).
- **Unavoidable: 9 of 15 research artifacts are not present locally**, so most dashboard cards show "not available".

---

## 9. Security and privacy findings

1. **Sensitive data in git history.** `data/neurobridge.db` (270 KB) was committed in `c0a9d2b` and is on `origin/neurobridge-development`. It is now untracked and ignored, but the blob remains reachable in history.
2. **Cross-site read and delete (C2).** Wildcard CORS plus no CSRF checks let any visited website read observations and the timeline export, or wipe observations with a plain form post. Closed.
3. **Deleted text survived (C5).** Observation text was copied into timeline rows that no delete touched, and into exports. Closed in code and repaired in existing databases by migration `m3`. That repair is irreversible on purpose, and **no automatic backup is made**, because a backup would preserve the very copies being removed. Live observations and strategies are never altered by it.
4. **Error leakage (I3, I11).** Raw exception text, SQL errors, and absolute filesystem paths reached clients. Now an opaque 500 with an id, or a path-free 503.
5. **Consent is not enforced (I9).** The page now says so; enforcement needs your decision.
6. **The database is plaintext (E5).** Anyone with file access can read it. The README and Privacy page call it sensitive; encryption is not implemented.
7. **Exports are sensitive.** The export now contains everything stored, including screening sessions. The response note says to handle it accordingly.
8. **Hard-coded secret (T12).** Replaced by a random per-process key (nothing uses it yet).
9. **`pickle.load`** of the calibrator from `results/` remains. It reads trusted local data today; a hash check would make that explicit.
10. **Logging.** Request bodies, which hold behaviour and health notes, are never logged.
11. **The migration will modify the real database on its next app start** (schema 0 to 3, including the one-time text clean-up). A copy was rehearsed successfully; the real file was not touched.

---

## 10. Decisions requiring your approval

1. **Purge the DB blob from history?** Rewriting a pushed branch and force-pushing is yours to approve. If the contents were real, treat them as exposed to anyone with access to the remote. A scratch clone is a safe place to rehearse `git filter-repo` beforehand.
2. **Enforce consent, or keep it as an acknowledgement log?** Enforcing means refusing to store data without a granted, unrevoked record per scope.
3. **Regenerate `results/` after the I10 fix?** `ablation_af` and `benchmark_multiseed` artifacts hold the old probability-scale slope. They are gitignored and were left untouched. Averaged slopes may become NaN where a run's predictions separate perfectly; consider `nanmean` there.
4. **The 12 environmental failures.** Restoring the `.sav` export and the matching frozen predictor files, or re-freezing the hashes, is a research-provenance choice.
5. **Dev tooling.** I added `jsdom` (a dev dependency, lockfile included) and `pyproject.toml` packaging metadata. Approve or drop.
6. **Commit messages.** The repo's phrase rule also applies to commit messages: avoid the words the markdown test forbids.
7. **Demo parity.** `scripts/demo_app.py` still binds each session to a held-out record and shows its label. T8 was applied to the web app alone; changing the demo alters pinned behaviour.
8. **T7** (decision vocabulary) stays skipped unless you want the artifact readers and the pinning test updated together.

---

## 11. Recommended next steps

1. Decide on the history purge (item 1 above) before anything else, because it gets harder after more people clone.
2. Start the app once against a **copy** of your database if you want to see the migration happen, or back the file up yourself before the next run.
3. Review the diff by area: `app/db.py` and `app/server.py` (the largest), then `src/screening/session.py`, then the tests.
4. Add a linter and type checker (ruff plus mypy on `app/` and `src/screening/` is a small start) and run the suite in CI with the data-dependent tests behind an explicit marker.
5. Finish T5 for the research scripts as one mechanical, separately reviewed change, then point them at `configs/config.yaml`.
6. Replace `penalty=None` in `external_metrics.py` before scikit-learn 1.10.
7. Add a CSP and security headers now that inline handlers are gone.
8. Consider a data fingerprint in the model cache key and a hash check before `pickle.load`.
9. Move the root-level report files into `docs/reports/` with an index.

---

## Addendum (2026-11-13): adaptive RL questionnaire

The RL ASK/STOP questionnaire work (`src/rl/`, steps 18-20, app policy `rl_adaptive`) is reported separately in
`RL_ADAPTIVE_REPORT.md`.
