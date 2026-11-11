# Final V2 Benchmark — Hybrid IRT+EVOI vs Baselines (2026-11-05)

**Verdict: HYBRID UNDERPERFORMS BASELINES.**
All numbers below are actual outputs of scripts executed this run: `scripts/benchmark_multiseed.py --seeds 10 --budget 6` (wall 13.1s, 70 completed, 0 failed) and `scripts/paired_bootstrap_hybrid.py` (2000 subject-level resamples).

> **CIRCULARITY DISCLOSURE (re-read before interpreting any number).** Saudi labels are a deterministic sum-threshold over the Q-CHAT-10 items (`audit_circularity: exact_match=1.0, Deterministic`). Everything below measures fit to the questionnaire's own scoring rule, **not** autism and **not** clinical diagnostic accuracy. λ = 0.0 is an unsigned research parameter; it is not clinically optimised. The existing v2 test-suite failures (486 passed / 15 environmental failures / 184 skipped) are unchanged by this run.

## 1. Benchmark configuration
- CLI: `python scripts/benchmark_multiseed.py --seeds 10 --budget 6 --lambda 0.0`
- Predictor: shared per-seed v3 `LogisticPredictor` (C=0.1, Platt on val) + `BayesianLogisticPredictor` Laplace wrapper (100 posterior draws). DQN/PPO are **not** in this runner by design: DQN trains to a sane policy but never beat greedy in Step-5, and PPO degrades with training at λ=0; neither participates in the multi-seed loop (documented in the runner header). They are not eligible here without retraining (2,000 episodes/arm), which is out of scope for this comparison.
- Budget: B=6 fixed. Split: `stratified-4fold-2nd-fold-val`, fingerprint **b2021998a83e7224**, 284 train / 95 val / 127 test. Provenance overlap 0, test synthetic 0, `label_source` = {questionnaire} only.
- Exact fixed subset: `ExactFixedSubsetPolicy(train, budget=6)` → subset [2,3,6,8,9,10] (1-based), selected on training data only.
- All 7 eligible arms evaluated on the same 127 subjects with the same `run_episode` evaluator and identical `compute_metrics`.

## 2. Policy comparison (mean ± SD across 10 seeds)

| policy | Brier ↓ | UAR ↑ | AUROC ↑ | asked | stop-early |
|---|---|---|---|---|---|
| greedy | 0.0923±0.0009 | 0.8274±0.0063 | 0.9865 | 6.00 | 0.000 |
| beta_greedy | 0.0994±0.0009 | **0.8500±0.0115** | 0.9859 | 6.00 | 0.000 |
| irt_cat | 0.0994±0.0005 | 0.8275±0.0000 | 0.9797 | 6.00 | 0.000 |
| static_rfe | 0.0950±0.0015 | 0.8333±0.0000 | 0.9881 | 6.00 | 0.000 |
| exact_fixed_subset | 0.0950±0.0015 | 0.8333±0.0000 | 0.9881 | 6.00 | 0.000 |
| **hybrid_irt_evoi** | **0.1215±0.0015** | **0.7976±0.0000** | **0.9751** | **3.31** | **0.921** |
| random | 0.1677±0.0120 | 0.6529±0.0378 | 0.8764 | 4.07 | 0.539 |

Seed-SD is tiny (≤0.0015) except random — the hybrid is consistently, not randomly, behind.

## 3. Paired subject-level bootstrap (hybrid − comparator, Brier; + = hybrid worse)
| comparison | diff | 95% CI |
|---|---|---|
| hybrid vs greedy | +0.0292 | [+0.0201, +0.0391] |
| hybrid vs beta_greedy | +0.0221 | [+0.0147, +0.0303] |
| hybrid vs exact_fixed_subset | +0.0266 | [+0.0175, +0.0366] |
| hybrid vs irt_cat | +0.0221 | [+0.0103, +0.0350] |

Every CI excludes zero: the deficit is statistically supported, not seed noise. Seed variability and subject-level bootstrap are reported separately — seed SD is small; the paired CI spans subject resampling.

## 4. Exact fixed-subset comparison
The exact best fixed subset (train-only) attains Brier 0.0950 vs hybrid 0.1215 (+0.0266, CI excludes zero). H1 is again not supported for the hybrid arm at B=6, consistent with the Step-5 finding (greedy/beta-greedy 0.0512/0.0482 on the v2 MLP vs 0.095 here on v3/Bayesian — predictor differences are noted, never mixed).

## 5. Research-validity checks (verified, not assumed)
- No train/test leakage: canonical split fingerprint stable across every artifact; overlap 0.
- No calibration leakage: Platt calibrator fit on val only (95 records).
- No target-label leakage into policy selection: all arms select items from training support or frozen predictor, never from test labels.
- No test-set threshold optimization: τ=0.5 frozen.
- No synthetic data in held-out evaluation: test synthetic count 0.
- No synthetic/real merging: enforced by `src/data/validate.py` (Stage 2 fix).

## 6. Scientific limitations
- Circular questionnaire labels (above) — the whole comparison is against the questionnaire rule.
- λ=0 unsigned; the hybrid's stop-early behaviour (3.31 questions, 92.1% early stop) is exactly the construct the V-6 decision exists to price. At λ>0 the hybrid might stop later, but no λ is signed off.
- UAR for hybrid 0.7976 is below beta_greedy 0.8500; question count 3.31 is not a benefit when performance does not hold.
- DQN/PPO excluded (see §1).

## 7. Reproducibility
Seeded end-to-end (torch, numpy, python random). Re-running `benchmark_multiseed.py --seeds 10 --budget 6` reproduces the JSON bit-identically except wall-clock fields. Predictor versions recorded in the artifact.

## 8. Git status
Nothing committed or pushed. New untracked artifacts: `results/benchmark_multiseed_saudi.{json,csv}`, `results/paired_bootstrap_hybrid.json`, `scripts/{paired_bootstrap_hybrid,summarise_benchmark}.py`, `BENCHMARK_REPORT.md`. Modified: `README.md` (486/15/184 only).
