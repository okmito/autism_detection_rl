# Stopping Investigation & Ablation A–F (VAL only, 2026-11-05)

**Verdict: HYBRID QUESTION SELECTION IS USEFUL, STOPPING NEEDS REVISION.**

All experiments below ran on TRAIN/VAL only (284/95). The held-out test
split (127 records, fingerprint `b2021998a83e7224`) remained **untouched**.
The prior 10-seed benchmark (`BENCHMARK_REPORT.md`, hybrid Brier 0.1215,
3.31 questions) is preserved as a historical result.

> Circular questionnaire labels. Nothing here is clinical accuracy. λ = 0
> remains an unsigned research parameter.

## 1. Root cause of early stopping

Verified defect **F** in `HybridIRT_EVOIPolicy.__call__` (now fixed):
the policy returned `STOP` when neither `min_questions` nor `max_questions`
applied and `best_net <= 0`, **even when `STOP` was not in the legal set**.
`src/policies/hybrid_irt_evoi.py` lines 157–163 returned STOP unconditionally.
The legality test `test_never_returns_stop_when_stop_not_legal` locks the fix.

Beyond the defect, the mechanism is **C (support collapse) + configuration**:
- `predictive_sd` ≈ 0.06, always below `abstain_sd = 0.25`, so the
  uncertainty gate never blocked a stop (A/B partially active, but the SD
  is an order of magnitude below the threshold; `uncertainty_stop_sd=None`).
- Trace (val, seed 0): best_evoi decays +0.108 (n=0) → −0.0036 (n=6);
  support collapses to ≤5 records by n=5. On a near-pure support the
  Beta-smoothed utility is already ~1.0, so EVOI is non-positive.
- Environment budget is 6; `max_questions=10` (benchmark construction) and
  `min_questions=0`, so nothing forces budget use.

**Stop tally over 95 val episodes was 93.7% policy-STOP, 6.3% env exhaustion
(benchmark applied to val reproduces 3.34 mean questions; see
`results/terminal_event_reconciliation.json`).**

## 2. Ablations (VAL, 10 seeds, mean over seeds)

| Arm | What differs | Brier ↓ | UAR ↑ | AUROC ↑ | asked | stop |
|---|---|---|---|---|---|---|
| A original hybrid | baseline | 0.1041 | 0.8954 | 0.9805 | 3.34 | 0.937 |
| B stopping disabled (full budget) | STOP removed from legal | **0.0821** | **0.9516** | **0.9980** | 6.00 | 0.000 |
| C IRT-only (Fisher) | EVOI ignored | 0.0835 | 0.9115 | 0.9922 | 6.00 | 0.000 |
| D EVOI-only (net score) | IRT ignored | **0.0821** | **0.9516** | **0.9980** | 6.00 | 0.000 |
| E uncertainty disabled in stopping | SD = 0 | 0.1041 | 0.8954 | 0.9805 | 3.34 | 0.937 |
| F greedy B3–B6 | matched budgets | 0.1299/0.1075/0.0907/**0.0750** | 0.50/0.90/0.85/**0.93** | — | 3/4/5/6 | 0 |
| F beta-greedy B3–B6 | matched budgets | 0.1304/0.1080/0.1014/0.0821 | 0.50/0.90/0.75/0.95 | — | 3/4/5/6 | 0 |

- **B == D:** forcing full-budget use reproduces EVOI-only performance
  exactly. The question-selection logic is doing the work; stopping is the deficit.
- **E == A:** uncertainty in the stopping gate is inert (SD < threshold) — disabling it changes nothing.
- **C < D:** IRT-only (0.0835) > EVOI-only (0.0821), but both beat A and both equal F beta-greedy B6 (0.0821).
- PPO/DQN not in this study (no retraining).

## 3. Pareto (burden vs Brier, VAL)
Greedy-B6 0.0750 (6.0) < D/B 0.0821 (6.0) = beta-greedy B6; F-B5 0.0907 (5.0); C 0.0835 (6.0); A/E 0.1041 (3.34); F-B3 0.13 (3.0). The only low-burden region is the low-performance region. No Pareto improvement over greedy at matched budget.

## 4. Components — recommendation
- **Question selection (EVOI net score): RETAIN** — it drives the full-budget gain.
- **IRT Fisher ranking: REVISE** — contribution is real but smaller than EVOI's; keep as tie-breaker/robustness, not primary.
- **Stopping rule: REVISE (blocking)** — never gate on a single non-positive EVOI with collapsed support. Uncertainty-aware stop requires upper-bound-on-net-EVOI ≤ 0 with supported uncertainty; currently the support-collapse makes that criterion degenerate. Until V-6, default to budget use or min_questions ≥ budget.
- **Bayesian predictive SD: RETAIN for reporting** — but it must not gate stopping at its current magnitude.

## 5. Tests
491 passed / 15 environmental failures / 184 skipped (identical failure set to baseline). New: legality regression test for the STOP-when-illegal defect.

## 6. Git
Nothing committed. Modified: `src/policies/hybrid_irt_evoi.py`, `scripts/demo_static/index.html`.
