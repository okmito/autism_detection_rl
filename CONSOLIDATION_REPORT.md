# V2 Architecture Consolidation Report (2026-11-05)

## 1. Files modified
- `configs/config.yaml` — `policy.type: greedy` is now the single source of truth for the default path. All experimental policies documented in-place.
- `scripts/demo_app.py` — `make_policy` exposes `greedy` (default), `random`, `beta_greedy`; unknown policies raise `ValueError` instead of silently returning greedy. `api_start` writes the same validated set.
- `scripts/demo_live.py` — `_default_policy(train)` single constructor for the default path; live demo and policy comparison route through it.
- `scripts/demo_static/index.html` — active-configuration bar (policy / stopping / uncertainty / data) added; footer records the consolidation decisions verbatim.

No policy implementation was deleted or altered (hybrid, IRT, DQN, PPO all preserved at source). No existing test was modified or weakened.

## 2. Default policy configuration
`configs/config.yaml: policy.type = greedy`. Greedy IG is the best matched-budget policy on the evidence (val Brier 0.0750 at B6, ECE 0.1987; 10-seed benchmark confirms greedy ≠ hybrid: Brier 0.0923 vs 0.1215). The demo, live demo, and benchmark default all use it consistently. Beta-Greedy remains selectable in the demo where supported; it is action-identical to EVOI-only on this data.

## 3. Experimental components preserved
- `src/policies/hybrid_irt_evoi.py` — removed from the default execution path; preserved exactly as implemented for reproducibility. Its selection output is bit-identical to EVOI-only (950/950 episodes) and its stopping rule is disabled.
- `src/policies/irt_cat.py` — retained for research comparisons.
- `src/policies/stopping.py` + `HybridIRT_EVOIPolicy` stopping path — disabled in the default workflow; no arbitrary minimum-question rule and no clinical threshold introduced.
- `src/simulator/` — isolated synthetic patient environment. `results/*` — all historical artifacts preserved (benchmark, ablation, audit, reconciliation).
- Bayesian predictive SD — kept for experimental reporting only.

## 4. Stopping behaviour
Default workflow: sessions run to the environment question budget (6) with `b_min=0`. STOP is only taken when the budget is exhausted or the user manually ends the session. A verified STOP-legality fix (never return STOP when STOP is absent from the legal set) remains in place. No adaptive stopping in the default path.

## 5. Dashboard changes
Active-configuration bar shows: active policy (from backend session), `Stopping: budget use (no adaptive stop)`, `Uncertainty: Bayesian SD — reporting only`, data source (real vs synthetic naming). The gauge labels remain "Research screening estimate / Elevated screening signal / Inconclusive" — no Low/Moderate/High clinical risk labels introduced. Disclaimer string preserved and extended with the ECE caveat. Responsive deep-navy layout preserved.

## 6. Security checks
`.gitignore` covers `data/*.sav`, `data/raw/`, `results/`, `.env` (tested). No new dependencies, no telemetry, no PII logging. Full security suite passes.

## 7. Test results
`python -m pytest tests -q` → **491 passed, 15 failed, 184 skipped**. The 15 failures are the same pre-existing environmental set from the original baseline (missing gitignored Polish `.sav`/CSV, frozen artifact-hash pins, 1 doc-rule failure) — byte-identical to the baseline recorded at V2 start. No newly introduced failures.

## 8. Remaining limitations
- SAE questionnaire-label circularity (deterministic sum-threshold) remains unresolved until Polish/other clinical cohorts are used.
- ECE 0.20–0.24 (all arms, held-out) — probabilities are research-grade, not clinically calibrated; documented in the dashboard footer.
- λ remains an unsigned research parameter (V-6). Support-aware fallback remains experimental (0.0709 val Brier, ~2× seed-SD vs greedy-B6 0.0750) and unpromoted pending paired subject-level comparison on untouched data.
- GRM/IRT-ordinal on Polish is BLOCKED (no verified item mapping/wording).

## 9. Git status
Nothing committed or pushed. Modified files listed in §1. The application now has a clear, reproducible default architecture consistent with the strongest experimental evidence.
