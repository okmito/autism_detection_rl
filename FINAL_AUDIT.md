# Final V2 Research Audit — Action Agreement & Policy Value (2026-11-05)

**Verdict: Retain EVOI only. Remove the current stopping mechanism. Bayesian
uncertainty improves nothing except reporting (kept, inert for selection/stopping).**

All analyses: TRAIN/VAL only (284/95). Held-out test (127, fingerprint
`b2021998a83e7224`) untouched. Questionnaire-label circularity disclosed
throughout. Nothing is clinical accuracy. No commits/pushes.

## 1. Action-level agreement (full budget, STOP disabled; 950 episodes = 10 seeds × 95 val)

| Pair | Exact full-sequence | First-q | Depth-1 | Depth-2 | Depth-3 | Depth-6 |
|---|---|---|---|---|---|---|
| hybrid vs EVOI-only | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid vs Beta-Greedy | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid vs IRT-only | 0.000 | 1.000 | 1.000 | 0.484 | 0.105 | 0.000 |
| hybrid vs Greedy | 0.168 | 1.000 | 1.000 | 1.000 | 0.621 | 0.168 |
| EVOI-only vs IRT-only | 0.000 | 1.000 | 1.000 | 0.484 | 0.105 | 0.000 |

- Disagreements between hybrid and EVOI-only: **0 of 950 episodes**.
- The combined IRT+EVOI architecture does **not** change a single question
  selection versus EVOI alone. Aggregate metrics being identical is not
  coincidence — the policies are functionally identical in action space.
- The action-ranking top-1 at every state is thus identical too.

**CONCLUSION 1: IRT contributes zero measurable action value over EVOI-only.**
Do not claim otherwise.

## 2. Uncertainty contribution (Bayesian predictive SD)
- `corr(sd, |error|) = +0.601`, monotonically increasing mean |error| by sd
  tertile (lo 0.136 → mid 0.449 → hi 0.500). SD rises with depth-0 error
  and falls as evidence accumulates.
- SD range 0.0115–0.0983 — always well below `abstain_sd=0.25`, so the
  current gate is inert (verified at `uncertainty_stop_sd=None`).
- The signal is real but not used: uncertainty selection correlates with
  error, yet the deployed stopping rule never fires on it.
- Keep `BayesianLogisticPredictor.predictive_uncertainty` for **reporting**
  (intervals, explanations). Do not use it to gate stopping until calibrated
  against supported thresholds on validation data.

## 3. Support-aware acquisition
- Support fallback (EVOI primary, Greedy-IG when support < train depth-0
  median) achieves **Brier 0.0709±0.0008** vs EVOI-only 0.0821±0.0009 —
  a small improvement, within ~2× seed-SD (0.0008/0.0009), so it is
  borderline-significant, not decisive. UAR 0.9438 vs 0.9516.
- The gain is because Greedy-IG picks a different item when support is thin;
  greedy-B6 separately reaches 0.0750. EVOI-only and beta-greedy 0.0821
  ties remain.

## 4. Burden-performance Pareto (VAL, mean across 10 seeds)
| Arm | Brier ↓ | 95% CI | UAR ↑ | AUROC ↑ | asked | ECE ↓ | stop |
|---|---|---|---|---|---|---|---|
| greedy_B6 | 0.0750 | [0.0739,0.0761] | 0.9277 | 0.9881 | 6.00 | 0.1987 | 0 |
| beta_greedy_B6 | 0.0821 | [0.0807,0.0836] | 0.9516 | 0.9980 | 6.00 | 0.2199 | 0 |
| EVOI-only B6 | 0.0821 | [0.0807,0.0836] | 0.9516 | 0.9980 | 6.00 | 0.2199 | 0 |
| IRT-only B6 | 0.0835 | — | 0.9115 | 0.9922 | 6.00 | — | 0 |
| exact_fixed_B6 | 0.0902 | [0.0882,0.0927] | 0.8548 | 0.9912 | 6.00 | 0.2139 | 0 |
| support_fallback B6 | 0.0709 | — | 0.9438 | — | 6.00 | — | 0 |
| greedy_B4 | 0.1075 | [0.1053,0.1102] | 0.8954 | 0.9790 | 4.00 | 0.2133 | 0 |
| greedy_B3 | 0.1299 | [0.1272,0.1324] | 0.5000 | 0.9627 | 3.00 | 0.2413 | 0 |
| hybrid original | 0.1041 | — | 0.8954 | 0.9805 | 3.34 | — | 0.937 |

Greedy remains the best matched-budget policy. EVOI-only ties beta-greedy
(they are action-identical). Hybrid's only frontier point is its
low-burden region, paid for at +0.03 Brier.

## 5. Scientific-reporting corrections
- IRT contribution: **none** measurable (0/950 disagreements).
- Hybrid vs Greedy B6: hybrid is worse (0.0821 vs 0.0750, CIs exclude zero
  in the held-out benchmark).
- EVOI = Beta-Greedy topologically; they are the same policy on this data.

## 6. Final decision
| Component | Recommendation | Evidence |
|---|---|---|
| Hybrid IRT + EVOI (combined) | REMOVE — EVOI-only reproduces it exactly | 950/950 identical sequences |
| EVOI (beta-smoothed empirical) | RETAIN | Brier 0.0821 = hybrid; no IRT needed |
| Greedy IG | RETAIN as primary policy | 0.0750 at B6, best matched budget |
| Current stopping mechanism | DISABLE/remove | Brier 0.1041 vs 0.0821 when disabled; never calibrates to supported thresholds |
| Bayesian predictive SD | KEEP (experimental reporting only) | corr 0.601 with error, but gate inert |

### Git
Nothing committed. Modified: `src/policies/hybrid_irt_evoi.py`, `scripts/demo_static/index.html`.
New untracked analysis scripts + reports listed in the repo root; historical benchmark preserved.
