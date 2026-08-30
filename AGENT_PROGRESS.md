# Current Project State

## Completed
- Repository scaffold per §8 — all src/*, configs, tests, data/, results/ created
- Data layer: schema.py, ingest.py (6 loaders + synthetic mode + Q-CHAT-10 binary map §15), dedupe.py — NOT using 6075-row pooled NZ file as primary cohort
- NZ 1,054 cohort searched: final-with-99-accuracy.ipynb references Toddler Autism dataset July 2018.csv (1054 rows, Case_No/A1-A10/Age_Mons/Qchat-10-Score/Sex/...) — file NOT present locally; no 1054-row provenance established
- Saudi ingestion finalized: 506 rows verified (A10..A1 reversed → corrected to A1..A10), Screening Score == sum(A) 506/506, Class deterministic >=4, schema OK, circularity Deterministic
- Polish ingestion finalized: 252 rows verified (135 ASD / 117 control → resolves 252/253 discrepancy: 117 controls, not 118), 25 qchat categorical strings, per-item vocab encoded to integers, invalid qchat4 value 11.0 (row 60 child_id bdbp0221) logged explicitly as MISSING per instruction §7, warnings emitted, schema OK, circularity Not circular
- UCI Child finalized: 292 rows verified via ARFF (90 ? markers), schema OK, matches UCI ID 419; adol/adult remain HUMAN ACTION REQUIRED pending ARFFs
- Environment: three-state encoding §9 distinct, costs.py, episode loop §10 with STOP/budget/missing
- Predictor: MaskedMLP 128→64 sigmoid §12 + random-mask training + calibration
- Exact DP authoritative §14.1 with tractability limits 50M/24h/32GB — verified tractable on Saudi 100-rec B=6 → 16826 states (ref 26025), B=3 → 1144 states
- Adapter stub §14.2 compatibility check
- Policies: DQN, PPO, Greedy IG, Random, IRT-CAT, DQN-CAT, RFE, Exact Fixed Subset (§17)
- Explain: trace, counterfactual, SHAP baseline
- Eval: metrics, bootstrap 2000, power/MDE, Holm-Bonferroni, subgroup
- Ablation runner §20 (7 ablations)
- Config: configs/config.yaml §13
- Tests: 26 tests covering §21 — all passing (3.25s) + real-data ingestion smoke tests (Saudi/Polish/UCI schema, circularity, episodes, DP)

## In Progress
- None — non-NZ infrastructure complete

## Verified
- 26 tests passed (pytest 3.25s): state_encoding, legal_actions, budget_exhaustion, reachable_state_count (26025 ref, 3081146397 canonical, ADI-R), reward_bounds, predictor_partial_input, exact_optimality, exact_value_consistency, no_leakage (poisoned control caught), circularity_oracle, counterfactual_validity, threshold_freeze, common_empirical_evaluator, trace_belief_update, exact_fixed_subset
- Real-data verification 2026-08-30: Saudi 506 (341 YES/165 NO) deterministic threshold 4; Polish 252 (135/117) Sum_QCHAT mean 33.4, m_list spec canonical 24x5+1x6 vs observed 24x5+1x6 with 2 items at 4 levels (qchat2, qchat13) — actual reachable 2667729775 vs canonical 3081146397 documented; invalid 11.0 logged to _POLISH_INVALID_LOG; UCI child 292 (151/141) verified

## Failed
- None

## Blocked
- HUMAN ACTION REQUIRED — NZ primary cohort missing: 1,054-row Toddler Autism dataset July 2018.csv not present (only Autism_Screening_Data_Combined.csv 6075 pooled remains unchanged at data/raw/Q-CHAT NZ/). Do NOT create 1054 subset by filtering. See §15 V-1.
- V-2 systematic literature search log (blocks novelty claims)
- V-4 MDE/comparison family freeze (before opening Polish for confirmatory)
- V-5 order-independence evidence
- V-6 lambda grid sign-off
- V-7 Polish denominator 252 vs 253 resolved to 117 controls — pending supervisor freeze
- V-9 ethics review
- V-10 Tier-3 go/no-go

## Next Required Steps
1. Human obtains Toddler Autism dataset July 2018.csv (1,054) or defensible equivalent and places at data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv — then implement _load_nz_toddler_csv provenance path
2. Complete V-2 PRISMA search log before any novelty wording
3. Train predictor on Saudi+UCI (NZ pending) and run exact DP tractability curves at B=3-6 (already verified on Saudi 100-rec sample)
4. Freeze V-4 MDE family before confirmatory Polish opening (Polish remains isolated — no threshold tuning)
5. Generate reports: performance-vs-budget, cost-utility vs lambda, gap plots, subgroup, faithfulness

## Important Decisions
- NZ 6075 pooled file retained unchanged; NOT used for training — HUMAN ACTION REQUIRED path enforced in load_nz()
- Saudi column order A10..A1 explicitly remapped to A1..A10; provenance raw_screening_score preserved
- Polish qchat4 11.0 treated as invalid data value → MISSING (NaN + missing_mask True) and logged to _POLISH_INVALID_LOG, not silently converted — per instruction item 7
- Polish categorical encoding uses per-item sorted vocab excluding 11.0; actual m_list recorded vs canonical 3081146397
- Primary state remains questions_only; age/sex excluded from policy state; subgroup via eval/subgroup.py

## Files Changed
- src/data/ingest.py (real Saudi/Polish/UCI loaders, provenance, 11.0 handling, m_list vocab)
- src/data/schema.py (unchanged but validated)
- AGENT_PROGRESS.md

## Tests Last Run
- command: python -m pytest tests -q
- result: 26 passed

## Last Known Good State
- 2026-08-30: non-NZ infrastructure complete, real-data ingest validated, NZ flagged missing, tests green
