# Current Project State

## Completed
- Repository scaffold per §8 — all src/*, configs, tests, data/, results/ created
- Data layer: schema.py, ingest.py (6 loaders + synthetic mode + Q-CHAT-10 binary map §15), dedupe.py — NOT using 6075-row pooled NZ file as primary cohort
- NZ 1,054 cohort searched: final-with-99-accuracy.ipynb cell 2 references Toddler Autism dataset July 2018.csv (1054 rows, Case_No/A1-A10/Age_Mons/Qchat-10-Score/Sex/...) — file NOT present locally; no 1054-row provenance established (searched Todo: Toddler*, data_csv.csv, autism_screening.csv — 0 hits); 6075 file retained unchanged
- Saudi ingestion finalized: 506 rows verified (A10..A1 reversed → corrected to A1..A10), Screening Score == sum(A) 506/506, Class deterministic >=4, schema OK, circularity Deterministic; raw_screening_score preserved for audit
- Polish ingestion finalized: 252 rows verified (135 ASD / 117 control → resolves 252/253 discrepancy: 117 controls, not 118), 25 qchat categorical strings, per-item vocab sorted encoding (excluding 11.0), invalid qchat4 value 11.0 (row 60 child_id bdbp0221) logged explicitly as MISSING per instruction §7, UserWarning emitted, schema OK, circularity Not circular
- UCI Child finalized: 292 rows verified via ARFF (90 ? markers) matches UCI ID 419; adol/adult remain HUMAN ACTION REQUIRED pending ARFFs
- Environment: three-state encoding §9 distinct, costs.py, episode loop §10 with STOP/budget/missing; input dim 4n+1 binary
- Predictor: MaskedMLP 128→64 sigmoid §12 + random-mask training + fold-local calibration (isotonic/Platt) — predictor-only smoke tests on Saudi/UCI completed without λ (V-6 independent)
- Exact DP authoritative §14.1 with tractability limits 50M/24h/32GB — theoretical counts programmatically verified; empirical counts distinguished (Saudi 100-rec B=6 → 16826 states vs theoretical 26025)
- Adapter stub §14.2 compatibility check (DL8.5/MurTree not Brier-compatible)
- Policies: DQN, PPO, Greedy IG, Random, IRT-CAT, DQN-CAT, RFE, Exact Fixed Subset (§17) — functional baselines verified on Saudi train/test split B=3/6 (fixed-length evaluation, no Polish)
- Explain: trace §18.1, counterfactual §18.2, SHAP baseline §18.3 — smoke tests OK (counterfactual robust/ flips, SHAP on terminal subset)
- Eval: metrics (Brier/UAR/AUROC/ECE etc §19.1), paired bootstrap 2000 §19.2, power/MDE §19.3, Holm-Bonferroni §19.3, subgroup §19.4 — subgroup smoke on Saudi OK
- Config: configs/config.yaml §13 — predictor.hidden [128,64], freeze true, τ=0.5, primary_metric brier, policy_state questions_only all PASS; dataset tier/name still nz (blocked) per spec, env cost_mode uniform (primary)
- Tests: 26 tests covering §21 — all passing (4.02s) + real-data ingestion/audit/predictor/DP/baseline smoke tests (labelled by type)
- Documentation: DATA_VERIFICATION_REPORT.md (2026-08-30), STATE_COUNT_VERIFICATION.md (theoretical vs empirical), AUDIT_REPORT.md (circularity/leakage/invalid handling)

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
- result: 26 passed (4.02s)
- command: python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))" → 26025 [THEORETICAL]
- command: python3 -c "from src.data.ingest import load_dataset; from src.audits.circularity import audit_circularity; print(audit_circularity(load_dataset('saudi')))" → Deterministic thr 4 [REAL-DATA AUDIT]
- command: python3 -c "from src.models.masked_predictor import MaskedPredictor; ..." → [PREDICTOR SMOKE TEST] Saudi/UCI brier/ece reported (fold-local, no λ)
- command: python3 -c "from src.solvers.exact_custom import ExactDP; ..." → [EMPIRICAL FINITE-SAMPLE] Saudi 100-rec B=6 → 16826 states (not theoretical)

## Last Known Good State
- 2026-08-30: non-NZ infrastructure complete, real-data ingest validated, theoretical vs empirical distinguished, NZ flagged missing, tests green, Polish isolated (V-4/V-7 pending, no tuning)
