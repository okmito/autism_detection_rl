# Current Project State

## Completed
- Repository scaffold per §8 (all src/*, configs, tests, data/, results/ created)
- Data layer: schema.py, ingest.py (6 loaders + synthetic mode + Q-CHAT-10 binary map §15), dedupe.py
- Circularity audit (§16.1) + leakage audit (§16.2) with LeakageTracker + poisoned control
- Environment: three-state encoding §9 (UNASKED/MISSING/OBSERVED distinct), costs.py, episode loop §10 with STOP/budget/missing semantics
- Predictor: MaskedMLP 128→64 sigmoid (§12) + random-mask training + isotonic/Platt calibration
- Exact solver: authoritative backward-induction DP §14.1 with empirical support, lambda cost, tractability limits (50M states/24h/32GB)
- Adapter stub §14.2 with compatibility check (DL8.5/MurTree not compatible for Brier)
- Policies: DQN (§11.2), PPO, Greedy IG, Random, IRT-CAT, DQN-CAT, Static RFE, Exact Fixed Subset (§17 10 baselines)
- Explain: trace §18.1, counterfactual §18.2, SHAP baseline §18.3
- Eval: metrics (Brier/AUROC/UAR/ECE etc §19.1), paired bootstrap §19.2, power/MDE §19.3, Holm-Bonferroni FWER, subgroup §19.4
- Ablation runner §20 (7 ablations)
- Config: configs/config.yaml §13
- Tests: 26 tests covering §21 table — all passing

## In Progress
- None

## Verified
- 26 tests passed (pytest 22s): state_encoding, legal_actions, budget_exhaustion, reachable_state_count (26025 and 3081146397 and ADI-R), reward_bounds, predictor_partial_input, exact_optimality, exact_value_consistency, no_leakage, circularity_oracle, counterfactual_validity, threshold_freeze, common_empirical_evaluator, trace_belief_update, exact_fixed_subset

## Failed
- None

## Blocked
- V-1 dataset licences/data download — HUMAN ACTION REQUIRED to place NZ/Saudi/Polish CSVs under data/raw/
- V-2 systematic literature search log
- V-4 MDE/comparison family freeze
- V-5 order-effects evidence
- V-6 lambda grid sign-off
- V-7 Polish denominator resolution
- V-9 ethics review
- V-10 Tier-3 go/no-go

## Next Required Steps
1. Human provides datasets (V-1) → run ingestion + circularity/leakage audits
2. Complete V-2 PRISMA search log before any novelty wording
3. Train predictor with real data, run exact DP at B=3-6 (tractability curves), train DQN/PPO
4. Freeze evaluation protocol + MDE (V-4) before opening Polish cohort
5. External evaluation + ablations + reporting (performance-vs-budget, cost-utility, gap plots)

## Important Decisions
- Binary Q-CHAT-10 mapping locked per Sollis et al. §15; raw preserved
- Primary state is questions_only (age/sex excluded from exact/learned); subgroup via eval/subgroup.py
- Exact DP treats MISSING as distinct branch discovered on ask; root all UNASKED with support filtering
- Threshold τ=0.5 frozen; polishing external evaluation untuned

## Files Changed
- All src/**/*.py, configs/config.yaml, tests/test_*.py, requirements.txt, .gitignore, README.md

## Tests Last Run
- command: python -m pytest tests -q
- result: 26 passed

## Last Known Good State
- 2026-08-30: scaffold complete, tests green, ready for real data ingest
