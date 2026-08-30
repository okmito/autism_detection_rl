# Adaptive RL for Explainable Autism Screening

System function: **adaptive screening and specialist-referral recommendation. Not a diagnostic tool.** Every output says SCREENING.

Source of truth: `Master-Project-Specification_FINAL.md`

## Quickstart
```bash
pip install -r requirements.txt
pytest tests -q
```

Synthetic demo (no real data required):
```python
from src.data.ingest import load_dataset
from src.models.masked_predictor import MaskedPredictor
from src.env.environment import run_episode
from src.policies.greedy import GreedyIGPolicy

records = load_dataset("nz", synthetic=True)
# etc. See tests/ for usage
```

## Repository layout — §8
```
src/data/ingest.py, schema.py, dedupe.py
src/audits/circularity.py, leakage.py
src/env/state.py, environment.py, costs.py
src/solvers/exact_custom.py, exact_adapter.py
src/models/masked_predictor.py
src/policies/dqn.py, ppo.py, greedy.py, random_policy.py, irt_cat.py, dqn_cat.py, static_rfe.py, static_fixed.py
src/explain/trace.py, counterfactual.py, shap_baseline.py
src/eval/metrics.py, bootstrap.py, power.py, fwer.py, subgroup.py
src/ablation/runner.py
configs/config.yaml (Hydra)
tests/ (26 tests, all passing)
```

## Key invariants — §9-11
- Three-state encoding: UNASKED(0) / OBSERVED(1) / MISSING(2) — distinct.
- STOP legal only after B_min unless no legal items remain; forced when budget==0.
- Reward: R = (1 - (p_hat - y)^2) - λ·Σc_j ; primary prediction is Brier-based.
- Primary threshold τ=0.5 frozen before Polish external cohort is opened.

## Exact reference — §14
Authoritative solver is `src/solvers/exact_custom.py` (backward induction DP).
External solvers (DL8.5/MurTree/STreeD) are cross-checks only when objective-compatible.

State counts verified: n=10,B=6 → 26,025 ; Q-CHAT-25 canonical (24×5 + 1×6) → 3,081,146,397.

## Data — §15
Real datasets require V-1 human action (place files under data/raw/). Loaders support `synthetic=True` for testing.
Q-CHAT-10 binary mapping: Q1-9 Sometimes/Rarely/Never→1 ; Q10 Always/Usually/Sometimes→1 — preserved raw for audit.

## Audits — §16 blocking
- `src/audits/circularity.py` — sum-threshold oracle detection.
- `src/audits/leakage.py` — LeakageTracker + poisoned control.

## Tests — §21
```
pytest tests -q   # 26 tests
```
Covers state encoding, legal actions, budget, state counts, reward bounds, predictor, exact optimality, circularity, leakage, counterfactual, threshold freeze, common evaluator, trace, fixed subset.

## Reproducibility
- configs/config.yaml records budget, lambda_grid, seeds, thresholds.
- Spec requires config hash + git SHA + seed per MLflow run (stub ready for integration).

## Pending verification gates — §24
V-1, V-2, V-4, V-5, V-6, V-7, V-9, V-10 remain NOT VERIFIED until human-supervised completion. No novelty claim uses "first/only" until V-2 logged.

## Limitations — §27
Questionnaire-derived labels (except Polish clinical), order-invariance assumption, small Polish cohort, no clinician-in-loop, no participatory design — see spec.

## Terminology lock — §25
Use screening / referral recommendation / risk estimate ; never diagnosis.
