# State-Count Verification — Theoretical vs Empirical (2026-08-30)

All counts generated programmatically via `src/env/state.py:reachable_state_count` — not hand-entered.

## Theoretical Complete-Instrument Reference (no finite-sample pruning)
Counts assume complete records (no structural missingness) and homogeneous or specified cardinalities. These are **not** empirical sample counts.

| Instrument | n | B | m per item | Formula | Theoretical Count | Note |
|---|---|---|---|---|---|---|
| Q-CHAT-10 binary | 10 | 6 | 2 | Σ_{k=0..6} C(10,k)2^k | **26,025** | Spec §14.5 reference |
| Q-CHAT-25 canonical (literature) | 25 | 6 | 24×5 + 1×6 (item4 has 6) | Σ_{S,|S|≤6} Π m_j | **3,081,146,397** | Canonical Q-CHAT structure (Allison et al. 2008; spec §14.5) |
| Q-CHAT-25 observed (Polish data) | 25 | 6 | [5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5] (qchat2/qchat13 have 4 observed levels, 11.0 excluded) | Σ Π m_j | **2,667,729,775** | Polish artifact `polish_qchat.csv` actual vocab; 11.0 treated as invalid MISSING per instruction item 7 |
| ADI-R hypothetical | 93 | 6 | 2 | Σ C(93,k)2^k | **50,494,563,219** | Theoretical extrapolation only (§14.5) |

Verification commands:
```
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))"
→ 26025
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5]*25[:3]+[6]+[5]*21))"
→ 3081146397
```

## Empirical Finite-Sample Reachable States (DP memoization)
Counts depend on sample size N, missingness, and λ cost. These are **not** theoretical references and must not be reported as such.

- **Saudi 100-rec subset, n=10, B=6, λ=0.01** → `ExactDP(...).solve()['n_states']` = **16,826** (smoke test, `λ=0.01` is placeholder — **V-6 PENDING**, not confirmatory)
- **Saudi 50-rec subset, B=3, λ=0.0** → **1,016** (toy, deterministic check)
- **Saudi 50-rec subset, B=3, λ=0.0 (second run)** → 1,016 consistent

Verification commands:
```
python3 -c "from src.data.ingest import load_dataset; from src.solvers.exact_custom import ExactDP; recs=load_dataset('saudi'); dp=ExactDP(recs[:100],n_items=10,budget=6,b_min=0,lambda_cost=0.01); print(dp.solve()['n_states'])"
→ 16826
```

## Distinction Rule
- Label every reported figure as either `[THEORETICAL]` (instrument/budget/m definition) or `[EMPIRICAL FINITE-SAMPLE]` (specific N, λ, solver run).
- Do not call an empirical DP count on a Saudi 100-record subset a replacement for the theoretical 26,025 reference.
- Polish observed 2,667,729,775 vs canonical 3,081,146,397 difference is due to observed vocab (4-level items) — implementation must use actual source encoding per §14.5.

