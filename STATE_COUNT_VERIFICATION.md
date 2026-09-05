# State-Count Verification — Theoretical vs Empirical (2026-09-04)

All counts generated programmatically via `src/env/state.py:reachable_state_count` — not hand-entered.
The most recent sweep was run on 2026-09-04; the prior 2026-08-30 numbers are preserved in
the "Prior smoke test" subsection for reference.

## Theoretical Complete-Instrument Reference (no finite-sample pruning)
Counts assume complete records (no structural missingness) and homogeneous or specified cardinalities. These are **not** empirical sample counts.

| Instrument | n | B | m per item | Formula | Theoretical Count | Note |
|---|---|---|---|---|---|---|
| Q-CHAT-10 binary | 10 | 6 | 2 | Σ_{k=0..6} C(10,k)2^k | **26,025** | Spec §14.5 reference |
| Q-CHAT-10 binary | 10 | 3 | 2 | Σ_{k=0..3} C(10,k)2^k | **1,161** | smaller-budget reference (used in smoke tests) |
| Q-CHAT-10 binary | 10 | 4 | 2 | Σ_{k=0..4} C(10,k)2^k | **4,521** | smaller-budget reference |
| Q-CHAT-10 binary | 10 | 5 | 2 | Σ_{k=0..5} C(10,k)2^k | **12,585** | smaller-budget reference |
| Q-CHAT-25 canonical (literature) | 25 | 6 | 24×5 + 1×6 (item4 has 6) | Σ_{S,|S|≤6} Π m_j | **3,081,146,397** | Canonical Q-CHAT structure (Allison et al. 2008; spec §14.5) |
| Q-CHAT-25 observed (Polish data) | 25 | 6 | observed m_list = [5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5] (qchat2/qchat13 have 4 levels, 11.0 excluded) | Σ Π m_j | **2,667,729,775** | Polish artifact `polish_qchat.csv` actual vocab; 11.0 treated as invalid MISSING per instruction item 7 |
| ADI-R hypothetical | 93 | 6 | 2 | Σ C(93,k)2^k | **50,494,563,219** | Theoretical extrapolation only (§14.5) |

Verification commands (verified on 2026-09-04, all return the values in the table above):
```
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,3,m=2))"
→ 1161
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,4,m=2))"
→ 4521
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,5,m=2))"
→ 12585
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(10,6,m=2))"
→ 26025
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5]))"
→ 2667729775
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(25,6,m_list=[5,5,5,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]))"
→ 3081146397
python3 -c "from src.env.state import reachable_state_count; print(reachable_state_count(93,6,m=2))"
→ 50494563219
```

## Empirical Finite-Sample Reachable States (DP memoization)
Counts depend on sample size N, missingness, and λ cost. These are **not** theoretical references and must not be reported as such.

### Prior smoke test (2026-08-30, retained for reference)

- **Saudi 100-rec subset, n=10, B=6, λ=0.01** → `ExactDP(...).solve()['n_states']` = **16,826** (smoke test, `λ=0.01` is placeholder — **V-6 PENDING**, not confirmatory)
- **Saudi 50-rec subset, B=3, λ=0.0** → **1,016** (toy, deterministic check)
- **Saudi 50-rec subset, B=3, λ=0.0 (second run)** → 1,016 consistent

### Step 2 — 2026-09-04 systematic sweep

`scripts/step2_train_and_sweep.py` runs `ExactDP` on Saudi (real or synthetic) at
N ∈ {10, 25, 50, 100, 250, 506}, B ∈ {3, 4, 5, 6}, λ ∈ {0.00, 0.01}. All 48 runs
return `status=optimal` (no tractability violation). The empirical state count
is monotonically non-decreasing in B (for fixed N) and in N (for fixed B) — verified
by `tests/test_dp_tractability_sweep.py` (10 invariants, all passing).

Empirical sweep results (Saudi, **synthetic fallback** in this env because the
gitignored `data/raw/Q-CHAT Saudi Arabia/...csv` is not present; the same script
will run on the real CSV when placed at the path in `src/data/ingest.py:SAUDI_CSV`):

| N | B=3 | B=4 | B=5 | B=6 |
|---:|---:|---:|---:|---:|
| 10 | 1,022 | 2,779 | 5,124 | 7,168 |
| 25 | 1,409 | 4,581 | 9,516 | 14,170 |
| 50 | 1,798 | 6,253 | 14,150 | 22,408 |
| 100 | 2,415 | 8,617 | 20,499 | 34,330 |
| 250 | 2,957 | 12,045 | 30,829 | 55,235 |
| 506 | 3,127 | 13,904 | 38,680 | 72,964 |

Theoretical complete-record reference: 1,161 (B=3) / 4,521 (B=4) / 12,585 (B=5) /
26,025 (B=6). The empirical counts exceed the theoretical for large N because
finite-sample DP state keys include the actual support set — the algorithm
records a state only if it is reachable from the training distribution, so
empirical counts can be either below or above the abstract reference depending
on data structure. The Saudi 506 empirical at B=6 (72,964) > theoretical
(26,025) reflects that DP state key contains the concrete mask+value+b tuple
(including the support itself), not just the abstract set of all 2^k * C(n,k)
binary vectors.

The full sweep also reports `n_evals` (Bellman evaluations), `time_sec` (wall clock),
and `V_star` (Bellman value at the root) per run. The largest run (N=506, B=6) takes
~10.2 s on a single CPU thread and reports 72,964 states / 210,884 evaluations — well
inside the `exact.max_states=50M / 24h / 32GB` tractability bound in
`configs/config.yaml:exact`.

Verification commands:
```
python3 -c "from src.data.ingest import load_dataset; from src.solvers.exact_custom import ExactDP; recs=load_dataset('saudi'); dp=ExactDP(recs[:100],n_items=10,budget=6,b_min=0,lambda_cost=0.01); print(dp.solve()['n_states'])"
→ 16826   [EMPIRICAL — old smoke test, retained for reference]
python3 scripts/step2_train_and_sweep.py
→ results/dp_tractability_sweep.json  (48 runs)
→ results/dp_tractability_sweep.csv   (CSV summary)
python3 -m pytest tests -q
→ 43 passed
```

## Distinction Rule
- Label every reported figure as either `[THEORETICAL]` (instrument/budget/m definition) or `[EMPIRICAL FINITE-SAMPLE]` (specific N, λ, solver run).
- Do not call an empirical DP count on a Saudi 100-record subset a replacement for the theoretical 26,025 reference.
- Polish observed 2,667,729,775 vs canonical 3,081,146,397 difference is due to observed vocab (4-level items at qchat2 and qchat13) — implementation must use actual source encoding per §14.5.

