# State-Count Verification — Theoretical vs Empirical (2026-09-04; re-verified 2026-10-01)

All counts generated programmatically via `src/env/state.py:reachable_state_count` — not hand-entered.
The most recent sweep was run on 2026-09-04; the prior 2026-08-30 numbers are preserved in
the "Prior smoke test" subsection for reference.

> **Updated 2026-10-01.** All counts re-verified after the datasets were re-fetched from scratch
> (`data/raw/` had been lost — it is gitignored). Every value reproduces; see
> §"2026-10-01 re-verification". Test count corrected 43 → 79.

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

Empirical sweep results (Saudi, **real CSV**, re-run 2026-09-30 after the raw
datasets were confirmed present at `src/data/ingest.py:SAUDI_CSV`; every row carries
`source=real` in `results/dp_tractability_sweep.{json,csv}`). Values below are the
λ=0.00 runs; the λ=0.01 runs have identical state/eval counts:

| N | B=3 | B=4 | B=5 | B=6 |
|---:|---:|---:|---:|---:|
| 10 | 514 | 1,381 | 2,625 | 3,805 |
| 25 | 959 | 2,896 | 5,889 | 8,865 |
| 50 | 1,016 | 3,268 | 7,043 | 11,081 |
| 100 | 1,144 | 4,159 | 9,981 | 16,826 |
| 250 | 1,159 | 4,474 | 11,971 | 22,446 |
| 506 | 1,161 | 4,519 | 12,501 | 25,023 |

Theoretical complete-record reference: 1,161 (B=3) / 4,521 (B=4) / 12,585 (B=5) /
26,025 (B=6). On real data the empirical counts converge to the theoretical
complete-record counts as N grows (N=506: 1,161 / 4,519 / 12,501 / 25,023 vs
1,161 / 4,521 / 12,585 / 26,025): with the full cohort nearly every item pattern is
observed, so finite-sample counts sit slightly below the abstract bound (a few
mask/value/support combinations are unreachable from 506 records). Smaller N
truncates the reachable set further. A prior **synthetic-fallback** run
(2026-09-04, retained in git history and in `AUDIT_UPDATE_2026-09-04.md`) produced
larger counts (e.g. N=506 B=6 → 72,964) because the synthetic generator's 5%
per-item missingness explodes the reachable mask×value space; that table is
superseded by the real-data sweep above.

The full sweep also reports `n_evals` (Bellman evaluations), `time_sec` (wall clock),
and `V_star` (Bellman value at the root) per run. The largest run (N=506, B=6) takes
~2.2 s on a single CPU thread and reports 25,023 states / 68,408 evaluations — well
inside the `exact.max_states=50M / 24h / 32GB` tractability bound in
`configs/config.yaml:exact`.

Verification commands:
```
python3 -c "from src.data.ingest import load_dataset; from src.solvers.exact_custom import ExactDP; recs=load_dataset('saudi'); dp=ExactDP(recs[:100],n_items=10,budget=6,b_min=0,lambda_cost=0.01); print(dp.solve()['n_states'])"
→ 16826   [EMPIRICAL — now reproduces exactly as the N=100, B=6 cell of the real-data sweep above]
python3 scripts/step2_train_and_sweep.py
→ results/dp_tractability_sweep.json  (48 runs)
→ results/dp_tractability_sweep.csv   (CSV summary)
python3 scripts/step5_policy_benchmark.py --episodes 400 --budgets 1,2,3,4,5,6
→ results/step5_policy_benchmark_saudi.{json,csv}
   (also solves ExactDP per budget; per-B state counts in `exact_dp` log lines)
python3 -m pytest tests -q
→ 79 passed
```

## 2026-10-01 re-verification

The repo had lost `data/raw/` (gitignored), so all datasets were re-fetched and every count recomputed. Results:

| Count | Expected (this doc) | Recomputed 2026-10-01 | Match |
|---|---|---|---|
| `reachable_state_count(10, 3, m=2)` | 1,161 | 1,161 | ✅ |
| `reachable_state_count(10, 6, m=2)` | 26,025 | 26,025 | ✅ |
| Q-CHAT-25 canonical (24×5 + 1×6) | 3,081,146,397 | 3,081,146,397 | ✅ |
| Q-CHAT-25 observed (Polish) | 2,667,729,775 | 2,667,729,775 | ✅ |
| Saudi DP sweep, 48 cells | all `status=optimal` | all `status=optimal` | ✅ |

**Note on deriving the observed `m_list`:** it is `len(vocabulary)` per item, **not** `len(vocabulary) - 1`. Recomputed from the restored Polish file: `[5,4,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5]` — 4-level items at qchat2 and qchat13, 5-level elsewhere. Using `n_vocab - 1` yields 692,824,784, which does **not** match this document. Use `n_vocab`.

`n_items=10` state counts are unaffected — they use `m=2` (binary), not `m_list`.

### Step 5 exact-DP runs (new, 2026-10-01)
`scripts/step5_policy_benchmark.py` solves ExactDP once per budget on the 284-record Saudi train split, providing the V\* reference for the optimality gap:

| B | V\* | n_states |
|---|---|---|
| 1 | 0.886826 | 21 |
| 2 | 0.927780 | 201 |
| 3 | 0.949005 | 1,160 |
| 4 | 0.973594 | 4,488 |
| 5 | 0.997653 | 12,152 |
| 6 | 1.000000 | 23,216 |

These are `[EMPIRICAL FINITE-SAMPLE]` counts on 284 records, not the theoretical references above.

## Distinction Rule
- Label every reported figure as either `[THEORETICAL]` (instrument/budget/m definition) or `[EMPIRICAL FINITE-SAMPLE]` (specific N, λ, solver run).
- Do not call an empirical DP count on a Saudi 100-record subset a replacement for the theoretical 26,025 reference.
- Polish observed 2,667,729,775 vs canonical 3,081,146,397 difference is due to observed vocab (4-level items at qchat2 and qchat13) — implementation must use actual source encoding per §14.5.

