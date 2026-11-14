"""Phase 2 instrumentation: trace hybrid decisions on TRAIN/VAL ONLY.

Minimal, aggregate-only instrumentation (no individual identifiers).
Reuses existing scores()/predictive_sd()/EVOI and the env's legal-action
and budget rules. Records per-decision: n_asked, predictor p,
predictive_sd (epistemic), theta_eap, best EVOI, best net score,
selected action, stop reason classification, support size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.ingest import load_dataset  # noqa: E402
from src.data.splits import stratified_split  # noqa: E402
from src.models.logistic_predictor import LogisticPredictor  # noqa: E402
from src.models.bayesian_logistic import BayesianLogisticPredictor  # noqa: E402
from src.policies.irt_cat import IRTCATPolicy  # noqa: E402
from src.policies.hybrid_irt_evoi import HybridIRT_EVOIPolicy, STOP  # noqa: E402
from src.env.state import init_state, update_state, get_legal_items  # noqa: E402

BUDGET = 6


def classify_stop(reason) -> str:
    return reason


def trace(records, policy, max_records=None):
    rows = []
    recs = list(records) if max_records is None else list(records)[:max_records]
    for rec in recs:
        state = init_state(rec)
        state["budget"] = BUDGET
        state["questions_remaining"] = BUDGET
        items_asked = []
        while True:
            legal_items = get_legal_items(state)
            stop_legal = len(items_asked) >= 0
            legal = legal_items.copy() + [STOP]
            s = policy.scores(state, legal)
            action = policy(state, legal)
            n_asked = len(items_asked)
            # EVOI signs at this decision point
            best_evoi = max(s["evoi"].values()) if s["evoi"] else 0.0
            best_net = max(s["net_score"].values()) if s["net_score"] else 0.0
            row = {
                "n_asked": n_asked,
                "predictor_p": s["predictor_probability"],
                "predictive_sd": s["predictive_sd"],
                "theta_eap": s["theta_eap"],
                "support_size": s["support_size"],
                "posterior": s["posterior"],
                "best_evoi": float(best_evoi),
                "best_net_score": float(best_net),
                "action": "STOP" if action == STOP else f"A{action+1}",
                "questions_remaining": state.get("questions_remaining", BUDGET - n_asked),
            }
            if action == STOP:
                if n_asked >= policy.max_questions:
                    row["stop_reason"] = "budget_cap"
                elif best_net <= 0 and s["predictive_sd"] <= policy.abstain_sd:
                    row["stop_reason"] = "evoi_nonpositive_and_low_uncertainty"
                elif best_net <= 0 and policy.uncertainty_stop_sd is not None and s["predictive_sd"] > policy.uncertainty_stop_sd:
                    row["stop_reason"] = "uncertainty_gate"
                elif best_net <= 0:
                    row["stop_reason"] = "evoi_nonpositive"
                else:
                    row["stop_reason"] = "other"
                rows.append(row)
                break
            state = update_state(state, int(action), int(rec["item_responses"][int(action)]), state["questions_remaining"] - 1)
            state["questions_remaining"] -= 1
            state["budget"] = BUDGET
            items_asked.append(int(action))
            rows.append(row)
            if len(items_asked) >= BUDGET + 1:
                break
    return rows


def main():
    recs = load_dataset("saudi")
    train, val, _ = stratified_split(recs, seed=0)
    base = LogisticPredictor.fit(train, val, feature_order=[f"A{i}" for i in range(1, 11)], seed=0)
    bayes = BayesianLogisticPredictor(base, train, n_samples=100, seed=0)
    irt = IRTCATPolicy(train, n_items=10)
    pol = HybridIRT_EVOIPolicy(bayes, irt, train, 10, lambda_cost=0.0)
    print(f"train={len(train)} val={len(val)} (tracing VAL only for validation; train support used for EVOI as in production)")
    rows = trace(val, pol)
    # aggregate by n_asked
    from collections import defaultdict
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n_asked"]].append(r)
    print(f"\ndecisions traced: {len(rows)}")
    for n in sorted(by_n):
        snap = by_n[n]
        sd = np.mean([r["predictive_sd"] for r in snap])
        be = np.mean([r["best_evoi"] for r in snap])
        bnet = np.mean([r["best_net_score"] for r in snap])
        sup = int(np.median([r["support_size"] for r in snap]))
        print(f" n_asked={n}: count={len(snap)} sd={sd:.4f} best_evoi={be:+.5f} best_net={bnet:+.5f} support_median={sup}")
    stops = [r for r in rows if r["action"] == "STOP"]
    from collections import Counter
    print("\nSTOP reasons (val):", Counter(r["stop_reason"] for r in stops))
    print(f"mean best_evoi at stop: {np.mean([r['best_evoi'] for r in stops]) if stops else 0:+.5f}")
    print(f"mean predictive_sd at stop: {np.mean([r['predictive_sd'] for r in stops]) if stops else 0:.4f}")
    print(f"abstain_sd={pol.abstain_sd} uncertainty_stop_sd={pol.uncertainty_stop_sd}")
    # also trace on TRAIN (support known) to check EVOI sign distribution
    rows_tr = trace(train, pol, max_records=95)
    stops_tr = [r for r in rows_tr if r["action"] == "STOP"]
    print("\nTRAIN (first 95) STOP reasons:", Counter(r["stop_reason"] for r in stops_tr))
    print(f"mean best_evoi at stop (train): {np.mean([r['best_evoi'] for r in stops_tr]) if stops_tr else 0:+.5f}")
    # support collapse rate
    supports = [r["support_size"] for r in rows_tr]
    print(f"support size: min={min(supports)} median={int(np.median(supports))} max={max(supports)}")
    Path("results").mkdir(exist_ok=True)
    Path("results/hybrid_stopping_audit.json").write_text(json.dumps(
        {"val_decisions": len(rows), "val_stops": len(stops),
         "val_stop_reasons": Counter(r["stop_reason"] for r in stops),
         "val_mean_best_evoi_at_stop": float(np.mean([r["best_evoi"] for r in stops])) if stops else None,
         "val_mean_predictive_sd_at_stop": float(np.mean([r["predictive_sd"] for r in stops])) if stops else None,
         "abstain_sd": pol.abstain_sd}, indent=2), encoding="utf-8")
    print("\nWrote results/hybrid_stopping_audit.json")


if __name__ == "__main__":
    main()
