"""Live mentor demo — composes EXISTING research modules only, no new research logic.

Pipeline shown (matches spec sections 10 / 12 / 14 / 16 / 18 / 19):
  data -> preprocessing/encoding -> predictor training -> live episode
  (policy picks which questions to ask) -> decision + counterfactual
  explanation -> policy comparison on the test set -> exact DP optimum.

Run from repo root:
  python scripts/demo_live.py               # auto demo (~2-3 min on CPU)
  python scripts/demo_live.py --interview   # additionally, interactive interview

Terminology lock: screening / referral recommendation ONLY, never diagnosis.
Label-circularity status is displayed next to results (section 16.1 gate).
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# A Windows console defaults to cp1252, which cannot encode the section-sign and
# arrow characters this walkthrough prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
os.chdir(REPO)

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.data.ingest import load_dataset
from src.audits.circularity import audit_circularity
from src.env.state import init_state, update_state, get_legal_items
from src.env.environment import run_episode
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy
from src.explain.counterfactual import find_counterfactual
from src.explain.outcome import explain_outcome
from src.explain.render import render_text
from src.env.costs import get_costs
from src.solvers.exact_custom import ExactDP

BUDGET = 6          # max questions per episode (configs/config.yaml: env.question_budget)
TAU = 0.5           # frozen referral threshold (configs/config.yaml: eval.operating_threshold)
SEED = 0
N_ITEMS = 10
# Platt-on-logit calibration keeps the live belief trace continuous; isotonic
# saturates to exactly 0/1 on this circular dataset (see scripts/demo_app.py).
DEMO_CALIBRATION = "platt"


def hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def split(records):
    """Same deterministic 4-fold stratified split scheme as scripts/step2."""
    y = np.array([r["label"] for r in records])
    idx = np.arange(len(records))
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    tr, te = next(skf.split(idx, y))
    tr, va = next(skf.split(tr, y[tr]))
    return [records[i] for i in tr], [records[i] for i in va], [records[i] for i in te]


def terminal_eval(predictor, records):
    """Predict with all 10 answers revealed (the 'ask everything' reference)."""
    ps, ys = [], []
    for rec in records:
        s = init_state(rec)
        for j in range(N_ITEMS):
            if not rec["missing_mask"][j]:
                s = update_state(s, j, int(rec["item_responses"][j]), 0)
        s["budget"] = N_ITEMS
        s["questions_remaining"] = 0
        ps.append(predictor(s))
        ys.append(rec["label"])
    ps, ys = np.array(ps), np.array(ys)
    return (float(brier_score_loss(ys, ps)), float(roc_auc_score(ys, ps)))


def policy_eval(predictor, policy, records):
    """Run full adaptive episodes on every test record, return (brier, UAR, avg questions)."""
    ps, ys, nq = [], [], []
    for rec in records:
        ep = run_episode(rec, BUDGET, 0, policy, predictor, 0.0, tau=TAU)
        ps.append(ep["p_hat"])
        ys.append(rec["label"])
        nq.append(len(ep["items_asked"]))
    ps, ys = np.array(ps), np.array(ys)
    uar = balanced_accuracy_score(ys, (ps >= TAU).astype(int))
    return (float(brier_score_loss(ys, ps)), float(uar), float(np.mean(nq)))


def main() -> int:
    np.random.seed(SEED)

    # ------------------------------------------------------------- STEP 1
    hr("STEP 1 - LOAD DATA (real CSVs from data/raw/)")
    records = load_dataset("saudi")
    y = np.array([r["label"] for r in records])
    n_missing = int(sum(int(r["missing_mask"].sum()) for r in records))
    print(f"dataset          : Saudi toddler screening (Q-CHAT-10, 10 items)")
    print(f"records          : {len(records)}")
    print(f"class balance    : {int(y.sum())} positive / {int((1 - y).sum())} negative")
    print(f"missing entries  : {n_missing} (handled as a separate state, not dropped)")

    # ------------------------------------------------------------- STEP 2
    hr("STEP 2 - DATA AUDIT: is the label just a score cutoff?")
    audit = audit_circularity(records)
    print(f"best threshold   : score >= {audit['best_threshold']}")
    print(f"exact match rate : {audit['exact_match_rate']:.4f}")
    print(f"classification   : {audit['classification']}")
    print("what this means  : the dataset's Class column is derived from the total")
    print("                   score. We report this openly as a data limitation.")

    # ------------------------------------------------------------- STEP 3
    hr("STEP 3 - SPLIT (stratified, seed fixed for reproducibility)")
    train, val, test = split(records)
    print(f"train {len(train)} / validation {len(val)} / test {len(test)}")
    print("the Polish cohort (252) is deliberately NOT touched in this demo")

    # ------------------------------------------------------------- STEP 4
    hr("STEP 4 - TRAIN PREDICTOR (neural net, random question masking)")
    print("architecture: input(41) -> 128 -> 64 -> 1 (sigmoid), Platt calibration (on logits)")
    t0 = time.time()
    predictor = MaskedPredictor(n_items=N_ITEMS, hidden=[128, 64], calibration=DEMO_CALIBRATION)
    predictor.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=SEED)
    predictor.fit_calibrator(val, method=DEMO_CALIBRATION)
    print(f"trained in {time.time() - t0:.1f}s (CPU)")

    # ------------------------------------------------------------- STEP 5
    hr("STEP 5 - TEST EVALUATION (all 10 questions revealed - the ceiling)")
    brier, auroc = terminal_eval(predictor, test)
    print(f"test Brier score : {brier:.4f}   (lower is better, 0 = perfect)")
    print(f"test AUROC       : {auroc:.4f}   (0.5 = chance, 1.0 = perfect)")
    print("note: this is high because of the label circularity found in STEP 2;")
    print("      these are algorithm-development numbers, NOT clinical accuracy.")

    # ------------------------------------------------------------- STEP 6
    hr(f"STEP 6 - LIVE ADAPTIVE EPISODE (budget B={BUDGET}, greedy policy picks questions)")
    rec = test[0]
    print(f"test record      : true class = {'ASD-positive' if rec['label'] else 'ASD-negative'}")
    print(f"                   (hidden from the policy; used only to score afterwards)")
    ep = run_episode(rec, BUDGET, 0, GreedyIGPolicy(train, N_ITEMS), predictor, 0.0, tau=TAU)
    print(f"\n  step | asked | answer | P(ASD) before -> after")
    print("  -----+-------+--------+-----------------------")
    for t in ep["trace"]:
        print(f"   {t['step']:>3} |  {t['item']:<3}  |   {t['value']}    |  "
              f"{t['belief_before']:.3f} -> {t['belief_after']:.3f}")
    print(f"\nstop reason       : {ep['stop_reason']}")
    print(f"questions asked   : {len(ep['items_asked'])} of 10")
    print(f"final risk score  : {ep['p_hat']:.3f}")
    print(f"DECISION          : {ep['decision']}  (threshold tau = {TAU})")

    cf = find_counterfactual(ep["final_state"], ep["p_hat"], predictor, tau=TAU)
    if cf is None:
        print("explanation       : robust - flipping any single answer would NOT")
        print("                    change the referral decision")
    else:
        print(f"explanation       : if answer to {cf['item']} were {cf['flipped_value']} instead of "
              f"{cf['original_value']}, risk would move to {cf['new_p']:.3f} and the")
        print(f"                    decision would flip - this is a minimal counterfactual")

    # ------------------------------------------------------------- STEP 6b
    hr("STEP 6b - OUTCOME EXPLANATION (exact group-Shapley over the answers given)")
    print("why the model produced this screening result, not which question it asked:\n")
    X_train = np.array([np.asarray(r["item_responses"], dtype=float) for r in train])
    explanation = explain_outcome(
        ep, predictor, tau=TAU, costs=get_costs(N_ITEMS), prior=None,
        reference_rows=X_train, dataset_tag="saudi",
        circularity=audit["classification"],
        label_source=rec["label_source"], budget=BUDGET,
        tag="terminal demo — research prototype, not a diagnostic device")
    print(render_text(explanation))

    # ------------------------------------------------------------- STEP 7
    hr(f"STEP 7 - POLICY COMPARISON ON THE FULL TEST SET (B={BUDGET} episodes)")
    print("running greedy policy on all test records...", flush=True)
    g = policy_eval(predictor, GreedyIGPolicy(train, N_ITEMS), test)
    print("running random policy on all test records...", flush=True)
    r = policy_eval(predictor, RandomPolicy(seed=SEED), test)
    print(f"{'policy':<12} {'Brier':>8} {'UAR':>8} {'avg questions':>15}")
    print(f"{'greedy IG':<12} {g[0]:>8.4f} {g[1]:>8.4f} {g[2]:>15.2f}")
    print(f"{'random':<12} {r[0]:>8.4f} {r[1]:>8.4f} {r[2]:>15.2f}")
    # This comparison is NOT at a matched budget. RandomPolicy draws uniformly
    # from the legal actions, which include STOP, so it ends some episodes early
    # (measured on the demo test split: mean 3.93 questions, ~9% ask nothing,
    # ~45% reach the full budget). Reporting a single "same budget" line here
    # overstated the result: part of the gap is that random answered fewer
    # questions. The matched-budget comparison is in POLICY_BENCHMARK_REPORT.md
    # §A.4, and it is the one that matters.
    print()
    print(f"NOTE: not a matched-budget comparison - random asked {r[2]:.2f} questions on")
    print(f"      average vs greedy's {g[2]:.2f}, because the random baseline can also")
    print("      choose STOP. Part of this gap is fewer questions, not better selection.")
    print("      Matched-budget result: POLICY_BENCHMARK_REPORT.md §A.4.")

    # ------------------------------------------------------------- STEP 8
    hr("STEP 8 - EXACT OPTIMUM (backward-induction DP on a 50-record subsample)")
    dp = ExactDP(train[:50], n_items=N_ITEMS, budget=4, b_min=0, lambda_cost=0.0)
    res = dp.solve()
    print(f"status            : {res['status']}")
    print(f"optimal value V*  : {res['V_star']:.4f}  (best achievable expected reward, max 1)")
    print(f"states explored   : {res['n_states']}")
    print(f"time              : {res['time_sec']:.2f}s")
    print("the exact solver computes the best possible questioning strategy;")
    print("policies like greedy are compared against this reference.")

    hr("DEMO COMPLETE")
    print("This system is a screening / referral-recommendation research prototype.")
    print("It is NOT a diagnosis tool and has no clinical validation.")
    return 0


def interview(predictor, train) -> None:
    """Interactive: the system interviews a human, asking which questions it chooses."""
    hr("INTERACTIVE INTERVIEW (prototype demo - NOT a medical screening)")
    print("Answer each question with 1 (yes/atypical) or 0 (no/typical).")
    print("Type s to stop early (allowed after 1 question). Max "
          f"{BUDGET} questions.\n")
    policy = GreedyIGPolicy(train, N_ITEMS)
    state = {"mask": np.zeros(N_ITEMS, dtype=int),
             "value": np.full(N_ITEMS, -1, dtype=int),
             "n": N_ITEMS, "questions_remaining": BUDGET, "budget": BUDGET}
    asked = 0
    while asked < BUDGET:
        legal = get_legal_items(state)
        if not legal:
            break
        item = policy(state, legal)
        while True:
            raw = input(f"  Q{asked + 1}: item A{item + 1} - answer (0/1/s): ").strip().lower()
            if raw in ("0", "1", "s"):
                break
            print("  please enter 0, 1, or s")
        if raw == "s" and asked >= 1:
            break
        if raw == "s":
            print("  at least one question is needed"); continue
        p_before = predictor(state)
        state = update_state(state, item, int(raw), BUDGET - asked - 1)
        state["questions_remaining"] = BUDGET - asked - 1
        p_after = predictor(state)
        asked += 1
        print(f"    P(ASD risk): {p_before:.3f} -> {p_after:.3f}")
    p = predictor(state)
    print(f"\n  answers given : {asked}")
    print(f"  risk score    : {p:.3f}")
    print(f"  DECISION      : {'REFER' if p >= TAU else 'NO_REFERRAL_INDICATED'} (tau={TAU})")
    cf = find_counterfactual(state, p, predictor, tau=TAU)
    if cf is None:
        print("  explanation   : robust to flipping any single answer")
    else:
        print(f"  explanation   : flipping {cf['item']} to {cf['flipped_value']} would flip the decision "
              f"(risk {cf['new_p']:.3f})")
    print("\n  Reminder: research prototype, screening/recommendation only, not diagnosis.")


if __name__ == "__main__":
    rc = main()
    if "--interview" in sys.argv:
        records = load_dataset("saudi")
        train, _, _ = split(records)
        predictor = MaskedPredictor(n_items=N_ITEMS, hidden=[128, 64], calibration=DEMO_CALIBRATION)
        predictor.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=SEED)
        predictor.fit_calibrator(split(records)[1], method=DEMO_CALIBRATION)
        interview(predictor, train)
    raise SystemExit(rc)
