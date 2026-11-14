"""Lower-variance estimate for the pre-registered cost selection (value-iteration agent).

The protocol selects the question cost ``c`` as the fewest mean questions whose inner-CV balanced accuracy is within
``delta_select`` of the full questionnaire, estimated from ONE inner 3-fold split. Step 21 showed that on the Polish
cohort this single-split estimate is noisy enough to accept a cost that violates the rule on average (c = 0.05:
-0.013 on the split used, -0.021 averaged over ten splits; it met the rule in 3/10 splits).

``repeated_inner_table`` keeps the rule, the margin, the grid, the inner fold count, the belief model and the agent
unchanged and averages the inner-CV estimate over ``repeats`` fold assignments; repeat 0 is exactly the protocol's
split, so ``repeats=1`` reproduces the original selection. Selection itself is delegated to ``Study.select``.
Not part of the frozen protocol code hash (see RL_ADAPTIVE_REPORT.md, stopping diagnosis).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.rl.baselines import FixedOrder
from src.rl.evaluate import rollout, screening_metrics
from src.rl.experiment import stratified_folds

SEED_STEP = 7919          # fold seed for repeat r is protocol seed + 1 + SEED_STEP * r (r = 0 is the protocol split)


def repeated_inner_table(study, X: np.ndarray, y: np.ndarray, repeats: int) -> Dict[str, Any]:
    """Mean over ``repeats`` inner fold assignments of the pooled inner-CV metrics per cost (value iteration)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    P, n = study.P, study.cfg.n_items
    X, y = np.asarray(X, float), np.asarray(y, int)
    costs = list(P["cost_grid"])
    per_repeat = []
    for rep in range(repeats):
        folds = stratified_folds(y, P["inner_folds"], P["seed"] + 1 + SEED_STEP * rep)
        full_p = np.zeros(len(y))
        pp = {c: np.zeros(len(y)) for c in costs}
        qq = {c: np.zeros(len(y)) for c in costs}
        for f in range(P["inner_folds"]):
            tr, te = folds != f, folds == f
            b = study.belief(X[tr], y[tr])
            full_p[te] = rollout(FixedOrder(range(n), n), X[te], b, study.bcfg)["p"]
            for c in costs:
                r = rollout(study.vi(b, c), X[te], b, study.cfg)
                pp[c][te], qq[c][te] = r["p"], r["n_q"]
        full = screening_metrics(y, full_p, np.full(len(y), n))
        tab = {}
        for c in costs:
            m = screening_metrics(y, pp[c], qq[c])
            tab[c] = {"balanced_accuracy": m["balanced_accuracy"], "mean_questions": m["mean_questions"],
                      "auroc": m["auroc"]}
        per_repeat.append({"full_ba": full["balanced_accuracy"], "table": tab})
    full_ba = float(np.mean([r["full_ba"] for r in per_repeat]))
    table = {c: {k: float(np.mean([r["table"][c][k] for r in per_repeat]))
                 for k in ("balanced_accuracy", "mean_questions", "auroc")} for c in costs}
    d = study.P["selection"]["delta_select"]
    meets = {c: int(sum(r["table"][c]["balanced_accuracy"] >= r["full_ba"] - d for r in per_repeat)) for c in costs}
    sel, rule = study.select(table, full_ba)
    return {"repeats": repeats, "full_ba": full_ba, "table": table, "meets_rule_in_n_repeats": meets,
            "selected": (sel, rule), "per_repeat": per_repeat}
