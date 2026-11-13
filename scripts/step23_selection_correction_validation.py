"""Step 23 — validate the selection correction by nested cross-validation (before vs after).

BEFORE: the pre-registered procedure (cost chosen from ONE inner 3-fold split).
AFTER:  the same rule, margin, grid, agent and belief, with the inner estimate averaged over R fold assignments
        (src/rl/selection.py; repeat 0 = the protocol split, so BEFORE is computed from the same runs).

For each outer seed, every child is tested once on an outer fold whose training data chose the cost. Reports, per
procedure: BA/sens/spec/AUROC/Brier, the paired bootstrap CI of the BA difference vs the full questionnaire, mean and
median questions, stopping distribution, quality at each stopping point vs the full questionnaire on the same children.
POST-HOC analysis (the correction was motivated by step 21): it does not replace the pre-registered step-19 result.
Writes results/rl_adaptive/selection_correction_validation.json. Research prototype; not a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
np.seterr(over="ignore")

from src.rl.baselines import FixedOrder  # noqa: E402
from src.rl.data import load_polish_qchat10  # noqa: E402
from src.rl.evaluate import rollout, screening_metrics  # noqa: E402
from src.rl.experiment import Study, stratified_folds  # noqa: E402
from src.rl.selection import repeated_inner_table  # noqa: E402

OUT = REPO / "results" / "rl_adaptive"


def _ba_boot(p, y, boots):
    d = p[boots] >= 0.5
    pos = y[boots] == 1
    return ((d & pos).sum(1) / np.maximum(pos.sum(1), 1) + (~d & ~pos).sum(1) / np.maximum((~pos).sum(1), 1)) / 2


def summarise(y, p, q, full_p):
    m = screening_metrics(y, p, q)
    boots = np.random.default_rng(7).integers(0, len(y), size=(5000, len(y)))
    d = _ba_boot(p, y, boots) - _ba_boot(full_p, y, boots)
    mf = screening_metrics(y, full_p, np.full(len(y), 10))
    df, da = full_p >= 0.5, p >= 0.5
    per_stop = {}
    for k in range(1, 11):
        s = q == k
        if s.any():
            per_stop[str(k)] = {"n": int(s.sum()), "agent_correct": int(np.sum(da[s] == y[s])),
                                "full_correct_same_children": int(np.sum(df[s] == y[s])),
                                "agree_with_full": int(np.sum(da[s] == df[s]))}
    return {"ba": m["balanced_accuracy"], "sens": m["sensitivity"], "spec": m["specificity"], "auroc": m["auroc"],
            "brier": m["brier"], "mean_q": float(q.mean()), "median_q": float(np.median(q)),
            "stop_distribution": {str(k): float(np.mean(q == k)) for k in range(1, 11)},
            "delta_ba": m["balanced_accuracy"] - mf["balanced_accuracy"],
            "delta_ba_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_ba_noninferior_0.05": float(np.mean(d <= -0.05)),
            "delta_auroc": m["auroc"] - mf["auroc"], "delta_brier": m["brier"] - mf["brier"],
            "agreement_with_full": float(np.mean(da == df)), "per_stopping_point": per_stop}


def one_seed(args):
    seed, repeats = args
    P = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    P["seed"] = seed
    X, y = load_polish_qchat10()
    S = Study(X, y, P, include_dqn=False)
    N, n = X.shape
    outer = stratified_folds(y, P["outer_folds"], seed)
    res = {k: {"p": np.zeros(N), "q": np.zeros(N), "costs": []} for k in ("before", "after")}
    full_p = np.zeros(N)
    for o in range(P["outer_folds"]):
        tr, te = outer != o, outer == o
        inner = repeated_inner_table(S, X[tr], y[tr], repeats)
        single = inner["per_repeat"][0]
        before, _ = S.select(single["table"], single["full_ba"])
        after, _ = inner["selected"]
        b = S.belief(X[tr], y[tr])
        full_p[te] = rollout(FixedOrder(range(n), n), X[te], b, S.bcfg)["p"]
        for key, c in (("before", before), ("after", after)):
            r = rollout(S.vi(b, c), X[te], b, S.cfg)
            res[key]["p"][te], res[key]["q"][te] = r["p"], r["n_q"]
            res[key]["costs"].append(float(c))
    out = {"outer_seed": seed, "full_questionnaire": screening_metrics(y, full_p, np.full(N, n))}
    for key in res:
        out[key] = {"selected_cost_per_fold": res[key]["costs"],
                    **summarise(y, res[key]["p"], res[key]["q"], full_p)}
    out["full_questionnaire"] = {k: out["full_questionnaire"][k] for k in
                                 ("balanced_accuracy", "sensitivity", "specificity", "auroc", "brier")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    P = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    seeds = [P["seed"] + 1000 * s for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=min(len(seeds), 8)) as ex:
        per_seed = list(ex.map(one_seed, [(s, args.repeats) for s in seeds]))
    for r in per_seed:
        print(r["outer_seed"], " | ".join(f"{k}: c={r[k]['selected_cost_per_fold']} Q={r[k]['mean_q']:.2f} "
                                          f"BA={r[k]['ba']:.3f} dBA={r[k]['delta_ba']:+.3f}" for k in ("before", "after")))
    art = {"artifact": "rl_adaptive_selection_correction_validation", "status": "POST-HOC (motivated by step 21)",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inner_repeats": args.repeats,
           "per_seed": per_seed, "runtime_seconds": round(time.time() - t0, 1), "contains_participant_rows": False}
    (OUT / "selection_correction_validation.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
