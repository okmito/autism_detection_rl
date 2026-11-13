"""Step 22 — EXPLORATORY decision support: what would the constrained cost selection choose under other quality requirements?

The pre-registered selection rule (protocol.json) is: the fewest mean questions whose inner-CV balanced accuracy is
within 0.02 of the full questionnaire. Step 21 found that balanced accuracy at tau saturates after ~3 questions on the
Polish cohort, so that requirement is slack and the selected cost sits at the edge of the grid. This script asks how the
*same* value-iteration agent, the *same* cost grid and the *same* nested CV behave if the quality requirement also
covers other properties (sensitivity, agreement with the full questionnaire, ranking, calibration).

Each rule is applied inside the inner CV of every outer fold; the selected cost is evaluated once on the outer fold.
The thresholds are ILLUSTRATIVE CANDIDATES, not choices: which requirement (and margin) applies is a research decision
for the supervisor. Nothing here changes the served policy, src/rl, or any existing artifact.
Writes results/rl_adaptive/quality_requirement_sweep.json. Research prototype; not a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
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

OUT = REPO / "results" / "rl_adaptive"


def metrics(y, p, q, p_full, tau=0.5):
    m = screening_metrics(y, p, q)
    return {"ba": m["balanced_accuracy"], "sens": m["sensitivity"], "spec": m["specificity"], "auroc": m["auroc"],
            "brier": m["brier"], "mean_q": float(np.mean(q)), "frac_stop_2": float(np.mean(q <= 2)),
            "agreement_with_full": float(np.mean((p >= tau) == (p_full >= tau)))}


# rule name -> predicate(candidate inner metrics, full inner metrics). Margins are illustrative.
RULES = {
    "R0 BA within 0.02 (pre-registered)": lambda m, f: m["ba"] >= f["ba"] - 0.02,
    "R1 R0 + sensitivity within 0.02": lambda m, f: m["ba"] >= f["ba"] - 0.02 and m["sens"] >= f["sens"] - 0.02,
    "R2 same referral decision as full for >= 95%": lambda m, f: m["agreement_with_full"] >= 0.95,
    "R3 R0 + AUROC within 0.02": lambda m, f: m["ba"] >= f["ba"] - 0.02 and m["auroc"] >= f["auroc"] - 0.02,
    "R4 R0 + Brier within 0.01": lambda m, f: m["ba"] >= f["ba"] - 0.02 and m["brier"] <= f["brier"] + 0.01,
}


def choose(table, full, rule):
    ok = [c for c, m in table.items() if rule(m, full)]
    if not ok:                                   # nothing qualifies: ask the most (cost 0 = full questionnaire)
        return min(table), "constraint_unmet"
    return min(ok, key=lambda c: (table[c]["mean_q"], -table[c]["ba"], c)), "constraint_met"


def one_seed(X, y, P, seed):
    P = {**P, "seed": seed}
    S = Study(X, y, P, include_dqn=False)
    N, n = X.shape
    outer = stratified_folds(y, P["outer_folds"], seed)
    out_p = {r: np.zeros(N) for r in RULES}
    out_q = {r: np.zeros(N) for r in RULES}
    full_p = np.zeros(N)
    chosen = {r: [] for r in RULES}
    for o in range(P["outer_folds"]):
        tr, te = outer != o, outer == o
        Xo, yo = X[tr], y[tr]
        inner = stratified_folds(yo, P["inner_folds"], seed + 1)
        ip = {c: np.zeros(len(yo)) for c in P["cost_grid"]}
        iq = {c: np.zeros(len(yo)) for c in P["cost_grid"]}
        ifull = np.zeros(len(yo))
        for f in range(P["inner_folds"]):
            itr, ite = inner != f, inner == f
            b = S.belief(Xo[itr], yo[itr])
            ifull[ite] = rollout(FixedOrder(range(n), n), Xo[ite], b, S.bcfg)["p"]
            for c in P["cost_grid"]:
                r = rollout(S.vi(b, c), Xo[ite], b, S.cfg)
                ip[c][ite], iq[c][ite] = r["p"], r["n_q"]
        full_m = metrics(yo, ifull, np.full(len(yo), n), ifull)
        table = {c: metrics(yo, ip[c], iq[c], ifull) for c in P["cost_grid"]}
        b = S.belief(Xo, yo)
        full_p[te] = rollout(FixedOrder(range(n), n), X[te], b, S.bcfg)["p"]
        cache = {}
        for name, rule in RULES.items():
            c, why = choose(table, full_m, rule)
            if c not in cache:
                cache[c] = rollout(S.vi(b, c), X[te], b, S.cfg)
            out_p[name][te], out_q[name][te] = cache[c]["p"], cache[c]["n_q"]
            chosen[name].append({"cost": c, "rule": why})
    res = {"full_questionnaire": metrics(y, full_p, np.full(N, n), full_p)}
    rng = np.random.default_rng(7)
    boots = rng.integers(0, N, size=(2000, N))
    yb = y[boots]

    def ba_boot(p):
        d = (p[boots] >= 0.5)
        pos = yb == 1
        sens = (d & pos).sum(1) / np.maximum(pos.sum(1), 1)
        spec = (~d & ~pos).sum(1) / np.maximum((~pos).sum(1), 1)
        return (sens + spec) / 2

    full_boot = ba_boot(full_p)
    for name in RULES:
        m = metrics(y, out_p[name], out_q[name], full_p)
        d_ba = ba_boot(out_p[name]) - full_boot
        m["delta_ba_vs_full"] = m["ba"] - res["full_questionnaire"]["ba"]
        m["delta_ba_ci95"] = [float(np.percentile(d_ba, 2.5)), float(np.percentile(d_ba, 97.5))]
        m["delta_auroc_vs_full"] = m["auroc"] - res["full_questionnaire"]["auroc"]
        m["stop_distribution"] = {str(k): float(np.mean(out_q[name] == k)) for k in range(2, n + 1)}
        m["selected_costs"] = chosen[name]
        res[name] = m
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    P = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    X, y = load_polish_qchat10()
    per_seed = []
    for s in range(args.seeds):
        t = time.time()
        per_seed.append({"outer_seed": P["seed"] + 1000 * s, **one_seed(X, y, P, P["seed"] + 1000 * s)})
        print(f"seed {s}: {time.time() - t:.0f}s  " + "  ".join(
            f"{k[:2]}: Q={v['mean_q']:.2f} BA={v['ba']:.3f}" for k, v in per_seed[-1].items() if k.startswith("R")))
    art = {"artifact": "rl_adaptive_quality_requirement_sweep", "status": "EXPLORATORY (not pre-registered)",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rules": list(RULES), "note": "margins are illustrative; choosing the requirement is a research decision",
           "per_seed": per_seed, "runtime_seconds": round(time.time() - t0, 1), "contains_participant_rows": False}
    (OUT / "quality_requirement_sweep.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
