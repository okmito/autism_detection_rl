"""Step 17 — outcome-level explainability artifact (P3-outcome).

Generates ``results/outcome_explainability_saudi.json``: for every held-out
episode, the structured explanation of the **final screening outcome** (exact
group-Shapley attributions, counterfactuals, prior-sensitivity band,
limitations), plus aggregate faithfulness metrics:

* counterfactual found/robust rates and feasibility breakdowns;
* additivity error of the Shapley attributions (the efficiency identity);
* deletion / insertion curves for the explanation against a random-order
  baseline — does the attribution order observed responses by their actual
  influence on the estimate?;
* rank stability of the attributions when one additional question is observed;
* explanation runtime overhead;
* subgroup attribution profiles by sex and age band where cells are powered.

Everything is computed post-hoc from the frozen predictor and the standard
``run_episode`` output; no model is retrained and no episode behaviour is
changed. Polish is not read here; the Saudi split is the canonical one.

Terminology (spec §25): screening / referral recommendation / risk estimate.
This artifact explains *model behaviour*; it is not clinical evidence and says
so in every explanation object.

Run from repo root:
    .venv/bin/python scripts/step17_outcome_explainability.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TAG = ("preliminary - V-4 / V-6 / V-7 PENDING; supervisor sign-off required; "
       "explainations describe model behaviour, not clinical evidence")

BUDGET = 6
B_MIN = 0
TAU = 0.5
LAMBDA_COST = 0.0
N_RANDOM_ORDERS = 20
MIN_CELL = 20
UNCERTAINTY_DRAWS = 200


# ---------------------------------------------------------------------------
# pure helpers (imported by tests)
# ---------------------------------------------------------------------------

def kept_state(final_state: Dict[str, Any], keep_items: Sequence[int]) -> Dict[str, Any]:
    """Copy of the terminal state with only ``keep_items`` still OBSERVED.

    All other observed items are set UNASKED (marginalised out by the
    predictor), which is the deletion operation for the faithfulness curves.
    """
    from src.env.state import UNASKED, OBSERVED

    mask = np.asarray(final_state["mask"], dtype=int).copy()
    value = np.asarray(final_state["value"], dtype=int).copy()
    keep = set(int(j) for j in keep_items)
    for j in range(mask.size):
        if mask[j] == OBSERVED and j not in keep:
            mask[j] = UNASKED
            value[j] = -1
    return {"mask": mask, "value": value, "n": final_state["n"],
            "questions_remaining": final_state.get("questions_remaining", 0),
            "budget": final_state.get("budget", final_state["n"])}


def deletion_curve(final_state: Dict[str, Any], predictor, order: Sequence[int],
                   tau: float = TAU) -> List[Dict[str, Any]]:
    """Delete items in ``order`` (first entry deleted first); record estimate path.

    Returns one entry per deletion step with the estimate, the decision, and
    whether the decision has flipped.
    """
    observed = [int(j) for j in range(int(final_state["n"]))
                if np.asarray(final_state["mask"], dtype=int)[j] == 1]
    p_final = float(predictor(final_state))
    d_final = 1 if p_final >= tau else 0
    curve: List[Dict[str, Any]] = []
    remaining = list(observed)
    for step, j in enumerate(order, start=1):
        j = int(j)
        if j not in remaining:
            raise ValueError(f"deletion order item {j} is not an observed item")
        remaining = [x for x in remaining if x != j]
        p = float(predictor(kept_state(final_state, remaining)))
        curve.append({
            "n_deleted": step,
            "p": p,
            "decision": 1 if p >= tau else 0,
            "decision_flipped": (1 if p >= tau else 0) != d_final,
            "abs_delta_from_final": abs(p - p_final),
        })
    return curve


def insertion_curve(final_state: Dict[str, Any], predictor, order: Sequence[int],
                    tau: float = TAU) -> List[Dict[str, Any]]:
    """Insert observed items in ``order`` from an all-UNASKED state."""
    from src.env.state import init_state, OBSERVED

    base = init_state({"item_responses": np.zeros(int(final_state["n"])),
                       "missing_mask": np.zeros(int(final_state["n"]), dtype=bool),
                       "label": 0, "label_source": "questionnaire",
                       "covariates": {}})
    base["questions_remaining"] = final_state.get("questions_remaining", 0)
    base["budget"] = final_state.get("budget", final_state["n"])
    value0 = np.asarray(final_state["value"], dtype=int)
    p_final = float(predictor(final_state))
    p_empty = float(predictor(base))
    curve: List[Dict[str, Any]] = []
    inserted: List[int] = []
    for step, j in enumerate(order, start=1):
        j = int(j)
        inserted.append(j)
        s = {"mask": base["mask"].copy(), "value": np.full(int(final_state["n"]), -1,
                                                           dtype=int),
             "n": final_state["n"],
             "questions_remaining": final_state.get("questions_remaining", 0),
             "budget": final_state.get("budget", final_state["n"])}
        for k in inserted:
            s["mask"][k] = OBSERVED
            s["value"][k] = int(value0[k])
        p = float(predictor(s))
        curve.append({"n_inserted": step, "p": p,
                      "progress": (p - p_empty) / (p_final - p_empty)
                      if abs(p_final - p_empty) > 1e-12 else 0.0})
    return curve


def curve_auc(curve: List[Dict[str, Any]], key: str,
              start_value: float = 0.0) -> float:
    """Trapezoidal AUC of ``key`` against step fraction (x in [0, 1]).

    ``start_value`` is the curve's value before the first step. Both
    faithfulness curves are 0 by construction there — no deletions means no
    deviation from the final estimate, no insertions means no progress — and
    the default encodes that convention; pass ``start_value`` explicitly for
    any other curve.
    """
    if not curve:
        return 0.0
    n = len(curve)
    xs = [i / n for i in range(n + 1)]
    ys = [float(start_value)] + [float(c[key]) for c in curve]
    return float(sum(0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
                     for i in range(n)))


def _mean(xs) -> float:
    """Mean of the finite entries (NaN-safe for empty/degenerate inputs)."""
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def rank_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (scipy is a project dependency)."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def _load_predictor(seed: int):
    """Prefer the frozen v3 artifact; fall back to an inline Saudi-only fit."""
    from src.models.logistic_predictor import LogisticPredictor

    frozen = RESULTS / "predictor_logistic_saudi_v3.pkl"
    if frozen.exists():
        return LogisticPredictor.load(frozen), "frozen_artifact"
    from src.data.ingest import load_dataset
    from src.data.splits import stratified_split
    saudi = load_dataset("saudi", synthetic=False)
    train, val, _ = stratified_split(saudi, seed=seed)
    pred = LogisticPredictor.fit(train, val, [f"Q{j+1}" for j in range(10)], seed=seed)
    return pred, "inline_fit_seed%d" % seed


def main() -> int:
    from src.data.ingest import load_dataset, SAUDI_CSV
    from src.data.splits import stratified_split, split_fingerprint, SCHEME
    from src.audits.circularity import audit_circularity
    from src.env.environment import run_episode
    from src.models.logistic_predictor import PREDICTOR_VERSION
    from src.policies.greedy import GreedyIGPolicy
    from src.env.state import UNASKED, OBSERVED
    from src.explain.attribution import exact_shapley
    from src.explain.outcome import explain_outcome

    print("=" * 78)
    print("Step 17 - outcome-level explainability")
    print(f"  {TAG}")
    print("=" * 78)

    # ----- data + predictor -------------------------------------------------
    try:
        saudi = load_dataset("saudi", synthetic=False)
        source = "real" if SAUDI_CSV.exists() else "synthetic"
    except FileNotFoundError:
        saudi = load_dataset("saudi", synthetic=True)
        source = "synthetic"
    train, val, test = stratified_split(saudi, seed=0)
    audit = audit_circularity(saudi)
    circularity = audit.get("classification", "unknown")
    predictor, predictor_origin = _load_predictor(seed=0)
    X_train = np.array([np.asarray(r["item_responses"], dtype=float) for r in train])
    prior = np.asarray(getattr(predictor, "prior", np.full(10, 0.5)), dtype=float)
    print(f"[1/4] saudi {len(saudi)} records (source={source}, circularity={circularity})")
    print(f"      predictor {PREDICTOR_VERSION} ({predictor_origin}); "
          f"splits {len(train)}/{len(val)}/{len(test)}")

    policy = GreedyIGPolicy(train, n_items=10)
    costs = np.ones(10, dtype=float)

    # ----- per-episode explanations and faithfulness metrics -----------------
    print(f"[2/4] explaining {len(test)} episodes at B={BUDGET}, tau={TAU} ...")
    n = len(test)
    cf_found = 0
    cf_items: Dict[str, int] = {}
    cf_feasible = 0
    cf_unseen = 0
    cf_flipping_total = 0
    additivity_max = 0.0
    abs_phi_per_item: Dict[str, List[float]] = {f"A{j+1}": [] for j in range(10)}
    top1_items: List[str] = []
    p_hats: List[float] = []
    deletion_auc_guided: List[float] = []
    deletion_auc_random: List[float] = []
    insertion_auc_guided: List[float] = []
    insertion_auc_random: List[float] = []
    flips_guided: List[bool] = []
    flips_random: List[bool] = []
    stability_rhos: List[float] = []
    overhead_ms: List[float] = []
    rng = np.random.default_rng(0)

    for rec in test:
        ep = run_episode(rec, question_budget=BUDGET, b_min=B_MIN, policy=policy,
                         predictor=predictor, lambda_cost=LAMBDA_COST, tau=TAU)
        t0 = time.perf_counter()
        ex = explain_outcome(ep, predictor, tau=TAU, costs=costs, prior=prior,
                             reference_rows=X_train, dataset_tag="saudi",
                             circularity=circularity,
                             label_source=rec.get("label_source"),
                             budget=BUDGET, tag=TAG,
                             uncertainty_draws=UNCERTAINTY_DRAWS,
                             uncertainty_seed=0)
        overhead_ms.append((time.perf_counter() - t0) * 1000.0)

        # ---- counterfactual aggregates
        cf = ex["counterfactuals"]
        flipping = [f for f in cf["single_flips"] if f["flips_decision"]]
        if flipping:
            cf_found += 1
            cf_flipping_total += len(flipping)
            for f in flipping:
                cf_items[f["item"]] = cf_items.get(f["item"], 0) + 1
            ratios = [f["prior_plausibility_ratio"] for f in flipping
                      if f["prior_plausibility_ratio"] is not None]
            if ratios and min(ratios) >= 1.0:
                cf_feasible += 1
            seen = [f["pattern_seen_in_reference"] for f in flipping
                    if f["pattern_seen_in_reference"] is not None]
            if seen and all(seen):
                cf_unseen += 1

        # ---- attribution aggregates
        additivity_max = max(additivity_max, ex["attribution"]["additivity_error"])
        contribs = ex["contributions"]
        for c in contribs:
            abs_phi_per_item[c["item"]].append(abs(c["phi"]))
        if contribs:
            top1_items.append(contribs[0]["item"])
        p_hats.append(ex["outcome"]["p_hat"])

        # ---- deletion / insertion faithfulness + stability
        fs = ep["final_state"]
        att = exact_shapley(fs, predictor)
        observed = [a.item_idx for a in att.items]
        if observed:
            order = [a.item_idx for a in att.ranked()]      # descending |phi|
            dg = deletion_curve(fs, predictor, order, tau=TAU)
            ig = insertion_curve(fs, predictor, order, tau=TAU)
            deletion_auc_guided.append(curve_auc(dg, "abs_delta_from_final"))
            insertion_auc_guided.append(curve_auc(ig, "progress"))
            flips_guided.append(bool(dg[-1]["decision_flipped"]))
            aucs_d, aucs_i, flips_r = [], [], []
            for _ in range(N_RANDOM_ORDERS):
                perm = [int(x) for x in rng.permutation(observed)]
                dr = deletion_curve(fs, predictor, perm, tau=TAU)
                ir = insertion_curve(fs, predictor, perm, tau=TAU)
                aucs_d.append(curve_auc(dr, "abs_delta_from_final"))
                aucs_i.append(curve_auc(ir, "progress"))
                flips_r.append(bool(dr[-1]["decision_flipped"]))
            deletion_auc_random.append(float(np.mean(aucs_d)))
            insertion_auc_random.append(float(np.mean(aucs_i)))
            flips_random.append(float(np.mean(flips_r)))

            # stability: observe one additional question, re-rank attributions
            unasked = [j for j in range(int(fs["n"]))
                       if np.asarray(fs["mask"], dtype=int)[j] == UNASKED]
            if unasked and len(observed) >= 2:
                u = int(unasked[0])
                base_phi = [a.phi for a in att.ranked()]
                for v in (0, 1):
                    probe = {"mask": np.asarray(fs["mask"], dtype=int).copy(),
                             "value": np.asarray(fs["value"], dtype=int).copy(),
                             "n": fs["n"],
                             "questions_remaining": fs.get("questions_remaining", 0),
                             "budget": fs.get("budget", fs["n"])}
                    probe["mask"][u] = OBSERVED
                    probe["value"][u] = v
                    att2 = exact_shapley(probe, predictor, items=observed)
                    rho = rank_correlation(base_phi,
                                           [a.phi for a in att2.ranked()])
                    if np.isfinite(rho):
                        stability_rhos.append(rho)

    # ----- aggregate ---------------------------------------------------------
    print("[3/4] aggregating faithfulness metrics ...")
    p_hat_arr = np.asarray(p_hats, dtype=float)
    top1 = np.asarray(top1_items, dtype=object) if top1_items else np.array([], dtype=object)

    subgroup: Dict[str, Any] = {}
    for label, key_fn in (("sex", lambda r: r["covariates"]["sex"]),
                          ("age_band", lambda r: r["covariates"]["age_band"])):
        for value in sorted({key_fn(r) for r in test}):
            idx = [i for i, r in enumerate(test) if key_fn(r) == value]
            if len(idx) < MIN_CELL:
                subgroup[f"{label}={value}"] = {"n": len(idx),
                                                "note": "underpowered - not interpreted"}
                continue
            cell_top1: Dict[str, int] = {}
            for i in idx:
                cell_top1[top1[i]] = cell_top1.get(top1[i], 0) + 1
            subgroup[f"{label}={value}"] = {
                "n": len(idx),
                "p_hat_mean": float(np.mean(p_hat_arr[idx])),
                "p_hat_sd": float(np.std(p_hat_arr[idx])),
                "top1_contribution_counts": cell_top1,
            }

    overhead_arr = np.asarray(overhead_ms, dtype=float)
    artifact = {
        "artifact": "outcome_explainability_saudi",
        "tag": TAG,
        "dataset": "saudi",
        "source": source,
        "circularity_status": circularity,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "predictor": {
            "version": PREDICTOR_VERSION,
            "origin": predictor_origin,
            "artifact": "predictor_logistic_saudi_v3.pkl" if predictor_origin == "frozen_artifact" else None,
            "prior_source": "saudi training prevalence (factorised)",
        },
        "split": {"scheme": SCHEME, "fingerprint": split_fingerprint(train, val, test),
                  "n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "episode_config": {"budget": BUDGET, "b_min": B_MIN, "tau": TAU,
                           "lambda_cost": LAMBDA_COST,
                           "policy": "greedy_information_gain"},
        "n_episodes": n,
        "attribution": {
            "method": "exact_group_shapley_over_observed_items",
            "abs_phi_mean_per_item": {k: _mean(v) for k, v in abs_phi_per_item.items()},
            "additivity_max_error": float(additivity_max),
            "efficiency_identity": "sum(phi) == p_hat - prior_only_estimate",
        },
        "counterfactuals": {
            "found_rate": cf_found / n if n else 0.0,
            "robust_rate": (n - cf_found) / n if n else 0.0,
            "flipping_items": dict(sorted(cf_items.items(), key=lambda kv: -kv[1])),
            "flipping_cf_total": cf_flipping_total,
            "found_and_all_plausible_rate": cf_feasible / n if n else 0.0,
            "found_and_all_seen_in_training_rate": cf_unseen / n if n else 0.0,
            "note": ("found = at least one single response change would move the "
                     "decision; robustness is the complement. Feasibility uses the "
                     "factorised training prior; seen-in-training compares the "
                     "flipped response pattern with the training cohort."),
        },
        "faithfulness": {
            "deletion_auc_guided": _mean(deletion_auc_guided),
            "deletion_auc_random_mean": _mean(deletion_auc_random),
            "insertion_auc_guided": _mean(insertion_auc_guided),
            "insertion_auc_random_mean": _mean(insertion_auc_random),
            "decision_flip_rate_full_delete_guided": _mean(flips_guided),
            "decision_flip_rate_full_delete_random": _mean(flips_random),
            "n_random_orders": N_RANDOM_ORDERS,
            "interpretation": ("deletion/insertion AUCs above the random baseline "
                               "mean the attribution ranks responses by their actual "
                               "influence on the estimate; they say nothing about "
                               "clinical validity."),
        },
        "stability": {
            "spearman_mean": _mean(stability_rhos),
            "spearman_min": float(np.min(stability_rhos)) if stability_rhos else float("nan"),
            "n_perturbations": len(stability_rhos),
            "interpretation": ("rank correlation of per-item attributions when one "
                               "additional question is observed (both possible "
                               "answers); model-stability only."),
        },
        "overhead_ms": {"mean": float(overhead_arr.mean()),
                        "p50": float(np.percentile(overhead_arr, 50)),
                        "max": float(overhead_arr.max())},
        "subgroup": subgroup,
        "limitations": [
            "Explanations describe the model's behaviour, not clinical evidence.",
            "Saudi training labels are questionnaire-derived and exactly circular.",
            "Attributions split credit among correlated items; they are not causal.",
            "The prior-sensitivity band is model stability, not clinical uncertainty.",
            "Screening result, not a diagnosis; supervisor sign-off gates remain open.",
        ],
    }
    out = RESULTS / "outcome_explainability_saudi.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"      episodes {n}; counterfactual found "
          f"{artifact['counterfactuals']['found_rate']:.3f} "
          f"(robust {artifact['counterfactuals']['robust_rate']:.3f})")
    print(f"      additivity max error {additivity_max:.2e}; deletion AUC guided "
          f"{artifact['faithfulness']['deletion_auc_guided']:.3f} vs random "
          f"{artifact['faithfulness']['deletion_auc_random_mean']:.3f}")
    print(f"      stability Spearman {artifact['stability']['spearman_mean']:.3f}; "
          f"overhead {artifact['overhead_ms']['mean']:.1f} ms/episode")
    print(f"[4/4] -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
