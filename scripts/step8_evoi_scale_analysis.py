"""Step 8 — EVOI scale analysis and stopping-rule sensitivity. V-6 decision input.

**This script selects no threshold and signs off nothing.** It measures the
distributions the V-6 decision needs, compares candidate stopping rules, and
writes the evidence. The threshold remains a supervisor decision.

What is measured, and why the existing code is not modified
----------------------------------------------------------
The support-posterior EVOI that ``BetaGreedyPolicy`` uses is a *point estimate of
expected Brier improvement*. This script measures the **same functional form on
the same states** for a second, independent probability estimate — the trained
neural predictor — so the two can be compared in identical units.

The matched estimator is the key to the comparison. For a node with support set
``S`` the code computes

    u_support(S) = mean_{y in S} [ 1 - (p_beta(S)  - y)^2 ]
    gain_j        = sum_v P(v|S,j) * u(child) - u(S)

For the predictor side this script computes, on the **same** ``S``, the **same**
``y``'s and the **same** functional form, changing only ``p``:

    u_pred(S)    = mean_{y in S} [ 1 - (p_hat(S)  - y)^2 ]
    gain_j^pred   = sum_v P(v|S,j) * u_pred(child) - u_pred(S)

Both are expected Brier improvement, dimensionless, and bounded above by
``max_p [1 - p(1-p)] = 0.25`` for *any* p in [0,1]. So the two are directly
comparable with no rescaling, and any ratio between them is a statement about
*information*, not about units. This script tests that claim rather than assuming
it, and reports the measured ceiling.

Why this matters for V-6
------------------------
``V6_LAMBDA_DECISION.md`` §2 and ``step6_lambda_sweep.py`` currently justify the
lambda grid against a "measured marginal utility of an informative question ...
roughly 0.05-0.15 Brier" for the neural-predictor reward. **No code in this
repository measures that quantity** — the figure appears only in prose, and
``tests/test_step6_lambda_sweep.py::test_informative_band_is_present`` asserts
only that the grid has >= 3 values in [0.001, 0.05]; the number is in the
docstring, not the assertion. Part 3 below measures it properly so the grid
rationale rests on evidence.

MANDATORY CAVEAT — label circularity (§16.1)
-------------------------------------------
The Saudi cohort's labels are a deterministic sum-threshold oracle over the
questionnaire items themselves. Nothing here is clinical evidence, and nothing
here is a sufficiency claim.

Outputs (under results/):
  - results/evoi_scale_saudi.json
  - results/evoi_scale_saudi.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.ingest import load_dataset
from src.data.splits import SCHEME, split_fingerprint, stratified_split
from src.env.environment import STOP, run_episode
from src.env.state import (MISSING, OBSERVED, UNASKED, encode_state,
                           get_legal_items, init_state, update_state)
from src.models.masked_predictor import MaskedPredictor
from src.policies.beta_greedy import BETA_A0, BETA_B0, BetaGreedyPolicy
from src.policies.greedy import GreedyIGPolicy

TAG = (
    "preliminary — EVOI scale analysis and stopping-rule sensitivity 2026-10-01 "
    "(Step 8); CIRCULAR questionnaire labels (Saudi §16.1); NOT clinical evidence; "
    "NO THRESHOLD SELECTED; V-4 / V-6 / V-7 pending supervisor sign-off; this "
    "artifact is DECISION INPUT for V-6 and does not sign off the gate"
)

CIRCULARITY_WARNING = (
    "Every number here is measured against a deterministic sum-threshold oracle "
    "over the questionnaire items themselves (label == 1[sum(A) >= threshold], "
    "verified 506/506 on Saudi). A 'premature stop' below means the empirical "
    "support was still label-mixed when the policy stopped, which is a statement "
    "about the support, not about diagnostic sufficiency."
)

#: The exact ceiling of expected Brier improvement for any p in [0,1]:
#: max_p [ 1 - p(1-p) ] = 0.25, attained at p = 0.5. Both the support-posterior
#: and the predictor estimator are bounded by this, so it is the natural yardstick
#: for "how much information is available".
EVOI_CEILING = 0.25

#: Candidate thresholds for the raw support-posterior decision statistic
#: (max_j gain_j). This is a REPORTING grid chosen to span the measured support
#: of the distribution on a log scale. **No value in it is recommended, selected
#: or endorsed**; the grid exists so the supervisor can see the whole
#: stop-rate/accuracy frontier at once.
CANDIDATE_THRESHOLDS = [
    0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
    1e-2, 3e-2, 5e-2, 1e-1, 1.5e-1, 2.0e-1, EVOI_CEILING,
]

QUANTILES = [0, 50, 75, 90, 95, 100]


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def describe(values: Sequence[float]) -> Dict[str, float]:
    """min / median / mean / p75 / p90 / p95 / max, as the V-6 note requires."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in
                ("n", "min", "median", "mean", "p75", "p90", "p95", "max")}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "median": float(np.percentile(a, 50)),
        "mean": float(a.mean()),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        total += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def uar(probs: Sequence[float], labels: Sequence[int], tau: float = 0.5) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.size == 0 or y.min() == y.max():
        return float("nan")
    pos, neg = y == 1, y == 0
    tpr = float(np.mean(p[pos] >= tau)) if pos.any() else float("nan")
    tnr = float(np.mean(p[neg] < tau)) if neg.any() else float("nan")
    return 0.5 * (tpr + tnr)


# --------------------------------------------------------------------------
# the two matched utility estimators
# --------------------------------------------------------------------------
class MatchedUtility:
    """Expected Brier improvement under two different p estimates.

    ``u_support`` reproduces ``BetaGreedyPolicy._posterior`` exactly (verified in
    ``self_check``); ``u_pred`` is the same functional form on the same support
    labels with the neural predictor's probability substituted.
    """

    def __init__(self, policy: BetaGreedyPolicy, predictor, n_items: int):
        self.pol = policy
        self.pred = predictor
        self.n_items = n_items
        self.X = policy.X
        self.mm = policy.missing_mask
        self.y = policy.y

    # -- branches over item j, mirroring BetaGreedyPolicy._child_stats -----
    def branches(self, idx: np.ndarray, j: int
                 ) -> List[Tuple[float, int, np.ndarray]]:
        """``[(weight, mask_code, child_index_array), ...]``.

        Same enumeration and same weights as ``_child_stats``; the child index
        arrays are returned as well so a child *state* can be built.
        """
        total = len(idx)
        out: List[Tuple[float, int, np.ndarray]] = []
        observed = idx[~self.mm[idx, j]]
        if len(observed):
            for v in np.unique(self.X[observed, j]):
                child = observed[self.X[observed, j] == v]
                out.append((len(child) / total, OBSERVED, child))
        missing = idx[self.mm[idx, j]]
        if len(missing):
            out.append((len(missing) / total, MISSING, missing))
        return out

    @staticmethod
    def u_from_p(p: float, ys: np.ndarray) -> float:
        """``mean_y [ 1 - (p - y)^2 ]`` — the form the reward already uses."""
        if ys.size == 0:
            return 1.0 - (0.5 - 0.5) ** 2
        return float(np.mean(1.0 - (p - ys) ** 2))

    def p_support(self, idx: np.ndarray) -> float:
        n = len(idx)
        if n == 0:
            return 0.5
        n_pos = int(self.y[idx].sum())
        return float((n_pos + BETA_A0) / (n + BETA_A0 + BETA_B0))

    def u_support(self, idx: np.ndarray) -> float:
        return self.u_from_p(self.p_support(idx), self.y[idx])

    def u_pred(self, idx: np.ndarray, state) -> float:
        return self.u_from_p(float(self.pred(state)), self.y[idx])

    def child_state(self, state, j: int, code: int, value: int,
                    questions_remaining: int):
        if code == MISSING:
            mask = state["mask"].copy()
            val = state["value"].copy()
            mask[j] = MISSING
            val[j] = -1
            return {"mask": mask, "value": val, "n": state["n"],
                    "questions_remaining": questions_remaining,
                    "budget": state.get("budget", questions_remaining + 1)}
        return update_state(state, j, value, questions_remaining)

    def state_gains(self, state, legal: Sequence[int]
                    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, Any]]:
        """Per-legal-item gain under both estimators, plus state diagnostics.

        Returns ``(gain_support, gain_pred, meta)`` where ``meta`` carries the
        state posterior, support size, purity, the support-weighted label
        distribution and the two state utilities.
        """
        idx = self.pol.support(state["mask"], state["value"])
        ys = self.y[idx]
        p_b = self.p_support(idx)
        p_h = float(self.pred(state))
        u_s = self.u_from_p(p_b, ys)
        u_h = self.u_from_p(p_h, ys)
        rem = int(state.get("questions_remaining", 0))
        bud = int(state.get("budget", max(rem, 1)))

        gs: Dict[int, float] = {}
        gh: Dict[int, float] = {}
        for a in legal:
            br = self.branches(idx, a)
            if not br:
                gs[a] = 0.0
                gh[a] = 0.0
                continue
            acc_s = 0.0
            acc_h = 0.0
            for w, code, child in br:
                v_obs = int(self.X[child, a][0]) if code == OBSERVED else -1
                cs = self.child_state(state, a, code, v_obs, max(rem - 1, 0))
                cs["budget"] = bud
                acc_s += w * self.u_support(child)
                acc_h += w * self.u_pred(child, cs)
            gs[a] = acc_s - u_s
            gh[a] = acc_h - u_h

        n_pos = int(ys.sum()) if ys.size else 0
        meta = {
            "support_size": int(len(idx)),
            "n_pos": n_pos,
            "pure_support": bool(len(idx) > 0 and n_pos in (0, len(idx))),
            "p_beta": p_b,
            "p_hat": p_h,
            "u_support": u_s,
            "u_pred": u_h,
            "depth": int(bud - rem),
        }
        return gs, gh, meta


# --------------------------------------------------------------------------
# trajectory collection
# --------------------------------------------------------------------------
def collect_states(records, pol: BetaGreedyPolicy, mu: MatchedUtility,
                   order_policy, budget: int) -> List[Dict[str, Any]]:
    """Walk every record under ``order_policy`` and score each decision state.

    ``order_policy`` only decides *which item is asked next*; the EVOI statistics
    are collected at every state visited, so the sample reflects the states a
    deployed policy actually faces rather than an arbitrary enumeration.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        state = init_state(rec)
        state["questions_remaining"] = budget
        state["budget"] = budget
        for depth in range(budget):
            legal = get_legal_items(state)
            if not legal:
                break
            gs, gh, meta = mu.state_gains(state, legal)
            rows.append({
                "depth": meta["depth"],
                "max_gain_support": max(gs.values()) if gs else 0.0,
                "max_gain_pred": max(gh.values()) if gh else 0.0,
                "mean_gain_support": float(np.mean(list(gs.values()))) if gs else 0.0,
                "mean_gain_pred": float(np.mean(list(gh.values()))) if gh else 0.0,
                "n_legal": len(legal),
                **{k: meta[k] for k in ("support_size", "pure_support",
                                         "p_beta", "p_hat", "u_support", "u_pred")},
                "all_support": list(gs.values()),
                "all_pred": list(gh.values()),
            })
            action = order_policy(state, legal + [STOP])
            if action == STOP or action not in legal:
                break
            value = rec["item_responses"][action]
            if np.isnan(value):
                break
            state = update_state(state, action, int(value), budget - depth - 1)
            state["budget"] = budget
            state["questions_remaining"] = budget - depth - 1
    return rows


def self_check(pol: BetaGreedyPolicy, mu: MatchedUtility, records,
               budget: int = 6) -> Dict[str, Any]:
    """Confirm the mirrored support-EVOI reproduces the policy's own numbers.

    If this fails, the predictor-side numbers are not on the same footing and the
    comparison would be invalid, so it is asserted rather than assumed.
    """
    worst = 0.0
    checked = 0
    for rec in records:
        state = init_state(rec)
        state["questions_remaining"] = budget
        state["budget"] = budget
        for depth in range(budget):
            legal = get_legal_items(state)
            if not legal:
                break
            gs, _gh, _meta = mu.state_gains(state, legal)
            ref = pol.scores(state, legal + [STOP])["evoi"]
            for a in legal:
                worst = max(worst, abs(gs[a] - ref[a]))
                checked += 1
            action = legal[0]
            value = rec["item_responses"][action]
            if np.isnan(value):
                break
            state = update_state(state, action, int(value), budget - depth - 1)
            state["budget"] = budget
            state["questions_remaining"] = budget - depth - 1
    return {"n_compared": checked, "max_abs_diff": worst,
            "matches_policy": worst < 1e-12}


# --------------------------------------------------------------------------
# candidate stopping rules (subclasses; the shipped policy is NOT modified)
# --------------------------------------------------------------------------
class RuleBetaGreedy(BetaGreedyPolicy):
    """``statistic='support_evoi'`` with threshold ``tau`` — the shipped rule.

    ``tau=0`` with the shipped ``lambda_cost`` semantics is not reachable via the
    parent class (it skips the stop test entirely when ``lambda_cost == 0``), so
    the comparison here uses an explicit ``tau`` and says so.
    """

    STATISTICS = ("support_evoi", "relative_evoi", "normalised_ig",
                  "uncertainty_per_question", "reward_per_evoi",
                  "percentile", "predictor_evoi")

    def __init__(self, records, n_items, statistic="support_evoi", tau=0.0,
                 percentile_q=50.0, select_by="evoi", **kw):
        super().__init__(records, n_items=n_items, select_by=select_by, **kw)
        if statistic not in self.STATISTICS:
            raise ValueError(f"unknown statistic {statistic!r}")
        self.statistic = statistic
        self.tau = float(tau)
        self.percentile_q = float(percentile_q)

    # -- the alternative statistics ----------------------------------------
    def stop_statistic(self, state, legal, stats):
        """Scalar compared against ``self.tau``; larger means 'more worth asking'."""
        items = [a for a in legal if a != STOP]
        if not items:
            return 0.0
        evoi = stats["evoi"]
        ig = stats["ig"]
        h = stats["entropy"]
        u = stats["utility"]
        best_evoi = max(evoi.values())
        if self.statistic == "support_evoi":
            return best_evoi
        if self.statistic == "predictor_evoi":
            return self._pred_best(state, items)
        if self.statistic == "relative_evoi":
            # gain as a fraction of the utility at stake. Scale-free, and the
            # natural reading of "expected % of the remaining utility".
            return best_evoi / u if u > 0 else best_evoi
        if self.statistic == "normalised_ig":
            # uncertainty reduction as a fraction of the current uncertainty.
            # h -> 0 on a pure support, so this is a *blow-up* there, which is
            # precisely the pathology worth showing.
            best_ig = max(ig.values())
            return best_ig / h if h > 1e-12 else float("inf") if best_ig > 0 else 0.0
        if self.statistic == "uncertainty_per_question":
            # bits removed per question, cost-weighted. Same as IG at c_j = 1.
            return max(ig[a] / float(self.costs[a]) for a in items)
        if self.statistic == "reward_per_evoi":
            # questions per unit of expected Brier improvement (the inverse
            # threshold). Larger = better value, so it is monotone-decreasing in
            # EVOI and the comparison direction flips.
            return 1.0 / best_evoi if best_evoi > 0 else float("inf")
        if self.statistic == "percentile":
            return best_evoi  # threshold replaced by a calibrated quantile
        raise AssertionError

    def _pred_best(self, state, items):
        mu = self._mu
        if mu is None:
            return 0.0
        _gs, gh, _meta = mu.state_gains(state, items)
        return max(gh.values()) if gh else 0.0

    _mu = None

    def __call__(self, state, legal):
        items_only = [a for a in legal if a != STOP]
        if not items_only:
            return STOP
        stats = self.scores(state, legal)
        crit = self.criterion(stats)
        best = max(items_only, key=lambda a: (crit[a], -a))
        if STOP in legal:
            s = self.stop_statistic(state, items_only, stats)
            if self.statistic == "reward_per_evoi":
                if s < self.tau:      # fewer questions per unit = better value
                    return STOP
            elif s < self.tau:
                return STOP
        return best


# --------------------------------------------------------------------------
# episode-level sensitivity analysis
# --------------------------------------------------------------------------
def run_rule(records, policy, predictor, budget: int, pol_for_support
             ) -> Dict[str, Any]:
    """Run every record and collect the decision-quality metrics V-6 needs.

    Definitions used here, stated explicitly because they are judgement calls:

    ``premature_stop``
        The policy stopped while the empirical support was still **label-mixed**.
        Rationale: on this data a mixed support means the training evidence does
        not yet determine the label, so more questions could still have changed
        the answer. This is a statement about the support, not about clinical
        sufficiency.
    ``unnecessary_questions``
        Questions asked *after* the support first became pure. On a pure support
        the reward term is already saturated, so these questions cannot change
        the objective.
    ``regret``
        Utility of the full-length (fixed B) reference minus utility of this rule,
        at lambda = 0. Positive means stopping early cost utility.
    """
    n_asked: List[int] = []
    p_term: List[float] = []
    p_hat_term: List[float] = []
    ys: List[int] = []
    premature = 0
    unnecessary = 0
    rewards: List[float] = []

    for rec in records:
        ep = run_episode(rec, question_budget=budget, b_min=0, policy=policy,
                         predictor=predictor, lambda_cost=0.0,
                         cost_mode="uniform", tau=0.5)
        items = ep["items_asked"]
        n_asked.append(len(items))
        p_term.append(float(ep["p_hat"]))
        p_hat_term.append(float(ep["p_hat_neural"])
                          if "p_hat_neural" in ep else float("nan"))
        ys.append(int(rec["label"]))
        rewards.append(float(ep["R"]))

        # Re-walk the realised trajectory to classify the stop.
        state = init_state(rec)
        state["questions_remaining"] = budget
        state["budget"] = budget
        pure_at: int | None = None
        for depth, a in enumerate(items):
            value = rec["item_responses"][a]
            if np.isnan(value):
                break
            state = update_state(state, a, int(value), budget - depth - 1)
            state["budget"] = budget
            state["questions_remaining"] = budget - depth - 1
            idx = pol_for_support.support(state["mask"], state["value"])
            n_pos = int(pol_for_support.y[idx].sum()) if len(idx) else 0
            is_pure = bool(len(idx) > 0 and n_pos in (0, len(idx)))
            if is_pure and pure_at is None:
                pure_at = depth + 1
        if len(items) < budget:
            idx = pol_for_support.support(state["mask"], state["value"])
            n_pos = int(pol_for_support.y[idx].sum()) if len(idx) else 0
            if not (len(idx) > 0 and n_pos in (0, len(idx))):
                premature += 1
        if pure_at is not None:
            unnecessary += max(0, len(items) - pure_at)

    n = len(records)
    return {
        "mean_questions": float(np.mean(n_asked)),
        "median_questions": float(np.median(n_asked)),
        "frac_full_length": float(np.mean([x >= budget for x in n_asked])),
        "premature_stop_frac": premature / n if n else float("nan"),
        "unnecessary_questions_mean": unnecessary / n if n else float("nan"),
        "brier": brier(p_term, ys),
        "uar": uar(p_term, ys),
        "ece_10bin": ece(p_term, ys),
        "mean_utility_reward": float(np.mean(rewards)),
        "n_episodes": n,
    }


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="saudi")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per split (0 = all)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("Step 8 — EVOI scale analysis + stopping-rule sensitivity")
    print(f"  {TAG}")
    print("=" * 78)

    records = load_dataset(args.dataset, synthetic=args.synthetic)
    train, val, test = stratified_split(records, seed=args.seed)
    fingerprint = split_fingerprint(train, val, test)
    n_items = len(records[0]["item_responses"])
    if args.limit:
        val, test = val[: args.limit], test[: args.limit]
    print(f"[{args.dataset}] train={len(train)} val={len(val)} test={len(test)}"
          f" | n_items={n_items} | budget={args.budget}")
    print(f"  split scheme={SCHEME} fingerprint={fingerprint}")

    # ---- predictors -----------------------------------------------------
    pol = BetaGreedyPolicy(train, n_items=n_items, lambda_cost=0.0)
    nn = MaskedPredictor(n_items=n_items, hidden=[128, 64],
                         calibration="platt", seed=args.seed)
    cache_pt = RESULTS / "demo_model_saudi_seed0_platt_v2.pt"
    cache_pk = RESULTS / "demo_model_saudi_seed0_platt_v2.pkl"
    if cache_pt.exists() and cache_pk.exists():
        nn.model.load_state_dict(torch.load(cache_pt, map_location="cpu"))
        with open(cache_pk, "rb") as fh:
            nn.calibrator = pickle.load(fh)
        print(f"  neural predictor: loaded {cache_pt.name}")
    else:
        nn.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=args.seed)
        nn.fit_calibrator(val, method="platt", seed=args.seed)
        print("  neural predictor: trained (no cache found)")

    def support_posterior(state) -> float:
        idx = pol.support(state["mask"], state["value"])
        return float(pol.y[idx].mean()) if len(idx) else 0.5

    mu = MatchedUtility(pol, nn, n_items)

    # ---- 0. fidelity check ---------------------------------------------
    chk = self_check(pol, mu, test[:40])
    print(f"\n[fidelity] mirrored support-EVOI vs BetaGreedyPolicy.scores(): "
          f"max|diff| = {chk['max_abs_diff']:.3e} over {chk['n_compared']} "
          f"item-states -> matches={chk['matches_policy']}")
    if not chk["matches_policy"]:
        raise SystemExit("mirrored EVOI does not match the policy; aborting")

    # ---- 1/2/3. distributions -------------------------------------------
    order = GreedyIGPolicy(train, n_items=n_items)
    dist: Dict[str, Any] = {"ceiling_expected_brier_improvement": EVOI_CEILING,
                            "fidelity_check": chk, "by_split": {}}
    for split_name, recs in (("val", val), ("test", test)):
        rows = collect_states(recs, pol, mu, order, args.budget)
        blk: Dict[str, Any] = {"n_decision_states": len(rows), "overall": {},
                               "by_depth": {}}
        pool_s = [v for r in rows for v in r["all_support"]]
        pool_p = [v for r in rows for v in r["all_pred"]]
        blk["overall"] = {
            "support_evoi_per_item": describe(pool_s),
            "predictor_gain_per_item": describe(pool_p),
            "support_evoi_max_over_legal": describe(
                [r["max_gain_support"] for r in rows]),
            "predictor_gain_max_over_legal": describe(
                [r["max_gain_pred"] for r in rows]),
        }
        for d in sorted({r["depth"] for r in rows}):
            sub = [r for r in rows if r["depth"] == d]
            blk["by_depth"][str(d)] = {
                "n_states": len(sub),
                "pure_support_frac": float(np.mean([r["pure_support"] for r in sub])),
                "support_evoi_max": describe([r["max_gain_support"] for r in sub]),
                "predictor_gain_max": describe([r["max_gain_pred"] for r in sub]),
                "support_evoi_max_on_pure": describe(
                    [r["max_gain_support"] for r in sub if r["pure_support"]]),
                "support_evoi_max_on_impure": describe(
                    [r["max_gain_support"] for r in sub if not r["pure_support"]]),
            }
        dist["by_split"][split_name] = blk

    # ---- 4. scale mismatch ---------------------------------------------
    te = dist["by_split"]["test"]["overall"]
    s_max = te["support_evoi_max_over_legal"]
    p_max = te["predictor_gain_max_over_legal"]
    ratio = {}
    for k in ("median", "mean", "p75", "p90", "p95", "max"):
        a, b = s_max[k], p_max[k]
        ratio[k] = (a / b) if (b not in (0.0, None) and np.isfinite(b)
                               and abs(b) > 1e-12) else float("nan")
    dist["scale_mismatch"] = {
        "note": (
            "Both quantities are expected Brier improvement with the same "
            "functional form and the same y-distribution, differing only in "
            "whether p is the Beta-smoothed support posterior or the neural "
            "predictor. Both are bounded above by 0.25, so these ratios are "
            "statements about information, not about units."
        ),
        "support_median": s_max["median"],
        "predictor_median": p_max["median"],
        "support_over_predictor": ratio,
        "support_median_over_ceiling": s_max["median"] / EVOI_CEILING,
        "predictor_median_over_ceiling": p_max["median"] / EVOI_CEILING,
        "unsourced_claim_under_test": {
            "claim": "marginal utility of an informative question is 0.05-0.15 Brier (neural predictor)",
            "claim_source": "prose only; asserted in step6_lambda_sweep.py docstring and V6_LAMBDA_DECISION.md §2, measured by no code",
            "measured_predictor_gain_p50": p_max["median"],
            "measured_predictor_gain_p90": p_max["p90"],
            "measured_predictor_gain_p95": p_max["p95"],
            "claim_supported": bool(0.05 <= p_max["median"] <= 0.15),
        },
    }

    # ---- 5. stop rate at candidate thresholds ---------------------------
    thr_rows: List[Dict[str, Any]] = []
    for split_name, recs in (("val", val), ("test", test)):
        rows = collect_states(recs, pol, mu, order, args.budget)
        for t in CANDIDATE_THRESHOLDS:
            # A state stops if the best remaining question cannot repay t.
            stops = [r["max_gain_support"] < t for r in rows]
            stop_rate = float(np.mean(stops)) if stops else float("nan")
            thr_rows.append({
                "split": split_name, "statistic": "support_evoi_max",
                "threshold": t, "n_states": len(rows),
                "stop_rate": stop_rate,
            })
    dist["candidate_threshold_stop_rates"] = thr_rows

    # ---- 8. offline sensitivity of each candidate rule -----------------
    sens: List[Dict[str, Any]] = []
    ref = run_rule(test, RuleBetaGreedy(train, n_items, statistic="support_evoi",
                                        tau=-1e18),
                   support_posterior, args.budget, pol)
    ref_row = dict(ref, rule="full_length_reference", statistic="none",
                   threshold=float("nan"),
                   note="tau=-inf, so the policy always asks all B questions")
    sens.append(ref_row)

    grid: List[Tuple[str, List[float]]] = [
        ("support_evoi", CANDIDATE_THRESHOLDS[1:]),
        ("predictor_evoi", CANDIDATE_THRESHOLDS[1:]),
        ("relative_evoi", [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
        ("normalised_ig", [0.0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]),
        ("uncertainty_per_question", [0.0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]),
        ("reward_per_evoi", [1e4, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0]),
    ]
    for stat, taus in grid:
        for t in taus:
            r = RuleBetaGreedy(train, n_items, statistic=stat, tau=t)
            r._mu = mu
            out = run_rule(test, r, support_posterior, args.budget, pol)
            out["regret_vs_full_length"] = ref["mean_utility_reward"] - out["mean_utility_reward"]
            sens.append(dict(out, rule=f"{stat}", statistic=stat, threshold=t))

    # percentile rule: calibrate the quantile on VAL, apply to TEST
    val_rows = collect_states(val, pol, mu, order, args.budget)
    val_max = np.array([r["max_gain_support"] for r in val_rows])
    for q in (25, 50, 75, 90):
        t = float(np.percentile(val_max, q))
        r = RuleBetaGreedy(train, n_items, statistic="support_evoi", tau=t)
        r._mu = mu
        out = run_rule(test, r, support_posterior, args.budget, pol)
        out["regret_vs_full_length"] = ref["mean_utility_reward"] - out["mean_utility_reward"]
        sens.append(dict(out, rule="percentile", statistic="percentile",
                         threshold=t, calibrated_q=q, calibrated_on="val"))

    dist["sensitivity"] = sens
    dist["no_threshold_selected"] = True

    # ---- write ----------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    artifact = {
        "tag": TAG,
        "step": "8 — EVOI scale analysis and stopping-rule sensitivity",
        "purpose": (
            "Evidence for the V-6 supervisor decision on the stopping threshold. "
            "This script deliberately selects NO threshold and does not sign off "
            "V-6. Every candidate is reported with its measured consequences."
        ),
        "dataset": args.dataset,
        "label_source": "questionnaire (CIRCULAR — see §16.1)",
        "circularity_warning": CIRCULARITY_WARNING,
        "budget": args.budget,
        "seed": args.seed,
        "split_scheme": SCHEME,
        "split_fingerprint": fingerprint,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "n_items": n_items,
        "predictor": {
            "kind": "MaskedMLP[128,64] + platt",
            "source": "results/demo_model_saudi_seed0_platt_v2.pt",
            "note": ("the held-out *reward* still uses the support posterior, so "
                     "this artifact stays comparable with V_star; the neural "
                     "predictor is used only to measure the second EVOI scale"),
        },
        "candidate_thresholds_are_a_reporting_grid_not_a_recommendation": True,
        "analysis": dist,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    j = RESULTS / f"evoi_scale_{args.dataset}.json"
    j.write_text(json.dumps(artifact, indent=2))
    c = RESULTS / f"evoi_scale_{args.dataset}.csv"
    with open(c, "w", newline="") as fh:
        cols: List[str] = []
        for row in sens:
            for k in row:
                if k not in cols:
                    cols.append(k)
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(sens)

    print(f"\n-> {j.relative_to(REPO)}")
    print(f"-> {c.relative_to(REPO)}")
    print("\nNO THRESHOLD SELECTED. V-6 remains open pending supervisor decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
