"""Nested cross-validation of adaptive (RL) and fixed questionnaires against independent labels.

Procedure (applied identically to every tunable arm, and frozen in ``results/rl_adaptive/protocol.json``):

1. Outer stratified K-fold over participants. Every participant is tested exactly once, by models that never saw
   them.
2. Inside each outer-training set, an inner stratified K-fold estimates, for every hyper-parameter value (question
   cost ``c`` for the RL agents, confidence ``hi`` for the threshold-stopping heuristic), the pooled balanced
   accuracy and mean number of questions, together with the full 10-item questionnaire's balanced accuracy.
3. **Constrained selection:** the value with the fewest mean questions whose inner balanced accuracy is at least
   ``full - delta_select``; if none qualifies, the value with the highest inner balanced accuracy.
4. The selected value is refitted on the whole outer-training set and applied to the outer-test fold.

Every arm is scored at STOP by the naive-Bayes belief fitted on the same outer-training set, so all differences
come from *which* and *how many* questions were asked.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.rl.baselines import (BeliefGreedyIG, FixedOrder, LegacyPolicyAdapter, RandomFixedLength,
                              ThresholdStopping, forward_selection_order, legacy_records)
from src.rl.belief import NaiveBayesBelief
from src.rl.dqn import DQNAgent, DQNConfig, QNet, train_dqn
from src.rl.evaluate import adaptivity_profile, rollout, screening_metrics
from src.rl.mdp import FEATURE_DIM, MDPConfig
from src.rl.planner import BeliefValueIteration

DEFAULT_PROTOCOL: Dict[str, Any] = {
    "mdp": {"n_items": 10, "max_questions": 10, "min_questions": 2, "terminal": "accuracy", "tau": 0.5},
    "cost_grid": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05],
    "threshold_grid": [0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    "outer_folds": 5,
    "inner_folds": 3,
    "seed": 20261113,
    "dqn": {"seeds": [0, 1, 2], "config": {"epochs": 400, "updates_per_epoch": 16}},
    "belief": {"smoothing": 1.0, "calibrate": True},
    "selection": {"metric": "balanced_accuracy", "delta_select": 0.02},
    "non_inferiority": {"metric": "balanced_accuracy", "delta": 0.05, "alpha_two_sided": 0.05,
                        "multiplicity": "holm over the two RL agents", "sensitivity_deltas": [0.02, 0.03, 0.10]},
    "bootstrap": {"n_resamples": 5000, "seed": 7},
    "fixed_k": [1, 2, 3, 4, 5, 6, 7, 8, 9],
}


def stratified_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def code_fingerprint(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- DQN jobs
def _dqn_job(args) -> Tuple[Any, List[Dict[str, np.ndarray]]]:
    import torch
    torch.set_num_threads(1)
    key, X, y, cfg_d, dcfg_d, seed, bk = args
    net = train_dqn(X, y, MDPConfig(**cfg_d), DQNConfig(**dcfg_d), seed=seed, belief_kwargs=bk)
    return key, {k: v.numpy() for k, v in net.state_dict().items()}


def _net_from(sd, n_items: int, hidden: int) -> QNet:
    import torch
    net = QNet(FEATURE_DIM(n_items), n_items + 1, hidden)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
    net.eval()
    return net


def run_dqn_jobs(jobs: List[tuple], workers: int, progress: Optional[Callable[[str], None]] = None):
    out: Dict[Any, Any] = {}
    if workers <= 1:
        for j in jobs:
            k, sd = _dqn_job(j)
            out[k] = sd
        return out
    with ProcessPoolExecutor(workers) as ex:
        for i, (k, sd) in enumerate(ex.map(_dqn_job, jobs, chunksize=1)):
            out[k] = sd
            if progress and (i + 1) % 25 == 0:
                progress(f"   DQN trainings {i + 1}/{len(jobs)}")
    return out


# --------------------------------------------------------------------------------------------- core
class Study:
    """Holds the protocol and the data of one nested-CV study (development set = the cohort being CV'd)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, protocol: Dict[str, Any], *, workers: int = 1,
                 progress: Optional[Callable[[str], None]] = None, include_dqn: bool = True,
                 legacy_dqn_loader: Optional[Callable[[int], Any]] = None):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, int)
        self.P = protocol
        self.cfg = MDPConfig(**protocol["mdp"])
        self.bcfg = replace(self.cfg, min_questions=0)          # fixed-length arms stop themselves
        self.dcfg = DQNConfig(**protocol["dqn"]["config"])
        self.workers = workers
        self.progress = progress or (lambda s: None)
        self.include_dqn = include_dqn
        self.legacy_dqn_loader = legacy_dqn_loader

    # ---------------------------------------------------------------- fitting helpers
    def belief(self, X, y) -> NaiveBayesBelief:
        return NaiveBayesBelief.fit(X, y, seed=self.P["seed"], **self.P["belief"])

    def vi(self, b: NaiveBayesBelief, c: float, terminal: Optional[str] = None) -> BeliefValueIteration:
        cfg = replace(self.cfg, cost=float(c), terminal=terminal or self.cfg.terminal)
        return BeliefValueIteration(b, cfg)

    def dqn_agent(self, sds, b: NaiveBayesBelief, c: float) -> DQNAgent:
        cfg = replace(self.cfg, cost=float(c))
        nets = [_net_from(sd, self.cfg.n_items, self.dcfg.hidden) for sd in sds]
        return DQNAgent(nets, b, cfg, self.dcfg, self.P["dqn"]["seeds"])

    def _dqn_jobs_for(self, tag, X, y) -> List[tuple]:
        return [((tag, c, s), X, y, replace(self.cfg, cost=float(c)).to_dict(), self.dcfg.to_dict(), int(s),
                 dict(self.P["belief"]))
                for c in self.P["cost_grid"] for s in self.P["dqn"]["seeds"]]

    # ---------------------------------------------------------------- constrained selection
    def select(self, table: Dict[float, Dict[str, float]], full_ba: float) -> Tuple[float, str]:
        d = self.P["selection"]["delta_select"]
        ok = [h for h, m in table.items() if m["balanced_accuracy"] >= full_ba - d]
        if ok:
            best = min(ok, key=lambda h: (table[h]["mean_questions"], -table[h]["balanced_accuracy"], h))
            return best, "constraint_met"
        return max(table, key=lambda h: (table[h]["balanced_accuracy"], -h)), "constraint_unmet_max_ba"

    def inner_tables(self, X, y, dqn_sds: Optional[Dict] = None, tag=None) -> Dict[str, Any]:
        """Pooled inner-CV (BA, mean questions) per hyper-parameter for vi / threshold / dqn, and the full-10 BA."""
        folds = stratified_folds(y, self.P["inner_folds"], self.P["seed"] + 1)
        pooled: Dict[str, Dict[float, Dict[str, list]]] = {"vi": {}, "threshold": {}, "dqn": {}}
        full_p = np.zeros(len(y))
        for f in range(self.P["inner_folds"]):
            tr, te = folds != f, folds == f
            b = self.belief(X[tr], y[tr])
            full_p[te] = rollout(FixedOrder(range(self.cfg.n_items), self.cfg.n_items), X[te], b, self.bcfg)["p"]
            for c in self.P["cost_grid"]:
                r = rollout(self.vi(b, c), X[te], b, self.cfg)
                pooled["vi"].setdefault(c, {"p": np.zeros(len(y)), "q": np.zeros(len(y))})
                pooled["vi"][c]["p"][te], pooled["vi"][c]["q"][te] = r["p"], r["n_q"]
                if dqn_sds is not None:
                    sds = [dqn_sds[((tag, f), c, s)] for s in self.P["dqn"]["seeds"]]
                    r = rollout(self.dqn_agent(sds, b, c), X[te], b, self.cfg)
                    pooled["dqn"].setdefault(c, {"p": np.zeros(len(y)), "q": np.zeros(len(y))})
                    pooled["dqn"][c]["p"][te], pooled["dqn"][c]["q"][te] = r["p"], r["n_q"]
            for hi in self.P["threshold_grid"]:
                ag = ThresholdStopping(b, hi, self.cfg.min_questions, self.cfg.max_questions)
                r = rollout(ag, X[te], b, self.cfg)
                pooled["threshold"].setdefault(hi, {"p": np.zeros(len(y)), "q": np.zeros(len(y))})
                pooled["threshold"][hi]["p"][te], pooled["threshold"][hi]["q"][te] = r["p"], r["n_q"]
        full_ba = screening_metrics(y, full_p, np.full(len(y), 10))["balanced_accuracy"]
        out = {"full_ba": full_ba, "tables": {}, "selected": {}}
        for arm, d in pooled.items():
            if not d:
                continue
            table = {h: {k: v for k, v in screening_metrics(y, v["p"], v["q"]).items()
                         if k in ("balanced_accuracy", "mean_questions", "auroc")} for h, v in d.items()}
            out["tables"][arm] = table
            out["selected"][arm] = self.select(table, full_ba)
        return out

    # ---------------------------------------------------------------- the study
    def run(self) -> Dict[str, Any]:
        P, X, y = self.P, self.X, self.y
        N, n = X.shape
        outer = stratified_folds(y, P["outer_folds"], P["seed"])
        # 1. all DQN trainings up front (parallel): inner fits and outer refits for every cost
        dqn_sds = None
        if self.include_dqn:
            jobs = []
            for o in range(P["outer_folds"]):
                Xo, yo = X[outer != o], y[outer != o]
                inner = stratified_folds(yo, P["inner_folds"], P["seed"] + 1)
                for f in range(P["inner_folds"]):
                    jobs += self._dqn_jobs_for(((o, "inner"), f), Xo[inner != f], yo[inner != f])
                jobs += self._dqn_jobs_for((o, "refit"), Xo, yo)
            self.progress(f"   training {len(jobs)} DQN networks on {self.workers} workers")
            dqn_sds = run_dqn_jobs(jobs, self.workers, self.progress)

        preds: Dict[str, Dict[str, np.ndarray]] = {}

        def put(name, te, r):
            d = preds.setdefault(name, {"p": np.full(N, np.nan), "n_q": np.zeros(N), "forced": np.zeros(N, bool),
                                        "seq": np.full((N, n), -1), "codes": np.zeros((N, n), int)})
            d["p"][te], d["n_q"][te] = r["p"], r["n_q"]
            d["forced"][te], d["seq"][te], d["codes"][te] = r["forced"], r["seq"], r["codes"]

        selection_log = []
        for o in range(P["outer_folds"]):
            self.progress(f"   outer fold {o + 1}/{P['outer_folds']}")
            tr, te = outer != o, outer == o
            Xo, yo, Xt = X[tr], y[tr], X[te]
            b = self.belief(Xo, yo)
            inner = self.inner_tables(Xo, yo, dqn_sds, tag=(o, "inner") if self.include_dqn else None)
            selection_log.append({"outer_fold": o, "inner_full_ba": inner["full_ba"],
                                  "selected": {k: {"value": v[0], "rule": v[1]} for k, v in inner["selected"].items()},
                                  "tables": {arm: {str(h): m for h, m in t.items()}
                                             for arm, t in inner["tables"].items()}})
            # RL agents (selected) + their full cost sweep (Pareto, descriptive)
            for c in P["cost_grid"]:
                vi = self.vi(b, c)
                r = rollout(vi, Xt, b, self.cfg)
                put(f"vi_c{c:g}", te, r)
                if c == inner["selected"]["vi"][0]:
                    put("rl_value_iteration", te, r)
                rb = rollout(self.vi(b, c, terminal="brier"), Xt, b, replace(self.cfg, cost=c, terminal="brier"))
                put(f"vi_brier_c{c:g}", te, rb)
                if self.include_dqn:
                    sds = [dqn_sds[((o, "refit"), c, s)] for s in P["dqn"]["seeds"]]
                    r = rollout(self.dqn_agent(sds, b, c), Xt, b, self.cfg)
                    put(f"dqn_c{c:g}", te, r)
                    if c == inner["selected"]["dqn"][0]:
                        put("rl_dqn", te, r)
            for hi in P["threshold_grid"]:
                r = rollout(ThresholdStopping(b, hi, self.cfg.min_questions, self.cfg.max_questions), Xt, b, self.cfg)
                put(f"threshold_{hi:g}", te, r)
                if hi == inner["selected"]["threshold"][0]:
                    put("threshold_stopping", te, r)
            # fixed-length baselines
            put("full_questionnaire", te, rollout(FixedOrder(range(n), n), Xt, b, self.bcfg))
            fs_order = forward_selection_order(b, Xo, yo, self.cfg.tau)
            recs = legacy_records(Xo, yo)
            from src.policies.beta_greedy import BetaGreedyPolicy
            from src.policies.greedy import GreedyIGPolicy
            greedy, beta = GreedyIGPolicy(recs, n), BetaGreedyPolicy(recs, n)
            for k in P["fixed_k"]:
                put(f"fixed_official_order_k{k}", te, rollout(FixedOrder(range(n), k), Xt, b, self.bcfg))
                put(f"fixed_best_subset_k{k}", te, rollout(FixedOrder(fs_order, k), Xt, b, self.bcfg))
                put(f"random_fixed_k{k}", te, rollout(RandomFixedLength(n, k, seed=P["seed"] + 100 * o + k),
                                                      Xt, b, self.bcfg))
                put(f"belief_greedy_ig_k{k}", te, rollout(BeliefGreedyIG(b, k), Xt, b, self.bcfg))
                put(f"greedy_ig_k{k}", te, rollout(LegacyPolicyAdapter(greedy, n, k, "greedy_ig"), Xt, b, self.bcfg))
                put(f"beta_greedy_k{k}", te, rollout(LegacyPolicyAdapter(beta, n, k, "beta_greedy"), Xt, b, self.bcfg))
            if self.legacy_dqn_loader is not None:
                for k in (2, 3, 4, 5):
                    pol = self.legacy_dqn_loader(k)
                    if pol is not None:
                        put(f"legacy_dqn_v3_k{k}", te, rollout(LegacyPolicyAdapter(pol, n, k, "legacy_dqn"),
                                                               Xt, b, self.bcfg))
        return {"outer_folds": outer, "preds": preds, "selection_log": selection_log}


# --------------------------------------------------------------------------------------------- statistics
def paired_bootstrap(y: np.ndarray, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], tau: float,
                     n_boot: int, seed: int) -> Dict[str, Any]:
    """Paired bootstrap over participants of (a - b) for BA, sensitivity, specificity, AUROC, Brier, questions."""
    from src.eval.fast_stats import auroc_rows

    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yy = y[idx]

    def stats(d):
        p = np.asarray(d["p"])[idx]
        q = np.asarray(d["n_q"], float)[idx]
        dec = (p >= tau)
        pos = yy == 1
        with np.errstate(invalid="ignore", divide="ignore"):
            sens = (dec & pos).sum(1) / pos.sum(1)
            spec = (~dec & ~pos).sum(1) / (~pos).sum(1)
        return {"balanced_accuracy": 0.5 * (sens + spec), "sensitivity": sens, "specificity": spec,
                "auroc": auroc_rows(yy, p), "brier": ((p - yy) ** 2).mean(1), "mean_questions": q.mean(1)}

    sa, sb = stats(a), stats(b)
    out = {}
    for k in sa:
        diff = sa[k] - sb[k]
        diff = diff[np.isfinite(diff)]
        pa = screening_point(y, a, tau)[k]
        pb = screening_point(y, b, tau)[k]
        out[k] = {"a": pa, "b": pb, "difference": pa - pb, "ci95": [float(np.percentile(diff, 2.5)),
                                                                    float(np.percentile(diff, 97.5))],
                  "n_valid_resamples": int(diff.size), "_draws": diff}
    return out


def screening_point(y, d, tau):
    m = screening_metrics(y, d["p"], d["n_q"], tau)
    return {k: m[k] for k in ("balanced_accuracy", "sensitivity", "specificity", "auroc", "brier", "mean_questions")}


def non_inferiority(boot: Dict[str, Any], metric: str, delta: float) -> Dict[str, Any]:
    draws = boot[metric]["_draws"]
    lo = boot[metric]["ci95"][0]
    # one-sided bootstrap p-value for H0: difference <= -delta
    p = float((np.sum(draws <= -delta) + 1) / (draws.size + 1))
    return {"metric": metric, "delta": delta, "difference": boot[metric]["difference"], "ci95_lower": lo,
            "non_inferior": bool(lo > -delta), "p_one_sided": p}


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject_h0": adj < alpha}
    return out


def strip_draws(obj):
    if isinstance(obj, dict):
        return {k: strip_draws(v) for k, v in obj.items() if k != "_draws"}
    if isinstance(obj, list):
        return [strip_draws(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
