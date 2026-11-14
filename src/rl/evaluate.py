"""Roll an agent over held-out records and measure screening quality and question burden."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.eval.fast_stats import auroc
from src.eval.metrics import calibration_fit
from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import C_SKIP, MDPConfig, legal_mask, obs_val


def rollout(agent, X: np.ndarray, belief: NaiveBayesBelief, cfg: MDPConfig) -> Dict[str, np.ndarray]:
    """Run ``agent.act_batch`` on every record until it STOPs (or the cap forces STOP).

    Illegal actions raise: an agent that re-asks an item or exceeds the cap is a bug, never silently corrected.
    Returns per-record ``p`` (belief at STOP), ``n_q`` (questions used), ``forced`` (budget/pool forced the STOP),
    ``seq`` (item order, -1 padded), ``codes`` (final history).
    """
    X = np.asarray(X, float)
    M = np.isnan(X)
    Xi = np.where(M, 0, X).astype(np.int64)
    N, n = Xi.shape
    codes = np.zeros((N, n), dtype=np.int64)
    active = np.ones(N, bool)
    forced = np.zeros(N, bool)
    seq = np.full((N, n), -1, dtype=np.int64)
    for _ in range(n + 1):
        if not active.any():
            break
        rows = np.flatnonzero(active)
        c = codes[rows]
        lm = legal_mask(c, cfg)
        must_stop = ~lm[:, :-1].any(axis=1)
        a = np.asarray(agent.act_batch(c), dtype=np.int64)
        a = np.where(must_stop, n, a)
        if not np.all(lm[np.arange(len(rows)), a]):
            bad = np.flatnonzero(~lm[np.arange(len(rows)), a])
            raise ValueError(f"{getattr(agent, 'name', agent)} took an illegal action at rows {rows[bad][:5].tolist()}")
        stop = a == n
        forced[rows[stop & must_stop]] = True
        active[rows[stop]] = False
        ask = np.flatnonzero(~stop)
        if len(ask):
            rr, jj = rows[ask], a[ask]
            used = np.count_nonzero(codes[rr], axis=1)
            seq[rr, used] = jj
            codes[rr, jj] = np.where(M[rr, jj], C_SKIP, 1 + Xi[rr, jj])
    if active.any():
        raise RuntimeError("rollout did not terminate")
    o, v = obs_val(codes)
    return {"p": belief.prob_arrays(o, v), "n_q": np.count_nonzero(codes, axis=1), "forced": forced, "seq": seq,
            "codes": codes}


def screening_metrics(y: np.ndarray, p: np.ndarray, n_q: np.ndarray, tau: float = 0.5,
                      max_questions: Optional[int] = None, forced: Optional[np.ndarray] = None) -> Dict[str, Any]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n_q = np.asarray(n_q, float)
    d = (p >= tau).astype(int)
    sens = float(np.mean(d[y == 1] == 1)) if (y == 1).any() else float("nan")
    spec = float(np.mean(d[y == 0] == 0)) if (y == 0).any() else float("nan")
    cal = calibration_fit(y, p)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = float(sum(np.abs(p[bins == b].mean() - y[bins == b].mean()) * np.mean(bins == b)
                    for b in range(10) if (bins == b).any()))
    out = {"n": int(len(y)), "balanced_accuracy": 0.5 * (sens + spec), "accuracy": float(np.mean(d == y)),
           "sensitivity": sens, "specificity": spec, "auroc": auroc(y, p), "brier": float(np.mean((p - y) ** 2)),
           "ece_10bin": ece, "calibration_slope": cal["slope"], "calibration_intercept": cal["intercept"],
           "mean_questions": float(n_q.mean()), "median_questions": float(np.median(n_q)),
           "sd_questions": float(n_q.std()),
           "question_histogram": {int(k): int((n_q == k).sum()) for k in np.unique(n_q)}}
    if max_questions is not None:
        out["frac_at_cap"] = float(np.mean(n_q >= max_questions))
    if forced is not None:
        out["frac_forced_stop"] = float(np.mean(forced))
    return out


def adaptivity_profile(seq: np.ndarray, codes: np.ndarray) -> Dict[str, Any]:
    """How response-dependent the realised behaviour was."""
    seq = np.asarray(seq)
    first = seq[:, 0]
    seqs = {tuple(int(x) for x in s if x >= 0) for s in seq}
    lens = np.sum(seq >= 0, axis=1)
    # second question conditioned on first answer: does the choice differ between answers to the same first item?
    branch_differs = []
    for j in np.unique(first[first >= 0]):
        rows = first == j
        a0 = rows & (codes[:, j] == 1)
        a1 = rows & (codes[:, j] == 2)
        s0 = {int(x) for x in seq[a0, 1]}
        s1 = {int(x) for x in seq[a1, 1]}
        if a0.any() and a1.any():
            branch_differs.append(s0 != s1)
    return {"distinct_sequences": len(seqs), "distinct_first_items": int(len(np.unique(first[first >= 0]))),
            "distinct_lengths": int(len(np.unique(lens))),
            "second_item_depends_on_first_answer": (bool(any(branch_differs)) if branch_differs else None)}
