"""Live browser demo for the adaptive screening project — stdlib only.

Composes EXISTING research modules only (ingest, circularity audit, state,
environment, masked predictor, policies, counterfactual, exact DP). No research
logic is changed. Serves one static page + a tiny JSON API.

Run from repo root (Windows cmd):
    .venv-win\\Scripts\\activate
    python scripts\\demo_app.py
Then open http://127.0.0.1:8000

First start trains the predictor (~20 s on CPU) and caches it under
results/demo_model_saudi_seed0.{pt,pkl}; later starts are instant.
Use --retrain to force retraining.

Terminology lock: screening / referral recommendation ONLY, never diagnosis.
Label circularity status is shown in the UI (spec 16.1 gate).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import torch
from src.data.splits import stratified_split

from src.data.ingest import load_dataset
from src.audits.circularity import audit_circularity
from src.env.state import init_state, update_state, get_legal_items
from src.env.environment import STOP, run_episode
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy, entropy
from src.policies.random_policy import RandomPolicy
from src.explain.counterfactual import find_counterfactual

BUDGET, TAU, SEED, N_ITEMS = 6, 0.5, 0, 10
# Demo uses Platt calibration on logits: keeps posteriors continuous instead of
# saturating at exactly 0/1 like isotonic does on this circular dataset
# (audit: isotonic has 4 breakpoints on 95 val records and maps everything to
# {0.0, 0.477, 1.0} - mathematically correct but unusable for a live demo).
# The offline research artifacts (Step 2/3) still use isotonic per configs/config.yaml.
DEMO_CALIBRATION = "platt"
STATIC = Path(__file__).resolve().parent / "demo_static"
# The cache filename encodes the predictor version. It previously did not, so a
# change to MaskedPredictor's training or calibration semantics left a stale v1
# cache on disk and the demo silently kept serving the old weights.
CACHE_STEM = (f"demo_model_saudi_seed{SEED}_{DEMO_CALIBRATION}"
              f"_v{MaskedPredictor.VERSION}")
CACHE_PT = REPO / "results" / f"{CACHE_STEM}.pt"
CACHE_PKL = REPO / "results" / f"{CACHE_STEM}.pkl"

# Short illustrative wording of the 10 Q-CHAT-10 items. In the dataset's binary
# encoding, 1 = atypical/concerning response, 0 = age-typical response.
ITEMS = [
    "Does your child look at you when you call their name?",
    "Can you easily get eye contact with your child?",
    "When you speak to your child, do they look at you and pay attention?",
    "Does your child point to ask for something they want?",
    "Does your child point to share interest with you?",
    "Does your child engage in pretend play?",
    "Does your child follow where you are looking?",
    "If someone is visibly upset, does your child try to comfort them?",
    "Were your child's first words typical for their age?",
    "Does your child use simple gestures typical for their age?",
]

_LOCK = threading.Lock()
S: dict = {"predictor": None, "train": None, "test": None, "records": None,
           "counter": 0, "sessions": {}}


def predict(state: dict) -> float:
    with _LOCK:
        return float(S["predictor"](state))


def split(records):
    """Canonical 4-fold stratified split, shared with every step script.

    The local copy this replaces overlapped the validation set with both training
    and test records, because it passed relative indices from a nested
    ``skf.split`` straight into the record list. See ``src/data/splits.py``.
    """
    return stratified_split(records, seed=SEED)


def build_predictor(retrain: bool = False):
    records = load_dataset("saudi")
    train, val, test = split(records)
    pred = MaskedPredictor(n_items=N_ITEMS, hidden=[128, 64], calibration=DEMO_CALIBRATION, seed=SEED)
    if not retrain and CACHE_PT.exists() and CACHE_PKL.exists():
        try:
            pred.model.load_state_dict(torch.load(CACHE_PT, map_location="cpu"))
            with open(CACHE_PKL, "rb") as f:
                pred.calibrator = pickle.load(f)
            print(f"model: loaded cached weights ({CACHE_PT.name})")
            return records, train, val, test, pred
        except Exception as e:  # fall through to retraining
            print(f"cache load failed ({e}); retraining")
    print(f"model: training MaskedMLP[128,64] + {DEMO_CALIBRATION} on Saudi 506 ... (~20 s)")
    pred.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=SEED)
    pred.fit_calibrator(val, method=DEMO_CALIBRATION)
    try:
        CACHE_PT.parent.mkdir(exist_ok=True)
        torch.save(pred.model.state_dict(), CACHE_PT)
        with open(CACHE_PKL, "wb") as f:
            pickle.dump(pred.calibrator, f)
        print("model: cached for instant startup (delete results/demo_model_* to retrain)")
    except Exception as e:
        print(f"model: caching skipped ({e})")
    return records, train, val, test, pred


def make_policy(name: str):
    if name == "random":
        # seed=None: each episode draws fresh entropy, so two Random runs differ;
        # greedy remains fully deterministic (argmax over information gain).
        return RandomPolicy(seed=None)
    return GreedyIGPolicy(S["train"], N_ITEMS)


def item_payload(j: int) -> dict:
    return {"index": int(j), "code": f"A{j + 1}", "text": ITEMS[int(j)]}


# --------------------------------------------------------------------------
# selection diagnostic
# --------------------------------------------------------------------------
# Why this exists
# ---------------
# The demo asks a viewer to watch which question the policy picks next. On this
# dataset it very often picks the lowest-numbered unasked item in order, and
# without an explanation that reads as a broken policy.
#
# It is not a bug, and this is the measured reason. ``GreedyIGPolicy`` selects on
# ``IG = H(Y|s) - E_v[H(Y|s,j,v)]`` over the *raw* empirical support mean. Once
# the support is **pure** — every training record consistent with the answers so
# far carries the same label — ``H(Y|s) = 0`` and every child support is pure too,
# so **every legal item's IG is exactly 0**. ``argmax`` then returns the first
# maximum, which is ``greedy.py``'s ``best_j = items_only[0]`` initialisation.
# The criterion is vacuous, not tied.
#
# Measured on the held-out test split at B=6: the support is already pure at
# 61.3% of decision states, and the criterion is identically zero at 195 of 195
# pure-support states (`POLICY_BENCHMARK_REPORT.md` §A.6, `diagnosisReady.md`
# §0A.1). A Beta(1,1)-smoothed posterior fixes the degeneracy
# (`src/policies/beta_greedy.py`) but the demo deliberately still runs the
# shipped greedy policy, so the UI reports the degeneracy rather than hiding it.
#
# ``GreedyIGPolicy`` is intentionally NOT modified, so this recomputes its
# formula rather than calling into it. ``tests/test_demo_behavior_audit.py::
# test_selection_diagnostic_agrees_with_policy_choice`` asserts the two agree,
# which is what keeps this from drifting away from the real policy.
DEGENERACY_TOL = 1e-12


def selection_diagnostic(policy, state, legal) -> dict:
    """Explain the item choice at ``state``: support size, purity, IG spread."""
    items_only = [a for a in legal if a != STOP]
    out = {
        "applicable": hasattr(policy, "_support"),
        "n_legal": len(items_only),
        "support_size": None, "n_pos": None, "posterior": None,
        "support_is_pure": None, "ig_spread": None, "criterion_vacuous": None,
        "tie_at_max": None, "top_items": [],
    }
    if not out["applicable"] or not items_only:
        out["criterion_vacuous"] = None
        return out

    idx = policy._support(state["mask"], state["value"])
    out["support_size"] = int(len(idx))
    if len(idx) == 0:
        out["criterion_vacuous"] = True
        return out

    ys = policy.y[idx]
    total = len(idx)
    n_pos = int(ys.sum())
    p_s = float(ys.mean())
    h_s = entropy(p_s)
    out["n_pos"] = n_pos
    out["posterior"] = p_s
    out["support_is_pure"] = bool(n_pos in (0, total))

    ig: dict[int, float] = {}
    for j in items_only:
        sub = idx[~policy.missing_mask[idx, j]]
        if len(sub) == 0:
            continue
        exp_h = 0.0
        vals, counts = np.unique(policy.X[sub, j], return_counts=True)
        for v, c in zip(vals, counts):
            child = idx[(~policy.missing_mask[idx, j]) & (policy.X[idx, j] == v)]
            if len(child) == 0:
                continue
            exp_h += (c / total) * entropy(float(policy.y[child].mean()))
        n_miss = int(np.sum(policy.missing_mask[idx, j]))
        if n_miss > 0:
            child = idx[policy.missing_mask[idx, j]]
            p_child = float(policy.y[child].mean()) if len(child) > 0 else 0.5
            exp_h += (n_miss / total) * entropy(p_child)
        ig[j] = h_s - exp_h

    if not ig:
        out["criterion_vacuous"] = True
        return out

    best = max(ig.values())
    worst = min(ig.values())
    out["ig_spread"] = best - worst
    # A spread of zero means every scored item is indistinguishable, so argmax
    # carries no information and the lowest legal index wins by default.
    out["criterion_vacuous"] = bool((best - worst) <= DEGENERACY_TOL)
    tied = sorted(j for j, v in ig.items() if best - v <= DEGENERACY_TOL)
    out["tie_at_max"] = len(tied)
    out["top_items"] = [{"code": f"A{j + 1}", "ig": ig[j]} for j in tied[:6]]
    return out


def read_offline() -> dict:
    """Read pre-generated artifacts from results/ (defensive: all optional)."""
    out: dict = {}
    p = REPO / "results" / "predictor_saudi_metrics.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            out.update({"test_brier": d.get("test_brier"), "test_auroc": d.get("test_auroc"),
                        "source": d.get("source")})
        except Exception:
            pass
    sw = REPO / "results" / "dp_tractability_sweep.json"
    if sw.exists():
        try:
            rows = json.loads(sw.read_text())
            out["dp_runs"] = len(rows)
            out["dp_all_optimal"] = all(r.get("status") == "optimal" for r in rows)
        except Exception:
            pass
    pf = REPO / "results" / "perf_vs_budget_saudi.json"
    if pf.exists():
        try:
            d = json.loads(pf.read_text())
            row = next((r for r in d.get("rows", []) if r.get("B") == 6), None)
            if row:
                out["b6"] = {k: row.get(k) for k in
                             ("policy_greedy_brier", "policy_random_brier",
                              "policy_greedy_uar", "policy_random_uar")}
        except Exception:
            pass
    # H1: adaptive vs the exact best fixed subset at matched budget. This is the
    # comparator that actually matters, and the demo should not imply the
    # adaptive policy wins when it does not.
    sb = REPO / "results" / "step5_policy_benchmark_saudi.json"
    if sb.exists():
        try:
            rows = json.loads(sb.read_text()).get("per_budget", [])
            h1 = {}
            for r in rows:
                if r.get("policy") in ("greedy", "exact_fixed_subset"):
                    # The artifact key is `brier` (held-out test split); the
                    # comparator that matters is the exact best fixed subset,
                    # and a LOWER brier is better.
                    h1.setdefault(r["B"], {})[r["policy"]] = r.get("brier")
            out["h1"] = {
                str(b): {
                    "greedy": v.get("greedy"),
                    "fixed": v.get("exact_fixed_subset"),
                    "supported": (v.get("greedy") is not None
                                  and v.get("exact_fixed_subset") is not None
                                  and v["greedy"] < v["exact_fixed_subset"]),
                }
                for b, v in sorted(h1.items())
                if v.get("greedy") is not None and v.get("exact_fixed_subset") is not None
            }
        except Exception:
            pass
    # V-6 gate status, read from the artifacts so the UI cannot drift from them.
    v6: dict = {"signed_off": False, "threshold_selected": False}
    ls = REPO / "results" / "lambda_sweep_saudi.json"
    if ls.exists():
        try:
            d = json.loads(ls.read_text())
            v6["lambdas"] = d.get("lambdas")
            v6["caveat"] = d.get("lambda_units_caveat")
            v6["saturated_at_zero_lambda"] = any(
                r.get("v_star_saturated") for r in d.get("rows", []))
        except Exception:
            pass
    es = REPO / "results" / "evoi_scale_saudi.json"
    if es.exists():
        try:
            d = json.loads(es.read_text())
            a = d.get("analysis", {})
            v6["no_threshold_selected"] = bool(a.get("no_threshold_selected"))
            v6["support_evoi_median"] = (
                a.get("by_split", {}).get("test", {}).get("overall", {})
                 .get("support_evoi_max_over_legal", {}).get("median"))
            v6["predictor_evoi_median"] = (
                a.get("by_split", {}).get("test", {}).get("overall", {})
                 .get("predictor_gain_max_over_legal", {}).get("median"))
        except Exception:
            pass
    out["v6"] = v6
    return out


def meta() -> dict:
    recs = S["records"]
    audit = audit_circularity(recs) if recs else {}
    y = np.array([r["label"] for r in recs]) if recs else np.array([0, 0])
    return {
        "budget": BUDGET, "tau": TAU, "n_items": N_ITEMS,
        "items": [{"code": f"A{i + 1}", "text": t} for i, t in enumerate(ITEMS)],
        "dataset": {"name": "Saudi toddler screening (Q-CHAT-10, real CSV)",
                    "n": int(len(recs)) if recs else 0,
                    "positive": int(y.sum()), "negative": int((1 - y).sum())},
        "circularity": {"threshold": audit.get("best_threshold"),
                        "exact_match": audit.get("exact_match_rate"),
                        "classification": audit.get("classification")},
        "policies_available": ["greedy", "random"],
        "offline": read_offline(),
    }


class Session:
    def __init__(self, mode: str, policy_name: str):
        test = S["test"]
        i = S["counter"] % len(test)
        S["counter"] += 1
        self.record = test[i]
        self.policy = make_policy(policy_name)
        self.policy_name = policy_name
        self.state = init_state(self.record)
        self.state["questions_remaining"] = BUDGET
        self.state["budget"] = BUDGET
        self.asked = 0
        self.items_asked: list[int] = []
        self.trace: list[dict] = []
        self.pending: int | None = None
        self.finished = False

    def _legal(self) -> list[int]:
        return get_legal_items(self.state)

    def first_question(self) -> dict:
        j = self.policy(self.state, self._legal())
        self.pending = int(j)
        return {"question": item_payload(j), "belief": predict(self.state), "asked": 0,
                "selection": selection_diagnostic(self.policy, self.state,
                                                  self._legal() + [STOP])}

    def answer(self, value: int | None = None, stop: bool = False) -> dict:
        if self.finished:
            raise ValueError("session already finished")
        if stop:
            if self.asked < 1:
                raise ValueError("answer at least one question before stopping")
            return {"finished": True, "result": self._finish("user_stop")}
        if value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        if self.pending is None:
            raise ValueError("no pending question")
        j = self.pending
        p_before = predict(self.state)
        qr = self.state["questions_remaining"] - 1
        # Diagnose the choice that was just made, evaluated at the state it was
        # made from, so the trace explains each pick rather than the next one.
        sel = selection_diagnostic(self.policy, self.state, self._legal() + [STOP])
        self.state = update_state(self.state, j, int(value), qr)
        self.state["questions_remaining"] = qr
        self.state["budget"] = BUDGET
        self.asked += 1
        self.items_asked.append(j)
        p_after = predict(self.state)
        step = {"step": self.asked, "item": f"A{j + 1}", "value": int(value),
                "belief_before": p_before, "belief_after": p_after,
                "selection": sel}
        self.trace.append(step)
        if self.asked >= BUDGET or not self._legal():
            return {"finished": True, "result": self._finish("budget_exhausted")}
        nj = int(self.policy(self.state, self._legal()))
        self.pending = nj
        return {"finished": False, "belief_before": p_before, "belief_after": p_after,
                "trace_step": step, "asked": self.asked, "next": item_payload(nj),
                "selection": selection_diagnostic(self.policy, self.state,
                                                  self._legal() + [STOP])}

    def _selection_summary(self) -> dict:
        """Aggregate the per-step diagnostics into one honest headline number."""
        steps = [t.get("selection") or {} for t in self.trace]
        scored = [s for s in steps if s.get("ig_spread") is not None]
        vacuous = [s for s in scored if s.get("criterion_vacuous")]
        pure = [s for s in steps if s.get("support_is_pure") is True]
        return {
            "steps_diagnosed": len(scored),
            "steps_criterion_vacuous": len(vacuous),
            "steps_support_pure": len(pure),
            "support_pure_at_stop": (pure[-1]["support_is_pure"]
                                     if pure else None),
            "support_size_at_stop": (steps[-1].get("support_size")
                                     if steps else None),
        }

    def _finish(self, reason: str) -> dict:
        p = predict(self.state)
        decision = "REFERRAL_RECOMMENDED" if p >= TAU else "NO_REFERRAL_INDICATED"
        cf = find_counterfactual(self.state, p, predict, tau=TAU)
        self.finished = True
        return {"p_hat": p, "decision": decision, "asked": self.asked,
                "items_asked": [f"A{j + 1}" for j in self.items_asked],
                "trace": self.trace, "counterfactual": cf,
                "true_label": int(self.record["label"]), "stop_reason": reason,
                "selection_summary": self._selection_summary()}

    def run_auto(self) -> dict:
        ep = run_episode(self.record, BUDGET, 0, self.policy, predict, 0.0, tau=TAU)
        cf = find_counterfactual(ep["final_state"], ep["p_hat"], predict, tau=TAU)
        # run_episode keeps its own trace and its own item list; adopt the item
        # list so the diagnostic replay below walks the same sequence.
        self.items_asked = [int(a) for a in ep["items_asked"]]
        # run_episode keeps its own trace, so attach the per-step diagnostic here
        # rather than changing the environment (which the RL trainers share).
        for st, t in enumerate(self._replayed_trace()):
            if st < len(ep["trace"]):
                ep["trace"][st]["selection"] = t.get("selection")
        self.trace = ep["trace"]
        self.finished = True
        return {"p_hat": float(ep["p_hat"]),
                "decision": "REFERRAL_RECOMMENDED" if ep["p_hat"] >= TAU else "NO_REFERRAL_INDICATED",
                "asked": len(ep["items_asked"]),
                "items_asked": [f"A{j + 1}" for j in ep["items_asked"]],
                "trace": ep["trace"], "counterfactual": cf,
                "true_label": int(self.record["label"]), "stop_reason": ep["stop_reason"],
                "selection_summary": self._selection_summary()}

    def _replayed_trace(self) -> list[dict]:
        """Re-walk the realised item sequence, diagnosing each decision point.

        Read-only with respect to ``self.state``, so it can run after
        ``run_episode`` has finished without disturbing the episode.
        """
        if not self.policy_name_is_greedy():
            return []
        state = init_state(self.record)
        state["questions_remaining"] = BUDGET
        state["budget"] = BUDGET
        out: list[dict] = []
        for depth, a in enumerate(self.items_asked):
            legal = get_legal_items(state) + [STOP]
            sel = selection_diagnostic(self.policy, state, legal)
            out.append({"step": depth + 1, "item": f"A{a + 1}", "selection": sel})
            value = self.record["item_responses"][a]
            if np.isnan(value):
                break
            state = update_state(state, a, int(value), BUDGET - depth - 1)
            state["budget"] = BUDGET
            state["questions_remaining"] = BUDGET - depth - 1
        return out

    def policy_name_is_greedy(self) -> bool:
        return isinstance(self.policy, GreedyIGPolicy)



def api_start(req: dict) -> dict:
    mode = req.get("mode", "interactive")
    pol = req.get("policy", "greedy")
    if mode not in ("interactive", "auto") or pol not in ("greedy", "random"):
        raise ValueError("mode must be interactive|auto; policy must be greedy|random")
    sid = uuid.uuid4().hex[:12]
    s = Session(mode, pol)
    S["sessions"][sid] = s
    if mode == "auto":
        return {"session_id": sid, "mode": mode, "budget": BUDGET, "tau": TAU,
                "policy": pol, "result": s.run_auto()}
    return {"session_id": sid, "mode": mode, "budget": BUDGET, "tau": TAU,
            "policy": pol, **s.first_question()}


def api_answer(req: dict) -> dict:
    sid = req.get("session_id", "")
    s = S["sessions"].get(sid)
    if s is None:
        raise ValueError("unknown session - start a new one")
    return s.answer(value=req.get("value"), stop=bool(req.get("stop")))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._json(meta())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/session/start":
                self._json(api_start(req))
            elif self.path == "/api/session/answer":
                self._json(api_answer(req))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 400)

    def log_message(self, *a):  # keep the demo console clean
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--retrain", action="store_true", help="ignore cached model and retrain")
    args = ap.parse_args()

    records, train, _val, test, pred = build_predictor(retrain=args.retrain)
    S.update({"records": records, "train": train, "test": test, "predictor": pred})

    port = args.port
    for p in range(port, port + 10):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    else:
        raise SystemExit("no free port in range")

    print()
    print("=" * 62)
    print("  Adaptive Screening - LIVE DEMO")
    print(f"  open in your browser:  http://127.0.0.1:{port}")
    print("  stop the server with Ctrl+C")
    print("=" * 62)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
