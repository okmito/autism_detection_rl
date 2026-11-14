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
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # entry-point script; no os.chdir - data paths are anchored to the repo root (T5)

import numpy as np
from src.data.splits import stratified_split

from src.audits.circularity import audit_circularity
from src.policies.greedy import GreedyIGPolicy
from src.policies.random_policy import RandomPolicy

BUDGET, TAU, SEED, N_ITEMS = 6, 0.5, 0, 10
# Demo uses Platt calibration on logits: keeps posteriors continuous instead of
# saturating at exactly 0/1 like isotonic does on this circular dataset
# (audit: isotonic has 4 breakpoints on 95 val records and maps everything to
# {0.0, 0.477, 1.0} - mathematically correct but unusable for a live demo).
# The offline research artifacts (Step 2/3) still use isotonic per configs/config.yaml.
DEMO_CALIBRATION = "platt"
STATIC = Path(__file__).resolve().parent / "demo_static"
# Cache location, key (dataset fingerprint) and hash verification: src/screening/model_cache.py.

# The wording is derived from the verified Q-CHAT-10 feature contract
# (src/data/qchat10_contract.py) rather than restated here. A hand-written copy
# previously drifted: from A3 onward the demo showed one construct while the
# answer was recorded as the feature for a different one. tests/test_support_
# questionnaire.py asserts this list equals the contract's items.
from src.screening import model_cache, session as engine
# selection_diagnostic and DEGENERACY_TOL are re-exported: existing tests and callers use demo_app.<name>.
from src.screening.diagnostics import (DEGENERACY_TOL, ITEMS, item_payload,  # noqa: F401
                                       selection_diagnostic)  # noqa: F401

assert len(ITEMS) == N_ITEMS, "questionnaire contract must supply exactly 10 items"

_LOCK = threading.Lock()
S: dict = {"predictor": None, "train": None, "test": None, "records": None,
           "counter": 0, "sessions": {}}

# The demo server keeps sessions in memory for its whole lifetime, so without a cap a long-running (or
# scripted) demo grows without bound. Finished sessions are dropped before unfinished ones.
MAX_SESSIONS = 200
_SESSIONS_LOCK = threading.Lock()


def next_record_index(n: int) -> int:
    """Cycle through the held-out records; the counter is advanced under a lock (the server is threaded)."""
    with _SESSIONS_LOCK:
        i = S["counter"] % n
        S["counter"] += 1
    return i


def register_session(sid: str, session) -> None:
    """Store a session, evicting the oldest finished (else the oldest) ones beyond ``MAX_SESSIONS``."""
    with _SESSIONS_LOCK:
        sessions = S["sessions"]
        sessions[sid] = session
        while len(sessions) > MAX_SESSIONS:
            victim = next((k for k, v in sessions.items() if k != sid and getattr(v, "finished", False)), None)
            if victim is None:
                victim = next(k for k in sessions if k != sid)
            del sessions[victim]


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
    """Load or train the demo predictor through the shared, hash-verified cache (src/screening/model_cache.py)."""
    return model_cache.build_predictor(
        REPO / "results", seed=SEED, n_items=N_ITEMS, calibration=DEMO_CALIBRATION, retrain=retrain,
        info=lambda m: print(f"model: {m}"), warn=lambda m: print(f"model: {m}"))


def make_policy(name: str):
    """Greedy Information Gain is the default (configs/config.yaml: policy.type).
    Anything other than an explicitly supported policy must be rejected, not
    silently treated as greedy."""
    if name in ("greedy", None):
        return GreedyIGPolicy(S["train"], N_ITEMS)
    if name == "random":
        # seed=None: each episode draws fresh entropy, so two Random runs differ;
        # greedy remains fully deterministic (argmax over information gain).
        return RandomPolicy(seed=None)
    if name == "beta_greedy":
        from src.policies.beta_greedy import BetaGreedyPolicy
        return BetaGreedyPolicy(S["train"], N_ITEMS)
    raise ValueError(f"unsupported policy {name!r}: supported = greedy (default), random, beta_greedy")


# item_payload / selection_diagnostic / DEGENERACY_TOL (and the long explanation of why the diagnostic exists)
# live in src/screening/diagnostics.py, shared with the web app. They are imported at the top of this file
# so ``demo_app.selection_diagnostic`` and ``demo_app.item_payload`` keep working for existing callers.


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
    """An in-memory demo session. The step/finish logic is ``src.screening.session`` (shared with the
    web app); this class only owns the state and binds it to a held-out dataset record."""

    def __init__(self, mode: str, policy_name: str):
        test = S["test"]
        i = next_record_index(len(test))
        self.record = test[i]
        self.policy = make_policy(policy_name)
        self.policy_name = policy_name
        self.state = engine.record_state(self.record, BUDGET)
        self.asked = 0
        self.items_asked: list[int] = []
        self.trace: list[dict] = []
        self.pending: int | None = None
        self.finished = False
        # P4: the finished session is the input to the support layer. Both are kept so the support
        # report can be assembled without re-running anything (read-only with respect to the outcome).
        self.result_id = f"demo-{uuid.uuid4().hex[:8]}"
        self.episode_result: dict | None = None
        self.explanation: dict | None = None
        self._support_report: dict | None = None

    @staticmethod
    def _ctx() -> engine.Context:
        return engine.Context(
            predict=predict, predictor=S["predictor"], train=S["train"] or [], records=S["records"] or [],
            n_items=N_ITEMS, budget=BUDGET, tau=TAU,
            tag="demo — research prototype, not a diagnostic device", input_reference="demo-session")

    def first_question(self) -> dict:
        j, sel = engine.next_question(self._ctx(), self.policy, self.state)
        self.pending = j
        return {"question": item_payload(j), "belief": predict(self.state), "asked": 0, "selection": sel}

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
        ctx = self._ctx()
        j = self.pending
        self.state, step = engine.apply_answer(ctx, self.policy, self.state, j, int(value), self.asked)
        self.asked += 1
        self.items_asked.append(j)
        self.trace.append(step)
        if engine.is_exhausted(ctx, self.state, self.asked):
            return {"finished": True, "result": self._finish("budget_exhausted")}
        self.pending, next_sel = engine.next_question(ctx, self.policy, self.state)
        return {"finished": False, "belief_before": step["belief_before"], "belief_after": step["belief_after"],
                "trace_step": step, "asked": self.asked, "next": item_payload(self.pending), "selection": next_sel}

    def _adopt(self, out: dict) -> dict:
        self.finished = True
        self.episode_result = out["episode_result"]
        self.explanation = out["explanation"]
        self._support_report = out["support_report"]
        return out["result"]

    def _finish(self, reason: str) -> dict:
        return self._adopt(engine.finalize(
            self._ctx(), state=self.state, trace=self.trace, items_asked=self.items_asked, stop_reason=reason,
            result_id=self.result_id, label_source=self.record.get("label_source"), true_label=self.record["label"]))

    def run_auto(self) -> dict:
        out = engine.run_auto(self._ctx(), self.policy, self.record, result_id=self.result_id)
        self.items_asked = list(out["episode_result"]["items_asked"])
        self.trace = out["episode_result"]["trace"]
        return self._adopt(out)

    # ------------------------------------------------------------------
    # P4 — the support report is generated from the answers already given
    # ------------------------------------------------------------------

    def support_report(self) -> dict:
        """The support report for this finished session.

        Read-only with respect to the screening outcome: it was built from this session's own episode
        result and explanation when the session finished. No second questionnaire is asked anywhere.
        """
        if not self.finished or self._support_report is None:
            raise ValueError("finish the screening session first — the support "
                             "report follows the result, it never precedes it")
        return self._support_report


def api_start(req: dict) -> dict:
    mode = req.get("mode", "interactive")
    pol = req.get("policy") or "greedy"
    if mode not in ("interactive", "auto") or pol not in ("greedy", "random", "beta_greedy"):
        raise ValueError("mode must be interactive|auto; policy must be greedy|random")
    sid = uuid.uuid4().hex[:12]
    s = Session(mode, pol)
    register_session(sid, s)
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
