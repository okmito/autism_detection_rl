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
from sklearn.model_selection import StratifiedKFold

from src.data.ingest import load_dataset
from src.audits.circularity import audit_circularity
from src.env.state import init_state, update_state, get_legal_items
from src.env.environment import run_episode
from src.models.masked_predictor import MaskedPredictor
from src.policies.greedy import GreedyIGPolicy
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
CACHE_PT = REPO / "results" / f"demo_model_saudi_seed{SEED}_{DEMO_CALIBRATION}.pt"
CACHE_PKL = REPO / "results" / f"demo_model_saudi_seed{SEED}_{DEMO_CALIBRATION}.pkl"

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
    """Same deterministic 4-fold stratified split scheme as scripts/step2."""
    y = np.array([r["label"] for r in records])
    idx = np.arange(len(records))
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    tr, te = next(skf.split(idx, y))
    tr, va = next(skf.split(tr, y[tr]))
    return [records[i] for i in tr], [records[i] for i in va], [records[i] for i in te]


def build_predictor(retrain: bool = False):
    records = load_dataset("saudi")
    train, val, test = split(records)
    pred = MaskedPredictor(n_items=N_ITEMS, hidden=[128, 64], calibration=DEMO_CALIBRATION)
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
        return {"question": item_payload(j), "belief": predict(self.state), "asked": 0}

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
        self.state = update_state(self.state, j, int(value), qr)
        self.state["questions_remaining"] = qr
        self.state["budget"] = BUDGET
        self.asked += 1
        self.items_asked.append(j)
        p_after = predict(self.state)
        step = {"step": self.asked, "item": f"A{j + 1}", "value": int(value),
                "belief_before": p_before, "belief_after": p_after}
        self.trace.append(step)
        if self.asked >= BUDGET or not self._legal():
            return {"finished": True, "result": self._finish("budget_exhausted")}
        nj = int(self.policy(self.state, self._legal()))
        self.pending = nj
        return {"finished": False, "belief_before": p_before, "belief_after": p_after,
                "trace_step": step, "asked": self.asked, "next": item_payload(nj)}

    def _finish(self, reason: str) -> dict:
        p = predict(self.state)
        decision = "REFERRAL_RECOMMENDED" if p >= TAU else "NO_REFERRAL_INDICATED"
        cf = find_counterfactual(self.state, p, predict, tau=TAU)
        self.finished = True
        return {"p_hat": p, "decision": decision, "asked": self.asked,
                "items_asked": [f"A{j + 1}" for j in self.items_asked],
                "trace": self.trace, "counterfactual": cf,
                "true_label": int(self.record["label"]), "stop_reason": reason}

    def run_auto(self) -> dict:
        ep = run_episode(self.record, BUDGET, 0, self.policy, predict, 0.0, tau=TAU)
        cf = find_counterfactual(ep["final_state"], ep["p_hat"], predict, tau=TAU)
        self.finished = True
        return {"p_hat": float(ep["p_hat"]),
                "decision": "REFERRAL_RECOMMENDED" if ep["p_hat"] >= TAU else "NO_REFERRAL_INDICATED",
                "asked": len(ep["items_asked"]),
                "items_asked": [f"A{j + 1}" for j in ep["items_asked"]],
                "trace": ep["trace"], "counterfactual": cf,
                "true_label": int(self.record["label"]), "stop_reason": ep["stop_reason"]}


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
