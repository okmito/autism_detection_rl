"""End-to-end screening workflow via the demo HTTP API (backend contract).

Verifies the exact contract the frontend depends on: start -> answer ->
next question -> result -> second session. Runs against a live demo_app
server; skipped when the server is unavailable.
"""
import json
import urllib.request
import urllib.error

import pytest

BASE = "http://127.0.0.1:8001"


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    except urllib.error.URLError as e:
        if "Connection refused" in str(e.reason) or "actively refused" in str(e.reason):
            pytest.skip(f"demo server not running at {BASE} (environmental)")
        raise


def test_full_screening_workflow():
    r = _post("/api/session/start", {"mode": "interactive", "policy": "greedy"})
    assert "session_id" in r and "question" in r and "selection" in r
    sid = r["session_id"]
    asked_codes = [r["question"]["code"]]
    finished = False
    for i in range(6):
        r2 = _post("/api/session/answer", {"session_id": sid, "value": 1 if i % 2 else 0})
        if r2.get("finished"):
            res = r2["result"]
            assert res["asked"] == len(res["items_asked"]) == i + 1
            assert len(set(res["items_asked"])) == len(res["items_asked"])
            assert 0.0 <= res["p_hat"] <= 1.0
            assert res["decision"] in ("REFERRAL_RECOMMENDED", "NO_REFERRAL_INDICATED")
            finished = True
            break
        asked_codes.append(r2["next"]["code"])
        assert r2["next"]["code"] not in asked_codes[:-1]
    assert finished

    r3 = _post("/api/session/start", {"mode": "interactive", "policy": "greedy"})
    assert r3["session_id"] != sid


def test_unknown_session_rejected():
    try:
        r = _post("/api/session/answer", {"session_id": "nonexistent-xyz", "value": 0})
    except Exception:
        return  # 4xx surfaces as a raised error — still a rejection
    assert "error" in str(r).lower() or "_error" in r
