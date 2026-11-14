"""End-to-end demo workflow verification (backend API level).

Runs: start -> answer x6 -> result -> new session. Verifies the API
contract the frontend depends on. Asserts result fidelity: the frontend
renders exactly what these responses contain.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8001"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    except Exception as e:
        return {"_error": str(e)}


def main():
    ok = True
    # 1. start
    r = post("/api/session/start", {"mode": "interactive", "policy": "greedy"})
    assert "session_id" in r, r
    sid = r["session_id"]
    assert "question" in r and "selection" in r, list(r.keys())
    print(f"start: session={sid} question={r['question']['code']} belief={r['belief']:.3f}")
    asked_codes = [r["question"]["code"]]
    # 2. answer until finished (max 6)
    for i in range(6):
        r2 = post("/api/session/answer", {"session_id": sid, "value": 1 if i % 2 else 0})
        if r2.get("finished"):
            res = r2["result"]
            print(f"finished at {i+1}: asked={res['asked']} items={res['items_asked']} "
                  f"p={res['p_hat']:.3f} decision={res['decision']} stop={res['stop_reason']}")
            assert res["asked"] == len(res["items_asked"]) == i + 1
            assert len(set(res["items_asked"])) == len(res["items_asked"]), "duplicate questions"
            assert 0.0 <= res["p_hat"] <= 1.0
            break
        asked_codes.append(r2["next"]["code"])
        assert r2["next"]["code"] not in asked_codes[:-1], "duplicate question displayed"
    else:
        raise AssertionError("session did not finish within budget")
    # 3. new session must not reuse the old one
    r3 = post("/api/session/start", {"mode": "interactive", "policy": "greedy"})
    assert r3["session_id"] != sid, "session id reused"
    print(f"new session: {r3['session_id']} != {sid} OK")
    # 4. unknown session id must error, not resurface the result
    r4 = post("/api/session/answer", {"session_id": "nonexistent", "value": 0})
    assert "_error" in r4 or "error" in str(r4).lower(), r4
    print("unknown session correctly rejected")
    print("E2E OK")

if __name__ == "__main__":
    main()
