"""Decision trace — §18.1"""
from __future__ import annotations
from typing import List, Dict, Any

def format_trace(trace: List[Dict[str, Any]], stop_reason: str, final_p: float) -> List[Dict[str, Any]]:
    out = []
    for t in trace:
        out.append({"step": t["step"], "item": t["item"], "value": t["value"], "belief_before": t["belief_before"], "belief_after": t["belief_after"]})
    out.append({"stop_reason": stop_reason, "final": final_p})
    return out
