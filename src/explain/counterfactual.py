"""Counterfactual — §18.2"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import copy

def find_counterfactual(state: Dict[str, Any], p_hat: float, predictor, tau: float = 0.5, costs=None) -> Optional[Dict[str, Any]]:
    """Search observed items for minimum single-response flip that crosses tau.
    Returns dict with item, original_value, flipped_value, new_p if found, else None (robust).
    Tie-break: minimum cost (if costs provided) else minimum index.
    """
    decision = 1 if p_hat >= tau else 0
    candidates = []
    for j in range(state["n"]):
        if state["mask"][j] != 1:
            continue
        orig = int(state["value"][j])
        # For binary primary, flip 0->1,1->0; for multi-category, try all other values (assume binary for now)
        for flipped in [0,1]:
            if flipped == orig:
                continue
            # create perturbed state
            s2 = {"mask": state["mask"].copy(), "value": state["value"].copy(), "n": state["n"],
                  "questions_remaining": state.get("questions_remaining",0), "budget": state.get("budget", state["n"])}
            s2["value"][j] = flipped
            p2 = float(predictor(s2))
            dec2 = 1 if p2 >= tau else 0
            if dec2 != decision:
                cost = costs[j] if costs is not None else 1
                candidates.append((cost, j, orig, flipped, p2))
    if not candidates:
        return None  # robust
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, j, orig, flipped, p2 = candidates[0]
    return {"item": f"A{j+1}", "item_idx": j, "original_value": orig, "flipped_value": flipped, "new_p": float(p2)}
