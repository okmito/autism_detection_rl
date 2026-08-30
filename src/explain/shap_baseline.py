"""SHAP comparison baseline — §18.3"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

def shap_for_static_subset(records: List[Dict[str, Any]], state: Dict[str, Any], predictor) -> Dict[str, Any]:
    """SHAP applied only to matched static classifier using same observed subset.
    This is a lightweight kernel-SHAP-like approximation using perturbation,
    to avoid heavy shap dependency in tests. If shap is installed, use it optionally.
    Returns dict with values per observed feature.
    """
    try:
        import shap  # type: ignore
        # If available, real SHAP could be used; fallback still works
        pass
    except Exception:
        pass
    # Simple perturbation-based attribution: change each observed feature and measure delta
    observed = [j for j in range(state["n"]) if state["mask"][j]==1]
    base_p = float(predictor(state))
    attributions = {}
    for j in observed:
        s_pert = {"mask": state["mask"].copy(), "value": state["value"].copy(), "n": state["n"],
                  "questions_remaining": state.get("questions_remaining",0), "budget": state.get("budget", state["n"])}
        # flip value for attribution
        orig = s_pert["value"][j]
        flipped = 1 - int(orig) if orig in (0,1) else 0
        s_pert["value"][j]=flipped
        p2 = float(predictor(s_pert))
        attributions[f"A{j+1}"] = float(base_p - p2)
    return {"base_p": base_p, "attributions": attributions, "note": "SHAP comparison baseline — explains final prediction, not sequential acquisition (§18.4)"}
