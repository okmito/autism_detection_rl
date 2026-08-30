"""Optional STreeD / DL8.5 / MurTree cross-checks — §14.2
These solvers are NOT authoritative; they are only valid when objective is equivalent.
This stub documents the compatibility check and provides a placeholder interface.
"""
from __future__ import annotations
from typing import Dict, Any, List

def check_compatibility(solver_name: str, objective: str) -> Dict[str, Any]:
    """Check if external solver's objective matches §14.1 Brier-minus-cost.
    Returns dict with compatible(bool) and reason.
    """
    # DL8.5/MurTree standard objective = misclassification/accuracy -> NOT compatible
    # STreeD supports broader separable objectives but must be verified per instance
    if solver_name in ("DL8.5", "MurTree") and objective == "empirical_brier_minus_cost":
        return {"compatible": False, "reason": f"{solver_name} standard objective is misclassification, not Brier-minus-cost (§14.2)"}
    if solver_name == "STreeD" and objective == "empirical_brier_minus_cost":
        return {"compatible": "unknown", "reason": "Requires formal separability proof per §14.2; not assumed compatible."}
    return {"compatible": False, "reason": "Unknown solver/objective combination"}

def run_external_solver(solver_name: str, records: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    comp = check_compatibility(solver_name, kwargs.get("objective", "empirical_brier_minus_cost"))
    if not comp["compatible"] is True:
        return {"status": "not_exact", "compatibility": comp, "note": "Result is NOT authoritative exact reference (§14.2). Report as surrogate/approximation only."}
    raise NotImplementedError(f"External solver {solver_name} integration not implemented in this stub.")
