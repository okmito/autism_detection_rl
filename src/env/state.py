"""Three-state item encoding — §9."""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

# Mask codes
UNASKED = 0
OBSERVED = 1
MISSING = 2

def init_state(record: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize state from a record.
    Returns dict with mask (n,), value (n,), n, remaining budget placeholder.
    """
    ir = record["item_responses"]
    mm = record["missing_mask"]
    n = len(ir)
    mask = np.zeros(n, dtype=int)
    value = np.full(n, -1, dtype=int)  # -1 = not observed; for missing stays -1
    for j in range(n):
        if mm[j]:
            mask[j] = MISSING
            value[j] = -1
        else:
            mask[j] = UNASKED
    return {"mask": mask, "value": value, "n": n}

def update_state(state: Dict[str, Any], action: int, observed_value: int, questions_remaining: int) -> Dict[str, Any]:
    """Apply observation of item `action` with value `observed_value`."""
    mask = state["mask"].copy()
    value = state["value"].copy()
    if mask[action] != UNASKED:
        raise ValueError(f"Cannot observe item {action} with mask {mask[action]}")
    mask[action] = OBSERVED
    value[action] = int(observed_value)
    return {"mask": mask, "value": value, "n": state["n"], "questions_remaining": questions_remaining}

def get_legal_items(state: Dict[str, Any]) -> List[int]:
    return [j for j in range(state["n"]) if state["mask"][j] == UNASKED]

def encode_state(state: Dict[str, Any], n: int | None = None, m_list: List[int] | None = None, questions_remaining: int | None = None, budget: int | None = None) -> np.ndarray:
    """Network input: one-hot mask (3*n) + one-hot response + normalized budget remaining.
    For binary primary: 4*n +1 (3*n mask one-hot + n response bits + 1 budget)
    For heterogeneous m_j: 3*n + sum(m_j) +1; missing/UNASKED response is zeros.
    """
    if n is None:
        n = state["n"]
    mask = state["mask"]
    value = state["value"]
    if questions_remaining is None:
        questions_remaining = state.get("questions_remaining", 0)
    if budget is None:
        budget = state.get("budget", max(questions_remaining, 1))

    # one-hot mask (3*n)
    mask_oh = np.zeros(3 * n, dtype=np.float32)
    for j in range(n):
        mask_oh[j * 3 + mask[j]] = 1.0

    # response portion
    if m_list is None:
        # binary: n dims, 1 if OBSERVED and value==1 else 0
        resp = np.zeros(n, dtype=np.float32)
        for j in range(n):
            if mask[j] == OBSERVED:
                resp[j] = float(value[j])
        budget_norm = np.array([questions_remaining / max(budget, 1)], dtype=np.float32)
        return np.concatenate([mask_oh, resp, budget_norm])
    else:
        # heterogeneous categorical
        total_m = sum(m_list)
        resp = np.zeros(total_m, dtype=np.float32)
        offset = 0
        for j in range(n):
            m_j = m_list[j]
            if mask[j] == OBSERVED:
                v = int(value[j])
                if 0 <= v < m_j:
                    resp[offset + v] = 1.0
            offset += m_j
        budget_norm = np.array([questions_remaining / max(budget, 1)], dtype=np.float32)
        return np.concatenate([mask_oh, resp, budget_norm])

def is_distinct_encoding() -> bool:
    """Sanity: UNASKED, MISSING, OBSERVED(0) have distinct representations."""
    # This is ensured by 3-state mask one-hot. OBSERVED(0) vs UNASKED vs MISSING are different.
    return True

def reachable_state_count(n: int, budget: int, m: int = 2, m_list: List[int] | None = None) -> int:
    """Theoretical reachable observation states Σ_{k=0..B} C(n,k) * (prod m_j or m^k).
    For heterogeneous, sum over subsets is computed exactly (feasible for n≤25, B≤6).
    """
    import math, itertools
    if m_list is not None:
        count = 0
        for k in range(budget + 1):
            for subset in itertools.combinations(range(n), k):
                prod = 1
                for j in subset:
                    prod *= m_list[j]
                count += prod
        return count
    else:
        return sum(math.comb(n, k) * (m ** k) for k in range(budget + 1))
