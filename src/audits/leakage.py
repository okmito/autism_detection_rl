"""Leakage audit — §16.2 blocking."""
from __future__ import annotations
import numpy as np
from typing import Any, List, Dict

class LeakageTracker:
    """Instrument fitting operations with fold provenance."""
    def __init__(self):
        self.fits: List[Dict[str, Any]] = []

    def record_fit(self, name: str, train_indices: List[int] | np.ndarray, all_indices: List[int] | np.ndarray | None = None):
        entry = {"name": name, "train_indices": set(map(int, train_indices))}
        if all_indices is not None:
            entry["all_indices"] = set(map(int, all_indices))
        self.fits.append(entry)

    def check_no_leakage(self, heldout_indices: List[int] | np.ndarray) -> List[str]:
        """Return list of violations where fitted object saw held-out rows."""
        held = set(map(int, heldout_indices))
        violations = []
        for f in self.fits:
            overlap = f["train_indices"] & held
            if overlap:
                violations.append(f"Transformer '{f['name']}' was fitted on held-out indices {sorted(overlap)[:5]}")
        return violations

    def assert_no_leakage(self, heldout_indices):
        v = self.check_no_leakage(heldout_indices)
        if v:
            raise AssertionError("LEAKAGE DETECTED: " + "; ".join(v))


def poisoned_control_demo(n: int = 100) -> bool:
    """Deliberately fit a scaler on full data and verify audit catches it.
    Returns True if audit correctly detects leakage.
    """
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 5))
    train_idx = list(range(n // 2))
    held_idx = list(range(n // 2, n))
    # Poisoned: fit on all data
    scaler = StandardScaler()
    scaler.fit(X)  # fitted on held-out rows
    tracker = LeakageTracker()
    tracker.record_fit("scaler_poisoned", train_indices=list(range(n)), all_indices=list(range(n)))
    violations = tracker.check_no_leakage(held_idx)
    return len(violations) > 0
