"""Per-item cost model — §11.1"""
from __future__ import annotations
import numpy as np

def get_costs(n: int, mode: str = "uniform", weights: list[float] | None = None) -> np.ndarray:
    if mode == "uniform":
        return np.ones(n, dtype=float)
    elif mode == "weighted":
        if weights is None:
            raise ValueError("weighted mode requires weights")
        if len(weights) != n:
            raise ValueError(f"weights len {len(weights)} != n {n}")
        return np.array(weights, dtype=float)
    else:
        raise ValueError(f"Unknown cost mode {mode}")
