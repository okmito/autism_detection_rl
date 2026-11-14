"""Lightweight synthetic patient simulator — Stage 6 (controlled research).

Generates reproducible synthetic questionnaire engagements via a 2PL IRT
response model over a latent trait. Deliberately NOT a clinical model:
labels are simulator-internal (a deterministic trait threshold) and are
never clinical outcomes. Includes incomplete responses, contradictory
answers (noise flips), configurable response cost, and distribution shift
(mean/slope offsets) for robustness evaluation.

Everything is seeded; returned records carry ``label_source="simulated"``
and ``provenance`` starting with ``simulated`` so schema validation keeps
them strictly separate from questionnaire/clinical records.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

SIMULATOR_VERSION = "simulator-v1-2pl"


class SimulatedPatient:
    """One synthetic engagement: latent trait -> per-item 2PL responses."""

    def __init__(self, a: np.ndarray, b: np.ndarray, *, seed: int = 0,
                 noise_flip: float = 0.0, missing_rate: float = 0.0,
                 threshold: float = 0.0, cost: float = 1.0):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        if self.a.shape != self.b.shape or self.a.ndim != 1:
            raise ValueError("a and b must be equal 1-D arrays")
        if not 0.0 <= noise_flip <= 1.0:
            raise ValueError("noise_flip must be in [0,1]")
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be in [0,1]")
        self.n_items = int(self.a.size)
        self.seed = int(seed)
        self.noise_flip = float(noise_flip)
        self.missing_rate = float(missing_rate)
        self.threshold = float(threshold)
        self.cost = float(cost)

    def generate(self, theta: float = 0.0, *, seed: int | None = None) -> Dict[str, Any]:
        """Responses for a given latent trait. Reproducible from ``seed``."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        theta = float(theta)
        p = 1.0 / (1.0 + np.exp(-self.a * (theta - self.b)))
        raw = (rng.random(self.n_items) < p).astype(float)
        # contradictory answers: flip a fraction
        if self.noise_flip > 0:
            flips = rng.random(self.n_items) < self.noise_flip
            raw[flips] = 1.0 - raw[flips]
        missing = rng.random(self.n_items) < self.missing_rate
        raw_out = raw.copy()
        raw_out[missing] = np.nan
        label = int(theta >= self.threshold)
        return {
            "item_responses": raw_out,
            "label": label,
            "label_source": "simulated",
            "covariates": {"age_band": "simulated", "sex": "unknown"},
            "missing_mask": missing,
            "provenance": f"simulated:{SIMULATOR_VERSION}:seed={self.seed}:theta={theta}",
            "simulator": True,
            "simulator_version": SIMULATOR_VERSION,
            "theta": theta,
            "response_cost": self.cost,
        }

    def generate_cohort(self, n: int, thetas: np.ndarray | None = None,
                        *, seed: int | None = None) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if thetas is None:
            thetas = rng.normal(0.0, 1.0, size=int(n))
        thetas = np.asarray(thetas, dtype=float)
        if len(thetas) != n:
            raise ValueError("thetas length must equal n")
        return [self.generate(float(t), seed=int(rng.integers(0, 2**31 - 1))) for t in thetas]


def generate_cohort(n: int, n_items: int = 10, *, seed: int = 0,
                    mean_shift: float = 0.0, slope_scale: float = 1.0,
                    noise_flip: float = 0.0, missing_rate: float = 0.0) -> List[Dict[str, Any]]:
    """Convenience cohort with default 2PL items; distribution shift via
    ``mean_shift`` (theta mean) and ``slope_scale`` (discrimination scale)."""
    rng = np.random.default_rng(seed)
    base_a = rng.uniform(0.5, 2.0, size=n_items) * slope_scale
    base_b = rng.normal(0.0, 1.0, size=n_items)
    sim = SimulatedPatient(base_a, base_b, seed=seed,
                           noise_flip=noise_flip, missing_rate=missing_rate)
    thetas = rng.normal(mean_shift, 1.0, size=int(n))
    return sim.generate_cohort(int(n), thetas, seed=seed)
