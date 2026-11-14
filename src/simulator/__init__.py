"""Digital patient simulator — Stage 6. Research environment ONLY.

Synthetic records are reproducible, labelled ``simulated``/``synthetic``,
kept separate from real data, and are NEVER clinical ground truth. The
final predictor is never trained on simulator output.
"""
from src.simulator.patient import SimulatedPatient, generate_cohort

__all__ = ["SimulatedPatient", "generate_cohort"]
