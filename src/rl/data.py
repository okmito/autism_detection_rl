"""Q-CHAT-10 binary matrices for the RL experiments (no participant identifiers leave this module)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def load_polish_qchat10() -> Tuple[np.ndarray, np.ndarray]:
    """Polish cohort projected onto the ten Q-CHAT-10 items (verified contract), clinician-established labels.

    Returns ``X`` (252, 10) with 0/1 (NaN if a response is missing) and ``y`` (1 = ASD group, 0 = control).
    """
    import pandas as pd

    from src.data.ingest import POLISH_CSV, _abs
    from src.data.qchat10_contract import MODEL_FEATURE_ORDER, build_feature_matrix

    df = pd.read_csv(_abs(POLISH_CSV), encoding="utf-8")
    feats = build_feature_matrix({e["polish_var"]: df[e["polish_var"]].tolist() for e in MODEL_FEATURE_ORDER},
                                 allow_missing=True)
    X = np.array([[np.nan if v is None else float(v) for v in row] for row in feats], dtype=float)
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]], dtype=int)
    return X, y


def load_saudi_qchat10() -> Tuple[np.ndarray, np.ndarray]:
    """Saudi Q-CHAT-10. Its label is 1[sum of answers >= 4] (circular); used for development and transfer only."""
    from src.data.ingest import load_dataset

    recs = load_dataset("saudi", synthetic=False)
    X = np.stack([np.asarray(r["item_responses"], float) for r in recs])
    y = np.array([int(r["label"]) for r in recs])
    return X, y


QCHAT10_OFFICIAL_CUTOFF = 4  # Q-CHAT-10 guidance: refer when more than 3 of 10 items are scored
