"""Dataset ingest — one loader per tier in §15.
All loaders return list[dict] conforming to common schema (§15 + schema.py).

For datasets that are not yet downloaded, loaders create synthetic placeholders
when synthetic=True (for testing). Real data paths are documented as
HUMAN ACTION REQUIRED — see docstrings.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Binary mapping for Q-CHAT-10 primary representation (§15)
# For Q1-Q9: Sometimes/Rarely/Never -> 1, Always/Usually -> 0
# For Q10: Always/Usually/Sometimes -> 1, Rarely/Never -> 0
# Raw ordinal scale assumed 0..4 => 0=Always,1=Usually,2=Sometimes,3=Rarely,4=Never
# This mapping must be verified against source encoding — locked per spec.

def qchat10_binary_map(raw: np.ndarray) -> np.ndarray:
    """Map raw Q-CHAT-10 ordinal responses (0-4) to binary representation.
    raw shape (n,10) with NaN for missing.
    Returns binary array same shape with NaN preserved.
    """
    out = np.full_like(raw, np.nan, dtype=float)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            if np.isnan(v):
                continue
            v = int(v)
            if j < 9:
                # Q1-9: Sometimes(2)/Rarely(3)/Never(4) -> 1
                out[i, j] = 1 if v >= 2 else 0
            else:
                # Q10
                out[i, j] = 1 if v <= 2 else 0
    return out


def _synthetic_records(n: int, n_items: int, label_source: str = "questionnaire", seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        # binary item responses 0/1 with 5% missing
        raw = rng.integers(0, 2, size=n_items).astype(float)
        missing = rng.random(n_items) < 0.05
        raw[missing] = np.nan
        label = int(rng.integers(0, 2))
        records.append({
            "item_responses": raw,
            "label": label,
            "label_source": label_source,
            "covariates": {"age_band": rng.choice(["1-2", "2-3", "3-5"]), "sex": rng.choice(["M", "F"])},
            "missing_mask": missing,
        })
    return records


def load_nz(synthetic: bool = False, data_dir: str | Path | None = None, **kwargs) -> List[Dict[str, Any]]:
    """NZ Q-CHAT-10 — 1054 rows. Requires V-1 data access."""
    if synthetic:
        return _synthetic_records(1054, 10, "questionnaire", seed=1)
    raise FileNotFoundError("HUMAN ACTION REQUIRED: NZ dataset not yet downloaded. Place file at data/raw/nz_qchat10.csv and re-run.")

def load_saudi(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(506, 10, "questionnaire", seed=2)
    raise FileNotFoundError("HUMAN ACTION REQUIRED: Saudi dataset not yet downloaded.")

def load_uci_child(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(292, 10, "questionnaire", seed=3)
    # Try UCI if present
    p = Path("data/raw/uci_child.csv")
    if p.exists():
        return _load_uci_csv(p, label_source="questionnaire")
    raise FileNotFoundError("HUMAN ACTION REQUIRED: UCI child not found.")

def load_uci_adol(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(104, 10, "questionnaire", seed=4)
    raise FileNotFoundError("HUMAN ACTION REQUIRED: UCI adol not found.")

def load_uci_adult(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(704, 10, "questionnaire", seed=5)
    raise FileNotFoundError("HUMAN ACTION REQUIRED: UCI adult not found.")

def load_polish(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Polish Q-CHAT-25 — 252 (or 253) rows, clinical labels."""
    if synthetic:
        return _synthetic_records(252, 25, "clinical", seed=6)
    raise FileNotFoundError("HUMAN ACTION REQUIRED: Polish dataset not yet downloaded. See V-1/V-7.")

def _load_uci_csv(path: Path, label_source: str = "questionnaire") -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    # Heuristic: last column is label, first 10 are items
    # Override if needed — caller must verify mapping
    records = []
    cols = [c for c in df.columns if c.lower().startswith("a") or c.lower().startswith("q")]
    if len(cols) < 10:
        cols = list(df.columns[:10])
    for _, row in df.iterrows():
        vals = row[cols].values.astype(float)
        label = int(row.iloc[-1])
        missing = np.isnan(vals)
        records.append({
            "item_responses": vals,
            "label": label,
            "label_source": label_source,
            "covariates": {"age_band": "unknown", "sex": "unknown"},
            "missing_mask": missing,
        })
    return records

# Registry for Hydra
LOADERS = {
    "nz": load_nz,
    "saudi": load_saudi,
    "uci_child": load_uci_child,
    "uci_adol": load_uci_adol,
    "uci_adult": load_uci_adult,
    "polish": load_polish,
}

def load_dataset(name: str, synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name}. Available: {list(LOADERS)}")
    return LOADERS[name](synthetic=synthetic, **kwargs)
