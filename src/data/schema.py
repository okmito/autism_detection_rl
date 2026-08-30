"""Dataset schema validation — §15 common schema."""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

VALID_LABEL_SOURCES = {"questionnaire", "clinical"}
VALID_SEX = {"M", "F", "unknown", "other"}
# age_band is free-form string but must be present


def validate_record(rec: Dict[str, Any], n_items: int | None = None) -> None:
    """Validate a single record dict against §15 common schema. Raises ValueError on violation."""
    required_keys = {"item_responses", "label", "label_source", "covariates", "missing_mask"}
    missing = required_keys - set(rec.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    ir = rec["item_responses"]
    mm = rec["missing_mask"]
    if not isinstance(ir, np.ndarray):
        raise ValueError("item_responses must be np.ndarray")
    if not isinstance(mm, np.ndarray):
        raise ValueError("missing_mask must be np.ndarray")
    if ir.shape != mm.shape:
        raise ValueError(f"Shape mismatch: {ir.shape} vs {mm.shape}")
    if ir.ndim != 1:
        raise ValueError("item_responses must be 1-D")
    if n_items is not None and ir.shape[0] != n_items:
        raise ValueError(f"Expected {n_items} items, got {ir.shape[0]}")
    if mm.dtype != bool:
        # allow castable but warn — enforce bool
        if not np.issubdtype(mm.dtype, np.bool_):
            raise ValueError("missing_mask must be bool dtype")

    label = rec["label"]
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}")

    if rec["label_source"] not in VALID_LABEL_SOURCES:
        raise ValueError(f"Invalid label_source: {rec['label_source']}")

    cov = rec["covariates"]
    if not isinstance(cov, dict) or "age_band" not in cov or "sex" not in cov:
        raise ValueError("covariates must contain age_band and sex")

    # missing entries must be NaN in item_responses
    for i in range(len(ir)):
        if mm[i]:
            if not np.isnan(ir[i]):
                raise ValueError(f"missing_mask True but item_responses[{i}]={ir[i]} is not NaN")
        else:
            if np.isnan(ir[i]):
                raise ValueError(f"missing_mask False but item_responses[{i}] is NaN")


def validate_dataset(records: list[Dict[str, Any]], n_items: int | None = None) -> None:
    if not records:
        raise ValueError("Empty dataset")
    for rec in records:
        validate_record(rec, n_items=n_items)


def to_arrays(records: list[Dict[str, Any]]):
    """Convert list of records to (X, y, missing_mask) arrays."""
    X = np.stack([r["item_responses"] for r in records])
    y = np.array([r["label"] for r in records], dtype=int)
    mm = np.stack([r["missing_mask"] for r in records])
    return X, y, mm
