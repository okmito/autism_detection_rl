"""Central data-validation pipeline — Stage 2 foundation.

Reuses the existing, already-tested components instead of inventing new
checks:

- ``src/data/schema.validate_dataset`` — per-record schema/label provenance
- ``src/data/splits.stratified_split`` + ``split_fingerprint`` — train/val/test
  separation and leak detection (canonical scheme)
- ``src/audits/circularity.audit_circularity`` — deterministic sum-threshold
  label detection (questionnaire-derived labels are not clinical diagnoses)
- ``src/data/dedupe.find_duplicates`` — cross-source duplicate detection

Nothing here downloads data or touches the sealed Polish cohort for fitting.
The pipeline is pure: it validates record lists it is given.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src.audits.circularity import audit_circularity
from src.data.dedupe import find_duplicates
from src.data.schema import validate_dataset, VALID_SYNTHETIC_SOURCES, VALID_LABEL_SOURCES
from src.data.splits import split_fingerprint, stratified_split

__all__ = [
    "validate_records",
    "check_split_separation",
    "validate_datasets",
]


def validate_records(records: Sequence[Dict[str, Any]], n_items: int | None = None) -> Dict[str, Any]:
    """Schema + label-provenance validation for one record list.

    Raises ValueError on the first invalid record (same behaviour as the
    underlying ``validate_dataset``); empty input is an error.
    Synthetic/simulated records must be labelled as such (see schema.py).
    """
    validate_dataset(list(records), n_items=n_items)
    sources = {r["label_source"] for r in records}
    is_synthetic_only = sources <= VALID_SYNTHETIC_SOURCES
    # A list returned by an ingest synthetic fixture must validate; a mixed
    # real+synthetic merge is refused here — callers must not silently merge.
    if sources & VALID_LABEL_SOURCES and sources & VALID_SYNTHETIC_SOURCES:
        raise ValueError(
            f"Mixed real (questionnaire/clinical) and synthetic/simulated labels "
            f"{sorted(sources)} in one dataset; keep synthetic and real separate"
        )
    return {
        "n": len(records),
        "label_sources": sorted(sources),
        "synthetic_only": bool(is_synthetic_only),
        "prevalence": float(np.mean([r["label"] for r in records])),
        "missing_frac": float(
            np.mean([r["missing_mask"].mean() for r in records])
        ),
    }


def check_split_separation(
    records: Sequence[Dict[str, Any]], seed: int = 0, n_splits: int = 4
) -> Dict[str, Any]:
    """Canonical split + overlap check. Returns fingerprint and sizes.

    Raises on any overlap between train/val/test (the P0-9 failure class).
    """
    train, val, test = stratified_split(list(records), seed=seed, n_splits=n_splits)
    def _prov(part):
        return {str(r.get("provenance", "")) for r in part}
    overlap = (_prov(train) & _prov(val)) | (_prov(train) & _prov(test)) | (_prov(val) & _prov(test))
    if overlap:
        raise ValueError(f"Train/val/test overlap on {len(overlap)} provenance keys")
    return {
        "fingerprint": split_fingerprint(train, val, test),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }


def validate_datasets(
    datasets: Mapping[str, Sequence[Dict[str, Any]]],
    n_items: int | None = None,
    run_circularity: bool = True,
) -> Dict[str, Any]:
    """Full pipeline over a mapping of named record lists.

    Returns per-dataset validation summaries, circularity reports and
    cross-source duplicates. Never merges datasets; never returns rows.
    """
    summaries: Dict[str, Any] = {}
    for name, records in datasets.items():
        summaries[name] = validate_records(records, n_items=n_items)
        if run_circularity:
            summaries[name]["circularity"] = audit_circularity(list(records))
    duplicates = find_duplicates({k: list(v) for k, v in datasets.items()})
    return {
        "datasets": summaries,
        "cross_source_duplicates": [
            {**d, "locations": d["locations"]} for d in duplicates
        ],
        "n_cross_source_duplicates": len(duplicates),
    }
