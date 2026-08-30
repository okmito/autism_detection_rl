"""Cross-source duplicate detection — §15."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
import hashlib

def _record_hash(rec: Dict[str, Any]) -> str:
    # hash item_responses + label + covariates
    arr = rec["item_responses"]
    # replace NaN with sentinel
    s = ",".join("NA" if np.isnan(x) else str(int(x)) for x in arr)
    s += f"|{rec['label']}|{rec['covariates'].get('sex','')}|{rec['covariates'].get('age_band','')}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def find_duplicates(datasets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Find records that appear in more than one dataset source.
    Returns list of dicts {hash, sources, indices}.
    """
    index: Dict[str, List[tuple]] = {}
    for src, records in datasets.items():
        for idx, rec in enumerate(records):
            h = _record_hash(rec)
            index.setdefault(h, []).append((src, idx))
    dups = []
    for h, locs in index.items():
        sources = set(s for s, _ in locs)
        if len(sources) > 1:
            dups.append({"hash": h, "locations": locs})
    return dups

def dedupe_report(datasets: Dict[str, List[Dict[str, Any]]]) -> str:
    dups = find_duplicates(datasets)
    if not dups:
        return "No cross-source duplicates found."
    lines = [f"Found {len(dups)} duplicate hash(es):"]
    for d in dups:
        lines.append(f"  {d['hash']}: {d['locations']}")
    return "\n".join(lines)
