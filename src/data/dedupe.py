"""Cross-source duplicate detection — §15."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
import hashlib

def _record_hash(rec: Dict[str, Any], include_label: bool = False) -> str:
    """Hash of the answer pattern (+ covariates). Label is excluded by
    default so a cross-source duplicate with a *disagreeing* label is
    still flagged; the label is reported alongside, not hidden in the hash."""
    arr = rec["item_responses"]
    s = ",".join("NA" if np.isnan(x) else str(int(x)) for x in arr)
    s += f"|{rec['covariates'].get('sex','')}|{rec['covariates'].get('age_band','')}"
    if include_label:
        s += f"|{rec['label']}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def find_duplicates(datasets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Find records that appear in more than one dataset source.
    Returns list of dicts {hash, sources, indices}. The hash excludes the
    label, so a cross-source duplicate whose labels disagree is flagged and
    the disagreement is reported in ``label_disagreement``.
    """
    index: Dict[str, List[tuple]] = {}
    for src, records in datasets.items():
        for idx, rec in enumerate(records):
            h = _record_hash(rec)
            index.setdefault(h, []).append((src, idx, int(rec["label"]),
                                            str(rec.get("label_source", ""))))
    dups = []
    for h, locs in index.items():
        sources = set(s for s, _, _, _ in locs)
        if len(sources) > 1:
            labels = {lbl for _, _, lbl, _ in locs}
            dups.append({
                "hash": h,
                "locations": [(s, i) for s, i, _, _ in locs],
                "label_disagreement": len(labels) > 1,
                "labels": sorted(labels),
            })
    return dups

def dedupe_report(datasets: Dict[str, List[Dict[str, Any]]]) -> str:
    dups = find_duplicates(datasets)
    if not dups:
        return "No cross-source duplicates found."
    lines = [f"Found {len(dups)} duplicate hash(es):"]
    for d in dups:
        lines.append(f"  {d['hash']}: {d['locations']}")
    return "\n".join(lines)
