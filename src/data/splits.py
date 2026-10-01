"""Canonical train/val/test split — §17, §19.

Every runnable baseline and every artifact must be produced on the *same* split.
Spec §17: "All runnable baselines use identical train/validation/test splits,
matched question-count budgets, missingness rules, seeds, and metric definitions
unless a method intrinsically requires a different representation."

This module exists because that invariant was not enforced in code. Four modules
had grown their own copy of a 4-fold ``StratifiedKFold`` (step2, step3, step5,
``scripts/demo_app.py``) and one had diverged: ``step4_train_policies.py`` used a
60/20/20 ``train_test_split``, giving 303/101/102 instead of 284/95/127. Its
policies were therefore trained on a different 303 records from the 284 every
other artifact used, so the step4 and step5 numbers were never comparable.

The scheme below is the 4-fold one, because that is what step2, step3, step5 and
both demos already used — adopting it keeps every existing artifact valid and
changes only step4.

Shape: ``n_splits=4`` stratified, shuffled, seeded. The first fold is held out as
test; the second fold of the remainder is held out as validation. On the 506-row
Saudi cohort at ``seed=0`` this yields 284 train / 95 val / 127 test.

The three parts are **disjoint by construction**. The pre-existing copies of this
split were not — see the indexing note in :func:`stratified_split`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

__all__ = ["stratified_split", "split_fingerprint"]

#: Recorded in every artifact so a split change is visible in the output rather
#: than inferred from row counts.
SCHEME = "stratified-4fold-2nd-fold-val"


def stratified_split(
    records: Sequence[Dict[str, Any]], seed: int = 0, n_splits: int = 4
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(train, val, test)`` record lists on the canonical split.

    Deterministic for a given ``seed``: the same call always yields the same
    partition, which is what makes artifacts comparable across runs.
    """
    if len(records) < n_splits:
        raise ValueError(f"need at least {n_splits} records, got {len(records)}")

    y = np.array([r["label"] for r in records])
    idx = np.arange(len(records))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(idx, y))

    # The second split must be indexed carefully. ``skf.split(train_idx, ...)``
    # returns indices *relative to train_idx*, not to the original record list.
    # The pre-existing copies of this split in step2/step3/step5/demo_app did
    # ``[records[i] for i in val_idx]`` with those relative indices, so the
    # "validation" set was drawn from the wrong records: on the 506-row Saudi
    # cohort 71 of its 95 records were also in the training set and 24 were in
    # the test set. The row *counts* came out right (284/95/127), which is why
    # it went unnoticed, but the calibrator was being fitted on data the
    # network had already memorised and partly on test records.
    inner_train_rel, val_rel = next(skf.split(train_idx, y[train_idx]))
    val_idx = train_idx[val_rel]
    train_idx = train_idx[inner_train_rel]

    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


def split_fingerprint(
    train: Sequence[Dict[str, Any]],
    val: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
) -> str:
    """Stable hash of the *identity* of a partition, not just its sizes.

    Two splits can have identical row counts and still differ in membership — the
    failure mode that let step4's 303-record split pass unnoticed. This hashes
    the ``provenance`` field of every record, so a mismatch is detectable even
    when the counts agree.
    """
    import hashlib

    h = hashlib.sha256()
    for name, part in (("train", train), ("val", val), ("test", test)):
        h.update(name.encode())
        for rec in part:
            h.update(str(rec.get("provenance", "")).encode())
            h.update(b"\x00")
    return h.hexdigest()[:16]
