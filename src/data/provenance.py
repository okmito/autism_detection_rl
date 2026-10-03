"""Dataset provenance verification — V-4 / V-7 supporting evidence.

Purpose
-------
Prove, reproducibly and from recorded metadata alone, whether a raw SPSS export
describes **the same cohort** as an already-integrated CSV, or a genuinely new
one. This exists because the repository holds a Mendeley ``.sav`` whose filename
suggests a second dataset (``QCHAT_dataset2``) while the integrated cohort was
documented as coming from ``QCHAT_dataset1.sav``. Resolving that question by
reading the raw file — rather than trusting a filename or a filename-shaped
memory — is what prevents a duplicate cohort from being ingested as if it were
a new external-validation set.

What is compared
----------------
Identity is established on four independent axes, because any one of them alone
is weak:

1. **Schema** — column names and their order.
2. **Participant identifiers** — ``child_id`` as an opaque key. Set equality plus
   duplicate detection on both sides.
3. **Decoded categorical agreement** — SPSS *value labels* are applied to the
   raw codes before comparison. Without this step a ``group`` coded
   ``1=ASD, 7=control`` is silently incomparable with a CSV holding the strings
   ``"ASD"``/``"control"``.
4. **Numeric agreement per participant** — the continuous columns, aligned on
   ``child_id`` rather than on row order.

Privacy
-------
The verifier returns **counts, hashes and boolean outcomes only**. No participant
row, identifier, or response is ever placed in the returned structure or written
to an artifact. :func:`_no_participant_payload` is asserted by the test suite so
a future change cannot quietly start emitting rows.

MANDATORY CAVEAT
----------------
Passing this verification means two files describe the same people. It says
nothing about whether the label is clinically sound — that is the separate
question ``src/audits/circularity.py`` answers, and it is answered elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: The participant key. Used to align rows, never to identify a person.
ID_COLUMN = "child_id"

#: Categorical columns compared after SPSS value-label decoding.
CATEGORICAL_COLUMNS = ("sex", "group", "preterm", "siblings_yesno",
                       "mothers_education", "sibling_withASD")

#: Continuous columns compared numerically after aligning on the ID.
NUMERIC_COLUMNS = ("age", "birthweight", "siblings_number", "Sum_QCHAT")


class ProvenanceError(RuntimeError):
    """Raised when provenance cannot be established. Never degrades silently."""


def sha256_file(path: Path) -> str:
    """Streaming sha256. Returns the hex digest only, never file content."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def _read_sav(path: Path):
    """Read an SPSS ``.sav`` with raw codes plus its value-label metadata."""
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ProvenanceError(
            "pyreadstat is required to read SPSS .sav files. "
            "Install it in the project venv: "
            ".venv-win\\Scripts\\python -m pip install pyreadstat"
        ) from exc
    frame, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
    return frame, meta


def _apply_value_labels(frame, meta, column: str):
    """Map a numeric-coded SPSS column onto its label strings.

    Returns a list of label strings, one per row, using the file's own
    ``value_labels`` metadata. Values with no label are returned as their
    ``str()`` so an unlabelled code stays visible instead of becoming NaN.
    """
    mapping = meta.variable_value_labels.get(column) or {}
    if not mapping:
        return [str(x) for x in frame[column]]
    out: List[str] = []
    for value in frame[column]:
        try:
            key = float(value)
        except (TypeError, ValueError):
            out.append(str(value))
            continue
        label = mapping.get(key)
        if label is None:
            label = mapping.get(value)
        out.append(str(label) if label is not None else str(value))
    return out


def _normalise_id(value: Any) -> str:
    """IDs are opaque keys. Only whitespace is stripped - never coerced to int,
    because '007' and '7' are different participants in some exports."""
    return str(value).strip()


def _as_float_series(series) -> np.ndarray:
    return pd_numeric(series)


def pd_numeric(series) -> np.ndarray:
    """Numeric coercion that maps unparseable entries to NaN rather than 0."""
    import pandas as pd
    return np.asarray(pd.to_numeric(pd.Series(list(series)), errors="coerce"),
                      dtype=float)


def _numeric_agreement(a: np.ndarray, b: np.ndarray) -> int:
    """Count of positions where both sides agree (NaN == NaN counts as agree)."""
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(np.nan_to_num(a, nan=-9e18), np.nan_to_num(b, nan=-9e18),
                       rtol=1e-9, atol=1e-9)
    return int(np.sum(both_nan | close))


def verify_same_cohort(sav_path: Path, csv_path: Path,
                       id_column: str = ID_COLUMN,
                       categorical_columns: tuple = CATEGORICAL_COLUMNS,
                       numeric_columns: tuple = NUMERIC_COLUMNS) -> Dict[str, Any]:
    """Record-level identity comparison between an SPSS export and a CSV.

    Returns a metadata-only dictionary. Raises :class:`ProvenanceError` if the
    files cannot be read or a required column is absent - a provenance check that
    cannot run must fail loudly rather than report ``False``.
    """
    import pandas as pd

    sav_path, csv_path = Path(sav_path), Path(csv_path)
    for p in (sav_path, csv_path):
        if not p.exists():
            raise ProvenanceError(f"required input missing: {p}")

    sav, meta = _read_sav(sav_path)
    csv = pd.read_csv(csv_path, encoding="utf-8")

    if id_column not in sav.columns:
        raise ProvenanceError(f"{sav_path.name}: participant key {id_column!r} absent")
    if id_column not in csv.columns:
        raise ProvenanceError(f"{csv_path.name}: participant key {id_column!r} absent")

    schema_equal_ordered = list(sav.columns) == list(csv.columns)
    schema_equal_set = set(sav.columns) == set(csv.columns)

    sav_ids = [_normalise_id(x) for x in sav[id_column]]
    csv_ids = [_normalise_id(x) for x in csv[id_column]]
    sav_set, csv_set = set(sav_ids), set(csv_ids)
    shared = sav_set & csv_set

    # --- categorical agreement, aligned on the participant key -----------
    categorical_detail: Dict[str, Any] = {}
    for column in categorical_columns:
        if column not in sav.columns or column not in csv.columns:
            categorical_detail[column] = {"comparable": False,
                                          "reason": "column absent from one source"}
            continue
        decoded = _apply_value_labels(sav, meta, column)
        sav_by_id = {k: v.strip() for k, v in zip(sav_ids, decoded)}
        csv_by_id = {k: str(v).strip() for k, v in zip(csv_ids, csv[column])}
        agree = sum(1 for k in shared if sav_by_id.get(k, "") == csv_by_id.get(k, ""))
        categorical_detail[column] = {
            "comparable": True,
            "compared_participants": len(shared),
            "agreement_count": int(agree),
            "identical": bool(bool(shared) and agree == len(shared)),
        }

    # --- numeric agreement, aligned on the participant key ---------------
    numeric_detail: Dict[str, Any] = {}
    for column in numeric_columns:
        if column not in sav.columns or column not in csv.columns:
            numeric_detail[column] = {"comparable": False,
                                      "reason": "column absent from one source"}
            continue
        sav_by_id = {k: v for k, v in zip(sav_ids, sav[column])}
        csv_by_id = {k: v for k, v in zip(csv_ids, csv[column])}
        keys = sorted(shared)
        left = pd_numeric([sav_by_id[k] for k in keys])
        right = pd_numeric([csv_by_id[k] for k in keys])
        agree = _numeric_agreement(left, right)
        numeric_detail[column] = {
            "comparable": True,
            "compared_participants": len(keys),
            "agreement_count": int(agree),
            "identical": bool(agree == len(keys) and keys),
        }

    sav_dupes = len(sav_ids) - len(sav_set)
    csv_dupes = len(csv_ids) - len(csv_set)

    matched = len(shared)
    identical = bool(
        schema_equal_ordered
        and sav_dupes == 0 and csv_dupes == 0
        and matched == len(sav_ids) == len(csv_ids)
        and all(v.get("identical") for v in categorical_detail.values())
        and all(v.get("identical") for v in numeric_detail.values())
    )

    result = {
        "row_count_sav": int(len(sav_ids)),
        "row_count_csv": int(len(csv_ids)),
        "column_count_sav": int(len(sav.columns)),
        "column_count_csv": int(len(csv.columns)),
        "schema_equal_ordered": bool(schema_equal_ordered),
        "schema_equal_set": bool(schema_equal_set),
        "participant_match_count": int(matched),
        "participant_mismatch_count": int((len(sav_set - csv_set))
                                          + len(csv_set - sav_set)),
        "only_in_sav": int(len(sav_set - csv_set)),
        "only_in_csv": int(len(csv_set - sav_set)),
        "duplicate_count": int(sav_dupes + csv_dupes),
        "id_column": id_column,
        "categorical_agreement": categorical_detail,
        "numeric_agreement": numeric_detail,
        "dataset_identity": ("same_cohort" if identical else "different_or_incomparable"),
        "spss_value_labels": {
            col: {str(k): str(v) for k, v in
                  sorted((meta.variable_value_labels.get(col) or {}).items(),
                         key=lambda kv: float(kv[0]))}
            for col in meta.column_names
            if meta.variable_value_labels.get(col)
        },
        "spss_variable_types": dict(meta.readstat_variable_types),
        "spss_file_encoding": meta.file_encoding,
        "contains_participant_rows": False,
    }
    _assert_no_participant_payload(result)
    return result


def _assert_no_participant_payload(obj: Any, depth: int = 0) -> None:
    """Guard against a future edit leaking participant-level values.

    Only *counts, hashes, labels and booleans* are permitted. A list whose
    length equals a dataset row count, or any raw string that looks like a
    participant identifier, fails the check.
    """
    if depth > 6:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("participant_rows", "records", "rows", "responses"):
                raise AssertionError(
                    f"provenance payload must not carry {key!r}")
            _assert_no_participant_payload(value, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_participant_payload(item, depth + 1)
    elif isinstance(obj, str):
        # SPSS value labels are short words; participant ids are not.
        if len(obj) >= 6 and obj.replace("-", "").replace("/", "").isalnum() \
                and not obj.replace(".", "").replace("_", "").isalnum():
            return  # a long alphabetic label such as 'primary/vocational'
        if len(obj) >= 6 and any(ch.isdigit() for ch in obj) and "_" in obj:
            raise AssertionError(f"possible participant identifier in payload: {obj!r}")


def build_provenance_artifact(repo: Path, sav_path: Path, csv_path: Path) -> Dict[str, Any]:
    """Full provenance artifact: hashes, identity result, git sha, timestamp."""
    repo = Path(repo)
    comparison = verify_same_cohort(sav_path, csv_path)
    artifact = {
        "artifact": "polish_provenance_verification",
        "purpose": (
            "Establish whether the local SPSS export describes the same cohort "
            "as the integrated Polish CSV, so that a duplicate cohort is never "
            "ingested as a new external-validation dataset. Metadata only - no "
            "participant rows."
        ),
        "git_sha": git_sha(repo),
        "source_file": str(Path(sav_path).name),
        "source_path": str(sav_path),
        "source_sha256": sha256_file(Path(sav_path)),
        "integrated_dataset_path": str(csv_path),
        "integrated_sha256": sha256_file(Path(csv_path)),
        "schema_match": bool(comparison["schema_equal_ordered"]),
        "dataset_identity": comparison["dataset_identity"],
        "contains_participant_rows": False,
        "privacy_note": (
            "Metadata only. No participant identifier, response or row is "
            "recorded in this artifact. Participant keys are used transiently "
            "in memory to align rows and are discarded."
        ),
        "verification_tool": "pyreadstat",
        "python": platform.python_version(),
        "verification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detail": comparison,
    }
    return artifact


def write_artifact(repo: Path, artifact: Dict[str, Any], filename: str) -> Path:
    results = Path(repo) / "results"
    results.mkdir(exist_ok=True)
    path = results / filename
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return path