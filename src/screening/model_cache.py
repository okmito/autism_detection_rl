"""Build or load the demo/app predictor (audit findings T1 and T12).

The web app and ``scripts/demo_app.py`` each had their own copy of this. They share it now, so both get the same
three properties:

1. **A real cache key.** The file stem includes a fingerprint of the dataset *content* and of the train/val/test
   partition, so changing the CSV (or the split code) can no longer silently reuse weights trained on other data.
2. **Verified before loaded.** A manifest records the SHA-256 of the weights and calibrator; both are checked
   before ``torch.load`` / ``pickle.load`` see a byte (see ``src.models.artifact_integrity`` for what that does
   and does not protect against). An absent or mismatching manifest means "retrain", never "load anyway".
3. **Logged, never silent.** Every hit, miss, and fallback goes through the caller's ``info``/``warn`` callbacks
   (the app logs; the demo prints).

Legacy files named ``demo_model_saudi_seed0_platt_v2.{pt,pkl}`` are frozen research artifacts that other scripts
read. This module never reads, rewrites, or deletes them: its files carry the fingerprint in the name.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from src.data.ingest import load_dataset
from src.data.splits import stratified_split
from src.models.artifact_integrity import ArtifactIntegrityError, load_state_dict, sha256_file, verify_sha256
from src.models.masked_predictor import MaskedPredictor

MANIFEST_SCHEMA = 1


def dataset_fingerprint(records: Sequence[Dict[str, Any]], train, val, test) -> str:
    """16-hex digest of the records' content and of which partition each record landed in.

    ``split_fingerprint`` hashes only the ``provenance`` field, so for records without one a different split
    gives the same digest. This hashes each record's answers, missing mask and label, per partition, so both a
    changed CSV and a changed split change the key.
    """
    h = hashlib.sha256()
    h.update(str(len(records)).encode())
    for name, part in (("all", records), ("train", train), ("val", val), ("test", test)):
        h.update(b"|" + name.encode() + b"|")
        for rec in part:
            h.update(rec["item_responses"].astype("float64").tobytes())
            h.update(rec["missing_mask"].astype("uint8").tobytes())
            h.update(str(int(rec["label"])).encode())
            h.update(str(rec.get("provenance", "")).encode())
            h.update(b"\x00")
    return h.hexdigest()[:16]


def cache_stem(seed: int, calibration: str, fingerprint: str) -> str:
    return f"demo_model_saudi_seed{seed}_{calibration}_v{MaskedPredictor.VERSION}_d{fingerprint[:12]}"


def _atomic_write(path: Path, write: Callable[[Any], None], mode: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode) as fh:
        write(fh)
    os.replace(tmp, path)


def _expected_manifest(seed, calibration, fingerprint, n_items, hidden) -> Dict[str, Any]:
    return {"schema": MANIFEST_SCHEMA, "predictor_version": MaskedPredictor.VERSION, "seed": seed,
            "calibration": calibration, "n_items": n_items, "hidden": list(hidden), "dataset_fingerprint": fingerprint}


def _try_load(pred: MaskedPredictor, pt: Path, pkl: Path, manifest_path: Path, expected: Dict[str, Any]) -> None:
    """Load weights + calibrator into ``pred`` or raise; nothing is deserialised before its hash is verified."""
    if not manifest_path.is_file():
        raise ArtifactIntegrityError("no manifest (cache written by an older version, or not written yet)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, want in expected.items():
        if manifest.get(key) != want:
            raise ArtifactIntegrityError(f"manifest field {key!r} differs from this run ({manifest.get(key)!r} != {want!r})")
    verify_sha256(pt, manifest.get("weights_sha256"), "predictor weights")
    verify_sha256(pkl, manifest.get("calibrator_sha256"), "calibrator")
    pred.model.load_state_dict(load_state_dict(pt))
    with open(pkl, "rb") as fh:
        pred.calibrator = pickle.load(fh)


def build_predictor(results_dir: Path, *, seed: int, n_items: int, calibration: str,
                    hidden: Sequence[int] = (128, 64), retrain: bool = False,
                    info: Callable[[str], None] = lambda m: None,
                    warn: Callable[[str], None] = lambda m: None) -> Tuple[List, List, List, List, MaskedPredictor]:
    """Return ``(records, train, val, test, predictor)``, from the verified cache when there is one."""
    records = load_dataset("saudi")
    train, val, test = stratified_split(records, seed=seed)
    fingerprint = dataset_fingerprint(records, train, val, test)
    stem = cache_stem(seed, calibration, fingerprint)
    results_dir = Path(results_dir)
    pt, pkl, manifest_path = (results_dir / f"{stem}.pt", results_dir / f"{stem}.pkl", results_dir / f"{stem}.manifest.json")
    expected = _expected_manifest(seed, calibration, fingerprint, n_items, hidden)

    pred = MaskedPredictor(n_items=n_items, hidden=list(hidden), calibration=calibration, seed=seed)
    if not retrain and pt.exists() and pkl.exists():
        try:
            _try_load(pred, pt, pkl, manifest_path, expected)
            info(f"loaded verified cached predictor {stem}")
            return records, train, val, test, pred
        except Exception as exc:  # corrupt, unverifiable, or from other data: retrain rather than trust it
            warn(f"cached predictor {stem} not used ({type(exc).__name__}: {exc}); retraining")
            pred = MaskedPredictor(n_items=n_items, hidden=list(hidden), calibration=calibration, seed=seed)
    else:
        info(f"no cached predictor {stem}; training (about 20 s)")
    pred.fit(train, epochs=20, lr=1e-3, batch_size=32, seed=seed)
    pred.fit_calibrator(val, method=calibration)
    try:
        import torch
        results_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(pt, lambda fh: torch.save(pred.model.state_dict(), fh), "wb")
        _atomic_write(pkl, lambda fh: pickle.dump(pred.calibrator, fh), "wb")
        manifest = {**expected, "weights_sha256": sha256_file(pt), "calibrator_sha256": sha256_file(pkl)}
        _atomic_write(manifest_path, lambda fh: json.dump(manifest, fh, indent=2, sort_keys=True), "w")
        info(f"cached predictor written to {stem}")
    except Exception as exc:
        warn(f"could not write predictor cache ({type(exc).__name__}); it will be retrained on next start")
    return records, train, val, test, pred
