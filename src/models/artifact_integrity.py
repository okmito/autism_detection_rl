"""SHA-256 verification before a model artifact is deserialised (audit finding T12).

``pickle.load`` executes code from the file it reads, and ``torch.load`` historically did too. Both are used
on files in ``results/``, which are trusted local outputs - but "trusted" should be checked, not assumed.
These helpers verify a recorded SHA-256 *before* the bytes are interpreted.

What this protects against, stated plainly: corruption, truncated writes, a weights file paired with the wrong
calibrator, and accidental replacement. It does not protect against someone who can write both the artifact
and its recorded hash into the same directory; the hash must come from somewhere that party cannot edit
(a committed manifest, a published release note) for that.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


class ArtifactIntegrityError(RuntimeError):
    """An artifact is missing, has no recorded hash, or does not match it. It must not be loaded."""


def sha256_file(path: PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: PathLike, expected: Any, label: str = "artifact") -> str:
    """Return the file's digest if it equals ``expected``; raise :class:`ArtifactIntegrityError` otherwise.

    A missing file, a missing or malformed expected hash, and a mismatch are all errors: an unverifiable
    artifact is never silently accepted.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactIntegrityError(f"{label} is missing: {path.name}")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected.lower()):
        raise ArtifactIntegrityError(f"{label} has no valid recorded SHA-256, refusing to load {path.name}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ArtifactIntegrityError(
            f"{label} {path.name} does not match its recorded SHA-256 "
            f"(recorded {expected[:12]}..., found {actual[:12]}...)")
    return actual


def load_state_dict(path: PathLike):
    """``torch.load`` restricted to tensors (``weights_only=True``): a state dict never needs arbitrary objects."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)
