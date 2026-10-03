"""External validation of a frozen predictor on the sealed Polish cohort.

Design constraints, all enforced rather than documented-and-hoped-for
-----------------------------------------------------------------------
This module is the V-4/V-7 payoff, and the whole point of the sealed Polish
cohort is that it stays *untouched by fitting*. Accordingly:

* :func:`FrozenPredictor` exposes prediction only. There is no ``fit``, no
  ``fit_calibrator``, and no threshold setter on the object handed to the
  evaluation path. The calibrator is attached as a loaded artefact.
* :func:`assert_no_fitting` is called before every evaluation. It inspects the
  predictor for fit-like methods and re-checks that the evaluation records did
  not overlap the training cohort.
* If the frozen model's declared feature contract does not match the cohort,
  :func:`preflight` raises :class:`ExternalValidationBlocked`. It does **not**
  coerce, pad, truncate, or subset items to make the shapes agree.

Label handling
--------------
The clinical target is ``group`` with the value-label mapping declared by the
source file (``1 -> ASD``, ``7 -> control``). :func:`group_to_label` accepts
either that numeric coding or the label strings, and raises on anything else,
so an unrecognised code can never be silently folded into class 0.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.eval.gates import feature_compatibility


class ExternalValidationBlocked(RuntimeError):
    """Raised when external validation must not proceed. Never degrades."""


class LabelError(ValueError):
    """Raised for an unrecognised clinical group code."""


#: Value-label mapping declared by the SPSS source for the ``group`` variable.
GROUP_VALUE_LABELS: Dict[str, int] = {"ASD": 1, "control": 0}

#: Numeric codes observed in the SPSS source, mapped through its own labels.
GROUP_NUMERIC_CODES: Dict[float, int] = {1.0: 1, 7.0: 0}


def group_to_label(value: Any) -> int:
    """Map one ``group`` cell to a binary clinical label.

    Accepts the label strings used by the integrated CSV, or the numeric codes
    declared by the SPSS source. Anything else raises - a silent fallback to 0
    would corrupt the reference target.
    """
    if isinstance(value, str):
        key = value.strip()
        if key in GROUP_VALUE_LABELS:
            return GROUP_VALUE_LABELS[key]
        raise LabelError(
            f"unrecognised clinical group {value!r}; expected one of "
            f"{sorted(GROUP_VALUE_LABELS)} or a declared numeric code")
    try:
        code = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelError(f"non-numeric clinical group {value!r}") from exc
    if code in GROUP_NUMERIC_CODES:
        return GROUP_NUMERIC_CODES[code]
    raise LabelError(
        f"unrecognised clinical group code {code!r}; declared codes are "
        f"{sorted(GROUP_NUMERIC_CODES)}")


class FrozenPredictor:
    """Prediction-only wrapper around a trained :class:`MaskedPredictor`.

    Deliberately exposes no training surface. ``fit``/``fit_calibrator`` are not
    reachable on this object, so the external-validation code path cannot fit
    even by accident.
    """

    __slots__ = ("_inner", "manifest")

    def __init__(self, inner, manifest: Dict[str, Any]):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "manifest", manifest)

    def __call__(self, state: Dict[str, Any]) -> float:
        return float(self._inner(state))

    def predict_state(self, state: Dict[str, Any]) -> float:
        return float(self._inner.predict_state(state))

    @property
    def n_items(self) -> int:
        return int(self._inner.n_items)

    @property
    def m_list(self):
        return self._inner.m_list

    def __setattr__(self, name, value):  # pragma: no cover - defensive
        raise AttributeError(
            f"FrozenPredictor is immutable; refusing to set {name!r}. The "
            f"external-validation cohort must not influence a fitted artefact.")


FITTING_METHOD_NAMES = ("fit", "fit_calibrator", "train", "partial_fit",
                        "fit_threshold", "calibrate", "set_params")


def assert_no_fitting(predictor) -> None:
    """Fail loudly if anything on the evaluation path can fit parameters."""
    for name in FITTING_METHOD_NAMES:
        if hasattr(predictor, name):
            raise ExternalValidationBlocked(
                f"external-validation path exposes a fitting method {name!r}; "
                f"the Polish cohort must never influence fitting")
    if isinstance(predictor, FrozenPredictor):
        return
    raise ExternalValidationBlocked(
        "external validation requires a FrozenPredictor so that the evaluated "
        "artefact cannot be modified by this phase")


def preflight(frozen: "FrozenPredictor", cohort_n_items: int,
              cohort_m_list: Optional[Sequence[int]]) -> Dict[str, Any]:
    """Validate the feature contract before touching any cohort row."""
    compat = feature_compatibility(
        frozen_n_items=frozen.n_items,
        frozen_m_list=frozen.m_list,
        cohort_n_items=cohort_n_items,
        cohort_m_list=list(cohort_m_list) if cohort_m_list is not None else None,
    )
    if not compat["compatible"]:
        raise ExternalValidationBlocked(
            "frozen predictor cannot consume this cohort's item representation:\n  - "
            + "\n  - ".join(compat["blocking_reasons"])
            + "\n"
            + compat["coercion_policy"])
    return compat


def load_frozen_predictor(cache_pt: Path, cache_pk: Path,
                          manifest: Dict[str, Any]) -> FrozenPredictor:
    """Load a trained predictor and its calibrator as a frozen artefact."""
    import torch
    from src.models.masked_predictor import MaskedPredictor

    cache_pt, cache_pk = Path(cache_pt), Path(cache_pk)
    if not cache_pt.exists():
        raise ExternalValidationBlocked(
            f"frozen predictor weights missing: {cache_pt}. Run the Step 2 "
            f"pipeline before external validation.")
    if not cache_pk.exists():
        raise ExternalValidationBlocked(
            f"frozen calibrator missing: {cache_pk}. Calibration is a frozen "
            f"artefact and is never fitted on the external cohort; regenerate "
            f"it on the development split via Step 2.")

    hidden = tuple(manifest.get("hidden", (128, 64)))
    n_items = int(manifest["n_items"])
    m_list = manifest.get("m_list")
    calibration = manifest.get("calibration", "platt")
    pred = MaskedPredictor(n_items=n_items, hidden=list(hidden),
                           calibration=calibration,
                           seed=int(manifest.get("seed", 0)), m_list=m_list)
    pred.model.load_state_dict(torch.load(cache_pt, map_location="cpu"))
    with open(cache_pk, "rb") as fh:
        pred.calibrator = pickle.load(fh)
    return FrozenPredictor(pred, manifest)


def build_qchat10_features(polish_row: Dict[str, Any]) -> List[float]:
    """Construct the 10 binary features the frozen Saudi predictor expects.

    Delegates to :mod:`src.data.qchat10_contract`, which holds the verified
    item-by-item mapping and the ordinal -> binary rule derived from the
    instrument's own scoring direction.

    Raises :class:`ExternalValidationBlocked` when the contract is incomplete.
    A 9-feature vector is never returned: the frozen model's input width is
    fixed, so a short vector would either crash deep inside the network or,
    worse, be padded with a fabricated value.
    """
    from src.data.qchat10_contract import (ContractError, build_feature_matrix,
                                           contract_completeness)

    completeness = contract_completeness()
    if not completeness["complete"]:
        raise ExternalValidationBlocked(
            "Q-CHAT-10 feature contract is incomplete: no verified Polish variable "
            f"for model feature index {completeness['missing_feature_indices']}. "
            "Q-CHAT-10 item 10 has no verified correspondence in the Q-CHAT-25 "
            "instrument, and substituting a guess would silently corrupt the "
            "feature vector. Refusing to construct a partial vector."
        )
    table = {var: [value] for var, value in polish_row.items()}
    try:
        return build_feature_matrix(table)[0]
    except ContractError as exc:
        raise ExternalValidationBlocked(str(exc)) from exc


def external_metrics(probs: Sequence[float], labels: Sequence[int],
                     tau: float = 0.5) -> Dict[str, float]:
    """Screening metrics for an external cohort. No threshold is fitted."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    out: Dict[str, float] = {"n": int(p.size)}
    if p.size == 0:
        return out
    out["brier"] = float(np.mean((p - y) ** 2))
    out["prevalence"] = float(np.mean(y))
    out["sensitivity"] = float(np.mean(p[y == 1] >= tau)) if np.any(y == 1) else float("nan")
    out["specificity"] = float(np.mean(p[y == 0] < tau)) if np.any(y == 0) else float("nan")
    pp = float(np.mean((p >= tau) & (y == 1))) if np.any(y == 1) else 0.0
    out["precision"] = float(pp / (pp + float(np.mean((p >= tau) & (y == 0))))
                              if (pp + float(np.mean((p >= tau) & (y == 0)))) > 0
                              else float("nan"))
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out["auprc"] = (float(average_precision_score(y, p))
                    if len(np.unique(y)) > 1 else float("nan"))
    out["ece_10bin"] = _ece(p, y)
    out["tau"] = float(tau)
    return out


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def bootstrap_ci(values: Sequence[float], statistic=np.mean,
                 n_resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI over evaluation instances (spec §19.2)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    if arr.size < 2:
        return {"point": float(statistic(arr)) if arr.size else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n_resamples": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    stats = statistic(arr[idx], axis=1)
    return {
        "point": float(statistic(arr)),
        "lo": float(np.percentile(stats, 100 * alpha / 2)),
        "hi": float(np.percentile(stats, 100 * (1 - alpha / 2))),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }