"""External-validation metrics for the sealed Polish cohort.

Scope
-----
Primary result only. Every quantity here is computed from a single frozen
prediction pass: the frozen Saudi predictor is loaded, the approved Q-CHAT-10
projection is applied to the Polish questionnaire responses, and the resulting
probability is compared against the clinician-established ``group`` label.

Nothing in this module fits, tunes or calibrates anything. In particular:

* the decision threshold is supplied by the caller and is never searched over;
* the calibration intercept/slope are **diagnostics** reported for interpretation
  (Van Calster's calibration-in-the-large / calibration slope), computed on the
  external data purely to describe miscalibration. They are never used to adjust a
  prediction, to recalibrate the frozen calibrator, or to select the threshold;
* confidence intervals come from a **paired** percentile bootstrap that resamples
  (label, probability) pairs together, so threshold-dependent metrics keep their
  pairing. No interval is ever invented: if a statistic is undefined for the data
  at hand it is returned as ``None`` with a reason, not as a number.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Metrics for which a paired bootstrap interval is computed.
_BOOTSTRAP_METRICS = ("brier", "auroc", "auprc", "sensitivity", "specificity",
                      "ppv", "npv", "ece_10bin", "accuracy")


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _safe(fn: Callable[[], Any]) -> Optional[float]:
    try:
        value = fn()
    except Exception:
        return None
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def confusion_counts(y: np.ndarray, p: np.ndarray,
                     tau: float) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= tau).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp,
            "false_negative": fn,
            "n": int(y.size),
            "rows_are_actual": "rows = actual class (0 control, 1 ASD)",
            "cols_are_predicted": "cols = predicted at tau",
            "matrix": [[tn, fp], [fn, tp]]}


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def core_metrics(y: np.ndarray, p: np.ndarray, tau: float) -> Dict[str, Any]:
    """Point estimates. Returns ``None`` for anything not legitimately computable."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    counts = confusion_counts(y, p, tau)
    tp, tn, fp, fn = (counts["true_positive"], counts["true_negative"],
                      counts["false_positive"], counts["false_negative"])

    out: Dict[str, Any] = {
        "n": int(y.size),
        "n_asd": int(np.sum(y == 1)),
        "n_control": int(np.sum(y == 0)),
        "prevalence": _safe(lambda: float(np.mean(y))),
        "tau": float(tau),
    }
    if p.size == 0 or np.unique(y).size < 2:
        out["undefined_reason"] = (
            "cohort is empty or single-class; discrimination metrics are undefined")
        for key in ("brier", "auroc", "auprc", "sensitivity", "specificity",
                    "ppv", "npv", "accuracy", "ece_10bin"):
            out[key] = None
        out["confusion_matrix"] = counts
        return out

    out["brier"] = _safe(lambda: float(np.mean((p - y) ** 2)))
    out["auroc"] = _safe(lambda: float(roc_auc_score(y, p)))
    out["auprc"] = _safe(lambda: float(average_precision_score(y, p)))
    out["auprc_baseline_prevalence"] = _safe(lambda: float(np.mean(y)))
    out["sensitivity"] = _safe(lambda: tp / (tp + fn) if (tp + fn) else None)
    out["specificity"] = _safe(lambda: tn / (tn + fp) if (tn + fp) else None)
    out["ppv"] = _safe(lambda: tp / (tp + fp) if (tp + fp) else None)
    out["npv"] = _safe(lambda: tn / (tn + fn) if (tn + fn) else None)
    out["accuracy"] = _safe(lambda: (tp + tn) / (tp + tn + fp + fn))
    out["ece_10bin"] = _safe(lambda: _ece(p, y))
    out["mean_predicted_risk"] = _safe(lambda: float(np.mean(p)))
    out["confusion_matrix"] = counts
    return out


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    """Calibration-in-the-large and calibration slope, as reported diagnostics.

    Standard external-validation practice (Van Calster et al.): an intercept-only
    logistic regression on the logit of the predicted probability gives the
    calibration-in-the-large; adding a slope gives the calibration slope. A
    perfectly calibrated model gives intercept 0 and slope 1.

    These are DESCRIPTIONS of the frozen model's behaviour on this cohort. Nothing
    here feeds back into the model, the calibrator or the threshold.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: Dict[str, Any] = {
        "role": "diagnostic_only",
        "note": ("Reported to describe miscalibration of the frozen model on this "
                 "cohort. Not used to recalibrate, to adjust any prediction, or to "
                 "select a threshold."),
        "interpretation": ("intercept 0 and slope 1 indicate perfect calibration; "
                           "intercept < 0 indicates systematic overprediction; "
                           "slope < 1 indicates predictions too extreme"),
    }
    if p.size < 10 or np.unique(y).size < 2:
        out["undefined_reason"] = "too few observations or single-class outcome"
        out["calibration_in_the_large"] = None
        out["calibration_slope"] = None
        return out

    z = _logit(p).reshape(-1, 1)
    try:
        intercept_only = LogisticRegression(penalty=None, solver="lbfgs",
                                            max_iter=2000).fit(z, y)
        out["calibration_in_the_large"] = float(intercept_only.intercept_[0])
        out["calibration_in_the_large_p"] = float(
            intercept_only.predict_proba(np.array([[0.0]]))[0, 1])
    except Exception as exc:
        out["calibration_in_the_large"] = None
        out["calibration_in_the_large_error"] = str(exc)

    try:
        with_slope = LogisticRegression(penalty=None, solver="lbfgs",
                                         max_iter=2000).fit(z, y)
        out["calibration_slope"] = float(with_slope.coef_[0][0])
    except Exception as exc:
        out["calibration_slope"] = None
        out["calibration_slope_error"] = str(exc)

    # Observed vs expected by decile, a directly readable calibration table.
    edges = np.linspace(0.0, 1.0, 11)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.any():
            table.append({
                "bin_lower": float(lo), "bin_upper": float(hi),
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
            })
    out["calibration_table"] = table
    return out


def bootstrap_intervals(y: np.ndarray, p: np.ndarray, tau: float,
                        n_resamples: int = 2000, seed: int = 0,
                        alpha: float = 0.05) -> Dict[str, Any]:
    """Paired percentile bootstrap CIs over evaluation instances.

    Rows are resampled with replacement and the label travels with the
    probability, which keeps threshold-dependent metrics correctly paired. A
    resample that is single-class yields an undefined AUROC; those resamples are
    excluded and the count is reported rather than silently imputed.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if p.size < 2:
        return {"status": "UNDEFINED", "reason": "fewer than 2 observations",
                "n_resamples": 0, "intervals": {}}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, p.size, size=(n_resamples, p.size))
    draws: Dict[str, List[float]] = {k: [] for k in _BOOTSTRAP_METRICS}
    degenerate = 0
    for row in idx:
        ys, ps = y[row], p[row]
        if np.unique(ys).size < 2:
            degenerate += 1
            continue
        stats = core_metrics(ys, ps, tau)
        for key in _BOOTSTRAP_METRICS:
            value = stats.get(key)
            if value is not None and np.isfinite(value):
                draws[key].append(float(value))

    intervals: Dict[str, Any] = {}
    for key, values in draws.items():
        if len(values) < 2:
            intervals[key] = {"point": None, "lo": None, "hi": None,
                              "n_usable_resamples": len(values),
                              "note": "insufficient usable resamples"}
            continue
        arr = np.asarray(values)
        intervals[key] = {
            "point": float(np.mean(arr)),
            "lo": float(np.percentile(arr, 100 * alpha / 2)),
            "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n_usable_resamples": len(values),
        }

    return {
        "status": "OK",
        "method": "paired percentile bootstrap over evaluation instances",
        "n_resamples": int(n_resamples),
        "n_degenerate_resamples_excluded": int(degenerate),
        "seed": int(seed),
        "alpha": float(alpha),
        "confidence_level": 1.0 - float(alpha),
        "intervals": intervals,
        "note": ("Intervals describe sampling variability of the estimate on this "
                 "cohort. They do not address cohort selection, label definition, or "
                 "cross-country generalisation."),
    }