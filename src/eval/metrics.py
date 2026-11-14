"""Metrics — §19.1"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, tau: float = 0.5) -> dict:
    y_pred = (y_prob >= tau).astype(int)
    out = {}
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["terminal_utility"] = float(np.mean(1 - (y_prob - y_true)**2))
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["auprc"] = float("nan")
    out["uar"] = float(balanced_accuracy_score(y_true, y_pred))
    # sensitivity/specificity
    if len(np.unique(y_true))==2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        out["sensitivity"] = float(tp/(tp+fn) if (tp+fn)>0 else 0)
        out["specificity"] = float(tn/(tn+fp) if (tn+fp)>0 else 0)
    else:
        out["sensitivity"]=out["specificity"]=float("nan")
    # ECE 10 bins
    out["ece"] = float(_ece(y_true, y_prob))
    cal = calibration_slope_intercept(y_true, y_prob)
    out["calibration_slope"] = cal["slope"]
    out["calibration_intercept"] = cal["intercept"]
    out["calibration_scale"] = CALIBRATION_SCALE
    return out


#: Marks which scale ``calibration_slope``/``calibration_intercept`` are on. Values written before the I10 fix
#: were fitted on the probability scale with an L2 penalty and are NOT comparable to these.
CALIBRATION_SCALE = "logit"
_EPS = 1e-6


def _logit(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _fit_logistic_1d(z: np.ndarray, y: np.ndarray, max_iter: int = 100):
    """Maximum-likelihood ``y ~ intercept + slope * z`` by Newton's method with step halving.

    Returns ``(intercept, slope)`` or ``None`` if it does not converge to a finite solution. Written out
    (two parameters) so the result does not depend on a library's regularisation defaults or deprecations.
    """
    from scipy.special import expit
    X = np.column_stack([np.ones_like(z), z])
    b = np.zeros(2)

    def nll(beta):
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    cur = nll(b)
    for _ in range(max_iter):
        mu = expit(X @ b)
        grad = X.T @ (y - mu)
        if np.max(np.abs(grad)) < 1e-8 * max(1.0, len(y)):
            return float(b[0]), float(b[1])
        hess = X.T @ (X * (mu * (1.0 - mu))[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while t > 1e-12:
            cand = b + t * step
            new = nll(cand)
            if np.isfinite(new) and new <= cur + 1e-12:
                break
            t /= 2.0
        else:
            return None
        b, cur = cand, new
        if not np.all(np.isfinite(b)) or np.max(np.abs(b)) > 1e4:
            return None
    return None


def calibration_fit(y_true, y_prob) -> dict:
    """Calibration slope and intercept (logistic recalibration of ``y`` on ``logit(p)``), with a reason when undefined.

    Slope 1 and intercept 0 mean perfect calibration (Van Calster et al.). Fitted **unpenalised on the logit
    scale** - the scale the definition requires; an earlier version regressed on the raw probability with an L2
    penalty, which shrinks the slope and is not this quantity.

    The estimate does not exist for some inputs, and a number is never invented for them: a single outcome
    class, constant predictions, or perfect separation (the maximum-likelihood slope is infinite) return NaN with
    ``undefined_reason``. Probabilities are clipped to ``[1e-6, 1 - 1e-6]`` before the logit so saturated (0/1)
    predictions stay finite.
    """
    out = {"slope": float("nan"), "intercept": float("nan"), "scale": CALIBRATION_SCALE, "undefined_reason": None}
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-D arrays of the same length")
    if y.size and not np.all(np.isin(y, (0, 1))):
        raise ValueError("y_true must contain only 0 and 1")
    y = y.astype(float)
    if not np.all(np.isfinite(p)):
        out["undefined_reason"] = "non-finite predictions"
        return out
    if y.size < 2 or np.unique(y).size < 2:
        out["undefined_reason"] = "single outcome class"
        return out
    z = _logit(p)
    if np.ptp(z) < 1e-9:
        out["undefined_reason"] = "constant predictions"
        return out
    z1, z0 = z[y == 1], z[y == 0]
    if z1.min() > z0.max() or z1.max() < z0.min():
        out["undefined_reason"] = "perfect separation (slope not finite)"
        return out
    fit = _fit_logistic_1d(z, y)
    if fit is None:
        out["undefined_reason"] = "maximum-likelihood fit did not converge"
        return out
    out["intercept"], out["slope"] = fit
    return out


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """``{"slope", "intercept"}`` on the logit scale; NaN when undefined. See :func:`calibration_fit`."""
    fit = calibration_fit(y_true, y_prob)
    return {"slope": fit["slope"], "intercept": fit["intercept"]}


def compute_metrics_with_ci(y_true: np.ndarray, y_prob: np.ndarray, tau: float = 0.5,
                            n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """compute_metrics + percentile bootstrap CIs for brier/auroc/uar.

    CIs are resampling (model-stability) statements over the evaluation
    sample; they are not clinical confidence intervals. Resamples that contain a single outcome class have no
    AUROC or UAR; they are excluded from those two metrics (never imputed) and counted in
    ``bootstrap_valid_resamples``.
    """
    from src.eval.fast_stats import bootstrap_core_metrics

    base = compute_metrics(y_true, y_prob, tau)
    draws = bootstrap_core_metrics(y_true, y_prob, tau=tau, n_bootstrap=n_bootstrap, seed=seed)
    out = dict(base)
    valid = {}
    for k, arr in draws.items():
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        valid[k] = int(finite.size)
        if finite.size:
            out[f"{k}_ci95"] = [float(np.nanpercentile(finite, 2.5)), float(np.nanpercentile(finite, 97.5))]
            out[f"{k}_bootstrap_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        else:
            out[f"{k}_ci95"] = [float("nan"), float("nan")]
            out[f"{k}_bootstrap_sd"] = float("nan")
    out["bootstrap_valid_resamples"] = valid
    out["bootstrap_n_resamples"] = int(n_bootstrap)
    return out

def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    e=0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob>=lo) & (y_prob<hi if i<n_bins-1 else y_prob<=hi)
        if mask.sum()==0: continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        e+= abs(acc-conf)*mask.sum()/len(y_true)
    return e

def acquisition_burden(items_asked_list):
    lens = [len(x) for x in items_asked_list]
    return {"mean": float(np.mean(lens)), "median": float(np.median(lens)), "lens": lens}
