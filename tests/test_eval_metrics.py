"""Calibration slope on the logit scale (audit finding I10) and the vectorised bootstrap (P7)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.eval import metrics
from src.eval.fast_stats import auroc, auroc_rows, bootstrap_core_metrics
from src.eval.metrics import calibration_fit, calibration_slope_intercept, compute_metrics, compute_metrics_with_ci


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def simulate(n, seed, stretch=1.0, shift=0.0):
    """Outcomes drawn from the true risk; predictions are the true logit stretched/shifted (miscalibrated on purpose)."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0, 1.5, n)
    y = (rng.random(n) < sigmoid(true_logit)).astype(int)
    return y, sigmoid(stretch * true_logit + shift)


# ------------------------------------------------------------------ I10: the estimate itself


@pytest.mark.filterwarnings("ignore:'penalty' was deprecated:FutureWarning")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_slope_and_intercept_match_an_unpenalised_logit_regression(seed):
    from sklearn.linear_model import LogisticRegression
    y, p = simulate(600, seed, stretch=1.7, shift=0.3)
    z = metrics._logit(p).reshape(-1, 1)
    ref = LogisticRegression(penalty=None, solver="newton-cholesky", max_iter=1000, tol=1e-12).fit(z, y)
    got = calibration_fit(y, p)
    assert got["slope"] == pytest.approx(float(ref.coef_[0][0]), rel=1e-5)
    assert got["intercept"] == pytest.approx(float(ref.intercept_[0]), abs=1e-5)


def test_a_calibrated_model_has_slope_one_and_intercept_zero():
    y, p = simulate(40000, 11)
    got = calibration_fit(y, p)
    assert got["slope"] == pytest.approx(1.0, abs=0.06) and got["intercept"] == pytest.approx(0.0, abs=0.06)


def test_overconfident_predictions_have_slope_below_one_and_underconfident_above():
    y, over = simulate(40000, 12, stretch=3.0)
    assert calibration_fit(y, over)["slope"] == pytest.approx(1 / 3, abs=0.04)
    y, under = simulate(40000, 13, stretch=0.5)
    assert calibration_fit(y, under)["slope"] == pytest.approx(2.0, abs=0.2)


def test_a_constant_overprediction_shows_up_in_the_intercept_not_the_slope():
    y, p = simulate(40000, 14, shift=1.0)
    got = calibration_fit(y, p)
    assert got["intercept"] == pytest.approx(-1.0, abs=0.08) and got["slope"] == pytest.approx(1.0, abs=0.06)


@pytest.mark.filterwarnings("ignore:'penalty' was deprecated:FutureWarning")
def test_the_estimate_is_on_the_logit_scale_not_the_old_probability_scale():
    """The pre-fix code regressed on raw p with an L2 penalty. For this sample that gave a slope near 6.7; the
    definition (logit scale) gives about 2.3. The difference is the whole reason for I10."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    y = (rng.random(506) < 0.55).astype(int)
    p = np.clip(0.5 + 0.35 * (y - 0.5) * 2 * rng.random(506) + rng.normal(0, 0.15, 506), 0.01, 0.99)
    old = LogisticRegression(penalty="l2", C=1.0, max_iter=5000).fit(np.clip(p, 1e-6, 1 - 1e-6).reshape(-1, 1), y)
    new = calibration_fit(y, p)["slope"]
    assert float(old.coef_[0][0]) > 6.0 and 2.0 < new < 2.6


def test_wrapper_returns_only_slope_and_intercept_for_existing_callers():
    y, p = simulate(300, 5)
    out = calibration_slope_intercept(y, p)
    assert set(out) == {"slope", "intercept"} and all(isinstance(v, float) for v in out.values())


def test_compute_metrics_carries_the_scale_marker():
    y, p = simulate(300, 6)
    m = compute_metrics(y, p)
    assert m["calibration_scale"] == "logit"
    assert m["calibration_slope"] == pytest.approx(calibration_fit(y, p)["slope"])


# ------------------------------------------------------------------ edge cases and numerical stability


@pytest.mark.parametrize("y, p, reason", [
    ([1, 1, 1, 1], [0.2, 0.4, 0.6, 0.8], "single outcome class"),
    ([0, 0, 0], [0.2, 0.4, 0.6], "single outcome class"),
    ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], "constant predictions"),
    ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], "perfect separation"),
    ([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], "perfect separation"),       # perfectly *inverted* ranking
    ([0, 1], [0.2, 0.8], "perfect separation"),
])
def test_undefined_cases_are_nan_with_a_reason_and_never_raise_or_warn(y, p, reason):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = calibration_fit(np.array(y), np.array(p))
    assert np.isnan(got["slope"]) and np.isnan(got["intercept"])
    assert reason in got["undefined_reason"]


def test_non_finite_predictions_are_undefined():
    got = calibration_fit(np.array([0, 1, 0, 1]), np.array([0.2, np.nan, 0.4, 0.9]))
    assert np.isnan(got["slope"]) and "non-finite" in got["undefined_reason"]


def test_saturated_predictions_with_some_errors_stay_finite():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])     # exact 0/1, two confident mistakes
    got = calibration_fit(y, p)
    assert np.isfinite(got["slope"]) and np.isfinite(got["intercept"]) and got["undefined_reason"] is None


def test_probabilities_outside_the_unit_interval_are_clipped_not_exploded():
    got = calibration_fit(np.array([0, 1, 0, 1, 1, 0]), np.array([0.0, 1.0, 1.0, 0.0, 0.7, 0.3]))
    assert np.isfinite(got["slope"])


def test_extreme_but_overlapping_predictions_converge():
    y, p = simulate(2000, 21, stretch=6.0)
    got = calibration_fit(y, p)
    assert np.isfinite(got["slope"]) and 0 < got["slope"] < 0.5


def test_small_samples_do_not_crash():
    got = calibration_fit(np.array([0, 1, 1, 0, 1]), np.array([0.6, 0.4, 0.7, 0.2, 0.5]))
    assert got["undefined_reason"] is None or isinstance(got["undefined_reason"], str)
    assert calibration_fit(np.array([]), np.array([]))["undefined_reason"] == "single outcome class"


@pytest.mark.parametrize("y, p", [([0, 2, 1], [0.1, 0.2, 0.3]), ([0, 1], [0.1, 0.2, 0.3]), ([[0, 1]], [[0.1, 0.2]])])
def test_invalid_inputs_are_rejected(y, p):
    with pytest.raises(ValueError):
        calibration_fit(np.array(y), np.array(p))


def test_boolean_and_float_labels_are_accepted():
    y, p = simulate(300, 9)
    base = calibration_fit(y, p)["slope"]
    assert calibration_fit(y.astype(bool), p)["slope"] == pytest.approx(base)
    assert calibration_fit(y.astype(float), p)["slope"] == pytest.approx(base)


# ------------------------------------------------------------------ P7: the bootstrap


def reference_bootstrap(y, p, tau, n_bootstrap, seed):
    """The pre-fix algorithm: one compute_metrics call per resample, drawing indices with the same generator."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = {k: [] for k in ("brier", "auroc", "uar")}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            continue
        m = compute_metrics(y[idx], p[idx], tau)
        for k in vals:
            vals[k].append(m[k])
    return {k: np.asarray(v) for k, v in vals.items()}


def test_vectorised_bootstrap_reproduces_the_loop_it_replaces():
    y, p = simulate(120, 31, stretch=0.8)
    ref = reference_bootstrap(y, p, 0.5, 60, seed=4)
    got = bootstrap_core_metrics(y, p, tau=0.5, n_bootstrap=60, seed=4)
    for k in ref:
        assert np.allclose(got[k][np.isfinite(got[k])], ref[k], rtol=0, atol=1e-12), k


def test_the_bootstrap_does_not_depend_on_the_chunk_size():
    y, p = simulate(80, 32)
    a = bootstrap_core_metrics(y, p, n_bootstrap=53, seed=9, chunk=7)
    b = bootstrap_core_metrics(y, p, n_bootstrap=53, seed=9, chunk=256)
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True)


def test_same_seed_same_result_different_seed_different_result():
    y, p = simulate(100, 33)
    a = compute_metrics_with_ci(y, p, n_bootstrap=100, seed=1)
    b = compute_metrics_with_ci(y, p, n_bootstrap=100, seed=1)
    c = compute_metrics_with_ci(y, p, n_bootstrap=100, seed=2)
    assert a == b and a["brier_ci95"] != c["brier_ci95"]


def test_vectorised_pieces_match_sklearn_including_ties():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=(5, 40))
    s = np.round(rng.random((5, 40)), 1)                       # heavy ties
    got = auroc_rows(y, s)
    for i in range(5):
        assert got[i] == pytest.approx(roc_auc_score(y[i], s[i])) == pytest.approx(auroc(y[i], s[i]))
    y2, p2 = simulate(60, 4)
    draws = bootstrap_core_metrics(y2, p2, tau=0.4, n_bootstrap=5, seed=0)
    rng = np.random.default_rng(0)
    for i in range(5):
        idx = rng.integers(0, 60, size=60)
        assert draws["uar"][i] == pytest.approx(balanced_accuracy_score(y2[idx], (p2[idx] >= 0.4).astype(int)))
        assert draws["brier"][i] == pytest.approx(np.mean((p2[idx] - y2[idx]) ** 2))


def test_single_class_resamples_are_excluded_and_counted_not_poisoning_the_interval():
    """With one positive in six, many resamples have no positive: AUROC/UAR are undefined there. The old code
    passed the resulting NaNs to np.percentile, making the whole interval NaN."""
    y = np.array([0, 0, 0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.9])
    out = compute_metrics_with_ci(y, p, n_bootstrap=400, seed=0)
    assert np.all(np.isfinite(out["auroc_ci95"])) and np.all(np.isfinite(out["uar_ci95"]))
    valid = out["bootstrap_valid_resamples"]
    assert 0 < valid["auroc"] < 400 and valid["uar"] == valid["auroc"] and valid["brier"] == 400
    assert out["bootstrap_n_resamples"] == 400


def test_a_single_class_evaluation_sample_gives_nan_for_auroc_but_not_brier():
    y = np.zeros(20, dtype=int)
    p = np.linspace(0.05, 0.6, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_metrics_with_ci(y, p, n_bootstrap=50, seed=0)
    assert np.all(np.isnan(out["auroc_ci95"])) and out["bootstrap_valid_resamples"]["auroc"] == 0
    assert np.all(np.isfinite(out["brier_ci95"]))


def test_bootstrap_validates_its_inputs():
    with pytest.raises(ValueError):
        bootstrap_core_metrics([], [], n_bootstrap=3)
    with pytest.raises(ValueError):
        bootstrap_core_metrics([0, 1], [0.1], n_bootstrap=3)


def test_the_point_estimate_inside_the_interval_for_a_typical_sample():
    y, p = simulate(500, 41, stretch=0.9)
    out = compute_metrics_with_ci(y, p, n_bootstrap=500, seed=0)
    for k in ("brier", "auroc", "uar"):
        lo, hi = out[f"{k}_ci95"]
        assert lo <= out[k] <= hi, k
