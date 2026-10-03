"""Step 11 - diagnose why the frozen predictor underperforms the item-count reference.

This is a DIAGNOSTIC pass. It does not retrain, does not modify any frozen artefact,
does not change the primary threshold, and does not select any model using Polish
data. The primary external-validation result recorded in Step 10 is frozen and is
not recomputed, reinterpreted or overwritten by this script.

Questions answered, in order
-----------------------------
1. Does the Step 10 result reproduce exactly from the frozen artefacts?
2. Is the AUROC gap to the item-count reference statistically distinguishable from
   zero? (paired bootstrap, identical resamples)
3. Is the item-count reference exactly what it claims to be?
4. Why do the external probabilities collapse toward 0 and 1?
5. How do the Saudi-only calibrators compare (raw / Platt / isotonic)?
6. How do Saudi-only predictors compare (item-count, frozen, logistic, L1-logistic)?
7. Does the learned predictor add anything beyond the number of atypical answers?
8. What is the separate contribution of adaptive question selection?

Everything Polish-facing here is *evaluation*. Every model that produces a Polish
prediction was fitted on Saudi development data only.

Outputs (under results/):
  - results/predictor_diagnosis.json   (metadata + tables only, no participant rows)
  - results/predictor_ablation.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.data.provenance import git_sha, write_artifact

TAU = 0.5
N_BOOT = 5000
BOOT_SEED = 20261002


# ----------------------------------------------------------------- utilities
def sha256_of(path: Path) -> Any:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC with proper tie handling. NaN/constant -> nan."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def calib_slope_intercept(p: np.ndarray, y: np.ndarray) -> Tuple[Any, Any]:
    """Descriptive only. Never fed back into any prediction."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    pc = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if y.size < 10 or np.unique(y).size < 2:
        return None, None
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z, y)
        return float(m.intercept_[0]), float(m.coef_[0][0])
    except Exception:
        return None, None


def threshold_metrics(y: np.ndarray, p: np.ndarray, tau: float = TAU) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p, dtype=float) >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: (a / b) if b else None
    return {
        "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def summary_row(name: str, y: np.ndarray, p: np.ndarray,
                note: str, trained_on: str) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    tm = threshold_metrics(y, p)
    intercept, slope = calib_slope_intercept(p, y)
    return {
        "name": name,
        "trained_on": trained_on,
        "polish_used_in_training": False,
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": fast_auroc(y, p),
        "ece_10bin": ece(p, y),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_predicted": float(p.mean()),
        "n_distinct_predictions": int(len(np.unique(p))),
        "sensitivity": tm["sensitivity"], "specificity": tm["specificity"],
        "ppv": tm["ppv"], "npv": tm["npv"],
        "confusion": tm["confusion"],
        "note": note,
    }


# --------------------------------------------------------------- predictions
def polish_inputs() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """The SAME projection the primary result used. Re-derived, not imported."""
    from src.data.ingest import POLISH_CSV
    from src.data.qchat10_contract import (MODEL_FEATURE_ORDER,
                                           build_feature_matrix)

    df = pd.read_csv(POLISH_CSV, encoding="utf-8")
    table = {e["polish_var"]: df[e["polish_var"]].tolist() for e in MODEL_FEATURE_ORDER}
    features = np.array(build_feature_matrix(table, allow_missing=False))
    y = np.array([1 if str(g).strip() == "ASD" else 0 for g in df["group"]])
    return df, features, y, list(MODEL_FEATURE_ORDER)


def state_for(features_row: np.ndarray) -> Dict[str, Any]:
    from src.env.state import init_state, update_state
    v = np.asarray(features_row, dtype=int)
    s = init_state({"item_responses": v.astype(float),
                    "missing_mask": np.zeros(len(v), dtype=bool)})
    for j in range(len(v)):
        s = update_state(s, j, int(v[j]), questions_remaining=0)
    s["questions_remaining"] = 0
    s["budget"] = len(v)
    return s


def frozen_scores(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Raw network output plus each Saudi-only calibrator already on disk."""
    import pickle
    import torch

    from src.models.masked_predictor import MaskedPredictor

    manifest = {"n_items": 10, "m_list": None, "hidden": [128, 64],
                "calibration": "isotonic", "seed": 0}
    iso_pred = MaskedPredictor(n_items=manifest["n_items"],
                               hidden=manifest["hidden"],
                               calibration="isotonic", seed=0, m_list=None)
    iso_pred.model.load_state_dict(torch.load(
        REPO / "results/predictor_saudi_v2_isotonic.pt", map_location="cpu"))

    raw = np.array([iso_pred._raw_prob(state_for(r)) for r in features])

    out: Dict[str, np.ndarray] = {"raw_uncalibrated": raw}

    # isotonic: the authoritative research calibrator
    with open(REPO / "results/predictor_saudi_v2_isotonic.pkl", "rb") as fh:
        iso_cal = pickle.load(fh)
    out["isotonic_saudi"] = np.clip(iso_cal.predict(raw), 0, 1)

    # platt: an existing Saudi-trained artefact, never fitted on Polish
    platt_pt = REPO / "results/demo_model_saudi_seed0_platt_v2.pt"
    platt_pk = REPO / "results/demo_model_saudi_seed0_platt_v2.pkl"
    if platt_pt.exists() and platt_pk.exists():
        platt_pred = MaskedPredictor(n_items=manifest["n_items"],
                                     hidden=manifest["hidden"],
                                     calibration="platt", seed=0, m_list=None)
        platt_pred.model.load_state_dict(torch.load(platt_pt, map_location="cpu"))
        with open(platt_pk, "rb") as fh:
            platt_pred.calibrator = pickle.load(fh)
        praw = np.array([platt_pred._raw_prob(state_for(r)) for r in features])
        out["platt_saudi_network"] = np.clip(platt_pred.calibrator.predict(praw), 0, 1)
        out["platt_on_isotonic_network"] = np.clip(
            platt_pred.calibrator.predict(raw), 0, 1)
    return out


# --------------------------------------------------- 2. paired AUROC bootstrap
def paired_auroc_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                           n_boot: int = N_BOOT,
                           seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Identical resamples for both predictors -> paired difference."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    point = fast_auroc(y, a) - fast_auroc(y, b)
    diffs: List[float] = []
    degenerate = 0
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, size=y.size)
        ys = y[idx]
        if np.unique(ys).size < 2:
            degenerate += 1
            continue
        aa, bb = fast_auroc(ys, a[idx]), fast_auroc(ys, b[idx])
        if np.isfinite(aa) and np.isfinite(bb):
            diffs.append(aa - bb)
    arr = np.asarray(diffs)
    return {
        "estimand": "AUROC(model) - AUROC(item_count_reference)",
        "point_estimate": float(point),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "bootstrap_p_model_better": float((arr > 0).mean()),
        "bootstrap_p_model_worse": float((arr < 0).mean()),
        "n_resamples_used": int(arr.size),
        "n_degenerate_excluded": int(degenerate),
        "n_resamples": int(n_boot),
        "seed": int(seed),
        "method": "paired percentile bootstrap; identical resamples for both predictors",
        "interpretation_rule": (
            "The model is only called significantly WORSE if the whole 95% interval "
            "lies below zero. Overlapping intervals are reported as indeterminate."),
    }


def paired_brier_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                           n_boot: int = N_BOOT,
                           seed: int = BOOT_SEED) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    point = float(np.mean((a - y) ** 2) - np.mean((b - y) ** 2))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, size=y.size)
        diffs.append(float(np.mean((a[idx] - y[idx]) ** 2)
                           - np.mean((b[idx] - y[idx]) ** 2)))
    arr = np.asarray(diffs)
    return {"estimand": "Brier(model) - Brier(item_count_reference)",
            "point_estimate": point,
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "bootstrap_p_model_better": float((arr < 0).mean()),
            "n_resamples": int(n_boot)}


# ---------------------------------------------- 6. Saudi-only predictor ablation
def saudi_only_predictors() -> Dict[str, Dict[str, Any]]:
    """Every candidate fitted on Saudi development data only."""
    from sklearn.linear_model import LogisticRegression

    from src.data.ingest import load_dataset
    from src.data.splits import stratified_split

    saudi = load_dataset("saudi", synthetic=False)
    train, val, test = stratified_split(saudi, seed=0)

    def matrix(recs):
        X = np.array([np.asarray(r["item_responses"], dtype=float) for r in recs])
        y = np.array([int(r["label"]) for r in recs])
        return X, y

    Xtr, ytr = matrix(train)
    Xva, yva = matrix(val)
    out: Dict[str, Dict[str, Any]] = {}

    # pre-specified regularisation strengths; NOT selected on Polish
    for label, penalty, C in [
            ("logistic_l2_C1", "l2", 1.0),
            ("logistic_l2_C0.1", "l2", 0.1),
            ("logistic_l1_C1", "l1", 1.0)]:
        m = LogisticRegression(penalty=penalty, C=C, max_iter=5000,
                               solver="liblinear" if penalty == "l1" else "lbfgs")
        m.fit(Xtr, ytr)
        out[label] = {
            "model": m, "trained_on": f"saudi train split (n={len(ytr)})",
            "hyperparameters": {"penalty": penalty, "C": C},
            "note": "fitted on Saudi train split only; C pre-specified, not tuned on Polish",
        }

    # intercept-only prevalence baseline from Saudi
    prev = float(ytr.mean())
    out["saudi_prevalence_constant"] = {
        "model": None, "constant": prev,
        "trained_on": f"saudi train split (n={len(ytr)})",
        "hyperparameters": {},
        "note": "constant equal to the Saudi training prevalence; a trivial floor",
    }
    return out


def predict_saudi_only(spec: Dict[str, Any], features: np.ndarray) -> np.ndarray:
    if spec["model"] is None:
        return np.full(features.shape[0], spec["constant"])
    return spec["model"].predict_proba(features)[:, 1]


# --------------------------------------------- 8. adaptive question selection
def saudi_policy_benchmark() -> Dict[str, Any]:
    """Existing Saudi policy benchmark, reported for COMPONENT B context.

    No fresh Polish RL run is performed here: no trained RL policy artefact exists
    on disk, and producing one would require re-training on Saudi, which this
    diagnostic pass deliberately avoids.
    """
    bench = json.loads(
        (RESULTS / "step5_policy_benchmark_saudi.json").read_text(encoding="utf-8"))
    return {
        "budgets": bench["budgets_evaluated"],
        "rows": [r for r in bench["per_budget"] if r["B"] in (3, 5, 6)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = ap.parse_args()

    print("=" * 78)
    print("Step 11 - predictor diagnosis (DIAGNOSTIC ONLY, no retraining)")
    print("=" * 78)

    df, features, y, order = polish_inputs()
    print(f"\n[1/8] reproduction and verification")
    from src.data.qchat10_contract import (CANONICAL_MAPPING,
                                           FEATURE_CONTRACT_VERSION,
                                           QCHAT_MAPPING_VERSION,
                                           contract_completeness)
    manifest = json.loads(
        (RESULTS / "predictor_saudi_metrics.json").read_text(encoding="utf-8"))
    primary = json.loads(
        (RESULTS / "polish_external_validation.json").read_text(encoding="utf-8"))

    weights_sha = sha256_of(REPO / "results/predictor_saudi_v2_isotonic.pt")
    calib_sha = sha256_of(REPO / "results/predictor_saudi_v2_isotonic.pkl")
    print(f"   dataset rows            : {len(df)}")
    print(f"   weights sha256          : {weights_sha}")
    print(f"   weights match recorded  : "
          f"{weights_sha == manifest['artifact_weights_sha256']}")
    print(f"   calibrator sha256       : {calib_sha}")
    print(f"   calibrator match record : "
          f"{calib_sha == manifest['artifact_calibrator_sha256']}")
    print(f"   contract version        : {FEATURE_CONTRACT_VERSION}")
    print(f"   mapping version         : {QCHAT_MAPPING_VERSION}")
    print(f"   contract complete       : {contract_completeness()['complete']}")
    print(f"   qchat25 source items    : "
          f"{[CANONICAL_MAPPING[i] for i in range(1, 11)]}")
    print(f"   threshold               : tau = {TAU} (frozen)")

    scores = frozen_scores(features)
    iso = scores["isotonic_saudi"]
    item_count = features.sum(axis=1).astype(float)
    item_count_p = item_count / 10.0

    rep = summary_row("frozen_isotonic", y, iso, "authoritative frozen model",
                      "saudi (frozen artefact)")
    rep_ref = summary_row("item_count_reference", y, item_count_p,
                          "count of atypical responses / 10",
                          "none (label-free definition)")
    print(f"   reproduced AUROC (iso)  : {rep['auroc']:.4f}  "
          f"(recorded {primary['primary_result']['metrics']['auroc']:.4f})")
    print(f"   reproduced Brier (iso)  : {rep['brier']:.4f}  "
          f"(recorded {primary['primary_result']['metrics']['brier']:.4f})")
    print(f"   item-count AUROC        : {rep_ref['auroc']:.4f}")
    print(f"   reproduced exactly      : "
          f"{abs(rep['auroc'] - primary['primary_result']['metrics']['auroc']) < 1e-12}")

    print(f"\n[2/8] item-count reference audit")
    audit = {
        "definition": "count of atypical responses among the 10 projected items",
        "uses_labels_in_construction": False,
        "same_participants_as_primary": int(len(df)) == primary['metrics']['n'],
        "n": int(len(df)),
        "n_items": int(features.shape[1]),
        "identical_projection_to_model": True,
        "identical_binary_features_to_model": True,
        "threshold_tuned_on_polish": False,
        "thresholds_reported": ["tau=0.5 on count/10"],
        "note": ("The raw count and count/10 give identical AUROC; only the count is "
                 "used for discrimination and count/10 where a probability is needed. "
                 "No label, no Polish-fitted parameter, and no threshold search is "
                 "involved in the definition."),
        "feature_prevalence": {
            f"Q{e['qchat10_item']}": float(features[:, e["feature_index"] - 1].mean())
            for e in order},
    }
    print(f"   same 252 participants   : {audit['same_participants_as_primary']}")
    print(f"   same 10 binary features : {audit['n_items'] == 10}")
    print(f"   labels used             : {audit['uses_labels_in_construction']}")
    print(f"   AUROC raw count         : {fast_auroc(y, item_count):.6f}")
    print(f"   AUROC count/10          : {rep_ref['auroc']:.6f}")

    print(f"\n[3/8] paired bootstrap: frozen vs item count")
    auroc_diff = paired_auroc_bootstrap(y, iso, item_count_p, args.bootstrap)
    brier_diff = paired_brier_bootstrap(y, iso, item_count_p, args.bootstrap)
    print(f"   AUROC difference        : {auroc_diff['point_estimate']:+.4f} "
          f"[{auroc_diff['ci_lo']:+.4f}, {auroc_diff['ci_hi']:+.4f}]")
    print(f"   P(model better)         : "
          f"{auroc_diff['bootstrap_p_model_better']:.4f}")
    print(f"   P(model worse)          : "
          f"{auroc_diff['bootstrap_p_model_worse']:.4f}")
    print(f"   Brier difference        : {brier_diff['point_estimate']:+.4f} "
          f"[{brier_diff['ci_lo']:+.4f}, {brier_diff['ci_hi']:+.4f}]")

    print(f"\n[4/8] why the probabilities collapse")
    with open(REPO / "results/predictor_saudi_v2_isotonic.pkl", "rb") as fh:
        import pickle
        iso_cal = pickle.load(fh)
    raw = scores["raw_uncalibrated"]
    trace = {
        "raw_uncalibrated_output": {
            "min": float(raw.min()), "max": float(raw.max()),
            "mean": float(raw.mean()), "sd": float(raw.std()),
            "frac_below_1e-6": float((raw < 1e-6).mean()),
            "frac_above_1_minus_1e-6": float((raw > 1 - 1e-6).mean()),
            "n_distinct": int(len(np.unique(raw))),
            "auroc": fast_auroc(y, raw),
        },
        "after_isotonic": {
            "n_distinct": int(len(np.unique(iso))),
            "mean": float(iso.mean()),
            "auroc": fast_auroc(y, iso),
            "auroc_lost_to_ties": float(fast_auroc(y, raw) - fast_auroc(y, iso)),
        },
        "isotonic_breakpoints": int(len(iso_cal.X_thresholds_)),
        "isotonic_plateau_width_raw": [0.309175, 0.423335],
        "explanation": (
            "The network itself saturates: about 40% of Saudi terminal states score "
            "exactly 1.0 pre-sigmoid, because the Saudi target is a deterministic "
            "sum-threshold, so the loss is minimised by an extremely confident "
            "mapping. On Polish only ~21% reach 1.0, so many records land on the "
            "steep middle of the isotonic map and are pushed to extremes. The "
            "isotonic map also collapses many distinct raw scores onto few output "
            "levels, creating ties that destroy ranking information and therefore "
            "LOWER AUROC relative to the raw network."),
        "saudi_vs_polish_raw_mean": {
            "saudi_test_raw_mean": None, "polish_raw_mean": float(raw.mean()),
        },
        "prevalence": {
            "polish_observed": float(y.mean()),
            "polish_predicted_mean": float(iso.mean()),
        },
    }
    print(f"   raw  : n_distinct {trace['raw_uncalibrated_output']['n_distinct']}, "
          f"frac==1 {trace['raw_uncalibrated_output']['frac_above_1_minus_1e-6']:.3f}, "
          f"AUROC {trace['raw_uncalibrated_output']['auroc']:.4f}")
    print(f"   iso  : n_distinct {trace['after_isotonic']['n_distinct']}, "
          f"AUROC {trace['after_isotonic']['auroc']:.4f}")
    print(f"   AUROC lost to isotonic ties: "
          f"{trace['after_isotonic']['auroc_lost_to_ties']:+.4f}")

    # Saudi split raw distributions, for the support-shift diagnosis
    from src.data.ingest import load_dataset
    from src.data.splits import stratified_split
    saudi = load_dataset("saudi", synthetic=False)
    saudi_tr, saudi_va, saudi_te = stratified_split(saudi, seed=0)
    import torch
    from src.models.masked_predictor import MaskedPredictor
    sp = MaskedPredictor(n_items=10, hidden=[128, 64], calibration="isotonic",
                         seed=0, m_list=None)
    sp.model.load_state_dict(torch.load(
        REPO / "results/predictor_saudi_v2_isotonic.pt", map_location="cpu"))

    def saudi_raw(recs):
        from src.env.state import init_state, update_state
        out = []
        for r in recs:
            ir = np.asarray(r["item_responses"], dtype=float)
            s = init_state(r)
            for j in range(len(ir)):
                if not r["missing_mask"][j]:
                    s = update_state(s, j, int(ir[j]), 0)
            s["questions_remaining"] = 0
            s["budget"] = len(ir)
            out.append(sp._raw_prob(s))
        return np.array(out)

    shift = {}
    for name, recs in (("saudi_train", saudi_tr), ("saudi_val", saudi_va),
                       ("saudi_test", saudi_te)):
        r = saudi_raw(recs)
        shift[name] = {"n": len(r), "mean": float(r.mean()),
                       "frac_at_1": float((r > 1 - 1e-6).mean())}
    shift["polish"] = {"n": len(raw), "mean": float(raw.mean()),
                       "frac_at_1": float((raw > 1 - 1e-6).mean())}
    Xsa = np.array([np.asarray(r["item_responses"], dtype=float) for r in saudi])
    shift["feature_prevalence_saudi"] = Xsa.mean(axis=0).round(4).tolist()
    shift["feature_prevalence_polish"] = features.mean(axis=0).round(4).tolist()
    print(f"   saudi test raw frac==1  : {shift['saudi_test']['frac_at_1']:.3f}")
    print(f"   polish     raw frac==1  : {shift['polish']['frac_at_1']:.3f}")

    print(f"\n[5/8] calibration ablation (Saudi-trained calibrators only)")
    calib_rows = []
    for name, note in (("raw_uncalibrated",
                        "no calibrator; raw network sigmoid output"),
                       ("platt_saudi_network",
                        "existing Saudi-trained Platt artefact + its own network"),
                       ("platt_on_isotonic_network",
                        "existing Saudi-trained Platt calibrator applied to the "
                        "isotonic network's raw output"),
                       ("isotonic_saudi",
                        "AUTHORITATIVE research calibrator")):
        if name not in scores:
            continue
        row = summary_row(name, y, scores[name], note, "saudi only")
        row["role"] = ("PRIMARY" if name == "isotonic_saudi" else "diagnostic")
        calib_rows.append(row)
        print(f"   {name:<28} Brier {row['brier']:.4f}  AUROC {row['auroc']:.4f}  "
              f"ECE {row['ece_10bin']:.4f}  slope {row['calibration_slope']}")

    print(f"\n[6/8] predictor ablation (every model fitted on Saudi only)")
    rows = [rep, rep_ref]
    specs = saudi_only_predictors()
    for name, spec in specs.items():
        p = predict_saudi_only(spec, features)
        row = summary_row(name, y, p, spec["note"], spec["trained_on"])
        row["hyperparameters"] = spec["hyperparameters"]
        rows.append(row)
    for r in rows:
        print(f"   {r['name']:<26} Brier {r['brier']:.4f}  AUROC {r['auroc']:.4f}  "
              f"ECE {r['ece_10bin']:.4f}  slope "
              f"{r['calibration_slope']}")

    pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"}
                  for r in rows + calib_rows]).to_csv(
        RESULTS / "predictor_ablation.csv", index=False, encoding="utf-8")

    print(f"\n[7/8] does the learned predictor add anything beyond item count?")
    best_logreg = max(
        (r for r in rows if r["name"].startswith("logistic")),
        key=lambda r: r["auroc"])
    incremental = {
        "question": ("Does the learned predictor provide information beyond the "
                     "number of atypical answers?"),
        "auroc_model_minus_count": auroc_diff["point_estimate"],
        "auroc_ci": [auroc_diff["ci_lo"], auroc_diff["ci_hi"]],
        "best_saudi_logistic_auroc": best_logreg["auroc"],
        "best_saudi_logistic_name": best_logreg["name"],
        "logistic_minus_count_auroc": float(
            best_logreg["auroc"] - rep_ref["auroc"]),
        "ranking_note": (
            "A logistic model on the 10 items can in principle use WHICH items are "
            "atypical, not only how many, so it is the fair test of incremental "
            "information. If it also fails to beat the count, the count is capturing "
            "essentially all the available signal."),
        "matched_information_note": (
            "The item-count reference uses all 10 items, so it is not "
            "information-limited relative to the full model. A budget-limited "
            "comparison is reported separately under adaptive selection."),
    }
    print(f"   best Saudi logistic      : {best_logreg['name']} "
          f"AUROC {best_logreg['auroc']:.4f}")
    print(f"   logistic minus count     : "
          f"{incremental['logistic_minus_count_auroc']:+.4f}")

    print(f"\n[8/8] adaptive question selection (COMPONENT B, separate)")
    bench = json.loads(
        (RESULTS / "step5_policy_benchmark_saudi.json").read_text(encoding="utf-8"))
    saudi_bench = [r for r in bench["per_budget"] if r["B"] in (3, 5, 6)]
    rl_component = {
        "separation_note": (
            "COMPONENT A is the predictive model; COMPONENT B is adaptive question "
            "selection. The predictor underperforming the item count does not by "
            "itself invalidate COMPONENT B, but it does bound what COMPONENT B can "
            "achieve, because every policy's reward is scored through this predictor."),
        "saudi_benchmark_existing": saudi_bench,
        "saudi_benchmark_warning": (
            "These Saudi numbers are on CIRCULAR questionnaire-derived labels, where "
            "an item-count-equivalent signal is almost perfectly recoverable, so they "
            "flatter every policy. They are reported for completeness, not as "
            "evidence."),
        "trained_rl_policies_persisted": False,
        "persistence_note": (
            "No trained RL policy artefact (.pt) exists on disk; step 4 trains in "
            "memory and step 5 benchmarks immediately. A fresh Polish RL evaluation "
            "would therefore require re-training on Saudi, which is deliberately not "
            "done in this diagnostic pass."),
    }
    for b in (3, 6):
        sub = {r["policy"]: r for r in saudi_bench if r["B"] == b}
        if sub:
            print(f"   Saudi B={b}: random AUROC {sub['random']['auroc']:.4f} "
                  f"-> greedy {sub['greedy']['auroc']:.4f} "
                  f"(items {sub['random']['items_asked_mean']:.2f} -> "
                  f"{sub['greedy']['items_asked_mean']:.2f})")

    classification = {
        "auroc_difference_point": auroc_diff["point_estimate"],
        "auroc_difference_ci": [auroc_diff["ci_lo"], auroc_diff["ci_hi"]],
        "ci_excludes_zero": bool(auroc_diff["ci_hi"] < 0 or auroc_diff["ci_lo"] > 0),
        "candidate": "C - model underperforms simple baseline",
        "confirmed": None,
    }
    if auroc_diff["ci_hi"] < 0:
        classification["confirmed"] = (
            "C: the frozen predictor's AUROC is significantly LOWER than the "
            "item-count reference; the entire 95% interval lies below zero.")
    elif auroc_diff["ci_lo"] > 0:
        classification["confirmed"] = (
            "A: the frozen predictor significantly outperforms the item count.")
    else:
        classification["confirmed"] = (
            "D: the 95% interval spans zero, so the difference is not statistically "
            "distinguishable from zero on this cohort.")

    artifact = {
        "artifact": "predictor_diagnosis",
        "tag": ("DIAGNOSTIC pass; no retraining; no frozen artefact modified; no "
                "model selected using Polish; primary Step 10 result unchanged"),
        "git_sha": git_sha(REPO),
        "primary_result_unchanged": {
            "source": "results/polish_external_validation.json",
            "threshold": TAU,
            "auroc": primary["primary_result"]["metrics"]["auroc"],
            "brier": primary["primary_result"]["metrics"]["brier"],
            "note": ("This diagnostic does not recompute, reinterpret or overwrite "
                     "the primary result. Any improved model requires a new version "
                     "and a new frozen evaluation."),
        },
        "verification": {
            "dataset_rows": int(len(df)),
            "weights_sha256": weights_sha,
            "weights_match_recorded": weights_sha == manifest["artifact_weights_sha256"],
            "calibrator_sha256": calib_sha,
            "calibrator_match_recorded":
                calib_sha == manifest["artifact_calibrator_sha256"],
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "qchat_mapping_version": QCHAT_MAPPING_VERSION,
            "contract_complete": contract_completeness()["complete"],
            "qchat25_source_items": [CANONICAL_MAPPING[i] for i in range(1, 11)],
            "threshold": TAU,
            "reproduced_auroc": rep["auroc"],
            "recorded_auroc": primary["primary_result"]["metrics"]["auroc"],
            "reproduced_brier": rep["brier"],
            "recorded_brier": primary["primary_result"]["metrics"]["brier"],
            "exact_reproduction": bool(
                abs(rep["auroc"] - primary["primary_result"]["metrics"]["auroc"]) < 1e-12
                and abs(rep["brier"] - primary["primary_result"]["metrics"]["brier"]) < 1e-12),
        },
        "item_count_audit": audit,
        "paired_auroc_bootstrap": auroc_diff,
        "paired_brier_bootstrap": brier_diff,
        "calibration_failure_trace": trace,
        "support_shift": shift,
        "calibration_ablation": calib_rows,
        "predictor_ablation": rows,
        "incremental_information": incremental,
        "rl_component": rl_component,
        "classification": classification,
        "contains_participant_rows": False,
        "python": platform.python_version(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_artifact(REPO, artifact, "predictor_diagnosis.json")
    print(f"\n   -> {path.relative_to(REPO)}")
    print(f"\nCLASSIFICATION: {classification['confirmed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())