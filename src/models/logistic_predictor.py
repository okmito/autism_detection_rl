"""Predictor v3 — L2 logistic regression on the ten Q-CHAT-10 binary items.

Why this replaces the neural predictor as the development scorer
--------------------------------------------------------------
The primary Polish external validation (2026-10-02) produced a decisive result:

    frozen neural predictor (v2, isotonic)   AUROC 0.8959
    item-count reference                     AUROC 0.9369
    paired difference                        -0.0410, 95% CI [-0.0636, -0.0186]

The whole confidence interval lies below zero, so the neural predictor was
**significantly worse** than simply counting atypical answers. Two separable causes
were identified:

1. *Network saturation.* The Saudi target is a deterministic sum-threshold, so the
   training loss is minimised by an extremely confident mapping; ~41% of Saudi
   terminal states score exactly 1.0 before calibration.
2. *Tie-creating calibration.* The isotonic calibrator maps 103 distinct raw scores
   onto 16 levels, destroying ranking and costing 0.033 AUROC on its own.

In a Saudi-only ablation, plain L2 logistic regression (C = 0.1) outperformed the
frozen network on **every** metric: Brier 0.1097 vs 0.1398, ECE 0.0775 vs 0.1370,
AUROC 0.9349 vs 0.8959.

This module therefore becomes the **development scorer**. It is not claimed to be
clinically superior — nothing here is a clinical claim. It is simpler, more
interpretable, better calibrated, and provides smoother probabilities for the RL
reward.

Scoring partially observed states
---------------------------------
The environment presents *partial* states, so a plain ``predict(x)`` is not enough.
Unobserved items are handled by **exact Bayesian marginalisation**: the ten binary
items admit only 2^10 = 1024 configurations, so the exact predictive distribution

    P(y=1 | x_observed) = sum_c sigmoid(beta0 + c . beta) * P(c)

can be precomputed once and then summed over the configurations consistent with the
observed entries. The per-item prior P(x_j = 1) is the **Saudi training**
prevalence, so no external-cohort information enters the scorer.

This is exact under an independence assumption across unobserved items, which is
stated rather than hidden. It is also strictly better than imputing unobserved items
as typical, which biases every partial state toward the majority class.

MANDATORY CAVEAT — research prototype, not a diagnostic device.
"""
from __future__ import annotations

import hashlib
import itertools
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PREDICTOR_VERSION = "logistic-saudi-v3"
N_ITEMS = 10

#: Hyperparameters are FIXED here and were selected in a Saudi-only ablation
#: (scripts/step11_predictor_diagnosis.py). They are never tuned on Polish.
HYPERPARAMETERS: Dict[str, Any] = {
    "model": "LogisticRegression",
    "penalty": "l2",
    "C": 0.1,
    "solver": "lbfgs",
    "max_iter": 5000,
    "selection_basis": (
        "Saudi-only ablation; C=0.1 gave the best Brier and ECE of the "
        "pre-specified grid {0.1, 1.0} x {l2, l1}"),
}

CALIBRATION_METHOD = "platt_sigmoid_on_raw_probability_saudi_val"


class PlattOnProbability:
    """Textbook Platt scaling: ``P(y=1 | p) = sigmoid(a * p + b)``.

    The obvious alternative — a logistic fit on ``logit(p)`` — is UNUSABLE here
    and was measured to be so. The Saudi target is a deterministic sum-threshold,
    so the unfitted logistic model is almost perfectly separable: its raw
    probabilities saturate at 0 and 1, its logits reach +-40, and an unpenalised
    logit-domain fit diverges (measured coefficient 60.9, intercept 8.97), which
    collapsed every external prediction to ~0.001.

    Regressing on the bounded raw probability keeps the fit well-posed while
    remaining the standard Platt functional form. This is the same parameterisation
    used by sklearn's ``CalibratedClassifierCV(method="sigmoid")`` and it is
    monotone, so discrimination is unaffected by construction.
    """

    def __init__(self, model: Any):
        self.model = model

    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=float).reshape(-1, 1), 0.0, 1.0)
        return self.model.predict_proba(p)[:, 1]

    predict_proba = predict

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_


class LogisticPredictor:
    """Frozen scorer: L2 logistic regression + Platt calibration on Saudi val."""

    def __init__(self, model: Any, calibrator: Any,
                 prior: np.ndarray, feature_order: Sequence[str],
                 metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.calibrator = calibrator
        self.prior = np.asarray(prior, dtype=float)
        self.feature_order = list(feature_order)
        self.metadata = dict(metadata or {})
        self._table: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._configs: Optional[np.ndarray] = None
        self._build_table()

    # ---------------------------------------------------------------- setup
    def _build_table(self) -> None:
        """Precompute P(y=1 | c) for all 2^10 item configurations.

        Row ``k`` holds item j as bit j, i.e. ``configs[k, j] == (k >> j) & 1``.
        That explicit convention matters: `itertools.product` varies the LAST
        element fastest, so relying on its ordering made ``predict_features`` look
        up effectively arbitrary configurations and report ~0.001 everywhere.
        """
        k = np.arange(1 << N_ITEMS)
        configs = np.array([[(int(row) >> j) & 1 for j in range(N_ITEMS)]
                            for row in k], dtype=float)
        logit = self.model.decision_function(configs)
        base = 1.0 / (1.0 + np.exp(-logit))
        if self.calibrator is not None:
            # Platt operates on the bounded raw probability, NOT on its logit; see
            # PlattOnProbability for why the logit-domain fit diverges here.
            base = np.clip(np.asarray(self.calibrator.predict(base), dtype=float),
                           0.0, 1.0)
        log_prior = np.zeros(configs.shape[0], dtype=float)
        for j in range(N_ITEMS):
            q = float(np.clip(self.prior[j], 1e-6, 1 - 1e-6))
            log_prior += np.where(configs[:, j] == 1, np.log(q), np.log(1 - q))
        weights = np.exp(log_prior - log_prior.max())
        self._configs = configs
        self._table = base * weights
        self._weights = weights

    def _marginal(self, consistent: np.ndarray) -> float:
        """Prior-weighted mean of the calibrated probability over consistent configs.

        The normaliser is the prior mass of the CONSISTENT subset, not of all
        2^10 configurations. Normalising globally returns ``base[k] * P(k)`` for a
        fully observed state, which silently reports ~0.001 everywhere.
        """
        mass = float(self._weights[consistent].sum())
        if mass <= 0.0:
            return float(1.0 / (1.0 + np.exp(-float(self.model.intercept_[0]))))
        return float(self._table[consistent].sum() / mass)

    # ------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, train_records: Sequence[Dict[str, Any]],
            val_records: Sequence[Dict[str, Any]],
            feature_order: Sequence[str],
            seed: int = 0) -> "LogisticPredictor":
        from sklearn.linear_model import LogisticRegression

        def mat(recs):
            X = np.array([np.asarray(r["item_responses"], dtype=float)
                          for r in recs], dtype=float)
            y = np.array([int(r["label"]) for r in recs], dtype=int)
            return X, y

        Xtr, ytr = mat(train_records)
        Xva, yva = mat(val_records)

        model = LogisticRegression(
            penalty=HYPERPARAMETERS["penalty"],
            C=HYPERPARAMETERS["C"],
            solver=HYPERPARAMETERS["solver"],
            max_iter=HYPERPARAMETERS["max_iter"],
        ).fit(Xtr, ytr)

        # Platt scaling fitted on the Saudi VALIDATION split only.
        from sklearn.linear_model import LogisticRegression as _LR
        raw = model.predict_proba(Xva)[:, 1]
        calibrator = PlattOnProbability(
            _LR(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000)
            .fit(np.clip(raw, 0.0, 1.0).reshape(-1, 1), yva))

        prior = Xtr.mean(axis=0)
        meta = {
            "predictor_version": PREDICTOR_VERSION,
            "hyperparameters": dict(HYPERPARAMETERS),
            "calibration_method": CALIBRATION_METHOD,
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "train_prevalence": float(ytr.mean()),
            "val_prevalence": float(yva.mean()),
            "seed": int(seed),
            "feature_order": list(feature_order),
            "trained_on": "saudi train split",
            "calibrated_on": "saudi validation split",
            "external_cohort_used": False,
        }
        return cls(model, calibrator, prior, feature_order, meta)

    # ---------------------------------------------------------- prediction
    def predict_features(self, features: Sequence[float]) -> float:
        """Exact marginal predictive probability from a COMPLETE item vector."""
        features = np.asarray(features, dtype=float)
        if features.shape != (N_ITEMS,):
            raise ValueError(f"expected {N_ITEMS} features, got {features.shape}")
        consistent = np.ones(self._configs.shape[0], dtype=bool)
        f = features.astype(int)
        for j in range(N_ITEMS):
            consistent &= (self._configs[:, j] == f[j])
        return self._marginal(consistent)

    def predict_state(self, state: Dict[str, Any]) -> float:
        """Probability from a partially observed state (the environment's call)."""
        from src.env.state import MISSING, OBSERVED

        mask = np.asarray(state["mask"], dtype=int)
        value = np.asarray(state["value"], dtype=int)
        consistent = np.ones(self._configs.shape[0], dtype=bool)
        any_observed = False
        for j in range(mask.size):
            if mask[j] == OBSERVED:
                consistent &= (self._configs[:, j] == value[j])
                any_observed = True
            elif mask[j] == MISSING:
                # a missing item is uninformative; marginalise it out
                pass
        if not any_observed:
            # nothing observed: fall back to the calibrated marginal prevalence
            raw = float(1.0 / (1.0 + np.exp(-float(self.model.intercept_[0]))))
            if self.calibrator is None:
                return float(np.clip(raw, 0.0, 1.0))
            return float(np.clip(self.calibrator.predict(np.array([raw]))[0],
                                 0.0, 1.0))
        return self._marginal(consistent)

    __call__ = predict_state

    def probability_for_partial(self, observed: Dict[int, int]) -> float:
        """Convenience wrapper used by policy information-gain calculations."""
        from src.env.state import OBSERVED
        mask = np.zeros(N_ITEMS, dtype=int)
        value = np.full(N_ITEMS, -1, dtype=int)
        for j, v in observed.items():
            mask[j] = OBSERVED
            value[j] = int(v)
        return self.predict_state({"mask": mask, "value": value, "n": N_ITEMS})

    # ------------------------------------------------------- explainability
    def explain(self) -> Dict[str, Any]:
        """log-odds = beta0 + sum_i beta_i * x_i, per item."""
        coef = np.asarray(self.model.coef_).ravel()
        rows = []
        for j, name in enumerate(self.feature_order):
            rows.append({
                "item": name,
                "feature_index": j + 1,
                "coefficient_log_odds": float(coef[j]),
                "direction": ("raises risk when atypical"
                              if coef[j] > 0 else
                              "lowers risk when atypical"),
                "odds_multiplier_when_atypical": float(np.exp(coef[j])),
                "probability_at_other_items_typical": float(
                    1.0 / (1.0 + np.exp(-(self.model.intercept_[0])))),
                "probability_at_other_items_typical_this_item_atypical": float(
                    1.0 / (1.0 + np.exp(-(self.model.intercept_[0] + coef[j])))),
            })
        base = float(1.0 / (1.0 + np.exp(-self.model.intercept_[0])))
        cal = None
        if self.calibrator is not None:
            cal = {
                "form": "P(y=1|p) = sigmoid(a * p + b), fitted on raw probability",
                "coefficient_a": float(np.ravel(self.calibrator.coef_)[0]),
                "intercept_b": float(np.ravel(self.calibrator.intercept_)[0]),
            }
        return {
            "form": "log-odds = beta0 + sum_i beta_i * x_i",
            "intercept": float(self.model.intercept_[0]),
            "intercept_probability_before_calibration": base,
            "platt_calibration": cal,
            "items": rows,
            "explanation_note": (
                "A positive coefficient means answering that item atypically "
                "increases the log-odds of the predicted class. Coefficients are "
                "reported exactly as fitted; no post-hoc interpretation is added."),
        }

    # --------------------------------------------------------- persistence
    def save(self, path_pt: Path, path_pkl: Path,
             extra: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        payload = {
            "model": self.model,
            "calibrator": self.calibrator,
            "prior": self.prior,
            "feature_order": self.feature_order,
            "metadata": self.metadata,
        }
        path_pkl = Path(path_pkl)
        path_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(path_pkl, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        digests = {
            "artifact": str(path_pkl),
            "artifact_sha256": sha256_of(path_pkl),
        }
        if extra:
            digests.update(extra)
        return digests

    @classmethod
    def load(cls, path_pkl: Path) -> "LogisticPredictor":
        with open(path_pkl, "rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["model"], payload["calibrator"], payload["prior"],
                   payload["feature_order"], payload.get("metadata"))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sha256_of(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None