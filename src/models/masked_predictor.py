"""Masked predictor — §12. MLP 128→64, sigmoid output, calibration."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from typing import List, Dict, Any, Callable
from src.env.state import encode_state, UNASKED, OBSERVED, MISSING

class MaskedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

class PlattWrap:
    """Module-level so instances are picklable (demo model cache)."""
    def __init__(self, m):
        self.m = m
    def predict(self, x):
        x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(x / (1 - x)).reshape(-1, 1)
        return self.m.predict_proba(z)[:, 1]


class MaskedPredictor:
    """Wrapper that handles state encoding, training with random masking, calibration.

    Predictor version 2 (2026-10-01) fixes three defects in version 1, all of
    which invalidated the offline artifacts:

    1. **One training state per record.** ``fit`` sampled a single random
       ``(k, subset)`` per record per call, so the 284-record Saudi train split
       yielded 284 distinct training states for a 41-dimensional input — and each
       was visited once. ``masks_per_record`` now defaults to 16.
    2. **The budget feature was out of distribution at inference.** Training set
       ``budget = n_items`` unconditionally, so the ``budget_norm`` input took
       values on the grid ``1.0, 0.9, ..., 0.0``. The environment runs episodes at
       ``budget = B`` (6 in the demo and in every script), which produces
       ``1.0, 0.833, 0.667, 0.5, 0.333, 0.167, 0.0`` — four of seven inference
       values were never seen in training. Training now samples the budget from
       ``eval_budgets`` and draws the number of revealed items *within* it, so the
       training distribution matches inference exactly.
    3. **The calibrator was fitted on fully observed states only.** The network
       is queried on partial states at inference, where its raw output lands
       mid-range, while the calibrator had only ever seen outputs on
       ``[0.001, 1.0]`` from fully revealed records — it was extrapolating. It is
       now fitted on states drawn from the same distribution as inference.
    """

    #: Bumped whenever training or calibration semantics change. Stamped into
    #: every artifact so pre- and post-fix numbers can never be silently mixed.
    VERSION = 2

    def __init__(self, n_items: int, hidden: List[int] = [128, 64], calibration: str = "isotonic",
                 device: str = "cpu", m_list=None, masks_per_record: int = 16,
                 eval_budgets: List[int] | None = None, seed: int | None = 0):
        self.n_items = n_items
        self.m_list = m_list
        self.calibration = calibration
        self.device = device
        self.seed = seed
        # input dim: 3*n + n (binary) +1  or 3*n + sum(m_j)+1
        if m_list is None:
            input_dim = 4 * n_items + 1
        else:
            input_dim = 3 * n_items + sum(m_list) + 1
        # Weight initialisation must be deterministic in its own right, not a
        # side effect of wherever the caller happened to leave the global torch
        # RNG. `MaskedMLP` builds `nn.Linear` layers, which draw from that global
        # stream — so constructing a predictor *before* the caller seeded torch
        # produced a different network on every run. `step3` and `step5` did
        # exactly that, which is why their artifacts drifted between runs
        # regardless of `--seed`.
        #
        # Seed only around the construction and then restore the previous global
        # state, so the predictor is reproducible without perturbing the caller's
        # stream.
        if seed is None:
            self.model = MaskedMLP(input_dim, hidden).to(device)
        else:
            prior_state = torch.get_rng_state()
            try:
                torch.manual_seed(seed)
                self.model = MaskedMLP(input_dim, hidden).to(device)
            finally:
                torch.set_rng_state(prior_state)
        self.calibrator = None
        self.hidden = hidden
        if masks_per_record < 1:
            raise ValueError("masks_per_record must be >= 1")
        self.masks_per_record = masks_per_record
        self.eval_budgets = list(eval_budgets) if eval_budgets else list(range(1, n_items + 1))
        if not self.eval_budgets:
            raise ValueError("eval_budgets must be non-empty")
        if max(self.eval_budgets) > n_items:
            raise ValueError(f"eval_budgets max {max(self.eval_budgets)} > n_items {n_items}")
        self.version = self.VERSION

    def _sample_state(self, rec, rng):
        """Draw one (budget, depth, revealed subset) triple from the inference grid.

        At inference the environment presents a state after ``t`` questions at
        budget ``B``: ``t`` items OBSERVED, ``questions_remaining = B - t`` and
        ``budget = B``, so ``budget_norm = (B - t) / B``. Sampling the same way
        makes the training distribution match inference, which is what fixes
        both the budget feature and the calibrator's extrapolation.
        """
        from src.env.state import init_state, update_state

        budget = int(rng.choice(self.eval_budgets))
        available = [j for j in range(self.n_items) if not rec["missing_mask"][j]]
        max_depth = min(budget, len(available))
        depth = int(rng.integers(0, max_depth + 1))
        chosen = rng.choice(available, size=depth, replace=False) if depth > 0 else []

        state = init_state(rec)
        state["budget"] = budget
        state["questions_remaining"] = budget
        for j in chosen:
            state = update_state(state, int(j), int(rec["item_responses"][int(j)]),
                                 state["questions_remaining"] - 1)
            state["budget"] = budget
        state["questions_remaining"] = budget - depth
        state["budget"] = budget
        return state

    def _encode_batch(self, states, budgets=None):
        # states: list of state dicts
        vecs = []
        for s in states:
            b = s.get("budget", self.n_items)
            qr = s.get("questions_remaining", b)
            vecs.append(encode_state(s, m_list=self.m_list, questions_remaining=qr, budget=b))
        return torch.tensor(np.stack(vecs), dtype=torch.float32, device=self.device)

    def predict_state(self, state: Dict[str, Any]) -> float:
        self.model.eval()
        with torch.no_grad():
            v = encode_state(state, m_list=self.m_list, questions_remaining=state.get("questions_remaining", 0), budget=state.get("budget", self.n_items))
            x = torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
            p = self.model(x).item()
        if self.calibrator is not None:
            p = float(self.calibrator.predict([p])[0])
            p = float(np.clip(p, 0.0, 1.0))
        return p

    def __call__(self, state: Dict[str, Any]) -> float:
        return self.predict_state(state)

    def fit(self, records: List[Dict[str, Any]], epochs: int = 20, lr: float = 1e-3, batch_size: int = 32, seed: int = 0):
        """Train on records with random masking across the evaluation budget grid.

        Draws ``masks_per_record`` independent masked views of every record, so
        the effective training set is ``len(records) * masks_per_record`` rather
        than ``len(records)``. See the class docstring for why version 1's single
        view per record was not enough.
        """
        import random
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        states = []
        labels = []
        for rec in records:
            for _ in range(self.masks_per_record):
                states.append(self._sample_state(rec, rng))
                labels.append(rec["label"])
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        self.model.train()
        n = len(states)
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                batch_states = [states[j] for j in idx.tolist()]
                xb = self._encode_batch(batch_states)
                yb = labels_t[idx]
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
        return self

    def fit_calibrator(self, records: List[Dict[str, Any]], method: str | None = None, seed: int = 0):
        """Fit isotonic or Platt calibration on states drawn from the inference grid.

        Version 1 fitted on fully revealed records only, so the calibrator never
        saw the mid-range raw outputs the network produces on partial states and
        had to extrapolate at inference. States are now drawn exactly as in
        :meth:`fit`.

        ``masks_per_record`` states are drawn per record, not one. With 95
        validation records a single view gives only 95 calibration points, which
        is too thin to constrain a monotone map across the range the partial
        states occupy — measured as a calibration step that made Brier *worse*
        (0.0982 -> 0.1254) before this was fixed.
        """
        if method is None:
            method = self.calibration
        from src.env.state import encode_state
        rng = np.random.default_rng(seed)
        probs = []
        labels = []
        for rec in records:
            for _ in range(self.masks_per_record):
                s = self._sample_state(rec, rng)
                with torch.no_grad():
                    v = encode_state(s, m_list=self.m_list,
                                     questions_remaining=s["questions_remaining"],
                                     budget=s["budget"])
                    x = torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                    probs.append(self.model(x).item())
                labels.append(rec["label"])
        probs = np.array(probs)
        labels = np.array(labels)
        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs, labels)
            self.calibrator = ir
        elif method == "platt":
            # Platt scaling on the LOGIT of the network output (Platt 1999):
            # z = log(p / (1 - p)); fit 1-D logistic regression on z.
            # Fitting on the raw probability compresses the decision boundary
            # against p∈{0,1} and yields near-saturated posteriors.
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            logits = np.log(probs / (1 - probs)).reshape(-1, 1)
            lr = LogisticRegression()
            lr.fit(logits, labels)
            self.calibrator = PlattWrap(lr)
        elif method == "none":
            self.calibrator = None
        else:
            raise ValueError(f"Unknown calibration {method}")
        self.calibration_fit_points = len(probs)
        return self

    def _raw_prob(self, state) -> float:
        """Uncalibrated network output for a state (diagnostics and tests)."""
        from src.env.state import encode_state
        self.model.eval()
        with torch.no_grad():
            v = encode_state(state, m_list=self.m_list,
                             questions_remaining=state.get("questions_remaining", 0),
                             budget=state.get("budget", self.n_items))
            x = torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.model(x).item())

    def calibration_report(self, records: List[Dict[str, Any]], seed: int = 0) -> Dict[str, float]:
        """Brier and ECE before/after calibration, on the inference distribution.

        Version 1 scored this on fully observed states, so the reported ECE did
        not describe the states the predictor is actually asked about. The same
        (budget, depth, subset) sampler as :meth:`fit` is used here, with the RNG
        offset so the uncalibrated and calibrated passes see identical states.
        """
        from sklearn.metrics import brier_score_loss
        from src.env.state import init_state, update_state
        import math

        def sample(seed_offset):
            rng = np.random.default_rng(seed + seed_offset)
            out = []
            for rec in records:
                out.append(self._sample_state(rec, rng))
            return out

        states = sample(0)
        cal_save = self.calibrator
        self.calibrator = None
        probs_uncal = np.array([self._raw_prob(s) for s in states])
        self.calibrator = cal_save
        probs_cal = np.array([float(self(s)) for s in states])
        y = np.array([r["label"] for r in records])

        def ece(probs, y, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            e = 0
            for i in range(n_bins):
                mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
                if mask.sum() == 0:
                    continue
                acc = y[mask].mean()
                conf = probs[mask].mean()
                e += abs(acc - conf) * mask.sum() / len(y)
            return float(e)

        return {"brier_before": float(brier_score_loss(y, np.clip(probs_uncal, 0, 1))),
                "brier_after": float(brier_score_loss(y, np.clip(probs_cal, 0, 1))),
                "ece_before": ece(probs_uncal, y),
                "ece_after": ece(probs_cal, y),
                "predictor_version": self.version,
                "n_states_scored": len(states)}
