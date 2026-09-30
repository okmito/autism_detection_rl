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
    """Wrapper that handles state encoding, training with random masking, calibration."""
    def __init__(self, n_items: int, hidden: List[int] = [128, 64], calibration: str = "isotonic", device: str = "cpu", m_list=None):
        self.n_items = n_items
        self.m_list = m_list
        self.calibration = calibration
        self.device = device
        # input dim: 3*n + n (binary) +1  or 3*n + sum(m_j)+1
        if m_list is None:
            input_dim = 4 * n_items + 1
        else:
            input_dim = 3 * n_items + sum(m_list) + 1
        self.model = MaskedMLP(input_dim, hidden).to(device)
        self.calibrator = None
        self.hidden = hidden

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
        """Train on records with random masking across budgets 0..n."""
        import random
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        from src.env.state import init_state, update_state
        # Build training states by sampling random subsets
        states = []
        labels = []
        for rec in records:
            # sample random k in 0..n and random subset
            k = int(rng.integers(0, self.n_items + 1))
            # only include items that are not MISSING
            available = [j for j in range(self.n_items) if not rec["missing_mask"][j]]
            if k > len(available):
                k = len(available)
            chosen = rng.choice(available, size=k, replace=False) if k > 0 else []
            s = init_state(rec)
            s["questions_remaining"] = self.n_items - k
            s["budget"] = self.n_items
            for j in chosen:
                s = update_state(s, int(j), int(rec["item_responses"][int(j)]), s["questions_remaining"]-1 if s["questions_remaining"]>0 else 0)
                # keep budget
                s["budget"] = self.n_items
                s["questions_remaining"] = s.get("questions_remaining", 0)
            # fix up questions_remaining
            s["questions_remaining"] = self.n_items - len(chosen)
            s["budget"] = self.n_items
            states.append(s)
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

    def fit_calibrator(self, records: List[Dict[str, Any]], method: str | None = None):
        """Fit isotonic or Platt calibration on provided records (should be val set)."""
        if method is None:
            method = self.calibration
        # Use fully observed state as proxy or random? Use fully observed for calibration
        from src.env.state import init_state, update_state
        probs = []
        labels = []
        for rec in records:
            s = init_state(rec)
            # reveal all non-missing
            for j in range(self.n_items):
                if not rec["missing_mask"][j]:
                    s = update_state(s, j, int(rec["item_responses"][j]), 0)
                    s["budget"] = self.n_items
                    s["questions_remaining"] = 0
            s["budget"] = self.n_items
            s["questions_remaining"] = 0
            # get uncalibrated prob
            self.model.eval()
            with torch.no_grad():
                v = encode_state(s, m_list=self.m_list, questions_remaining=0, budget=self.n_items)
                x = torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                p = self.model(x).item()
            probs.append(p)
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
        return self

    def calibration_report(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        from sklearn.metrics import brier_score_loss
        from src.env.state import init_state, update_state
        import math
        probs_uncal = []
        # temporarily disable calibrator to get uncal
        cal_save = self.calibrator
        self.calibrator = None
        for rec in records:
            s = init_state(rec)
            for j in range(self.n_items):
                if not rec["missing_mask"][j]:
                    s = update_state(s, j, int(rec["item_responses"][j]), 0)
                    s["budget"] = self.n_items; s["questions_remaining"] = 0
            s["budget"]=self.n_items; s["questions_remaining"]=0
            probs_uncal.append(self.predict_state(s))
        self.calibrator = cal_save
        probs_cal = []
        for rec in records:
            s = init_state(rec)
            for j in range(self.n_items):
                if not rec["missing_mask"][j]:
                    s = update_state(s, j, int(rec["item_responses"][j]), 0)
                    s["budget"]=self.n_items; s["questions_remaining"]=0
            s["budget"]=self.n_items; s["questions_remaining"]=0
            probs_cal.append(self.predict_state(s))
        y = np.array([r["label"] for r in records])
        brier_before = brier_score_loss(y, np.clip(probs_uncal, 0, 1))
        brier_after = brier_score_loss(y, np.clip(probs_cal, 0, 1))
        # ECE simple 10 bins
        def ece(probs, y, n_bins=10):
            bins = np.linspace(0, 1, n_bins+1)
            e=0
            for i in range(n_bins):
                mask = (probs >= bins[i]) & (probs < bins[i+1] if i < n_bins-1 else probs <= bins[i+1])
                if mask.sum()==0: continue
                acc = y[mask].mean()
                conf = probs[mask].mean()
                e += abs(acc-conf)*mask.sum()/len(y)
            return float(e)
        return {"brier_before": float(brier_before), "brier_after": float(brier_after), "ece_before": ece(np.array(probs_uncal), y), "ece_after": ece(np.array(probs_cal), y)}
