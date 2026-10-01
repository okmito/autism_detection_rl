"""Step 2 — Train MaskedMLP predictor on Saudi + UCI Child (Q-CHAT-10 binary)
and run ExactDP tractability sweep at B ∈ {3, 4, 5, 6} on multiple sample sizes.

§12, §14, §21.

Outputs (all under results/):
  - results/predictor_saudi_metrics.json
  - results/predictor_uci_metrics.json
  - results/dp_tractability_sweep.json
  - results/dp_tractability_sweep.csv

Polish cohort (252) is excluded — isolation enforced per AUDIT_REPORT.md
until V-4 (MDE family) and V-7 (denominator freeze) are supervisor-approved.

Run from repo root:
  /tmp/aar/bin/python scripts/step2_train_and_sweep.py
  # or with a venv that has requirements.txt installed:
  python scripts/step2_train_and_sweep.py

Datasets:
  - If real Saudi / UCI CSVs are present at the paths in src/data/ingest.py
    they are used directly (label_source=questionnaire, real circularity).
  - Otherwise the synthetic fallback from src/data/ingest.py is used so the
    pipeline can still be smoke-tested. Outputs clearly state which mode ran.
"""
from __future__ import annotations
import json
import os
import sys
import time
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# A Windows console defaults to cp1252, which cannot encode the lambda / arrow /
# section-sign characters this script prints. Without this the script aborts
# mid-run with UnicodeEncodeError instead of completing and writing its artifacts.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
os.chdir(REPO)

import numpy as np
import torch

from src.data.ingest import (
    load_dataset, SAUDI_CSV, UCI_CHILD_ARFF, POLISH_CSV,
)
from src.models.masked_predictor import MaskedPredictor
from src.solvers.exact_custom import ExactDP
from src.env.state import reachable_state_count
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from src.data.splits import stratified_split, split_fingerprint, SCHEME

RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def _split(records, val_frac: float = 0.25, seed: int = 0):
    """Canonical split — delegates to ``src.data.splits.stratified_split``.

    ``val_frac`` is accepted for call-site compatibility and ignored: the
    canonical scheme is fixed at 4-fold by spec §17/§19 so that every artifact
    lands on the same partition. The local copy this replaces passed the relative
    indices returned by the second ``skf.split(train_idx, ...)`` call straight
    into ``records[i]``, so the "validation" set overlapped both train and test.
    """
    return stratified_split(records, seed=seed)


def _terminal_probs(predictor: MaskedPredictor, records) -> tuple[np.ndarray, np.ndarray]:
    """Predict p on the terminal fully-observed state for each record."""
    from src.env.state import init_state, update_state
    p_list, y_list = [], []
    for rec in records:
        s = init_state(rec)
        n = s["n"]
        for j in range(n):
            if not rec["missing_mask"][j]:
                s = update_state(s, j, int(rec["item_responses"][j]), 0)
        s["questions_remaining"] = 0
        s["budget"] = n
        p_list.append(predictor(s))
        y_list.append(rec["label"])
    return np.array(p_list), np.array(y_list)


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = probs[mask].mean()
        e += abs(acc - conf) * mask.sum() / len(y)
    return float(e)


def _train_one(name: str, records, hidden=(128, 64), epochs=20, lr=1e-3,
               batch_size=32, seed: int = 0) -> dict:
    """Train a MaskedMLP with fold-local calibration; return metrics + serializable model."""
    real_data = name == "saudi" and SAUDI_CSV.exists() or (
        name == "uci_child" and UCI_CHILD_ARFF.exists()
    )
    source_tag = "real" if (
        (name == "saudi" and SAUDI_CSV.exists())
        or (name == "uci_child" and UCI_CHILD_ARFF.exists())
    ) else "synthetic"

    train, val, test = _split(records, seed=seed)

    pred = MaskedPredictor(
        n_items=10, hidden=list(hidden), calibration="isotonic", seed=seed
    )
    pred.fit(train, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    pred.fit_calibrator(val, method="isotonic")

    # Test metrics
    p_test, y_test = _terminal_probs(pred, test)
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y_test, p_test))
    ece = _ece(p_test, y_test)
    try:
        auroc = float(roc_auc_score(y_test, p_test))
    except ValueError:
        auroc = float("nan")
    try:
        ll = float(log_loss(y_test, p_test))
    except ValueError:
        ll = float("nan")

    # Val metrics (calibration check)
    p_val, y_val = _terminal_probs(pred, val)
    brier_val = float(brier_score_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)))

    return {
        "dataset": name,
        "source": source_tag,
        "n_total": len(records),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "label_source": records[0]["label_source"],
        "hidden": list(hidden),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "test_brier": brier,
        "test_ece": ece,
        "test_auroc": auroc,
        "test_logloss": ll,
        "val_brier": brier_val,
        "lambda": 0.0,
        "calibration": "isotonic",
        "predictor_version": MaskedPredictor.VERSION,
        "masks_per_record": pred.masks_per_record,
        "split_scheme": SCHEME,
        "split_fingerprint": split_fingerprint(train, val, test),
        "config": str(REPO / "configs" / "config.yaml"),
        "git_sha": _git_sha(),
    }


# ---------------------------------------------------------------------------
# DP tractability sweep
# ---------------------------------------------------------------------------

def _sweep_one(records, N: int, B: int, lambda_cost: float, seed: int = 0):
    """Run ExactDP on first N records of `records` at budget B and λ."""
    subset = records[:N]
    dp = ExactDP(
        records=subset,
        n_items=10,
        budget=B,
        b_min=0,
        lambda_cost=lambda_cost,
        cost_mode="uniform",
    )
    t0 = time.time()
    out = dp.solve()
    return {
        "N": N,
        "B": B,
        "lambda_cost": lambda_cost,
        "n_states": int(out.get("n_states", 0)),
        "n_evals": int(out.get("n_evals", 0)),
        "time_sec": float(out.get("time_sec", time.time() - t0)),
        "status": out.get("status", "unknown"),
        "V_star": float(out.get("V_star", float("nan"))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print(f"Step 2 — train predictor + DP tractability sweep")
    print(f"  repo:  {REPO}")
    print(f"  git:   {_git_sha()}")
    print(f"  py:    {platform.python_version()}")
    print(f"  torch: {torch.__version__}")
    print("=" * 70)

    # ----- 1. Predictor training (Saudi + UCI Child) -----
    predictor_results = {}
    for name in ["saudi", "uci_child"]:
        try:
            recs = load_dataset(name, synthetic=False)
        except FileNotFoundError as e:
            print(f"[{name}] real data not present → falling back to synthetic: {e}")
            recs = load_dataset(name, synthetic=True)
        print(f"[{name}] loaded {len(recs)} records ({recs[0]['label_source']})")
        res = _train_one(name, recs)
        predictor_results[name] = res
        out_path = RESULTS / f"predictor_{name}_metrics.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(
            f"[{name}] test_brier={res['test_brier']:.4f} "
            f"ece={res['test_ece']:.4f} auroc={res['test_auroc']:.4f} "
            f"→ {out_path.name}"
        )

    # ----- 2. DP tractability sweep (Saudi, multiple N × B) -----
    try:
        saudi = load_dataset("saudi", synthetic=False)
        saudi_source = "real" if SAUDI_CSV.exists() else "synthetic"
    except FileNotFoundError as e:
        print(f"[saudi] real data not present → falling back to synthetic: {e}")
        saudi = load_dataset("saudi", synthetic=True)
        saudi_source = "synthetic"

    # We only sweep Saudi to respect the spec's tractability bound (50M / 24h / 32GB)
    # and the §16.1 circularity gate. Polish is NOT used here per isolation.
    ns = [10, 25, 50, 100, 250, 506]
    budgets = [3, 4, 5, 6]
    lambda_grid = [0.0, 0.01]  # V-6 PENDING; placeholder values only — see config.yaml
    sweep = []
    for N in ns:
        for B in budgets:
            for lam in lambda_grid:
                rec = _sweep_one(saudi, N=min(N, len(saudi)), B=B, lambda_cost=lam)
                rec["dataset"] = "saudi"
                rec["source"] = saudi_source
                rec["n_items"] = 10
                rec["config"] = str(REPO / "configs" / "config.yaml")
                rec["git_sha"] = _git_sha()
                sweep.append(rec)
                print(
                    f"[saudi N={rec['N']:>3} B={B} λ={lam:.2f}] "
                    f"n_states={rec['n_states']:>7,} n_evals={rec['n_evals']:>7,} "
                    f"time={rec['time_sec']:.3f}s status={rec['status']}"
                )

    sweep_path = RESULTS / "dp_tractability_sweep.json"
    sweep_path.write_text(json.dumps(sweep, indent=2))
    print(f"→ {sweep_path.name} (n_runs={len(sweep)})")

    # CSV summary
    csv_path = RESULTS / "dp_tractability_sweep.csv"
    with csv_path.open("w") as f:
        f.write("dataset,source,N,B,lambda_cost,n_states,n_evals,time_sec,status,V_star\n")
        for r in sweep:
            f.write(
                f"{r['dataset']},{r['source']},{r['N']},{r['B']},"
                f"{r['lambda_cost']},{r['n_states']},{r['n_evals']},"
                f"{r['time_sec']:.4f},{r['status']},{r['V_star']:.6f}\n"
            )
    print(f"→ {csv_path.name}")

    # Theoretical reference
    print()
    print("Theoretical reference (complete-record, n=10, m=2):")
    for B in budgets:
        print(f"  B={B}: reachable_state_count(10,{B},2) = {reachable_state_count(10, B, 2)}")
    print()
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
