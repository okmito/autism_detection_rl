from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("NEUROBRIDGE_DB", str(ROOT / "data" / "neurobridge.db")))
RESULTS_DIR = ROOT / "results"
STATIC_DIR = APP_DIR / "static"
SECRET_KEY = os.environ.get("NEUROBRIDGE_SECRET", "dev-research-prototype-change-me")
BUDGET_DEFAULT = 6
TAU_DEFAULT = 0.5
N_ITEMS = 10
SEED = 0
# Demo cache (reuses existing predictor cache naming semantics)
DEMO_CALIBRATION = "platt"