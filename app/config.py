from __future__ import annotations
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("NEUROBRIDGE_DB", str(ROOT / "data" / "neurobridge.db")))
RESULTS_DIR = ROOT / "results"
STATIC_DIR = APP_DIR / "static"
# Not used for anything yet (the app sets no cookies or Flask sessions). It must never be a guessable
# constant, so without NEUROBRIDGE_SECRET each process gets a fresh random key.
SECRET_KEY = os.environ.get("NEUROBRIDGE_SECRET") or secrets.token_hex(32)
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def _session_ttl_hours(raw: str | None) -> float:
    """Hours an unfinished screening session stays resumable. Invalid values are an error, not a silent default."""
    if raw is None or not raw.strip():
        return 24.0
    try:
        hours = float(raw)
    except ValueError:
        raise ValueError("NEUROBRIDGE_SESSION_TTL_HOURS must be a number of hours") from None
    if not 0.0 < hours <= 24.0 * 365:
        raise ValueError("NEUROBRIDGE_SESSION_TTL_HOURS must be greater than 0 and at most 8760")
    return hours


SESSION_TTL_HOURS = _session_ttl_hours(os.environ.get("NEUROBRIDGE_SESSION_TTL_HOURS"))


def load_defaults(path: Path = CONFIG_PATH) -> dict:
    """Execution defaults the app reads from ``configs/config.yaml`` (audit finding T4).

    Only the keys the app actually uses are read: the question budget, the referral threshold and the default
    policy. The file is the single source for them; the literals below are the fallback when it is absent
    (a partial checkout) and tests assert both agree. A key that is *present* but invalid is an error, not a
    silent fallback, because a mistyped threshold would change every screening result.
    """
    import yaml

    defaults = {"budget": 6, "tau": 0.5, "policy": "greedy"}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return defaults
    out = dict(defaults)
    budget = (data.get("env") or {}).get("question_budget")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 10:
            raise ValueError(f"{path.name}: env.question_budget must be an integer in 1..10, got {budget!r}")
        out["budget"] = budget
    tau = (data.get("eval") or {}).get("operating_threshold")
    if tau is not None:
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 < float(tau) < 1.0:
            raise ValueError(f"{path.name}: eval.operating_threshold must be a number in (0, 1), got {tau!r}")
        out["tau"] = float(tau)
    policy = (data.get("policy") or {}).get("type")
    if policy is not None:
        if policy not in ("greedy", "random", "beta_greedy", "rl_adaptive"):
            # The research config may name policies the app does not serve (dqn, ppo, ...). The app keeps its default.
            policy = defaults["policy"]
        out["policy"] = policy
    return out


_DEFAULTS = load_defaults()
BUDGET_DEFAULT = _DEFAULTS["budget"]
TAU_DEFAULT = _DEFAULTS["tau"]
POLICY_DEFAULT = _DEFAULTS["policy"]
N_ITEMS = 10
SEED = 0
# Demo cache (reuses existing predictor cache naming semantics)
DEMO_CALIBRATION = "platt"