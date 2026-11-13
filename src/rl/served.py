"""The adaptive-RL policy the application serves, and the gate that decides whether it may be served.

``results/rl_adaptive/served_policy.json`` is written by ``scripts/step20_build_served_policy.py`` *after* the
nested-CV experiment. It records which agent passed the pre-registered integration gate, the belief model and MDP
configuration fitted on the whole development cohort with the same constrained cost-selection procedure the
experiment evaluated, and SHA-256 hashes for everything loaded.

:func:`load_served_agent` refuses (``ServedPolicyUnavailable``) when the artifact is missing, malformed, altered, or
records a failed gate. The caller decides what to do then; the application falls back to greedy information gain
and labels the session as a fallback, never silently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.rl.belief import NaiveBayesBelief
from src.rl.mdp import MDPConfig

ARTIFACT_KIND = "rl_adaptive_served_policy"
ARTIFACT_VERSION = 1
SERVED_NAME = "served_policy.json"


class ServedPolicyUnavailable(RuntimeError):
    """The RL policy cannot be served; ``str(exc)`` is a reason safe to show to a user."""


def _canonical_sha(payload: Dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "payload_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_served_policy(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload["kind"], payload["version"] = ARTIFACT_KIND, ARTIFACT_VERSION
    payload["payload_sha256"] = _canonical_sha(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return payload


def load_served_agent(results_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """Return ``(agent, info)`` or raise :class:`ServedPolicyUnavailable` with a user-safe reason."""
    path = Path(results_dir) / "rl_adaptive" / SERVED_NAME
    if not path.is_file():
        raise ServedPolicyUnavailable("the trained RL policy artifact has not been built "
                                      "(run scripts/step19 and scripts/step20)")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ServedPolicyUnavailable("the RL policy artifact is unreadable") from None
    if payload.get("kind") != ARTIFACT_KIND or payload.get("version") != ARTIFACT_VERSION:
        raise ServedPolicyUnavailable("the RL policy artifact has an unknown format")
    if payload.get("payload_sha256") != _canonical_sha(payload):
        raise ServedPolicyUnavailable("the RL policy artifact failed its integrity check")
    gate = payload.get("gate") or {}
    agent_name = payload.get("agent")
    if not gate.get("passed") or agent_name not in ("rl_value_iteration", "rl_dqn"):
        raise ServedPolicyUnavailable(gate.get("reason") or "no RL agent passed the integration gate")
    belief = NaiveBayesBelief.from_dict(payload["belief"])
    cfg = MDPConfig(**payload["mdp"])
    if agent_name == "rl_value_iteration":
        from src.rl.planner import BeliefValueIteration
        agent = BeliefValueIteration(belief, cfg)
    else:
        import torch
        from src.models.artifact_integrity import verify_sha256
        from src.rl.dqn import DQNAgent, DQNConfig
        wpath = path.parent / payload["dqn_weights"]["file"]
        try:
            verify_sha256(wpath, payload["dqn_weights"]["sha256"], "RL DQN weights")
        except Exception:
            raise ServedPolicyUnavailable("the RL policy weights failed their integrity check") from None
        blob = torch.load(wpath, map_location="cpu", weights_only=True)
        agent = DQNAgent.from_state_dicts(blob, belief, cfg, DQNConfig(**payload["dqn_config"]),
                                          payload["dqn_weights"].get("seeds", ()))
    info = {"agent": agent_name, "artifact_sha256": payload["payload_sha256"], "mdp": cfg.to_dict(),
            "trained_on": belief.trained_on, "cost": cfg.cost, "gate": gate,
            "expected_performance": payload.get("expected_performance"), "built_utc": payload.get("built_utc"),
            "protocol_code_sha256": payload.get("protocol_code_sha256")}
    return agent, info


def development_records(X: np.ndarray, y: np.ndarray) -> list:
    """Records in the engine's format (used for explanation reference rows and the circularity audit)."""
    X = np.asarray(X, float)
    return [{"item_responses": X[i], "label": int(y[i]), "missing_mask": np.isnan(X[i]),
             "label_source": "clinical"} for i in range(len(y))]
