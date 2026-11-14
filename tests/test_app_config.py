"""configs/config.yaml is read by the app (audit finding T4) and must not drift from the demo constants."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app import config

REPO = Path(__file__).resolve().parent.parent


def test_the_app_reads_budget_threshold_and_policy_from_the_yaml():
    raw = yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert config.BUDGET_DEFAULT == raw["env"]["question_budget"]
    assert config.TAU_DEFAULT == raw["eval"]["operating_threshold"]
    assert config.POLICY_DEFAULT == raw["policy"]["type"]


def test_the_demo_scripts_use_the_same_budget_and_threshold_as_the_config():
    demo = (REPO / "scripts" / "demo_app.py").read_text(encoding="utf-8")
    live = (REPO / "scripts" / "demo_live.py").read_text(encoding="utf-8")
    m = re.search(r"^BUDGET, TAU, SEED, N_ITEMS = (\d+), ([\d.]+), (\d+), (\d+)$", demo, re.M)
    assert m, "demo_app constants line changed shape: update this test"
    assert (int(m.group(1)), float(m.group(2))) == (config.BUDGET_DEFAULT, config.TAU_DEFAULT)
    assert int(re.search(r"^BUDGET = (\d+)", live, re.M).group(1)) == config.BUDGET_DEFAULT
    assert float(re.search(r"^TAU = ([\d.]+)", live, re.M).group(1)) == config.TAU_DEFAULT
    assert int(m.group(4)) == config.N_ITEMS


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_a_missing_file_falls_back_to_the_built_in_defaults(tmp_path):
    assert config.load_defaults(tmp_path / "absent.yaml") == {"budget": 6, "tau": 0.5, "policy": "greedy"}


def test_valid_values_are_used(tmp_path):
    out = config.load_defaults(write(tmp_path, "env: {question_budget: 4}\neval: {operating_threshold: 0.3}\npolicy: {type: beta_greedy}\n"))
    assert out == {"budget": 4, "tau": 0.3, "policy": "beta_greedy"}


@pytest.mark.parametrize("text", [
    "env: {question_budget: 0}", "env: {question_budget: 11}", "env: {question_budget: 'six'}",
    "env: {question_budget: true}", "env: {question_budget: 3.5}",
    "eval: {operating_threshold: 0}", "eval: {operating_threshold: 1}", "eval: {operating_threshold: 'half'}",
    "eval: {operating_threshold: true}",
])
def test_invalid_values_fail_loudly_instead_of_silently_changing_results(tmp_path, text):
    with pytest.raises(ValueError):
        config.load_defaults(write(tmp_path, text))


def test_a_policy_the_app_does_not_serve_keeps_the_app_default(tmp_path):
    assert config.load_defaults(write(tmp_path, "policy: {type: dqn}"))["policy"] == "greedy"


def test_an_empty_yaml_is_all_defaults(tmp_path):
    assert config.load_defaults(write(tmp_path, "")) == {"budget": 6, "tau": 0.5, "policy": "greedy"}


def test_start_uses_the_configured_default_policy_when_none_is_given(api, monkeypatch):
    from app import server
    seen = {}
    monkeypatch.setattr(server.screening, "start_session", lambda mode, policy: seen.update(mode=mode, policy=policy) or {"ok": 1})
    monkeypatch.setattr(server, "POLICY_DEFAULT", "beta_greedy")
    assert api.post("/api/screening/start", json={}).status_code == 200
    assert seen == {"mode": "interactive", "policy": "beta_greedy"}
