"""Step 3 preliminary-artifact contract tests.

Lock the shape and metadata of the preliminary reports so future runs must
keep the tag `preliminary — V-4 / V-6 / V-7 PENDING` until those gates clear.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"

PERF_CSV = RESULTS / "perf_vs_budget_saudi.csv"
PERF_JSON = RESULTS / "perf_vs_budget_saudi.json"
FAITH_JSON = RESULTS / "faithfulness_saudi.json"
SUB_JSON = RESULTS / "subgroup_saudi.json"

EXPECTED_TAG_FRAGMENT = "preliminary"


def _require(p: Path):
    return pytest.mark.skipif(
        not p.exists(),
        reason=f"{p.name} not generated; run scripts/step3_preliminary_reports.py first",
    )


@pytest.mark.skipif(not PERF_CSV.exists(), reason="perf_vs_budget_saudi.csv not generated")
def test_perf_csv_shape():
    lines = PERF_CSV.read_text().strip().split("\n")
    header = lines[0]
    expected_cols = {"B", "policy", "items_asked_mean", "brier", "uar", "auroc", "ece"}
    assert set(header.split(",")) == expected_cols
    # 6 budgets × 2 policies + 1 terminal = 13 rows
    assert len(lines) == 1 + 13, f"unexpected row count: {len(lines)}"


@pytest.mark.skipif(not PERF_JSON.exists(), reason="perf_vs_budget_saudi.json not generated")
def test_perf_json_metadata():
    obj = json.loads(PERF_JSON.read_text())
    assert EXPECTED_TAG_FRAGMENT in obj["tag"].lower()
    assert "V-4" in obj["tag"] and "V-6" in obj["tag"]
    assert obj["dataset"] == "saudi"
    assert obj["source"] in ("real", "synthetic")
    assert "Deterministic" in obj["circularity_status"] or "Synthetic" in obj["circularity_status"]
    assert obj["lambda"] == 0.0
    # config hash + git sha (accept both path separators — script runs on Linux and Windows)
    assert "configs/config.yaml" in str(obj["config"]).replace("\\", "/")
    assert "git_sha" in obj
    # terminal reference present
    assert "terminal_B10_brier" in obj["terminal_reference"]
    # rows have B=1..6
    Bs = sorted({r["B"] for r in obj["rows"]})
    assert Bs == [1, 2, 3, 4, 5, 6]


@pytest.mark.skipif(not FAITH_JSON.exists(), reason="faithfulness_saudi.json not generated")
def test_faithfulness_metadata():
    obj = json.loads(FAITH_JSON.read_text())
    assert EXPECTED_TAG_FRAGMENT in obj["tag"].lower()
    assert obj["dataset"] == "saudi"
    assert obj["B"] == 6
    assert obj["tau"] == 0.5
    assert 0.0 <= obj["counterfactual_found_rate"] <= 1.0
    assert 0.0 <= obj["counterfactual_robust_rate"] <= 1.0
    assert abs(obj["counterfactual_found_rate"] + obj["counterfactual_robust_rate"] - 1.0) < 1e-9
    # SHAP must have 10 features
    assert set(obj["shap_abs_attribution_mean"].keys()) == {f"A{j+1}" for j in range(10)}


@pytest.mark.skipif(not SUB_JSON.exists(), reason="subgroup_saudi.json not generated")
def test_subgroup_metadata():
    obj = json.loads(SUB_JSON.read_text())
    assert EXPECTED_TAG_FRAGMENT in obj["tag"].lower()
    assert obj["dataset"] == "saudi"
    assert obj["tau"] == 0.5
    # subgroups should include sex and age groups
    keys = list(obj["subgroups"].keys())
    assert any("sex_" in k for k in keys)
    assert any("age_" in k for k in keys)
    # differences logged never suppressed (§25)
    assert "note" in obj and "never suppressed" in obj["note"]
