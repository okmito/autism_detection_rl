"""Tests for the CORRECTED fixed-budget experimental design.

Locks the corrections that were requested after the first evaluation:
budget-matched random, two separate references, prespecified F1 with Holm, DQN
multi-seed reproducibility, the ExactDP oracle under the environment reward, the
Bayesian prior audit, and the surrogate nature of the reward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

from src.eval.baselines import (REQUIRED_BASELINES, STOPPING_EXPERIMENT_ONLY_BASELINES,
                                assert_all_baselines, missing_baselines)
from src.eval.fast_stats import auroc, holm_adjust, paired_auroc_comparison


def _load(name: str):
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _design():
    d = _load("polish_fixed_budget_protocol.json")
    if not d:
        pytest.skip("run scripts/step15_corrected_design.py first")
    return d


def _evaluation():
    d = _load("polish_fixed_budget_external_evaluation.json")
    if not d:
        pytest.skip("run scripts/step16_fixed_budget_external.py first")
    return d


# ================================== 1. random baseline is budget matched
def test_random_fixed_policy_never_selects_stop():
    from src.env.environment import STOP
    from src.policies.random_fixed import RandomFixedLengthPolicy

    pol = RandomFixedLengthPolicy(seed=0)
    # STOP deliberately offered; the policy must still return an item
    for _ in range(50):
        action = pol({"mask": np.zeros(10, dtype=int), "value": np.full(10, -1),
                      "n": 10}, [0, 1, 2, 3, STOP])
        assert action != STOP
        assert action in (0, 1, 2, 3)


def test_random_fixed_asks_exactly_b_questions_every_episode():
    """Regression test for the defect that invalidated the first comparison."""
    from src.data.ingest import load_dataset
    from src.env.environment import run_episode
    from src.models.logistic_predictor import LogisticPredictor
    from src.policies.random_fixed import RandomFixedLengthPolicy

    path = RESULTS / "predictor_logistic_saudi_v3.pkl"
    if not path.exists():
        pytest.skip("run scripts/step12_new_predictor.py first")
    predictor = LogisticPredictor.load(path)
    saudi = load_dataset("saudi", synthetic=False)
    records = saudi[:12]
    for B in (2, 3, 4, 5, 10):
        pol = RandomFixedLengthPolicy(seed=0)
        counts = []
        for rec in records:
            ep = run_episode(rec, question_budget=B, b_min=B, policy=pol,
                             predictor=predictor.predict_state, lambda_cost=0.0,
                             cost_mode="uniform", tau=0.5)
            asked = ep["items_asked"]
            counts.append(len(asked))
            assert len(set(asked)) == len(asked), "a question was repeated"
            assert set(asked).issubset(set(range(10)))
        assert all(c == B for c in counts), f"B={B} gave {sorted(set(counts))}"


def test_random_policy_with_stop_is_untouched_and_excluded():
    """RandomPolicy keeps STOP for a separate stopping experiment."""
    from src.policies.random_policy import RandomPolicy
    assert "random_questioning" in STOPPING_EXPERIMENT_ONLY_BASELINES
    assert "random_questioning" not in REQUIRED_BASELINES
    pol = RandomPolicy(seed=0)
    assert callable(pol)


def test_every_evaluated_arm_is_budget_matched():
    """min == max == B for every policy arm in the corrected evaluation."""
    ev = _evaluation()
    arms = [r for r in ev["results"] if r["arm"] != "item_count"]
    assert arms
    for r in arms:
        assert r["min_questions"] == r["B"], r
        assert r["max_questions"] == r["B"], r
        assert r["questions_used"] == float(r["B"]), r


# ================================ 2. two references, neither the sole ceiling
def test_both_references_are_reported_separately():
    ev = _evaluation()
    refs = ev["references"]
    assert refs["A_v3_full_information"]["name"] == "A_v3_full_information"
    assert refs["B_item_count"]["name"] == "B_item_count"
    assert "Neither is called the sole ceiling" in refs["note"]
    # they are genuinely different numbers
    assert refs["A_v3_full_information"]["auroc"] != refs["B_item_count"]["auroc"]


def test_both_deltas_are_reported_for_every_row():
    for r in _evaluation()["results"]:
        assert "delta_vs_v3_full" in r
        assert "delta_vs_item_count" in r


def test_item_count_baseline_is_reported_as_a_row():
    arms = {r["arm"] for r in _evaluation()["results"]}
    assert "item_count" in arms
    assert_all_baselines(sorted(arms), context="corrected evaluation")
    assert not missing_baselines(sorted(arms))


# ================================== 3. corrected B=10 wording, no sole ceiling
def test_b10_wording_is_not_a_sole_ceiling_claim():
    """The artifact must not ASSERT a sole-ceiling framing.

    The phrase legitimately appears inside the *prohibition* note, so this checks
    the notes that make claims, not a raw substring scan of the whole document.
    """
    ev = _evaluation()
    assert "Neither is called the sole ceiling" in ev["references"]["note"]
    for key in ("headline_claim", "conclusion", "claim", "interpretation"):
        assert key not in ev, f"unexpected assertive field {key!r}"
    for r in ev["results"]:
        for banned in ("sole ceiling", "no policy can exceed"):
            assert banned not in json.dumps(r).lower()


def test_b10_corrected_wording_is_recorded():
    ev = _evaluation()
    assert "asking all 10 questions" in ev["b10_interpretation"]["statement"]
    assert "slightly stronger externally" in ev["b10_interpretation"]["statement"]


def test_no_equivalence_claim_anywhere():
    """Margin OPEN, and no comparison carries an equivalence verdict.

    The words appear inside ``claims_forbidden``, which is the point of that
    field, so the test inspects verdicts rather than vocabulary.
    """
    ev = _evaluation()
    assert ev["equivalence"]["acceptable_AUROC_margin"] == "OPEN"
    assert ev["equivalence"]["state"] == "OPEN"
    for c in ev["confirmatory_family_F1"]["comparisons"]:
        assert c.get("equivalence_declared", "NOT DECLARED") == "NOT DECLARED"
        assert "equivalent" not in json.dumps(
            {k: v for k, v in c.items() if k != "comparison"}).lower()
    for row in ev["results"]:
        assert "non_inferior" not in row
        assert "equivalent" not in row


def test_b10_arms_are_identical_because_no_selection_remains():
    """At B=10 every item is asked, so all arms must coincide exactly."""
    ev = _evaluation()
    at10 = {r["arm"]: r["auroc"] for r in ev["results"] if r["B"] == 10}
    arms = {k: v for k, v in at10.items() if k != "item_count"}
    assert len(set(round(v, 12) for v in arms.values())) == 1, at10


# ============================= 4. F1 is the only confirmatory analysis
def test_f1_has_exactly_the_eight_prespecified_comparisons():
    f1 = _evaluation()["confirmatory_family_F1"]
    assert f1["family_size"] == 8
    assert f1["exclusive"] is True
    assert f1["correction"] == "holm-bonferroni"
    assert f1["alpha"] == 0.05
    got = sorted(c["comparison"] for c in f1["comparisons"])
    expected = sorted(
        f"{a} - random_fixed @ B={B}"
        for B in (2, 3, 4, 5)
        for a in ("greedy_information_gain", "beta_greedy_evoi"))
    assert got == expected


def test_holm_correction_is_applied_and_never_over_claims():
    f1 = _evaluation()["confirmatory_family_F1"]
    for c in f1["comparisons"]:
        assert c["holm_adjusted_p"] is not None
        assert 0.0 <= c["holm_adjusted_p"] <= 1.0
        assert c["holm_reject_at_0.05"] in (True, False)
        assert c["ci_lo"] <= c["difference"] <= c["ci_hi"]


def test_exploratory_comparisons_are_labelled_uncorrected():
    for c in _evaluation()["exploratory_comparisons"]:
        assert c["family"] in ("exploratory", "exploratory_vs_reference")
        assert "none" in c["correction"]


def test_no_equivalence_claim_anywhere():
    """Margin OPEN, and no comparison carries an equivalence verdict.

    The words appear inside ``claims_forbidden``, which is the point of that
    field, so the test inspects verdicts rather than vocabulary.
    """
    ev = _evaluation()
    assert ev["equivalence"]["acceptable_AUROC_margin"] == "OPEN"
    assert ev["equivalence"]["state"] == "OPEN"
    for c in ev["confirmatory_family_F1"]["comparisons"]:
        assert c.get("equivalence_declared", "NOT DECLARED") == "NOT DECLARED"
    for row in ev["results"]:
        assert "non_inferior" not in row
        assert "equivalent" not in row
    for c in ev["exploratory_comparisons"]:
        assert c.get("equivalence_declared", "NOT DECLARED") == "NOT DECLARED"


# =========================================== 5. DQN multi-seed reproducibility
def test_five_seeds_were_trained_per_budget_and_all_reported():
    multi = _design()["dqn_multi_seed"]
    assert multi["n_seeds"] >= 5
    assert multi["all_seeds_reported"] is True
    assert multi["polish_inspected_during_selection"] is False
    for B, rec in multi["per_budget"].items():
        assert len(multi["per_seed_records"][B]) == multi["n_seeds"], B
        assert 0.0 <= rec["validation_reward_sd"] < 1.0
        assert "validation_auroc_sd" in rec


def test_seed_selection_rule_is_prespecified_and_saudi_only():
    multi = _design()["dqn_multi_seed"]
    assert "SAUDI" in multi["selection_rule"].upper()
    assert "lowest seed" in multi["selection_rule"]
    assert multi["polish_inspected_during_selection"] is False


def test_every_seed_policy_file_exists_with_matching_hash():
    multi = _design()["dqn_multi_seed"]
    for B, rec in multi["per_budget"].items():
        for row in multi["per_seed_records"][B]:
            path = REPO / row["path"]
            assert path.exists(), row["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            # new artefacts live in their own directory, never over the originals
            assert row["path"].replace("\\", "/").startswith(
                "results/policies_v3_multiseed/"), row["path"]


def test_single_seed_policy_files_were_not_overwritten():
    legacy = RESULTS / "policies_v3"
    for B in (2, 3, 4, 5, 10):
        assert (legacy / f"dqn_v3_B{B}.pt").exists(), (
            "the original single-seed policy files must be retained")


def test_seed_variability_is_visible_not_hidden():
    spread = _evaluation()["dqn_seed_spread"]
    varied = [k for k, v in spread.items() if v["saudi_val_auroc_sd"] > 0]
    assert varied, "seed variability should be non-zero at reduced budgets"


# ================================================= 6. ExactDP under the reward
def test_exactdp_ran_for_every_budget():
    dp = _design()["exact_dp"]
    assert dp["predictor_supplied"] is True
    got = sorted(r["B"] for r in dp["results"])
    assert got == [2, 3, 4, 5, 10]
    for r in dp["results"]:
        assert r["J_exact"] is not None
        assert r["status"] == "optimal"


def test_exactdp_default_behaviour_is_backward_compatible():
    """Without a predictor the solver must keep its original objective."""
    from src.data.ingest import load_dataset
    from src.data.splits import stratified_split
    from src.solvers.exact_custom import ExactDP

    saudi = load_dataset("saudi", synthetic=False)
    train, _, _ = stratified_split(saudi, seed=0)
    dp = ExactDP(train, n_items=10, budget=2, b_min=2, lambda_cost=0.0)
    assert dp.predictor is None
    sol = dp.solve()
    # identical to the value recorded before the predictor argument existed
    assert sol["V_star"] == pytest.approx(0.93862870, abs=1e-6)


def test_optimality_gap_is_reported_and_labelled_simulator_only():
    ev = _evaluation()
    gaps = ev["exact_dp"]["optimality_gaps"]
    assert gaps
    for g in gaps:
        assert g["J_exact"] is not None
        assert g["optimality_gap"] == pytest.approx(g["J_exact"] - g["J_policy"])
    assert "NOT clinically optimal" in ev["exact_dp"]["interpretation_limit"]


# ==================================================== 7. reward is a surrogate
def test_reward_is_recorded_as_a_surrogate_objective():
    audit = _evaluation()["reward_surrogate_audit"]
    assert audit["status"] == "SURROGATE OBJECTIVE"
    assert "questionnaire" in audit["development_label"]
    assert "clinician" in audit["external_target"]
    assert "clinical diagnosis" in audit["prohibited"]
    assert audit["n_points"] > 0
    assert -1.0 <= audit["pearson"] <= 1.0


def test_reward_formula_unchanged():
    assert _design()["reward_audit"]["formula"] == \
        "R = 1 - (p_hat - y)^2, lambda = 0"


# ============================================ 8. Bayesian prior audit
def test_prior_support_audit_reports_the_real_sparsity():
    support = _evaluation()["bayesian_prior_summary"]["support"]
    assert support["n_possible_configurations"] == 1024
    assert support["train_observed_configurations"] == 155
    assert support["train_zero_mass_configurations"] == 869
    assert support["train_cells_with_count_one"] == 121
    assert support["effective_support_participation_ratio"] < 60


def test_empirical_joint_prior_is_a_documented_failed_variant():
    audit = _design()["bayesian_prior"]["audit"]
    variants = {v["variant"]: v for v in audit["sensitivity"]}
    a = variants["A_empirical_joint"]
    assert a["prior_zero_mass_cells"] > 0
    assert a["meets_definition_everywhere"] is False
    assert a["n_states_with_zero_prior_mass"] > 0
    lowered = audit["conclusion"].lower()
    assert "a (empirical joint)" in lowered
    assert "fails" in lowered
    assert "primary" in lowered


def test_factorized_prior_is_primary_and_was_not_polish_selected():
    ba = _design()["bayesian_prior"]
    assert ba["primary_variant"] == "C_factorized_item"
    assert ba["alpha_tuned"] is False
    assert ba["selected_using_polish"] is False
    variants = {v["variant"]: v for v in ba["audit"]["sensitivity"]}
    assert variants["C_factorized_item"]["meets_definition_everywhere"] is True
    assert variants["B_add_alpha_joint"]["meets_definition_everywhere"] is True


def test_alpha_for_variant_b_was_prespecified():
    ba = _design()["bayesian_prior"]
    assert ba["prespecified_alpha_variant_B"] == 0.5
    assert ba["alpha_tuned"] is False


# ==================================================== fast_stats correctness
def test_auroc_is_tie_aware_and_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    for s in (rng.random(200), np.round(rng.random(200), 1), y.astype(float)):
        assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_paired_comparison_uses_identical_resamples_and_reports_all_fields():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=120)
    a = y * 0.6 + rng.random(120) * 0.3
    b = rng.random(120)
    out = paired_auroc_comparison(y, a, b, n_resamples=800, seed=3)
    assert out["status"] == "OK"
    assert out["ci_lo"] <= out["difference"] <= out["ci_hi"]
    assert 0.0 <= out["bootstrap_p_a_greater"] <= 1.0
    assert out["n_resamples"] == 800
    assert "identical participant resamples" in out["method"]


def test_holm_is_step_down_and_monotone():
    out = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}, alpha=0.05)
    adj = out["adjusted_p"]
    assert out["family_size"] == 4
    assert adj["a"] <= adj["b"] <= adj["c"] <= adj["d"]
    assert out["reject"]["a"] is True
    assert out["reject"]["d"] is False


# ============================================== 12/13 future work recorded
def test_ordinal_predictor_is_future_work_not_implemented():
    fw = _design()["ordinal_future_work"]
    assert fw["status"] == "DOCUMENTED_NOT_IMPLEMENTED"
    assert fw["binary_contract_changed"] is False


def test_nz_cross_dataset_is_deferred_and_loader_untouched():
    nz = _design()["nz_cross_dataset"]
    assert nz["status"] == "DEFERRED"
    assert nz["nz_loader_modified"] is False
    assert "1054" in nz["future_work"]
    src = (REPO / "src" / "data" / "ingest.py").read_text(encoding="utf-8")
    assert "raise NotImplementedError" in src


# ================================================ frozen material untouched
@pytest.mark.parametrize("name,expected", [
    ("predictor_saudi_v2_isotonic.pt",
     "2366a28353f77b19b628073b9f97393e3218a80e9643b93dd4456e58f02cf1d4"),
    ("predictor_saudi_v2_isotonic.pkl",
     "a1978f274797b463c44b7c7a1b9b93a007b813f5e019a6534855865664583863"),
])
def test_v2_artifacts_unchanged(name, expected):
    path = RESULTS / name
    if not path.exists():
        pytest.skip("absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_v3_predictor_not_retrained_by_the_corrected_design():
    d = _design()["predictor"]
    assert d["retrained"] is False
    assert d["polish_used"] is False
    ev = _evaluation()
    assert ev["design_sha_verified"] is True
    assert ev["predictor_sha256"] == d["sha256"]


def test_first_evaluation_results_are_retained_not_deleted():
    path = RESULTS / "polish_rl_external_evaluation.json"
    assert path.exists(), "the first evaluation must be retained"
    assert "retains_first_evaluation" in _evaluation()


def test_step10_primary_result_untouched():
    path = RESULTS / "polish_external_validation.json"
    if not path.exists():
        pytest.skip("run scripts/step10_external_validation.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["primary_result"]["threshold"] == 0.5
    assert data["primary_result"]["metrics"]["auroc"] == pytest.approx(
        0.8959, abs=5e-4)


def test_no_participant_rows_in_the_new_artifact():
    assert _evaluation()["contains_participant_rows"] is False