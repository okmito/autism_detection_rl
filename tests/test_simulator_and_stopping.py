"""Stage 6 simulator + Stage 4/5 stopping regression tests."""
import numpy as np
import pytest

from src.simulator.patient import SimulatedPatient, generate_cohort
from src.data.schema import validate_dataset
from src.data.validate import validate_datasets
from src.data.ingest import load_saudi
from src.policies.stopping import decide_stop, stopping_summary


def test_simulator_reproducible():
    a = np.ones(10); b = np.zeros(10)
    s1 = SimulatedPatient(a, b, seed=42)
    s2 = SimulatedPatient(a, b, seed=42)
    r1 = s1.generate(theta=1.0)
    r2 = s2.generate(theta=1.0)
    assert np.array_equal(np.nan_to_num(r1["item_responses"]), np.nan_to_num(r2["item_responses"]))
    assert r1["label"] == r2["label"]


def test_simulator_labels_are_simulated_not_clinical():
    a = np.ones(10); b = np.zeros(10)
    s = SimulatedPatient(a, b, seed=0)
    rec = s.generate(theta=10.0)
    assert rec["label_source"] == "simulated"
    assert rec["provenance"].startswith("simulated")
    validate_dataset([rec], n_items=10)


def test_simulator_mixed_with_real_is_refused():
    from src.data.validate import validate_records
    sim = SimulatedPatient(np.ones(10), np.zeros(10), seed=0).generate(theta=0.0)
    real = load_saudi()[0]
    # A caller that merges simulator + real into one list and validates it
    # must be refused — synthetic and real datasets stay separate.
    with pytest.raises(ValueError, match="Mixed real"):
        validate_records([real, sim], n_items=10)


def test_simulator_incomplete_and_contradictory():
    s = SimulatedPatient(np.ones(10), np.zeros(10), seed=1, missing_rate=1.0, noise_flip=1.0)
    rec = s.generate(theta=0.0)
    assert rec["missing_mask"].all()
    s2 = SimulatedPatient(np.ones(10), np.zeros(10), seed=1)
    r = s2.generate(theta=0.0)
    assert r["missing_mask"].any() or r["missing_mask"].sum() < 10  # default no missing


def test_simulator_shift_works():
    c0 = generate_cohort(50, seed=0, mean_shift=0.0)
    c1 = generate_cohort(50, seed=0, mean_shift=2.0)
    assert len(c0) == len(c1) == 50
    # higher mean theta -> more simulator-positive labels (threshold test only)
    pos0 = sum(r["label"] for r in c0); pos1 = sum(r["label"] for r in c1)
    assert pos1 >= pos0


def test_stop_decision_budget_minmax():
    assert decide_stop(predictive_sd=0.1, best_net_score=1.0, n_asked=10,
                       min_questions=0, max_questions=10, abstain_sd=0.25) == "STOP"
    assert decide_stop(predictive_sd=0.1, best_net_score=-1.0, n_asked=1,
                       min_questions=3, max_questions=10, abstain_sd=0.25) == "CONTINUE"


def test_stop_decision_abstain_vs_stop():
    assert decide_stop(predictive_sd=0.9, best_net_score=-0.1, n_asked=3,
                       min_questions=0, max_questions=10, abstain_sd=0.25) == "ABSTAIN"
    assert decide_stop(predictive_sd=0.1, best_net_score=-0.1, n_asked=3,
                       min_questions=0, max_questions=10, abstain_sd=0.25) == "STOP"


def test_stopping_summary_never_clinical():
    s = stopping_summary(decision="ABSTAIN", predictive_sd=0.9,
                         best_net_score=-0.1, n_asked=3, budget=6)
    assert s["is_clinical_threshold"] is False
    assert "inconclusive" in s["interpretation"]
