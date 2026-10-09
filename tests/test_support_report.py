"""Assembly tests for the support report — P4.

Covers the guarantees ``src.support.report`` promises: validation at every
boundary, no fabricated fields, reference resolution, honest generation status,
determinism, and the end-to-end path from ``run_episode`` through
``explain_outcome`` to a validated, JSON-serialisable report.

All fixtures are synthetic: a deterministic toy predictor with an exact
prior-marginalisation method, and synthetic records. No participant data is
used anywhere in this file.
"""
from __future__ import annotations

import json
import math
from itertools import product

import numpy as np
import pytest

from src.env.environment import run_episode
from src.env.state import OBSERVED
from src.explain.outcome import explain_outcome
from src.support.report import (build_support_report,
                                evidence_from_episode,
                                explanation_from_dict,
                                screening_result_from_episode,
                                screening_result_from_explanation)
from src.support.schemas import (AssessmentStatus, DecisionEnum,
                                 ExplanationResult, GenerationStatus,
                                 ScreeningResult, SourceType,
                                 SupportRecommendation)

PRIOR = [0.2] * 10


# ---------------------------------------------------------------------------
# a synthetic predictor with v3-like exact marginalisation
# ---------------------------------------------------------------------------

def _score(cfg) -> float:
    s = -1.5 + sum(0.9 if v == 1 else 0.1 for v in cfg)
    return 1.0 / (1.0 + math.exp(-s))


def make_predictor():
    """Deterministic toy scorer: logistic over observed-plus-marginalised
    items, with an exact marginalisation under a factorised prior (the v3
    interface, so an uncertainty band is computable)."""
    def predict(state):
        return _probability(state, PRIOR)

    def _probability(state, prior):
        mask = np.asarray(state["mask"], dtype=int)
        value = np.asarray(state["value"], dtype=int)
        n = int(mask.size)
        total = 0.0
        for cfg in product((0, 1), repeat=n):
            p = 1.0
            for j in range(n):
                if mask[j] == OBSERVED:
                    p *= 1.0 if cfg[j] == int(value[j]) else 0.0
                else:
                    p *= prior[j] if cfg[j] == 1 else (1.0 - prior[j])
            total += p * _score(cfg)
        return float(total)

    def probability_with_prior(state, prior):
        return _probability(state, list(prior))

    predict.probability_with_prior = probability_with_prior
    predict.prior = list(PRIOR)
    predict.metadata = {"predictor_version": "synthetic-test-v3like",
                        "n_train": 200, "trained_on": "synthetic",
                        "calibration_method": "platt"}
    return predict


def _record(responses):
    return {"item_responses": list(responses),
            "missing_mask": [False] * len(responses),
            "label": 1}


def _ask_lowest(state, legal):
    items = [a for a in legal if a != -1]
    return min(items) if items else -1


def _episode(responses=(1, 0, 1, 1, 0, 0, 1, 0, 1, 0), budget=6):
    return run_episode(_record(responses), budget, 0, _ask_lowest,
                       make_predictor(), 0.0, tau=0.5)


# ---------------------------------------------------------------------------
# the screening result
# ---------------------------------------------------------------------------

def test_screening_result_from_episode_records_the_outcome():
    ep = _episode()
    sr = screening_result_from_episode(ep, result_id="sr-1", model_version="3",
                                       input_reference="episode:e-1")
    assert isinstance(sr, ScreeningResult)
    assert sr.result_id == "sr-1"
    assert sr.decision is DecisionEnum.REFERRAL_RECOMMENDED or \
        sr.decision is DecisionEnum.NO_REFERRAL_INDICATED
    assert 0.0 <= sr.p_hat <= 1.0
    assert sr.items_asked == ["A1", "A2", "A3", "A4", "A5", "A6"]
    assert sr.n_questions_asked == 6
    assert sr.budget == 6
    assert sr.stop_reason == "budget_exhausted"


def test_screening_result_requires_episode_output():
    with pytest.raises(ValueError, match="decision"):
        screening_result_from_episode({}, result_id="sr-1")


def test_screening_result_from_a_stored_explanation():
    ep = _episode()
    expl = explain_outcome(ep, make_predictor(), tau=0.5, prior=PRIOR)
    sr = screening_result_from_explanation(expl, result_id="sr-replay")
    # the episode records item *indices*; the contract carries item *codes*
    assert sr.items_asked == ["A1", "A2", "A3", "A4", "A5", "A6"]
    assert sr.n_questions_asked == expl["outcome"]["n_questions_asked"]
    assert sr.p_hat == expl["outcome"]["p_hat"]
    assert sr.stop_reason == expl["outcome"]["stop_reason"]
    assert sr.decision is DecisionEnum(expl["outcome"]["decision"])


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def test_evidence_from_trace_and_from_state_agree():
    ep = _episode()
    from_trace = evidence_from_episode(ep)
    assert [e.item_code for e in from_trace] == ["A1", "A2", "A3", "A4", "A5",
                                                 "A6"]
    assert all(e.source_type is SourceType.OBSERVED_RESPONSE for e in from_trace)
    assert all(e.observed_value in (0, 1) for e in from_trace)
    assert all(e.source_reference.startswith("episode:trace[")
               for e in from_trace)

    state_only = {"final_state": ep["final_state"],
                  "items_asked": ep["items_asked"]}
    from_state = evidence_from_episode(state_only)
    assert [(e.item_code, e.observed_value) for e in from_state] == \
        [(e.item_code, e.observed_value) for e in from_trace]
    assert all(e.source_reference == "episode:final_state" for e in from_state)


# ---------------------------------------------------------------------------
# the explanation adapter
# ---------------------------------------------------------------------------

def _complete_explanation(**over):
    expl = {
        "schema_version": "outcome-explanation/1.0",
        "outcome": {"decision": "REFERRAL_RECOMMENDED", "p_hat": 0.62,
                    "tau": 0.5, "n_questions_asked": 2,
                    "items_asked": ["A1", "A8"]},
        "attribution": {"method": "predictor_partial_state_value_function"},
        "contributions": [
            {"item": "A1", "phi": 0.20, "direction": "raises"},
            {"item": "A8", "phi": -0.10, "direction": "lowers"},
        ],
        "evidence": {"observed_responses": [
                         {"item": "A1", "response": 1},
                         {"item": "A8", "response": 1}],
                     "supporting": ["A1"], "opposing": ["A8"]},
        "uncertainty": {"available": True, "method": "prior_sensitivity_band",
                        "interval": [0.5, 0.7],
                        "interpretation": "model stability, not clinical"},
        "limitations": ["a stated limitation"],
        "warnings": [],
    }
    expl.update(over)
    return expl


def test_explanation_adapter_maps_every_field():
    er = explanation_from_dict(_complete_explanation(), result_id="sr-1")
    assert isinstance(er, ExplanationResult)
    assert er.screening_result_id == "sr-1"
    assert [c.item_code for c in er.feature_contributions] == ["A1", "A8"]
    assert er.feature_contributions[0].contribution_value == 0.20
    assert er.feature_contributions[0].direction.value == "raises"
    assert er.feature_contributions[1].direction.value == "lowers"
    assert er.feature_contributions[0].baseline_reference == \
        "prior_only_unasked_all_items"
    assert [e.item_code for e in er.supporting_evidence] == ["A1"]
    assert [e.item_code for e in er.opposing_evidence] == ["A8"]
    assert all(e.source_type is SourceType.OBSERVED_RESPONSE
               for e in er.supporting_evidence + er.opposing_evidence)
    assert er.uncertainty is not None and er.uncertainty.available is True
    assert er.uncertainty.interval_low == 0.5
    assert er.generation_status is GenerationStatus.COMPLETE
    assert er.limitations == ["a stated limitation"]


def test_missing_uncertainty_band_is_reported_not_fabricated():
    er = explanation_from_dict(_complete_explanation(
        uncertainty={"available": False, "method": "prior_sensitivity_band",
                     "reason": "predictor version has no prior mechanism"}),
        result_id="sr-1")
    assert er.uncertainty.available is False
    assert er.uncertainty.interval_low is None
    assert er.uncertainty.interval_high is None
    assert "no prior mechanism" in er.uncertainty.interpretation
    # and the report says the explanation is partial, not complete
    assert er.generation_status is GenerationStatus.PARTIAL


def test_absent_uncertainty_block_is_not_an_error():
    er = explanation_from_dict(_complete_explanation(uncertainty=None),
                               result_id="sr-1")
    assert er.uncertainty.available is False
    assert er.generation_status is GenerationStatus.PARTIAL


def test_warnings_and_empty_attributions_make_the_explanation_partial():
    er = explanation_from_dict(_complete_explanation(
        warnings=["decision mismatch"]), result_id="sr-1")
    assert er.generation_status is GenerationStatus.PARTIAL
    er = explanation_from_dict(_complete_explanation(
        contributions=[], evidence={"observed_responses": [], "supporting": [],
                                    "opposing": []}), result_id="sr-1")
    assert er.generation_status is GenerationStatus.PARTIAL


def test_adapter_rejects_non_dicts():
    with pytest.raises(TypeError):
        explanation_from_dict(["not", "a", "dict"], result_id="sr-1")


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def test_report_with_a_skipped_questionnaire_is_honest():
    rep = build_support_report(
        episode_result=_episode(), result_id="sr-skip", input_reference="episode:e-2")
    assert rep.screening_result.result_id == "sr-skip"
    assert rep.report_id == "rep-sr-skip"
    assert rep.explanation is None                 # no explanation was produced
    hyp = [a for a in rep.support_assessments
           if a.status is AssessmentStatus.HYPOTHESIS_FROM_OBSERVED]
    assert hyp, "atypical responses should produce labelled hypotheses"
    assert rep.recommendations == []
    assert "not a diagnosis" in rep.disclaimer
    # the hypothesis disclosure is part of the report limitations
    assert any("hypothesis" in lim for lim in rep.limitations)


def test_report_limitations_disclose_the_pending_review_state():
    rep = build_support_report(
        episode_result=_episode(),
        followup_answers=[{"question_id": "q_emotion_support", "value": "yes"}],
        result_id="sr-yes")
    assert rep.recommendations, "a confirmed need must produce suggestions"
    assert all(r.review_status == "pending_expert_review"
               for r in rep.recommendations)
    joined = " ".join(rep.limitations)
    assert "not yet reviewed" in joined or "pending" in joined
    assert "is a diagnosis" in joined


def test_report_carries_the_screening_disclaimer_and_optionality():
    rep = build_support_report(
        episode_result=_episode(),
        followup_answers=[{"question_id": "q_comm_support", "value": "yes"}],
        result_id="sr-disclaimer")
    assert rep.disclaimer.startswith("Screening support only.")
    assert "not a diagnosis" in rep.disclaimer
    assert "optional" in rep.disclaimer


def test_confirming_social_communication_fires_only_that_domain():
    rep = build_support_report(
        episode_result=_episode(),
        followup_answers=[{"question_id": "q_comm_support", "value": "yes"}],
        result_id="sr-comm")
    domains = {r.domain for r in rep.recommendations}
    assert domains == {"social_communication"}
    confirmed = [a for a in rep.support_assessments
                 if a.status is AssessmentStatus.USER_CONFIRMED]
    assert [a.domain for a in confirmed] == ["social_communication"]
    for r in rep.recommendations:
        assert r.related_assessment_ids == ["as-social_communication"]
        assert isinstance(r, SupportRecommendation)


def test_declining_everything_produces_an_empty_but_valid_report():
    rep = build_support_report(
        episode_result=_episode(),
        followup_answers=[{"question_id": "q_emotion_support", "value": "no"}],
        result_id="sr-no")
    assert rep.recommendations == []
    assert [a.status for a in rep.support_assessments
            if a.domain == "emotional_regulation"] == \
        [AssessmentStatus.USER_DECLINED]
    assert rep.limitations


def test_invalid_followup_answers_are_refused():
    with pytest.raises(ValueError, match="invalid follow-up answers"):
        build_support_report(
            episode_result=_episode(),
            followup_answers=[{"question_id": "q_nope", "value": "yes"}],
            result_id="sr-bad")
    with pytest.raises(ValueError, match="invalid follow-up answers"):
        build_support_report(
            episode_result=_episode(),
            followup_answers=[{"question_id": "q_comm_preference",
                               "value": "yes"}],
            result_id="sr-bad2")


def test_report_requires_an_outcome_source():
    with pytest.raises(ValueError, match="episode_result or an explanation"):
        build_support_report(result_id="sr-none")


def test_report_from_a_stored_explanation_only():
    ep = _episode()
    expl = explain_outcome(ep, make_predictor(), tau=0.5, prior=PRIOR)
    rep = build_support_report(
        explanation=expl,
        followup_answers=[{"question_id": "q_emotion_support", "value": "yes"}],
        result_id="sr-replay")
    assert rep.screening_result.input_reference == "outcome-explanation"
    assert rep.explanation is not None
    assert rep.explanation.screening_result_id == "sr-replay"
    assert rep.recommendations


# ---------------------------------------------------------------------------
# end to end: run_episode -> explain_outcome -> report
# ---------------------------------------------------------------------------

def test_end_to_end_screening_then_support():
    predictor = make_predictor()
    ep = _episode()
    expl = explain_outcome(ep, predictor, tau=0.5, prior=PRIOR)
    rep = build_support_report(
        episode_result=ep, explanation=expl,
        followup_answers=[{"question_id": "q_emotion_support", "value": "yes"},
                          {"question_id": "q_comm_preference", "value": "yes",
                           "choice": "written"}],
        result_id="sr-e2e", model_name="synthetic",
        model_version="synthetic-test-v3like", input_reference="episode:e2e")

    # screening
    assert rep.screening_result.decision in (DecisionEnum.REFERRAL_RECOMMENDED,
                                            DecisionEnum.NO_REFERRAL_INDICATED)
    assert rep.screening_result.model_version == "synthetic-test-v3like"

    # explanation survived the adapter
    assert rep.explanation is not None
    assert rep.explanation.generation_status is GenerationStatus.COMPLETE
    assert rep.explanation.feature_contributions
    assert rep.explanation.screening_result_id == "sr-e2e"

    # support layer fired on the confirmed need and the stated preference
    confirmed = [a for a in rep.support_assessments
                 if a.status is AssessmentStatus.USER_CONFIRMED]
    prefs = [a for a in rep.support_assessments
             if a.status is AssessmentStatus.USER_STATED_PREFERENCE]
    assert {a.domain for a in confirmed} == {"emotional_regulation"}
    assert {a.domain for a in prefs} == {"communication_accessibility"}
    assert {r.domain for r in rep.recommendations} == {"emotional_regulation",
                                                      "communication_accessibility"}

    # every reference inside the report resolves, and it serialises cleanly
    dumped = rep.model_dump(mode="json")
    json.dumps(dumped)
    ids = {a.assessment_id for a in rep.support_assessments}
    for r in rep.recommendations:
        assert set(r.related_assessment_ids) <= ids
    evidence_ids = [e.evidence_id for a in rep.support_assessments
                    for e in a.supporting_evidence]
    assert len(evidence_ids) == len(set(evidence_ids))


def test_end_to_end_determinism():
    predictor = make_predictor()
    ep = _episode()
    expl = explain_outcome(ep, predictor, tau=0.5, prior=PRIOR)
    answers = [{"question_id": "q_sensory_support", "value": "yes"}]
    a = build_support_report(episode_result=ep, explanation=expl,
                             followup_answers=answers, result_id="sr-det")
    b = build_support_report(episode_result=ep, explanation=expl,
                             followup_answers=answers, result_id="sr-det")
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    da.pop("created_at"), db.pop("created_at")
    assert da == db


def test_end_to_end_hypothesis_only_when_screening_suggests_it():
    """A domain with no screening-data linkage must not be marked as a
    hypothesis even when the interview observed atypical responses."""
    predictor = make_predictor()
    ep = _episode(responses=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    expl = explain_outcome(ep, predictor, tau=0.5, prior=PRIOR)
    rep = build_support_report(episode_result=ep, explanation=expl,
                               result_id="sr-hyp")
    hypotheses = {a.domain for a in rep.support_assessments
                  if a.status is AssessmentStatus.HYPOTHESIS_FROM_OBSERVED}
    assert "social_communication" in hypotheses
    assert "sensory_environment" not in hypotheses
    assert rep.recommendations == []
