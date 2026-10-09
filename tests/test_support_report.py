"""Assembly tests for the support report — P4 (revised: no second form).

Covers the guarantees ``src/support/report.py`` promises: validation at every
boundary, question ids checked against the verified instrument, fail-closed
evidence sufficiency, honest disclosure of unmeasured areas, determinism, and
the end-to-end path from ``run_episode`` through ``explain_outcome`` to a
validated, JSON-serialisable report — assembled **only** from the answers the
person already gave.

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
from src.support.domains import DOMAINS, UNASSESSED_AREAS
from src.support.questionnaire import ITEM_BY_CODE, ITEM_CODES
from src.support.report import (build_support_report, evidence_from_episode,
                                explanation_from_dict,
                                questionnaire_evidence_from_episode,
                                screening_result_from_episode,
                                screening_result_from_explanation)
from src.support.schemas import (AssessmentStatus, DecisionEnum,
                                 ExplanationResult, GenerationStatus,
                                 QuestionnaireEvidence, ScreeningResult,
                                 SupportRecommendation)
from src.support.strategies import STRATEGIES

PRIOR = [0.2] * 10


# ---------------------------------------------------------------------------
# a synthetic predictor with v3-like exact marginalisation
# ---------------------------------------------------------------------------

def _score(cfg) -> float:
    s = -1.5 + sum(0.9 if v == 1 else 0.1 for v in cfg)
    return 1.0 / (1.0 + math.exp(-s))


def make_predictor():
    """Deterministic toy scorer: logistic over observed-plus-marginalised
    items, with exact marginalisation under a factorised prior (the v3
    interface, so an uncertainty band is computable)."""
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

    def predict(state):
        return _probability(state, PRIOR)

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


ATYPICAL = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)


def _episode(responses=(1, 0, 1, 1, 0, 0, 1, 0, 1, 0), budget=6):
    return run_episode(_record(responses), budget, 0, _ask_lowest,
                       make_predictor(), 0.0, tau=0.5)


def _full_episode(responses: tuple = ATYPICAL):
    """Budget 10: every item is asked, so every triggerable area is reachable."""
    return _episode(responses=responses, budget=10)


# ---------------------------------------------------------------------------
# screening result and evidence
# ---------------------------------------------------------------------------

def test_screening_result_records_the_outcome():
    ep = _episode()
    sr = screening_result_from_episode(ep, result_id="sr-1", model_version="3",
                                       input_reference="episode:e-1")
    assert sr.result_id == "sr-1"
    assert sr.decision in (DecisionEnum.REFERRAL_RECOMMENDED,
                           DecisionEnum.NO_REFERRAL_INDICATED)
    assert 0.0 <= sr.p_hat <= 1.0
    assert sr.items_asked == ["A1", "A2", "A3", "A4", "A5", "A6"]
    assert sr.n_questions_asked == 6
    assert sr.budget == 6
    assert sr.stop_reason == "budget_exhausted"


def test_screening_result_requires_episode_output():
    with pytest.raises(ValueError, match="decision"):
        screening_result_from_episode({}, result_id="sr-1")


def test_questionnaire_evidence_uses_the_verified_instrument():
    ep = _episode()
    ev = questionnaire_evidence_from_episode(ep)
    assert [e.question_id for e in ev] == ["A1", "A2", "A3", "A4", "A5", "A6"]
    for e in ev:
        item = ITEM_BY_CODE[e.question_id]
        assert e.feature_id == item.feature_id
        assert e.question_text == item.question_text
        assert e.response in (0, 1)
    assert evidence_from_episode({"trace": [{"item": "A9", "value": 1}]})[0].item_code == "A9"


def test_unknown_recorded_item_is_refused():
    with pytest.raises(ValueError, match="does not define"):
        questionnaire_evidence_from_episode(
            {"trace": [{"item": "A11", "value": 1}]})


def test_screening_result_from_a_stored_explanation():
    ep = _episode()
    expl = explain_outcome(ep, make_predictor(), tau=0.5, prior=PRIOR)
    sr = screening_result_from_explanation(expl, result_id="sr-replay")
    assert sr.items_asked == ["A1", "A2", "A3", "A4", "A5", "A6"]
    assert sr.n_questions_asked == expl["outcome"]["n_questions_asked"]
    assert sr.p_hat == expl["outcome"]["p_hat"]
    assert sr.decision is DecisionEnum(expl["outcome"]["decision"])


# ---------------------------------------------------------------------------
# explanation adapter
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
    assert [c.item_code for c in er.feature_contributions] == ["A1", "A8"]
    assert er.feature_contributions[0].contribution_value == 0.20
    assert er.feature_contributions[0].baseline_reference == \
        "prior_only_unasked_all_items"
    assert [e.item_code for e in er.supporting_evidence] == ["A1"]
    assert [e.item_code for e in er.opposing_evidence] == ["A8"]
    assert er.uncertainty.available is True and er.uncertainty.interval_low == 0.5
    assert er.generation_status is GenerationStatus.COMPLETE


def test_missing_uncertainty_band_is_reported_not_fabricated():
    er = explanation_from_dict(_complete_explanation(
        uncertainty={"available": False, "method": "prior_sensitivity_band",
                     "reason": "predictor version has no prior mechanism"}),
        result_id="sr-1")
    assert er.uncertainty.available is False
    assert er.uncertainty.interval_low is None
    assert "no prior mechanism" in er.uncertainty.interpretation
    assert er.generation_status is GenerationStatus.PARTIAL


def test_absent_uncertainty_block_is_not_an_error():
    er = explanation_from_dict(_complete_explanation(uncertainty=None),
                               result_id="sr-1")
    assert er.uncertainty.available is False
    assert er.generation_status is GenerationStatus.PARTIAL


def test_warnings_and_empty_attributions_make_the_explanation_partial():
    assert explanation_from_dict(_complete_explanation(warnings=["mismatch"]),
                                  result_id="sr-1").generation_status is \
        GenerationStatus.PARTIAL
    er = explanation_from_dict(_complete_explanation(contributions=[]),
                               result_id="sr-1")
    assert er.generation_status is GenerationStatus.PARTIAL


def test_adapter_rejects_non_dicts():
    with pytest.raises(TypeError):
        explanation_from_dict(["not", "a", "dict"], result_id="sr-1")


# ---------------------------------------------------------------------------
# report assembly — there is no answers parameter
# ---------------------------------------------------------------------------

def test_build_support_report_takes_no_answers():
    import inspect
    params = inspect.signature(build_support_report).parameters
    assert "answers" not in params and "followup_answers" not in params, (
        "the second questionnaire must not be reintroduced as a parameter")


def test_report_from_a_session_with_no_atypical_answers():
    ep = _episode(responses=tuple([0] * 10))
    rep = build_support_report(episode_result=ep, result_id="sr-none")
    assert rep.recommendations == []
    assert all(a.status is AssessmentStatus.NO_EVIDENCE
               for a in rep.support_assessments
               if a.domain in {d.domain_id for d in DOMAINS})
    assert rep.unassessed_areas and len(rep.unassessed_areas) == len(UNASSESSED_AREAS)
    assert "not a diagnosis" in rep.disclaimer


def test_report_from_atypical_answers_offers_linked_suggestions():
    ep = _full_episode()
    rep = build_support_report(episode_result=ep, result_id="sr-yes")
    assert rep.recommendations
    for r in rep.recommendations:
        assert r.basis.value == "observed_response_pattern"
        assert set(r.triggering_question_ids) <= set(ITEM_CODES)
        assert all(v == 1 for v in r.triggering_responses)
        assert r.review_status == "pending_expert_review"
    domains = {r.domain for r in rep.recommendations}
    assert domains == {d.domain_id for d in DOMAINS}


def test_every_suggestion_is_justified_by_the_recorded_answers():
    ep = _full_episode()
    rep = build_support_report(episode_result=ep, result_id="sr-just")
    recorded = {e.question_id: e.response for e in rep.questionnaire_evidence}
    for r in rep.recommendations:
        assert r.triggering_question_ids
        for code, value in zip(r.triggering_question_ids,
                               r.triggering_responses):
            assert code in recorded, (r.recommendation_id, code)
            assert recorded[code] == value == 1


def test_unmeasured_areas_are_listed_and_never_offered():
    ep = _full_episode()
    rep = build_support_report(episode_result=ep, result_id="sr-areas")
    offered = {r.domain for r in rep.recommendations}
    assert not offered & {a.area_id for a in UNASSESSED_AREAS}
    for area in UNASSESSED_AREAS:
        assert any(area.label in label for label in rep.unassessed_areas)
    measured = {a.status for a in rep.support_assessments}
    assert AssessmentStatus.NOT_MEASURED in measured


def test_report_requires_an_outcome_source():
    with pytest.raises(ValueError, match="episode_result or an explanation"):
        build_support_report(result_id="sr-none")


def test_report_from_a_stored_explanation_only():
    ep = _episode(responses=ATYPICAL)
    expl = explain_outcome(ep, make_predictor(), tau=0.5, prior=PRIOR)
    rep = build_support_report(explanation=expl, result_id="sr-replay")
    assert rep.screening_result.input_reference == "outcome-explanation"
    assert rep.explanation is not None
    assert rep.explanation.screening_result_id == "sr-replay"
    assert rep.questionnaire_evidence
    assert rep.recommendations


# ---------------------------------------------------------------------------
# end to end: run_episode -> explain_outcome -> report
# ---------------------------------------------------------------------------

def test_end_to_end_screening_then_support():
    predictor = make_predictor()
    ep = _episode(responses=ATYPICAL)
    expl = explain_outcome(ep, predictor, tau=0.5, prior=PRIOR)
    rep = build_support_report(
        episode_result=ep, explanation=expl,
        result_id="sr-e2e", model_name="synthetic",
        model_version="synthetic-test-v3like", input_reference="episode:e2e")

    assert rep.screening_result.model_version == "synthetic-test-v3like"
    assert rep.explanation is not None
    assert rep.explanation.generation_status is GenerationStatus.COMPLETE
    assert rep.explanation.feature_contributions
    assert [e.question_id for e in rep.questionnaire_evidence] == \
        ["A1", "A2", "A3", "A4", "A5", "A6"]

    # the screening values are untouched by the support layer
    assert rep.screening_result.p_hat == expl["outcome"]["p_hat"]
    assert rep.screening_result.decision is \
        DecisionEnum(expl["outcome"]["decision"])

    dumped = rep.model_dump(mode="json")
    json.dumps(dumped)
    ids = {a.assessment_id for a in rep.support_assessments}
    for r in rep.recommendations:
        assert set(r.related_assessment_ids) <= ids


def test_end_to_end_is_deterministic():
    predictor = make_predictor()
    ep = _episode(responses=ATYPICAL)
    expl = explain_outcome(ep, predictor, tau=0.5, prior=PRIOR)
    a = build_support_report(episode_result=ep, explanation=expl,
                             result_id="sr-det")
    b = build_support_report(episode_result=ep, explanation=expl,
                             result_id="sr-det")
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    da.pop("created_at"), db.pop("created_at")
    assert da == db


def test_only_observed_items_can_trigger():
    """A5 is asked but has no curated suggestion; being atypical must not drag
    in anything for another area, and the report must say why nothing came."""
    ep = _episode(responses=(0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
    rep = build_support_report(episode_result=ep, result_id="sr-asked")
    assert rep.recommendations == []
    assert any("A5" in lim for lim in rep.limitations)
    # A10 was never asked in this session, so it must not appear as if it were
    assert all("A10" not in lim for lim in rep.limitations)


def test_report_limitations_disclose_pending_review_and_scope():
    ep = _full_episode()
    rep = build_support_report(episode_result=ep, result_id="sr-lim")
    joined = " ".join(rep.limitations)
    assert "not yet reviewed" in joined
    assert "not personalised" in joined
    assert "does not ask about sensory" in joined
    assert "Screening support only." in rep.disclaimer


def test_every_curated_strategy_is_reachable_or_explained():
    """Nothing in the triggerable library may be dead weight: either a
    reachable answer pattern fires it, or it is in the non-triggerable list
    with a reason."""
    from src.support.strategies import NON_TRIGGERABLE_STRATEGIES
    excluded = {s.recommendation_id for s in NON_TRIGGERABLE_STRATEGIES}
    ep = _full_episode()
    rep = build_support_report(episode_result=ep, result_id="sr-reach")
    reachable_titles = {r.title for r in rep.recommendations}
    for s in STRATEGIES:
        assert s.recommendation_id not in excluded
        assert s.title in reachable_titles, (
            f"{s.recommendation_id} is triggerable but no answer pattern in "
            f"this test fires it")
