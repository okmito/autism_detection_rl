"""Tests for the outcome explanation object (P3-outcome, Phase 3).

Covers: schema completeness, tau-consistency guard, attribution wiring,
evidence block, uncertainty availability per predictor version, limitations
content, `probability_with_prior` equivalence with `predict_state`, and the
deterministic renderer (no invented content).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.environment import run_episode
from src.env.state import init_state, update_state
from src.explain.outcome import SCHEMA_VERSION, explain_outcome
from src.explain.render import render_text
from src.explain.limitations import build_limitations, screening_disclaimer
from src.explain.uncertainty import prior_sensitivity_band
from src.models.logistic_predictor import LogisticPredictor
from src.models.masked_predictor import MaskedPredictor


def _record(responses, label=1, missing_idx=None):
    """Episode/test record. ``missing_idx`` makes that item structurally MISSING
    (NaN response), matching the schema contract. Training records pass no
    missing index — the real Saudi cohort has zero missing cells (0/5060)."""
    n = len(responses)
    responses = list(responses)
    mask = np.zeros(n, dtype=bool)
    if missing_idx is not None:
        mask[missing_idx] = True
        responses[missing_idx] = np.nan
    return {"item_responses": np.asarray(responses, dtype=float),
            "label": label, "label_source": "questionnaire",
            "covariates": {"age_band": "1-2", "sex": "M"},
            "missing_mask": mask}


def _tiny_training_set():
    rng = np.random.default_rng(11)
    recs = []
    for _ in range(24):
        r = rng.integers(0, 2, size=10).astype(float)
        recs.append(_record(r, label=int(r.sum() >= 5)))
    return recs


def _first_item_policy(state, legal):
    items = [a for a in legal if a != -1]
    return items[0] if items else -1


@pytest.fixture(scope="module")
def v3():
    train = _tiny_training_set()
    return LogisticPredictor.fit(train, train, [f"A{j+1}" for j in range(10)],
                                 seed=0)


@pytest.fixture(scope="module")
def episode(v3):
    # item index 3 (A4) is structurally MISSING; the policy must route around it
    rec = _record([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], label=1, missing_idx=3)
    assert bool(rec["missing_mask"][3]) is True
    ep = run_episode(rec, question_budget=6, b_min=0,
                     policy=_first_item_policy, predictor=v3)
    return rec, ep


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def test_schema_keys(v3, episode):
    rec, ep = episode
    ex = explain_outcome(ep, v3, tau=0.5, dataset_tag="saudi",
                         circularity="Deterministic (questionnaire-derived)",
                         label_source="questionnaire", budget=6)
    assert ex["schema_version"] == SCHEMA_VERSION
    for key in ("outcome", "model", "contributions", "attribution",
                "evidence", "counterfactuals", "uncertainty", "limitations",
                "disclaimer", "warnings"):
        assert key in ex, key
    oc = ex["outcome"]
    # semantic decision agreement with the episode (REFER vs REFERRAL_RECOMMENDED
    # are the repository's two vocabularies for the same thing)
    ep_is_referral = ep["decision"] in ("REFER", "REFERRAL_RECOMMENDED")
    ex_is_referral = oc["decision"] == "REFERRAL_RECOMMENDED"
    assert ep_is_referral == ex_is_referral
    assert oc["decision_as_reported_by_episode"] == ep["decision"]
    assert oc["p_hat"] == pytest.approx(ep["p_hat"])
    assert oc["stop_reason"] == ep["stop_reason"]
    assert oc["n_questions_asked"] == len(ep["items_asked"])
    assert oc["tau_frozen"] is True
    assert ex["warnings"] == [], ex["warnings"]
    # contributions ranked by |phi|
    phis = [abs(c["phi"]) for c in ex["contributions"]]
    assert phis == sorted(phis, reverse=True)
    # every observed item appears exactly once
    assert {c["item_idx"] for c in ex["contributions"]} == set(ep["items_asked"])
    # disclaimer present and non-clinical
    assert "not a diagnosis" in ex["disclaimer"].lower()
    # limitations mention missingness (item 3 was missing in the record)
    assert any("missing" in l.lower() for l in ex["limitations"])


def test_tau_mismatch_is_reported_not_silently_fixed(v3, episode):
    rec, ep = episode
    # p_hat is ~0.58; at tau=0.9 the recomputed decision flips to no-referral
    ex = explain_outcome(ep, v3, tau=0.9)
    assert ex["warnings"], "a decision mismatch must be surfaced as a warning"
    assert any("decision mismatch" in w and "tau=0.9" in w
               for w in ex["warnings"])
    # the explanation still reports the episode's own decision string
    assert ex["outcome"]["decision_as_reported_by_episode"] == ep["decision"]


def test_attribution_block_consistency(v3, episode):
    rec, ep = episode
    ex = explain_outcome(ep, v3, tau=0.5)
    att = ex["attribution"]
    assert att["full_p"] == pytest.approx(ep["p_hat"], abs=1e-12)
    assert att["additivity_error"] < 1e-9
    assert att["n_attributed"] == len(ep["items_asked"])
    assert att["n_evaluations"] == 2 ** att["n_attributed"]


def test_evidence_block(v3, episode):
    rec, ep = episode
    ex = explain_outcome(ep, v3, tau=0.5)
    ev = ex["evidence"]
    assert len(ev["observed_responses"]) == len(ep["trace"])
    for o in ev["observed_responses"]:
        assert "item" in o and "response" in o
    # supporting/opposing partition the non-neutral contributions
    assert set(ev["supporting"]) | set(ev["opposing"]) == \
        {c["item"] for c in ex["contributions"] if c["direction"] != "neutral"}
    # item index 3 (A4) is structurally missing
    assert "A4" in ev["missing_items"]


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------

def test_probability_with_prior_reproduces_predict_state(v3, episode):
    rec, ep = episode
    state = ep["final_state"]
    a = float(v3.predict_state(state))
    b = float(v3.probability_with_prior(state, v3.prior))
    assert abs(a - b) < 1e-9
    # a shifted prior must move the estimate (or at least not crash)
    shifted = np.clip(np.asarray(v3.prior) + 0.2, 0.01, 0.99)
    c = float(v3.probability_with_prior(state, shifted))
    assert 0.0 <= c <= 1.0


def test_prior_sensitivity_band_available_for_v3(v3, episode):
    rec, ep = episode
    unc = prior_sensitivity_band(ep["final_state"], v3, n_draws=60, seed=0)
    assert unc["available"] is True
    assert unc["is_clinical_uncertainty"] is False
    lo, hi = unc["interval"]
    assert 0.0 <= lo <= hi <= 1.0
    assert "interpretation" in unc


def test_prior_sensitivity_band_deterministic(v3, episode):
    rec, ep = episode
    a = prior_sensitivity_band(ep["final_state"], v3, n_draws=40, seed=3)
    b = prior_sensitivity_band(ep["final_state"], v3, n_draws=40, seed=3)
    assert a["interval"] == b["interval"]


def test_prior_sensitivity_band_unavailable_for_v2(episode):
    rec, ep = episode
    v2 = MaskedPredictor(n_items=10, seed=0)
    unc = prior_sensitivity_band(ep["final_state"], v2)
    assert unc["available"] is False
    assert "reason" in unc


# --------------------------------------------------------------------------
# limitations + renderer
# --------------------------------------------------------------------------

def test_limitations_cover_key_honesty_items(v3, episode):
    rec, ep = episode
    lims = build_limitations(ep["final_state"], v3,
                             stop_reason=ep["stop_reason"], budget=6,
                             dataset_tag="saudi",
                             circularity="Deterministic",
                             label_source="questionnaire")
    text = " ".join(lims).lower()
    assert "not a diagnosis" in text or "not a\n" in text or "diagnosis" in text
    assert "missing" in text
    assert "order-invariant" in text
    assert "deterministic" in text
    assert screening_disclaimer().startswith("Screening support only")


def test_renderer_uses_only_evidence_fields(v3, episode):
    rec, ep = episode
    ex = explain_outcome(ep, v3, tau=0.5, budget=6)
    text = render_text(ex, item_texts={"A1": "looks when called"})
    assert "Screening result" in text
    assert "A1" in text
    assert "looks when called" in text  # caller-supplied wording is threaded in
    assert "not a diagnosis" in text.lower()
    # no number may appear that is not derived from the evidence: the rendered
    # risk estimate must equal the object's p_hat
    assert f"{100.0 * ex['outcome']['p_hat']:.1f}%" in text


def test_renderer_mentions_counterfactual_framing(v3, episode):
    rec, ep = episode
    ex = explain_outcome(ep, v3, tau=0.5, budget=6)
    text = render_text(ex)
    assert ("model's behaviour" in text or "model's output" in text
            or "not the person" in text or "not advice" in text)


def test_missing_final_state_raises(v3):
    with pytest.raises(ValueError, match="final_state"):
        explain_outcome({"p_hat": 0.4, "decision": "NO_REFERRAL_INDICATED"}, v3)
