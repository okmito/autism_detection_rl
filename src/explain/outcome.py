"""Outcome-level explanation for the final screening result — P3-outcome.

One entry point, :func:`explain_outcome`, turns the result of
``src.env.environment.run_episode`` into a structured, auditable explanation of
**why the system produced this screening result**:

1. what the outcome is (calibrated probability + thresholded referral decision),
2. which observed questionnaire responses moved it, ranked (exact group-Shapley,
   ``attribution.py``),
3. the supporting and opposing evidence,
4. counterfactual analysis (``counterfactual.py``),
5. the only uncertainty statement this architecture supports (prior
   sensitivity, ``uncertainty.py``),
6. limitations and the non-diagnostic disclaimer (``limitations.py``).

Design contract
---------------
* **Post-hoc and read-only.** Nothing here trains, mutates the predictor, or
  changes the episode. The same ``run_episode`` result and the same frozen
  predictor produce the same explanation (deterministic given the seed).
* **Explains the model, not the person.** Every attribution, counterfactual
  and uncertainty value is a statement about the model's output; the schema
  carries that framing explicitly, and the disclaimer is mandatory.
* **No fabrication.** Fields the architecture cannot support (e.g. an
  uncertainty interval for the v2 MLP) are reported as unavailable, never
  filled in.

Schema: ``SCHEMA_VERSION`` below; consumers (demo result screen, batch
artifact, tests) validate against the same keys.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.env.state import UNASKED, OBSERVED, MISSING
from src.explain.attribution import exact_shapley
from src.explain.counterfactual import find_counterfactual_rich
from src.explain.uncertainty import prior_sensitivity_band
from src.explain.limitations import build_limitations, screening_disclaimer

SCHEMA_VERSION = "outcome-explanation/1.0"

#: Decision vocabularies present in the repository. ``run_episode``
#: (src/env/environment.py) emits "REFER"; the demo's Session._finish
#: (scripts/demo_app.py) emits "REFERRAL_RECOMMENDED". Both are preserved; the
#: explanation layer compares semantic content, not string equality.
_REFERRAL_VOCAB = {"REFER", "REFERRAL_RECOMMENDED"}
_NO_REFERRAL_VOCAB = {"NO_REFERRAL_INDICATED"}

#: Tau is frozen at 0.5 by spec §10 and never tuned on an external cohort.
TAU_FROZEN = True
_TOL = 1e-12


def _model_block(predictor) -> Dict[str, Any]:
    meta = getattr(predictor, "metadata", {}) or {}
    if hasattr(predictor, "probability_with_prior"):
        method = "exact_marginalisation_factorised_prior"
    else:
        method = "masked_mlp_partial_state"
    return {
        "predictor_version": str(meta.get("predictor_version",
                                          type(predictor).__name__)),
        "calibration": str(meta.get("calibration_method", "unknown")),
        "trained_on": str(meta.get("trained_on", "unknown")),
        "external_cohort_used": meta.get("external_cohort_used"),
        "partial_state_method": method,
    }


def _evidence_block(episode_result: Dict[str, Any], state: Dict[str, Any],
                    attributions) -> Dict[str, Any]:
    trace = episode_result.get("trace") or []
    observed: List[Dict[str, Any]] = []
    if trace:
        for t in trace:
            if "item" not in t:
                continue
            observed.append({
                "item": t["item"], "item_idx": t.get("item_idx"),
                "response": t.get("value"),
                "belief_before": t.get("belief_before"),
                "belief_after": t.get("belief_after"),
            })
    else:  # fall back to the state itself
        mask = np.asarray(state["mask"], dtype=int)
        value = np.asarray(state["value"], dtype=int)
        for j in range(mask.size):
            if mask[j] == OBSERVED:
                observed.append({"item": f"A{j + 1}", "item_idx": int(j),
                                 "response": int(value[j])})
    supporting = [a.item for a in attributions if a.phi > _TOL]
    opposing = [a.item for a in attributions if a.phi < -_TOL]
    mask = np.asarray(state["mask"], dtype=int)
    n = int(mask.size)
    return {
        "observed_responses": observed,
        "supporting": supporting,
        "opposing": opposing,
        "unobserved_items_marginalised": [f"A{j + 1}" for j in range(n)
                                          if mask[j] == UNASKED],
        "missing_items": [f"A{j + 1}" for j in range(n) if mask[j] == MISSING],
        "note": ("supporting/opposing are the sign of each response's Shapley "
                 "value on the probability scale, relative to the model's "
                 "prior-only estimate; they are model attributions, not "
                 "clinical evidence"),
    }


def explain_outcome(
    episode_result: Dict[str, Any],
    predictor,
    *,
    tau: float = 0.5,
    costs=None,
    prior=None,
    reference_rows=None,
    dataset_tag: Optional[str] = None,
    circularity: Optional[str] = None,
    label_source: Optional[str] = None,
    budget: Optional[int] = None,
    tag: Optional[str] = None,
    uncertainty_draws: int = 200,
    uncertainty_seed: int = 0,
) -> Dict[str, Any]:
    """Build the structured explanation for one screening outcome.

    Parameters
    ----------
    episode_result:
        The dict returned by ``run_episode`` (``final_state``, ``p_hat``,
        ``decision``, ``trace``, ``items_asked``, ``stop_reason``).
    predictor:
        The same frozen predictor that produced ``p_hat``.
    tau:
        The operating threshold the decision was made at. Must match the
        threshold used for the episode; a mismatch is reported in
        ``warnings`` rather than silently reconciled.
    costs, prior, reference_rows:
        Optional inputs for counterfactual tie-breaks, feasibility scoring and
        seen-in-training flags.
    dataset_tag, circularity, label_source, budget, tag:
        Context for the limitations block and artifact tagging.
    """
    state = episode_result.get("final_state")
    if state is None:
        raise ValueError("episode_result has no final_state; run_episode output "
                         "is required to explain the outcome")
    warnings: List[str] = []

    p_hat = float(episode_result.get("p_hat", float(predictor(state))))
    episode_decision = episode_result.get("decision")
    # The repository carries two decision vocabularies: run_episode emits
    # "REFER" (environment.py) while the demo emits "REFERRAL_RECOMMENDED"
    # (demo_app.py). Neither is changed here; the explanation compares the
    # semantic content (referral vs not) and keeps the schema's canonical
    # wording alongside the string the episode actually reported.
    recomputed_flag = bool(p_hat >= tau)
    recomputed = ("REFERRAL_RECOMMENDED" if recomputed_flag
                  else "NO_REFERRAL_INDICATED")
    if episode_decision is not None:
        if episode_decision in _REFERRAL_VOCAB:
            episode_flag = True
        elif episode_decision in _NO_REFERRAL_VOCAB:
            episode_flag = False
        else:
            episode_flag = None
            warnings.append(
                f"unrecognised decision string '{episode_decision}'; could not "
                f"cross-check against the recomputed decision at tau={tau}")
        if episode_flag is not None and episode_flag != recomputed_flag:
            warnings.append(
                f"decision mismatch: episode reported '{episode_decision}' but "
                f"p_hat={p_hat:.6f} against tau={tau} implies "
                f"'{recomputed}'; the explanation uses tau={tau}")

    attribution = exact_shapley(state, predictor)
    counterfactuals = find_counterfactual_rich(
        state, predictor, tau=tau, costs=costs, prior=prior,
        reference_rows=reference_rows)
    uncertainty = prior_sensitivity_band(
        state, predictor, prior=prior, n_draws=uncertainty_draws,
        seed=uncertainty_seed)
    limitations = build_limitations(
        state, predictor, stop_reason=episode_result.get("stop_reason"),
        budget=budget, dataset_tag=dataset_tag, circularity=circularity,
        label_source=label_source)

    mask = np.asarray(state["mask"], dtype=int)
    explanation: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "outcome": {
            "decision": recomputed,
            "decision_as_reported_by_episode": episode_decision,
            "p_hat": p_hat,
            "tau": float(tau),
            "tau_frozen": TAU_FROZEN,
            "stop_reason": episode_result.get("stop_reason"),
            "n_questions_asked": int((mask == OBSERVED).sum()),
            "budget": budget,
            "items_asked": list(episode_result.get("items_asked", [])),
        },
        "model": _model_block(predictor),
        "contributions": [a.as_dict() for a in attribution.ranked()],
        "attribution": {
            "method": attribution.value_function,
            "baseline_p": float(attribution.baseline_p),
            "full_p": float(attribution.full_p),
            "n_attributed": int(attribution.n_attributed),
            "n_evaluations": int(attribution.n_evaluations),
            "additivity_error": float(attribution.additivity_error),
        },
        "evidence": _evidence_block(episode_result, state, attribution.items),
        "counterfactuals": counterfactuals,
        "uncertainty": uncertainty,
        "limitations": limitations,
        "disclaimer": screening_disclaimer(),
        "warnings": warnings,
    }
    return _json_safe(explanation)


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python types.

    The explanation object is written to JSON artifacts and served by the demo
    API; numpy scalars (np.float64, np.int64, np.bool_) are not serialisable by
    the stdlib json module, so the layer that produces the object guarantees
    its own serialisability instead of pushing that onto every consumer.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj
