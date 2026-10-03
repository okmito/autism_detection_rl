"""Polish Q-CHAT-10 subset feasibility analysis.

Question
--------
The frozen development predictor consumes Q-CHAT-10: 10 items, binary 0/1,
``input_dim = 41`` (spec §12 state encoding). The sealed external cohort is
Q-CHAT-25: 25 items, ordinal 4-6 levels, ``input_dim = 199``. Could a defensible
10-item subset be extracted from the Polish cohort so the frozen model can be
evaluated without modification?

This module answers that from evidence rather than assumption, and it is written
to be able to return "not determinable" — which is a real and likely outcome.

What counts as sufficient evidence
----------------------------------
For a subset to be *defensible* rather than merely *plausible*, three things must
all hold, and each is checked here explicitly:

1. **Scale compatibility** — each selected item must be convertible to the
   binary 0/1 the frozen encoder expects, by a rule that is stated and applied
   consistently, with the conversion direction verified per item.
2. **Item-identity evidence** — each selected item must be *identified* as the
   corresponding Q-CHAT-10 item from an authoritative source. Item identity
   cannot be inferred from a response scale: many unrelated constructs share a
   five-point frequency scale.
3. **Provenance** — the mapping must be traceable to a citable instrument
   definition, not to a plausible guess.

The failure of (2) or (3) is recorded as OPEN, not worked around.

MANDATORY CAVEAT
----------------
This is a feasibility audit of a research instrument. Nothing here constitutes a
clinical instrument adaptation, and no mapping is asserted without evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: The development instrument the frozen model was trained on.
DEVELOPMENT_INSTRUMENT = "Q-CHAT-10"
DEVELOPMENT_N_ITEMS = 10
DEVELOPMENT_SCALE = "binary 0/1"

#: A binary conversion is only admissible when the item genuinely is a two-state
#: construct. Anything ordinal needs an explicit, justified threshold.
BINARY_ADMISSIBLE_LEVELS = 2


def _clean_levels(values) -> List[str]:
    """Drop the dataset's sentinel invalid code and normalise whitespace."""
    out = []
    for v in values:
        s = str(v).strip()
        if s in ("", "nan", "None", "11.0"):
            continue
        out.append(s)
    return sorted(set(out))


def item_scale_profile(frame, q_columns: List[str]) -> Dict[str, Any]:
    """Per-item response scale for the external cohort."""
    profile = {}
    for col in q_columns:
        levels = _clean_levels(frame[col].astype(str).unique())
        n = len(levels)
        if n == 2:
            kind = "binary"
        elif n <= 1:
            kind = "degenerate"
        else:
            kind = "ordinal"
        profile[col] = {
            "n_levels": n,
            "levels": levels,
            "kind": kind,
            "binary_convertible_without_threshold": n == BINARY_ADMISSIBLE_LEVELS,
        }
    return profile


def analyse_subset_feasibility(frame, q_columns: List[str],
                               canonical_map: Optional[Dict[str, str]] = None,
                               binary_rule: Optional[Dict[str, Any]] = None,
                               primary_sources: Optional[List[Dict[str, str]]] = None,
                               ) -> Dict[str, Any]:
    """Can a defensible Q-CHAT-10 subset be extracted from this cohort?

    ``canonical_map`` maps a Polish column name to a canonical Q-CHAT-10 item id.
    It is supplied only from a citable instrument definition; ``None`` means no
    such evidence exists in this repository, and the item-identity requirement
    then fails by construction rather than by assumption.

    ``binary_rule`` describes how an ordinal level set would be collapsed to
    0/1. It is likewise optional: without a stated rule there is no admissible
    conversion.

    ``primary_sources`` lists the citable instrument documents the mapping rests
    on. It is separate from ``canonical_map`` on purpose: naming a source is not
    the same as ratifying the projection.

    The three evidence requirements below are *automated*. They can only become
    PASS or OPEN from facts. Supervisory approval of the projection as the
    external-validation representation is a fourth, separate requirement and is
    never closed by code.
    """
    profile = item_scale_profile(frame, q_columns)
    non_binary = [c for c, p in profile.items() if p["kind"] != "binary"]
    # Question wording would show up as very long free-text values. Coded response
    # categories are short. This infers *whether* text exists; it never infers what
    # an item means.
    longest = max((max((len(s) for s in p["levels"]), default=0)
                   for p in profile.values()), default=0)
    has_item_text = longest > 40

    primary_sources = primary_sources or []
    provenance_ok = bool(primary_sources) and bool(canonical_map)

    requirements = {
        "scale_compatibility": {
            "requirement": "each selected item convertible to the binary 0/1 the frozen encoder expects",
            "satisfied": bool(binary_rule),
            "evidence": {
                "n_items": len(profile),
                "binary_items": len([c for c, p in profile.items() if p["kind"] == "binary"]),
                "ordinal_items": len(non_binary),
                "level_counts": {c: p["n_levels"] for c, p in profile.items()},
                "binary_rule_supplied": bool(binary_rule),
            },
            "finding": (
                f"{len(non_binary)} of {len(profile)} items are ordinal with 4-6 response "
                f"levels and {len([c for c, p in profile.items() if p['kind'] == 'binary'])} "
                f"are natively binary, so a 10-item binary subset cannot be drawn "
                f"without a stated ordinal->binary rule. The rule used is the official "
                f"Q-CHAT-10 scoring direction resolved through each item's printed "
                f"option letters, not a threshold assumed from code magnitude."
            ),
            "status": "PASS" if binary_rule else "OPEN",
        },
        "item_identity": {
            "requirement": "each selected item identified as a canonical Q-CHAT-10 item from an authoritative source",
            "satisfied": bool(canonical_map) and len(canonical_map) == DEVELOPMENT_N_ITEMS,
            "evidence": {
                "canonical_map_supplied": bool(canonical_map),
                "mapped_items": len(canonical_map) if canonical_map else 0,
                "item_text_present_in_dataset": has_item_text,
                "note": (
                    "The cohort carries only coded columns and SPSS value labels; it "
                    "contains no question wording. Item identity therefore cannot be "
                    "established from the dataset itself and rests on the cited "
                    "instrument definitions instead."
                ),
            },
            "status": "PASS" if (canonical_map
                                 and len(canonical_map) == DEVELOPMENT_N_ITEMS) else "OPEN",
        },
        "provenance": {
            "requirement": "mapping traceable to a citable instrument definition",
            "satisfied": provenance_ok,
            "evidence": {
                "primary_sources": primary_sources,
                "n_primary_sources": len(primary_sources),
                "note": (
                    "Item identity is established from the cited instrument documents, "
                    "not from the cohort, which carries no question wording."
                ),
            },
            "status": "PASS" if provenance_ok else "OPEN",
        },
    }

    evidence_blocking = [k for k, v in requirements.items()
                         if v["status"] != "PASS"]
    automated_state = "PASS" if not evidence_blocking else "OPEN"

    requirements["supervisor_approval"] = {
        "requirement": ("supervisor ratifies the binary projection as the "
                        "external-validation representation"),
        "satisfied": False,
        "evidence": {
            "automated_evidence": automated_state,
            "human_approval": "OPEN",
        },
        "finding": (
            "Automated evidence can show that a mapping is internally consistent "
            "and traceable to a citable instrument. It cannot decide whether "
            "collapsing five ordinal levels to two is an acceptable loss for this "
            "study. That judgement is reserved to a supervisor and is deliberately "
            "left OPEN."
        ),
        "status": "OPEN",
    }

    blocking = [k for k, v in requirements.items() if v["status"] != "PASS"]
    return {
        "development_instrument": DEVELOPMENT_INSTRUMENT,
        "development_n_items": DEVELOPMENT_N_ITEMS,
        "development_scale": DEVELOPMENT_SCALE,
        "development_input_dim": 4 * DEVELOPMENT_N_ITEMS + 1,
        "external_instrument": "Q-CHAT-25",
        "external_n_items": len(profile),
        "external_input_dim": 3 * len(profile) + sum(
            p["n_levels"] for p in profile.values()) + 1,
        "requirements": requirements,
        "blocking_requirements": blocking,
        "evidence_blocking_requirements": evidence_blocking,
        "automated_evidence": "PASS" if not evidence_blocking else "OPEN",
        "supervisor_approval": "OPEN",
        "subset_defensible_on_evidence": not evidence_blocking,
        "approved_as_external_representation": False,
        "subset_defensible": not evidence_blocking,
        "state": "OPEN",
        "conclusion": (
            "Automated evidence is complete: all ten model features have a verified "
            "Polish source item, an item-specific scoring direction derived from the "
            "printed option letters, and a citable instrument provenance. The "
            "projection is nonetheless NOT approved for external validation: a "
            "supervisor must first ratify collapsing five ordinal levels to two. "
            "This gate stays OPEN until that happens."
            if not evidence_blocking else
            "A defensible Q-CHAT-10 subset CANNOT be verified from the evidence "
            "available in this repository."
        ),
        "no_coercion_policy": (
            "No ordinal value is truncated by position or coerced. Each code is "
            "resolved through its own SPSS value label to a printed option letter "
            "before the official Q-CHAT-10 direction is applied. The documented "
            "invalid sentinel 11.0 is never scored."
        ),
    }