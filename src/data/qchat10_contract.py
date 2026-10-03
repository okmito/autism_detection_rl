"""Canonical Q-CHAT-10 -> Polish Q-CHAT-25 feature contract.

Single source of truth for mapping Polish Q-CHAT-25 questionnaire responses onto
the ten binary features consumed by the frozen Saudi predictor. No other module
may hard-code this mapping.

Provenance of every fact in this file
-------------------------------------
* **Item mapping** (``CANONICAL_MAPPING``): Q-CHAT-10 (Autism Research Centre) item
  order matched against the original 25-item Q-CHAT instrument. The 25-item
  instrument shipped at ``data/raw/Q-CHAT Polish/QCHAT.pdf`` is the local copy.
* **Printed option order** (``QCHAT25_PRINTED_OPTIONS_EN``): transcribed from the
  option lists printed under each question in that same PDF. Order matters because
  the Q-CHAT-10 scoring rules are stated in terms of the printed letters A-E.
* **SPSS codes and labels** (``ORDINAL_VERIFICATION[...]["labels"]``): read from the
  value-label dictionary of ``data/QCHAT_dataset2 mendeley.sav``. These are *codes*,
  which for some items run opposite to the printed order (see ``qchat25recode``).
* **Scoring direction** (``QCHAT10_SCORED_LETTERS``): the official Q-CHAT-10 rule.
  Items 1-9 score C/D/E as 1; item 10 scores A/B/C as 1.

How the binary value is produced
--------------------------------
Not by thresholding the code. For each code we take its SPSS label, translate the
label to the English wording printed in the instrument, find that wording's
position in the printed option list to recover the printed letter, then apply the
official direction for that Q-CHAT-10 item. That derivation is the answer.

The simpler rule ``code >= 2`` happens to agree for all ten items, but it is a
*consequence* of the label/letter derivation, not the assumption behind it. This
matters: ``qchat2recode`` is coded ``0,1,2,3,5`` (code 4 never observed) and
``qchat25recode`` runs 0=never .. 4=many times a day, i.e. opposite to its printed
order. A code-magnitude assumption would be unjustified. ``verify_split_rule()``
proves the two agree and is exercised by the test suite.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CANONICAL_MAPPING",
    "ContractError",
    "FEATURE_CONTRACT_VERSION",
    "MODEL_FEATURE_ORDER",
    "ORDINAL_SPLIT",
    "ORDINAL_VERIFICATION",
    "POLISH_INVALID_SENTINEL",
    "QCHAT10_SCORED_LETTERS",
    "QCHAT10_SCORING_DIRECTION",
    "QCHAT25_PRINTED_OPTIONS_EN",
    "QCHAT_MAPPING_VERSION",
    "binary_for_letter",
    "build_feature_matrix",
    "code_to_binary",
    "contract_completeness",
    "describe_contract",
    "encode_qchat10_features",
    "letter_for_code",
    "polish_var_for_qchat10_item",
    "qchat25_item_for_qchat10_item",
    "response_to_feature",
    "scoring_table",
    "verify_split_rule",
]

# The 25-item questionnaire records this value for an item that was not answered.
# It is a missing marker, not a scale point, and must never receive a score.
POLISH_INVALID_SENTINEL = 11.0

#: Version of this module's mapping/projection contract. Bump whenever the item
#: mapping, the scoring derivation, or the encoding changes in any way. Recorded in
#: the reproducibility artifact so a pre-validation representation can always be
#: tied back to the exact contract that produced it.
FEATURE_CONTRACT_VERSION = "qchat10-contract/2.0.0"

#: Version of the canonical item mapping alone, independent of any code change.
#: The 2.x series replaced the superseded Q1-Q9 mapping that was shifted after Q2.
QCHAT_MAPPING_VERSION = "qchat10-mapping/2.0.0"

# --------------------------------------------------------------------------
# 1. The canonical 10-item mapping (the single definition in the project).
#    Key = Q-CHAT-10 item. Value = source item in the 25-item Q-CHAT.
# --------------------------------------------------------------------------
CANONICAL_MAPPING: Dict[int, int] = {
    1: 1,
    2: 2,
    3: 5,
    4: 6,
    5: 9,
    6: 10,
    7: 15,
    8: 17,
    9: 19,
    10: 25,
}

# --------------------------------------------------------------------------
# 2. Official Q-CHAT-10 scoring. Stated over printed option letters, not codes.
# --------------------------------------------------------------------------
QCHAT10_SCORING_DIRECTION: Dict[int, str] = {
    **{item: "C/D/E scores 1" for item in range(1, 10)},
    10: "A/B/C scores 1",
}

QCHAT10_SCORED_LETTERS: Dict[int, frozenset] = {
    **{item: frozenset("CDE") for item in range(1, 10)},
    10: frozenset("ABC"),
}

# --------------------------------------------------------------------------
# 3. Printed option order per 25-item question, transcribed from QCHAT.pdf.
#    Index 0 is the printed letter "A".
# --------------------------------------------------------------------------
_FREQUENCY_EN = (
    "many times a day",
    "a few times a day",
    "a few times a week",
    "less than once a week",
    "never",
)
_AGRREEMENT_EN = ("always", "usually", "sometimes", "rarely", "never")

QCHAT25_PRINTED_OPTIONS_EN: Dict[int, Tuple[str, ...]] = {
    1: _AGRREEMENT_EN,
    2: ("very easy", "quite easy", "quite difficult", "very difficult",
        "impossible"),
    5: _FREQUENCY_EN,
    6: _FREQUENCY_EN,
    9: _FREQUENCY_EN,
    10: _FREQUENCY_EN,
    15: _AGRREEMENT_EN,
    17: ("very typical", "quite typical", "slightly unusual", "very unusual",
         "my child does not speak"),
    19: _FREQUENCY_EN,
    25: _FREQUENCY_EN,
}

# --------------------------------------------------------------------------
# 4. Polish value label -> the English wording printed in the instrument.
#    This is the bridge that lets a numeric code be resolved to a printed letter.
# --------------------------------------------------------------------------
POLISH_LABEL_EN: Dict[str, str] = {
    "zawsze": "always",
    "zazwyczaj": "usually",
    "czasami": "sometimes",
    "rzadko": "rarely",
    "nigdy": "never",
    "b.łatwo": "very easy",
    "dość łatwo": "quite easy",
    "dość trudno": "quite difficult",
    "b. trudno": "very difficult",
    "niemożliwe": "impossible",
    "wiele razy/dzień": "many times a day",
    "kilka razy/dzień": "a few times a day",
    "kilka razy/tydzień": "a few times a week",
    "mniej niż raz/tydzień": "less than once a week",
    "b.typowe": "very typical",
    "całkiem typowe": "quite typical",
    "trochę niezwykłe": "slightly unusual",
    "b.niezwykłe": "very unusual",
    "nie mówi": "my child does not speak",
}

# --------------------------------------------------------------------------
# 5. Per-source-item verification records. Labels are verbatim SPSS strings.
# --------------------------------------------------------------------------
_FREQ_LABELS = {
    0: "wiele razy/dzień",
    1: "kilka razy/dzień",
    2: "kilka razy/tydzień",
    3: "mniej niż raz/tydzień",
    4: "nigdy",
}
_AGREE_LABELS = {
    0: "zawsze",
    1: "zazwyczaj",
    2: "czasami",
    3: "rzadko",
    4: "nigdy",
}

# Construct note per item, used to sanity-check direction by meaning.
ORDINAL_VERIFICATION: Dict[int, Dict[str, Any]] = {
    1: {
        "polish_var": "qchat1recode",
        "labels": dict(_AGREE_LABELS),
        "construct_en": "looks at you when you call his/her name",
        "typical_end": "always",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q1 options A-E = always..never; SPSS codes 0-4 follow "
                    "the same order.",
    },
    2: {
        "polish_var": "qchat2recode",
        "labels": {0: "b.łatwo", 1: "dość łatwo", 2: "dość trudno",
                   3: "b. trudno", 5: "niemożliwe"},
        "construct_en": "how easy it is to get eye contact",
        "typical_end": "very easy",
        "atypical_end": "impossible",
        "evidence": "QCHAT.pdf Q2 options A-E = very easy..impossible; SPSS codes are "
                    "0,1,2,3,5 - code 4 is NOT used, so codes must never be assumed "
                    "contiguous.",
    },
    5: {
        "polish_var": "qchat5recode",
        "labels": dict(_FREQ_LABELS),
        "construct_en": "points to indicate that s/he wants something",
        "typical_end": "many times a day",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q5 frequency options; SPSS codes 0-4 follow the same "
                    "order.",
    },
    6: {
        "polish_var": "qchat6recode",
        "labels": dict(_FREQ_LABELS),
        "construct_en": "points to share interest with you",
        "typical_end": "many times a day",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q6 frequency options; SPSS codes 0-4 follow the same "
                    "order.",
    },
    9: {
        "polish_var": "qchat9recode",
        "labels": dict(_FREQ_LABELS),
        "construct_en": "pretends (e.g. cares for dolls, talks on a toy phone)",
        "typical_end": "many times a day",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q9 frequency options; SPSS codes 0-4 follow the same "
                    "order.",
    },
    10: {
        "polish_var": "qchat10recode",
        "labels": dict(_FREQ_LABELS),
        "construct_en": "follows where you're looking",
        "typical_end": "many times a day",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q10 frequency options; SPSS codes 0-4 follow the same "
                    "order.",
    },
    15: {
        "polish_var": "qchat15recode",
        "labels": dict(_AGREE_LABELS),
        "construct_en": "shows signs of wanting to comfort a visibly upset person",
        "typical_end": "always",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q15 options A-E = always..never; SPSS codes 0-4 follow "
                    "the same order.",
    },
    17: {
        "polish_var": "qchat17recode",
        "labels": {0: "b.typowe", 1: "całkiem typowe", 2: "trochę niezwykłe",
                   3: "b.niezwykłe", 4: "nie mówi"},
        "construct_en": "how typical the child's first words were",
        "typical_end": "very typical",
        "atypical_end": "very unusual",
        "evidence": "QCHAT.pdf Q17 options A-E = very typical..does not speak; SPSS "
                    "codes 0-4 follow the same order.",
    },
    19: {
        "polish_var": "qchat19recode",
        "labels": dict(_FREQ_LABELS),
        "construct_en": "uses simple gestures",
        "typical_end": "many times a day",
        "atypical_end": "never",
        "evidence": "QCHAT.pdf Q19 frequency options; SPSS codes 0-4 follow the same "
                    "order.",
    },
    25: {
        "polish_var": "qchat25recode",
        "labels": {0: "nigdy", 1: "mniej niż raz/tydzień",
                   2: "kilka razy/tydzień", 3: "kilka razy/dzień",
                   4: "wiele razy/dzień"},
        "construct_en": "stares at nothing with no apparent purpose",
        "typical_end": "never",
        "atypical_end": "many times a day",
        "evidence": "QCHAT.pdf Q25 printed options run many-times-a-day (A) down to "
                    "never (E), but the SPSS codes run the OPPOSITE way: 0=never, "
                    "4=many times a day. More frequent staring therefore carries a "
                    "HIGHER code. This is the only reverse-coded item and it is the "
                    "target of Q-CHAT-10 item 10.",
    },
}

# The verified consequence of the label/letter derivation below. Not the premise.
ORDINAL_SPLIT = 2

# --------------------------------------------------------------------------
# 6. Explicit chain: Q10 item -> canonical number -> Q25 item -> variable ->
#    model feature index -> scoring direction.
# --------------------------------------------------------------------------
MODEL_FEATURE_ORDER: Tuple[Dict[str, Any], ...] = tuple(
    {
        "model_feature": f"feature_{item}",
        "feature_index": item,
        "saudi_column": f"A{item}",
        "qchat10_item": item,
        "qchat25_item": CANONICAL_MAPPING[item],
        "polish_var": ORDINAL_VERIFICATION[CANONICAL_MAPPING[item]]["polish_var"],
        "scoring_direction": QCHAT10_SCORING_DIRECTION[item],
    }
    for item in sorted(CANONICAL_MAPPING)
)

_FEATURE_BY_ITEM: Dict[int, Dict[str, Any]] = {e["qchat10_item"]: e
                                               for e in MODEL_FEATURE_ORDER}
_VAR_TO_ENTRY: Dict[str, Dict[str, Any]] = {e["polish_var"]: e
                                            for e in MODEL_FEATURE_ORDER}


class ContractError(ValueError):
    """The requested conversion is not defined by the verified contract."""


def _lookup_for(polish_var: str) -> Dict[float, str]:
    """SPSS code -> verbatim value label for one Polish item."""
    for record in ORDINAL_VERIFICATION.values():
        if record["polish_var"] == polish_var:
            return {float(code): label
                    for code, label in sorted(record["labels"].items())}
    raise ContractError(
        f"{polish_var!r} is not part of the verified Q-CHAT-10 feature contract")


def qchat25_item_for_qchat10_item(item: int) -> int:
    try:
        return CANONICAL_MAPPING[item]
    except KeyError:
        raise ContractError(f"no Q-CHAT-10 item {item}") from None


def polish_var_for_qchat10_item(item: int) -> str:
    return _FEATURE_BY_ITEM[item]["polish_var"]


def letter_for_code(polish_var: str, code: float) -> str:
    """Resolve an SPSS code to its printed option letter A-E.

    code -> Polish label -> English wording printed in the instrument -> position
    in that item's printed option list. Deliberately does not use the code's
    magnitude, because some items are reverse-coded relative to print order.
    """
    entry = _VAR_TO_ENTRY.get(polish_var)
    if entry is None:
        raise ContractError(
            f"{polish_var!r} is not part of the verified Q-CHAT-10 feature contract")
    qchat25_item = entry["qchat25_item"]

    lookup = _lookup_for(polish_var)
    key = _normalise_code(code, lookup, polish_var)
    if key is None:
        raise ContractError(f"missing response for {polish_var!r}")

    label = lookup[key]
    english = POLISH_LABEL_EN.get(label)
    if english is None:
        raise ContractError(
            f"value label {label!r} of {polish_var!r} has no entry in "
            "POLISH_LABEL_EN, so the printed letter cannot be resolved")
    options = QCHAT25_PRINTED_OPTIONS_EN[qchat25_item]
    if english not in options:
        raise ContractError(
            f"label {label!r} ({english!r}) of {polish_var!r} is not one of the "
            f"printed options for Q-CHAT-25 Q{qchat25_item}: {options}")
    return "ABCDE"[options.index(english)]


def binary_for_letter(letter: str, qchat10_item: int) -> int:
    """Apply the official Q-CHAT-10 scoring rule for this item."""
    try:
        scored = QCHAT10_SCORED_LETTERS[qchat10_item]
    except KeyError:
        raise ContractError(f"no scoring rule for Q-CHAT-10 item {qchat10_item}") \
            from None
    if letter not in "ABCDE":
        raise ContractError(f"invalid printed letter {letter!r}")
    return 1 if letter in scored else 0


def label_to_code(polish_var: str, label: str) -> float:
    """Resolve a verbatim Polish value label back to its SPSS code.

    The integrated Polish CSV stores the label text, not the numeric code, so
    the contract accepts either representation.
    """
    for code, text in _lookup_for(polish_var).items():
        if text == label:
            return code
    raise ContractError(
        f"value label {label!r} is not observed for {polish_var!r}; observed "
        f"labels are {sorted(_lookup_for(polish_var).values())}")


def _normalise_code(code: Any, lookup: Mapping[float, str],
                    polish_var: str) -> Optional[float]:
    if code is None:
        return None
    if isinstance(code, str):
        stripped = code.strip()
        if stripped == "" or stripped.lower() in {"nan", "none"}:
            return None
        try:
            code = float(stripped)
        except ValueError:
            return label_to_code(polish_var, stripped)
    if isinstance(code, bool) or not isinstance(code, (int, float)):
        raise ContractError(
            f"non-numeric response {code!r} for {polish_var!r}; the Polish "
            "questionnaire is categorical")
    value = float(code)
    if value != value:  # NaN
        return None
    if value == POLISH_INVALID_SENTINEL:
        raise ContractError(
            f"invalid sentinel {POLISH_INVALID_SENTINEL!r} for {polish_var!r} means "
            "the item was not answered; it must stay missing and never be scored")
    if value not in lookup:
        raise ContractError(
            f"code {value:g} for {polish_var!r} is outside the item's observed "
            f"codes {sorted(lookup)}")
    return value


def code_to_binary(code: Any, polish_var: str) -> Optional[float]:
    """SPSS response code -> the frozen model's binary value, or None if missing.

    Derived through the printed letter so that reverse-coded and non-contiguous
    items cannot be mis-scored.
    """
    entry = _VAR_TO_ENTRY.get(polish_var)
    if entry is None:
        raise ContractError(
            f"{polish_var!r} is not part of the verified Q-CHAT-10 feature contract")
    key = _normalise_code(code, _lookup_for(polish_var), polish_var)
    if key is None:
        return None
    letter = letter_for_code(polish_var, key)
    return float(binary_for_letter(letter, entry["qchat10_item"]))


def response_to_feature(responses: Sequence[Any], polish_var: str,
                        allow_missing: bool = True) -> List[float]:
    """Vectorise :func:`code_to_binary` over several responses."""
    out: List[float] = []
    for code in responses:
        value = code_to_binary(code, polish_var)
        if value is None:
            if not allow_missing:
                raise ContractError(
                    f"missing response for {polish_var!r} is not permitted here")
            out.append(float("nan"))
        else:
            out.append(value)
    return out


def verify_split_rule() -> Dict[str, Any]:
    """Prove the letter derivation agrees with ``code >= ORDINAL_SPLIT``.

    Returns a per-item report. Raises :class:`ContractError` on any disagreement,
    which is what turns "it happens to work" into a checked invariant.
    """
    report: Dict[str, Any] = {}
    for entry in MODEL_FEATURE_ORDER:
        polish_var = entry["polish_var"]
        lookup = _lookup_for(polish_var)
        per_code = {}
        for code in sorted(lookup):
            derived = code_to_binary(code, polish_var)
            naive = 1.0 if code >= ORDINAL_SPLIT else 0.0
            if derived != naive:
                raise ContractError(
                    f"letter derivation and code>={ORDINAL_SPLIT} disagree for "
                    f"{polish_var!r} code {code:g}: derived {derived}, naive {naive}")
            per_code[code] = {
                "label": lookup[code],
                "letter": letter_for_code(polish_var, code),
                "binary": derived,
            }
        report[polish_var] = {
            "qchat10_item": entry["qchat10_item"],
            "qchat25_item": entry["qchat25_item"],
            "model_feature": entry["model_feature"],
            "scoring_direction": entry["scoring_direction"],
            "codes": per_code,
        }
    return report


def build_feature_matrix(
        table: Mapping[str, Sequence[Any]],
        allow_missing: bool = False) -> List[List[float]]:
    """Build ``(n_rows, 10)`` binary features in Q1..Q10 order.

    Every one of the ten verified variables must be present. A short vector is
    never emitted: the frozen predictor has a fixed input width, so silently
    padding or dropping a position would corrupt the features without any error.
    """
    completeness = contract_completeness()
    if not completeness["complete"]:
        raise ContractError(
            "Q-CHAT-10 feature contract is incomplete; missing feature index "
            f"{completeness['missing_feature_indices']}")

    required = [entry["polish_var"] for entry in MODEL_FEATURE_ORDER]
    absent = [var for var in required if var not in table]
    if absent:
        raise ContractError(
            f"Polish row is missing required variable(s) {absent}; the vector must "
            "have exactly 10 positions with no padding")

    lengths = {len(table[var]) for var in required}
    if len(lengths) != 1:
        raise ContractError(f"inconsistent row lengths across items: {lengths}")
    n_rows = lengths.pop()

    rows: List[List[float]] = []
    for i in range(n_rows):
        row: List[float] = []
        for entry in MODEL_FEATURE_ORDER:
            value = code_to_binary(table[entry["polish_var"]][i],
                                   entry["polish_var"])
            if value is None:
                if not allow_missing:
                    raise ContractError(
                        f"missing response for {entry['polish_var']!r} (feature "
                        f"{entry['model_feature']}) is not permitted")
                row.append(float("nan"))
            else:
                row.append(value)
        rows.append(row)
    return rows


def encode_qchat10_features(features: Sequence[float]) -> Any:
    """Project the ten binary features into the frozen model's network input.

    Uses the project's own ``init_state`` / ``update_state`` so the encoding
    convention is identical to the one used at training and scoring time rather
    than hand-rolled here. A fully answered questionnaire gives mask ``OBSERVED``
    for all ten items.

    ``questions_remaining = 0`` and ``budget = n`` reproduce the terminal
    fully-observed state that produced the Saudi research metrics
    (``scripts/step2_train_and_sweep.py::_terminal_probs``). Because
    ``encode_state`` computes ``questions_remaining / max(budget, 1)``, the
    normalised budget term is 0.0 either way here, but the values are set to the
    scoring convention so the two paths cannot silently diverge.

    For a binary instrument ``encode_state`` emits ``4n + 1 = 41`` dimensions:
    ``3n`` mask one-hot + ``n`` response bits + 1 normalised budget term. This is
    a shape check on the frozen interface; it does not alter the architecture.
    """
    import numpy as np

    from src.env.state import init_state, update_state

    values = list(features)
    if len(values) != len(MODEL_FEATURE_ORDER):
        raise ContractError(
            f"expected exactly {len(MODEL_FEATURE_ORDER)} features, got "
            f"{len(values)}; the vector must not be padded or truncated")

    n = len(MODEL_FEATURE_ORDER)
    record = {"item_responses": np.asarray(values, dtype=float),
              "missing_mask": np.zeros(n, dtype=bool)}
    state = init_state(record)
    for j in range(n):
        state = update_state(state, j, int(values[j]), questions_remaining=0)

    from src.env.state import encode_state
    return encode_state(state, m_list=None, questions_remaining=0, budget=n)


def contract_completeness() -> Dict[str, Any]:
    """Whether all ten model features have a verified Polish source."""
    missing = [e["feature_index"] for e in MODEL_FEATURE_ORDER
               if e["polish_var"] is None or e["qchat25_item"] is None]
    return {
        "n_features_required": len(MODEL_FEATURE_ORDER),
        "n_features_verified": len(MODEL_FEATURE_ORDER) - len(missing),
        "missing_feature_indices": missing,
        "complete": not missing,
        "state": "COMPLETE" if not missing else "OPEN",
    }


def scoring_table() -> List[Dict[str, Any]]:
    """Full ``code -> label -> letter -> binary`` rows for all ten items."""
    report = verify_split_rule()
    rows: List[Dict[str, Any]] = []
    for entry in MODEL_FEATURE_ORDER:
        block = report[entry["polish_var"]]
        for code, detail in block["codes"].items():
            rows.append({
                "qchat10_item": entry["qchat10_item"],
                "model_feature": entry["model_feature"],
                "saudi_column": entry["saudi_column"],
                "qchat25_item": entry["qchat25_item"],
                "polish_var": entry["polish_var"],
                "code": code,
                "label": detail["label"],
                "printed_letter": detail["letter"],
                "binary": detail["binary"],
            })
    return rows


def describe_contract() -> Dict[str, Any]:
    completeness = contract_completeness()
    return {
        "canonical_mapping": {f"Q{item}": f"Q{canon}"
                              for item, canon in CANONICAL_MAPPING.items()},
        "model_feature_order": {e["model_feature"]: f"Q{e['qchat10_item']}"
                                for e in MODEL_FEATURE_ORDER},
        "saudi_feature_construction": (
            "src/data/ingest.py reads A1..A10 verbatim and in order; the model "
            "feature i is column A{i} with no thresholding, casting or reordering."),
        "ordinal_split": ORDINAL_SPLIT,
        "ordinal_split_provenance": (
            "Derived from the printed option letters and the official Q-CHAT-10 "
            "direction per item, then confirmed to equal code >= 2 for all ten "
            "items by verify_split_rule()."),
        "binary_semantics": {
            "0": "response in the typical direction for that item",
            "1": "response in the atypical/risk-side direction for that item",
        },
        "information_loss": (
            "Each 5-level response is collapsed to 2 levels; 3 categories are "
            "discarded. Irreversible and inherent to the instrument."),
        "invalid_sentinel": POLISH_INVALID_SENTINEL,
        "missing_policy": (
            "11.0 stays missing and is never scored; out-of-vocabulary codes raise; "
            "missing values are NaN rather than an imputed 0."),
        "ml_decision_threshold": {
            "value": 0.5,
            "kind": "calibrated probability",
            "note": "The published Q-CHAT-10 rule 'more than 3 points suggests ASD "
                    "traits' is a rule for the raw questionnaire and is NOT used as "
                    "the model decision threshold.",
        },
        "completeness": completeness,
        "external_validation_status": (
            "BLOCKED - the feature contract is complete but V-4/V-7 human sign-off "
            "remains OPEN."),
    }