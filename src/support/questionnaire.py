"""The screening questionnaire as the verified contract defines it — P4.

One module owns the questionnaire the user actually answers. Everything in it
is **derived from** ``src/data/qchat10_contract.py`` (the signed-off Q-CHAT-10
feature contract) rather than restated here, so the support layer, the
explanation layer and the demo cannot disagree about what a question asks:

* ``construct``      — the verbatim construct string from the contract
                       (``ORDINAL_VERIFICATION[...]["construct_en"]``);
* ``feature_id``     — ``feature_{item}``, the contract's own feature name;
* ``qchat25_item``   — the Q-CHAT-25 source item the contract maps it to;
* ``scoring_direction`` — the official Q-CHAT-10 rule for that item.

Why this module exists
----------------------
``scripts/demo_app.py`` historically carried its own hand-written list of item
wording, which drifted from the contract: from A3 onward the demo asked about
one construct while the answer was recorded as the feature for a *different*
construct. No test compared the two, so the drift survived review. The display
wording below is keyed by the contract's Q-CHAT-10 item number, and
``tests/test_support_questionnaire.py`` asserts that the demo's list, the
contract's feature order and this table all agree. A misalignment now fails a
test instead of quietly mislabelling eight questions.

Answer semantics (unchanged from the contract): each item is binary, where

    ``1`` = the atypical / risk-side response for that item
    ``0`` = the typical response for that item

and the polarity is per item (items 1-9 score C/D/E as 1; item 10 scores
A/B/C as 1). The 5-level Polish responses are collapsed to these two levels
before they ever reach here; that collapse is lossy and is documented in the
contract, not hidden here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.data.qchat10_contract import (MODEL_FEATURE_ORDER,
                                       ORDINAL_VERIFICATION,
                                       QCHAT10_SCORING_DIRECTION,
                                       QCHAT_MAPPING_VERSION)

#: Answer values the instrument produces (binary projection, contract §6).
ATYPICAL_VALUE = 1
TYPICAL_VALUE = 0

#: Display wording, keyed by the contract's Q-CHAT-10 item number. Keying by the
#: contract's own numbering — not by list position — is what makes a silent
#: reordering impossible: an item whose construct is not in the contract cannot
#: be described here at all.
_DISPLAY_TEXT: Dict[int, str] = {
    1: "Does your child look at you when you call their name?",
    2: "Is it easy for you to get eye contact with your child?",
    3: "Does your child point to ask for something they want?",
    4: "Does your child point to share interest with you?",
    5: "Does your child pretend — for example caring for a doll, or talking on "
       "a toy phone?",
    6: "Does your child follow where you are looking?",
    7: "If someone is visibly upset, does your child show signs of wanting to "
       "comfort them?",
    8: "Were your child's first words typical for their age?",
    9: "Does your child use simple gestures, like waving goodbye?",
    10: "Does your child stare at nothing with no apparent purpose?",
}


@dataclass(frozen=True)
class QuestionnaireItem:
    """One screening question, fixed by the contract."""
    item_code: str                     # "A1".."A10" — the project's item code
    question_text: str                 # wording shown to the user
    construct: str                     # the verified construct this feature measures
    feature_id: str                    # "feature_1".."feature_10"
    feature_index: int                 # 0-based index into the model's 10 features
    qchat10_item: int                  # the contract's 1-based item number
    qchat25_item: int                  # the Q-CHAT-25 item it maps to
    polish_var: str                    # the Polish variable it comes from
    scoring_direction: str             # official Q-CHAT-10 direction

    @property
    def atypical_value(self) -> int:
        return ATYPICAL_VALUE

    @property
    def typical_value(self) -> int:
        return TYPICAL_VALUE

    def answer_label(self, response: int) -> str:
        """Human wording for a recorded binary answer."""
        if response == ATYPICAL_VALUE:
            return "recorded as the atypical response"
        return "recorded as the typical response"


ITEMS: Tuple[QuestionnaireItem, ...] = tuple(
    QuestionnaireItem(
        item_code=entry["saudi_column"],
        question_text=_DISPLAY_TEXT[entry["qchat10_item"]],
        construct=ORDINAL_VERIFICATION[entry["qchat25_item"]]["construct_en"],
        feature_id=entry["model_feature"],
        feature_index=int(entry["feature_index"]) - 1,
        qchat10_item=int(entry["qchat10_item"]),
        qchat25_item=int(entry["qchat25_item"]),
        polish_var=entry["polish_var"],
        scoring_direction=QCHAT10_SCORING_DIRECTION[entry["qchat10_item"]],
    )
    for entry in MODEL_FEATURE_ORDER
)

#: Ordered by feature index, which is the order the project asks and records in.
ITEM_BY_CODE: Dict[str, QuestionnaireItem] = {i.item_code: i for i in ITEMS}
ITEM_CODES: Tuple[str, ...] = tuple(i.item_code for i in ITEMS)

#: Items the questionnaire asks that carry no curated, reviewed suggestion in
#: this project. Kept here so the report can disclose them instead of
#: pretending they were covered.
ASKED_WITHOUT_SUGGESTION_NOTE: str = (
    "The questionnaire also asks about pretending (A5) and staring at nothing "
    "with no apparent purpose (A10) where those items came up. This project "
    "has no curated, expert-reviewed suggestion linked to them, so none is "
    "offered and nothing is inferred from them.")


def is_valid_item_code(code: str) -> bool:
    """Whether ``code`` names an item the questionnaire actually asks."""
    return code in ITEM_BY_CODE


def item_code_for_feature_index(j: int) -> str:
    """Model feature index (0-based) -> item code (``A{j+1}``)."""
    return ITEMS[j].item_code


def questionnaire_table() -> List[Dict[str, object]]:
    """The whole questionnaire as plain data (used for the report's evidence
    block and by tests)."""
    return [{
        "item_code": i.item_code,
        "question_text": i.question_text,
        "construct": i.construct,
        "feature_id": i.feature_id,
        "feature_index": i.feature_index,
        "qchat10_item": i.qchat10_item,
        "qchat25_item": i.qchat25_item,
        "polish_var": i.polish_var,
        "scoring_direction": i.scoring_direction,
        "atypical_value": ATYPICAL_VALUE,
        "typical_value": TYPICAL_VALUE,
    } for i in ITEMS]


def questionnaire_version() -> str:
    """Which questionnaire contract this module is bound to."""
    return QCHAT_MAPPING_VERSION
