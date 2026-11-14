"""Request validation for the JSON API (audit finding I3).

Every route parses its body through one of these models *before* touching the database, so a bad
request is a clean 400 that names the offending field - never a ``TypeError``/``IntegrityError`` 500 -
and the stored data has bounded sizes and known enums.

Conventions:

* strict types: ``what_happened: 123`` or ``insufficient_info: "yes"`` are rejected, not coerced;
* unknown keys are ignored (the SPA may send extra form fields);
* optional text fields: blank/whitespace becomes ``None`` (audit finding I7), so "no value" has one
  representation in the database;
* timestamps are normalised to UTC (``app.db.to_utc_iso``);
* error messages never echo the submitted value.
"""
from __future__ import annotations

from typing import Annotated, Any, List, Literal, Optional, Type, TypeVar

from flask import request
from pydantic import (BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints,
                      ValidationError, field_validator, model_validator)

from .db import to_utc_iso
from .errors import BadRequest

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------- limits (characters)
MAX_LABEL = 120        # short labels: context, domain, frequency, consistency
MAX_TITLE = 200
MAX_TEXT = 4000        # free text a person types
MAX_ID = 64
MAX_QUERY = 200

STATUSES = ("saved", "tried", "helpful", "not_helpful", "mixed")
CONSENT_SCOPES = ("general_research_use", "store_observations", "store_screening_sessions")


def _blank_to_none(v: Any) -> Any:
    if isinstance(v, str) and not v.strip():
        return None
    return v


def _required(max_len: int):
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=max_len)]


def _optional(max_len: int):
    return Annotated[Optional[Annotated[str, StringConstraints(strip_whitespace=True, max_length=max_len)]],
                     BeforeValidator(_blank_to_none)]


class _Body(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")

    #: Columns that may be omitted on an update but never set to null.
    _not_null: tuple = ()

    @model_validator(mode="after")
    def _reject_explicit_null(self):
        for name in self._not_null:
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be empty")
        return self

    def provided(self) -> dict:
        """Only the fields the client actually sent (what a PUT should change)."""
        return {k: getattr(self, k) for k in self.model_fields_set}


def _utc(v: Optional[str]) -> Optional[str]:
    try:
        return to_utc_iso(v)
    except ValueError:
        raise ValueError("must be an ISO-8601 date/time") from None


# ---------------------------------------------------------------------------- observations


class ObservationIn(_Body):
    context: _required(MAX_LABEL)
    what_happened: _required(MAX_TEXT)
    frequency: _optional(MAX_LABEL) = None
    difficulty: _optional(MAX_TEXT) = None
    what_helped: _optional(MAX_TEXT) = None
    consistency: _optional(MAX_LABEL) = None
    insufficient_info: bool = False
    notes: _optional(MAX_TEXT) = None
    session_id: _optional(MAX_ID) = None
    occurred_at: _optional(MAX_LABEL) = None

    @field_validator("occurred_at")
    @classmethod
    def _normalise_time(cls, v):
        return _utc(v)


class ObservationUpdate(_Body):
    _not_null = ("context", "what_happened", "insufficient_info")
    context: Optional[_required(MAX_LABEL)] = None
    what_happened: Optional[_required(MAX_TEXT)] = None
    frequency: _optional(MAX_LABEL) = None
    difficulty: _optional(MAX_TEXT) = None
    what_helped: _optional(MAX_TEXT) = None
    consistency: _optional(MAX_LABEL) = None
    insufficient_info: Optional[bool] = None
    notes: _optional(MAX_TEXT) = None
    session_id: _optional(MAX_ID) = None
    occurred_at: _optional(MAX_LABEL) = None

    @field_validator("occurred_at")
    @classmethod
    def _normalise_time(cls, v):
        return _utc(v)


# ---------------------------------------------------------------------------- support strategies


class StrategyIn(_Body):
    title: _required(MAX_TITLE)
    domain: _optional(MAX_LABEL) = None
    description: _optional(MAX_TEXT) = None
    triggered_observation_id: _optional(MAX_ID) = None
    why_useful: _optional(MAX_TEXT) = None
    how_to_try: _optional(MAX_TEXT) = None
    how_to_record: _optional(MAX_TEXT) = None
    status: Literal["saved", "tried", "helpful", "not_helpful", "mixed"] = "saved"
    outcome: _optional(MAX_TEXT) = None
    notes: _optional(MAX_TEXT) = None
    source: _optional(MAX_TITLE) = None


class StrategyUpdate(_Body):
    _not_null = ("title", "status")
    title: Optional[_required(MAX_TITLE)] = None
    domain: _optional(MAX_LABEL) = None
    description: _optional(MAX_TEXT) = None
    triggered_observation_id: _optional(MAX_ID) = None
    why_useful: _optional(MAX_TEXT) = None
    how_to_try: _optional(MAX_TEXT) = None
    how_to_record: _optional(MAX_TEXT) = None
    status: Optional[Literal["saved", "tried", "helpful", "not_helpful", "mixed"]] = None
    outcome: _optional(MAX_TEXT) = None
    notes: _optional(MAX_TEXT) = None
    source: _optional(MAX_TITLE) = None


class AttemptIn(_Body):
    status: Literal["saved", "tried", "helpful", "not_helpful", "mixed"] = "tried"
    outcome: _optional(MAX_TEXT) = None
    notes: _optional(MAX_TEXT) = None


class SuggestIn(_Body):
    observation_id: _required(MAX_ID)


# ---------------------------------------------------------------------------- consent


class ConsentIn(_Body):
    scope: Literal["general_research_use", "store_observations", "store_screening_sessions"] = "general_research_use"
    granted: bool = False
    details: _optional(2000) = None


# ---------------------------------------------------------------------------- screening / RL lab


class ScreeningStart(_Body):
    mode: Literal["interactive", "auto"] = "interactive"
    policy: Optional[Literal["greedy", "random", "beta_greedy", "rl_adaptive"]] = None


class ScreeningAnswer(_Body):
    session_id: _required(MAX_ID)
    value: Optional[int] = None
    item_index: Optional[int] = Field(default=None, ge=0, le=999)
    stop: bool = False
    skip: bool = False   # "prefer not to answer": rl_adaptive sessions record the item as asked, without evidence

    @field_validator("value")
    @classmethod
    def _binary(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("must be 0 or 1")
        return v


class RlReplay(_Body):
    scenario: Optional[_required(MAX_ID)] = None
    policy: Optional[_required(MAX_ID)] = None
    budget: Optional[int] = Field(default=None, ge=1, le=10)


class RlCompare(_Body):
    scenario: Optional[_required(MAX_ID)] = None
    policies: Optional[List[_required(MAX_ID)]] = Field(default=None, max_length=6)
    budget: Optional[int] = Field(default=None, ge=1, le=10)


# ---------------------------------------------------------------------------- parsing helpers


def _fields(exc: ValidationError) -> List[dict]:
    out = []
    for e in exc.errors(include_input=False, include_url=False, include_context=False):
        loc = ".".join(str(p) for p in e.get("loc", ()) if p != "__root__") or "body"
        out.append({"field": loc, "message": e.get("msg", "invalid")})
    return out


def parse_body(model: Type[T]) -> T:
    """Parse the JSON request body into ``model`` or raise a 400 naming the bad fields.

    A missing/empty body is treated as ``{}`` (several endpoints take no body); a body that is present
    but not valid JSON, or not a JSON object, is rejected.
    """
    raw = request.get_json(silent=True)
    if raw is None:
        if request.get_data(cache=True).strip():
            raise BadRequest("Request body is not valid JSON")
        raw = {}
    if not isinstance(raw, dict):
        raise BadRequest("Request body must be a JSON object")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        fields = _fields(exc)
        summary = "; ".join(f"{f['field']}: {f['message']}" for f in fields[:5])
        raise BadRequest(f"Invalid request - {summary}", fields=fields) from None


def query_text(name: str) -> str:
    """A bounded, trimmed search/filter parameter (empty string when absent)."""
    value = (request.args.get(name) or "").strip()
    if len(value) > MAX_QUERY:
        raise BadRequest(f"query parameter '{name}' is too long (max {MAX_QUERY})")
    return value


def query_limit(default: int, maximum: int) -> int:
    raw = request.args.get("limit")
    if raw is None or raw == "":
        return default
    try:
        n = int(raw)
    except ValueError:
        raise BadRequest("query parameter 'limit' must be an integer") from None
    if not 1 <= n <= maximum:
        raise BadRequest(f"query parameter 'limit' must be between 1 and {maximum}")
    return n
