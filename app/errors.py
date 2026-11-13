"""Errors that are *meant* to reach the client, with the HTTP status they map to.

Anything raised that is not an :class:`ApiError` (or a werkzeug ``HTTPException``) is a server fault:
``app.server`` logs it with an error id and answers 500 without echoing the exception text.

``ApiError`` subclasses ``ValueError`` so service-level callers that already ``except ValueError``
keep working; a *plain* ``ValueError`` (for example the internal check on policy output) is not an
``ApiError`` and therefore stays a 500.
"""
from __future__ import annotations


class ApiError(ValueError):
    status = 400

    def __init__(self, message: str, status: int | None = None, **extra):
        super().__init__(message)
        if status is not None:
            self.status = status
        self.extra = extra


class BadRequest(ApiError):
    status = 400


class NotFound(ApiError):
    status = 404


class Conflict(ApiError):
    """The request is valid but conflicts with the stored state (stale/duplicate submission)."""
    status = 409


class Unavailable(ApiError):
    """A required resource (dataset, model) is not available on this machine."""
    status = 503
