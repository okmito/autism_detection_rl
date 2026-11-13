"""Shared fixtures for the NeuroBridge web-app tests (tests/test_app_*.py).

Only fixtures live here; the older research tests do not use any of them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REAL_DB = REPO / "data" / "neurobridge.db"


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A Flask test client bound to a throw-away SQLite file.

    ``app.db`` resolves ``DB_PATH`` on every connection (per thread, migrated on first use), so
    pointing it at a temp file is enough. The real ``data/neurobridge.db`` must never be opened by
    a test.
    """
    from app import db, server

    path = tmp_path / "neurobridge-test.db"
    assert path.resolve() != REAL_DB.resolve()
    monkeypatch.setattr(db, "DB_PATH", path)
    db.close_connection()
    client = server.app.test_client()
    try:
        yield client
    finally:
        db.close_connection()


@pytest.fixture
def screening_api(api):
    """``api`` plus the screening model/data; skipped when the gitignored dataset is absent."""
    from src.data import ingest

    if not (REPO / ingest.SAUDI_CSV).exists():
        pytest.skip("Saudi Q-CHAT-10 CSV not present locally (data/raw/ is gitignored)")
    from app.services import screening

    screening.ensure_ready()
    return api
