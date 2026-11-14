"""Stage 8: security & privacy regression tests (no secrets, no PII,
no unbalanced error surfaces, CORS-free, gitignore coverage)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_no_committed_raw_data_or_sav():
    out = subprocess.check_output(["git", "ls-files", "data/"], cwd=str(REPO), text=True)
    assert out.strip() == "", f"tracked data files must not be committed: {out}"
    savs = [p for p in subprocess.check_output(["git", "ls-files"], cwd=str(REPO), text=True).splitlines() if p.endswith(".sav")]
    assert savs == [], f".sav participant exports must never be tracked: {savs}"


def test_gitignore_covers_sensitive_paths():
    gi = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pat in ("data/raw/", "results/", "*.sav", ".env", "data/*.sav"):
        assert pat in gi, f".gitignore missing {pat!r}"


def test_no_hardcoded_api_keys():
    hits = []
    for p in (REPO / "src").rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        for marker in ("AKIA", "OPENAI_API_KEY", "SECRET_", "xoxb"):
            if marker in t:
                hits.append((str(p), marker))
        # 'sk-' alone is a false positive (hits English words like 'task-');
        # only a plausible OpenAI key shape is a real hit.
        import re
        if re.search(r"sk-(?:proj-)?[A-Za-z0-9]{20,}", t):
            hits.append((str(p), "sk-..."))
    assert hits == [], f"potential hardcoded secrets in src/: {hits}"


def test_polish_loader_withholds_participant_ids_from_warnings():
    import warnings
    from src.data import ingest
    if not ingest.POLISH_CSV.exists():
        pytest.skip("Polish CSV not present locally")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ingest.load_polish()
    msgs = " ".join(str(x.message) for x in w)
    assert "child_id" not in msgs and "bdbp" not in msgs


def test_source_has_no_network_transmission():
    from src.data import ingest, schema, validate  # noqa: F401
    import src.data.ingest as m
    import inspect
    src_text = open(m.__file__, encoding="utf-8").read()
    for token in ("requests.get", "requests.post", "urllib.request", "urlopen", "socket."):
        assert token not in src_text, f"ingest must not transmit data externally ({token})"


def test_demo_app_input_validation_contract():
    text = (REPO / "scripts/demo_app.py").read_text(encoding="utf-8", errors="ignore")
    # the P4 revision removed the support questionnaire endpoint
    assert "POST /api/session/support" not in text
    assert "/api/session/support" not in text or "404" in text
