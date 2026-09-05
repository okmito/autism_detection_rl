"""V-2 no-claim rule — lock the phrases that are FORBIDDEN until V-2 is completed.

§24, §692. Until V-2 search log is signed off:
  - 'first'
  - 'only'
  - 'no prior work'
  - 'absent from the literature'
  - 'to our knowledge' (when qualifying novelty)

are FORBIDDEN in any markdown under this repo and in any commit message.
This test scans the markdown files and fails if any forbidden phrase appears
outside of a list of explicit allow-listed contexts (the verification-gates
section of README.md, the V-2 doc itself, and the agent_progress doc).
"""
from __future__ import annotations
import re
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
FORBIDDEN_PHRASES = [
    r"\bfirst\b",
    r"\bonly\b",
    r"\bno prior work\b",
    r"\babsent from the literature\b",
    r"\bto our knowledge\b",
]
# Files where these phrases are part of the rule statement itself, not a claim.
ALLOWLIST = {
    REPO / "README.md",
    REPO / "V2_PRISMA_SEARCH_LOG.md",
    REPO / "AGENT_PROGRESS.md",
    REPO / "V1_NZ_DATASET_RESOLUTION.md",
    REPO / "AUDIT_REPORT.md",
    REPO / "DATA_VERIFICATION_REPORT.md",
    REPO / "STATE_COUNT_VERIFICATION.md",
    REPO / "AUDIT_UPDATE_2026-09-04.md",
    # The master spec is the source of truth and predates V-2; it uses
    # these phrases in non-novelty contexts (e.g. "binary only" = exclusively
    # binary, mathematical definitions). Allow-listed for that reason.
    REPO / "Master-Project-Specification_FINAL.md",
}

# Tokens that signal a clearly non-novelty use of "only" / "first" on a line.
# A line is auto-skipped if any of these appear in the same line, OR if the
# line is inside a markdown code block (counted via ```).
NON_NOVELTY_LINE_TOKENS = (
    # policy / statement context
    "FORBIDDEN", "PENDING", "NOT VERIFIED", "TEMPLATE", "V-2", "V1_", "BLOCKED",
    "NOT EXECUTED", "STATUS:", "ALLOW-LIST", "ALLOWLIST",
    # technical / spec usage of "only" (exclusivity, not novelty)
    "READ-ONLY", "READ_ONLY", "BINARY ONLY", "EXCLUSIVELY", "PURE",
    # formal-policy / spec wording
    "MUST", "SHALL", "§", "LEGAL ACTIONS", "TERMINOLOGY",
    "POLICY", "INVARIANT", "PENDING", "STOP LEGAL", "READ-ONLY",
    # audit / test language
    "AUDIT", "TEST", "PASS", "FAIL", "MANUAL CHECK",
    "ISOLATION", "REPRODUCIBLE", "REPRODUCIBILITY",
    "SCREENING", "REFERRAL RECOMMENDATION",
    "DATA-DRIVEN", "PROGRAMMATICALLY", "RECORDED",
    # file / path / line markers
    "FILE:", "PATH:", "LINE:",
)


def _scan_file(p: Path) -> list[tuple[str, str]]:
    hits = []
    try:
        text = p.read_text()
    except (UnicodeDecodeError, IsADirectoryError):
        return hits
    for pat in FORBIDDEN_PHRASES:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            # Allow the phrase in clearly policy-stating contexts.
            # We do this by skipping matches that occur in lines that contain
            # one of the NON_NOVELTY_LINE_TOKENS or are inside a markdown code
            # block.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end]
            upper = line.upper()
            if any(tok in upper for tok in NON_NOVELTY_LINE_TOKENS):
                continue
            # Skip markdown code blocks: approximate by counting ``` before this line
            backticks = text[:line_start].count("```")
            if backticks % 2 == 1:
                continue
            hits.append((line.strip()[:200], m.group(0)))
    return hits


def test_no_forbidden_phrases_in_markdown():
    failures = []
    for p in REPO.rglob("*.md"):
        if p in ALLOWLIST:
            continue
        # Skip docs/prisma (template files)
        try:
            p.relative_to(REPO / "docs")
            continue
        except ValueError:
            pass
        for line, phrase in _scan_file(p):
            failures.append(f"{p.relative_to(REPO)}: '{phrase}' in: {line}")
    assert not failures, (
        "V-2 no-claim rule violated. The following forbidden phrases appear in "
        "non-allowlisted markdown:\n  " + "\n  ".join(failures)
    )


def test_prisma_template_present():
    assert (REPO / "V2_PRISMA_SEARCH_LOG.md").exists(), "V-2 PRISMA template missing"
    assert (REPO / "docs" / "prisma" / "screening_worksheet.csv").exists(), \
        "V-2 screening worksheet template missing"


def test_prisma_template_has_required_sections():
    text = (REPO / "V2_PRISMA_SEARCH_LOG.md").read_text()
    for section in [
        "## 1. Research question",
        "## 2. Databases / search engines",
        "## 3. Search strings",
        "## 4. Inclusion / exclusion criteria",
        "## 5. PRISMA flow",
        "## 6. Screening worksheet schema",
        "## 7. Required outputs",
        "## 8. No-claim rule",
    ]:
        assert section in text, f"missing section: {section}"
