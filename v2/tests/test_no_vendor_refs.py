#!/usr/bin/env python3
"""Guard: vendor-specific references must not regress into the repo.

Scans every tracked text file for product-decoupling violations. The README
is excluded (a sponsorship acknowledgment may legitimately live there).
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

FORBIDDEN = [
    r"happycapy",
    r"ai-gateway",
    r"capy_agent_id",
    r"\bcapy_",
    r"ai_gateway",
    r"cortex_fast_model",
    r"surfaced-cto",
    r"cortex\.db",
    r"capy-cortex",
]
PATTERN = re.compile("|".join(FORBIDDEN), re.IGNORECASE)

# README: sponsorship exception. This test file: defines the patterns.
EXCLUDED = {"README.md", "v2/tests/test_no_vendor_refs.py"}


def _tracked_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.splitlines() if p.strip()]


def test_no_vendor_references():
    violations = []
    for rel in _tracked_files():
        if rel in EXCLUDED:
            continue
        path = REPO / rel
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                violations.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not violations, "vendor references found:\n" + "\n".join(violations)
