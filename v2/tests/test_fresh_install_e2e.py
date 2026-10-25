#!/usr/bin/env python3
"""Fresh-install end-to-end tests: hooks as subprocesses on a bare HOME.

Proves plug-and-play: no environment setup beyond ANAMNESIS_HOME, no config
files, no pre-existing identity. The system must provision identity, create
stores, capture events, surface knowledge, and block dangerous commands.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent


def run_hook(repo_home: str, script: str, payload: dict,
             extra_env: dict = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, ANAMNESIS_HOME=repo_home)
    env.pop("ANAMNESIS_AGENT_ID", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(REPO / script)],
        input=json.dumps(payload), capture_output=True, text=True,
        timeout=20, env=env)


@pytest.fixture
def home(tmp_path):
    return str(tmp_path / "data")


def test_bare_install_provisions_and_stores(home):
    """Session start on a bare machine: identity provisioned, store created."""
    proc = run_hook(home, "on_session_start.py",
                    {"session_id": "s1", "cwd": "/tmp/proj-alpha"})
    assert proc.returncode == 0, proc.stderr
    agents = list((Path(home) / "agents").iterdir())
    assert len(agents) == 1
    assert agents[0].name.startswith("project-proj-alpha-")
    assert (agents[0] / "anamnesis.db").exists()


def test_project_isolation(home):
    """Two projects get separate identities and separate stores."""
    run_hook(home, "on_session_start.py", {"session_id": "s1", "cwd": "/tmp/p1"})
    run_hook(home, "on_session_start.py", {"session_id": "s2", "cwd": "/tmp/p2"})
    agents = sorted(p.name for p in (Path(home) / "agents").iterdir())
    assert len(agents) == 2
    assert agents[0] != agents[1]


def test_event_capture_and_retrieval_roundtrip(home):
    """Events recorded via post_tool are retrievable in the store."""
    cwd = "/tmp/proj-roundtrip"
    run_hook(home, "on_session_start.py", {"session_id": "s1", "cwd": cwd})
    run_hook(home, "on_post_tool.py", {
        "session_id": "s1", "cwd": cwd, "tool_name": "Bash",
        "tool_input": {"command": "npm test"},
        "tool_response": "FAIL 3 tests ModuleNotFoundError: flangle"})
    agents = list((Path(home) / "agents").iterdir())
    db = sqlite3.connect(str(agents[0] / "anamnesis.db"))
    events = db.execute(
        "SELECT event_type, error_text FROM empeiria").fetchall()
    db.close()
    assert any(e[0] == "tool_error" and "flangle" in e[1] for e in events)


def test_guardian_blocks_dangerous_on_bare_install(home):
    """Sophia axioms work with zero prior knowledge."""
    proc = run_hook(home, "on_pre_bash.py", {
        "session_id": "s1", "cwd": "/tmp/p",
        "tool_input": {"command": "rm -rf /var/data/production"}})
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_guardian_allows_safe_command(home):
    proc = run_hook(home, "on_pre_bash.py", {
        "session_id": "s1", "cwd": "/tmp/p",
        "tool_input": {"command": "git status"}})
    assert proc.returncode == 0
    assert proc.stdout.strip() == "", f"safe command should pass silently: {proc.stdout}"


def test_prompt_submit_surfacing(home):
    """Record a learning via CLI, then prompt-submit surfaces it."""
    cwd = "/tmp/proj-surf"
    run_hook(home, "on_session_start.py", {"session_id": "s1", "cwd": cwd})
    env = dict(os.environ, ANAMNESIS_HOME=home)
    env.pop("ANAMNESIS_AGENT_ID", None)
    subprocess.run(
        [sys.executable, str(REPO / "anamnesis_cli.py"), "record",
         "WHEN using sqlite3 with concurrent writers THEN database locked "
         "BECAUSE WAL mode not enabled. FIX: PRAGMA journal_mode=WAL",
         "--category", "error_fix", "--cwd", cwd],
        capture_output=True, text=True, timeout=20, env=env)
    proc = run_hook(home, "on_prompt_submit.py", {
        "session_id": "s1", "cwd": cwd,
        "prompt": "how do I fix sqlite3 database locked errors with wal mode?"})
    assert proc.returncode == 0
    assert "journal_mode" in proc.stdout or "WAL" in proc.stdout, proc.stdout


def test_session_end_hook_without_llm(home, tmp_path):
    """Stop hook degrades gracefully with no LLM key: maintenance runs,
    no crash, no false output."""
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("\n".join([
        json.dumps({"type": "user", "message": {"content": "please fix the bug in the parser module"}}),
        json.dumps({"type": "assistant", "message": {"content": "I found the issue and fixed the parser bug"}}),
    ]))
    env = dict(os.environ)
    for var in ("ANAMNESIS_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(var, None)
    proc = run_hook(home, "agent_learn.py", {
        "session_id": "s1", "cwd": "/tmp/p",
        "transcript_path": str(transcript)}, extra_env=env)
    assert proc.returncode == 0, proc.stderr
