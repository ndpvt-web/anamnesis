#!/usr/bin/env python3
"""Tests for migrate_store.py: generic store importer."""

import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
import migrate_store


@pytest.fixture
def old_layout(tmp_path):
    """Fixture of a previous installation layout with two agents."""
    src = tmp_path / "old-agents"
    # agent-a: v1 + v2-legacy stores
    (src / "agent-a" / "learnings").mkdir(parents=True)
    (src / "agent-a" / "hermes").mkdir(parents=True)
    for db_path in (src / "agent-a" / "learnings" / "learnings.db",
                    src / "agent-a" / "hermes" / "hermes.db"):
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE marker (id INTEGER)")
        conn.execute("INSERT INTO marker VALUES (1)")
        conn.commit()
        conn.close()
    # agent-b: only v1
    (src / "agent-b" / "learnings").mkdir(parents=True)
    conn = sqlite3.connect(str(src / "agent-b" / "learnings" / "learnings.db"))
    conn.execute("CREATE TABLE marker (id INTEGER)")
    conn.commit()
    conn.close()
    # noise directory without stores
    (src / "not-an-agent").mkdir()
    return src


def test_discovers_only_store_dirs(old_layout):
    found = migrate_store.find_source_agents(old_layout)
    assert set(found) == {"agent-a", "agent-b"}
    assert len(found["agent-a"]) == 2
    assert len(found["agent-b"]) == 1


def test_import_renames_legacy_layout(old_layout, tmp_path):
    to_dir = tmp_path / "new-agents"
    results = migrate_store.import_agent(old_layout, to_dir, "agent-a")
    assert all(r["status"] == "copied" for r in results)
    assert (to_dir / "agent-a" / "learnings" / "learnings.db").exists()
    assert (to_dir / "agent-a" / "anamnesis.db").exists()
    # source untouched
    assert (old_layout / "agent-a" / "hermes" / "hermes.db").exists()
    # content preserved
    conn = sqlite3.connect(str(to_dir / "agent-a" / "anamnesis.db"))
    assert conn.execute("SELECT COUNT(*) FROM marker").fetchone()[0] == 1
    conn.close()


def test_refuses_overwrite_without_force(old_layout, tmp_path):
    to_dir = tmp_path / "new-agents"
    migrate_store.import_agent(old_layout, to_dir, "agent-a")
    results = migrate_store.import_agent(old_layout, to_dir, "agent-a")
    assert all(r["status"].startswith("refused") for r in results)
    results = migrate_store.import_agent(old_layout, to_dir, "agent-a",
                                         force=True)
    assert all(r["status"] == "copied" for r in results)


def test_dry_run_writes_nothing(old_layout, tmp_path):
    to_dir = tmp_path / "new-agents"
    results = migrate_store.import_agent(old_layout, to_dir, "agent-a",
                                         dry_run=True)
    assert all(r["status"] == "would copy" for r in results)
    assert not to_dir.exists()


def test_unknown_agent_errors(old_layout, tmp_path):
    results = migrate_store.import_agent(old_layout, tmp_path / "t", "ghost")
    assert "error" in results[0]
