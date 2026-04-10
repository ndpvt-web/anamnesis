#!/usr/bin/env python3
"""Shared test fixtures for Hermes v2 test suite.
Provides in-memory DB creation, sample data, and cleanup utilities."""

import sqlite3
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Ensure v2 modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from v2 import db as v2db
from v2 import knowledge_graph as kg
from v2 import retrieval
from v2 import feedback
from v2 import maintenance

# Test agent ID (never collides with real agents)
TEST_AGENT = "__test_hermes_v2__"
TEST_SESSION = "test-session-001"


def create_test_db() -> sqlite3.Connection:
    """Create a fresh in-memory hermes.db with full schema.
    Returns an open connection. Caller must close."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    v2db._ensure_schema(db)
    return db


def create_temp_agent_db():
    """Create a temp directory with a real on-disk hermes.db.
    Returns (db, tmp_dir). Caller must close db and remove tmp_dir."""
    tmp_dir = tempfile.mkdtemp(prefix="hermes_test_")
    hermes_dir = Path(tmp_dir) / "hermes"
    hermes_dir.mkdir()
    db_path = hermes_dir / "hermes.db"
    db = sqlite3.connect(str(db_path), timeout=5.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=3000")
    db.execute("PRAGMA foreign_keys=ON")
    v2db._ensure_schema(db)
    return db, tmp_dir


def cleanup_temp(tmp_dir: str):
    """Remove temp directory."""
    shutil.rmtree(tmp_dir, ignore_errors=True)


def seed_nodes(db, agent_id: str = TEST_AGENT, count: int = 10) -> list:
    """Insert sample kg_nodes. Returns list of node IDs."""
    node_ids = []
    categories = ["anti_pattern", "preference", "principle", "general"]
    for i in range(count):
        cat = categories[i % len(categories)]
        confidence = 0.50 + (i * 0.05)
        if confidence > 0.95:
            confidence = 0.95
        cursor = db.execute("""
            INSERT INTO kg_nodes (agent_id, node_type, layer, content, summary,
                                  applies_when, category, severity, confidence,
                                  helpful_count, maturity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (agent_id, v2db.NODE_FACT, v2db.LAYER_TECHNE,
              f"Test learning #{i}: {cat} pattern about topic-{i}",
              f"Summary for node {i}", f"When doing thing-{i}",
              cat, "medium", confidence, i % 5, "candidate"))
        node_ids.append(cursor.lastrowid)
        # Also insert into FTS
        db.execute("""
            INSERT INTO kg_fts(rowid, content, summary, category)
            VALUES (?, ?, ?, ?)
        """, (cursor.lastrowid,
              f"Test learning #{i}: {cat} pattern about topic-{i}",
              f"Summary for node {i}", cat))
    db.commit()
    return node_ids


def seed_episteme_node(db, agent_id: str = TEST_AGENT,
                       content: str = "Validated episteme knowledge",
                       confidence: float = 0.90) -> int:
    """Insert a single episteme/crystallized node. Returns node ID."""
    cursor = db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content,
                              category, confidence, helpful_count, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (agent_id, v2db.NODE_FACT, v2db.LAYER_EPISTEME,
          content, "principle", confidence, 5, "crystallized"))
    nid = cursor.lastrowid
    db.execute("""
        INSERT INTO kg_fts(rowid, content, summary, category)
        VALUES (?, ?, ?, ?)
    """, (nid, content, "", "principle"))
    db.commit()
    return nid


def seed_edges(db, agent_id: str = TEST_AGENT,
               node_ids: list = None) -> list:
    """Insert sample edges between nodes. Returns list of edge IDs."""
    if not node_ids or len(node_ids) < 3:
        return []
    edge_ids = []
    # SUPPORTS edges
    for i in range(1, min(4, len(node_ids))):
        cursor = db.execute("""
            INSERT INTO kg_edges (agent_id, from_node_id, to_node_id, edge_type, weight)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, node_ids[i], node_ids[0], v2db.EDGE_SUPPORTS, 1.0))
        edge_ids.append(cursor.lastrowid)
    # CONTRADICTS edge
    if len(node_ids) >= 5:
        cursor = db.execute("""
            INSERT INTO kg_edges (agent_id, from_node_id, to_node_id, edge_type, weight)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, node_ids[3], node_ids[4], v2db.EDGE_CONTRADICTS, 1.0))
        edge_ids.append(cursor.lastrowid)
    db.commit()
    return edge_ids


def seed_events(db, agent_id: str = TEST_AGENT,
                session_id: str = TEST_SESSION, count: int = 5) -> list:
    """Insert sample empeiria events. Returns list of event IDs."""
    event_ids = []
    event_types = ["prompt_submit", "tool_success", "tool_error",
                   "tool_success", "prompt_submit"]
    for i in range(count):
        etype = event_types[i % len(event_types)]
        cursor = db.execute("""
            INSERT INTO empeiria (agent_id, session_id, event_type, tool_name,
                                  command, error_text)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_id, session_id, etype,
              f"Tool{i}", f"command-{i}",
              f"Error text {i}" if "error" in etype else ""))
        event_ids.append(cursor.lastrowid)
    db.commit()
    return event_ids


def seed_surfaced(db, agent_id: str = TEST_AGENT,
                  session_id: str = TEST_SESSION,
                  node_ids: list = None) -> list:
    """Insert surfaced_nodes records. Returns list of surfaced IDs."""
    if not node_ids:
        return []
    surfaced_ids = []
    for nid in node_ids[:5]:
        cursor = db.execute("""
            INSERT INTO surfaced_nodes (agent_id, session_id, node_id, feedback)
            VALUES (?, ?, ?, 'pending')
        """, (agent_id, session_id, nid))
        surfaced_ids.append(cursor.lastrowid)
    db.commit()
    return surfaced_ids
