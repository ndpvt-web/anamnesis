#!/usr/bin/env python3
"""Hermes v2 Core Database Library.

Manages hermes.db -- the v2 knowledge store with 8 tables implementing
the Aristotelian five-layer memory stack.

Tables:
  empeiria        - Append-only event log (L2)
  kg_nodes        - Knowledge graph nodes (L3-L5)
  kg_edges        - Typed relationships between nodes
  kg_fts          - FTS5 virtual table on kg_nodes.content
  aisthesis       - Session-scoped working memory (L1)
  surfaced_nodes  - Tracking surfaced nodes + feedback
  reflection_runs - Audit trail for reflection pipeline
  sophia          - Human-approved axioms (L5)

Zero external dependencies. stdlib only."""

import sqlite3
import json
import re
import os
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAPPYCAPY_DIR = Path.home() / ".happycapy"
AGENTS_DIR = HAPPYCAPY_DIR / "agents"

# Confidence thresholds
MIN_CONFIDENCE = 0.50
MAX_CONFIDENCE = 0.95
CONFIDENCE_BOOST = 0.02
CONFIDENCE_DECAY = 0.03

# Promotion thresholds (techne -> episteme)
PROMOTE_MIN_CONFIDENCE = 0.85
PROMOTE_MIN_HELPFUL = 3
PROMOTE_MIN_SUPPORTS = 2  # Must have 2+ SUPPORTS edges (new in v2)

# Layer labels
LAYER_SOPHIA = "sophia"
LAYER_EPISTEME = "episteme"
LAYER_TECHNE = "techne"
LAYER_EMPEIRIA = "empeiria"
LAYER_AISTHESIS = "aisthesis"

# Node types
NODE_FACT = "fact"
NODE_REFLECTION = "reflection"
NODE_CONCEPT = "concept"
NODE_AXIOM = "axiom"

# Edge types
EDGE_DERIVED_FROM = "DERIVED_FROM"
EDGE_SUPPORTS = "SUPPORTS"
EDGE_CONTRADICTS = "CONTRADICTS"
EDGE_GENERALIZES = "GENERALIZES"
EDGE_SUPERSEDES = "SUPERSEDES"

# FTS5 stop words (shared with v1 for consistency)
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "i", "you", "he", "she",
    "it", "we", "they", "me", "my", "your", "his", "her", "its", "our",
    "their", "this", "that", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "and", "or", "not", "but", "if", "then", "so", "just",
    "want", "need", "please", "help", "make", "let", "get",
})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def hermes_dir_for_agent(agent_id: str) -> Path:
    """Return path to agent's hermes/ directory."""
    return AGENTS_DIR / agent_id / "hermes"


def db_path_for_agent(agent_id: str) -> Path:
    """Return path to agent's hermes.db."""
    return hermes_dir_for_agent(agent_id) / "hermes.db"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def open_db(agent_id: str) -> sqlite3.Connection:
    """Open (and init if needed) the agent's hermes.db."""
    path = db_path_for_agent(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=5.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=3000")
    db.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(db)
    return db


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- L2: Append-only event log. Never updated.
CREATE TABLE IF NOT EXISTS empeiria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL,
    tool_name TEXT DEFAULT '',
    command TEXT DEFAULT '',
    file_path TEXT DEFAULT '',
    error_text TEXT DEFAULT '',
    prompt_text TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_empeiria_agent_session
    ON empeiria(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_empeiria_event_type
    ON empeiria(event_type);
CREATE INDEX IF NOT EXISTS idx_empeiria_created
    ON empeiria(created_at);

-- L3-L5: Knowledge graph nodes.
CREATE TABLE IF NOT EXISTS kg_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'fact',
    layer TEXT NOT NULL DEFAULT 'techne',
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    applies_when TEXT DEFAULT '',
    category TEXT NOT NULL DEFAULT 'general',
    severity TEXT DEFAULT 'medium',
    confidence REAL NOT NULL DEFAULT 0.70,
    helpful_count INTEGER NOT NULL DEFAULT 0,
    harmful_count INTEGER NOT NULL DEFAULT 0,
    irrelevant_count INTEGER NOT NULL DEFAULT 0,
    surface_count INTEGER NOT NULL DEFAULT 0,
    source_session TEXT DEFAULT '',
    origin_agent_id TEXT DEFAULT '',
    derived_from TEXT DEFAULT '[]',
    reflection_run_id TEXT DEFAULT '',
    maturity TEXT NOT NULL DEFAULT 'candidate',
    visibility TEXT NOT NULL DEFAULT 'private',
    embedding BLOB DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_confirmed TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_agent ON kg_nodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_layer ON kg_nodes(layer);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_category ON kg_nodes(category);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_confidence ON kg_nodes(confidence);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_maturity ON kg_nodes(maturity);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_node_type ON kg_nodes(node_type);

-- Typed relationships between KG nodes.
CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    from_node_id INTEGER NOT NULL REFERENCES kg_nodes(id),
    to_node_id INTEGER NOT NULL REFERENCES kg_nodes(id),
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(from_node_id, to_node_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_from ON kg_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_to ON kg_edges(to_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_type ON kg_edges(edge_type);

-- L1: Session-scoped working memory. Ephemeral.
CREATE TABLE IF NOT EXISTS aisthesis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    tool_name TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    command TEXT DEFAULT '',
    error_text TEXT DEFAULT '',
    files_touched TEXT DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_aisthesis_session
    ON aisthesis(agent_id, session_id);

-- Tracking which KG nodes were surfaced + feedback.
CREATE TABLE IF NOT EXISTS surfaced_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    node_id INTEGER NOT NULL REFERENCES kg_nodes(id),
    feedback TEXT DEFAULT 'pending',
    surfaced_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_surfaced_session
    ON surfaced_nodes(session_id, feedback);

-- Audit trail for reflection pipeline runs.
CREATE TABLE IF NOT EXISTS reflection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    trigger_type TEXT NOT NULL DEFAULT 'session_end',
    status TEXT NOT NULL DEFAULT 'started',
    events_processed INTEGER DEFAULT 0,
    nodes_created INTEGER DEFAULT 0,
    edges_created INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    error_text TEXT DEFAULT '',
    steps_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- L5: Human-approved axioms. Rarely written.
CREATE TABLE IF NOT EXISTS sophia (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    approved_by TEXT DEFAULT 'human',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS kg_fts USING fts5(
    content, summary, category,
    content='kg_nodes',
    content_rowid='id',
    tokenize='porter unicode61'
);
"""

_FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS kg_fts_ai AFTER INSERT ON kg_nodes BEGIN
    INSERT INTO kg_fts(rowid, content, summary, category)
    VALUES (new.id, new.content, new.summary, new.category);
END;

CREATE TRIGGER IF NOT EXISTS kg_fts_ad AFTER DELETE ON kg_nodes BEGIN
    INSERT INTO kg_fts(kg_fts, rowid, content, summary, category)
    VALUES ('delete', old.id, old.content, old.summary, old.category);
END;

CREATE TRIGGER IF NOT EXISTS kg_fts_au AFTER UPDATE ON kg_nodes BEGIN
    INSERT INTO kg_fts(kg_fts, rowid, content, summary, category)
    VALUES ('delete', old.id, old.content, old.summary, old.category);
    INSERT INTO kg_fts(rowid, content, summary, category)
    VALUES (new.id, new.content, new.summary, new.category);
END;
"""


def _ensure_schema(db: sqlite3.Connection):
    """Create all tables, indices, FTS5, and triggers."""
    db.executescript(_SCHEMA_SQL)
    try:
        db.execute(_FTS_SQL)
        db.executescript(_FTS_TRIGGERS_SQL)
    except sqlite3.OperationalError:
        pass  # FTS5 already exists
    _migrate_schema(db)
    db.commit()


def _migrate_schema(db: sqlite3.Connection):
    """Add new columns to existing tables. Safe to re-run."""
    migrations = [
        "ALTER TABLE empeiria ADD COLUMN exit_code INTEGER DEFAULT NULL",
        "ALTER TABLE empeiria ADD COLUMN tool_output TEXT DEFAULT ''",
        "ALTER TABLE empeiria ADD COLUMN duration_ms INTEGER DEFAULT NULL",
        "ALTER TABLE kg_nodes ADD COLUMN error_family TEXT DEFAULT ''",
        # v3 migrations: domain genus, node lifecycle
        "ALTER TABLE kg_nodes ADD COLUMN domain TEXT DEFAULT ''",
        "ALTER TABLE kg_nodes ADD COLUMN status TEXT DEFAULT 'active'",
        "ALTER TABLE kg_nodes ADD COLUMN archived_reason TEXT DEFAULT ''",
        "ALTER TABLE kg_nodes ADD COLUMN superseded_by INTEGER DEFAULT NULL",
    ]
    for sql in migrations:
        try:
            db.execute(sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    # Index for domain-filtered queries
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_kg_nodes_domain ON kg_nodes(domain)")
    except sqlite3.OperationalError:
        pass
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_kg_nodes_status ON kg_nodes(status)")
    except sqlite3.OperationalError:
        pass


# ---------------------------------------------------------------------------
# EMPEIRIA: Append-only event store (L2)
# ---------------------------------------------------------------------------

def append_event(db: sqlite3.Connection, agent_id: str, event_type: str,
                 session_id: str = "", tool_name: str = "",
                 command: str = "", file_path: str = "",
                 error_text: str = "", prompt_text: str = "",
                 metadata: dict = None,
                 exit_code: int = None, tool_output: str = "",
                 duration_ms: int = None) -> int:
    """Append an event to empeiria. Returns event ID."""
    cursor = db.execute("""
        INSERT INTO empeiria (agent_id, session_id, event_type, tool_name,
                              command, file_path, error_text, prompt_text, metadata,
                              exit_code, tool_output, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (agent_id, session_id, event_type, tool_name,
          command, file_path, error_text, prompt_text,
          json.dumps(metadata or {}),
          exit_code, tool_output or "", duration_ms))
    db.commit()
    return cursor.lastrowid


def get_session_events(db: sqlite3.Connection, agent_id: str,
                       session_id: str, limit: int = 50) -> list:
    """Get events for a session, most recent first."""
    rows = db.execute("""
        SELECT * FROM empeiria
        WHERE agent_id = ? AND session_id = ?
        ORDER BY created_at DESC, id DESC LIMIT ?
    """, (agent_id, session_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_recent_events(db: sqlite3.Connection, agent_id: str,
                      limit: int = 30) -> list:
    """Get most recent events across all sessions."""
    rows = db.execute("""
        SELECT * FROM empeiria
        WHERE agent_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (agent_id, limit)).fetchall()
    return [dict(r) for r in rows]


def count_events(db: sqlite3.Connection, agent_id: str) -> int:
    """Count total events for an agent."""
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM empeiria WHERE agent_id = ?",
        (agent_id,)
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# KG NODES: Knowledge graph (L3-L5)
# ---------------------------------------------------------------------------

def add_node(db: sqlite3.Connection, agent_id: str, content: str,
             node_type: str = NODE_FACT, layer: str = LAYER_TECHNE,
             summary: str = "", applies_when: str = "",
             category: str = "general", severity: str = "medium",
             confidence: float = 0.70, source_session: str = "",
             origin_agent_id: str = "", derived_from: list = None,
             reflection_run_id: str = "", maturity: str = "candidate",
             helpful_count: int = 0, error_family: str = "",
             domain: str = "") -> int:
    """Add a KG node. Returns node ID or -1 if duplicate."""
    if is_duplicate_node(db, agent_id, content):
        _bump_existing_node(db, agent_id, content)
        return -1
    cursor = db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, summary,
                              applies_when, category, severity, confidence,
                              source_session, origin_agent_id, derived_from,
                              reflection_run_id, maturity, helpful_count,
                              error_family, domain)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (agent_id, node_type, layer, content, summary, applies_when,
          category, severity, confidence, source_session,
          origin_agent_id or agent_id,
          json.dumps(derived_from or []),
          reflection_run_id, maturity, helpful_count,
          error_family or "", domain or ""))
    db.commit()
    return cursor.lastrowid


def get_node(db: sqlite3.Connection, node_id: int) -> dict:
    """Get a single node by ID."""
    row = db.execute("SELECT * FROM kg_nodes WHERE id = ?", (node_id,)).fetchone()
    return dict(row) if row else {}


def get_nodes_by_layer(db: sqlite3.Connection, agent_id: str,
                       layer: str, limit: int = 50) -> list:
    """Get active nodes for a specific layer (excludes archived)."""
    rows = db.execute("""
        SELECT * FROM kg_nodes
        WHERE agent_id = ? AND layer = ? AND maturity != 'deprecated'
        AND COALESCE(status, 'active') != 'archived'
        ORDER BY confidence DESC, helpful_count DESC
        LIMIT ?
    """, (agent_id, layer, limit)).fetchall()
    return [dict(r) for r in rows]


def count_nodes_by_layer(db: sqlite3.Connection, agent_id: str) -> dict:
    """Count active nodes per layer for an agent (excludes archived)."""
    rows = db.execute("""
        SELECT layer, COUNT(*) as cnt FROM kg_nodes
        WHERE agent_id = ? AND maturity != 'deprecated'
        AND COALESCE(status, 'active') != 'archived'
        GROUP BY layer
    """, (agent_id,)).fetchall()
    return {r["layer"]: r["cnt"] for r in rows}


def update_node_confidence(db: sqlite3.Connection, node_id: int,
                           delta: float) -> float:
    """Adjust confidence by delta. Clamps to [0.10, MAX_CONFIDENCE]. Returns new value."""
    node = get_node(db, node_id)
    if not node:
        return 0.0
    new_conf = max(0.10, min(MAX_CONFIDENCE, node["confidence"] + delta))
    db.execute(
        "UPDATE kg_nodes SET confidence = ?, last_seen = datetime('now') WHERE id = ?",
        (new_conf, node_id)
    )
    db.commit()
    return new_conf


def archive_node(db: sqlite3.Connection, node_id: int,
                  reason: str, superseded_by: int = None):
    """Archive a node (soft-delete). Does NOT delete -- preserved for audit."""
    db.execute("""
        UPDATE kg_nodes SET status = 'archived', archived_reason = ?,
        superseded_by = ? WHERE id = ?
    """, (reason, superseded_by, node_id))
    db.commit()


# ---------------------------------------------------------------------------
# KG EDGES: Typed relationships
# ---------------------------------------------------------------------------

def add_edge(db: sqlite3.Connection, agent_id: str,
             from_node_id: int, to_node_id: int,
             edge_type: str, weight: float = 1.0,
             metadata: dict = None) -> int:
    """Add a KG edge. Returns edge ID or -1 if duplicate."""
    try:
        cursor = db.execute("""
            INSERT INTO kg_edges (agent_id, from_node_id, to_node_id,
                                  edge_type, weight, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_id, from_node_id, to_node_id, edge_type, weight,
              json.dumps(metadata or {})))
        db.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return -1  # Duplicate edge


def get_edges_from(db: sqlite3.Connection, node_id: int,
                   edge_type: str = None) -> list:
    """Get outgoing edges from a node."""
    if edge_type:
        rows = db.execute("""
            SELECT e.*, n.content as to_content, n.layer as to_layer,
                   n.confidence as to_confidence
            FROM kg_edges e JOIN kg_nodes n ON e.to_node_id = n.id
            WHERE e.from_node_id = ? AND e.edge_type = ?
        """, (node_id, edge_type)).fetchall()
    else:
        rows = db.execute("""
            SELECT e.*, n.content as to_content, n.layer as to_layer,
                   n.confidence as to_confidence
            FROM kg_edges e JOIN kg_nodes n ON e.to_node_id = n.id
            WHERE e.from_node_id = ?
        """, (node_id,)).fetchall()
    return [dict(r) for r in rows]


def get_edges_to(db: sqlite3.Connection, node_id: int,
                 edge_type: str = None) -> list:
    """Get incoming edges to a node."""
    if edge_type:
        rows = db.execute("""
            SELECT e.*, n.content as from_content, n.layer as from_layer,
                   n.confidence as from_confidence
            FROM kg_edges e JOIN kg_nodes n ON e.from_node_id = n.id
            WHERE e.to_node_id = ? AND e.edge_type = ?
        """, (node_id, edge_type)).fetchall()
    else:
        rows = db.execute("""
            SELECT e.*, n.content as from_content, n.layer as from_layer,
                   n.confidence as from_confidence
            FROM kg_edges e JOIN kg_nodes n ON e.from_node_id = n.id
            WHERE e.to_node_id = ?
        """, (node_id,)).fetchall()
    return [dict(r) for r in rows]


def count_supports(db: sqlite3.Connection, node_id: int) -> int:
    """Count SUPPORTS edges pointing to a node."""
    row = db.execute("""
        SELECT COUNT(*) as cnt FROM kg_edges
        WHERE to_node_id = ? AND edge_type = ?
    """, (node_id, EDGE_SUPPORTS)).fetchone()
    return row["cnt"] if row else 0


def has_unresolved_contradictions(db: sqlite3.Connection, node_id: int) -> bool:
    """Check if node has CONTRADICTS edges where both nodes are active."""
    rows = db.execute("""
        SELECT e.id FROM kg_edges e
        JOIN kg_nodes n1 ON e.from_node_id = n1.id
        JOIN kg_nodes n2 ON e.to_node_id = n2.id
        WHERE (e.from_node_id = ? OR e.to_node_id = ?)
        AND e.edge_type = ?
        AND n1.maturity != 'deprecated' AND n2.maturity != 'deprecated'
    """, (node_id, node_id, EDGE_CONTRADICTS)).fetchall()
    return len(rows) > 0


# ---------------------------------------------------------------------------
# ERROR FAMILY HELPERS
# ---------------------------------------------------------------------------

def find_nodes_by_error_family(db: sqlite3.Connection, agent_id: str,
                                error_family: str, exclude_ids: list = None,
                                limit: int = 5) -> list:
    """Find nodes tagged with a specific error family."""
    rows = db.execute("""
        SELECT id, content, layer, confidence, error_family FROM kg_nodes
        WHERE agent_id = ? AND maturity != 'deprecated'
        AND error_family = ?
        ORDER BY confidence DESC LIMIT ?
    """, (agent_id, error_family, limit + len(exclude_ids or []))).fetchall()
    exclude = set(exclude_ids or [])
    return [dict(r) for r in rows if r["id"] not in exclude][:limit]


# ---------------------------------------------------------------------------
# FTS5 RETRIEVAL
# ---------------------------------------------------------------------------

def fts5_search(db: sqlite3.Connection, query: str, agent_id: str,
                top_k: int = 10, category_filter: str = None,
                layer_filter: str = None,
                min_confidence: float = MIN_CONFIDENCE,
                domain_filter: str = None) -> list:
    """FTS5 search over kg_nodes. Returns list of node dicts with fts_rank.
    domain_filter: if set, only return nodes with matching domain."""
    words = _clean_query(query)
    if not words:
        return []
    fts_query = " OR ".join(words)
    try:
        sql = """
            SELECT n.*, rank AS fts_rank
            FROM kg_fts
            JOIN kg_nodes n ON kg_fts.rowid = n.id
            WHERE kg_fts MATCH ? AND n.agent_id = ?
            AND n.maturity != 'deprecated' AND n.confidence >= ?
            AND COALESCE(n.status, 'active') != 'archived'
        """
        params = [fts_query, agent_id, min_confidence]
        if category_filter:
            sql += " AND n.category = ?"
            params.append(category_filter)
        if layer_filter:
            sql += " AND n.layer = ?"
            params.append(layer_filter)
        if domain_filter:
            sql += " AND n.domain = ?"
            params.append(domain_filter)
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k * 3)  # Overfetch for scoring
        raw = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    scored = []
    for r in raw:
        d = dict(r)
        score = -d["fts_rank"]  # FTS5 rank is negative
        # Helpful boost
        helpful = d.get("helpful_count", 0)
        if helpful > 0:
            score *= (1 + min(helpful / 10, 0.5))
        # Layer boost: episteme > techne > sophia (sophia always shown)
        if d["layer"] == LAYER_EPISTEME:
            score *= 2.0
        elif d["maturity"] == "crystallized":
            score *= 1.5
        scored.append((d, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# DEDUPLICATION
# ---------------------------------------------------------------------------

def is_duplicate_node(db: sqlite3.Connection, agent_id: str,
                      content: str, threshold: float = 0.8,
                      error_family: str = "") -> bool:
    """Check if content duplicates an existing node.
    threshold: Jaccard similarity threshold (0.8 for general, 0.65 for same error_family).
    error_family: if provided, same-family nodes use a lower threshold (0.65)."""
    content_lower = content.lower().strip()
    if len(content_lower) < 10:
        return True

    # Exact match
    exact = db.execute(
        "SELECT id FROM kg_nodes WHERE agent_id = ? AND LOWER(content) = ? AND maturity != 'deprecated'",
        (agent_id, content_lower)
    ).fetchone()
    if exact:
        return True

    # FTS5 candidate search
    words = _clean_query(content)
    if not words:
        return False
    fts_query = " OR ".join(words[:5])
    try:
        candidates = db.execute("""
            SELECT n.content AS node_content, n.error_family AS node_error_family
            FROM kg_fts
            JOIN kg_nodes n ON kg_fts.rowid = n.id
            WHERE kg_fts MATCH ? AND n.agent_id = ? AND n.maturity != 'deprecated'
            LIMIT 20
        """, (fts_query, agent_id)).fetchall()
    except sqlite3.OperationalError:
        return False

    for row in candidates:
        existing = row["node_content"].lower()
        if content_lower in existing or existing in content_lower:
            return True
        words_new = set(content_lower.split())
        words_old = set(existing.split())
        if not words_new or not words_old:
            continue
        intersection = words_new & words_old
        union = words_new | words_old
        jaccard = len(intersection) / len(union)
        # Same error_family -> lower threshold (more aggressive dedup within domain)
        node_family = row["node_error_family"] if "node_error_family" in row.keys() else ""
        if error_family and node_family and error_family == node_family:
            if jaccard > 0.65:
                return True
        elif jaccard > threshold:
            return True
    return False


def _bump_existing_node(db: sqlite3.Connection, agent_id: str, content: str):
    """Find closest match and bump its last_seen."""
    words = _clean_query(content)
    if not words:
        return
    fts_query = " OR ".join(words[:5])
    try:
        row = db.execute("""
            SELECT n.id FROM kg_fts
            JOIN kg_nodes n ON kg_fts.rowid = n.id
            WHERE kg_fts MATCH ? AND n.agent_id = ? AND n.maturity != 'deprecated'
            ORDER BY rank LIMIT 1
        """, (fts_query, agent_id)).fetchone()
        if row:
            db.execute(
                "UPDATE kg_nodes SET last_seen = datetime('now') WHERE id = ?",
                (row["id"],)
            )
            db.commit()
    except sqlite3.OperationalError:
        pass


# ---------------------------------------------------------------------------
# AISTHESIS: Session working memory (L1)
# ---------------------------------------------------------------------------

def record_aisthesis(db: sqlite3.Connection, agent_id: str,
                     session_id: str, event_type: str,
                     tool_name: str = "", summary: str = "",
                     command: str = "", error_text: str = "",
                     files_touched: list = None) -> int:
    """Record a working memory entry. Returns row ID."""
    cursor = db.execute("""
        INSERT INTO aisthesis (agent_id, session_id, event_type, tool_name,
                               summary, command, error_text, files_touched)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (agent_id, session_id, event_type, tool_name, summary,
          command, error_text, json.dumps(files_touched or [])))
    db.commit()
    return cursor.lastrowid


def get_session_memory(db: sqlite3.Connection, agent_id: str,
                       session_id: str, limit: int = 30) -> list:
    """Get working memory for current session."""
    rows = db.execute("""
        SELECT * FROM aisthesis
        WHERE agent_id = ? AND session_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (agent_id, session_id, limit)).fetchall()
    return [dict(r) for r in rows]


def cleanup_old_aisthesis(db: sqlite3.Connection, agent_id: str,
                          ttl_minutes: int = 30):
    """Delete working memory older than TTL."""
    cutoff = (datetime.utcnow() - timedelta(minutes=ttl_minutes)).isoformat()
    db.execute(
        "DELETE FROM aisthesis WHERE agent_id = ? AND created_at < ?",
        (agent_id, cutoff)
    )
    db.commit()


# ---------------------------------------------------------------------------
# SURFACED NODES: Feedback tracking
# ---------------------------------------------------------------------------

def record_surfaced(db: sqlite3.Connection, agent_id: str,
                    session_id: str, node_ids: list):
    """Record which nodes were shown to the agent."""
    for nid in node_ids:
        db.execute("""
            INSERT INTO surfaced_nodes (agent_id, session_id, node_id)
            VALUES (?, ?, ?)
        """, (agent_id, session_id, nid))
        db.execute(
            "UPDATE kg_nodes SET surface_count = surface_count + 1 WHERE id = ?",
            (nid,)
        )
    db.commit()


def update_feedback(db: sqlite3.Connection, surfaced_id: int,
                    feedback: str):
    """Update feedback on a surfaced node (confirmed/refuted/irrelevant)."""
    db.execute(
        "UPDATE surfaced_nodes SET feedback = ? WHERE id = ?",
        (feedback, surfaced_id)
    )
    db.commit()


# ---------------------------------------------------------------------------
# REFLECTION RUNS: Audit trail
# ---------------------------------------------------------------------------

def start_reflection_run(db: sqlite3.Connection, agent_id: str,
                         session_id: str = "",
                         trigger_type: str = "session_end") -> int:
    """Start a reflection run. Returns run ID."""
    cursor = db.execute("""
        INSERT INTO reflection_runs (agent_id, session_id, trigger_type)
        VALUES (?, ?, ?)
    """, (agent_id, session_id, trigger_type))
    db.commit()
    return cursor.lastrowid


def complete_reflection_run(db: sqlite3.Connection, run_id: int,
                            status: str = "completed",
                            events_processed: int = 0,
                            nodes_created: int = 0,
                            edges_created: int = 0,
                            duration_ms: int = 0,
                            error_text: str = "",
                            steps_json: dict = None):
    """Complete a reflection run with results."""
    db.execute("""
        UPDATE reflection_runs
        SET status = ?, events_processed = ?, nodes_created = ?,
            edges_created = ?, duration_ms = ?, error_text = ?,
            steps_json = ?
        WHERE id = ?
    """, (status, events_processed, nodes_created, edges_created,
          duration_ms, error_text, json.dumps(steps_json or {}), run_id))
    db.commit()


# ---------------------------------------------------------------------------
# SOPHIA: Human-approved axioms (L5)
# ---------------------------------------------------------------------------

def add_axiom(db: sqlite3.Connection, agent_id: str, content: str,
              category: str = "general", approved_by: str = "human") -> int:
    """Add a sophia axiom. Returns axiom ID."""
    cursor = db.execute("""
        INSERT INTO sophia (agent_id, content, category, approved_by)
        VALUES (?, ?, ?, ?)
    """, (agent_id, content, category, approved_by))
    db.commit()
    return cursor.lastrowid


def get_axioms(db: sqlite3.Connection, agent_id: str) -> list:
    """Get all sophia axioms for an agent."""
    rows = db.execute(
        "SELECT * FROM sophia WHERE agent_id = ? ORDER BY created_at",
        (agent_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------------

def get_stats(db: sqlite3.Connection, agent_id: str) -> dict:
    """Get comprehensive stats for an agent's hermes.db."""
    stats = {}

    # Node counts by layer
    stats["nodes_by_layer"] = count_nodes_by_layer(db, agent_id)

    # Total nodes
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM kg_nodes WHERE agent_id = ? AND maturity != 'deprecated'",
        (agent_id,)
    ).fetchone()
    stats["total_nodes"] = row["cnt"] if row else 0

    # Maturity counts
    rows = db.execute("""
        SELECT maturity, COUNT(*) as cnt FROM kg_nodes
        WHERE agent_id = ? GROUP BY maturity
    """, (agent_id,)).fetchall()
    stats["maturity"] = {r["maturity"]: r["cnt"] for r in rows}

    # Event count
    stats["total_events"] = count_events(db, agent_id)

    # Edge count
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM kg_edges WHERE agent_id = ?",
        (agent_id,)
    ).fetchone()
    stats["total_edges"] = row["cnt"] if row else 0

    # Axiom count
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM sophia WHERE agent_id = ?",
        (agent_id,)
    ).fetchone()
    stats["axioms"] = row["cnt"] if row else 0

    # Reflection runs
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM reflection_runs WHERE agent_id = ?",
        (agent_id,)
    ).fetchone()
    stats["reflection_runs"] = row["cnt"] if row else 0

    return stats


# ---------------------------------------------------------------------------
# QUERY HELPERS (shared with v1)
# ---------------------------------------------------------------------------

def _extract_technical_tokens(word: str) -> list:
    """Extract searchable tokens from a technical term."""
    word = word.strip(".,;:!?'\"()-[]{}#")
    if not word:
        return []
    parts = re.split(r'[_=./\\]+', word.lower())
    tokens = []
    for p in parts:
        p = re.sub(r'[^a-z0-9]', '', p)
        if len(p) > 2 and p not in STOP_WORDS:
            tokens.append(p)
    return tokens


def _clean_query(text: str) -> list:
    """Extract meaningful search words from text."""
    words = []
    seen = set()
    for w in text.lower().split():
        tokens = _extract_technical_tokens(w)
        for t in tokens:
            if t not in seen:
                words.append(t)
                seen.add(t)
    return words[:15]
