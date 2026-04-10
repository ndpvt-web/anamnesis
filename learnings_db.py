#!/usr/bin/env python3
"""Hermes Micro-Cortex: Per-agent SQLite + FTS5 knowledge base.

Each agent gets its own learnings.db in ~/.happycapy/agents/<id>/learnings/.
Provides: schema init, CRUD, FTS5 retrieval, dedup, and feedback credit.

Zero external dependencies. stdlib only."""

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime

HAPPYCAPY_DIR = Path.home() / ".happycapy"
AGENTS_DIR = HAPPYCAPY_DIR / "agents"
CONFIG_PATH = HAPPYCAPY_DIR / "happycapy.json"

# FTS5 stop words for query cleaning
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "i", "you", "he", "she",
    "it", "we", "they", "me", "my", "your", "his", "her", "its", "our",
    "their", "this", "that", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "and", "or", "not", "but", "if", "then", "so", "just",
    "want", "need", "please", "help", "make", "let", "get",
})

# Confidence thresholds
MIN_CONFIDENCE = 0.5
MAX_CONFIDENCE = 0.95
CONFIDENCE_BOOST = 0.02
CONFIDENCE_DECAY_RATE = 0.10  # 10% decay per maintenance cycle


def db_path_for_agent(agent_id: str) -> Path:
    """Return path to agent's learnings.db."""
    return AGENTS_DIR / agent_id / "learnings" / "learnings.db"


def open_db(agent_id: str) -> sqlite3.Connection:
    """Open (and init if needed) the agent's learning database."""
    path = db_path_for_agent(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=2000")
    _ensure_schema(db)
    return db


def _ensure_schema(db: sqlite3.Connection):
    """Create tables and FTS5 index if they don't exist."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            severity TEXT DEFAULT 'medium',
            confidence REAL DEFAULT 0.7,
            occurrences INTEGER DEFAULT 1,
            helpful_count INTEGER DEFAULT 0,
            applies_when TEXT DEFAULT '',
            source_session TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            last_seen TEXT DEFAULT (datetime('now')),
            maturity TEXT DEFAULT 'candidate'
        );

        CREATE TABLE IF NOT EXISTS surfaced_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            rule_id INTEGER NOT NULL,
            credited INTEGER DEFAULT 0,
            surfaced_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (rule_id) REFERENCES rules(id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            tool_name TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            command TEXT DEFAULT '',
            error_text TEXT DEFAULT '',
            files_touched TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_rules_category ON rules(category);
        CREATE INDEX IF NOT EXISTS idx_rules_confidence ON rules(confidence);
        CREATE INDEX IF NOT EXISTS idx_rules_maturity ON rules(maturity);
        CREATE INDEX IF NOT EXISTS idx_surfaced_session ON surfaced_rules(session_id, credited);
        CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id, created_at);
    """)
    # FTS5: content-sync'd to rules table
    try:
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
                content, category,
                content='rules',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
        # Auto-sync triggers
        db.executescript("""
            CREATE TRIGGER IF NOT EXISTS rules_ai AFTER INSERT ON rules BEGIN
                INSERT INTO rules_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
            END;
            CREATE TRIGGER IF NOT EXISTS rules_ad AFTER DELETE ON rules BEGIN
                INSERT INTO rules_fts(rules_fts, rowid, content, category)
                VALUES ('delete', old.id, old.content, old.category);
            END;
            CREATE TRIGGER IF NOT EXISTS rules_au AFTER UPDATE ON rules BEGIN
                INSERT INTO rules_fts(rules_fts, rowid, content, category)
                VALUES ('delete', old.id, old.content, old.category);
                INSERT INTO rules_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
            END;
        """)
    except sqlite3.OperationalError:
        pass  # Already exists
    db.commit()


# ---------------------------------------------------------------------------
# CRUD: Add rules with deduplication
# ---------------------------------------------------------------------------

def is_duplicate(db: sqlite3.Connection, content: str, threshold: float = 0.7) -> bool:
    """Check if content is a near-duplicate of an existing rule.
    Uses FTS5 to find candidates, then substring overlap check."""
    content_lower = content.lower().strip()
    if len(content_lower) < 10:
        return True  # Too short to be meaningful

    # Exact match check first
    exact = db.execute(
        "SELECT id FROM rules WHERE LOWER(content) = ? AND maturity != 'deprecated'",
        (content_lower,)
    ).fetchone()
    if exact:
        return True

    # FTS5 candidate search for substring overlap
    words = _clean_query(content)
    if not words:
        return False
    fts_query = " OR ".join(words[:5])
    try:
        candidates = db.execute("""
            SELECT r.content AS rule_content FROM rules_fts
            JOIN rules r ON rules_fts.rowid = r.id
            WHERE rules_fts MATCH ? AND r.maturity != 'deprecated'
            LIMIT 20
        """, (fts_query,)).fetchall()
    except sqlite3.OperationalError:
        return False

    for row in candidates:
        existing = row["rule_content"].lower()
        # Bidirectional substring check (either contains the other)
        if content_lower in existing or existing in content_lower:
            return True
        # Jaccard similarity on word sets
        words_new = set(content_lower.split())
        words_old = set(existing.split())
        if not words_new or not words_old:
            continue
        intersection = words_new & words_old
        union = words_new | words_old
        if len(intersection) / len(union) > threshold:
            return True
    return False


def add_rule(db: sqlite3.Connection, content: str, category: str = "general",
             severity: str = "medium", applies_when: str = "",
             source_session: str = "") -> int:
    """Add a rule if not duplicate. Returns rule ID or -1 if deduped."""
    if is_duplicate(db, content):
        # Bump occurrences of the closest match
        _bump_existing(db, content)
        return -1
    cursor = db.execute("""
        INSERT INTO rules (content, category, severity, applies_when, source_session)
        VALUES (?, ?, ?, ?, ?)
    """, (content, category, severity, applies_when, source_session))
    db.commit()
    return cursor.lastrowid


def _bump_existing(db: sqlite3.Connection, content: str):
    """Find closest match and bump its occurrences + last_seen."""
    words = _clean_query(content)
    if not words:
        return
    fts_query = " OR ".join(words[:5])
    try:
        row = db.execute("""
            SELECT r.id FROM rules_fts
            JOIN rules r ON rules_fts.rowid = r.id
            WHERE rules_fts MATCH ? AND r.maturity != 'deprecated'
            ORDER BY rank LIMIT 1
        """, (fts_query,)).fetchone()
        if row:
            db.execute("""
                UPDATE rules SET occurrences = occurrences + 1,
                last_seen = datetime('now') WHERE id = ?
            """, (row["id"],))
            db.commit()
    except sqlite3.OperationalError:
        pass


# ---------------------------------------------------------------------------
# RETRIEVAL: FTS5 search with quality gates
# ---------------------------------------------------------------------------

def fts5_retrieve(db: sqlite3.Connection, query: str, top_k: int = 5,
                  category_filter: str = None) -> list:
    """Retrieve top-K rules by FTS5 relevance. Returns list of dicts."""
    words = _clean_query(query)
    if not words:
        return []
    fts_query = " OR ".join(words)
    try:
        sql = """
            SELECT r.id, r.content, r.category, r.severity, r.confidence,
                   r.helpful_count, r.occurrences, r.applies_when,
                   r.maturity, rank AS fts_rank
            FROM rules_fts
            JOIN rules r ON rules_fts.rowid = r.id
            WHERE rules_fts MATCH ? AND r.maturity != 'deprecated'
        """
        params = [fts_query]
        if category_filter:
            sql += " AND r.category = ?"
            params.append(category_filter)
        sql += " ORDER BY rank LIMIT 30"
        raw = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    scored = []
    for r in raw:
        if r["confidence"] < MIN_CONFIDENCE:
            continue
        score = -r["fts_rank"]  # FTS5 rank is negative (lower = better)
        helpful = r["helpful_count"] or 0
        if helpful > 0:
            score *= (1 + min(helpful / 10, 0.5))
        # Crystallized rules get priority boost (Layer 4)
        if r["maturity"] == "crystallized":
            score *= 2.0
        scored.append((dict(r), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


def match_anti_patterns(db: sqlite3.Connection, command: str) -> list:
    """Find anti-patterns that match a bash command. Returns list of dicts."""
    anti = db.execute("""
        SELECT id, content, severity, confidence
        FROM rules WHERE category = 'anti_pattern'
        AND maturity != 'deprecated' AND confidence >= ?
        ORDER BY CASE severity
            WHEN 'critical' THEN 0 WHEN 'high' THEN 1
            WHEN 'medium' THEN 2 ELSE 3 END
    """, (MIN_CONFIDENCE,)).fetchall()

    cmd_lower = command.lower().strip()
    matches = []
    for row in anti:
        if _command_matches_pattern(cmd_lower, row["content"].lower()):
            matches.append(dict(row))
    return matches


def match_error_fixes(db: sqlite3.Connection, error_text: str, top_k: int = 3) -> list:
    """Find rules that address a specific error. Uses FTS5 on error text."""
    # Search across all categories -- error fixes might be in any category
    return fts5_retrieve(db, error_text, top_k=top_k)


# ---------------------------------------------------------------------------
# FEEDBACK: Surface tracking and success credit
# ---------------------------------------------------------------------------

def record_surfaced(db: sqlite3.Connection, session_id: str, rule_ids: list):
    """Record which rules were shown for later success crediting."""
    db.execute(
        "DELETE FROM surfaced_rules WHERE session_id = ? AND credited = 0",
        (session_id,)
    )
    for rid in rule_ids:
        db.execute(
            "INSERT INTO surfaced_rules (session_id, rule_id) VALUES (?, ?)",
            (session_id, rid)
        )
    db.commit()


def credit_surfaced(db: sqlite3.Connection, session_id: str) -> int:
    """Credit recently surfaced rules on tool success. Returns count."""
    surfaced = db.execute("""
        SELECT id, rule_id FROM surfaced_rules
        WHERE session_id = ? AND credited = 0
        ORDER BY surfaced_at DESC LIMIT 10
    """, (session_id,)).fetchall()
    if not surfaced:
        return 0
    count = 0
    for surf_id, rule_id in surfaced:
        db.execute("""
            UPDATE rules SET helpful_count = helpful_count + 1,
            confidence = MIN(confidence + ?, ?),
            last_seen = datetime('now')
            WHERE id = ? AND maturity != 'deprecated'
        """, (CONFIDENCE_BOOST, MAX_CONFIDENCE, rule_id))
        db.execute("UPDATE surfaced_rules SET credited = 1 WHERE id = ?", (surf_id,))
        count += 1
    db.commit()
    return count


# ---------------------------------------------------------------------------
# AGENT DETECTION (shared utility)
# ---------------------------------------------------------------------------

def detect_active_agent() -> str:
    """Find the active agent. Returns agent_id or empty string.

    Detection order:
    1. CAPY_AGENT_ID env var (set by HappyCapy runtime)
    2. Scan happycapy.json for any active agent (all types)
       - Prefer agents that already have a learnings/ dir
       - Fall back to most recently created active agent
    """
    env_agent = os.environ.get("CAPY_AGENT_ID", "")
    if env_agent:
        return env_agent
    if not CONFIG_PATH.exists():
        return ""
    try:
        config = json.loads(CONFIG_PATH.read_text())
        agents = config.get("agents", {})
        with_learnings = []
        without_learnings = []
        for aid, agent in agents.items():
            if not agent.get("active", True):
                continue
            agent_dir = AGENTS_DIR / aid
            if (agent_dir / "learnings").exists():
                with_learnings.append((aid, agent.get("created", "")))
            elif agent_dir.exists():
                without_learnings.append((aid, agent.get("created", "")))
        # Prefer agents that already have learnings
        pool = with_learnings if with_learnings else without_learnings
        if not pool:
            return ""
        if len(pool) == 1:
            return pool[0][0]
        # Most recently created
        pool.sort(key=lambda x: x[1], reverse=True)
        return pool[0][0]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _extract_technical_tokens(word: str) -> list:
    """Extract searchable tokens from a technical term.

    Preserves sub-components of compound identifiers:
      '.dockerignore'  -> ['dockerignore']
      'journal_mode=WAL' -> ['journal', 'mode', 'wal']
      'pathlib.Path'   -> ['pathlib', 'path']
      'os.path.join'   -> ['path', 'join']
      'JSONEncoder'    -> ['jsonencoder']
      'busy_timeout'   -> ['busy', 'timeout']
      'check_same_thread' -> ['check', 'same', 'thread']
    """
    # Strip leading dots/punctuation but keep internal structure
    word = word.strip(".,;:!?'\"()-[]{}#")
    if not word:
        return []
    # Split on common technical separators: _ = . /
    parts = re.split(r'[_=./\\]+', word.lower())
    tokens = []
    for p in parts:
        p = re.sub(r'[^a-z0-9]', '', p)
        if len(p) > 2 and p not in STOP_WORDS:
            tokens.append(p)
    return tokens


# Concept synonym expansion: maps high-level concepts to additional
# search terms that appear in technical rules but not in natural prompts.
CONCEPT_SYNONYMS = {
    "docker": ["dockerfile", "dockerignore", "container", "multistage"],
    "database": ["sqlite", "postgres", "sql", "pragma", "wal"],
    "sqlite": ["busy_timeout", "journal_mode", "wal", "pragma", "check_same_thread"],
    "serialization": ["json", "jsonencoder", "isoformat", "datetime", "decimal"],
    "json": ["jsonencoder", "isoformat", "serializable", "datetime"],
    "import": ["circular", "importerror", "type_checking", "init"],
    "path": ["pathlib", "ospath", "join", "crossplatform"],
    "dependency": ["requirements", "pin", "freeze", "pyproject"],
    "test": ["pytest", "conftest", "importlib", "modulenotfounderror"],
    "ci": ["actions", "cache", "workflow", "pipeline", "github"],
    "git": ["merge", "conflict", "pyproject", "force"],
    "environment": ["env", "dotenv", "environ", "keyerror"],
    "containerize": ["docker", "dockerfile", "dockerignore", "multistage"],
    "pipeline": ["actions", "cache", "workflow", "github"],
    # Node.js / Web Dev domain
    "npm": ["package", "lockfile", "peer", "legacy", "node_modules"],
    "express": ["middleware", "router", "cors", "json", "bodyparser"],
    "typescript": ["tsconfig", "moduleresolution", "tsc", "esm", "declarations"],
    "vite": ["importmeta", "env", "production", "build", "frontend"],
    "cors": ["origin", "preflight", "whitelist", "accesscontrol"],
    "react": ["vite", "webpack", "jsx", "component", "frontend"],
    "node": ["npm", "express", "typescript", "package", "modules"],
    "middleware": ["express", "router", "cors", "bodyparser", "json"],
    "frontend": ["vite", "react", "webpack", "build", "env"],
}


def _clean_query(text: str) -> list:
    """Extract meaningful search words from text.

    Improvements over naive alphanumeric-only stripping:
    1. Preserves technical identifiers by splitting on _ = . / separators
    2. Expands concept synonyms for better recall on technical terms
    3. Higher word limit (15) for richer queries
    """
    words = []
    seen = set()
    for w in text.lower().split():
        tokens = _extract_technical_tokens(w)
        for t in tokens:
            if t not in seen:
                words.append(t)
                seen.add(t)
                # Synonym expansion: add related terms
                if t in CONCEPT_SYNONYMS:
                    for syn in CONCEPT_SYNONYMS[t]:
                        if syn not in seen:
                            words.append(syn)
                            seen.add(syn)
    return words[:15]


def _command_matches_pattern(cmd: str, pattern_content: str) -> bool:
    """Check if a bash command matches an anti-pattern's key phrases.

    STRICT matching: only matches on technical identifiers (paths, flags,
    dotted names, specific tool names). Generic English words are ignored
    to prevent false positives like "sessions" matching "on_session_start.py".
    """
    # Extract ONLY technical keywords from the anti-pattern
    keywords = []
    # Strip trailing punctuation only -- leading - and . are meaningful
    _trailing_punct = str.maketrans("", "", ".,;:!?'\"()")
    for word in pattern_content.split():
        word = word.rstrip(".,;:!?'\"()-")
        if "/" in word and len(word) > 5:
            keywords.append(word)  # paths
        elif word.startswith("--") and len(word) > 3:
            keywords.append(word)  # flags like --force
        elif word.startswith("-") and len(word) == 2:
            keywords.append(word)  # short flags like -rf
        elif "." in word and len(word) > 4:
            keywords.append(word)  # dotted names like package.json
        # DO NOT match generic English words -- they cause massive false positives
    if not keywords:
        return False

    # Read-only commands are never blocked
    read_prefixes = (
        "cat ", "head ", "tail ", "less ", "more ", "wc ", "file ",
        "stat ", "ls ", "find ", "grep ", "rg ", "diff ", "echo ",
        "python3 -c", "node -e",
    )
    if cmd.lstrip().startswith(read_prefixes):
        return False

    for kw in keywords:
        if kw in cmd:
            return True
    return False


def log_event(db: sqlite3.Connection, event_type: str, metadata: dict = None):
    """Log an event to the events table."""
    db.execute(
        "INSERT INTO events (event_type, metadata) VALUES (?, ?)",
        (event_type, json.dumps(metadata or {}))
    )
    db.commit()


# Need os for detect_active_agent
import os
