#!/usr/bin/env python3
"""Hermes Layer 1: Working Memory.

Session-scoped ephemeral context store. Tracks recent tool calls, errors,
file touches, and task intent within a single session. Auto-purges stale
entries beyond TTL.

Aristotelian analogy: phantasmata -- the immediate sensory impressions
that inform current reasoning but do not persist long-term.

Zero external dependencies. stdlib only."""

import json
import sqlite3
from datetime import datetime, timedelta

# TTL for working memory entries (30 minutes)
WM_TTL_MINUTES = 30
# Max entries per session to prevent bloat
WM_MAX_PER_SESSION = 50
# Max context lines returned to hooks
WM_CONTEXT_LIMIT = 5


def record_tool_event(db: sqlite3.Connection, session_id: str,
                      event_type: str, tool_name: str = "",
                      summary: str = "", command: str = "",
                      error_text: str = "",
                      files_touched: list = None):
    """Record a tool event in working memory.

    event_type: 'command', 'error', 'file_write', 'file_read', 'success'
    """
    if not session_id:
        return
    # Enforce per-session limit: delete oldest if at capacity
    count = db.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
        (session_id,)
    ).fetchone()[0]
    if count >= WM_MAX_PER_SESSION:
        db.execute("""
            DELETE FROM working_memory WHERE id IN (
                SELECT id FROM working_memory
                WHERE session_id = ? ORDER BY created_at ASC LIMIT 5
            )
        """, (session_id,))

    db.execute("""
        INSERT INTO working_memory
        (session_id, event_type, tool_name, summary, command, error_text,
         files_touched)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, event_type, tool_name,
        summary[:200] if summary else "",
        command[:300] if command else "",
        error_text[:500] if error_text else "",
        json.dumps(files_touched[:10] if files_touched else []),
    ))
    db.commit()


def get_recent_context(db: sqlite3.Connection, session_id: str,
                       limit: int = WM_CONTEXT_LIMIT) -> str:
    """Get formatted recent context for injection into prompts.

    Returns a compact summary of recent activity, most recent first."""
    if not session_id:
        return ""
    rows = db.execute("""
        SELECT event_type, tool_name, summary, command, error_text,
               created_at
        FROM working_memory
        WHERE session_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (session_id, limit)).fetchall()

    if not rows:
        return ""

    lines = ["## Working Memory (recent session context)"]
    for r in rows:
        etype = r["event_type"]
        if etype == "error":
            err = r["error_text"][:120] if r["error_text"] else "unknown"
            lines.append(f"- [ERROR] {r['tool_name']}: {err}")
        elif etype == "command":
            cmd = r["command"][:80] if r["command"] else ""
            lines.append(f"- [CMD] {cmd}")
        elif etype == "file_write":
            lines.append(f"- [WRITE] {r['summary']}")
        elif etype == "success":
            lines.append(f"- [OK] {r['tool_name']}: {r['summary'][:80]}")
        else:
            lines.append(f"- [{etype.upper()}] {r['summary'][:100]}")

    return "\n".join(lines)


def get_recent_errors(db: sqlite3.Connection, session_id: str,
                      limit: int = 3) -> list:
    """Get recent errors for contextual error matching.

    Returns list of dicts with error_text and tool_name."""
    if not session_id:
        return []
    rows = db.execute("""
        SELECT tool_name, error_text, command, created_at
        FROM working_memory
        WHERE session_id = ? AND event_type = 'error'
        ORDER BY created_at DESC LIMIT ?
    """, (session_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_active_files(db: sqlite3.Connection, session_id: str,
                     limit: int = 10) -> list:
    """Get files recently touched in this session.

    Returns deduplicated list of file paths."""
    if not session_id:
        return []
    rows = db.execute("""
        SELECT files_touched FROM working_memory
        WHERE session_id = ? AND files_touched != '[]'
        ORDER BY created_at DESC LIMIT 20
    """, (session_id,)).fetchall()

    seen = set()
    files = []
    for r in rows:
        try:
            touched = json.loads(r["files_touched"])
            for f in touched:
                if f and f not in seen:
                    seen.add(f)
                    files.append(f)
                    if len(files) >= limit:
                        return files
        except (json.JSONDecodeError, TypeError):
            continue
    return files


def purge_session(db: sqlite3.Connection, session_id: str) -> int:
    """Remove all working memory for a completed session.

    Called by Stop hook after learning extraction."""
    if not session_id:
        return 0
    cursor = db.execute(
        "DELETE FROM working_memory WHERE session_id = ?",
        (session_id,)
    )
    db.commit()
    return cursor.rowcount


def purge_stale(db: sqlite3.Connection,
                ttl_minutes: int = WM_TTL_MINUTES) -> int:
    """Remove working memory entries older than TTL.

    Safety net for sessions that don't clean up properly."""
    cutoff = (datetime.utcnow() - timedelta(minutes=ttl_minutes)).isoformat()
    cursor = db.execute(
        "DELETE FROM working_memory WHERE created_at < ?",
        (cutoff,)
    )
    db.commit()
    return cursor.rowcount


def session_summary(db: sqlite3.Connection, session_id: str) -> dict:
    """Get session statistics for the Stop hook.

    Returns dict with counts by event_type, total events, unique files."""
    if not session_id:
        return {}
    rows = db.execute("""
        SELECT event_type, COUNT(*) as cnt
        FROM working_memory WHERE session_id = ?
        GROUP BY event_type
    """, (session_id,)).fetchall()

    stats = {r["event_type"]: r["cnt"] for r in rows}
    stats["total"] = sum(stats.values())
    stats["unique_files"] = len(get_active_files(db, session_id))
    return stats
