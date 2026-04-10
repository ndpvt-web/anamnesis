#!/usr/bin/env python3
"""Hermes Hook: SessionStart - Morning briefing.

Injects top anti-patterns, preferences, and principles from the agent's
own learnings.db into the session context. Also pulls from global cortex.db
as a supplementary source (read-only, for backward compatibility).

Replaces the old cortex on_session_start.py which queried WRONG legacy tables.
This version queries the UNIFIED rules table via FTS5.

Must be fast (<200ms) -- pure SQLite queries, no LLM calls.
Zero external dependencies. stdlib only."""

import json
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from learnings_db import detect_active_agent, open_db, log_event
from crystallized import maintenance_cycle

# Global cortex.db for supplementary rules (read-only)
CORTEX_DB_PATH = Path.home() / ".claude" / "skills" / "capy-cortex" / "cortex.db"

# Noise filter
NOISE_SUBSTRINGS = [
    "session is being continued",
    "Base directory for this skill",
    "<teammate-message",
    "<system-reminder>",
]

MAX_ANTI_PATTERNS = 5
MAX_PREFERENCES = 3
MAX_PRINCIPLES = 5


def _is_noise(content):
    """Filter garbage that shouldn't be injected."""
    if not content or len(content) > 300:
        return True
    content_lower = content.lower()
    for noise in NOISE_SUBSTRINGS:
        if noise.lower() in content_lower:
            return True
    if content.strip().startswith("<"):
        return True
    return False


def _query_agent_rules(db):
    """Query the agent's learnings.db for top rules by category."""
    results = {"anti_pattern": [], "preference": [], "principle": []}
    for cat in results:
        rows = db.execute("""
            SELECT content, severity, confidence, helpful_count, maturity
            FROM rules
            WHERE category = ? AND maturity != 'deprecated' AND confidence >= 0.5
            ORDER BY
                CASE maturity WHEN 'crystallized' THEN 0 ELSE 1 END,
                confidence DESC, helpful_count DESC
            LIMIT 10
        """, (cat,)).fetchall()
        for r in rows:
            if not _is_noise(r["content"]):
                results[cat].append(dict(r))
    return results


def _query_cortex_supplementary():
    """Pull high-confidence rules from global cortex.db (read-only).
    Only used as a supplement when agent's own DB is sparse."""
    if not CORTEX_DB_PATH.exists():
        return {"anti_pattern": [], "preference": [], "principle": []}
    try:
        db = sqlite3.connect(str(CORTEX_DB_PATH))
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=2000")
        results = {"anti_pattern": [], "preference": [], "principle": []}
        # Query the UNIFIED rules table (not the broken legacy tables)
        for cat in results:
            rows = db.execute("""
                SELECT content, confidence, helpful_count
                FROM rules
                WHERE category = ? AND maturity != 'deprecated'
                AND confidence >= 0.7
                ORDER BY helpful_count DESC, confidence DESC
                LIMIT 5
            """, (cat,)).fetchall()
            for r in rows:
                if not _is_noise(r["content"]):
                    results[cat].append(dict(r))
        db.close()
        return results
    except Exception:
        return {"anti_pattern": [], "preference": [], "principle": []}


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        data = {}

    agent_id = detect_active_agent()
    if not agent_id:
        return

    try:
        db = open_db(agent_id)

        # Run crystallization maintenance at session start (housekeeping)
        maint = maintenance_cycle(db)

        # Get agent's own rules
        agent_rules = _query_agent_rules(db)

        # Count total active rules
        rule_count = db.execute(
            "SELECT COUNT(*) FROM rules WHERE maturity != 'deprecated'"
        ).fetchone()[0]

        # Log session start
        log_event(db, "session_start", {
            "agent_id": agent_id,
            "session_id": data.get("session_id", ""),
            "cwd": data.get("cwd", ""),
            "rule_count": rule_count,
            "maintenance": maint,
        })
        db.close()

        # --- Hermes v2 dual-write (event capture + maintenance, never crash) ---
        _v2_briefing = ""
        v2_conn = None
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from v2.observer import safe_append_event
            from v2.maintenance import maintenance_cycle as v2_maintenance
            from v2.retrieval import (format_session_briefing,
                                       format_query_based_briefing,
                                       dual_retrieve, classify_error_family)
            from v2 import db as v2db
            safe_append_event(agent_id, 'session_start', session_id=data.get('session_id', ''),
                              metadata={'cwd': data.get('cwd', ''), 'rule_count': rule_count})
            v2_conn = v2db.open_db(agent_id)
            v2_maintenance(v2_conn, agent_id)
            # Query-based briefing: use cwd + initial prompt for domain context
            domain_query = f"{data.get('cwd', '')} {data.get('prompt', '')}".strip()
            if domain_query and len(domain_query) > 5:
                error_family = classify_error_family(domain_query)
                v2_results = dual_retrieve(v2_conn, agent_id, domain_query,
                                            top_k=5, error_family=error_family)
                _v2_briefing = format_query_based_briefing(v2_conn, agent_id, v2_results)
            else:
                _v2_briefing = format_session_briefing(v2_conn, agent_id)
        except Exception:
            pass  # v2 failure must never affect v1
        finally:
            if v2_conn:
                try:
                    v2_conn.close()
                except Exception:
                    pass

        # Supplement from cortex.db if agent's DB is sparse
        cortex_rules = {"anti_pattern": [], "preference": [], "principle": []}
        if rule_count < 20:
            cortex_rules = _query_cortex_supplementary()

        # Merge: agent rules first, then cortex supplement (deduplicated)
        merged = {}
        for cat in ("anti_pattern", "preference", "principle"):
            seen_content = set()
            merged[cat] = []
            for r in agent_rules.get(cat, []):
                key = r["content"].lower().strip()
                if key not in seen_content:
                    seen_content.add(key)
                    merged[cat].append(r)
            for r in cortex_rules.get(cat, []):
                key = r["content"].lower().strip()
                if key not in seen_content:
                    seen_content.add(key)
                    merged[cat].append(r)

        # Build output
        lines = []
        anti = merged["anti_pattern"][:MAX_ANTI_PATTERNS]
        prefs = merged["preference"][:MAX_PREFERENCES]
        princs = merged["principle"][:MAX_PRINCIPLES]

        if anti or prefs or princs:
            lines.append(f"## Cortex v3: Knowledge")
            lines.append(f"({rule_count} rules, agent: {agent_id})\n")

        if anti:
            lines.append("### NEVER DO (learned from past mistakes)")
            for ap in anti:
                sev = ap.get("severity", "medium")
                proven = " [PROVEN]" if ap.get("maturity") == "crystallized" else ""
                lines.append(f"- [{sev.upper()}]{proven} {ap['content']}")
            lines.append("")

        if prefs:
            lines.append("### User Preferences")
            for p in prefs:
                lines.append(f"- {p['content']}")
            lines.append("")

        if princs:
            lines.append("### Principles (distilled from experience)")
            for p in princs:
                proven = " [PROVEN]" if p.get("maturity") == "crystallized" else ""
                lines.append(f"-{proven} {p['content']}")
            lines.append("")

        # Prepend v2 briefing if available
        if _v2_briefing:
            lines.insert(0, "")
            lines.insert(0, _v2_briefing)

        if lines:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n".join(lines)
                }
            }
            print(json.dumps(output))

    except Exception:
        pass  # Hooks must NEVER crash


if __name__ == "__main__":
    main()
