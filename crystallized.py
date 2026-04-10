#!/usr/bin/env python3
"""Hermes Layer 4: Crystallized Knowledge.

Promotion engine for battle-tested rules. Rules that prove themselves
through repeated use, high confidence, and positive feedback get promoted
to 'crystallized' maturity -- the highest tier.

Aristotelian analogy: episteme (scientific knowledge) + nous (intuitive
understanding). Knowledge so well-established it becomes axiomatic.

Crystallized rules get a 2x score boost in FTS5 retrieval (see
learnings_db.py fts5_retrieve). They represent the agent's most
reliable, proven knowledge.

Zero external dependencies. stdlib only."""

import sqlite3
from datetime import datetime, timedelta

# Promotion thresholds (all must be met)
PROMOTE_MIN_CONFIDENCE = 0.85
PROMOTE_MIN_HELPFUL = 3
PROMOTE_MIN_OCCURRENCES = 2
PROMOTE_MIN_AGE_HOURS = 1  # Must exist for at least 1 hour

# Demotion thresholds
DEMOTE_STALE_DAYS = 30  # Not seen in 30 days -> demote back
DEMOTE_LOW_CONFIDENCE = 0.4  # Confidence dropped -> demote

# Deprecation thresholds
DEPRECATE_CONFIDENCE = 0.3  # Below this -> deprecate entirely


def promote_rules(db: sqlite3.Connection) -> dict:
    """Scan candidate rules and promote qualified ones to crystallized.

    Returns dict with 'promoted' count and list of promoted rule IDs."""
    min_age = (datetime.utcnow() - timedelta(
        hours=PROMOTE_MIN_AGE_HOURS
    )).isoformat()

    candidates = db.execute("""
        SELECT id, content, category, confidence, helpful_count, occurrences
        FROM rules
        WHERE maturity = 'candidate'
        AND confidence >= ?
        AND helpful_count >= ?
        AND occurrences >= ?
        AND created_at <= ?
    """, (
        PROMOTE_MIN_CONFIDENCE, PROMOTE_MIN_HELPFUL,
        PROMOTE_MIN_OCCURRENCES, min_age,
    )).fetchall()

    promoted_ids = []
    for rule in candidates:
        db.execute("""
            UPDATE rules SET maturity = 'crystallized',
            last_seen = datetime('now')
            WHERE id = ?
        """, (rule["id"],))
        promoted_ids.append(rule["id"])

    if promoted_ids:
        db.commit()

    return {"promoted": len(promoted_ids), "ids": promoted_ids}


def demote_stale(db: sqlite3.Connection,
                 stale_days: int = DEMOTE_STALE_DAYS) -> dict:
    """Demote crystallized rules that haven't been seen recently.

    Also demotes rules whose confidence has dropped below threshold.
    Returns dict with 'demoted' count and 'deprecated' count."""
    stale_cutoff = (datetime.utcnow() - timedelta(
        days=stale_days
    )).isoformat()

    # Demote stale crystallized -> candidate
    stale = db.execute("""
        UPDATE rules SET maturity = 'candidate'
        WHERE maturity = 'crystallized'
        AND last_seen < ?
    """, (stale_cutoff,))
    demoted_count = stale.rowcount

    # Demote low-confidence crystallized -> candidate
    low_conf = db.execute("""
        UPDATE rules SET maturity = 'candidate'
        WHERE maturity = 'crystallized'
        AND confidence < ?
    """, (DEMOTE_LOW_CONFIDENCE,))
    demoted_count += low_conf.rowcount

    # Deprecate very low confidence rules (any maturity)
    deprecated = db.execute("""
        UPDATE rules SET maturity = 'deprecated'
        WHERE maturity != 'deprecated'
        AND confidence < ?
    """, (DEPRECATE_CONFIDENCE,))

    db.commit()
    return {
        "demoted": demoted_count,
        "deprecated": deprecated.rowcount,
    }


def get_crystallized_stats(db: sqlite3.Connection) -> dict:
    """Get statistics about the crystallized knowledge layer.

    Returns counts by maturity level and top crystallized rules."""
    maturity_counts = db.execute("""
        SELECT maturity, COUNT(*) as cnt FROM rules
        GROUP BY maturity
    """).fetchall()
    stats = {r["maturity"]: r["cnt"] for r in maturity_counts}

    top = db.execute("""
        SELECT id, content, category, confidence, helpful_count, occurrences
        FROM rules WHERE maturity = 'crystallized'
        ORDER BY helpful_count DESC, confidence DESC
        LIMIT 10
    """).fetchall()
    stats["top_crystallized"] = [dict(r) for r in top]

    return stats


def confidence_decay(db: sqlite3.Connection,
                     decay_rate: float = 0.05,
                     stale_days: int = 30) -> dict:
    """Apply time-based confidence decay to rules not seen recently.

    EverMemOS-inspired: rules that aren't re-confirmed lose confidence
    over time. 5% per maintenance cycle for rules not seen in 30+ days.
    This prevents stale knowledge from dominating retrieval.

    Returns dict with 'decayed' count."""
    stale_cutoff = (datetime.utcnow() - timedelta(
        days=stale_days
    )).isoformat()

    # Decay confidence for stale rules (but don't drop below 0.15)
    result = db.execute("""
        UPDATE rules
        SET confidence = MAX(0.15, confidence * (1.0 - ?))
        WHERE last_seen < ?
        AND confidence > 0.15
        AND maturity != 'deprecated'
    """, (decay_rate, stale_cutoff))

    db.commit()
    return {"decayed": result.rowcount}


def maintenance_cycle(db: sqlite3.Connection) -> dict:
    """Run a full maintenance cycle: promote + demote + decay + stats.

    Called by Stop hook after learning extraction, and optionally
    at session start for housekeeping."""
    promote_result = promote_rules(db)
    demote_result = demote_stale(db)
    decay_result = confidence_decay(db)
    stats = get_crystallized_stats(db)

    return {
        "promoted": promote_result["promoted"],
        "promoted_ids": promote_result["ids"],
        "demoted": demote_result["demoted"],
        "deprecated": demote_result["deprecated"],
        "decayed": decay_result["decayed"],
        "maturity_stats": {
            k: v for k, v in stats.items() if k != "top_crystallized"
        },
        "top_crystallized_count": len(stats.get("top_crystallized", [])),
    }
