#!/usr/bin/env python3
"""Hermes v2 Migration: learnings.db (v1) -> hermes.db (v2).

Each v1 rule becomes a kg_node with:
  - node_type='fact'
  - layer='episteme' if crystallized, else 'techne'
  - All original metadata carried over
  - Dedup check before each insert

Idempotent: safe to re-run (skips duplicates).
Non-destructive: learnings.db is read-only, never modified.

Usage:
  python3 migrate.py --agent surfaced-cto
  python3 migrate.py --all
  python3 migrate.py --agent surfaced-cto --dry-run
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2 import db as v2db

HAPPYCAPY_DIR = Path.home() / ".happycapy"
AGENTS_DIR = HAPPYCAPY_DIR / "agents"
CONFIG_PATH = HAPPYCAPY_DIR / "happycapy.json"


def find_agents_with_learnings() -> list:
    """Find all agent IDs that have a learnings.db."""
    agents = []
    if not AGENTS_DIR.exists():
        return agents
    for agent_dir in AGENTS_DIR.iterdir():
        if not agent_dir.is_dir():
            continue
        learnings_db = agent_dir / "learnings" / "learnings.db"
        if learnings_db.exists():
            agents.append(agent_dir.name)
    return sorted(agents)


def open_v1_db(agent_id: str) -> sqlite3.Connection:
    """Open v1 learnings.db read-only."""
    path = AGENTS_DIR / agent_id / "learnings" / "learnings.db"
    if not path.exists():
        raise FileNotFoundError(f"No learnings.db for agent {agent_id}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def migrate_agent(agent_id: str, dry_run: bool = False) -> dict:
    """Migrate a single agent's v1 rules to v2 kg_nodes.

    Returns migration stats dict."""
    t0 = time.time()
    stats = {
        "agent_id": agent_id,
        "v1_rules": 0,
        "migrated": 0,
        "skipped_duplicate": 0,
        "skipped_deprecated": 0,
        "errors": 0,
        "duration_ms": 0,
    }

    # Open v1 read-only
    try:
        v1 = open_v1_db(agent_id)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return stats

    # Count v1 rules
    v1_count = v1.execute("SELECT COUNT(*) as cnt FROM rules").fetchone()["cnt"]
    stats["v1_rules"] = v1_count
    print(f"  v1 rules: {v1_count}")

    if dry_run:
        print(f"  DRY RUN: would migrate up to {v1_count} rules")
        v1.close()
        stats["duration_ms"] = int((time.time() - t0) * 1000)
        return stats

    # Open/create v2
    v2 = v2db.open_db(agent_id)

    # Start a reflection run to log the migration
    run_id = v2db.start_reflection_run(v2, agent_id, trigger_type="migration")

    # Read all v1 rules (batch for efficiency)
    v1_rules = v1.execute("""
        SELECT id, content, category, severity, confidence, occurrences,
               helpful_count, applies_when, source_session, created_at,
               last_seen, maturity
        FROM rules ORDER BY id
    """).fetchall()

    for rule in v1_rules:
        try:
            # Skip v1 deprecated
            if rule["maturity"] == "deprecated":
                stats["skipped_deprecated"] += 1
                continue

            # Determine v2 layer
            if rule["maturity"] == "crystallized":
                layer = v2db.LAYER_EPISTEME
            else:
                layer = v2db.LAYER_TECHNE

            # Determine maturity mapping
            v2_maturity = "candidate"
            if rule["maturity"] == "crystallized":
                v2_maturity = "crystallized"

            # Add to v2 (dedup is handled inside add_node)
            node_id = v2db.add_node(
                db=v2,
                agent_id=agent_id,
                content=rule["content"],
                node_type=v2db.NODE_FACT,
                layer=layer,
                category=rule["category"] or "general",
                severity=rule["severity"] or "medium",
                confidence=rule["confidence"] or 0.70,
                applies_when=rule["applies_when"] or "",
                source_session=rule["source_session"] or "",
                origin_agent_id=agent_id,
                reflection_run_id=str(run_id),
                maturity=v2_maturity,
                helpful_count=rule["helpful_count"] or 0,
            )

            if node_id == -1:
                stats["skipped_duplicate"] += 1
            else:
                stats["migrated"] += 1

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"  ERROR on rule {rule['id']}: {e}")

    # Complete the reflection run
    duration_ms = int((time.time() - t0) * 1000)
    v2db.complete_reflection_run(
        v2, run_id,
        status="completed",
        events_processed=v1_count,
        nodes_created=stats["migrated"],
        duration_ms=duration_ms,
        steps_json={
            "type": "v1_migration",
            "v1_rules": v1_count,
            "migrated": stats["migrated"],
            "skipped_duplicate": stats["skipped_duplicate"],
            "skipped_deprecated": stats["skipped_deprecated"],
        }
    )

    stats["duration_ms"] = duration_ms
    v1.close()
    v2.close()
    return stats


def migrate_all(dry_run: bool = False) -> list:
    """Migrate all agents with learnings.db."""
    agents = find_agents_with_learnings()
    if not agents:
        print("No agents with learnings.db found.")
        return []

    print(f"Found {len(agents)} agents with learnings.db: {', '.join(agents)}")
    results = []
    for agent_id in agents:
        print(f"\nMigrating: {agent_id}")
        result = migrate_agent(agent_id, dry_run=dry_run)
        results.append(result)
        print(f"  -> migrated={result['migrated']}, "
              f"dupes={result['skipped_duplicate']}, "
              f"deprecated={result['skipped_deprecated']}, "
              f"errors={result['errors']}, "
              f"time={result['duration_ms']}ms")
    return results


def verify_migration(agent_id: str) -> dict:
    """Verify migration integrity for an agent."""
    v2 = v2db.open_db(agent_id)
    stats = v2db.get_stats(v2, agent_id)

    # Check FTS5 works
    test_results = v2db.fts5_search(v2, "error handling", agent_id, top_k=3)

    # Check reflection run exists
    runs = v2.execute(
        "SELECT * FROM reflection_runs WHERE agent_id = ? AND trigger_type = 'migration'",
        (agent_id,)
    ).fetchall()

    v2.close()

    return {
        "agent_id": agent_id,
        "stats": stats,
        "fts5_test_results": len(test_results),
        "migration_runs": len(runs),
    }


def main():
    parser = argparse.ArgumentParser(description="Hermes v1 -> v2 migration")
    parser.add_argument("--agent", help="Migrate a specific agent")
    parser.add_argument("--all", action="store_true", help="Migrate all agents")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--verify", action="store_true", help="Verify migration integrity")
    args = parser.parse_args()

    if args.verify:
        agent_id = args.agent
        if not agent_id:
            print("--verify requires --agent")
            sys.exit(1)
        result = verify_migration(agent_id)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.all:
        results = migrate_all(dry_run=args.dry_run)
    elif args.agent:
        print(f"Migrating: {args.agent}")
        result = migrate_agent(args.agent, dry_run=args.dry_run)
        results = [result]
        print(f"  -> migrated={result['migrated']}, "
              f"dupes={result['skipped_duplicate']}, "
              f"deprecated={result['skipped_deprecated']}, "
              f"errors={result['errors']}, "
              f"time={result['duration_ms']}ms")
    else:
        parser.print_help()
        sys.exit(1)

    # Summary
    total_migrated = sum(r["migrated"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    print(f"\n{'='*50}")
    print(f"Total migrated: {total_migrated}")
    print(f"Total errors: {total_errors}")


if __name__ == "__main__":
    main()
