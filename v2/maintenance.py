#!/usr/bin/env python3
"""Maintenance cycle for the v2 knowledge store (replaces crystallized.py).
Import: from v2 import db as v2db; from v2 import knowledge_graph as kg"""

import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2 import db as v2db
from v2 import knowledge_graph as kg


def maintenance_cycle(db: sqlite3.Connection, agent_id: str) -> dict:
    """Full maintenance: promote + demote + decay + stats.
    Called at session start and session end.
    Returns: {
        "promoted": int, "promoted_ids": [int],
        "demoted": int, "decayed": int, "deprecated": int,
        "nodes_by_layer": {layer: count},
        "total_nodes": int
    }"""
    print(f"[anamnesis] Running maintenance cycle for agent={agent_id}", file=sys.stderr)

    # Step 1: Decay stale nodes first (lowers confidence before promotion checks)
    decay_result = kg.decay_stale_nodes(db, agent_id, stale_days=30, decay_rate=0.05)

    # Step 2: Promote techne -> episteme (only truly active nodes qualify now)
    promote_result = kg.promote_candidates(db, agent_id)

    # Step 3: Demote stale/low-confidence episteme -> techne
    demote_result = kg.demote_stale_episteme(db, agent_id, stale_days=30)

    # Step 4: Gather stats
    nodes_by_layer = v2db.count_nodes_by_layer(db, agent_id)
    total_nodes = sum(nodes_by_layer.values())

    result = {
        "promoted": promote_result["promoted"],
        "promoted_ids": promote_result["ids"],
        "demoted": demote_result["demoted"],
        "decayed": decay_result["decayed"],
        "deprecated": decay_result["deprecated"],
        "nodes_by_layer": nodes_by_layer,
        "total_nodes": total_nodes,
    }

    print(
        f"[anamnesis] Maintenance done: +{result['promoted']} promoted, "
        f"-{result['demoted']} demoted, {result['decayed']} decayed, "
        f"{result['deprecated']} deprecated, {result['total_nodes']} total nodes",
        file=sys.stderr
    )

    return result
