#!/usr/bin/env python3
"""Knowledge graph operations: traversal, promotion, decay, contradiction detection.
All functions take an open db connection and agent_id.
Import: from v2 import db as v2db"""

import math
import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2 import db as v2db


def find_related_nodes(db: sqlite3.Connection, agent_id: str,
                       node_id: int, max_depth: int = 2,
                       max_nodes: int = 20) -> list:
    """BFS graph traversal from a node. Returns list of dicts with
    keys: id, content, layer, confidence, edge_type, depth.
    Follows all edge types. Excludes deprecated nodes.
    Max depth=2, max_nodes=20 for performance."""
    visited = {node_id}
    results = []
    # Queue items: (node_id, depth, edge_type_that_led_here)
    queue = [(node_id, 0, None)]

    while queue and len(results) < max_nodes:
        current_id, depth, edge_type = queue.pop(0)

        if depth >= max_depth:
            continue

        # Get outgoing edges
        out_edges = v2db.get_edges_from(db, current_id)
        # Get incoming edges
        in_edges = v2db.get_edges_to(db, current_id)

        for edge in out_edges:
            neighbor_id = edge["to_node_id"]
            if neighbor_id in visited:
                continue
            # Fetch the node to check it's not deprecated
            row = db.execute(
                "SELECT id, content, layer, confidence, maturity FROM kg_nodes "
                "WHERE id = ? AND agent_id = ? AND maturity != 'deprecated'",
                (neighbor_id, agent_id)
            ).fetchone()
            if row is None:
                continue
            visited.add(neighbor_id)
            results.append({
                "id": row["id"],
                "content": row["content"],
                "layer": row["layer"],
                "confidence": row["confidence"],
                "edge_type": edge["edge_type"],
                "depth": depth + 1,
            })
            if len(results) >= max_nodes:
                break
            queue.append((neighbor_id, depth + 1, edge["edge_type"]))

        if len(results) >= max_nodes:
            break

        for edge in in_edges:
            neighbor_id = edge["from_node_id"]
            if neighbor_id in visited:
                continue
            row = db.execute(
                "SELECT id, content, layer, confidence, maturity FROM kg_nodes "
                "WHERE id = ? AND agent_id = ? AND maturity != 'deprecated'",
                (neighbor_id, agent_id)
            ).fetchone()
            if row is None:
                continue
            visited.add(neighbor_id)
            results.append({
                "id": row["id"],
                "content": row["content"],
                "layer": row["layer"],
                "confidence": row["confidence"],
                "edge_type": edge["edge_type"],
                "depth": depth + 1,
            })
            if len(results) >= max_nodes:
                break
            queue.append((neighbor_id, depth + 1, edge["edge_type"]))

    return results[:max_nodes]


def build_context_subgraph(db: sqlite3.Connection, agent_id: str,
                           seed_node_ids: list,
                           max_nodes: int = 30) -> dict:
    """Build a context subgraph from seed nodes.
    Returns: {"nodes": [node_dicts], "edges": [edge_dicts], "summary": str}
    Expands each seed node 1 hop. Deduplicates. Generates summary string."""
    all_node_ids = set()
    node_map = {}
    edge_list = []
    edge_seen = set()

    # Collect seed nodes themselves
    for nid in seed_node_ids:
        row = db.execute(
            "SELECT id, content, layer, confidence, category, severity, maturity "
            "FROM kg_nodes WHERE id = ? AND agent_id = ? AND maturity != 'deprecated'",
            (nid, agent_id)
        ).fetchone()
        if row:
            all_node_ids.add(nid)
            node_map[nid] = dict(row)

    # 1-hop expansion for each seed
    for nid in list(seed_node_ids):
        if len(all_node_ids) >= max_nodes:
            break
        neighbors = find_related_nodes(db, agent_id, nid, max_depth=1, max_nodes=max_nodes)
        for neighbor in neighbors:
            nnode_id = neighbor["id"]
            if nnode_id not in all_node_ids and len(all_node_ids) < max_nodes:
                all_node_ids.add(nnode_id)
                row = db.execute(
                    "SELECT id, content, layer, confidence, category, severity, maturity "
                    "FROM kg_nodes WHERE id = ? AND agent_id = ? AND maturity != 'deprecated'",
                    (nnode_id, agent_id)
                ).fetchone()
                if row:
                    node_map[nnode_id] = dict(row)

    # Collect edges between collected nodes
    for nid in all_node_ids:
        out_edges = v2db.get_edges_from(db, nid)
        for edge in out_edges:
            if edge["to_node_id"] in all_node_ids:
                edge_key = (edge["from_node_id"], edge["to_node_id"], edge["edge_type"])
                if edge_key not in edge_seen:
                    edge_seen.add(edge_key)
                    edge_list.append({
                        "id": edge["id"],
                        "from_node_id": edge["from_node_id"],
                        "to_node_id": edge["to_node_id"],
                        "edge_type": edge["edge_type"],
                        "weight": edge.get("weight", 1.0),
                    })

    nodes_list = list(node_map.values())

    # Generate summary
    if not nodes_list:
        summary = "Empty subgraph."
    else:
        layer_counts = {}
        for n in nodes_list:
            layer_counts[n["layer"]] = layer_counts.get(n["layer"], 0) + 1
        layer_str = ", ".join(f"{cnt} {layer}" for layer, cnt in layer_counts.items())
        summary = (
            f"Subgraph: {len(nodes_list)} nodes ({layer_str}), "
            f"{len(edge_list)} edges from {len(seed_node_ids)} seed(s)."
        )

    return {"nodes": nodes_list, "edges": edge_list, "summary": summary}


def detect_contradictions(db: sqlite3.Connection, agent_id: str,
                          node_id: int) -> list:
    """Find nodes that CONTRADICT the given node.
    Returns list of dicts with keys: node_id, content, confidence, edge_id.
    Checks both directions of CONTRADICTS edges."""
    results = []

    # Outgoing CONTRADICTS edges (node_id -> other)
    out_edges = v2db.get_edges_from(db, node_id, edge_type=v2db.EDGE_CONTRADICTS)
    for edge in out_edges:
        other_id = edge["to_node_id"]
        row = db.execute(
            "SELECT id, content, confidence, maturity FROM kg_nodes "
            "WHERE id = ? AND agent_id = ? AND maturity != 'deprecated'",
            (other_id, agent_id)
        ).fetchone()
        if row:
            results.append({
                "node_id": row["id"],
                "content": row["content"],
                "confidence": row["confidence"],
                "edge_id": edge["id"],
            })

    # Incoming CONTRADICTS edges (other -> node_id)
    in_edges = v2db.get_edges_to(db, node_id, edge_type=v2db.EDGE_CONTRADICTS)
    for edge in in_edges:
        other_id = edge["from_node_id"]
        row = db.execute(
            "SELECT id, content, confidence, maturity FROM kg_nodes "
            "WHERE id = ? AND agent_id = ? AND maturity != 'deprecated'",
            (other_id, agent_id)
        ).fetchone()
        if row:
            results.append({
                "node_id": row["id"],
                "content": row["content"],
                "confidence": row["confidence"],
                "edge_id": edge["id"],
            })

    return results


def promote_candidates(db: sqlite3.Connection, agent_id: str) -> dict:
    """Promote techne nodes to episteme when they meet ALL criteria:
    - confidence >= 0.85
    - helpful_count >= 3
    - count_supports() >= 2
    - has_unresolved_contradictions() == False
    Updates node layer to 'episteme' and maturity to 'crystallized'.
    Returns: {"promoted": int, "ids": [int]}"""
    candidates = db.execute("""
        SELECT id, confidence, helpful_count FROM kg_nodes
        WHERE agent_id = ? AND layer = ? AND maturity != 'deprecated'
        AND confidence >= ? AND helpful_count >= ?
    """, (agent_id, v2db.LAYER_TECHNE,
          v2db.PROMOTE_MIN_CONFIDENCE, v2db.PROMOTE_MIN_HELPFUL)).fetchall()

    promoted_ids = []
    for row in candidates:
        nid = row["id"]
        # Transitional relief: strong empirical evidence reduces edge requirement
        required_supports = v2db.PROMOTE_MIN_SUPPORTS
        if row["helpful_count"] >= 5:
            required_supports = 1
        if v2db.count_supports(db, nid) < required_supports:
            continue
        if v2db.has_unresolved_contradictions(db, nid):
            continue
        db.execute("""
            UPDATE kg_nodes SET layer = ?, maturity = ?, last_seen = datetime('now')
            WHERE id = ?
        """, (v2db.LAYER_EPISTEME, "crystallized", nid))
        promoted_ids.append(nid)
        print(f"[hermes-v2] Promoted node {nid} to episteme/crystallized", file=sys.stderr)

    if promoted_ids:
        db.commit()

    return {"promoted": len(promoted_ids), "ids": promoted_ids}


def create_supports_from_similarity(db: sqlite3.Connection, agent_id: str,
                                     new_node_ids: list,
                                     threshold: float = 0.4) -> int:
    """For each new node, find existing nodes with high content similarity.
    Creates SUPPORTS edges where token-based cosine similarity exceeds threshold.
    Returns total edge count created."""
    from v2.feedback import cosine_similarity
    edges_created = 0

    new_nodes = [v2db.get_node(db, nid) for nid in new_node_ids if nid > 0]
    new_nodes = [n for n in new_nodes if n]

    existing = v2db.get_nodes_by_layer(db, agent_id, v2db.LAYER_TECHNE, limit=30)
    existing += v2db.get_nodes_by_layer(db, agent_id, v2db.LAYER_EPISTEME, limit=10)

    new_ids_set = set(new_node_ids)

    for new_node in new_nodes:
        for ex in existing:
            if ex["id"] in new_ids_set:
                continue
            sim = cosine_similarity(new_node.get("content", ""), ex.get("content", ""))
            if sim >= threshold:
                eid = v2db.add_edge(db, agent_id, new_node["id"], ex["id"],
                                    v2db.EDGE_SUPPORTS, weight=round(sim, 3),
                                    metadata={"source": "similarity", "score": round(sim, 3)})
                if eid > 0:
                    edges_created += 1
    return edges_created


def ebbinghaus_strength(base_confidence: float, days_elapsed: float,
                        helpful_count: int, base_s: float = 30.0) -> float:
    """Ebbinghaus forgetting curve with SM-2 reinforcement.
    Formula: strength = base_confidence * e^(-t/S)
    where S = base_s * (1 + helpful_count * 0.3)

    - base_confidence: the node's stored confidence at last access
    - days_elapsed: days since last_confirmed (or last_seen if never confirmed)
    - helpful_count: number of times the node was helpfully used
    - base_s: base significance in days (30 = halves in ~21 days with no usage)

    SM-2 insight: each helpful access increases S, making the node decay slower.
    A node accessed 10 times has S = 30 * (1 + 10*0.3) = 120 days -- very durable.
    A node never accessed has S = 30 days -- decays quickly.
    """
    significance = base_s * (1.0 + helpful_count * 0.3)
    if days_elapsed <= 0:
        return base_confidence
    return base_confidence * math.exp(-days_elapsed / significance)


def decay_stale_nodes(db: sqlite3.Connection, agent_id: str,
                      stale_days: int = 30,
                      decay_rate: float = 0.05) -> dict:
    """Apply Ebbinghaus exponential decay to nodes not seen in stale_days.
    Decay formula: strength = confidence * e^(-days_elapsed / S)
    where S = 30 * (1 + helpful_count * 0.3)  [SM-2 reinforcement]

    Nodes used frequently (high helpful_count) decay slowly.
    Nodes never used decay exponentially faster.
    Floor: 0.15 (never drop below unless deprecating).
    Deprecates nodes with computed strength < 0.15.
    Returns: {"decayed": int, "deprecated": int}"""
    stale_nodes = db.execute("""
        SELECT id, confidence, helpful_count, last_seen, last_confirmed
        FROM kg_nodes
        WHERE agent_id = ? AND maturity != 'deprecated'
        AND layer IN (?, ?)
        AND last_seen < datetime('now', ? || ' days')
    """, (agent_id, v2db.LAYER_TECHNE, v2db.LAYER_EPISTEME,
          f"-{stale_days}")).fetchall()

    decayed_count = 0
    deprecated_count = 0

    for row in stale_nodes:
        nid = row["id"]
        old_conf = row["confidence"]
        helpful = row["helpful_count"]
        # Use last_confirmed if available, else last_seen
        anchor = row["last_confirmed"] or row["last_seen"]

        # Compute days elapsed since last meaningful access
        days_row = db.execute(
            "SELECT julianday('now') - julianday(?) as days",
            (anchor,)
        ).fetchone()
        days_elapsed = days_row["days"] if days_row and days_row["days"] else float(stale_days)

        new_conf = ebbinghaus_strength(old_conf, days_elapsed, helpful)

        # Deprecate if confidence drops below threshold
        if new_conf < 0.15:
            db.execute(
                "UPDATE kg_nodes SET maturity = 'deprecated', confidence = ?, "
                "last_seen = datetime('now') WHERE id = ?",
                (new_conf, nid)
            )
            deprecated_count += 1
            print(f"[hermes-v2] Deprecated node {nid} (ebbinghaus: {old_conf:.3f}->{new_conf:.3f}, "
                  f"helpful={helpful}, days={days_elapsed:.0f})", file=sys.stderr)
        else:
            new_conf = max(0.15, new_conf)
            db.execute(
                "UPDATE kg_nodes SET confidence = ?, last_seen = datetime('now') WHERE id = ?",
                (new_conf, nid)
            )
            decayed_count += 1

    if stale_nodes:
        db.commit()

    return {"decayed": decayed_count, "deprecated": deprecated_count}


def demote_stale_episteme(db: sqlite3.Connection, agent_id: str,
                          stale_days: int = 30) -> dict:
    """Demote episteme/crystallized nodes not seen in stale_days back to techne.
    Also demotes nodes whose confidence dropped below 0.50.
    Returns: {"demoted": int}"""
    # Stale episteme nodes
    stale = db.execute("""
        SELECT id FROM kg_nodes
        WHERE agent_id = ? AND layer = ? AND maturity != 'deprecated'
        AND last_seen < datetime('now', ? || ' days')
    """, (agent_id, v2db.LAYER_EPISTEME, f"-{stale_days}")).fetchall()

    # Low-confidence episteme nodes
    low_conf = db.execute("""
        SELECT id FROM kg_nodes
        WHERE agent_id = ? AND layer = ? AND maturity != 'deprecated'
        AND confidence < 0.50
    """, (agent_id, v2db.LAYER_EPISTEME)).fetchall()

    demote_ids = set()
    for row in stale:
        demote_ids.add(row["id"])
    for row in low_conf:
        demote_ids.add(row["id"])

    for nid in demote_ids:
        db.execute("""
            UPDATE kg_nodes SET layer = ?, maturity = 'established',
            last_seen = datetime('now') WHERE id = ?
        """, (v2db.LAYER_TECHNE, nid))
        print(f"[hermes-v2] Demoted node {nid} from episteme to techne", file=sys.stderr)

    if demote_ids:
        db.commit()

    return {"demoted": len(demote_ids)}
