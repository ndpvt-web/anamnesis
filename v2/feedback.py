#!/usr/bin/env python3
"""Three-signal feedback engine.
Signal 1: Confirmation  -> confidence +0.02, helpful_count +1
Signal 2: Refutation    -> confidence -0.03, harmful_count +1
Signal 3: Irrelevance   -> irrelevant_count +1 (no confidence change unless count > 10 and helpful = 0)
Import: from v2 import db as v2db"""

import math
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2 import db as v2db


# --- Token-based cosine similarity (no embeddings, <1ms) ---

_TOKEN_RE = re.compile(r'[a-z0-9_]+(?:\.[a-z0-9_]+)*|--?[a-z]+')
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "not", "no", "or", "and", "but", "if", "then",
    "than", "when", "up", "out", "so", "all", "each", "any", "use",
})


def _tokenize(text: str) -> Counter:
    """Extract meaningful tokens from text, filtered for stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    return Counter(t for t in tokens if t not in _STOP_WORDS and len(t) > 1)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Token-based cosine similarity between two texts.
    Returns 0.0 to 1.0. Fast (<1ms for typical inputs)."""
    vec_a = _tokenize(text_a)
    vec_b = _tokenize(text_b)
    if not vec_a or not vec_b:
        return 0.0
    # Dot product
    dot = sum(vec_a[k] * vec_b[k] for k in vec_a if k in vec_b)
    if dot == 0:
        return 0.0
    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    return dot / (mag_a * mag_b)


def confirm_surfaced_nodes(db, agent_id: str, session_id: str,
                           action_context: str = "") -> int:
    """Credit the most relevant surfaced node(s) on tool success.
    Uses cosine similarity between node content and action_context to rank
    candidates. Only the top-scoring node (or top 2 if scores are close)
    gets full credit. Others get partial credit or are marked irrelevant.

    action_context: concatenation of tool_name + tool_input + tool_output
                    that triggered the success signal. If empty, falls back
                    to crediting all pending nodes equally (backward compat).

    Also creates SUPPORTS edges between confirmed nodes in same session.
    Returns count of confirmed nodes."""
    rows = db.execute("""
        SELECT sn.id as surfaced_id, sn.node_id
        FROM surfaced_nodes sn
        WHERE sn.agent_id = ? AND sn.session_id = ? AND sn.feedback = 'pending'
    """, (agent_id, session_id)).fetchall()

    if not rows:
        return 0

    # Get already-confirmed node IDs for this session (for SUPPORTS edges)
    prev_confirmed = db.execute("""
        SELECT DISTINCT sn.node_id
        FROM surfaced_nodes sn
        WHERE sn.agent_id = ? AND sn.session_id = ? AND sn.feedback = 'confirmed'
    """, (agent_id, session_id)).fetchall()
    prev_confirmed_ids = {r["node_id"] for r in prev_confirmed}

    # --- Cosine-similarity ranking ---
    # Score each pending node against action_context
    scored = []
    for row in rows:
        node_id = row["node_id"]
        node = v2db.get_node(db, node_id)
        content = node["content"] if node else ""
        if action_context:
            sim = cosine_similarity(content, action_context)
        else:
            sim = 1.0  # No context = credit all equally (backward compat)
        scored.append((row, sim, content))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Determine which nodes get full credit vs partial vs irrelevant
    # - Top node always gets full credit (if sim > 0)
    # - Second node gets full credit if within 80% of top score
    # - Others: if sim >= 0.1, partial credit (half boost); else mark irrelevant
    top_sim = scored[0][1] if scored else 0.0
    FULL_THRESHOLD = 0.8  # Within 80% of top score
    PARTIAL_THRESHOLD = 0.1  # Minimum similarity for any credit

    count = 0
    new_confirmed_ids = []
    for i, (row, sim, _content) in enumerate(scored):
        surfaced_id = row["surfaced_id"]
        node_id = row["node_id"]

        if top_sim == 0 or (not action_context):
            # No context or all zeros: credit all equally (backward compat)
            v2db.update_node_confidence(db, node_id, v2db.CONFIDENCE_BOOST)
            db.execute("""
                UPDATE kg_nodes
                SET helpful_count = helpful_count + 1,
                    last_confirmed = datetime('now')
                WHERE id = ?
            """, (node_id,))
            v2db.update_feedback(db, surfaced_id, "confirmed")
            new_confirmed_ids.append(node_id)
            count += 1
        elif i == 0 or (top_sim > 0 and sim >= top_sim * FULL_THRESHOLD):
            # Full credit: top node or close runner-up
            v2db.update_node_confidence(db, node_id, v2db.CONFIDENCE_BOOST)
            db.execute("""
                UPDATE kg_nodes
                SET helpful_count = helpful_count + 1,
                    last_confirmed = datetime('now')
                WHERE id = ?
            """, (node_id,))
            v2db.update_feedback(db, surfaced_id, "confirmed")
            new_confirmed_ids.append(node_id)
            count += 1
        elif sim >= PARTIAL_THRESHOLD:
            # Partial credit: some relevance but not the best match
            v2db.update_node_confidence(db, node_id, v2db.CONFIDENCE_BOOST * 0.5)
            db.execute("""
                UPDATE kg_nodes
                SET helpful_count = helpful_count + 1,
                    last_confirmed = datetime('now')
                WHERE id = ?
            """, (node_id,))
            v2db.update_feedback(db, surfaced_id, "confirmed")
            new_confirmed_ids.append(node_id)
            count += 1
        else:
            # Irrelevant: surfaced but not related to the successful action
            db.execute("""
                UPDATE kg_nodes
                SET irrelevant_count = irrelevant_count + 1
                WHERE id = ?
            """, (node_id,))
            v2db.update_feedback(db, surfaced_id, "irrelevant")

    # Create SUPPORTS edges between confirmed nodes
    edges_created = 0
    if new_confirmed_ids and prev_confirmed_ids:
        anchor_id = min(prev_confirmed_ids)
        for new_id in new_confirmed_ids:
            if new_id != anchor_id:
                eid = v2db.add_edge(db, agent_id, anchor_id, new_id,
                                    v2db.EDGE_SUPPORTS, weight=0.5)
                if eid > 0:
                    edges_created += 1

    if count:
        db.commit()
        msg = f"[anamnesis] confirm_surfaced_nodes: confirmed {count}/{len(rows)} nodes"
        if action_context:
            msg += f" (top_sim={top_sim:.3f})"
        if edges_created:
            msg += f", created {edges_created} SUPPORTS edges"
        msg += f" for session {session_id}"
        print(msg, file=sys.stderr)

    return count


def refute_surfaced_nodes(db, agent_id: str, session_id: str,
                          error_text: str = "") -> int:
    """Refute recently surfaced nodes on tool failure.
    Finds surfaced_nodes with feedback='pending' for this session (limit 3).
    For each: confidence -= 0.03, harmful_count += 1.
    Updates surfaced_nodes.feedback = 'refuted'.
    Returns count of refuted nodes."""
    rows = db.execute("""
        SELECT sn.id as surfaced_id, sn.node_id
        FROM surfaced_nodes sn
        WHERE sn.agent_id = ? AND sn.session_id = ? AND sn.feedback = 'pending'
        LIMIT 3
    """, (agent_id, session_id)).fetchall()

    count = 0
    for row in rows:
        surfaced_id = row["surfaced_id"]
        node_id = row["node_id"]

        # Update confidence - decay + harmful_count
        v2db.update_node_confidence(db, node_id, -v2db.CONFIDENCE_DECAY)
        db.execute("""
            UPDATE kg_nodes
            SET harmful_count = harmful_count + 1
            WHERE id = ?
        """, (node_id,))

        # Update surfaced_nodes feedback
        v2db.update_feedback(db, surfaced_id, "refuted")
        count += 1

    if count:
        db.commit()
        print(f"[anamnesis] refute_surfaced_nodes: refuted {count} nodes for session {session_id}", file=sys.stderr)

    return count


def record_irrelevance(db, agent_id: str, session_id: str) -> int:
    """Mark remaining pending surfaced nodes as irrelevant at session end.
    For each: irrelevant_count += 1.
    If irrelevant_count > 10 and helpful_count == 0: confidence -= 0.01.
    Updates surfaced_nodes.feedback = 'irrelevant'.
    Returns count."""
    rows = db.execute("""
        SELECT sn.id as surfaced_id, sn.node_id
        FROM surfaced_nodes sn
        WHERE sn.agent_id = ? AND sn.session_id = ? AND sn.feedback = 'pending'
    """, (agent_id, session_id)).fetchall()

    count = 0
    for row in rows:
        surfaced_id = row["surfaced_id"]
        node_id = row["node_id"]

        # Increment irrelevant_count
        db.execute("""
            UPDATE kg_nodes
            SET irrelevant_count = irrelevant_count + 1
            WHERE id = ?
        """, (node_id,))

        # Check thresholds: if irrelevant_count > 10 and helpful_count == 0, decay confidence
        node = db.execute("""
            SELECT irrelevant_count, helpful_count FROM kg_nodes WHERE id = ?
        """, (node_id,)).fetchone()

        if node and node["irrelevant_count"] > 10 and node["helpful_count"] == 0:
            v2db.update_node_confidence(db, node_id, -0.01)

        # Update surfaced_nodes feedback
        v2db.update_feedback(db, surfaced_id, "irrelevant")
        count += 1

    if count:
        db.commit()
        print(f"[anamnesis] record_irrelevance: marked {count} nodes irrelevant for session {session_id}", file=sys.stderr)

    return count


def apply_explicit_feedback(db, node_id: int, feedback: str) -> dict:
    """Apply explicit feedback from MCP tool.
    feedback must be 'confirmed', 'refuted', or 'irrelevant'.
    Returns: {"node_id": int, "new_confidence": float, "feedback": str}"""
    valid_feedback = {"confirmed", "refuted", "irrelevant"}
    if feedback not in valid_feedback:
        print(f"[anamnesis] apply_explicit_feedback: invalid feedback '{feedback}'", file=sys.stderr)
        return {"node_id": node_id, "new_confidence": 0.0, "feedback": feedback}

    node = v2db.get_node(db, node_id)
    if not node:
        print(f"[anamnesis] apply_explicit_feedback: node {node_id} not found", file=sys.stderr)
        return {"node_id": node_id, "new_confidence": 0.0, "feedback": feedback}

    if feedback == "confirmed":
        new_conf = v2db.update_node_confidence(db, node_id, v2db.CONFIDENCE_BOOST)
        db.execute("""
            UPDATE kg_nodes
            SET helpful_count = helpful_count + 1,
                last_confirmed = datetime('now')
            WHERE id = ?
        """, (node_id,))
        db.commit()

    elif feedback == "refuted":
        new_conf = v2db.update_node_confidence(db, node_id, -v2db.CONFIDENCE_DECAY)
        db.execute("""
            UPDATE kg_nodes
            SET harmful_count = harmful_count + 1
            WHERE id = ?
        """, (node_id,))
        db.commit()

    elif feedback == "irrelevant":
        db.execute("""
            UPDATE kg_nodes
            SET irrelevant_count = irrelevant_count + 1
            WHERE id = ?
        """, (node_id,))

        # Check threshold for confidence decay
        updated_node = db.execute("""
            SELECT irrelevant_count, helpful_count, confidence FROM kg_nodes WHERE id = ?
        """, (node_id,)).fetchone()

        if updated_node and updated_node["irrelevant_count"] > 10 and updated_node["helpful_count"] == 0:
            new_conf = v2db.update_node_confidence(db, node_id, -0.01)
        else:
            new_conf = updated_node["confidence"] if updated_node else node.get("confidence", 0.0)
        db.commit()

    else:
        new_conf = node.get("confidence", 0.0)

    print(f"[anamnesis] apply_explicit_feedback: node {node_id} -> {feedback}, conf={new_conf:.3f}", file=sys.stderr)
    return {"node_id": node_id, "new_confidence": new_conf, "feedback": feedback}
