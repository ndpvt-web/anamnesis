#!/usr/bin/env python3
"""Tests for Hermes v2 feedback engine (three-signal) and knowledge graph operations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2 import feedback
from v2 import knowledge_graph as kg
from v2 import maintenance
from v2.tests.fixtures import (
    create_test_db, TEST_AGENT, TEST_SESSION,
    seed_nodes, seed_edges, seed_surfaced, seed_episteme_node
)

passed = 0
failed = 0


def check(name, condition, detail=""):
    """Hard assertion: a failed check fails the test (no soft PASS/FAIL)."""
    assert condition, f"{name} {detail}"


# --- Feedback tests ---

def test_confirm_surfaced():
    """confirm_surfaced_nodes boosts confidence and helpful_count."""
    print("\n--- test_confirm_surfaced ---")
    db = create_test_db()
    nids = seed_nodes(db, count=3)
    sids = seed_surfaced(db, node_ids=nids)
    old_node = vdb_get(db, nids[0])
    old_conf = old_node["confidence"]

    count = feedback.confirm_surfaced_nodes(db, TEST_AGENT, TEST_SESSION)
    check("confirmed 3 nodes", count == 3, f"got {count}")

    new_node = vdb_get(db, nids[0])
    check("confidence boosted", new_node["confidence"] > old_conf,
          f"old={old_conf}, new={new_node['confidence']}")
    check("helpful_count incremented", new_node["helpful_count"] > old_node["helpful_count"])

    # Check surfaced_nodes feedback updated
    row = db.execute("SELECT feedback FROM surfaced_nodes WHERE id = ?",
                     (sids[0],)).fetchone()
    check("feedback set to confirmed", row["feedback"] == "confirmed")
    db.close()


def test_refute_surfaced():
    """refute_surfaced_nodes decays confidence and increments harmful_count."""
    print("\n--- test_refute_surfaced ---")
    db = create_test_db()
    nids = seed_nodes(db, count=5)
    sids = seed_surfaced(db, node_ids=nids[:3])
    old_node = vdb_get(db, nids[0])
    old_conf = old_node["confidence"]

    count = feedback.refute_surfaced_nodes(db, TEST_AGENT, TEST_SESSION,
                                            error_text="test error")
    check("refuted 3 nodes (limit 3)", count == 3, f"got {count}")

    new_node = vdb_get(db, nids[0])
    check("confidence decayed", new_node["confidence"] < old_conf,
          f"old={old_conf}, new={new_node['confidence']}")
    check("harmful_count incremented", new_node["harmful_count"] > old_node["harmful_count"])

    row = db.execute("SELECT feedback FROM surfaced_nodes WHERE id = ?",
                     (sids[0],)).fetchone()
    check("feedback set to refuted", row["feedback"] == "refuted")
    db.close()


def test_record_irrelevance():
    """record_irrelevance increments irrelevant_count."""
    print("\n--- test_record_irrelevance ---")
    db = create_test_db()
    nids = seed_nodes(db, count=3)
    seed_surfaced(db, node_ids=nids)

    count = feedback.record_irrelevance(db, TEST_AGENT, TEST_SESSION)
    check("marked 3 irrelevant", count == 3, f"got {count}")

    node = vdb_get(db, nids[0])
    check("irrelevant_count incremented", node["irrelevant_count"] == 1)
    db.close()


def test_irrelevance_confidence_decay():
    """Confidence decays when irrelevant_count > 10 and helpful_count == 0."""
    print("\n--- test_irrelevance_confidence_decay ---")
    db = create_test_db()
    nids = seed_nodes(db, count=1)
    nid = nids[0]
    # Set irrelevant_count to 10, helpful_count to 0
    db.execute("UPDATE kg_nodes SET irrelevant_count = 10, helpful_count = 0 WHERE id = ?",
               (nid,))
    db.commit()
    old_conf = vdb_get(db, nid)["confidence"]

    # Surface and mark irrelevant (pushes count to 11)
    seed_surfaced(db, node_ids=[nid])
    feedback.record_irrelevance(db, TEST_AGENT, TEST_SESSION)

    new_conf = vdb_get(db, nid)["confidence"]
    check("confidence decayed at count > 10", new_conf < old_conf,
          f"old={old_conf}, new={new_conf}")
    db.close()


def test_apply_explicit_feedback_confirmed():
    """apply_explicit_feedback with 'confirmed' boosts confidence."""
    print("\n--- test_apply_explicit_feedback_confirmed ---")
    db = create_test_db()
    nids = seed_nodes(db, count=1)
    old_conf = vdb_get(db, nids[0])["confidence"]

    result = feedback.apply_explicit_feedback(db, nids[0], "confirmed")
    check("returns node_id", result["node_id"] == nids[0])
    check("returns feedback", result["feedback"] == "confirmed")
    check("confidence boosted", result["new_confidence"] > old_conf)
    db.close()


def test_apply_explicit_feedback_refuted():
    """apply_explicit_feedback with 'refuted' decays confidence."""
    print("\n--- test_apply_explicit_feedback_refuted ---")
    db = create_test_db()
    nids = seed_nodes(db, count=1)
    old_conf = vdb_get(db, nids[0])["confidence"]

    result = feedback.apply_explicit_feedback(db, nids[0], "refuted")
    check("confidence decayed", result["new_confidence"] < old_conf)

    node = vdb_get(db, nids[0])
    check("harmful_count incremented", node["harmful_count"] == 1)
    db.close()


def test_apply_explicit_feedback_invalid():
    """apply_explicit_feedback with invalid feedback returns 0.0 confidence."""
    print("\n--- test_apply_explicit_feedback_invalid ---")
    db = create_test_db()
    result = feedback.apply_explicit_feedback(db, 99999, "confirmed")
    check("non-existent node returns 0.0", result["new_confidence"] == 0.0)
    db.close()


# --- Knowledge Graph tests ---

def test_find_related_nodes():
    """find_related_nodes does BFS traversal."""
    print("\n--- test_find_related_nodes ---")
    db = create_test_db()
    nids = seed_nodes(db, count=5)
    seed_edges(db, node_ids=nids)  # nids[1,2,3] -> nids[0] SUPPORTS

    related = kg.find_related_nodes(db, TEST_AGENT, nids[0], max_depth=1)
    check("finds related nodes", len(related) > 0, f"got {len(related)}")
    check("has edge_type", all(r.get("edge_type") for r in related))
    check("has depth", all(r.get("depth", -1) >= 1 for r in related))
    # Should find the SUPPORTS sources
    related_ids = {r["id"] for r in related}
    check("finds node 1", nids[1] in related_ids)
    check("finds node 2", nids[2] in related_ids)
    db.close()


def test_find_related_max_depth():
    """find_related_nodes respects max_depth."""
    print("\n--- test_find_related_max_depth ---")
    db = create_test_db()
    nids = seed_nodes(db, count=5)
    # Chain: 0 -> 1 -> 2 -> 3
    v2db.add_edge(db, TEST_AGENT, nids[0], nids[1], v2db.EDGE_SUPPORTS)
    v2db.add_edge(db, TEST_AGENT, nids[1], nids[2], v2db.EDGE_SUPPORTS)
    v2db.add_edge(db, TEST_AGENT, nids[2], nids[3], v2db.EDGE_SUPPORTS)

    depth1 = kg.find_related_nodes(db, TEST_AGENT, nids[0], max_depth=1)
    depth2 = kg.find_related_nodes(db, TEST_AGENT, nids[0], max_depth=2)
    check("depth 1 finds node 1", any(r["id"] == nids[1] for r in depth1))
    check("depth 1 does NOT find node 2",
          not any(r["id"] == nids[2] for r in depth1))
    check("depth 2 finds node 2", any(r["id"] == nids[2] for r in depth2))
    db.close()


def test_detect_contradictions():
    """detect_contradictions finds CONTRADICTS edges in both directions."""
    print("\n--- test_detect_contradictions ---")
    db = create_test_db()
    nids = seed_nodes(db, count=5)
    seed_edges(db, node_ids=nids)  # CONTRADICTS between nids[3] and nids[4]

    contras_3 = kg.detect_contradictions(db, TEST_AGENT, nids[3])
    check("node 3 has contradictions", len(contras_3) > 0)
    check("contradiction points to node 4",
          any(c["node_id"] == nids[4] for c in contras_3))

    contras_4 = kg.detect_contradictions(db, TEST_AGENT, nids[4])
    check("node 4 has contradictions (reverse)", len(contras_4) > 0)

    contras_0 = kg.detect_contradictions(db, TEST_AGENT, nids[0])
    check("node 0 has no contradictions", len(contras_0) == 0)
    db.close()


def test_promote_candidates():
    """promote_candidates promotes qualifying techne nodes to episteme."""
    print("\n--- test_promote_candidates ---")
    db = create_test_db()
    # Create a node that meets all promotion criteria
    cursor = db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, helpful_count, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (TEST_AGENT, "fact", "techne", "Promotion candidate node",
          "principle", 0.90, 5, "candidate"))
    nid = cursor.lastrowid
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (nid, "Promotion candidate node", "", "principle"))
    # Add 2+ SUPPORTS edges (need source nodes)
    for i in range(3):
        src = db.execute("""
            INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                                  confidence, maturity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (TEST_AGENT, "fact", "techne", f"Support source {i}",
              "general", 0.70, "candidate")).lastrowid
        db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
                   (src, f"Support source {i}", "", "general"))
        db.execute("""
            INSERT INTO kg_edges (agent_id, from_node_id, to_node_id, edge_type)
            VALUES (?, ?, ?, ?)
        """, (TEST_AGENT, src, nid, v2db.EDGE_SUPPORTS))
    db.commit()

    result = kg.promote_candidates(db, TEST_AGENT)
    check("promoted 1 node", result["promoted"] == 1, f"got {result['promoted']}")
    check("promoted correct node", nid in result["ids"])

    node = vdb_get(db, nid)
    check("layer changed to episteme", node["layer"] == v2db.LAYER_EPISTEME)
    check("maturity changed to crystallized", node["maturity"] == "crystallized")
    db.close()


def test_promote_blocked_by_contradiction():
    """Nodes with unresolved contradictions are NOT promoted."""
    print("\n--- test_promote_blocked_by_contradiction ---")
    db = create_test_db()
    # Create qualifying node
    nid = db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, helpful_count, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (TEST_AGENT, "fact", "techne", "Contradicted candidate",
          "principle", 0.90, 5, "candidate")).lastrowid
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (nid, "Contradicted candidate", "", "principle"))
    # Add SUPPORTS
    for i in range(3):
        src = db.execute("""
            INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                                  confidence, maturity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (TEST_AGENT, "fact", "techne", f"Contradict support {i}",
              "general", 0.70, "candidate")).lastrowid
        db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
                   (src, f"Contradict support {i}", "", "general"))
        db.execute("""
            INSERT INTO kg_edges (agent_id, from_node_id, to_node_id, edge_type)
            VALUES (?, ?, ?, ?)
        """, (TEST_AGENT, src, nid, v2db.EDGE_SUPPORTS))
    # Add CONTRADICTS edge
    contra_id = db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (TEST_AGENT, "fact", "techne", "Contradicting node",
          "general", 0.70, "candidate")).lastrowid
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (contra_id, "Contradicting node", "", "general"))
    db.execute("""
        INSERT INTO kg_edges (agent_id, from_node_id, to_node_id, edge_type)
        VALUES (?, ?, ?, ?)
    """, (TEST_AGENT, contra_id, nid, v2db.EDGE_CONTRADICTS))
    db.commit()

    result = kg.promote_candidates(db, TEST_AGENT)
    check("NOT promoted due to contradiction", result["promoted"] == 0)
    node = vdb_get(db, nid)
    check("layer still techne", node["layer"] == v2db.LAYER_TECHNE)
    db.close()


def test_decay_stale_nodes():
    """decay_stale_nodes reduces confidence of old nodes."""
    print("\n--- test_decay_stale_nodes ---")
    db = create_test_db()
    nids = seed_nodes(db, count=3)
    # Make nodes stale (last_seen 60 days ago)
    db.execute("UPDATE kg_nodes SET last_seen = datetime('now', '-60 days')")
    db.commit()
    old_conf = vdb_get(db, nids[0])["confidence"]

    result = kg.decay_stale_nodes(db, TEST_AGENT, stale_days=30, decay_rate=0.05)
    check("decayed or deprecated some nodes",
          result["decayed"] + result["deprecated"] > 0,
          f"decayed={result['decayed']}, deprecated={result['deprecated']}")

    new_conf = vdb_get(db, nids[0])["confidence"]
    # Node might be deprecated or decayed depending on initial confidence
    if new_conf > 0:
        check("confidence reduced", new_conf <= old_conf, f"old={old_conf}, new={new_conf}")
    db.close()


def test_demote_stale_episteme():
    """demote_stale_episteme moves stale episteme back to techne."""
    print("\n--- test_demote_stale_episteme ---")
    db = create_test_db()
    nid = seed_episteme_node(db, confidence=0.90)
    # Make it stale
    db.execute("UPDATE kg_nodes SET last_seen = datetime('now', '-60 days') WHERE id = ?",
               (nid,))
    db.commit()

    result = kg.demote_stale_episteme(db, TEST_AGENT, stale_days=30)
    check("demoted 1 node", result["demoted"] == 1, f"got {result['demoted']}")

    node = vdb_get(db, nid)
    check("layer changed to techne", node["layer"] == v2db.LAYER_TECHNE)
    check("maturity changed to established", node["maturity"] == "established")
    db.close()


def test_maintenance_cycle():
    """maintenance_cycle runs all three steps without error."""
    print("\n--- test_maintenance_cycle ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    result = maintenance.maintenance_cycle(db, TEST_AGENT)
    check("returns promoted count", "promoted" in result)
    check("returns demoted count", "demoted" in result)
    check("returns decayed count", "decayed" in result)
    check("returns nodes_by_layer", "nodes_by_layer" in result)
    check("returns total_nodes", result["total_nodes"] >= 10)
    db.close()


# --- Helper ---

def vdb_get(db, node_id):
    """Get node as dict."""
    row = db.execute("SELECT * FROM kg_nodes WHERE id = ?", (node_id,)).fetchone()
    return dict(row) if row else {}


def main():
    test_confirm_surfaced()
    test_refute_surfaced()
    test_record_irrelevance()
    test_irrelevance_confidence_decay()
    test_apply_explicit_feedback_confirmed()
    test_apply_explicit_feedback_refuted()
    test_apply_explicit_feedback_invalid()
    test_find_related_nodes()
    test_find_related_max_depth()
    test_detect_contradictions()
    test_promote_candidates()
    test_promote_blocked_by_contradiction()
    test_decay_stale_nodes()
    test_demote_stale_episteme()
    test_maintenance_cycle()

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
