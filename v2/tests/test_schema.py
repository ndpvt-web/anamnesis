#!/usr/bin/env python3
"""Tests for Hermes v2 database schema, CRUD operations, and FTS5."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2.tests.fixtures import (
    create_test_db, TEST_AGENT, TEST_SESSION,
    seed_nodes, seed_edges, seed_events, seed_episteme_node
)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def test_schema_creation():
    """All 8 tables exist after schema init."""
    print("\n--- test_schema_creation ---")
    db = create_test_db()
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    expected = {"aisthesis", "empeiria", "kg_edges", "kg_nodes",
                "reflection_runs", "sophia", "surfaced_nodes"}
    for t in expected:
        check(f"table {t} exists", t in tables, f"tables={tables}")
    # FTS virtual table
    vtables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_fts'"
    ).fetchall()]
    check("kg_fts virtual table exists", len(vtables) == 1)
    db.close()


def test_append_event():
    """append_event writes to empeiria and returns ID."""
    print("\n--- test_append_event ---")
    db = create_test_db()
    eid = v2db.append_event(db, TEST_AGENT, "test_event",
                            session_id=TEST_SESSION,
                            tool_name="Bash",
                            command="echo hello")
    check("event ID positive", eid > 0)
    row = db.execute("SELECT * FROM empeiria WHERE id = ?", (eid,)).fetchone()
    check("event_type correct", row["event_type"] == "test_event")
    check("tool_name correct", row["tool_name"] == "Bash")
    check("command correct", row["command"] == "echo hello")
    check("agent_id correct", row["agent_id"] == TEST_AGENT)
    db.close()


def test_get_session_events():
    """get_session_events returns events in DESC order."""
    print("\n--- test_get_session_events ---")
    db = create_test_db()
    ids = seed_events(db, count=5)
    events = v2db.get_session_events(db, TEST_AGENT, TEST_SESSION, limit=10)
    check("returns 5 events", len(events) == 5)
    check("DESC order (newest first)", events[0]["id"] > events[-1]["id"])
    db.close()


def test_add_node():
    """add_node creates a kg_node with correct fields."""
    print("\n--- test_add_node ---")
    db = create_test_db()
    nid = v2db.add_node(db, TEST_AGENT, "Test learning content",
                        node_type=v2db.NODE_FACT,
                        layer=v2db.LAYER_TECHNE,
                        category="anti_pattern",
                        severity="high",
                        confidence=0.75)
    check("node ID positive", nid > 0)
    node = v2db.get_node(db, nid)
    check("content matches", node["content"] == "Test learning content")
    check("layer is techne", node["layer"] == v2db.LAYER_TECHNE)
    check("category is anti_pattern", node["category"] == "anti_pattern")
    check("confidence is 0.75", abs(node["confidence"] - 0.75) < 0.001)
    check("maturity is candidate", node["maturity"] == "candidate")
    db.close()


def test_add_node_dedup():
    """add_node returns -1 for duplicate content."""
    print("\n--- test_add_node_dedup ---")
    db = create_test_db()
    nid1 = v2db.add_node(db, TEST_AGENT, "Exact duplicate test content here")
    check("first insert succeeds", nid1 > 0)
    nid2 = v2db.add_node(db, TEST_AGENT, "Exact duplicate test content here")
    check("duplicate returns -1", nid2 == -1)
    nid3 = v2db.add_node(db, TEST_AGENT, "EXACT DUPLICATE TEST CONTENT HERE")
    check("case-insensitive dedup", nid3 == -1)
    db.close()


def test_add_edge():
    """add_edge creates typed relationships."""
    print("\n--- test_add_edge ---")
    db = create_test_db()
    nids = seed_nodes(db, count=3)
    eid = v2db.add_edge(db, TEST_AGENT, nids[0], nids[1],
                        v2db.EDGE_SUPPORTS, weight=1.0)
    check("edge ID positive", eid > 0)
    # Duplicate edge returns -1
    eid2 = v2db.add_edge(db, TEST_AGENT, nids[0], nids[1],
                         v2db.EDGE_SUPPORTS, weight=1.0)
    check("duplicate edge returns -1", eid2 == -1)
    # Different edge type on same nodes is allowed
    eid3 = v2db.add_edge(db, TEST_AGENT, nids[0], nids[1],
                         v2db.EDGE_CONTRADICTS, weight=1.0)
    check("different edge type allowed", eid3 > 0)
    db.close()


def test_get_edges():
    """get_edges_from and get_edges_to return correct edges."""
    print("\n--- test_get_edges ---")
    db = create_test_db()
    nids = seed_nodes(db, count=3)
    v2db.add_edge(db, TEST_AGENT, nids[0], nids[1], v2db.EDGE_SUPPORTS)
    v2db.add_edge(db, TEST_AGENT, nids[0], nids[2], v2db.EDGE_GENERALIZES)
    out = v2db.get_edges_from(db, nids[0])
    check("2 outgoing edges", len(out) == 2)
    inc = v2db.get_edges_to(db, nids[1])
    check("1 incoming edge to node 1", len(inc) == 1)
    # Filter by type
    supports = v2db.get_edges_from(db, nids[0], edge_type=v2db.EDGE_SUPPORTS)
    check("1 SUPPORTS edge filtered", len(supports) == 1)
    db.close()


def test_fts5_search():
    """FTS5 search returns relevant results."""
    print("\n--- test_fts5_search ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    results = v2db.fts5_search(db, "anti_pattern topic", TEST_AGENT, top_k=5)
    check("returns results", len(results) > 0)
    check("results have content", all(r.get("content") for r in results))
    check("results have confidence", all(r.get("confidence", 0) > 0 for r in results))
    # Category filter
    ap_results = v2db.fts5_search(db, "pattern topic", TEST_AGENT,
                                   top_k=10, category_filter="anti_pattern")
    check("category filter works",
          all(r["category"] == "anti_pattern" for r in ap_results))
    db.close()


def test_fts5_empty_query():
    """FTS5 search with empty/stopword-only query returns empty."""
    print("\n--- test_fts5_empty_query ---")
    db = create_test_db()
    seed_nodes(db, count=5)
    check("empty string returns []", v2db.fts5_search(db, "", TEST_AGENT) == [])
    check("stopwords only returns []",
          v2db.fts5_search(db, "the a an is are", TEST_AGENT) == [])
    db.close()


def test_count_supports():
    """count_supports returns correct SUPPORTS edge count."""
    print("\n--- test_count_supports ---")
    db = create_test_db()
    nids = seed_nodes(db, count=5)
    seed_edges(db, node_ids=nids)  # Creates 3 SUPPORTS to nids[0]
    count = v2db.count_supports(db, nids[0])
    check("3 SUPPORTS edges", count == 3, f"got {count}")
    count_other = v2db.count_supports(db, nids[1])
    check("0 SUPPORTS for other node", count_other == 0)
    db.close()


def test_has_unresolved_contradictions():
    """has_unresolved_contradictions detects active CONTRADICTS edges."""
    print("\n--- test_has_unresolved_contradictions ---")
    db = create_test_db()
    nids = seed_nodes(db, count=5)
    seed_edges(db, node_ids=nids)  # Creates CONTRADICTS between nids[3] and nids[4]
    check("node 3 has contradiction", v2db.has_unresolved_contradictions(db, nids[3]))
    check("node 4 has contradiction", v2db.has_unresolved_contradictions(db, nids[4]))
    check("node 0 has no contradiction",
          not v2db.has_unresolved_contradictions(db, nids[0]))
    db.close()


def test_update_node_confidence():
    """update_node_confidence clamps correctly."""
    print("\n--- test_update_node_confidence ---")
    db = create_test_db()
    nid = v2db.add_node(db, TEST_AGENT, "Confidence test node", confidence=0.50)
    new = v2db.update_node_confidence(db, nid, 0.02)
    check("boost works", abs(new - 0.52) < 0.001, f"got {new}")
    new = v2db.update_node_confidence(db, nid, -0.50)
    check("clamps to floor 0.10", abs(new - 0.10) < 0.001, f"got {new}")
    new = v2db.update_node_confidence(db, nid, 1.0)
    check("clamps to MAX_CONFIDENCE", abs(new - v2db.MAX_CONFIDENCE) < 0.001, f"got {new}")
    db.close()


def test_record_aisthesis():
    """record_aisthesis writes to working memory."""
    print("\n--- test_record_aisthesis ---")
    db = create_test_db()
    rid = v2db.record_aisthesis(db, TEST_AGENT, TEST_SESSION,
                                "command", tool_name="Bash",
                                command="ls -la")
    check("record ID positive", rid > 0)
    mem = v2db.get_session_memory(db, TEST_AGENT, TEST_SESSION)
    check("1 memory entry", len(mem) == 1)
    check("tool_name correct", mem[0]["tool_name"] == "Bash")
    db.close()


def test_surfaced_nodes():
    """record_surfaced and update_feedback work correctly."""
    print("\n--- test_surfaced_nodes ---")
    db = create_test_db()
    nids = seed_nodes(db, count=3)
    v2db.record_surfaced(db, TEST_AGENT, TEST_SESSION, nids)
    rows = db.execute(
        "SELECT * FROM surfaced_nodes WHERE session_id = ?",
        (TEST_SESSION,)
    ).fetchall()
    check("3 surfaced records", len(rows) == 3)
    check("all pending", all(r["feedback"] == "pending" for r in rows))
    # Update feedback
    v2db.update_feedback(db, rows[0]["id"], "confirmed")
    updated = db.execute(
        "SELECT feedback FROM surfaced_nodes WHERE id = ?",
        (rows[0]["id"],)
    ).fetchone()
    check("feedback updated to confirmed", updated["feedback"] == "confirmed")
    db.close()


def test_reflection_runs():
    """start/complete reflection_run audit trail."""
    print("\n--- test_reflection_runs ---")
    db = create_test_db()
    rid = v2db.start_reflection_run(db, TEST_AGENT, TEST_SESSION)
    check("run ID positive", rid > 0)
    v2db.complete_reflection_run(db, rid, status="completed",
                                  events_processed=10, nodes_created=3,
                                  duration_ms=500)
    row = db.execute("SELECT * FROM reflection_runs WHERE id = ?", (rid,)).fetchone()
    check("status completed", row["status"] == "completed")
    check("events_processed 10", row["events_processed"] == 10)
    check("nodes_created 3", row["nodes_created"] == 3)
    check("duration_ms 500", row["duration_ms"] == 500)
    db.close()


def test_sophia_axioms():
    """add_axiom and get_axioms work correctly."""
    print("\n--- test_sophia_axioms ---")
    db = create_test_db()
    aid = v2db.add_axiom(db, TEST_AGENT, "Never delete user data without confirmation")
    check("axiom ID positive", aid > 0)
    axioms = v2db.get_axioms(db, TEST_AGENT)
    check("1 axiom", len(axioms) == 1)
    check("content matches", axioms[0]["content"] == "Never delete user data without confirmation")
    db.close()


def test_get_stats():
    """get_stats returns comprehensive statistics."""
    print("\n--- test_get_stats ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    seed_events(db, count=5)
    seed_episteme_node(db)
    v2db.add_axiom(db, TEST_AGENT, "Test axiom")
    stats = v2db.get_stats(db, TEST_AGENT)
    check("total_nodes >= 11", stats["total_nodes"] >= 11)
    check("total_events == 5", stats["total_events"] == 5)
    check("has nodes_by_layer", isinstance(stats["nodes_by_layer"], dict))
    check("has techne nodes", stats["nodes_by_layer"].get("techne", 0) >= 10)
    check("has episteme nodes", stats["nodes_by_layer"].get("episteme", 0) >= 1)
    check("axioms == 1", stats["axioms"] == 1)
    db.close()


def main():
    test_schema_creation()
    test_append_event()
    test_get_session_events()
    test_add_node()
    test_add_node_dedup()
    test_add_edge()
    test_get_edges()
    test_fts5_search()
    test_fts5_empty_query()
    test_count_supports()
    test_has_unresolved_contradictions()
    test_update_node_confidence()
    test_record_aisthesis()
    test_surfaced_nodes()
    test_reflection_runs()
    test_sophia_axioms()
    test_get_stats()

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
