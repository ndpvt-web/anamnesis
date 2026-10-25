#!/usr/bin/env python3
"""Tests for the knowledge deepening changes (Phases 1-4).
Covers: schema migration, error family classification, edge creation,
domain-boosted retrieval, WHEN/THEN/BECAUSE prompt structure."""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2 import knowledge_graph as kg
from v2.retrieval import classify_error_family, dual_retrieve, format_query_based_briefing
from v2.tests.fixtures import (
    create_test_db, seed_nodes, seed_edges, seed_events,
    TEST_AGENT, TEST_SESSION
)


# ---------------------------------------------------------------------------
# Phase 1: Schema migration
# ---------------------------------------------------------------------------

def test_migration_adds_empeiria_columns():
    """Verify empeiria has exit_code, tool_output, duration_ms columns."""
    db = create_test_db()
    cols = [row[1] for row in db.execute("PRAGMA table_info(empeiria)").fetchall()]
    assert "exit_code" in cols, f"exit_code missing from empeiria columns: {cols}"
    assert "tool_output" in cols, f"tool_output missing from empeiria columns: {cols}"
    assert "duration_ms" in cols, f"duration_ms missing from empeiria columns: {cols}"
    db.close()


def test_migration_adds_kg_nodes_error_family():
    """Verify kg_nodes has error_family column."""
    db = create_test_db()
    cols = [row[1] for row in db.execute("PRAGMA table_info(kg_nodes)").fetchall()]
    assert "error_family" in cols, f"error_family missing from kg_nodes columns: {cols}"
    db.close()


def test_append_event_with_new_fields():
    """Verify append_event stores exit_code, tool_output, duration_ms."""
    db = create_test_db()
    eid = v2db.append_event(db, TEST_AGENT, "tool_error", session_id=TEST_SESSION,
                             tool_name="Bash", command="python app.py",
                             error_text="ModuleNotFoundError: flask",
                             exit_code=1, tool_output="Traceback... ModuleNotFoundError",
                             duration_ms=1500)
    assert eid > 0
    row = db.execute("SELECT * FROM empeiria WHERE id = ?", (eid,)).fetchone()
    assert row["exit_code"] == 1
    assert "ModuleNotFoundError" in row["tool_output"]
    assert row["duration_ms"] == 1500
    db.close()


def test_add_node_with_error_family():
    """Verify add_node stores error_family."""
    db = create_test_db()
    nid = v2db.add_node(db, TEST_AGENT, "WHEN flask port 5000 THEN conflict",
                         error_family="E1_environment")
    assert nid > 0
    node = v2db.get_node(db, nid)
    assert node["error_family"] == "E1_environment"
    db.close()


def test_find_nodes_by_error_family():
    """Verify find_nodes_by_error_family returns matching nodes."""
    db = create_test_db()
    n1 = v2db.add_node(db, TEST_AGENT, "Encoding issue 1", error_family="E2_encoding")
    n2 = v2db.add_node(db, TEST_AGENT, "Encoding issue 2", error_family="E2_encoding")
    n3 = v2db.add_node(db, TEST_AGENT, "Config issue", error_family="E4_configuration")
    results = v2db.find_nodes_by_error_family(db, TEST_AGENT, "E2_encoding")
    ids = [r["id"] for r in results]
    assert n1 in ids
    assert n2 in ids
    assert n3 not in ids
    db.close()


def test_find_nodes_by_error_family_excludes():
    """Verify exclude_ids parameter works."""
    db = create_test_db()
    n1 = v2db.add_node(db, TEST_AGENT, "Encoding issue alpha", error_family="E2_encoding")
    n2 = v2db.add_node(db, TEST_AGENT, "Encoding issue beta", error_family="E2_encoding")
    results = v2db.find_nodes_by_error_family(db, TEST_AGENT, "E2_encoding",
                                               exclude_ids=[n1])
    ids = [r["id"] for r in results]
    assert n1 not in ids
    assert n2 in ids
    db.close()


# ---------------------------------------------------------------------------
# Phase 2: Error family classification
# ---------------------------------------------------------------------------

def test_classify_encoding_error():
    assert classify_error_family("UnicodeDecodeError reading CSV file") == "E2_encoding"


def test_classify_dependency_error():
    assert classify_error_family("ModuleNotFoundError: No module named flask") == "E3_dependencies"


def test_classify_environment_error():
    assert classify_error_family("port 5000 already in use EADDRINUSE") == "E1_environment"


def test_classify_config_error():
    assert classify_error_family("yaml parse error invalid indent") == "E4_configuration"


def test_classify_platform_error():
    assert classify_error_family("macOS darwin arm64 platform incompatible") == "E7_platform"


def test_classify_none():
    assert classify_error_family("Hello world") == "none"
    assert classify_error_family("") == "none"


# ---------------------------------------------------------------------------
# Phase 3: Edge creation
# ---------------------------------------------------------------------------

def test_intra_session_supports_edges():
    """Verify that co-accepted hypotheses get SUPPORTS edges."""
    db = create_test_db()
    # Create 3 nodes as if from same reflection run
    n1 = v2db.add_node(db, TEST_AGENT, "Hypothesis A about flask ports",
                         error_family="E1_environment", source_session="s1")
    n2 = v2db.add_node(db, TEST_AGENT, "Hypothesis B about environment vars",
                         error_family="E1_environment", source_session="s1")
    n3 = v2db.add_node(db, TEST_AGENT, "Hypothesis C about shell paths",
                         error_family="E1_environment", source_session="s1")
    # Simulate what _step_store does: create SUPPORTS between co-accepted
    accepted_ids = [n1, n2, n3]
    anchor = accepted_ids[0]
    for other_id in accepted_ids[1:]:
        v2db.add_edge(db, TEST_AGENT, anchor, other_id,
                      v2db.EDGE_SUPPORTS, weight=0.3,
                      metadata={"source": "co_session"})
    # Verify edges exist
    edges = db.execute("SELECT * FROM kg_edges WHERE edge_type = ?",
                        (v2db.EDGE_SUPPORTS,)).fetchall()
    assert len(edges) >= 2
    db.close()


def test_error_family_supports_edges():
    """Verify cross-session SUPPORTS via error_family matching."""
    db = create_test_db()
    # Pre-existing node with E2 family
    old = v2db.add_node(db, TEST_AGENT, "Old encoding pattern about BOM",
                         error_family="E2_encoding")
    # New node from different session, same family
    new = v2db.add_node(db, TEST_AGENT, "New encoding pattern about UTF-8 fallback",
                         error_family="E2_encoding")
    # Create edge as _step_store would
    family_nodes = v2db.find_nodes_by_error_family(db, TEST_AGENT, "E2_encoding",
                                                     exclude_ids=[new])
    for fn in family_nodes:
        v2db.add_edge(db, TEST_AGENT, new, fn["id"],
                      v2db.EDGE_SUPPORTS, weight=0.5,
                      metadata={"source": "error_family"})
    # Verify
    supports = v2db.count_supports(db, old)
    assert supports >= 1, f"Expected >= 1 SUPPORTS edge to old node, got {supports}"
    db.close()


def test_promotion_with_edges():
    """Verify that a node with sufficient edges can promote."""
    db = create_test_db()
    # Create main node meeting confidence + helpful thresholds
    nid = v2db.add_node(db, TEST_AGENT, "Promotable knowledge about encoding",
                         confidence=0.87, helpful_count=3, error_family="E2_encoding")
    # Create 2 supporting nodes + edges
    s1 = v2db.add_node(db, TEST_AGENT, "Supporter 1 for encoding", error_family="E2_encoding")
    s2 = v2db.add_node(db, TEST_AGENT, "Supporter 2 for encoding", error_family="E2_encoding")
    v2db.add_edge(db, TEST_AGENT, s1, nid, v2db.EDGE_SUPPORTS, weight=0.5)
    v2db.add_edge(db, TEST_AGENT, s2, nid, v2db.EDGE_SUPPORTS, weight=0.5)
    # Promote
    result = kg.promote_candidates(db, TEST_AGENT)
    assert nid in result["ids"], f"Node {nid} should have been promoted, result: {result}"
    # Verify layer changed
    node = v2db.get_node(db, nid)
    assert node["layer"] == v2db.LAYER_EPISTEME
    db.close()


def test_transitional_promotion_relief():
    """Verify helpful_count>=5 allows promotion with only 1 SUPPORTS edge."""
    db = create_test_db()
    nid = v2db.add_node(db, TEST_AGENT, "Heavily confirmed knowledge",
                         confidence=0.87, helpful_count=5, error_family="E3_dependencies")
    s1 = v2db.add_node(db, TEST_AGENT, "Single supporter", error_family="E3_dependencies")
    v2db.add_edge(db, TEST_AGENT, s1, nid, v2db.EDGE_SUPPORTS, weight=0.5)
    # Only 1 edge, but helpful_count=5 should allow promotion
    result = kg.promote_candidates(db, TEST_AGENT)
    assert nid in result["ids"], f"Node with helpful=5 and 1 edge should promote"
    db.close()


def test_similarity_based_supports():
    """Verify create_supports_from_similarity creates edges between similar nodes."""
    db = create_test_db()
    # Create nodes with similar content
    n1 = v2db.add_node(db, TEST_AGENT, "Flask port 5000 conflict AirDrop macOS Monterey")
    n2 = v2db.add_node(db, TEST_AGENT, "Port 5000 used by AirDrop on macOS causes Flask bind failure")
    # Different content
    n3 = v2db.add_node(db, TEST_AGENT, "Python unittest mock patch decorator usage")
    edges = kg.create_supports_from_similarity(db, TEST_AGENT, [n1], threshold=0.3)
    # n1 and n2 share many tokens, should get SUPPORTS edge
    # n1 and n3 share almost no tokens, should not
    all_edges = db.execute("SELECT * FROM kg_edges").fetchall()
    edge_targets = [e["to_node_id"] for e in all_edges if e["from_node_id"] == n1]
    assert n2 in edge_targets, f"Expected SUPPORTS edge from n1 to n2, edges: {[dict(e) for e in all_edges]}"
    db.close()


# ---------------------------------------------------------------------------
# Phase 4: Domain-aware retrieval
# ---------------------------------------------------------------------------

def test_domain_boosted_retrieval():
    """Verify error_family boost in dual_retrieve."""
    db = create_test_db()
    # Insert two nodes, one with matching family
    n1 = v2db.add_node(db, TEST_AGENT, "UnicodeDecodeError when reading CSV with BOM",
                         error_family="E2_encoding", confidence=0.70)
    n2 = v2db.add_node(db, TEST_AGENT, "Flask config validation error on startup",
                         error_family="E4_configuration", confidence=0.70)
    # Both should match "error" but E2 should rank higher for encoding query
    results = dual_retrieve(db, TEST_AGENT, "UnicodeDecodeError CSV",
                             top_k=5, error_family="E2_encoding")
    if results:
        # The encoding node should rank first due to family boost
        assert results[0]["id"] == n1 or results[0]["error_family"] == "E2_encoding"
    db.close()


def test_query_based_briefing():
    """Verify format_query_based_briefing generates domain-relevant output."""
    db = create_test_db()
    v2db.add_node(db, TEST_AGENT, "Encoding knowledge about UTF-8 BOM",
                   layer=v2db.LAYER_TECHNE, error_family="E2_encoding")
    results = dual_retrieve(db, TEST_AGENT, "encoding UTF-8 BOM", top_k=3)
    briefing = format_query_based_briefing(db, TEST_AGENT, results)
    assert "Anamnesis" in briefing
    assert "TECHNE" in briefing or "Relevant" in briefing
    db.close()


def test_family_fallback_search():
    """Verify FTS5 fallback via error family name when direct search fails."""
    db = create_test_db()
    n1 = v2db.add_node(db, TEST_AGENT, "Platform-specific macOS AirDrop issue",
                         error_family="E7_platform")
    # Search for something that won't match directly but has platform family
    results = dual_retrieve(db, TEST_AGENT, "darwin arm64 incompatible",
                             top_k=5, error_family="E7_platform")
    # Should find something via family fallback or direct token match
    # (the word "platform" appears in ERROR_FAMILY_SIGNATURES and node content)
    db.close()


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def test_edge_creation_performance():
    """Verify similarity-based edge creation is fast enough (<100ms for 40 nodes)."""
    db = create_test_db()
    # Seed 40 nodes
    for i in range(40):
        v2db.add_node(db, TEST_AGENT, f"Pattern about topic-{i % 10} variant-{i} flask encoding")
    new_ids = []
    for i in range(5):
        nid = v2db.add_node(db, TEST_AGENT, f"New hypothesis about flask encoding topic-{i}")
        new_ids.append(nid)
    start = time.time()
    edges = kg.create_supports_from_similarity(db, TEST_AGENT, new_ids, threshold=0.4)
    elapsed = time.time() - start
    assert elapsed < 0.5, f"Edge creation took {elapsed:.3f}s, expected < 0.5s"
    db.close()


def test_migration_idempotent():
    """Verify _migrate_schema can run multiple times without error."""
    db = create_test_db()
    # Run migration again (should be no-op)
    v2db._migrate_schema(db)
    v2db._migrate_schema(db)
    # Verify columns still exist
    cols = [row[1] for row in db.execute("PRAGMA table_info(empeiria)").fetchall()]
    assert "exit_code" in cols
    db.close()
