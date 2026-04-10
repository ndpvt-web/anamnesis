#!/usr/bin/env python3
"""Tests for Hermes v2 retrieval engine (dual-process) and anti-pattern matching."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2 import retrieval
from v2.tests.fixtures import (
    create_test_db, TEST_AGENT, TEST_SESSION,
    seed_nodes, seed_edges, seed_episteme_node
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


def test_dual_retrieve_basic():
    """dual_retrieve returns results from FTS5."""
    print("\n--- test_dual_retrieve_basic ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    results = retrieval.dual_retrieve(db, TEST_AGENT, "pattern topic", top_k=5)
    check("returns results", len(results) > 0, f"got {len(results)}")
    check("max top_k results", len(results) <= 5)
    check("results have content", all(r.get("content") for r in results))
    check("results have source field", all(r.get("source") in ("fts5", "graph") for r in results))
    check("no _score in output", all("_score" not in r for r in results))
    db.close()


def test_dual_retrieve_empty_query():
    """dual_retrieve with empty/None query returns []."""
    print("\n--- test_dual_retrieve_empty_query ---")
    db = create_test_db()
    seed_nodes(db, count=5)
    check("empty string returns []", retrieval.dual_retrieve(db, TEST_AGENT, "", top_k=5) == [])
    check("None returns []", retrieval.dual_retrieve(db, TEST_AGENT, None, top_k=5) == [])
    check("whitespace returns []", retrieval.dual_retrieve(db, TEST_AGENT, "   ", top_k=5) == [])
    db.close()


def test_dual_retrieve_with_graph():
    """dual_retrieve includes graph-expanded results."""
    print("\n--- test_dual_retrieve_with_graph ---")
    db = create_test_db()
    nids = seed_nodes(db, count=10)
    seed_edges(db, node_ids=nids)
    results = retrieval.dual_retrieve(db, TEST_AGENT, "pattern topic",
                                       top_k=10, include_graph=True)
    graph_results = [r for r in results if r.get("source") == "graph"]
    # Graph expansion may or may not find results depending on edge structure
    check("returns results", len(results) > 0)
    # Check that graph_context field exists on some results
    has_context = any(r.get("graph_context") for r in results)
    check("some results have graph_context", has_context or len(graph_results) == 0,
          "graph expansion happened but no context")
    db.close()


def test_dual_retrieve_no_graph():
    """dual_retrieve with include_graph=False skips graph expansion."""
    print("\n--- test_dual_retrieve_no_graph ---")
    db = create_test_db()
    nids = seed_nodes(db, count=10)
    seed_edges(db, node_ids=nids)
    results = retrieval.dual_retrieve(db, TEST_AGENT, "pattern topic",
                                       top_k=10, include_graph=False)
    graph_results = [r for r in results if r.get("source") == "graph"]
    check("no graph results", len(graph_results) == 0)
    db.close()


def test_dual_retrieve_episteme_boost():
    """Episteme nodes get score boost over techne."""
    print("\n--- test_dual_retrieve_episteme_boost ---")
    db = create_test_db()
    # Insert a techne node and an episteme node with same content
    db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, helpful_count, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (TEST_AGENT, "fact", "techne", "error handling pattern for APIs",
          "principle", 0.70, 1, "candidate"))
    techne_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (techne_id, "error handling pattern for APIs", "", "principle"))

    db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, helpful_count, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (TEST_AGENT, "fact", "episteme", "error handling best practice for APIs",
          "principle", 0.90, 5, "crystallized"))
    epist_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (epist_id, "error handling best practice for APIs", "", "principle"))
    db.commit()

    results = retrieval.dual_retrieve(db, TEST_AGENT, "error handling APIs", top_k=5)
    check("returns results", len(results) >= 2, f"got {len(results)}")
    if len(results) >= 2:
        # Episteme should rank higher
        epist_idx = next((i for i, r in enumerate(results) if r["id"] == epist_id), None)
        techne_idx = next((i for i, r in enumerate(results) if r["id"] == techne_id), None)
        if epist_idx is not None and techne_idx is not None:
            check("episteme ranks higher", epist_idx < techne_idx,
                  f"episteme at {epist_idx}, techne at {techne_idx}")
        else:
            check("both nodes found", False, f"epist_idx={epist_idx}, techne_idx={techne_idx}")
    db.close()


def test_retrieve_for_error():
    """retrieve_for_error returns results for error text."""
    print("\n--- test_retrieve_for_error ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    results = retrieval.retrieve_for_error(db, TEST_AGENT, "pattern topic error", top_k=3)
    check("returns results", len(results) > 0)
    check("max 3 results", len(results) <= 3)
    check("results have source=fts5", all(r.get("source") == "fts5" for r in results))
    db.close()


def test_retrieve_anti_patterns_safe_commands():
    """retrieve_anti_patterns returns [] for read-only commands."""
    print("\n--- test_retrieve_anti_patterns_safe ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    safe_commands = [
        "cat /etc/hosts",
        "ls -la /home",
        "grep -r pattern .",
        "echo hello world",
        "head -n 10 file.txt",
        "python3 -c 'print(1)'",
        "mkdir -p /tmp/test",
        "touch /tmp/testfile",
    ]
    for cmd in safe_commands:
        result = retrieval.retrieve_anti_patterns(db, TEST_AGENT, cmd)
        check(f"safe: '{cmd[:30]}' returns []", result == [], f"got {len(result)} results")
    db.close()


def test_retrieve_anti_patterns_empty():
    """retrieve_anti_patterns handles edge cases."""
    print("\n--- test_retrieve_anti_patterns_empty ---")
    db = create_test_db()
    check("empty command returns []",
          retrieval.retrieve_anti_patterns(db, TEST_AGENT, "") == [])
    check("whitespace returns []",
          retrieval.retrieve_anti_patterns(db, TEST_AGENT, "   ") == [])
    # Command with no technical tokens
    check("plain words return []",
          retrieval.retrieve_anti_patterns(db, TEST_AGENT, "hello world") == [])
    db.close()


def test_retrieve_anti_patterns_no_false_positives():
    """Short flags don't match inside words (e.g. -p in happy-path)."""
    print("\n--- test_retrieve_anti_patterns_no_false_positives ---")
    db = create_test_db()
    # Insert an anti-pattern containing "happy-path"
    db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (TEST_AGENT, "fact", "techne",
          "Testing only happy-path scenarios without edge cases",
          "anti_pattern", 0.80, "candidate"))
    nid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (nid, "Testing only happy-path scenarios without edge cases", "", "anti_pattern"))
    db.commit()

    # Command with -p flag should NOT match "happy-path"
    result = retrieval.retrieve_anti_patterns(db, TEST_AGENT,
                                               "tar -xzf /some/archive.tar.gz -p")
    check("-p does not match happy-path", len(result) == 0, f"got {len(result)} matches")
    db.close()


def test_format_session_briefing():
    """format_session_briefing produces layered output."""
    print("\n--- test_format_session_briefing ---")
    db = create_test_db()
    seed_nodes(db, count=10)
    seed_episteme_node(db, content="Validated episteme knowledge")
    v2db.add_axiom(db, TEST_AGENT, "Never ignore user intent")

    briefing = retrieval.format_session_briefing(db, TEST_AGENT)
    check("non-empty briefing", len(briefing) > 0)
    check("contains agent ID", TEST_AGENT in briefing)
    check("contains SOPHIA header", "SOPHIA" in briefing)
    check("contains EPISTEME header", "EPISTEME" in briefing)
    check("contains TECHNE header", "TECHNE" in briefing)
    db.close()


def test_format_session_briefing_empty():
    """format_session_briefing returns '' when no nodes exist."""
    print("\n--- test_format_session_briefing_empty ---")
    db = create_test_db()
    briefing = retrieval.format_session_briefing(db, TEST_AGENT)
    check("empty briefing for empty DB", briefing == "")
    db.close()


def main():
    test_dual_retrieve_basic()
    test_dual_retrieve_empty_query()
    test_dual_retrieve_with_graph()
    test_dual_retrieve_no_graph()
    test_dual_retrieve_episteme_boost()
    test_retrieve_for_error()
    test_retrieve_anti_patterns_safe_commands()
    test_retrieve_anti_patterns_empty()
    test_retrieve_anti_patterns_no_false_positives()
    test_format_session_briefing()
    test_format_session_briefing_empty()

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
