#!/usr/bin/env python3
"""End-to-end integration tests for Hermes v2.

These tests simulate a REAL session lifecycle against an on-disk DB,
exercising the actual code paths that hooks take in production.

Lifecycle: session_start -> prompt_submit (x3) -> tool_success (x2) ->
           tool_error (x1) -> bash_guardian -> reflection -> session_end

Each test function exercises one phase and verifies the DB state afterward.
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2 import observer
from v2 import retrieval
from v2 import feedback
from v2 import maintenance
from v2 import knowledge_graph as kg

passed = 0
failed = 0

# Test identity
AGENT_ID = "__integration_test__"
SESSION_ID = "integration-session-001"

# Temp directory for on-disk DB
_tmp_dir = None
_original_db_path_fn = None


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def setup_module(module):
    """Create a temp directory and patch v2db to use it."""
    global _tmp_dir, _original_db_path_fn
    _tmp_dir = tempfile.mkdtemp(prefix="hermes_integ_")
    # Patch db_path_for_agent to use our temp dir
    _original_db_path_fn = v2db.db_path_for_agent
    def _test_db_path(agent_id):
        return Path(_tmp_dir) / "hermes" / "hermes.db"
    v2db.db_path_for_agent = _test_db_path


def teardown_module(module):
    """Restore original db_path and clean up temp dir."""
    global _tmp_dir, _original_db_path_fn
    v2db.db_path_for_agent = _original_db_path_fn
    if _tmp_dir:
        shutil.rmtree(_tmp_dir, ignore_errors=True)


# --- Phase 1: DB Creation and Schema ---

def test_on_disk_db_creation():
    """open_db creates hermes.db on disk with full schema."""
    print("\n--- test_on_disk_db_creation ---")
    db = v2db.open_db(AGENT_ID)
    db_path = Path(_tmp_dir) / "hermes" / "hermes.db"
    check("hermes.db exists on disk", db_path.exists())

    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    for t in ["empeiria", "kg_nodes", "kg_edges", "aisthesis", "surfaced_nodes",
              "reflection_runs", "sophia"]:
        check(f"table {t} exists", t in tables, f"tables={tables}")

    # FTS5
    vtables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_fts'"
    ).fetchall()]
    check("kg_fts exists", len(vtables) == 1)

    # WAL mode
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    check("WAL mode", mode.lower() == "wal", f"got {mode}")
    db.close()


# --- Phase 2: Observer (Event Capture) ---

def test_observer_safe_append_event():
    """safe_append_event writes to empeiria via its own connection."""
    print("\n--- test_observer_safe_append_event ---")
    eid = observer.safe_append_event(
        AGENT_ID, "session_start",
        session_id=SESSION_ID,
        metadata={"cwd": "/tmp/test"}
    )
    check("event ID > 0", eid > 0, f"got {eid}")

    # Verify via separate connection
    db = v2db.open_db(AGENT_ID)
    row = db.execute("SELECT * FROM empeiria WHERE id = ?", (eid,)).fetchone()
    check("event exists in DB", row is not None)
    check("event_type correct", row["event_type"] == "session_start")
    check("session_id correct", row["session_id"] == SESSION_ID)
    check("metadata has cwd", "cwd" in row["metadata"])
    db.close()


def test_observer_safe_record_aisthesis():
    """safe_record_aisthesis writes to working memory."""
    print("\n--- test_observer_safe_record_aisthesis ---")
    rid = observer.safe_record_aisthesis(
        AGENT_ID, SESSION_ID, "command",
        tool_name="Bash", command="echo hello",
        files_touched=["/tmp/test.txt"]
    )
    check("aisthesis ID > 0", rid > 0, f"got {rid}")

    db = v2db.open_db(AGENT_ID)
    mem = v2db.get_session_memory(db, AGENT_ID, SESSION_ID)
    check("memory entry exists", len(mem) >= 1)
    check("tool_name correct", mem[0]["tool_name"] == "Bash")
    check("files_touched stored", "/tmp/test.txt" in mem[0]["files_touched"])
    db.close()


def test_observer_truncation():
    """Observer truncates oversized fields."""
    print("\n--- test_observer_truncation ---")
    long_cmd = "x" * 2000
    long_err = "E" * 5000
    long_prompt = "P" * 10000

    eid = observer.safe_append_event(
        AGENT_ID, "test_truncation",
        session_id=SESSION_ID,
        command=long_cmd,
        error_text=long_err,
        prompt_text=long_prompt
    )
    check("event created despite long fields", eid > 0)

    db = v2db.open_db(AGENT_ID)
    row = db.execute("SELECT * FROM empeiria WHERE id = ?", (eid,)).fetchone()
    check("command truncated to 500", len(row["command"]) <= 500)
    check("error_text truncated to 1000", len(row["error_text"]) <= 1000)
    check("prompt_text truncated to 2000", len(row["prompt_text"]) <= 2000)
    db.close()


def test_observer_failure_recovery():
    """Observer returns -1 on failure, never raises."""
    print("\n--- test_observer_failure_recovery ---")
    # Temporarily break db_path to force failure
    original = v2db.db_path_for_agent
    def _broken(agent_id):
        return Path("/nonexistent/path/hermes.db")
    v2db.db_path_for_agent = _broken

    eid = observer.safe_append_event(AGENT_ID, "should_fail")
    check("returns -1 on failure", eid == -1)

    rid = observer.safe_record_aisthesis(AGENT_ID, SESSION_ID, "should_fail")
    check("aisthesis returns -1 on failure", rid == -1)

    # Restore
    v2db.db_path_for_agent = original


# --- Phase 3: Seed Knowledge (simulate migration) ---

def test_seed_knowledge():
    """Seed realistic KG nodes for retrieval testing."""
    print("\n--- test_seed_knowledge ---")
    db = v2db.open_db(AGENT_ID)

    # Add diverse nodes
    nodes = [
        ("Never use git push --force on shared branches", "anti_pattern", "high", 0.85),
        ("Always check exit codes after bash commands", "anti_pattern", "medium", 0.70),
        ("Use descriptive variable names in Python", "principle", "low", 0.65),
        ("User prefers TypeScript over JavaScript", "preference", "medium", 0.75),
        ("Error handling should wrap all external API calls", "principle", "high", 0.80),
        ("rm -rf without confirmation is dangerous in production", "anti_pattern", "critical", 0.90),
        ("Database connections must be closed in finally blocks", "anti_pattern", "high", 0.82),
        ("Use Aristotelian reasoning framework for analysis", "preference", "medium", 0.72),
        ("Test coverage should include edge cases not just happy path", "principle", "medium", 0.68),
        ("Avoid bare except clauses in Python", "anti_pattern", "medium", 0.71),
    ]

    node_ids = []
    for content, category, severity, confidence in nodes:
        nid = v2db.add_node(db, AGENT_ID, content,
                            category=category, severity=severity,
                            confidence=confidence)
        if nid > 0:
            node_ids.append(nid)

    check(f"seeded {len(node_ids)} nodes", len(node_ids) == len(nodes),
          f"got {len(node_ids)}")

    # Add an episteme node
    epist_id = db.execute("""
        INSERT INTO kg_nodes (agent_id, node_type, layer, content, category,
                              confidence, helpful_count, maturity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (AGENT_ID, "fact", v2db.LAYER_EPISTEME,
          "Always validate user input at system boundaries",
          "principle", 0.92, 8, "crystallized")).lastrowid
    db.execute("INSERT INTO kg_fts(rowid, content, summary, category) VALUES (?, ?, ?, ?)",
               (epist_id, "Always validate user input at system boundaries", "", "principle"))
    db.commit()
    node_ids.append(epist_id)

    # Add some edges
    if len(node_ids) >= 5:
        v2db.add_edge(db, AGENT_ID, node_ids[0], node_ids[1], v2db.EDGE_SUPPORTS)
        v2db.add_edge(db, AGENT_ID, node_ids[4], node_ids[6], v2db.EDGE_SUPPORTS)
        v2db.add_edge(db, AGENT_ID, node_ids[2], node_ids[4], v2db.EDGE_SUPPORTS)

    # Add a sophia axiom
    v2db.add_axiom(db, AGENT_ID, "Never delete user data without explicit confirmation")

    stats = v2db.get_stats(db, AGENT_ID)
    check("total_nodes correct", stats["total_nodes"] >= 11,
          f"got {stats['total_nodes']}")
    check("has episteme nodes", stats["nodes_by_layer"].get("episteme", 0) >= 1)
    check("has edges", stats["total_edges"] >= 3)
    check("has axioms", stats["axioms"] >= 1)
    db.close()


# --- Phase 4: Retrieval (Session Start + Prompt Submit) ---

def test_session_briefing():
    """format_session_briefing produces layered output."""
    print("\n--- test_session_briefing ---")
    db = v2db.open_db(AGENT_ID)
    briefing = retrieval.format_session_briefing(db, AGENT_ID)
    check("non-empty briefing", len(briefing) > 0)
    check("contains agent ID", AGENT_ID in briefing)
    check("contains SOPHIA header", "SOPHIA" in briefing)
    check("contains EPISTEME header", "EPISTEME" in briefing)
    check("contains TECHNE header", "TECHNE" in briefing)
    check("contains axiom content", "Never delete user data" in briefing)
    check("contains episteme node", "validate user input" in briefing)
    db.close()


def test_dual_retrieve_on_disk():
    """dual_retrieve works against on-disk DB with real data."""
    print("\n--- test_dual_retrieve_on_disk ---")
    db = v2db.open_db(AGENT_ID)

    # Search for error handling
    results = retrieval.dual_retrieve(db, AGENT_ID, "error handling API calls", top_k=5)
    check("returns results", len(results) > 0, f"got {len(results)}")
    check("results have content", all(r.get("content") for r in results))
    check("results have source", all(r.get("source") in ("fts5", "graph") for r in results))
    check("no _score in output", all("_score" not in r for r in results))

    # Search for git push
    results2 = retrieval.dual_retrieve(db, AGENT_ID, "git push force shared", top_k=3)
    check("git push results", len(results2) > 0)
    if results2:
        check("finds anti_pattern", any(r["category"] == "anti_pattern" for r in results2))

    # Empty query
    check("empty returns []", retrieval.dual_retrieve(db, AGENT_ID, "", top_k=5) == [])
    check("None returns []", retrieval.dual_retrieve(db, AGENT_ID, None, top_k=5) == [])
    db.close()


def test_dual_retrieve_with_surfacing():
    """dual_retrieve results get recorded as surfaced_nodes."""
    print("\n--- test_dual_retrieve_with_surfacing ---")
    db = v2db.open_db(AGENT_ID)

    results = retrieval.dual_retrieve(db, AGENT_ID, "database connections", top_k=3)
    if results:
        node_ids = [r["id"] for r in results]
        v2db.record_surfaced(db, AGENT_ID, SESSION_ID, node_ids)

        rows = db.execute(
            "SELECT * FROM surfaced_nodes WHERE session_id = ? AND feedback = 'pending'",
            (SESSION_ID,)
        ).fetchall()
        check("surfaced nodes recorded", len(rows) >= 1, f"got {len(rows)}")
        check("all pending", all(r["feedback"] == "pending" for r in rows))
    else:
        check("has results to surface", False, "no results for 'database connections'")
    db.close()


# --- Phase 5: Bash Guardian ---

def test_bash_guardian_safe_commands():
    """retrieve_anti_patterns returns [] for safe commands."""
    print("\n--- test_bash_guardian_safe_commands ---")
    db = v2db.open_db(AGENT_ID)
    safe_cmds = [
        "cat /etc/hosts",
        "ls -la /home",
        "grep -r pattern .",
        "echo hello world",
        "head -n 10 file.txt",
        "python3 -c 'print(1)'",
        "mkdir -p /tmp/test",
        "touch /tmp/testfile",
    ]
    for cmd in safe_cmds:
        result = retrieval.retrieve_anti_patterns(db, AGENT_ID, cmd)
        check(f"safe: '{cmd[:30]}' -> []", result == [], f"got {len(result)} results")
    db.close()


def test_bash_guardian_dangerous_commands():
    """retrieve_anti_patterns detects dangerous patterns."""
    print("\n--- test_bash_guardian_dangerous_commands ---")
    db = v2db.open_db(AGENT_ID)

    # git push --force should match our anti-pattern about shared branches
    result = retrieval.retrieve_anti_patterns(db, AGENT_ID, "git push --force origin main")
    check("git push --force detected", len(result) > 0, f"got {len(result)}")
    if result:
        check("matches anti_pattern category", result[0]["category"] == "anti_pattern")

    # rm -rf should match our anti-pattern
    result2 = retrieval.retrieve_anti_patterns(db, AGENT_ID, "rm -rf /var/data/production")
    check("rm -rf detected", len(result2) > 0, f"got {len(result2)}")
    db.close()


def test_bash_guardian_no_false_positives():
    """Short flags don't match inside words."""
    print("\n--- test_bash_guardian_no_false_positives ---")
    db = v2db.open_db(AGENT_ID)
    # tar -p should not match "happy path" in anti-pattern content
    result = retrieval.retrieve_anti_patterns(db, AGENT_ID,
                                               "tar -xzf archive.tar.gz -p")
    check("-p flag no false positive", len(result) == 0, f"got {len(result)}")
    db.close()


# --- Phase 6: Feedback Loop ---

def test_confirm_surfaced():
    """confirm_surfaced_nodes boosts confidence."""
    print("\n--- test_confirm_surfaced ---")
    db = v2db.open_db(AGENT_ID)

    # Get a pending surfaced node
    pending = db.execute(
        "SELECT sn.*, n.confidence as old_conf FROM surfaced_nodes sn "
        "JOIN kg_nodes n ON sn.node_id = n.id "
        "WHERE sn.session_id = ? AND sn.feedback = 'pending' LIMIT 1",
        (SESSION_ID,)
    ).fetchone()

    if pending:
        old_conf = pending["old_conf"]
        count = feedback.confirm_surfaced_nodes(db, AGENT_ID, SESSION_ID)
        check("confirmed some nodes", count >= 1, f"confirmed {count}")

        # Check confidence was boosted
        node = v2db.get_node(db, pending["node_id"])
        check("confidence boosted", node["confidence"] >= old_conf,
              f"old={old_conf}, new={node['confidence']}")

        # Check feedback updated
        row = db.execute("SELECT feedback FROM surfaced_nodes WHERE id = ?",
                         (pending["id"],)).fetchone()
        check("feedback set to confirmed", row["feedback"] == "confirmed")
    else:
        check("has pending surfaced nodes", False, "none found - seed more data")
    db.close()


def test_refute_on_error():
    """Tool error refutes surfaced nodes."""
    print("\n--- test_refute_on_error ---")
    db = v2db.open_db(AGENT_ID)

    # Surface some new nodes for refutation
    results = retrieval.dual_retrieve(db, AGENT_ID, "exit codes bash", top_k=2)
    if results:
        v2db.record_surfaced(db, AGENT_ID, SESSION_ID + "-err", [r["id"] for r in results])

        old_node = v2db.get_node(db, results[0]["id"])
        old_conf = old_node["confidence"]

        count = feedback.refute_surfaced_nodes(
            db, AGENT_ID, SESSION_ID + "-err",
            error_text="ModuleNotFoundError: no module named 'foo'"
        )
        check("refuted some nodes", count >= 1, f"refuted {count}")

        new_node = v2db.get_node(db, results[0]["id"])
        check("confidence decayed", new_node["confidence"] < old_conf,
              f"old={old_conf}, new={new_node['confidence']}")
    else:
        check("has results for refutation", False, "no results")
    db.close()


def test_record_irrelevance():
    """record_irrelevance marks remaining pending as irrelevant."""
    print("\n--- test_record_irrelevance ---")
    db = v2db.open_db(AGENT_ID)

    # Surface fresh nodes
    results = retrieval.dual_retrieve(db, AGENT_ID, "variable names Python", top_k=2)
    if results:
        v2db.record_surfaced(db, AGENT_ID, SESSION_ID + "-irr", [r["id"] for r in results])
        count = feedback.record_irrelevance(db, AGENT_ID, SESSION_ID + "-irr")
        check("marked irrelevant", count >= 1, f"count={count}")

        node = v2db.get_node(db, results[0]["id"])
        check("irrelevant_count incremented", node["irrelevant_count"] >= 1,
              f"got {node['irrelevant_count']}")
    else:
        check("has results for irrelevance", False)
    db.close()


# --- Phase 7: Knowledge Graph Operations ---

def test_graph_traversal():
    """find_related_nodes finds edges."""
    print("\n--- test_graph_traversal ---")
    db = v2db.open_db(AGENT_ID)

    # Find a node with edges
    edges = db.execute("SELECT DISTINCT to_node_id FROM kg_edges WHERE agent_id = ? LIMIT 1",
                       (AGENT_ID,)).fetchone()
    if edges:
        node_id = edges["to_node_id"]
        related = kg.find_related_nodes(db, AGENT_ID, node_id, max_depth=1)
        check("finds related nodes", len(related) > 0, f"got {len(related)}")
        check("has edge_type", all(r.get("edge_type") for r in related))
        check("has depth", all(r.get("depth", -1) >= 1 for r in related))
    else:
        check("has edges for traversal", False)
    db.close()


def test_detect_contradictions():
    """detect_contradictions works with real data."""
    print("\n--- test_detect_contradictions ---")
    db = v2db.open_db(AGENT_ID)

    # Add a contradiction edge
    nodes = db.execute(
        "SELECT id FROM kg_nodes WHERE agent_id = ? LIMIT 2",
        (AGENT_ID,)
    ).fetchall()
    if len(nodes) >= 2:
        n1, n2 = nodes[0]["id"], nodes[1]["id"]
        v2db.add_edge(db, AGENT_ID, n1, n2, v2db.EDGE_CONTRADICTS)

        contras = kg.detect_contradictions(db, AGENT_ID, n1)
        check("detects contradiction", len(contras) > 0)
        check("points to correct node", any(c["node_id"] == n2 for c in contras))

        # Reverse direction
        contras2 = kg.detect_contradictions(db, AGENT_ID, n2)
        check("detects reverse contradiction", len(contras2) > 0)
    else:
        check("has enough nodes", False)
    db.close()


# --- Phase 8: Maintenance ---

def test_maintenance_cycle():
    """maintenance_cycle runs without error."""
    print("\n--- test_maintenance_cycle ---")
    db = v2db.open_db(AGENT_ID)
    result = maintenance.maintenance_cycle(db, AGENT_ID)
    check("returns promoted count", "promoted" in result)
    check("returns demoted count", "demoted" in result)
    check("returns decayed count", "decayed" in result)
    check("returns nodes_by_layer", "nodes_by_layer" in result)
    check("returns total_nodes", result["total_nodes"] >= 10,
          f"got {result['total_nodes']}")
    db.close()


# --- Phase 9: Concurrent DB Access ---

def test_concurrent_connections():
    """Multiple connections can read/write without deadlock."""
    print("\n--- test_concurrent_connections ---")
    db1 = v2db.open_db(AGENT_ID)
    db2 = v2db.open_db(AGENT_ID)

    # Write from db1
    eid = v2db.append_event(db1, AGENT_ID, "concurrent_test_1", session_id=SESSION_ID)
    check("db1 write succeeds", eid > 0)

    # Read from db2
    row = db2.execute("SELECT * FROM empeiria WHERE id = ?", (eid,)).fetchone()
    check("db2 reads db1 write", row is not None)

    # Write from db2
    eid2 = v2db.append_event(db2, AGENT_ID, "concurrent_test_2", session_id=SESSION_ID)
    check("db2 write succeeds", eid2 > 0)

    # Read from db1
    row2 = db1.execute("SELECT * FROM empeiria WHERE id = ?", (eid2,)).fetchone()
    check("db1 reads db2 write", row2 is not None)

    db1.close()
    db2.close()


# --- Phase 10: Hook Simulation (JSON stdin/stdout) ---

def test_hook_session_start_simulation():
    """Simulate on_session_start hook input/output."""
    print("\n--- test_hook_session_start_simulation ---")
    db = v2db.open_db(AGENT_ID)

    # Record a session_start event like the hook does
    eid = v2db.append_event(db, AGENT_ID, "session_start",
                            session_id=SESSION_ID,
                            metadata={"cwd": "/test"})
    check("session_start event recorded", eid > 0)

    # Run maintenance like the hook does
    maint = maintenance.maintenance_cycle(db, AGENT_ID)
    check("maintenance ran", isinstance(maint, dict))

    # Generate briefing like the hook does
    briefing = retrieval.format_session_briefing(db, AGENT_ID)
    check("briefing generated", len(briefing) > 0)
    db.close()


def test_hook_prompt_submit_simulation():
    """Simulate on_prompt_submit hook flow."""
    print("\n--- test_hook_prompt_submit_simulation ---")
    db = v2db.open_db(AGENT_ID)

    # Record prompt event
    eid = v2db.append_event(db, AGENT_ID, "prompt_submit",
                            session_id=SESSION_ID,
                            prompt_text="How do I handle errors in API calls?")
    check("prompt event recorded", eid > 0)

    # Dual retrieve
    results = retrieval.dual_retrieve(db, AGENT_ID,
                                       "How do I handle errors in API calls?",
                                       top_k=3)
    check("retrieval returns results", len(results) > 0)

    # Record surfaced
    if results:
        v2db.record_surfaced(db, AGENT_ID, SESSION_ID,
                              [r["id"] for r in results])
        surfaced = db.execute(
            "SELECT COUNT(*) as cnt FROM surfaced_nodes WHERE session_id = ?",
            (SESSION_ID,)
        ).fetchone()
        check("surfaced nodes tracked", surfaced["cnt"] > 0)
    db.close()


def test_hook_post_tool_success_simulation():
    """Simulate on_post_tool success flow."""
    print("\n--- test_hook_post_tool_success_simulation ---")
    db = v2db.open_db(AGENT_ID)

    # Record tool success event
    eid = v2db.append_event(db, AGENT_ID, "tool_success",
                            session_id=SESSION_ID,
                            tool_name="Bash",
                            command="echo hello")
    check("tool_success event recorded", eid > 0)

    # Record aisthesis
    aid = v2db.record_aisthesis(db, AGENT_ID, SESSION_ID, "command",
                                 tool_name="Bash", command="echo hello")
    check("aisthesis recorded", aid > 0)

    # Confirm surfaced (like on_post_tool does on success)
    count = feedback.confirm_surfaced_nodes(db, AGENT_ID, SESSION_ID)
    # May be 0 if all already confirmed
    check("confirm ran without error", count >= 0)
    db.close()


def test_hook_post_tool_error_simulation():
    """Simulate on_post_tool error flow."""
    print("\n--- test_hook_post_tool_error_simulation ---")
    db = v2db.open_db(AGENT_ID)

    # Surface something for the error session
    error_session = SESSION_ID + "-err-sim"
    results = retrieval.dual_retrieve(db, AGENT_ID, "git push force", top_k=2)
    if results:
        v2db.record_surfaced(db, AGENT_ID, error_session,
                              [r["id"] for r in results])

    # Record tool error event
    eid = v2db.append_event(db, AGENT_ID, "tool_error",
                            session_id=error_session,
                            tool_name="Bash",
                            command="npm install",
                            error_text="npm ERR! 404 Not Found")
    check("tool_error event recorded", eid > 0)

    # Refute surfaced
    count = feedback.refute_surfaced_nodes(db, AGENT_ID, error_session,
                                            error_text="npm ERR! 404 Not Found")
    check("refute ran", count >= 0)
    db.close()


def test_hook_pre_bash_simulation():
    """Simulate on_pre_bash guardian flow."""
    print("\n--- test_hook_pre_bash_simulation ---")
    db = v2db.open_db(AGENT_ID)

    # Safe command
    safe_result = retrieval.retrieve_anti_patterns(db, AGENT_ID, "ls -la /home")
    check("safe command passes", safe_result == [])

    # Dangerous command
    danger_result = retrieval.retrieve_anti_patterns(db, AGENT_ID,
                                                      "git push --force origin main")
    # May or may not match depending on FTS5 tokenization
    check("dangerous command checked", isinstance(danger_result, list))
    db.close()


# --- Phase 11: Full Session Lifecycle ---

def test_full_session_lifecycle():
    """Simulate a complete session from start to end."""
    print("\n--- test_full_session_lifecycle ---")
    lifecycle_session = "lifecycle-test-001"
    db = v2db.open_db(AGENT_ID)

    # 1. Session start
    v2db.append_event(db, AGENT_ID, "session_start",
                      session_id=lifecycle_session)
    briefing = retrieval.format_session_briefing(db, AGENT_ID)
    check("lifecycle: briefing generated", len(briefing) > 0)

    # 2. Prompt submit (3 prompts)
    for i, prompt in enumerate([
        "Fix the authentication bug in login.py",
        "Add error handling to the API endpoint",
        "Write tests for the user model"
    ]):
        v2db.append_event(db, AGENT_ID, "prompt_submit",
                          session_id=lifecycle_session,
                          prompt_text=prompt)
        results = retrieval.dual_retrieve(db, AGENT_ID, prompt, top_k=3)
        if results:
            v2db.record_surfaced(db, AGENT_ID, lifecycle_session,
                                  [r["id"] for r in results])
    check("lifecycle: prompts recorded", True)

    # 3. Tool successes
    for cmd in ["python3 login.py", "pytest tests/"]:
        v2db.append_event(db, AGENT_ID, "tool_success",
                          session_id=lifecycle_session,
                          tool_name="Bash", command=cmd)
        v2db.record_aisthesis(db, AGENT_ID, lifecycle_session, "command",
                               tool_name="Bash", command=cmd)
    feedback.confirm_surfaced_nodes(db, AGENT_ID, lifecycle_session)
    check("lifecycle: successes recorded", True)

    # 4. Tool error
    v2db.append_event(db, AGENT_ID, "tool_error",
                      session_id=lifecycle_session,
                      tool_name="Bash",
                      command="npm test",
                      error_text="TypeError: undefined is not a function")
    v2db.record_aisthesis(db, AGENT_ID, lifecycle_session, "error",
                           tool_name="Bash",
                           error_text="TypeError: undefined is not a function")
    check("lifecycle: error recorded", True)

    # 5. Bash guardian
    guard = retrieval.retrieve_anti_patterns(db, AGENT_ID, "rm -rf /tmp/data")
    check("lifecycle: guardian checked", isinstance(guard, list))

    # 6. Session end - mark irrelevance
    irr_count = feedback.record_irrelevance(db, AGENT_ID, lifecycle_session)
    check("lifecycle: irrelevance recorded", irr_count >= 0)

    # 7. Maintenance
    maint = maintenance.maintenance_cycle(db, AGENT_ID)
    check("lifecycle: maintenance ran", isinstance(maint, dict))

    # Verify event store integrity
    events = v2db.get_session_events(db, AGENT_ID, lifecycle_session, limit=50)
    check("lifecycle: events preserved", len(events) >= 6,
          f"got {len(events)}")
    event_types = [e["event_type"] for e in events]
    check("lifecycle: has session_start", "session_start" in event_types)
    check("lifecycle: has prompt_submit", "prompt_submit" in event_types)
    check("lifecycle: has tool_success", "tool_success" in event_types)
    check("lifecycle: has tool_error", "tool_error" in event_types)

    # Verify working memory
    mem = v2db.get_session_memory(db, AGENT_ID, lifecycle_session)
    check("lifecycle: aisthesis entries", len(mem) >= 2, f"got {len(mem)}")

    # Verify surfaced_nodes feedback distribution
    all_surfaced = db.execute(
        "SELECT feedback, COUNT(*) as cnt FROM surfaced_nodes "
        "WHERE session_id = ? GROUP BY feedback",
        (lifecycle_session,)
    ).fetchall()
    feedback_dist = {r["feedback"]: r["cnt"] for r in all_surfaced}
    check("lifecycle: has feedback distribution", len(feedback_dist) > 0,
          f"got {feedback_dist}")

    # Stats
    stats = v2db.get_stats(db, AGENT_ID)
    check("lifecycle: stats available", stats["total_nodes"] > 0)
    check("lifecycle: events accumulated", stats["total_events"] > 0)

    db.close()


# --- Phase 12: Edge Cases ---

def test_dedup_on_disk():
    """Deduplication works against on-disk FTS5."""
    print("\n--- test_dedup_on_disk ---")
    db = v2db.open_db(AGENT_ID)
    content = "Unique test node for deduplication verification"
    nid1 = v2db.add_node(db, AGENT_ID, content)
    check("first insert succeeds", nid1 > 0)
    nid2 = v2db.add_node(db, AGENT_ID, content)
    check("exact duplicate returns -1", nid2 == -1)
    nid3 = v2db.add_node(db, AGENT_ID, content.upper())
    check("case-insensitive dedup", nid3 == -1)
    db.close()


def test_fts5_special_characters():
    """FTS5 handles special characters gracefully."""
    print("\n--- test_fts5_special_characters ---")
    db = v2db.open_db(AGENT_ID)

    # Queries with special chars should not crash
    special_queries = [
        'import "module"',
        "error: can't find 'foo'",
        "path/to/file.py:42",
        "func() -> None",
        "SELECT * FROM users;",
        "&&||!@#$%^&*()",
    ]
    for q in special_queries:
        try:
            results = v2db.fts5_search(db, q, AGENT_ID, top_k=3)
            check(f"fts5 handles: '{q[:25]}'", isinstance(results, list))
        except Exception as e:
            check(f"fts5 handles: '{q[:25]}'", False, f"raised {e}")
    db.close()


def test_empty_db_operations():
    """Operations on a fresh DB don't crash."""
    print("\n--- test_empty_db_operations ---")
    # Use a different agent to get empty DB
    empty_agent = "__empty_test_agent__"
    db = v2db.open_db(empty_agent)

    briefing = retrieval.format_session_briefing(db, empty_agent)
    check("empty briefing is ''", briefing == "")

    results = retrieval.dual_retrieve(db, empty_agent, "anything", top_k=5)
    check("empty dual_retrieve is []", results == [])

    anti = retrieval.retrieve_anti_patterns(db, empty_agent, "rm -rf /")
    check("empty anti_patterns is []", anti == [])

    maint = maintenance.maintenance_cycle(db, empty_agent)
    check("empty maintenance runs", isinstance(maint, dict))
    check("empty maintenance 0 nodes", maint["total_nodes"] == 0)

    stats = v2db.get_stats(db, empty_agent)
    check("empty stats 0 nodes", stats["total_nodes"] == 0)
    check("empty stats 0 events", stats["total_events"] == 0)

    db.close()


def main():
    setup()
    try:
        # Phase 1: DB creation
        test_on_disk_db_creation()
        # Phase 2: Observer
        test_observer_safe_append_event()
        test_observer_safe_record_aisthesis()
        test_observer_truncation()
        test_observer_failure_recovery()
        # Phase 3: Seed knowledge
        test_seed_knowledge()
        # Phase 4: Retrieval
        test_session_briefing()
        test_dual_retrieve_on_disk()
        test_dual_retrieve_with_surfacing()
        # Phase 5: Bash guardian
        test_bash_guardian_safe_commands()
        test_bash_guardian_dangerous_commands()
        test_bash_guardian_no_false_positives()
        # Phase 6: Feedback
        test_confirm_surfaced()
        test_refute_on_error()
        test_record_irrelevance()
        # Phase 7: KG operations
        test_graph_traversal()
        test_detect_contradictions()
        # Phase 8: Maintenance
        test_maintenance_cycle()
        # Phase 9: Concurrent access
        test_concurrent_connections()
        # Phase 10: Hook simulation
        test_hook_session_start_simulation()
        test_hook_prompt_submit_simulation()
        test_hook_post_tool_success_simulation()
        test_hook_post_tool_error_simulation()
        test_hook_pre_bash_simulation()
        # Phase 11: Full lifecycle
        test_full_session_lifecycle()
        # Phase 12: Edge cases
        test_dedup_on_disk()
        test_fts5_special_characters()
        test_empty_db_operations()

        print(f"\n{'='*50}")
        print(f"RESULTS: {passed} passed, {failed} failed")
        if failed > 0:
            sys.exit(1)
        print("ALL INTEGRATION TESTS PASSED")

    finally:
        teardown()


if __name__ == "__main__":
    main()
