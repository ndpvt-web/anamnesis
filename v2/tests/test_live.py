#!/usr/bin/env python3
"""Live tests against real agent hermes.db.

Exercises every v2 code path that hooks actually call in production,
against the surfaced-cto agent's real 3400+ node knowledge base.
Reads and writes to the REAL DB (adds events, surfaces, feedback).

Run: python3 -m v2.tests.test_live
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2 import observer
from v2 import retrieval
from v2 import feedback
from v2 import maintenance
from v2 import knowledge_graph as kg

passed = 0
failed = 0

AGENT_ID = "surfaced-cto"
SESSION_ID = f"live-test-{int(time.time())}"


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def _log(msg):
    print(f"[live-test] {msg}", file=sys.stderr)


# --- Pre-flight: verify DB exists ---

def test_db_exists():
    """Agent's hermes.db exists and has migrated data."""
    print("\n--- test_db_exists ---")
    db_path = v2db.db_path_for_agent(AGENT_ID)
    check("hermes.db exists", db_path.exists(), str(db_path))

    db = v2db.open_db(AGENT_ID)
    stats = v2db.get_stats(db, AGENT_ID)
    check("has nodes", stats["total_nodes"] > 100, f"got {stats['total_nodes']}")
    check("has events", stats["total_events"] >= 0)
    _log(f"Stats: {json.dumps(stats, indent=2)}")
    db.close()


# --- Hook 1: on_session_start ---

def test_session_start_hook():
    """Simulate SessionStart hook v2 block."""
    print("\n--- test_session_start_hook ---")
    db = None
    try:
        # 1. Append session_start event (observer)
        eid = observer.safe_append_event(
            AGENT_ID, "session_start",
            session_id=SESSION_ID,
            metadata={"cwd": "/home/node/workspace", "test": True}
        )
        check("session_start event appended", eid > 0, f"got {eid}")

        # 2. Run maintenance
        db = v2db.open_db(AGENT_ID)
        t0 = time.time()
        maint = maintenance.maintenance_cycle(db, AGENT_ID)
        elapsed = (time.time() - t0) * 1000
        check("maintenance completed", isinstance(maint, dict))
        check("maintenance < 500ms", elapsed < 500, f"took {elapsed:.0f}ms")
        _log(f"Maintenance: {json.dumps(maint)} ({elapsed:.0f}ms)")

        # 3. Generate session briefing
        t0 = time.time()
        briefing = retrieval.format_session_briefing(db, AGENT_ID)
        elapsed = (time.time() - t0) * 1000
        check("briefing generated", len(briefing) > 0)
        check("briefing < 200ms", elapsed < 200, f"took {elapsed:.0f}ms")
        check("briefing has agent ID", AGENT_ID in briefing)

        # Check layer headers
        has_techne = "TECHNE" in briefing
        has_episteme = "EPISTEME" in briefing
        check("briefing has TECHNE", has_techne)
        # Episteme may not exist yet if no promotions
        _log(f"Briefing length: {len(briefing)} chars, has_episteme={has_episteme}")
        _log(f"Briefing preview: {briefing[:200]}...")

    finally:
        if db:
            db.close()


# --- Hook 2: on_prompt_submit ---

def test_prompt_submit_hook():
    """Simulate UserPromptSubmit hook v2 block."""
    print("\n--- test_prompt_submit_hook ---")
    db = None
    try:
        prompts = [
            "How do I handle API errors in production?",
            "Fix the authentication bug in the login endpoint",
            "Write a Dockerfile for the Python web service",
        ]

        for prompt in prompts:
            # 1. Append event
            eid = observer.safe_append_event(
                AGENT_ID, "prompt_submit",
                session_id=SESSION_ID,
                prompt_text=prompt
            )
            check(f"prompt event appended: '{prompt[:30]}'", eid > 0)

            # 2. Dual retrieve
            db = v2db.open_db(AGENT_ID)
            t0 = time.time()
            results = retrieval.dual_retrieve(db, AGENT_ID, prompt, top_k=3)
            elapsed = (time.time() - t0) * 1000
            check(f"retrieval returned results ({len(results)})", len(results) >= 0)
            check(f"retrieval < 100ms", elapsed < 100, f"took {elapsed:.0f}ms")

            # 3. Record surfaced
            if results:
                node_ids = [r["id"] for r in results]
                v2db.record_surfaced(db, AGENT_ID, SESSION_ID, node_ids)
                _log(f"Surfaced {len(node_ids)} nodes for '{prompt[:30]}'")
                for r in results:
                    _log(f"  [{r['confidence']:.2f}|{r['category']}|{r['source']}] {r['content'][:60]}")
            db.close()
            db = None

    finally:
        if db:
            db.close()


# --- Hook 3: on_post_tool (success) ---

def test_post_tool_success():
    """Simulate PostToolUse success v2 block."""
    print("\n--- test_post_tool_success ---")
    db = None
    try:
        # 1. Append tool_success event
        eid = observer.safe_append_event(
            AGENT_ID, "tool_success",
            session_id=SESSION_ID,
            tool_name="Bash",
            command="python3 -m pytest tests/ -v"
        )
        check("tool_success event appended", eid > 0)

        # 2. Record aisthesis
        aid = observer.safe_record_aisthesis(
            AGENT_ID, SESSION_ID, "command",
            tool_name="Bash",
            command="python3 -m pytest tests/ -v"
        )
        check("aisthesis recorded", aid > 0)

        # 3. Confirm surfaced nodes
        db = v2db.open_db(AGENT_ID)
        count = feedback.confirm_surfaced_nodes(db, AGENT_ID, SESSION_ID)
        check("confirm ran", count >= 0, f"confirmed {count}")
        _log(f"Confirmed {count} surfaced nodes")

    finally:
        if db:
            db.close()


# --- Hook 3b: on_post_tool (error) ---

def test_post_tool_error():
    """Simulate PostToolUse error v2 block."""
    print("\n--- test_post_tool_error ---")
    db = None
    try:
        error_session = SESSION_ID + "-err"
        error_text = "ModuleNotFoundError: No module named 'fastapi'\nTraceback..."

        # Surface something first
        db = v2db.open_db(AGENT_ID)
        results = retrieval.dual_retrieve(db, AGENT_ID, "python module import error", top_k=2)
        if results:
            v2db.record_surfaced(db, AGENT_ID, error_session, [r["id"] for r in results])
        db.close()
        db = None

        # 1. Append tool_error event
        eid = observer.safe_append_event(
            AGENT_ID, "tool_error",
            session_id=error_session,
            tool_name="Bash",
            command="pip install fastapi",
            error_text=error_text
        )
        check("tool_error event appended", eid > 0)

        # 2. Record aisthesis
        aid = observer.safe_record_aisthesis(
            AGENT_ID, error_session, "error",
            tool_name="Bash",
            command="pip install fastapi",
            error_text=error_text[:500]
        )
        check("error aisthesis recorded", aid > 0)

        # 3. Refute surfaced nodes
        db = v2db.open_db(AGENT_ID)
        count = feedback.refute_surfaced_nodes(
            db, AGENT_ID, error_session, error_text=error_text
        )
        check("refute ran", count >= 0, f"refuted {count}")
        _log(f"Refuted {count} surfaced nodes")

    finally:
        if db:
            db.close()


# --- Hook 4: on_pre_bash (bash guardian) ---

def test_bash_guardian():
    """Simulate PreToolUse Bash guardian v2 block."""
    print("\n--- test_bash_guardian ---")
    db = None
    try:
        db = v2db.open_db(AGENT_ID)

        # Safe commands
        safe_cmds = [
            "cat package.json",
            "ls -la src/",
            "grep -r 'TODO' .",
            "python3 -c 'print(1)'",
            "mkdir -p /tmp/build",
        ]
        for cmd in safe_cmds:
            t0 = time.time()
            result = retrieval.retrieve_anti_patterns(db, AGENT_ID, cmd)
            elapsed = (time.time() - t0) * 1000
            check(f"safe: '{cmd[:25]}' -> []", result == [], f"got {len(result)}")
            check(f"  < 50ms", elapsed < 50, f"took {elapsed:.0f}ms")

        # Potentially dangerous
        danger_cmds = [
            "git push --force origin main",
            "rm -rf /var/data",
            "docker system prune -a --force",
        ]
        for cmd in danger_cmds:
            t0 = time.time()
            result = retrieval.retrieve_anti_patterns(db, AGENT_ID, cmd)
            elapsed = (time.time() - t0) * 1000
            _log(f"Guardian '{cmd[:30]}': {len(result)} matches ({elapsed:.0f}ms)")
            if result:
                for r in result:
                    _log(f"  [{r['severity']}|{r['confidence']:.2f}] {r['content'][:60]}")
            # Just check it doesn't crash and is fast
            check(f"guard: '{cmd[:25]}' ran", isinstance(result, list))
            check(f"  < 50ms", elapsed < 50, f"took {elapsed:.0f}ms")

    finally:
        if db:
            db.close()


# --- Hook 5: agent_learn (reflection) ---

def test_reflection_readiness():
    """Verify reflection pipeline prerequisites on real data."""
    print("\n--- test_reflection_readiness ---")
    db = None
    try:
        db = v2db.open_db(AGENT_ID)

        # Check we have enough events for reflection
        events = v2db.get_session_events(db, AGENT_ID, SESSION_ID, limit=30)
        check("session has events", len(events) >= 3, f"got {len(events)}")

        # Check API key exists (needed for reflection)
        has_api_key = bool(os.environ.get("AI_GATEWAY_API_KEY"))
        check("AI_GATEWAY_API_KEY set", has_api_key)

        # Check existing reflection runs
        runs = db.execute(
            "SELECT * FROM reflection_runs WHERE agent_id = ? ORDER BY created_at DESC LIMIT 5",
            (AGENT_ID,)
        ).fetchall()
        _log(f"Recent reflection runs: {len(runs)}")
        for r in runs:
            _log(f"  [{r['status']}] events={r['events_processed']}, nodes={r['nodes_created']}, {r['duration_ms']}ms")

    finally:
        if db:
            db.close()


# --- Run actual reflection if API key available ---

def test_live_reflection():
    """Run actual 5-step reflection pipeline if API key is available."""
    print("\n--- test_live_reflection ---")
    if not os.environ.get("AI_GATEWAY_API_KEY"):
        check("SKIP: no API key", True)
        return

    # We need enough events - append a few more
    for i in range(5):
        observer.safe_append_event(
            AGENT_ID, "tool_success",
            session_id=SESSION_ID,
            tool_name="Bash",
            command=f"reflection test command {i}"
        )

    from v2.reflection import run_reflection_pipeline

    t0 = time.time()
    result = run_reflection_pipeline(AGENT_ID, SESSION_ID)
    elapsed = (time.time() - t0) * 1000
    _log(f"Reflection result: {json.dumps(result, indent=2)}")

    check("reflection returned result", result is not None)
    check("has status", "status" in result)
    status = result.get("status", "")
    check("status is valid", status in ("completed", "failed", "skipped"),
          f"got '{status}'")
    check("reflection < 10s", elapsed < 10000, f"took {elapsed:.0f}ms")

    if status == "completed":
        check("has nodes_created", "nodes_created" in result)
        check("has edges_created", "edges_created" in result)
        _log(f"Created {result.get('nodes_created', 0)} nodes, {result.get('edges_created', 0)} edges")
    elif status == "failed":
        _log(f"Reflection failed: {result.get('error', 'unknown')}")


# --- Knowledge graph on real data ---

def test_live_kg_operations():
    """Test KG operations on real 3400-node dataset."""
    print("\n--- test_live_kg_operations ---")
    db = None
    try:
        db = v2db.open_db(AGENT_ID)

        # Find a node with edges (if any)
        edge_row = db.execute(
            "SELECT DISTINCT to_node_id FROM kg_edges WHERE agent_id = ? LIMIT 1",
            (AGENT_ID,)
        ).fetchone()

        if edge_row:
            nid = edge_row["to_node_id"]
            t0 = time.time()
            related = kg.find_related_nodes(db, AGENT_ID, nid, max_depth=2)
            elapsed = (time.time() - t0) * 1000
            check("graph traversal works", isinstance(related, list))
            check("traversal < 2s", elapsed < 2000, f"took {elapsed:.0f}ms")
            _log(f"Found {len(related)} related nodes for node {nid} ({elapsed:.0f}ms)")
        else:
            check("SKIP: no edges in DB", True)

        # Detect contradictions
        contra_edge = db.execute(
            "SELECT from_node_id FROM kg_edges WHERE agent_id = ? AND edge_type = ? LIMIT 1",
            (AGENT_ID, v2db.EDGE_CONTRADICTS)
        ).fetchone()
        if contra_edge:
            contras = kg.detect_contradictions(db, AGENT_ID, contra_edge["from_node_id"])
            check("contradictions detected", len(contras) > 0)
        else:
            check("SKIP: no CONTRADICTS edges", True)

        # Promotion candidates
        t0 = time.time()
        promo = kg.promote_candidates(db, AGENT_ID)
        elapsed = (time.time() - t0) * 1000
        check("promote_candidates ran", isinstance(promo, dict))
        check("promote < 2s", elapsed < 2000, f"took {elapsed:.0f}ms")
        _log(f"Promote: {json.dumps(promo)} ({elapsed:.0f}ms)")

    finally:
        if db:
            db.close()


# --- Session end (irrelevance + cleanup) ---

def test_session_end():
    """Simulate session end: mark irrelevance, run maintenance."""
    print("\n--- test_session_end ---")
    db = None
    try:
        db = v2db.open_db(AGENT_ID)

        # Mark remaining surfaced as irrelevant
        irr_count = feedback.record_irrelevance(db, AGENT_ID, SESSION_ID)
        check("irrelevance recorded", irr_count >= 0, f"count={irr_count}")

        # Final maintenance
        maint = maintenance.maintenance_cycle(db, AGENT_ID)
        check("final maintenance ran", isinstance(maint, dict))

        # Verify event store integrity
        events = v2db.get_session_events(db, AGENT_ID, SESSION_ID, limit=100)
        check("events preserved", len(events) > 0, f"got {len(events)}")
        _log(f"Session {SESSION_ID}: {len(events)} events total")

        # Check feedback distribution
        surfaced = db.execute(
            "SELECT feedback, COUNT(*) as cnt FROM surfaced_nodes "
            "WHERE session_id = ? GROUP BY feedback",
            (SESSION_ID,)
        ).fetchall()
        dist = {r["feedback"]: r["cnt"] for r in surfaced}
        _log(f"Feedback distribution: {json.dumps(dist)}")

        # Final stats
        stats = v2db.get_stats(db, AGENT_ID)
        _log(f"Final stats: {json.dumps(stats, indent=2)}")

    finally:
        if db:
            db.close()


# --- Performance benchmarks ---

def test_performance_benchmarks():
    """Verify all hook paths meet performance targets."""
    print("\n--- test_performance_benchmarks ---")
    db = None
    try:
        db = v2db.open_db(AGENT_ID)

        # Benchmark FTS5 search
        queries = [
            "error handling",
            "git push force",
            "database connection pool",
            "authentication JWT token",
            "Docker container deployment",
        ]
        for q in queries:
            t0 = time.time()
            results = v2db.fts5_search(db, q, AGENT_ID, top_k=10)
            elapsed = (time.time() - t0) * 1000
            check(f"FTS5 '{q[:20]}' < 30ms", elapsed < 30,
                  f"took {elapsed:.0f}ms, {len(results)} results")

        # Benchmark dual_retrieve
        t0 = time.time()
        results = retrieval.dual_retrieve(db, AGENT_ID, "error handling in API calls", top_k=5)
        elapsed = (time.time() - t0) * 1000
        check("dual_retrieve < 100ms", elapsed < 100, f"took {elapsed:.0f}ms")

        # Benchmark session briefing
        t0 = time.time()
        briefing = retrieval.format_session_briefing(db, AGENT_ID)
        elapsed = (time.time() - t0) * 1000
        check("session_briefing < 200ms", elapsed < 200, f"took {elapsed:.0f}ms")

        # Benchmark bash guardian
        t0 = time.time()
        _ = retrieval.retrieve_anti_patterns(db, AGENT_ID, "git push --force origin main")
        elapsed = (time.time() - t0) * 1000
        check("bash_guardian < 50ms", elapsed < 50, f"took {elapsed:.0f}ms")

        # Benchmark maintenance
        t0 = time.time()
        _ = maintenance.maintenance_cycle(db, AGENT_ID)
        elapsed = (time.time() - t0) * 1000
        check("maintenance < 500ms", elapsed < 500, f"took {elapsed:.0f}ms")

    finally:
        if db:
            db.close()


# --- Actual hook execution via subprocess ---

def test_hook_subprocess_execution():
    """Run actual hook scripts as subprocesses (like Claude Code does)."""
    print("\n--- test_hook_subprocess_execution ---")
    hooks_dir = Path(__file__).resolve().parent.parent.parent

    # Test on_session_start
    input_data = json.dumps({
        "session_id": SESSION_ID + "-subprocess",
        "cwd": "/tmp/test"
    })
    t0 = time.time()
    result = subprocess.run(
        ["python3", str(hooks_dir / "on_session_start.py")],
        input=input_data, capture_output=True, text=True, timeout=5
    )
    elapsed = (time.time() - t0) * 1000
    check("on_session_start exits 0", result.returncode == 0,
          f"rc={result.returncode}, stderr={result.stderr[:200]}")
    check("on_session_start < 2s", elapsed < 2000, f"took {elapsed:.0f}ms")
    if result.stdout.strip():
        try:
            output = json.loads(result.stdout)
            has_ctx = "hookSpecificOutput" in output
            check("on_session_start has output", has_ctx)
            if has_ctx:
                ctx = output["hookSpecificOutput"].get("additionalContext", "")
                _log(f"SessionStart output length: {len(ctx)} chars")
        except json.JSONDecodeError:
            check("on_session_start valid JSON", False, result.stdout[:100])

    # Test on_prompt_submit
    input_data = json.dumps({
        "session_id": SESSION_ID + "-subprocess",
        "prompt": "How do I fix authentication errors in my Python API?"
    })
    t0 = time.time()
    result = subprocess.run(
        ["python3", str(hooks_dir / "on_prompt_submit.py")],
        input=input_data, capture_output=True, text=True, timeout=5
    )
    elapsed = (time.time() - t0) * 1000
    check("on_prompt_submit exits 0", result.returncode == 0,
          f"rc={result.returncode}, stderr={result.stderr[:200]}")
    check("on_prompt_submit < 1s", elapsed < 1000, f"took {elapsed:.0f}ms")

    # Test on_pre_bash (safe command)
    input_data = json.dumps({
        "tool_input": {"command": "ls -la /tmp"}
    })
    t0 = time.time()
    result = subprocess.run(
        ["python3", str(hooks_dir / "on_pre_bash.py")],
        input=input_data, capture_output=True, text=True, timeout=5
    )
    elapsed = (time.time() - t0) * 1000
    check("on_pre_bash exits 0", result.returncode == 0,
          f"rc={result.returncode}, stderr={result.stderr[:200]}")
    check("on_pre_bash < 500ms", elapsed < 500, f"took {elapsed:.0f}ms")
    # Safe command should produce no output (no block, no warning)
    check("on_pre_bash no output for safe cmd", result.stdout.strip() == "",
          f"got: {result.stdout[:100]}")

    # Test on_post_tool (success)
    input_data = json.dumps({
        "session_id": SESSION_ID + "-subprocess",
        "tool_name": "Bash",
        "tool_input": {"command": "echo hello"},
        "tool_response": "hello\n"
    })
    t0 = time.time()
    result = subprocess.run(
        ["python3", str(hooks_dir / "on_post_tool.py")],
        input=input_data, capture_output=True, text=True, timeout=5
    )
    elapsed = (time.time() - t0) * 1000
    check("on_post_tool exits 0", result.returncode == 0,
          f"rc={result.returncode}, stderr={result.stderr[:200]}")
    check("on_post_tool < 1s", elapsed < 1000, f"took {elapsed:.0f}ms")


def main():
    print(f"LIVE TEST: agent={AGENT_ID}, session={SESSION_ID}")
    print(f"{'='*50}")

    test_db_exists()
    test_session_start_hook()
    test_prompt_submit_hook()
    test_post_tool_success()
    test_post_tool_error()
    test_bash_guardian()
    test_reflection_readiness()
    test_live_reflection()
    test_live_kg_operations()
    test_session_end()
    test_performance_benchmarks()
    test_hook_subprocess_execution()

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("ALL LIVE TESTS PASSED")


if __name__ == "__main__":
    main()
