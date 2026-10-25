#!/usr/bin/env python3
"""Live tests against a real, populated agent knowledge store.

These exercise v2 code paths against an existing on-disk store with real
data. They are SKIPPED unless both are set:
  ANAMNESIS_TEST_LIVE_AGENT   agent id with an existing store
  ANAMNESIS_HOME (optional)   data home containing that agent's store

Every check is a real assertion: pytest's pass/fail counts are accurate.
"""

import os
import sys
import time
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

LIVE_AGENT = os.environ.get("ANAMNESIS_TEST_LIVE_AGENT", "").strip()

pytestmark = pytest.mark.skipif(
    not LIVE_AGENT,
    reason="no live agent configured; set ANAMNESIS_TEST_LIVE_AGENT",
)

from v2 import db as v2db          # noqa: E402
from v2 import observer            # noqa: E402
from v2 import retrieval           # noqa: E402
from v2 import feedback            # noqa: E402
from v2 import maintenance         # noqa: E402
from v2 import knowledge_graph as kg  # noqa: E402

SESSION_ID = f"live-test-{int(time.time())}"


@pytest.fixture
def db():
    conn = v2db.open_db(LIVE_AGENT)
    yield conn
    conn.close()


def test_store_exists():
    db_path = v2db.db_path_for_agent(LIVE_AGENT)
    assert db_path.exists(), f"no knowledge store at {db_path}"


def test_store_has_data(db):
    stats = v2db.get_stats(db, LIVE_AGENT)
    assert stats["total_nodes"] > 0, "live store has no nodes"


def test_observer_append_event(db):
    eid = observer.safe_append_event(
        LIVE_AGENT, "session_start", session_id=SESSION_ID,
        metadata={"test": True})
    assert eid > 0


def test_maintenance_cycle(db):
    result = maintenance.maintenance_cycle(db, LIVE_AGENT)
    assert "total_nodes" in result
    assert result["total_nodes"] >= 0


def test_dual_retrieve(db):
    results = retrieval.dual_retrieve(db, LIVE_AGENT, "error handling", top_k=3)
    assert isinstance(results, list)


def test_retrieve_anti_patterns_safe_command(db):
    matches = retrieval.retrieve_anti_patterns(db, LIVE_AGENT, "git status")
    assert matches == [], f"safe command matched: {matches}"


def test_retrieve_anti_patterns_dangerous(db):
    matches = retrieval.retrieve_anti_patterns(db, LIVE_AGENT,
                                               "rm -rf /var/data/production")
    assert any(m.get("severity") == "critical" for m in matches)


def test_surfacing_and_feedback(db):
    node_ids = [n["id"] for n in
                v2db.get_nodes_by_layer(db, LIVE_AGENT, "techne", limit=1)]
    if not node_ids:
        pytest.skip("no techne nodes in live store")
    v2db.record_surfaced(db, LIVE_AGENT, SESSION_ID, node_ids)
    count = feedback.confirm_surfaced_nodes(
        db, LIVE_AGENT, SESSION_ID, action_context="test action")
    assert count >= 1


def test_refutation_path(db):
    node_ids = [n["id"] for n in
                v2db.get_nodes_by_layer(db, LIVE_AGENT, "techne", limit=1)]
    if not node_ids:
        pytest.skip("no techne nodes in live store")
    v2db.record_surfaced(db, LIVE_AGENT, SESSION_ID + "-r", node_ids)
    count = feedback.refute_surfaced_nodes(db, LIVE_AGENT, SESSION_ID + "-r",
                                           error_text="test error")
    assert count >= 1


def test_irrelevance_recording(db):
    node_ids = [n["id"] for n in
                v2db.get_nodes_by_layer(db, LIVE_AGENT, "techne", limit=1)]
    if not node_ids:
        pytest.skip("no techne nodes in live store")
    v2db.record_surfaced(db, LIVE_AGENT, SESSION_ID + "-i", node_ids)
    count = feedback.record_irrelevance(db, LIVE_AGENT, SESSION_ID + "-i")
    assert count >= 1


def test_graph_traversal(db):
    node_ids = [n["id"] for n in
                v2db.get_nodes_by_layer(db, LIVE_AGENT, "techne", limit=1)]
    if not node_ids:
        pytest.skip("no techne nodes in live store")
    related = kg.find_related_nodes(db, LIVE_AGENT, node_ids[0], max_depth=1)
    assert isinstance(related, list)


def test_session_briefing(db):
    text = retrieval.format_session_briefing(db, LIVE_AGENT)
    assert isinstance(text, str)
    if text:
        assert "Anamnesis" in text
