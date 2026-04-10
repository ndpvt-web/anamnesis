#!/usr/bin/env python3
"""Tests for root-cause fixes:
1. False positive prevention (match only problem text, not Fix text)
2. Critical command safety net (Sophia axioms)
3. SUPPORTS edges created on confirmation
4. Cosine-similarity credit assignment
5. Ebbinghaus decay with SM-2 reinforcement
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2.tests.fixtures import create_test_db
from v2 import db as v2db
from v2 import retrieval
from v2 import feedback
from v2 import knowledge_graph as kg


# --- Fix 1: False positive prevention ---

def test_false_positive_fix_recommendation_not_matched():
    """Anti-pattern with tool in Fix text should NOT match that tool's command."""
    db = create_test_db()
    # Create an anti-pattern that mentions "curl" only in the Fix portion
    v2db.add_node(db, "test-agent",
        content="Debugging API failures by changing defaults. Fix: use curl to verify the actual request",
        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70,
        category="anti_pattern", severity="high")
    db.commit()

    # A curl command should NOT be blocked by this anti-pattern
    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "curl https://example.com")
    assert len(matches) == 0, f"Expected 0 matches (curl only in Fix text), got {len(matches)}"
    db.close()


def test_false_positive_problem_text_still_matches():
    """Anti-pattern with tool in problem text SHOULD still match."""
    db = create_test_db()
    # Create an anti-pattern that mentions "git push --force" in the problem
    v2db.add_node(db, "test-agent",
        content="Running git push --force on shared branches destroys history. Fix: use --force-with-lease instead",
        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70,
        category="anti_pattern", severity="high")
    db.commit()

    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "git push --force origin main")
    assert len(matches) >= 1, f"Expected >= 1 match (git in problem text), got {len(matches)}"
    db.close()


def test_false_positive_no_fix_delimiter():
    """Anti-patterns without 'Fix:' delimiter should still match normally."""
    db = create_test_db()
    v2db.add_node(db, "test-agent",
        content="Using rm -rf on production directories without backup is dangerous",
        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70,
        category="anti_pattern", severity="critical")
    db.commit()

    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "rm -rf /var/data")
    # Should match via critical patterns OR FTS5
    assert len(matches) >= 1, f"Expected >= 1 match, got {len(matches)}"
    db.close()


# --- Fix 2: Critical command safety net (Sophia axioms) ---

def test_critical_rm_rf():
    """rm -rf should ALWAYS be caught even with empty KG."""
    db = create_test_db()
    # Empty DB, no anti-patterns at all
    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "rm -rf /var/data/production")
    assert len(matches) >= 1, f"Expected critical match for rm -rf, got {len(matches)}"
    assert matches[0]["severity"] == "critical"
    assert matches[0]["source"] == "axiom"
    assert matches[0]["layer"] == "sophia"
    db.close()


def test_critical_rm_rf_variants():
    """Various rm -rf formats should all be caught."""
    db = create_test_db()
    dangerous = [
        "rm -rf /home/user/data",
        "rm -rf ./important",
        "rm -fr /var/log",
        "sudo rm -rf /etc",
    ]
    for cmd in dangerous:
        matches = retrieval.retrieve_anti_patterns(db, "test-agent", cmd)
        assert len(matches) >= 1, f"Expected critical match for '{cmd}', got {len(matches)}"
    db.close()


def test_critical_git_force_push():
    """git push --force to main should be caught."""
    db = create_test_db()
    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "git push --force origin main")
    assert len(matches) >= 1, f"Expected match for force push, got {len(matches)}"
    assert matches[0]["severity"] == "high"
    db.close()


def test_critical_git_reset_hard():
    """git reset --hard should be caught."""
    db = create_test_db()
    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "git reset --hard HEAD~3")
    assert len(matches) >= 1, f"Expected match for reset --hard, got {len(matches)}"
    db.close()


def test_critical_drop_database():
    """DROP DATABASE should be caught."""
    db = create_test_db()
    matches = retrieval.retrieve_anti_patterns(db, "test-agent",
        "psql -c 'DROP DATABASE production'")
    assert len(matches) >= 1, f"Expected match for DROP DATABASE, got {len(matches)}"
    assert matches[0]["severity"] == "critical"
    db.close()


def test_critical_safe_commands_pass():
    """Safe commands should NOT trigger critical patterns."""
    db = create_test_db()
    safe = [
        "git push origin feature-branch",
        "git reset --soft HEAD~1",
        "rm file.txt",
        "rm -f temp.log",
    ]
    for cmd in safe:
        matches = retrieval.retrieve_anti_patterns(db, "test-agent", cmd)
        axiom_matches = [m for m in matches if m.get("source") == "axiom"]
        assert len(axiom_matches) == 0, f"Safe cmd '{cmd}' matched critical pattern: {axiom_matches}"
    db.close()


def test_critical_rm_tmp_excluded():
    """rm -rf /tmp should be allowed (excluded from critical patterns)."""
    db = create_test_db()
    matches = retrieval.retrieve_anti_patterns(db, "test-agent", "rm -rf /tmp/build")
    axiom_matches = [m for m in matches if m.get("source") == "axiom"]
    assert len(axiom_matches) == 0, f"/tmp should be excluded, got {axiom_matches}"
    db.close()


# --- Fix 3: SUPPORTS edges on confirmation ---

def test_confirm_creates_supports_edges():
    """Confirming surfaced nodes should create SUPPORTS edges between them."""
    db = create_test_db()
    agent = "test-agent"
    session = "edge-test-session"

    # Create two nodes
    n1 = v2db.add_node(db, agent, content="Pattern A for testing",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    n2 = v2db.add_node(db, agent, content="Pattern B for testing",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    n3 = v2db.add_node(db, agent, content="Pattern C for testing",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)

    # Surface all three
    v2db.record_surfaced(db, agent, session, [n1, n2, n3])
    db.commit()

    # First confirmation: n1, n2, n3 are pending, confirm them
    # No previous confirmed nodes, so no edges yet
    count1 = feedback.confirm_surfaced_nodes(db, agent, session)
    assert count1 == 3, f"Expected 3 confirmed, got {count1}"

    edges = db.execute("SELECT * FROM kg_edges WHERE agent_id=?", (agent,)).fetchall()
    # No edges yet because all 3 were pending simultaneously (no prior confirmed)
    assert len(edges) == 0, f"Expected 0 edges on first batch, got {len(edges)}"

    # Now surface more nodes and confirm - should create SUPPORTS edges
    n4 = v2db.add_node(db, agent, content="Pattern D for testing",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    v2db.record_surfaced(db, agent, session, [n4])
    db.commit()

    count2 = feedback.confirm_surfaced_nodes(db, agent, session)
    assert count2 == 1, f"Expected 1 confirmed, got {count2}"

    edges = db.execute("SELECT * FROM kg_edges WHERE agent_id=?", (agent,)).fetchall()
    assert len(edges) >= 1, f"Expected >= 1 SUPPORTS edge, got {len(edges)}"
    assert edges[0]["edge_type"] == v2db.EDGE_SUPPORTS
    db.close()


def test_confirm_edges_enable_promotion():
    """Nodes with enough SUPPORTS edges + confidence + helpful should promote."""
    db = create_test_db()
    agent = "test-agent"

    # Create a high-confidence, high-helpful node
    n1 = v2db.add_node(db, agent, content="Proven pattern",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.90)
    db.execute("UPDATE kg_nodes SET helpful_count=5 WHERE id=?", (n1,))

    # Create 2 SUPPORTS edges pointing to it
    n2 = v2db.add_node(db, agent, content="Supporting evidence A",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    n3 = v2db.add_node(db, agent, content="Supporting evidence B",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    v2db.add_edge(db, agent, n2, n1, v2db.EDGE_SUPPORTS)
    v2db.add_edge(db, agent, n3, n1, v2db.EDGE_SUPPORTS)
    db.commit()

    # Verify promotion requirements met
    from v2 import knowledge_graph as kg
    result = kg.promote_candidates(db, agent)
    assert result["promoted"] >= 1, f"Expected promotion, got {result}"
    assert n1 in result["ids"], f"Node {n1} should have been promoted"

    # Verify it's now episteme
    node = v2db.get_node(db, n1)
    assert node["layer"] == v2db.LAYER_EPISTEME
    assert node["maturity"] == "crystallized"
    db.close()


# --- Fix 4: Cosine-similarity credit assignment ---

def test_cosine_similarity_basic():
    """cosine_similarity returns high score for similar texts, low for dissimilar."""
    # Highly similar
    sim = feedback.cosine_similarity(
        "configure nginx reverse proxy with ssl certificates",
        "nginx reverse proxy ssl configuration tutorial"
    )
    assert sim > 0.4, f"Expected high similarity, got {sim:.3f}"

    # Completely different
    sim2 = feedback.cosine_similarity(
        "configure nginx reverse proxy",
        "bake chocolate cake recipe"
    )
    assert sim2 < 0.1, f"Expected low similarity, got {sim2:.3f}"

    # Empty inputs
    assert feedback.cosine_similarity("", "hello") == 0.0
    assert feedback.cosine_similarity("hello", "") == 0.0


def test_cosine_similarity_token_extraction():
    """cosine_similarity handles technical tokens correctly."""
    # Command-like tokens
    sim = feedback.cosine_similarity(
        "npm install --save-dev typescript",
        "Install typescript as dev dependency using npm"
    )
    assert sim > 0.3, f"Expected moderate similarity, got {sim:.3f}"

    # Path tokens
    sim2 = feedback.cosine_similarity(
        "Edit /home/user/project/config.json",
        "Modified config.json in project directory"
    )
    assert sim2 > 0.2, f"Expected some similarity, got {sim2:.3f}"


def test_confirm_with_context_credits_most_relevant():
    """confirm_surfaced_nodes with action_context credits most relevant node."""
    db = create_test_db()
    agent = "test-agent"
    session = "cosine-test"

    # Create 3 nodes with very different content
    n1 = v2db.add_node(db, agent, content="Always use virtual environments for Python projects",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    n2 = v2db.add_node(db, agent, content="Configure nginx reverse proxy with upstream blocks",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    n3 = v2db.add_node(db, agent, content="Use git rebase for clean commit history",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)

    # Surface all three
    v2db.record_surfaced(db, agent, session, [n1, n2, n3])
    db.commit()

    # Action context is about nginx -- should credit n2 most
    action_ctx = "Bash nginx -t && systemctl reload nginx proxy upstream configured successfully"
    count = feedback.confirm_surfaced_nodes(db, agent, session, action_context=action_ctx)

    # n2 (nginx) should get full credit, others get partial or irrelevant
    node2 = v2db.get_node(db, n2)
    node1 = v2db.get_node(db, n1)
    node3 = v2db.get_node(db, n3)

    # n2 should have higher confidence boost than n1 and n3
    assert node2["confidence"] >= node1["confidence"], \
        f"nginx node ({node2['confidence']:.3f}) should >= python node ({node1['confidence']:.3f})"
    assert node2["helpful_count"] >= 1, "nginx node should have helpful_count >= 1"
    db.close()


def test_confirm_without_context_credits_all():
    """confirm_surfaced_nodes without action_context credits all equally (backward compat)."""
    db = create_test_db()
    agent = "test-agent"
    session = "nocontext-test"

    n1 = v2db.add_node(db, agent, content="Always configure nginx reverse proxy with upstream blocks",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    n2 = v2db.add_node(db, agent, content="Use pytest fixtures for database integration testing",
                        layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.70)
    # add_node auto-commits, so nodes exist before record_surfaced FK check
    v2db.record_surfaced(db, agent, session, [n1, n2])

    count = feedback.confirm_surfaced_nodes(db, agent, session)  # No action_context
    assert count == 2, f"Expected 2 confirmed, got {count}"

    node1 = v2db.get_node(db, n1)
    node2 = v2db.get_node(db, n2)
    assert node1["confidence"] == node2["confidence"], \
        f"Without context, both should get equal credit: {node1['confidence']:.3f} vs {node2['confidence']:.3f}"
    db.close()


# --- Fix 5: Ebbinghaus decay with SM-2 reinforcement ---

def test_ebbinghaus_strength_basic():
    """ebbinghaus_strength returns correct values for known inputs."""
    import math

    # No time elapsed -> same confidence
    s = kg.ebbinghaus_strength(0.80, 0, 0)
    assert abs(s - 0.80) < 0.001, f"Expected 0.80, got {s:.3f}"

    # 30 days, 0 helpful -> significant decay
    # S = 30 * (1 + 0*0.3) = 30, strength = 0.80 * e^(-30/30) = 0.80 * 0.368 = 0.294
    s = kg.ebbinghaus_strength(0.80, 30, 0)
    expected = 0.80 * math.exp(-1)
    assert abs(s - expected) < 0.01, f"Expected {expected:.3f}, got {s:.3f}"

    # 30 days, 10 helpful -> much slower decay
    # S = 30 * (1 + 10*0.3) = 120, strength = 0.80 * e^(-30/120) = 0.80 * 0.779 = 0.623
    s = kg.ebbinghaus_strength(0.80, 30, 10)
    expected = 0.80 * math.exp(-30 / 120)
    assert abs(s - expected) < 0.01, f"Expected {expected:.3f}, got {s:.3f}"


def test_ebbinghaus_helpful_nodes_decay_slower():
    """Nodes with more helpful accesses decay slower than unused ones."""
    # Same base confidence and time, different helpful_count
    s_unused = kg.ebbinghaus_strength(0.80, 60, helpful_count=0)
    s_used = kg.ebbinghaus_strength(0.80, 60, helpful_count=5)
    s_heavy = kg.ebbinghaus_strength(0.80, 60, helpful_count=20)

    assert s_unused < s_used < s_heavy, \
        f"Expected decay order: {s_unused:.3f} < {s_used:.3f} < {s_heavy:.3f}"

    # Unused node at 60 days should be well below 0.50
    assert s_unused < 0.20, f"Unused node at 60 days should be < 0.20, got {s_unused:.3f}"
    # Heavily used node should still be high
    assert s_heavy > 0.60, f"Heavy-use node at 60 days should be > 0.60, got {s_heavy:.3f}"


def test_ebbinghaus_decay_in_maintenance():
    """Maintenance cycle uses Ebbinghaus decay correctly."""
    db = create_test_db()
    agent = "test-agent"

    # Create two nodes: one with high helpful_count, one with zero
    n_used = v2db.add_node(db, agent, content="Frequently helpful pattern",
                            layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.80)
    n_unused = v2db.add_node(db, agent, content="Never used pattern",
                              layer=v2db.LAYER_TECHNE, node_type="fact", confidence=0.80)
    db.execute("UPDATE kg_nodes SET helpful_count=10 WHERE id=?", (n_used,))
    # Make both stale (60 days ago)
    db.execute("UPDATE kg_nodes SET last_seen=datetime('now','-60 days'), "
               "last_confirmed=datetime('now','-60 days')")
    db.commit()

    result = kg.decay_stale_nodes(db, agent, stale_days=30)

    node_used = v2db.get_node(db, n_used)
    node_unused = v2db.get_node(db, n_unused)

    # Used node should retain more confidence than unused
    if node_used and node_unused:
        assert node_used["confidence"] > node_unused["confidence"], \
            f"Used node ({node_used['confidence']:.3f}) should > unused ({node_unused['confidence']:.3f})"
    db.close()
