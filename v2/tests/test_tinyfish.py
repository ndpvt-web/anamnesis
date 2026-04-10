#!/usr/bin/env python3
"""Tests for Tinyfish integration (Didaxis: external teaching knowledge).
Tests the 3 integration points:
  1. Error-triggered search (search_for_error)
  2. Reflection enrichment (enrich_with_web_context)
  3. Knowledge seeding (seed_knowledge)

Note: Tests that call Tinyfish API are marked with _live suffix
and require TINYFISH_API_KEY to be set. They are skipped if no key."""

import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from v2 import db as v2db
from v2 import tinyfish
from v2.tinyfish import (
    search_for_error, enrich_with_web_context, seed_knowledge,
    _tinyfish_search, _tinyfish_fetch, _call_llm_parse,
    DOMAIN_SEED_QUERIES, classify_domain,
)
from v2.tests.fixtures import create_test_db, TEST_AGENT

HAS_KEY = bool(os.environ.get("TINYFISH_API_KEY"))


# ---------------------------------------------------------------------------
# Unit tests (no API calls -- mocked)
# ---------------------------------------------------------------------------

def test_search_for_error_no_key():
    """Without API key, returns empty string."""
    with patch.object(tinyfish, 'TINYFISH_API_KEY', ''):
        result = search_for_error("ModuleNotFoundError: flask")
    assert result == "", f"Expected empty, got: {result}"


def test_search_for_error_with_mock():
    """With mocked Tinyfish response, returns formatted context."""
    mock_results = [
        {"title": "Fix ModuleNotFoundError", "url": "https://example.com/1",
         "snippet": "Install flask with pip install flask"},
        {"title": "Python import errors", "url": "https://example.com/2",
         "snippet": "Check your virtualenv is activated"},
    ]
    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'):
        with patch.object(tinyfish, '_tinyfish_search', return_value=mock_results):
            result = search_for_error("ModuleNotFoundError: flask",
                                       tool_name="Bash",
                                       command="python app.py")
    assert "Tinyfish" in result, f"Missing Tinyfish header in: {result}"
    assert "ModuleNotFoundError" in result or "flask" in result or "Install" in result


def test_search_for_error_empty_results():
    """When Tinyfish returns no results, returns empty string."""
    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'):
        with patch.object(tinyfish, '_tinyfish_search', return_value=[]):
            result = search_for_error("some random error text")
    assert result == ""


def test_enrich_no_key():
    """Without API key, returns empty string."""
    with patch.object(tinyfish, 'TINYFISH_API_KEY', ''):
        result = enrich_with_web_context([{"error_text": "test"}])
    assert result == ""


def test_enrich_empty_signatures():
    """With empty error signatures, returns empty string."""
    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'):
        result = enrich_with_web_context([])
    assert result == ""


def test_enrich_with_mock():
    """With mocked results, returns web context string."""
    mock_results = [
        {"title": "UnicodeDecodeError fix", "url": "https://so.com/1",
         "snippet": "Use encoding='utf-8-sig' to handle BOM"},
    ]
    sigs = [{"error_text": "UnicodeDecodeError: 'utf-8' codec can't decode"}]
    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'):
        with patch.object(tinyfish, '_tinyfish_search', return_value=mock_results):
            result = enrich_with_web_context(sigs, error_family="E2_encoding")
    assert "Web Context" in result
    assert "UnicodeDecodeError" in result or "BOM" in result


def test_seed_no_key():
    """Without API key, returns error."""
    db = create_test_db()
    with patch.object(tinyfish, 'TINYFISH_API_KEY', ''):
        result = seed_knowledge(db, TEST_AGENT, domains=["web_backend"])
    assert result["nodes_created"] == 0
    assert "TINYFISH_API_KEY" in result["errors"][0]
    db.close()


def test_seed_with_mock():
    """With mocked API responses, creates nodes in DB."""
    db = create_test_db()
    mock_search = [
        {"title": "Flask errors", "url": "https://example.com/flask",
         "snippet": "Common Flask pitfalls"},
    ]
    mock_fetch = [
        {"url": "https://example.com/flask",
         "text": "Flask port 5000 conflicts with macOS AirDrop..."},
    ]
    mock_llm = [
        {
            "content": "WHEN running Flask on macOS Monterey+ THEN ConnectionRefusedError on port 5000 BECAUSE AirDrop Control Center occupies port 5000. FIX: Use port 5001 or set --port 8080",
            "category": "error_fix",
            "severity": "high",
            "error_signature": "ConnectionRefusedError",
            "error_family": "E7_platform",
            "confidence": 0.60,
        }
    ]

    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'), \
         patch.object(tinyfish, 'AI_GATEWAY_KEY', 'test-key'), \
         patch.object(tinyfish, '_tinyfish_search', return_value=mock_search), \
         patch.object(tinyfish, '_tinyfish_fetch', return_value=mock_fetch), \
         patch.object(tinyfish, '_call_llm_parse', return_value=mock_llm):
        result = seed_knowledge(db, TEST_AGENT, domains=["web_backend"])

    assert result["nodes_created"] >= 1, f"Expected >=1 nodes, got {result}"
    assert "web_backend" in result["domains_seeded"]

    # Verify node was stored correctly
    nodes = v2db.get_nodes_by_layer(db, TEST_AGENT, "techne")
    flask_nodes = [n for n in nodes if "Flask" in n["content"]]
    assert len(flask_nodes) >= 1, f"No Flask node found in {[n['content'][:50] for n in nodes]}"

    node = flask_nodes[0]
    assert node["domain"] == "web_backend"
    assert node["error_family"] == "E7_platform"
    assert "tinyfish_seed" in node["origin_agent_id"]
    assert node["confidence"] <= 0.65  # Capped for web-sourced knowledge
    db.close()


def test_seed_rejects_generic():
    """Seeder rejects generic advice without WHEN/BECAUSE."""
    db = create_test_db()
    mock_search = [{"title": "Tips", "url": "https://x.com", "snippet": "tips"}]
    mock_fetch = [{"url": "https://x.com", "text": "Always test your code"}]
    mock_llm = [
        {"content": "Always test your code before deploying",
         "category": "best_practice", "severity": "low", "confidence": 0.60}
    ]

    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'), \
         patch.object(tinyfish, 'AI_GATEWAY_KEY', 'test-key'), \
         patch.object(tinyfish, '_tinyfish_search', return_value=mock_search), \
         patch.object(tinyfish, '_tinyfish_fetch', return_value=mock_fetch), \
         patch.object(tinyfish, '_call_llm_parse', return_value=mock_llm):
        result = seed_knowledge(db, TEST_AGENT, domains=["testing"])

    # Generic content without WHEN/BECAUSE should be rejected
    assert result["nodes_created"] == 0
    db.close()


def test_domain_seed_queries_coverage():
    """All 8 domains have seed queries defined."""
    assert len(DOMAIN_SEED_QUERIES) == 8
    for domain, queries in DOMAIN_SEED_QUERIES.items():
        assert len(queries) >= 2, f"Domain {domain} needs >=2 queries"
        assert domain in ("web_backend", "data_processing", "database",
                          "cli_tools", "config_deploy", "testing",
                          "frontend", "devops"), f"Unknown domain: {domain}"


def test_confidence_capped():
    """Web-sourced nodes must have confidence <= 0.65."""
    db = create_test_db()
    mock_search = [{"title": "T", "url": "https://x.com", "snippet": "s"}]
    mock_fetch = [{"url": "https://x.com", "text": "content"}]
    mock_llm = [
        {"content": "WHEN using sqlite3 THEN database locked BECAUSE WAL mode not enabled. FIX: PRAGMA journal_mode=WAL",
         "category": "error_fix", "severity": "high",
         "confidence": 0.95,  # High confidence from LLM -- should be capped
         "error_family": "E8_state"}
    ]

    with patch.object(tinyfish, 'TINYFISH_API_KEY', 'test-key'), \
         patch.object(tinyfish, 'AI_GATEWAY_KEY', 'test-key'), \
         patch.object(tinyfish, '_tinyfish_search', return_value=mock_search), \
         patch.object(tinyfish, '_tinyfish_fetch', return_value=mock_fetch), \
         patch.object(tinyfish, '_call_llm_parse', return_value=mock_llm):
        result = seed_knowledge(db, TEST_AGENT, domains=["database"])

    if result["nodes_created"] > 0:
        nodes = v2db.get_nodes_by_layer(db, TEST_AGENT, "techne")
        for n in nodes:
            assert n["confidence"] <= 0.65, \
                f"Web-sourced node confidence {n['confidence']} exceeds 0.65 cap"
    db.close()


# ---------------------------------------------------------------------------
# Live integration tests (require TINYFISH_API_KEY)
# ---------------------------------------------------------------------------

def test_search_live():
    """Live Tinyfish search -- requires API key."""
    if not HAS_KEY:
        print("  SKIP: test_search_live (no TINYFISH_API_KEY)")
        return
    results = _tinyfish_search("Python Flask ModuleNotFoundError fix", num_results=3)
    assert len(results) > 0, "Live search returned no results"
    assert results[0].get("title"), "Missing title in live result"
    assert results[0].get("url"), "Missing URL in live result"
    print(f"  LIVE: search returned {len(results)} results")


def test_fetch_live():
    """Live Tinyfish fetch -- requires API key."""
    if not HAS_KEY:
        print("  SKIP: test_fetch_live (no TINYFISH_API_KEY)")
        return
    results = _tinyfish_fetch(["https://docs.python.org/3/tutorial/errors.html"])
    assert len(results) > 0, "Live fetch returned no results"
    assert len(results[0].get("text", "")) > 100, "Fetched text too short"
    print(f"  LIVE: fetched {len(results[0]['text'])} chars")


def test_error_search_live():
    """Live error-triggered search -- requires API key."""
    if not HAS_KEY:
        print("  SKIP: test_error_search_live (no TINYFISH_API_KEY)")
        return
    result = search_for_error(
        "ModuleNotFoundError: No module named 'flask'\nExit code: 1",
        tool_name="Bash",
        command="python app.py"
    )
    assert "Tinyfish" in result, f"Missing header in: {result[:100]}"
    print(f"  LIVE: error search returned {len(result)} chars")
