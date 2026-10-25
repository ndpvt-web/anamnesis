#!/usr/bin/env python3
"""Tinyfish integration for Anamnesis: Didaxis (external teaching knowledge).

Three integration points:
  1. search_for_error()   -- real-time error lookup via Search API
  2. enrich_reflection()  -- web context for reflection hypothesis generation
  3. seed_knowledge()     -- bootstrap KG with domain-specific knowledge

Aristotelian justification:
  Anamnesis currently has a closed epistemic loop (empeiria only).
  Tinyfish opens Didaxis -- knowledge from authoritative external sources.
  This complements experiential learning with curated web knowledge.

API: Tinyfish Search API (GET https://api.search.tinyfish.ai)
     Tinyfish Fetch API (POST https://api.fetch.tinyfish.ai)
Cost: ~1 step per search, ~1 step per fetch. 500 free steps for new accounts.
"""

import json
import os
import sys
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anamnesis_config
from v2 import db as v2db
from v2 import llm
from v2.retrieval import classify_domain, classify_error_family

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TINYFISH_API_KEY = os.environ.get("TINYFISH_API_KEY", "")
SEARCH_URL = "https://api.search.tinyfish.ai"
FETCH_URL = "https://api.fetch.tinyfish.ai"

# Seeding configuration
SEED_QUERIES_PER_DOMAIN = 3
MAX_FETCH_PER_SEED = 5

# Domain-specific seed queries (common pitfalls for each domain genus)
DOMAIN_SEED_QUERIES = {
    "web_backend": [
        "Flask common errors and fixes site:stackoverflow.com",
        "FastAPI common mistakes beginners",
        "Python web server port conflict CORS errors",
    ],
    "data_processing": [
        "pandas CSV UnicodeDecodeError encoding fix",
        "Python data pipeline common errors",
        "pandas dataframe common mistakes and solutions",
    ],
    "database": [
        "SQLite database locked error fix Python",
        "SQL migration common pitfalls",
        "Python database connection pool errors",
    ],
    "cli_tools": [
        "Python argparse common mistakes",
        "subprocess common errors Python",
        "CLI tool error handling best practices Python",
    ],
    "config_deploy": [
        "YAML parsing errors common fixes Python",
        "Docker build common errors solutions",
        "environment variable configuration mistakes",
    ],
    "testing": [
        "pytest common errors and fixtures",
        "Python unittest mock pitfalls",
        "test isolation common mistakes",
    ],
    "frontend": [
        "React common errors and solutions",
        "JavaScript build errors webpack vite",
        "CSS layout common mistakes fixes",
    ],
    "devops": [
        "git common mistakes and recovery",
        "CI/CD pipeline common errors",
        "GitHub Actions common pitfalls fixes",
    ],
}


# ---------------------------------------------------------------------------
# Core API calls
# ---------------------------------------------------------------------------

def _tinyfish_search(query: str, num_results: int = 5) -> list:
    """Search via Tinyfish Search API. Returns list of {title, url, snippet}."""
    if not TINYFISH_API_KEY:
        return []
    try:
        encoded_q = urllib.request.quote(query)
        url = f"{SEARCH_URL}?q={encoded_q}&num={num_results}&language=en"
        req = urllib.request.Request(url, headers={
            "X-API-Key": TINYFISH_API_KEY,
            "User-Agent": "anamnesis/3.1",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("snippet", "")} for r in results[:num_results]]
    except Exception as e:
        print(f"[anamnesis] tinyfish search error: {e}", file=sys.stderr)
        return []


def _tinyfish_fetch(urls: list, fmt: str = "markdown") -> list:
    """Fetch page content via Tinyfish Fetch API. Returns list of {url, text}."""
    if not TINYFISH_API_KEY or not urls:
        return []
    try:
        payload = json.dumps({"urls": urls[:5], "format": fmt}).encode("utf-8")
        req = urllib.request.Request(FETCH_URL, data=payload, headers={
            "X-API-Key": TINYFISH_API_KEY,
            "Content-Type": "application/json",
            "User-Agent": "anamnesis/3.1",
        }, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        return [{"url": r.get("url", ""), "text": r.get("text", "")[:3000]}
                for r in results if r.get("text")]
    except Exception as e:
        print(f"[anamnesis] tinyfish fetch error: {e}", file=sys.stderr)
        return []


def _call_llm_parse(system: str, prompt: str, timeout: float = 15) -> dict:
    """Parse web content into structured knowledge with the fast model."""
    result = llm.chat_json(system, prompt, purpose="fast",
                           timeout=timeout, max_tokens=2048)
    return result if isinstance(result, (dict, list)) else {}


# ---------------------------------------------------------------------------
# Point 1: Error-Triggered Search
# ---------------------------------------------------------------------------

def search_for_error(error_text: str, tool_name: str = "",
                     command: str = "") -> str:
    """Search web for error solutions when local knowledge is empty.
    Returns formatted context string for hook output, or empty string."""
    if not TINYFISH_API_KEY:
        return ""
    # Build search query from error signature
    # Extract the most specific error line
    lines = error_text.strip().split("\n")
    error_line = ""
    for line in reversed(lines):
        if re.search(r'Error|Exception|FAILED|fatal|denied', line, re.IGNORECASE):
            error_line = line.strip()[:120]
            break
    if not error_line:
        error_line = lines[-1].strip()[:120] if lines else error_text[:120]

    query = f"{error_line} fix solution"
    if tool_name == "Bash" and command:
        # Add command context for better results
        cmd_short = command.split()[0] if command else ""
        if cmd_short:
            query = f"{cmd_short} {error_line} fix"

    results = _tinyfish_search(query, num_results=3)
    if not results:
        return ""

    # Format as ephemeral context (not stored in KG)
    lines_out = ["## Anamnesis: Web Search Results (Tinyfish)"]
    for r in results[:3]:
        title = r["title"][:80]
        snippet = r["snippet"][:200]
        lines_out.append(f"- **{title}**: {snippet}")
    return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Point 2: Reflection Enrichment
# ---------------------------------------------------------------------------

def enrich_with_web_context(error_signatures: list,
                             error_family: str = "") -> str:
    """Search web for error signatures found during narration.
    Returns web context string to inject into HYPOTHESIZE prompt.
    Called during reflection pipeline, between diagnose and hypothesize."""
    if not TINYFISH_API_KEY:
        return ""
    if not error_signatures:
        return ""

    web_results = []
    seen_urls = set()
    for sig in error_signatures[:3]:
        error_text = sig.get("error_text", "")[:100]
        if not error_text:
            continue
        query = f"{error_text} root cause fix"
        results = _tinyfish_search(query, num_results=2)
        for r in results:
            if r["url"] not in seen_urls:
                web_results.append(r)
                seen_urls.add(r["url"])

    if not web_results:
        return ""

    lines = ["## Web Context (via Tinyfish search):"]
    for r in web_results[:5]:
        lines.append(f"- {r['title'][:80]}: {r['snippet'][:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Point 3: Knowledge Seeding
# ---------------------------------------------------------------------------

SEED_SYSTEM_PROMPT = """You are a knowledge extraction system for the Anamnesis memory architecture.
Given web content about common programming errors and solutions, extract structured knowledge entries.

Each entry MUST follow the WHEN/THEN/BECAUSE format:
- WHEN: specific triggering condition (environment, tool, version)
- THEN: observable symptom (error message, behavior)
- BECAUSE: root cause explanation
- FIX: concrete remediation step

Return JSON array of objects:
[{
  "content": "WHEN ... THEN ... BECAUSE ... FIX: ...",
  "category": "anti_pattern" | "best_practice" | "error_fix" | "environment",
  "severity": "low" | "medium" | "high",
  "error_signature": "the specific error text if any",
  "error_family": "E1_environment" | "E2_encoding" | "E3_dependency" | "E4_configuration" | "E5_silent_failure" | "E6_cascading" | "E7_platform" | "E8_state" | "E9_api_contract" | "E10_build" | "",
  "confidence": 0.60
}]

Rules:
- Extract 2-5 entries per page. Quality over quantity.
- Each entry must contain specific technical details (error names, file paths, port numbers, versions).
- Reject generic advice ("always test your code"). Only specific, actionable knowledge.
- confidence should be 0.55-0.65 for web-sourced knowledge (lower than empirical).
- Preserve exact error messages and commands when found."""


def seed_knowledge(db, agent_id: str, domains: list = None,
                   session_id: str = "seed") -> dict:
    """Bootstrap KG with domain-specific knowledge from the web.
    Uses Tinyfish Search + Fetch + AI Gateway to extract structured knowledge.

    Args:
        db: open sqlite3 connection
        agent_id: agent to seed knowledge for
        domains: list of domain names to seed (default: all 8)
        session_id: session ID for provenance tracking

    Returns: {"nodes_created": int, "domains_seeded": list, "errors": list}
    """
    if not TINYFISH_API_KEY:
        return {"nodes_created": 0, "domains_seeded": [],
                "errors": ["TINYFISH_API_KEY not set"]}
    if not anamnesis_config.llm_config("fast")["configured"]:
        return {"nodes_created": 0, "domains_seeded": [],
                "errors": ["no LLM API key configured (ANAMNESIS_LLM_API_KEY)"]}

    if domains is None:
        domains = list(DOMAIN_SEED_QUERIES.keys())

    total_nodes = 0
    seeded_domains = []
    errors = []

    for domain in domains:
        queries = DOMAIN_SEED_QUERIES.get(domain, [])
        if not queries:
            continue

        domain_nodes = 0
        print(f"[anamnesis] seeding domain: {domain} ({len(queries)} queries)",
              file=sys.stderr)

        for query in queries[:SEED_QUERIES_PER_DOMAIN]:
            try:
                # Step 1: Search for relevant pages
                search_results = _tinyfish_search(query, num_results=3)
                if not search_results:
                    continue

                # Step 2: Fetch top results
                urls = [r["url"] for r in search_results[:2]]
                pages = _tinyfish_fetch(urls)
                if not pages:
                    continue

                # Step 3: Extract knowledge via LLM
                for page in pages:
                    page_text = page["text"][:2500]
                    prompt = (
                        f"Extract structured knowledge entries from this web page "
                        f"about {domain} programming issues.\n\n"
                        f"Source URL: {page['url']}\n\n"
                        f"Content:\n{page_text}\n\n"
                        f"Return a JSON array of knowledge entries."
                    )
                    result = _call_llm_parse(SEED_SYSTEM_PROMPT, prompt)
                    entries = result if isinstance(result, list) else result.get("entries", [])

                    # Step 4: Store each entry as a techne node
                    for entry in entries[:5]:
                        content = entry.get("content", "").strip()
                        if not content or len(content) < 30:
                            continue
                        # Quality check: must have WHEN or BECAUSE
                        if "WHEN" not in content and "BECAUSE" not in content:
                            continue

                        error_family = entry.get("error_family", "")
                        node_id = v2db.add_node(
                            db,
                            agent_id=agent_id,
                            content=content,
                            node_type=v2db.NODE_FACT,
                            layer=v2db.LAYER_TECHNE,
                            summary=content[:200],
                            applies_when=content.split("THEN")[0][:200]
                                if "THEN" in content else "",
                            category=entry.get("category", "error_fix"),
                            severity=entry.get("severity", "medium"),
                            confidence=min(float(entry.get("confidence", 0.60)), 0.65),
                            source_session=session_id,
                            origin_agent_id="tinyfish_seed",
                            maturity="candidate",
                            error_family=error_family,
                            domain=domain,
                        )
                        if node_id > 0:
                            domain_nodes += 1
                            print(f"[anamnesis] seeded node {node_id} "
                                  f"({domain}): {content[:50]}",
                                  file=sys.stderr)

            except Exception as e:
                errors.append(f"{domain}/{query[:30]}: {e}")
                print(f"[anamnesis] seed error: {e}", file=sys.stderr)

        if domain_nodes > 0:
            seeded_domains.append(domain)
            total_nodes += domain_nodes
            print(f"[anamnesis] seeded {domain_nodes} nodes for {domain}",
                  file=sys.stderr)

    return {
        "nodes_created": total_nodes,
        "domains_seeded": seeded_domains,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI entry point for standalone seeding
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anamnesis Knowledge Seeder (Tinyfish)")
    parser.add_argument("agent_id", help="Agent ID to seed knowledge for")
    parser.add_argument("--domains", nargs="*", help="Specific domains to seed")
    parser.add_argument("--list-domains", action="store_true",
                        help="List available domain categories")
    args = parser.parse_args()

    if args.list_domains:
        for d, queries in DOMAIN_SEED_QUERIES.items():
            print(f"  {d}: {len(queries)} seed queries")
        sys.exit(0)

    db = v2db.open_db(args.agent_id)
    before = v2db.count_nodes_by_layer(db, args.agent_id)
    print(f"KG before: {sum(before.values())} nodes ({before})")

    result = seed_knowledge(db, args.agent_id, domains=args.domains)

    after = v2db.count_nodes_by_layer(db, args.agent_id)
    print(f"\nKG after: {sum(after.values())} nodes ({after})")
    print(f"Seeded: {result['nodes_created']} nodes across {result['domains_seeded']}")
    if result["errors"]:
        print(f"Errors: {len(result['errors'])}")
        for e in result["errors"][:5]:
            print(f"  - {e}")
    db.close()
