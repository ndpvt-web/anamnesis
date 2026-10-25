#!/usr/bin/env python3
"""Dual-process retrieval engine with domain-aware routing.
System 1: FTS5 search (<30ms) -- fast keyword matching.
System 2: Graph expansion (<20ms) -- follow edges from top FTS5 results.
Domain routing: Error family classification boosts relevant nodes.
Import: from v2 import db as v2db, knowledge_graph as kg"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2 import db as v2db
from v2 import knowledge_graph as kg

# ---------------------------------------------------------------------------
# Error Family Classifier (lightweight, no LLM needed)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Domain Genus Classifier (v3: prevents cross-domain retrieval pollution)
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "web_backend": {"flask", "django", "fastapi", "express", "endpoint", "route",
                    "api", "http", "server", "cors", "middleware", "wsgi", "gunicorn"},
    "data_processing": {"csv", "pandas", "dataframe", "json", "parse", "transform",
                        "pipeline", "etl", "encoding", "utf", "numpy", "data"},
    "database": {"sql", "sqlite", "postgres", "mysql", "query", "migration",
                 "schema", "table", "index", "orm", "alembic"},
    "cli_tools": {"argparse", "click", "cli", "terminal", "command", "flag",
                  "subprocess", "argv", "getopt"},
    "config_deploy": {"yaml", "toml", "docker", "env", "config", "deploy",
                      "nginx", "systemd", "compose", "dockerfile"},
    "testing": {"pytest", "unittest", "mock", "assert", "fixture", "coverage",
                "test", "spec", "describe"},
    "frontend": {"react", "vue", "html", "css", "javascript", "dom", "component",
                 "render", "webpack", "vite", "typescript"},
    "devops": {"git", "ci", "cd", "github", "actions", "pipeline", "build",
               "release", "workflow", "jenkins"},
}


def classify_domain(text: str) -> str:
    """Classify text into a domain genus. Returns domain name or 'general'.
    Uses token intersection scoring. Requires 2+ keyword matches."""
    if not text:
        return "general"
    words = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower()))
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in words)
        if score >= 2:
            scores[domain] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


ERROR_FAMILY_SIGNATURES = {
    "E1_environment": {"PATH", "env", "environ", "variable", "export", "source",
                       "bash_profile", "zshenv", "profile", "shell", "port",
                       "EADDRINUSE", "address already in use", "bind"},
    "E2_encoding": {"unicode", "encode", "decode", "utf", "ascii", "bytes",
                    "UnicodeDecodeError", "UnicodeEncodeError", "charset", "BOM",
                    "codec", "latin", "cp1252", "surrogate"},
    "E3_dependencies": {"import", "module", "package", "install", "require",
                        "ModuleNotFoundError", "ImportError", "pip", "npm",
                        "version", "dependency", "missing"},
    "E4_configuration": {"config", "yaml", "json", "toml", "settings", "env_file",
                         "dotenv", "parse", "invalid", "malformed", "indent", "tab"},
    "E5_silent_failure": {"exit_code", "returncode", "silent", "no output",
                          "nothing happened", "empty", "no error"},
    "E6_cascading": {"caused by", "traceback", "because of", "downstream",
                     "chain", "propagat"},
    "E7_platform": {"macos", "linux", "windows", "darwin", "arm", "x86",
                    "platform", "os.name", "sys.platform", "airdrop"},
    "E8_state": {"stale", "cache", "leftover", "previous", "already", "exists",
                 "__pycache__", "lock", "pid", "zombie"},
    "E9_api_contract": {"400", "401", "403", "404", "422", "500", "response",
                        "schema", "payload", "endpoint", "header", "content-type"},
    "E10_build_system": {"Makefile", "cmake", "gradle", "cargo", "webpack",
                         "build", "compile", "linker", "gcc", "make"},
}


def classify_error_family(text: str) -> str:
    """Classify text into an error family (E1-E10) or 'none'.
    Uses token intersection scoring. No LLM needed."""
    if not text:
        return "none"
    text_lower = text.lower()
    words = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text))
    words_lower = {w.lower() for w in words}
    scores = {}
    for family, signatures in ERROR_FAMILY_SIGNATURES.items():
        score = 0
        for sig in signatures:
            sig_lower = sig.lower()
            # For short signatures (<=4 chars), require word-boundary match
            if len(sig) <= 4:
                if sig_lower in words_lower or sig in words:
                    score += 1
            else:
                if sig_lower in text_lower or sig in words:
                    score += 1
        if score > 0:
            scores[family] = score
    if not scores:
        return "none"
    return max(scores, key=scores.get)


def _adaptive_threshold(node_count: int) -> float:
    """Aristotle's Doctrine of the Mean: returns a RATIO of max score.
    At 10 nodes: keep results >= 5% of best. At 75: >= 20%. At 200+: >= 50%.
    Returns a fraction [0.0, 1.0] to multiply against the top score."""
    base = 0.03       # minimum ratio (very permissive)
    per_node = 0.003  # tighten 0.3% per node
    cap = 0.50        # never filter more than half the top score
    return min(base + per_node * node_count, cap)


def dual_retrieve(db, agent_id: str, query: str, top_k: int = 5,
                  include_graph: bool = True,
                  error_family: str = None) -> list:
    """Dual-process retrieval combining FTS5 + graph expansion + domain routing.
    Step 1 (System 1): FTS5 search -> top_k * 2 candidates (domain-filtered first)
    Step 2 (System 2): For top 3 FTS5 results, expand 1 hop via graph
    Step 3: Domain boost + error_family boost in re-scoring
    Step 4: Adaptive threshold filtering based on graph size
    Step 5: Layer-aware caps (episteme uncapped, techne capped)
    Returns: list of dicts with keys:
        id, content, layer, confidence, category, severity,
        helpful_count, maturity, source (str: "fts5"|"graph"),
        graph_context (str: related node summaries, only if from graph expansion)
    """
    if not query or not query.strip():
        return []
    # Auto-detect error family if not provided
    if error_family is None:
        error_family = classify_error_family(query)
    # Auto-detect domain for genus-bounded retrieval
    query_domain = classify_domain(query)

    # Step 1: Domain-filtered FTS5 search first, then fallback to unfiltered
    fts_results = []
    if query_domain != "general":
        fts_results = v2db.fts5_search(
            db, query, agent_id, top_k=top_k * 2, domain_filter=query_domain
        )
    # Fallback: if domain search returned <2 results, do unfiltered
    if len(fts_results) < 2:
        fts_results = v2db.fts5_search(db, query, agent_id, top_k=top_k * 2)
    # Fallback: if FTS5 found nothing but we have an error family, search by family
    if not fts_results and error_family and error_family != "none":
        fts_results = v2db.fts5_search(
            db, error_family.replace("_", " "), agent_id, top_k=top_k * 2
        )
    print(f"[anamnesis] dual_retrieve: FTS5 returned {len(fts_results)} results "
          f"(domain={query_domain})", file=sys.stderr)

    # Build a dict of node_id -> result dict for dedup tracking
    seen_ids = {}
    for node in fts_results:
        nid = node["id"]
        if nid not in seen_ids:
            result = {
                "id": nid,
                "content": node.get("content", ""),
                "layer": node.get("layer", ""),
                "confidence": node.get("confidence", 0.0),
                "category": node.get("category", "general"),
                "severity": node.get("severity", "medium"),
                "helpful_count": node.get("helpful_count", 0),
                "maturity": node.get("maturity", "candidate"),
                "error_family": node.get("error_family", ""),
                "domain": node.get("domain", ""),
                "source": "fts5",
                "_score": -float(node.get("fts_rank", 0)) if node.get("fts_rank") else 0.0,
            }
            seen_ids[nid] = result

    # Step 2: Graph expansion for top 3 FTS5 results
    if include_graph and fts_results:
        top3 = fts_results[:3]
        for fts_node in top3:
            seed_id = fts_node["id"]
            try:
                related = kg.find_related_nodes(db, agent_id, seed_id, max_depth=1, max_nodes=20)
            except Exception as e:
                print(f"[anamnesis] graph expand error for node {seed_id}: {e}", file=sys.stderr)
                related = []

            # Build graph_context summary from this seed's related nodes
            seed_context_parts = []
            for rel_node in related:
                rel_id = rel_node["id"]
                if rel_id == seed_id:
                    continue
                content_snippet = rel_node.get("content", "")[:80]
                seed_context_parts.append(content_snippet)

                if rel_id not in seen_ids:
                    # New node from graph expansion
                    # Score: base on confidence, slightly lower than fts5
                    # (helpful boost applied once in _rescore, not here)
                    conf = rel_node.get("confidence", 0.0)
                    helpful = rel_node.get("helpful_count", 0)
                    base_score = conf * 0.8

                    # Gather graph_context: summaries of what led to this node
                    result = {
                        "id": rel_id,
                        "content": rel_node.get("content", ""),
                        "layer": rel_node.get("layer", ""),
                        "confidence": conf,
                        "category": rel_node.get("category", "general"),
                        "severity": rel_node.get("severity", "medium"),
                        "helpful_count": helpful,
                        "maturity": rel_node.get("maturity", "candidate"),
                        "error_family": rel_node.get("error_family", ""),
                        "domain": rel_node.get("domain", ""),
                        "source": "graph",
                        "graph_context": f"Related to: {fts_node.get('content', '')[:60]}",
                        "_score": base_score,
                    }
                    seen_ids[rel_id] = result

            # Update graph_context on the seed node itself if it has related nodes
            if seed_context_parts and seed_id in seen_ids:
                seen_ids[seed_id]["graph_context"] = "; ".join(seed_context_parts[:3])

    # Step 3: Re-score and sort
    all_results = list(seen_ids.values())

    def _rescore(r):
        score = r.get("_score", 0.0)
        # Layer boost
        layer = r.get("layer", "")
        if layer == v2db.LAYER_EPISTEME:
            score *= 2.0
        elif r.get("maturity") == "crystallized":
            score *= 1.5
        # Helpful boost
        helpful = r.get("helpful_count", 0)
        if helpful > 0:
            score *= (1 + min(helpful / 10, 0.5))
        # Confidence boost
        score += r.get("confidence", 0.0) * 0.1
        # Error family boost: same domain = strong relevance signal
        node_family = r.get("error_family", "")
        if error_family and error_family != "none" and node_family == error_family:
            score *= 1.5
        # Domain genus boost: same domain = relevant knowledge
        node_domain = r.get("domain", "")
        if query_domain != "general" and node_domain == query_domain:
            score *= 1.3
        return score

    for r in all_results:
        r["_final_score"] = _rescore(r)
    all_results.sort(key=lambda r: r["_final_score"], reverse=True)

    # Step 4: Adaptive threshold filtering based on graph size
    # Threshold is a ratio of the best score -- scales with any FTS5 magnitude
    layer_counts = v2db.count_nodes_by_layer(db, agent_id)
    total_active = sum(layer_counts.values())
    threshold_ratio = _adaptive_threshold(total_active)
    max_score = max((r["_final_score"] for r in all_results), default=0.0)
    abs_threshold = max_score * threshold_ratio

    filtered = []
    for r in all_results:
        score = r["_final_score"]
        layer = r.get("layer", "")
        # Episteme/Sophia always pass threshold (proven knowledge)
        if layer in (v2db.LAYER_EPISTEME, v2db.LAYER_SOPHIA):
            filtered.append(r)
        elif score >= abs_threshold:
            filtered.append(r)

    # Step 5: Layer-aware caps (prevent techne flood)
    episteme_count = sum(1 for r in filtered
                         if r.get("layer") in (v2db.LAYER_EPISTEME, v2db.LAYER_SOPHIA))
    techne_cap = max(2, top_k - episteme_count)

    final = []
    techne_seen = 0
    for r in filtered:
        layer = r.get("layer", "")
        if layer in (v2db.LAYER_EPISTEME, v2db.LAYER_SOPHIA):
            r.pop("_score", None)
            r.pop("_final_score", None)
            final.append(r)
        elif techne_seen < techne_cap:
            r.pop("_score", None)
            r.pop("_final_score", None)
            final.append(r)
            techne_seen += 1
        if len(final) >= top_k:
            break

    print(f"[anamnesis] dual_retrieve: returning {len(final)} results "
          f"(top_k={top_k}, threshold={abs_threshold:.3f} ({threshold_ratio:.0%}), "
          f"active_nodes={total_active})",
          file=sys.stderr)
    return final


def retrieve_for_error(db, agent_id: str, error_text: str, top_k: int = 3) -> list:
    """Specialized retrieval for error context.
    Searches across all categories (not just anti_pattern).
    Returns same format as dual_retrieve."""
    print(f"[anamnesis] retrieve_for_error: searching for error context", file=sys.stderr)
    fts_results = v2db.fts5_search(db, error_text, agent_id, top_k=top_k)

    results = []
    for node in fts_results:
        result = {
            "id": node["id"],
            "content": node.get("content", ""),
            "layer": node.get("layer", ""),
            "confidence": node.get("confidence", 0.0),
            "category": node.get("category", "general"),
            "severity": node.get("severity", "medium"),
            "helpful_count": node.get("helpful_count", 0),
            "maturity": node.get("maturity", "candidate"),
            "source": "fts5",
        }
        results.append(result)

    return results


def _check_critical_patterns(command: str) -> list:
    """Sophia-level axioms: hardcoded critical command patterns that are always
    dangerous regardless of knowledge base content. Returns synthetic anti-pattern
    nodes for any matches. These are the safety net when the KG has no coverage."""
    import re as _re
    cmd = command.strip().lower()
    # Normalize whitespace for pattern matching
    cmd_norm = " ".join(cmd.split())

    # Helper: check if rm target is only /tmp (safe)
    def _rm_targets_tmp_only(cmd_str):
        """Return True if all path args to rm are under /tmp."""
        parts = cmd_str.split()
        paths = [p for p in parts[1:] if not p.startswith("-")]
        return bool(paths) and all(p.startswith("/tmp") or p.startswith("./tmp") for p in paths)

    CRITICAL_PATTERNS = [
        (_re.compile(r'(?<!\w)rm\s+.*-[a-z]*r[a-z]*f|(?<!\w)rm\s+.*-[a-z]*f[a-z]*r'),
         "rm -rf without confirmation is dangerous in production environments",
         "critical"),
        (_re.compile(r'(?<!\w)rm\s+-[a-z]*r\s+/(?!tmp\b)'),
         "Recursive rm on system paths risks catastrophic data loss",
         "critical"),
        (_re.compile(r'>\s*/dev/sd|mkfs\.|fdisk\s'),
         "Direct disk/filesystem operations risk data destruction",
         "critical"),
        (_re.compile(r'(?<!\w)dd\s+.*of=/dev/'),
         "dd writing to device files can destroy data irreversibly",
         "critical"),
        (_re.compile(r':\(\)\s*\{\s*:\|:\s*&\s*\}\s*;'),
         "Fork bomb will crash the system",
         "critical"),
        (_re.compile(r'chmod\s+.*-[rR]\s+.*777\s+/(?!tmp\b)'),
         "Recursive chmod 777 on system paths is a severe security risk",
         "high"),
        (_re.compile(r'(?<!\w)git\s+push\s+.*--force\s+.*(?:main|master)\b'),
         "Force-pushing to main/master can destroy shared commit history",
         "high"),
        (_re.compile(r'(?<!\w)git\s+reset\s+--hard'),
         "git reset --hard discards uncommitted changes permanently",
         "high"),
        (_re.compile(r'drop\s+(?:database|table)\s', _re.IGNORECASE),
         "DROP DATABASE/TABLE is irreversible data destruction",
         "critical"),
        (_re.compile(r'truncate\s+table\s', _re.IGNORECASE),
         "TRUNCATE TABLE deletes all data irreversibly",
         "critical"),
    ]

    matches = []
    for pattern, description, severity in CRITICAL_PATTERNS:
        if pattern.search(cmd_norm):
            # Exclude rm commands targeting only /tmp (safe temp directory)
            if description.startswith("rm ") and _rm_targets_tmp_only(cmd_norm):
                continue
            matches.append({
                "id": -1,
                "content": description,
                "layer": "sophia",
                "confidence": 1.0,
                "category": "anti_pattern",
                "severity": severity,
                "helpful_count": 0,
                "maturity": "crystallized",
                "source": "axiom",
            })

    return matches


def retrieve_anti_patterns(db, agent_id: str, command: str) -> list:
    """Find anti-patterns matching a bash command.
    Two-stage: (1) hardcoded critical patterns (Sophia axioms, always checked),
    then (2) FTS5 + token verification against the knowledge base.
    Returns list of matching anti-pattern nodes."""
    cmd = command.strip()
    if not cmd:
        return []

    import re as _re
    cmd_lower = cmd.lower()

    # Read-only / diagnostic commands are never blocked
    read_prefixes = (
        "cat ", "head ", "tail ", "less ", "more ", "wc ", "file ",
        "stat ", "ls ", "find ", "grep ", "rg ", "diff ", "echo ",
        "python3 -c", "python3 -m json", "node -e",
        "mkdir ", "touch ", "which ", "type ", "env ", "printenv ",
        "curl ", "wget ", "ping ", "dig ", "nslookup ", "traceroute ",
    )
    if cmd_lower.lstrip().startswith(read_prefixes):
        return []
    # Multi-word safe commands (read-only + constructive package installs)
    read_commands = (
        "git status", "git log", "git diff", "git show", "git branch",
        "git remote", "git stash list", "git tag",
        "npm list", "npm ls", "npm info", "npm view", "npm outdated",
        "npm install", "npm i ", "npm ci", "npm add",
        "pip list", "pip show", "pip freeze", "pip install",
        "docker ps", "docker images", "docker logs",
        "brew install", "brew info", "brew list",
        "cargo install", "cargo build", "cargo test",
        "apt install", "apt-get install",
    )
    cmd_stripped = " ".join(cmd_lower.split())
    if any(cmd_stripped.startswith(rc) for rc in read_commands):
        return []

    # Stage 1: Sophia axioms (hardcoded critical patterns)
    critical_matches = _check_critical_patterns(cmd)
    if critical_matches:
        return critical_matches[:5]

    # Stage 2: Knowledge-based matching (FTS5 + token verification)

    # Subcommands that are also common English words -- too ambiguous to match
    # against anti-pattern text without causing false positives.
    _SUBCOMMAND_STOPWORDS = {
        "status", "list", "show", "log", "help", "info", "view", "run",
        "test", "start", "stop", "set", "get", "add", "new", "init",
        "check", "search", "update", "version", "config", "clean",
    }

    # Extract technical tokens from command, categorized by match strategy:
    #   substring_tokens: long enough to safely do substring match (paths, long flags, dotted names)
    #   boundary_tokens: short flags that need word-boundary regex to avoid false positives
    #   command_tokens: the command name + compound flags (e.g. "rm", "-rf", "git")
    substring_tokens = set()
    boundary_tokens = set()
    command_tokens = set()

    words = cmd_lower.split()
    for i, word in enumerate(words):
        word = word.rstrip(".,;:!?'\"()-")
        if not word:
            continue
        if "/" in word and len(word) > 5:
            substring_tokens.add(word)       # paths: /var/data/foo
        elif word.startswith("--") and len(word) > 3:
            substring_tokens.add(word)       # long flags: --force, --hard
        elif word.startswith("-") and len(word) == 2:
            boundary_tokens.add(word)        # single short flags: -p, -r
        elif word.startswith("-") and len(word) > 2 and not word.startswith("--"):
            command_tokens.add(word)          # compound flags: -rf, -xzf
        elif "." in word and len(word) > 4:
            substring_tokens.add(word)       # dotted names: module.foo
        elif i == 0 and _re.match(r'^[a-z][a-z0-9_-]*$', word) and len(word) <= 20:
            command_tokens.add(word)          # command name: rm, git, npm
        elif i > 0 and not word.startswith("-") and _re.match(r'^[a-z][a-z0-9_-]*$', word) and len(word) <= 15:
            # subcommands: git push, npm install -- skip common English words
            if words[0].rstrip(".,;:!?'\"()-") in ("git", "npm", "docker", "kubectl", "pip", "cargo", "brew"):
                if word not in _SUBCOMMAND_STOPWORDS:
                    command_tokens.add(word)

    all_tokens = substring_tokens | boundary_tokens | command_tokens
    if not all_tokens:
        return []

    # FTS5 to find candidate anti-patterns
    results = v2db.fts5_search(
        db, cmd, agent_id,
        top_k=10,
        category_filter="anti_pattern"
    )

    # Second pass: verify at least one token from the command appears in
    # the PROBLEM portion of the anti-pattern (before "Fix:"), not in the
    # recommendation text. This prevents false positives where the fix
    # recommends a tool (e.g. "Fix: use curl to verify") and the command
    # using that tool gets blocked.
    matches = []
    fix_split = _re.compile(r'\.\s*fix:\s*', _re.IGNORECASE)
    for node in results:
        content_lower = node.get("content", "").lower()
        # Extract only the problem portion (before "Fix:")
        parts = fix_split.split(content_lower, maxsplit=1)
        problem_text = parts[0] if parts else content_lower
        matched = False

        # Substring match for long tokens (paths, long flags, dotted names)
        for token in substring_tokens:
            if token in problem_text:
                matched = True
                break

        # Word-boundary match for single short flags (-p, -r)
        if not matched:
            for token in boundary_tokens:
                pattern = r'(?<!\w)' + _re.escape(token) + r'(?!\w)'
                if _re.search(pattern, problem_text):
                    matched = True
                    break

        # Word-boundary match for command names and compound flags (rm, -rf, git, push)
        if not matched:
            for token in command_tokens:
                pattern = r'(?<!\w)' + _re.escape(token) + r'(?!\w)'
                if _re.search(pattern, problem_text):
                    matched = True
                    break

        if matched:
            matches.append(dict(node))

    return matches[:5]


def format_query_based_briefing(db, agent_id: str, results: list) -> str:
    """Generate briefing from query results, not static layer dump.
    Used when session start has domain context (cwd, initial prompt)."""
    if not results:
        return format_session_briefing(db, agent_id)  # Fallback

    lines = [f"## Anamnesis: Relevant Knowledge ({agent_id})"]

    # Always include sophia axioms
    axioms = v2db.get_axioms(db, agent_id)[:3]
    if axioms:
        lines.append("### SOPHIA (Axioms)")
        for ax in axioms:
            lines.append(f"- {ax.get('content', '')}")

    # Query-matched nodes, sorted by layer
    episteme = [r for r in results if r.get("layer") == "episteme"]
    techne = [r for r in results if r.get("layer") == "techne"]

    if episteme:
        lines.append("### EPISTEME (Validated - Highly Relevant)")
        for node in episteme:
            conf = node.get("confidence", 0.0)
            lines.append(f"- [{conf:.2f}] {node['content']} [PROVEN]")

    if techne:
        lines.append("### TECHNE (Patterns - Relevant)")
        for node in techne:
            conf = node.get("confidence", 0.0)
            lines.append(f"- [{conf:.2f}] {node['content']}")

    return "\n".join(lines)


def format_session_briefing(db, agent_id: str) -> str:
    """Generate session start briefing string.
    Shows nodes grouped by layer (Sophia first, then Episteme, then Techne).
    Format:
      ## Anamnesis: Agent Knowledge ({agent_id})
      ### SOPHIA (Axioms)
      - [content]
      ### EPISTEME (Validated Knowledge)
      - [confidence] [category] content [PROVEN]
      ### TECHNE (Patterns)
      - [confidence] [category] content
    Max: 5 sophia, 5 episteme, 5 techne.
    Returns empty string if no nodes."""
    lines = []

    # Sophia axioms
    axioms = v2db.get_axioms(db, agent_id)[:5]

    # Episteme nodes
    episteme_nodes = v2db.get_nodes_by_layer(db, agent_id, v2db.LAYER_EPISTEME, limit=5)

    # Techne nodes
    techne_nodes = v2db.get_nodes_by_layer(db, agent_id, v2db.LAYER_TECHNE, limit=5)

    if not axioms and not episteme_nodes and not techne_nodes:
        return ""

    lines.append(f"## Anamnesis: Agent Knowledge ({agent_id})")

    if axioms:
        lines.append("### SOPHIA (Axioms)")
        for ax in axioms:
            lines.append(f"- {ax.get('content', '')}")

    if episteme_nodes:
        lines.append("### EPISTEME (Validated Knowledge)")
        for node in episteme_nodes:
            conf = node.get("confidence", 0.0)
            cat = node.get("category", "general")
            content = node.get("content", "")
            lines.append(f"- [{conf:.2f}] [{cat}] {content} [PROVEN]")

    if techne_nodes:
        lines.append("### TECHNE (Patterns)")
        for node in techne_nodes:
            conf = node.get("confidence", 0.0)
            cat = node.get("category", "general")
            content = node.get("content", "")
            lines.append(f"- [{conf:.2f}] [{cat}] {content}")

    return "\n".join(lines)
