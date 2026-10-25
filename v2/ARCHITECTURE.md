# Anamnesis Architecture

## Overview

Anamnesis is a production-grade agent memory platform implementing the Aristotelian epistemological hierarchy. It replaces v1's flat `rules` table with a typed knowledge graph, append-only event store, 5-step LLM reflection pipeline, dual-process retrieval, and three-signal feedback loop.

**3,110 lines of Python. Zero external dependencies (stdlib only, except LLM calls).**

## The Five-Layer Memory Stack

```
Layer 5  SOPHIA      Human-approved axioms. Read-only. Written once, rarely.
Layer 4  EPISTEME    Validated KG nodes (conf >= 0.85, 2+ SUPPORTS, no CONTRADICTS)
Layer 3  TECHNE      Proven patterns from reflection pipeline (conf 0.50-0.84)
Layer 2  EMPEIRIA    Append-only event log. Never updated. Source of truth.
Layer 1  AISTHESIS   Working memory. Session-scoped. Ephemeral.
```

Each layer has increasing epistemic status. Knowledge flows upward through reflection and feedback; stale knowledge flows downward through decay and demotion.

## Database Schema

**File:** `<data_home>/agents/<agent_id>/anamnesis.db` (SQLite, WAL mode), where `<data_home>` is `$ANAMNESIS_HOME` or `~/.anamnesis`

| Table | Purpose | Layer |
|---|---|---|
| `empeiria` | Append-only event log | L2 |
| `kg_nodes` | Knowledge graph nodes | L3-L5 |
| `kg_edges` | Typed relationships (5 edge types) | L3-L4 |
| `kg_fts` | FTS5 virtual table on content+summary+category | L3-L4 |
| `aisthesis` | Session-scoped working memory | L1 |
| `surfaced_nodes` | Tracks which nodes were shown + user feedback | Feedback |
| `reflection_runs` | Audit trail for the 5-step pipeline | Audit |
| `sophia` | Human-approved axioms | L5 |

### Key Schema Features

- **`kg_nodes.harmful_count` + `irrelevant_count`**: Three-signal feedback (confirmation, refutation, irrelevance)
- **`kg_nodes.derived_from`**: JSON list of empeiria IDs (causal provenance)
- **`kg_nodes.maturity`**: `candidate` -> `established` -> `crystallized` (lifecycle tracking)
- **`kg_edges`**: Unique constraint on `(agent_id, from_node_id, to_node_id, edge_type)`
- **FTS5**: Porter unicode61 tokenizer, auto-synced via triggers on INSERT/UPDATE/DELETE

### Node Types

- `fact` - Concrete observations or rules
- `reflection` - Higher-order patterns derived from facts
- `concept` - Abstract generalizations

### Edge Types

- `SUPPORTS` - Evidence reinforcing a node
- `CONTRADICTS` - Evidence opposing a node
- `DERIVED_FROM` - Causal provenance (node came from this source)
- `GENERALIZES` - Abstraction relationship
- `SUPERSEDES` - Replacement (newer replaces older)

### Maturity Lifecycle

```
candidate ──(feedback)──> established ──(promotion)──> crystallized
    ^                         |                             |
    |                         |                             |
    └───────(demotion)────────┘<──────(demotion)────────────┘
```

## Six Subsystems

### 1. Observer (`observer.py`, 70 lines)

Dumb recorder. Captures events to empeiria and aisthesis. Two safe functions:

- `safe_append_event(agent_id, event_type, ...)` - Writes to empeiria. Never throws.
- `safe_record_aisthesis(agent_id, session_id, ...)` - Writes to aisthesis. Never throws.

Both open their own DB connection inside a try/except, so hook callers never crash.

### 2. Event Store (`db.py`, empeiria section)

Append-only. Events are never updated or deleted. Fields: `event_type`, `tool_name`, `command`, `error_text`, `exit_code`, `duration_ms`, `session_id`.

Accessed via:
- `append_event()` - Insert one event
- `get_session_events()` - Fetch events for a session (DESC order, `id DESC` tiebreaker)

### 3. Reflection Pipeline (`reflection.py`, 470 lines)

5-step Sonnet pipeline triggered at session end (Stop hook). Falls back to v1 Haiku on failure.

| Step | Name | Input | Output | LLM? |
|---|---|---|---|---|
| 1 | NARRATE | Last 30 empeiria events | `{summary, key_moments, tools_used}` | Yes |
| 2 | DIAGNOSE | Step 1 + error events | `{errors_diagnosed, patterns, successes}` | Yes |
| 3 | HYPOTHESIZE | Steps 1+2 | `{hypotheses: [...]}` (max 5) | Yes |
| 4 | VALIDATE | Step 3 + top 20 existing nodes | `{accepted, rejected, merges, contradictions}` | Yes |
| 5 | STORE | Steps 3+4 + empeiria IDs | KG writes: nodes + DERIVED_FROM edges | No (pure DB) |

**Budget:** Global 10s timeout with shrinking deadline across sequential LLM calls. Each step receives `timeout=_remaining()` so the pipeline never exceeds the Stop hook budget.

**Merge handling:** Step 4 can identify merges (new hypothesis is similar to existing node). Merges update existing node content in-place + boost confidence. No spurious nodes created.

**Contradiction handling:** Step 4 can flag contradictions. CONTRADICTS edges are created between the new node and the conflicting existing node.

### 4. Knowledge Graph (`knowledge_graph.py`, 325 lines)

Graph operations on `kg_nodes` + `kg_edges`:

- **`find_related_nodes(db, agent_id, node_id, max_depth)`** - BFS traversal following all edge types. Returns neighbors with edge_type and depth annotations.
- **`build_context_subgraph(db, agent_id, node_ids)`** - Given a set of FTS5 results, expand via 1-hop graph traversal to add related nodes. Filters by agent_id and excludes deprecated nodes.
- **`detect_contradictions(db, agent_id, node_id)`** - Find all CONTRADICTS edges in both directions for a node.
- **`promote_candidates(db, agent_id)`** - Promote qualifying techne nodes to episteme. Requirements: confidence >= 0.85, helpful_count >= 3, 2+ SUPPORTS edges, 0 unresolved CONTRADICTS.
- **`decay_stale_nodes(db, agent_id, stale_days, decay_rate)`** - Reduce confidence of nodes not seen for `stale_days`. Deprecates nodes that fall below 0.15 threshold.
- **`demote_stale_episteme(db, agent_id, stale_days)`** - Move stale episteme nodes back to techne layer with `established` maturity.

### 5. Retrieval Engine (`retrieval.py`, ~350 lines)

Dual-process retrieval inspired by System 1/System 2 neuroscience:

- **System 1** (`_fts5_retrieve`): FTS5 full-text search, < 30ms. Handles query sanitization (strips special chars, removes stopwords).
- **System 2** (`build_context_subgraph`): 1-hop graph expansion on FTS5 results, < 20ms.
- **Scoring** (`_rescore`): Ranks results by `confidence * (1 + 0.1*helpful - 0.05*harmful)` with episteme boost (1.2x) and anti-pattern boost (1.15x).

Entry points:
- `dual_retrieve(db, agent_id, query, top_k, include_graph)` - Full dual-process retrieval
- `retrieve_for_error(db, agent_id, error_text, top_k)` - Simplified FTS5-only retrieval for error matching
- `retrieve_anti_patterns(db, agent_id, command)` - Bash guardian with two-stage matching (see below)
- `format_session_briefing(db, agent_id)` - Generates layered text for session start injection (Sophia -> Episteme -> Techne)

**Bash guardian two-stage architecture:**

Stage 1 -- Sophia axioms (`_check_critical_patterns`): Hardcoded regex patterns for universally dangerous commands (rm -rf, git push --force main, DROP DATABASE, dd to devices, fork bombs, chmod 777). Returns synthetic anti-pattern nodes with `source="axiom"`, `layer="sophia"`, `confidence=1.0`. These fire regardless of KG content. `/tmp` paths are excluded.

Stage 2 -- Knowledge-based (FTS5 + token verification): Extracts technical tokens from the command in three categories (substring_tokens, boundary_tokens, command_tokens), then verifies at least one token appears in the **problem portion** of the anti-pattern (before "Fix:").

**Three layers of false positive prevention:**
1. **Safe command allowlist**: Read-only commands (cat, ls, grep, curl, git status, npm list, pip list, etc.) return `[]` immediately.
2. **Problem-only matching**: Token verification runs against the problem text, not the Fix recommendation. This prevents "Fix: use curl to verify" from blocking curl commands.
3. **Subcommand stopwords**: Common English words used as subcommands (status, list, log, show, help, etc.) are excluded from token extraction to prevent matching against their natural English usage in anti-pattern text.
4. **Word-boundary regex**: Short flags (e.g. `-p`) use `(?<!\w)-p(?!\w)` to avoid matching inside words like "happy-path".

### 6. Feedback Loop (`feedback.py`, ~200 lines)

Three signals for confidence evolution + structural edge creation:

| Signal | Trigger | Confidence Change | Counter |
|---|---|---|---|
| Confirmation | Tool success after surfacing | +0.02 | `helpful_count` +1 |
| Refutation | Explicit via MCP or failure detection | -0.03 | `harmful_count` +1 |
| Irrelevance | Surfaced but never used | 0 (decays if count > 10 and helpful = 0) | `irrelevant_count` +1 |

Functions:
- `confirm_surfaced_nodes(db, agent_id, session_id)` - Boost all pending surfaced nodes + create SUPPORTS edges
- `refute_surfaced_nodes(db, agent_id, session_id, error_text)` - Decay pending surfaced nodes
- `record_irrelevance(db, agent_id, session_id)` - Mark pending as irrelevant, conditionally decay
- `apply_explicit_feedback(db, node_id, feedback_type)` - Direct feedback on a specific node (via MCP)

**SUPPORTS edge creation**: When nodes are confirmed, `confirm_surfaced_nodes` creates SUPPORTS edges between newly confirmed nodes and previously confirmed nodes from the same session. This builds graph structure incrementally through real usage, enabling the promotion pipeline (which requires 2+ SUPPORTS edges) to work without depending solely on the reflection pipeline.

### MCP Server (`mcp_server.py`, 502 lines)

stdio JSON-RPC server providing 5 tools for active agent cognition:

| Tool | Purpose |
|---|---|
| `anamnesis_recall` | Active knowledge retrieval by query |
| `anamnesis_record` | Immediate learning capture (bypass reflection) |
| `anamnesis_feedback` | Explicit confirm/refute/irrelevant on a node |
| `anamnesis_status` | Memory statistics |
| `anamnesis_graph` | Graph traversal from a node |

All handlers use `db = None` / `try/finally: if db: db.close()` pattern to prevent connection leaks.

### Maintenance (`maintenance.py`, 54 lines)

Single `maintenance_cycle(db, agent_id)` function that runs:
1. `decay_stale_nodes()` - Reduce stale confidence
2. `promote_candidates()` - Techne -> Episteme
3. `demote_stale_episteme()` - Episteme -> Techne

Order matters: decay first so freshly-decayed nodes aren't promoted, then promote survivors, then demote stale episteme.

## Hook Integration (Dual-Write)

All hooks write to BOTH v1 (`learnings.db`) and v2 (`anamnesis.db`). v2 failures are caught and logged; they never affect v1 operation.

| Hook | v2 Action |
|---|---|
| `on_session_start.py` | `format_session_briefing()` injected into additionalContext |
| `on_prompt_submit.py` | `dual_retrieve()` on user prompt text |
| `on_pre_bash.py` | `retrieve_anti_patterns()` on bash command |
| `on_post_tool.py` | `safe_append_event()` + `safe_record_aisthesis()` |
| `agent_learn.py` | `run_reflection_pipeline()`, falls back to v1 Haiku on failure |

All hooks use `try/finally` to close v2 DB connections even on error.

## Migration (`migrate.py`, 262 lines)

One-time migration from v1 `learnings.db` to v2 `anamnesis.db`:

- Each v1 rule -> `kg_node` (layer assigned by confidence: >= 0.85 -> episteme, else techne)
- Carries over: confidence, helpful_count, category, severity, applies_when, timestamps
- Case-insensitive dedup check before each insert
- Logged as `reflection_run` with trigger_type='migration'
- Idempotent (safe to re-run; second run = 0 new nodes)

Usage: `python3 -m v2.migrate --agent <agent-id>`

## Test Suite

**142 tests across 3 files, all passing.**

| File | Tests | Coverage |
|---|---|---|
| `tests/test_schema.py` | 64 | Schema, CRUD, FTS5, edges, confidence, axioms, stats |
| `tests/test_retrieval.py` | 34 | Dual retrieve, anti-patterns, session briefing |
| `tests/test_feedback.py` | 45 | Feedback signals, KG traversal, promotion, decay, maintenance |

All tests use in-memory SQLite via `fixtures.create_test_db()`. No disk I/O, no network calls.

## File Map

```
<anamnesis-repo>/v2/
  __init__.py              12 lines   Package init
  db.py                   824 lines   Core DB library (schema, CRUD, FTS5)
  migrate.py              262 lines   v1 -> v2 migration
  observer.py              70 lines   Hook event capture utilities
  knowledge_graph.py      325 lines   Graph traversal, promotion, decay
  maintenance.py           54 lines   Maintenance cycle
  reflection.py           470 lines   5-step Sonnet pipeline
  prompts.py              127 lines   LLM prompt constants
  retrieval.py            277 lines   Dual-process retrieval engine
  mcp_server.py           502 lines   stdio MCP server (5 tools)
  feedback.py             187 lines   Three-signal feedback engine
  ARCHITECTURE.md                     This file
  tests/
    __init__.py
    fixtures.py           164 lines   Shared test infrastructure
    test_schema.py        323 lines   Schema + CRUD tests
    test_retrieval.py     244 lines   Retrieval tests
    test_feedback.py      393 lines   Feedback + KG tests
```

## Design Decisions

1. **Separate `anamnesis.db`**: Zero risk to v1 data. Both coexist. v1 is the fallback.
2. **Sonnet over Haiku for reflection**: v1's 0.03% crystallization rate proves Haiku extraction produces too-shallow rules. Sonnet's 5-step pipeline produces causal, validated knowledge.
3. **Three-signal feedback**: Refutation is the critical missing signal. A system that can only confirm, never refute, accumulates beliefs indefinitely.
4. **CONTRADICTS edges block promotion**: Forces resolution rather than silent accumulation of inconsistencies.
5. **Global timeout budget in reflection**: Shrinking deadline ensures the 5-step pipeline never exceeds the Stop hook's 10s budget, even if individual LLM calls are slow.
6. **`id DESC` tiebreaker on all temporal queries**: In-memory SQLite (tests) and fast sequential inserts can produce identical `created_at` timestamps. The `id DESC` tiebreaker guarantees deterministic ordering.
7. **Anti-pattern word-boundary matching**: Short flags like `-p` use regex `(?<!\w)-p(?!\w)` to prevent false positives against words like "happy-path".

## Performance Targets

| Hook | Budget | Mechanism |
|---|---|---|
| `on_session_start` | < 200ms | FTS5 briefing generation |
| `on_prompt_submit` | < 100ms | Dual-process retrieval |
| `on_pre_bash` | < 50ms | Anti-pattern token matching |
| `on_post_tool` | < 100ms | Event append + aisthesis write |
| `agent_learn` (Stop) | < 10s | 5-step reflection with global deadline |
