# Anamnesis

**Aristotelian Memory Architecture for AI Coding Agents**

Anamnesis is a persistent memory system for AI coding agents, grounded in Aristotle's *Posterior Analytics* and his 5-layer epistemological hierarchy. It enables agents to learn from experience, retain causal knowledge, and improve over time.

## Aristotelian Foundation

The system implements Aristotle's hierarchy of knowledge acquisition:

| Layer | Greek | Role | In Anamnesis |
|-------|-------|------|-------------|
| **Sophia** | Wisdom | Unchanging first principles | Hardcoded safety axioms (rm -rf, force push) |
| **Episteme** | Scientific Knowledge | Justified, causal, universal | Promoted nodes: confidence >= 0.85, helpful >= 3, supports >= 2 |
| **Techne** | Craft Knowledge | Practical know-how | Reflection-generated hypotheses with WHEN/THEN/BECAUSE structure |
| **Empeiria** | Experience | Accumulated observations | Tool events, errors, successes recorded per session |
| **Aisthesis** | Perception | Raw sense data | Real-time tool invocations and outputs |

## Architecture

```
Session Start ──> Briefing (relevant knowledge surfaced)
      │
Prompt Submit ──> Retrieval (domain-filtered, threshold-gated)
      │
Tool Execution ──> Observer (records empeiria)
      │
Tool Result ──> Feedback (confirm/refute surfaced knowledge)
      │
Session End ──> Reflection Pipeline (5-step LLM: Narrate→Diagnose→Hypothesize→Validate→Store)
      │
      └──> Knowledge Graph (nodes + edges + decay + promotion)
```

### Key Components

- **v2/db.py** -- SQLite schema with FTS5 full-text search, domain classification, node lifecycle
- **v2/retrieval.py** -- Dual-process retrieval: FTS5 (System 1) + Graph expansion (System 2) + adaptive threshold
- **v2/reflection.py** -- 5-step LLM pipeline with quality gating (salience scoring)
- **v2/knowledge_graph.py** -- Graph operations: BFS traversal, promotion, Ebbinghaus decay
- **v2/feedback.py** -- Cosine-similarity credit assignment, confidence updates
- **Hook entry points** -- `on_session_start.py`, `on_prompt_submit.py`, `on_post_tool.py`, `on_pre_bash.py`

## v3 Features (Retrieval Precision)

Addressing the *Doctrine of the Mean* -- too little knowledge = baseline, too much = noise:

1. **Domain Genus Classification** -- 8 domains (web_backend, data_processing, database, etc.) prevent cross-domain retrieval pollution
2. **Adaptive Relevance Threshold** -- Ratio-based filtering that tightens as the knowledge graph grows
3. **Write-Time Quality Gating** -- Salience scoring (novelty + specificity + actionability) rejects platitudes
4. **Enhanced Deduplication** -- Jaccard 0.8 threshold, error-family-aware at 0.65

## Experiment Results

| Phase | Description | Turns | Reduction |
|-------|-----------|-------|-----------|
| P1 | Baseline (no memory) | 554 | -- |
| P2 | Anamnesis v2 (basic) | 426 | -23.1% |
| P3 | Deep Knowledge (WHEN/THEN/BECAUSE) | 489 | -11.7% |
| P4 | v3 (domain + threshold + quality gate) | TBD | TBD |

25-task benchmark across 5 categories (Flask microservices, CSV pipelines, config apps, SQLite migrations, API clients).

## Test Suite

```bash
cd /path/to/anamnesis && python3 -m pytest v2/tests/ -q
```

The suite covers schema, retrieval, feedback, reflection, integration,
fresh-install end-to-end (hooks as subprocesses on a bare HOME), the MCP
stdio protocol, installer behavior, and LLM client shapes against a local
HTTP fixture. Live-store tests are skipped unless
`ANAMNESIS_TEST_LIVE_AGENT` is set.

## Installation

Zero dependencies (Python 3.8+ stdlib only). Clone, install, verify:

```bash
git clone https://github.com/ndpvt-web/anamnesis
cd anamnesis
python3 install.py            # creates ~/.anamnesis + config
python3 install.py --check    # doctor: diagnostics + hook/MCP smoke tests
```

With no further configuration, Anamnesis works out of the box: agent
identity is derived per project, stores are created on first use, and the
safety guardian is active immediately. LLM-powered reflection and session
extraction turn on when you export a key:

```bash
export ANAMNESIS_LLM_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY
export ANAMNESIS_LLM_BASE_URL=...       # optional; any OpenAI-compatible
                                        # endpoint or api.anthropic.com
export ANAMNESIS_LLM_MODEL=...          # optional overrides
export ANAMNESIS_LLM_FAST_MODEL=...
```

### Claude Code

```bash
python3 install.py --claude-code
```

Registers all five hooks and the MCP server in `~/.claude/settings.json`
(existing settings are backed up and preserved). Remove with
`python3 install.py --uninstall`.

### Any other harness

- **MCP client** -- add the stdio server printed by `install.py`; gives
  `anamnesis_recall / record / feedback / status / graph`.
- **Lifecycle hooks** -- call the five hook scripts with JSON on stdin
  (contract in `docs/INTEGRATIONS.md`).
- **Shell CLI** -- `python3 anamnesis_cli.py briefing|record|status|doctor`.

See `docs/INTEGRATIONS.md` for the full capability matrix and per-harness
limits. Upgrading from a pre-portability install? See `MIGRATION.md`.

## Research

- Novel application of Aristotle's *Posterior Analytics* to ML memory architecture (zero prior art found in literature review)
- Complementary Learning Systems theory (hippocampal fast binding + cortical slow consolidation)
- Informed by: ExpeL, CLIN, G-Memory, SEEKR, LightRAG, Karpathy's LLM Wiki

See `docs/anamnesis_v3_brainstorm.md` for the full literature review and design rationale.

## License

MIT
