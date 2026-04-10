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

146 tests across 11 test files:

```bash
cd /path/to/anamnesis && python3 -m pytest v2/tests/ -v
```

## Installation (Claude Code Hooks)

Configure in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_pre_bash.py"}]}],
    "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_post_tool.py"}]}],
    "PrePromptSubmit": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_prompt_submit.py"}]}],
    "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_session_start.py"}]}],
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/agent_learn.py"}]}]
  }
}
```

## Research

- Novel application of Aristotle's *Posterior Analytics* to ML memory architecture (zero prior art found in literature review)
- Complementary Learning Systems theory (hippocampal fast binding + cortical slow consolidation)
- Informed by: ExpeL, CLIN, G-Memory, SEEKR, LightRAG, Karpathy's LLM Wiki

See `docs/anamnesis_v3_brainstorm.md` for the full literature review and design rationale.

## License

MIT
