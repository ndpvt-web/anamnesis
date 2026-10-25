# Migration: pre-portability installs

This release makes Anamnesis harness- and provider-agnostic. A few
breaking changes are deliberate; each has a one-time migration step.

## 1. Environment variables renamed

Provider-specific variable names were replaced by generic ones:

| Setting | New variable |
|---|---|
| Agent identity | `ANAMNESIS_AGENT_ID` |
| LLM API key | `ANAMNESIS_LLM_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| LLM endpoint | `ANAMNESIS_LLM_BASE_URL` (or `OPENAI_BASE_URL`) |
| Reflection model | `ANAMNESIS_LLM_MODEL` |
| Fast/extraction model | `ANAMNESIS_LLM_FAST_MODEL` |
| Data directory | `ANAMNESIS_HOME` |

There is no compatibility shim for the old names by design. If you ran a
previous internal deployment, rename its variables once.

## 2. Data home moved

Stores now live under `$ANAMNESIS_HOME` (default `~/.anamnesis`). To keep
existing knowledge, import it explicitly - nothing is copied automatically:

```bash
python3 migrate_store.py --from-dir /path/to/old/agents --dry-run
python3 migrate_store.py --from-dir /path/to/old/agents
```

The importer recognizes per-agent `learnings/learnings.db` and
`hermes/hermes.db` layouts, copies them into the new home (renaming the
v2 store to `anamnesis.db`), never modifies the source, and refuses to
overwrite an existing target without `--force`.

Within an existing data home, a legacy `hermes/hermes.db` store is copied
forward to `anamnesis.db` on first open (non-destructive; the old file
remains as a backup).

## 3. MCP tool names renamed

`hermes_recall/record/feedback/status/graph` are now
`anamnesis_recall/record/feedback/status/graph`; the server name is
`anamnesis`. Update MCP client configs (the installer writes the new
names) and any prompts/scripts that call the old tool names.

## 4. Cross-project knowledge source removed

The session-start briefing no longer reads a sibling tool's shared
database. Import any such knowledge explicitly instead: dump it to rules
and load via `anamnesis_cli.py record`, or migrate the store with
`migrate_store.py`.
