# Anamnesis Integrations

Anamnesis is harness-agnostic: every integration point is a process that
reads JSON on stdin, writes JSON on stdout, and keeps all state under the
data home (`$ANAMNESIS_HOME` or `~/.anamnesis`).

## Capability matrix

| Capability | Claude Code hooks | Generic MCP client | Shell CLI |
|---|---|---|---|
| Safety guardian (blocks dangerous commands) | yes | no (no lifecycle events) | no |
| Prompt-time knowledge retrieval | yes | yes (`anamnesis_recall`) | yes (`briefing`) |
| Automatic event observation | yes | no | no |
| Success/failure feedback loop | yes | manual (`anamnesis_feedback`) | no |
| Session-end LLM reflection | yes | no | no |
| Manual learning capture | yes | yes (`anamnesis_record`) | yes (`record`) |
| Stats / diagnostics | yes | yes (`anamnesis_status`) | yes (`status`, `doctor`) |

Honest limits:

- MCP alone does **not** give Anamnesis lifecycle visibility. Without
  hook-equivalent events there is no automatic observation, feedback, or
  reflection; memory grows only through explicit `anamnesis_record` calls.
- With no LLM API key configured, retrieval, observation, feedback, and
  maintenance all work; reflection, session extraction, and web-knowledge
  seeding are disabled. `python3 install.py --check` reports this.
- The Tinyfish web-enrichment integration is optional and off without
  `TINYFISH_API_KEY`.

## Claude Code

`python3 install.py --claude-code` merges this into
`~/.claude/settings.json` (backup first, unrelated keys preserved):

```json
{
  "hooks": {
    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_pre_bash.py"}]}],
    "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_post_tool.py"}]}],
    "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_prompt_submit.py"}]}],
    "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/on_session_start.py"}]}],
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/anamnesis/agent_learn.py"}]}]
  },
  "mcpServers": {
    "anamnesis": {"type": "stdio", "command": "python3", "args": ["/path/to/anamnesis/v2/mcp_server.py"]}
  }
}
```

## Generic hook protocol

Each hook reads one JSON object on stdin and either stays silent or prints
one JSON object on stdout. Hooks never crash (exit 0 on internal errors)
and never require setup beyond the data home.

| Script | stdin fields used | stdout contract |
|---|---|---|
| `on_session_start.py` | `session_id`, `cwd`, `prompt` | `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}` |
| `on_prompt_submit.py` | `session_id`, `cwd`, `prompt` | same shape, event `UserPromptSubmit` |
| `on_pre_bash.py` | `tool_input.command` | same shape, event `PreToolUse`; may add `permissionDecision: "deny"` + `permissionDecisionReason` |
| `on_post_tool.py` | `session_id`, `tool_name`, `tool_input`, `tool_response` | same shape, event `PostToolUse` |
| `agent_learn.py` | `session_id`, `cwd`, `transcript_path` (JSONL transcript) | `{"suppressOutput": true}` |

Agent identity resolution order: `ANAMNESIS_AGENT_ID` env -> `agent_id` /
`agentId` / `agent` / `agent_name` in the payload -> `agent_id` in
`<data_home>/config.json` -> per-project id derived from the payload's
`cwd` -> persisted machine default. Identity is never empty, so hooks are
never silently inert. Memory is isolated per project by default; set
`ANAMNESIS_AGENT_ID` to share one memory across projects or to run
multiple named agents.

## MCP tools

`v2/mcp_server.py` speaks MCP (stdio, JSON-RPC 2.0, protocol 2024-11-05):

- `anamnesis_recall {query, top_k?, layer?}` -- search the knowledge base
- `anamnesis_record {content, category?, severity?, applies_when?}` -- record a learning now
- `anamnesis_feedback {node_id, feedback: confirmed|refuted|irrelevant}`
- `anamnesis_status {}` -- memory statistics
- `anamnesis_graph {node_id, max_depth?}` -- traverse related knowledge

## LLM providers

Reflection and extraction use any of:

- **OpenAI-compatible chat completions** (default base
  `https://api.openai.com/v1`; override with `ANAMNESIS_LLM_BASE_URL` or
  `OPENAI_BASE_URL`). Works with OpenAI, OpenRouter, and self-hosted
  compatible gateways.
- **Anthropic native** (auto-detected when the base URL is
  `api.anthropic.com`, or force with `ANAMNESIS_LLM_PROVIDER=anthropic`).

Two model slots: `ANAMNESIS_LLM_MODEL` (reflection) and
`ANAMNESIS_LLM_FAST_MODEL` (extraction/parsing). Defaults are matched to
the detected provider. Other provider APIs (e.g. Gemini native) are not
currently supported; the doctor reports unconfigured providers rather
than failing silently.
