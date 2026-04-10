#!/usr/bin/env python3
"""Hermes Micro-Cortex Hook: PostToolUse.

Three functions across two layers:
  Layer 1 (Working Memory): Record every tool event for session context
  Layer 2 (Feedback):       Credit surfaced rules on success
  Layer 2 (Error Fixes):    Search for relevant fixes on failure

Must be fast (<100ms) -- FTS5 query on failure, simple SQL on success."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from learnings_db import (
    detect_active_agent, open_db,
    credit_surfaced, match_error_fixes, log_event
)
from working_memory import record_tool_event

# Tools where success crediting is meaningful
CREDIT_TOOLS = {"Bash", "Write", "Edit", "NotebookEdit"}
SKIP_TOOLS = {"TaskOutput", "TodoWrite", "SendMessage", "AskUserQuestion"}

# Error indicators in Bash output
BASH_FAIL_INDICATORS = [
    "Exit code", "Error:", "error:", "ENOENT", "EACCES",
    "Permission denied", "command not found", "No such file",
    "npm ERR", "ModuleNotFoundError", "SyntaxError",
    "TypeError", "ReferenceError", "fatal:",
    "FAILED", "failed", "Cannot find module",
]


def _is_failure(tool_name: str, response: str) -> bool:
    """Check if tool response indicates failure."""
    if tool_name == "Bash":
        for indicator in BASH_FAIL_INDICATORS:
            if indicator in response:
                return True
    elif tool_name in ("Write", "Edit", "NotebookEdit"):
        lower = response.lower()
        if "error" in lower or "failed" in lower:
            return True
    return False


def _extract_files(data: dict) -> list:
    """Extract file paths from tool input."""
    tool_input = data.get("tool_input", {})
    files = []
    for key in ("file_path", "path", "notebook_path"):
        v = tool_input.get(key, "")
        if v:
            files.append(v)
    return files


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return

    tool_name = data.get("tool_name", "")
    if tool_name in SKIP_TOOLS:
        return

    agent_id = detect_active_agent()
    if not agent_id:
        return

    tool_response = str(data.get("tool_response", ""))
    session_id = data.get("session_id", "")
    tool_input = data.get("tool_input", {})

    try:
        db = open_db(agent_id)  # Auto-creates DB + schema for new agents
        is_fail = _is_failure(tool_name, tool_response)

        # Layer 1: Record to working memory (always)
        if session_id:
            if is_fail:
                record_tool_event(
                    db, session_id, event_type="error",
                    tool_name=tool_name,
                    command=tool_input.get("command", "")[:300],
                    error_text=tool_response[:500],
                    files_touched=_extract_files(data),
                )
            elif tool_name == "Bash":
                record_tool_event(
                    db, session_id, event_type="command",
                    tool_name=tool_name,
                    command=tool_input.get("command", "")[:300],
                    summary=tool_response[:100],
                )
            elif tool_name in ("Write", "Edit", "NotebookEdit"):
                record_tool_event(
                    db, session_id, event_type="file_write",
                    tool_name=tool_name,
                    summary=tool_input.get("file_path", "")
                            or tool_input.get("notebook_path", ""),
                    files_touched=_extract_files(data),
                )

        if is_fail:
            # FAILURE PATH: search for relevant fixes
            error_snippet = tool_response[:500]
            fixes = match_error_fixes(db, error_snippet, top_k=2)
            context_parts = []
            if fixes:
                lines = ["## Hermes: Relevant Learnings for This Error"]
                for f in fixes:
                    mat = f.get("maturity", "")
                    proven = " [PROVEN]" if mat == "crystallized" else ""
                    lines.append(
                        f"- [{f['category']}]{proven} {f['content']}"
                    )
                context_parts.append("\n".join(lines))

            # Point 1: Tinyfish error-triggered web search (didaxis)
            # Only search web if NO local knowledge found
            if not fixes:
                try:
                    from v2.tinyfish import search_for_error
                    web_ctx = search_for_error(
                        error_snippet,
                        tool_name=tool_name,
                        command=tool_input.get("command", "")[:200],
                    )
                    if web_ctx:
                        context_parts.append(web_ctx)
                except Exception:
                    pass  # Tinyfish failure must never block hook

            if context_parts:
                db.close()
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "\n\n".join(context_parts)
                    }
                }
                print(json.dumps(output))
                return

        elif tool_name in CREDIT_TOOLS and session_id:
            # SUCCESS PATH: credit surfaced rules
            count = credit_surfaced(db, session_id)
            if count > 0:
                log_event(db, "success_credit", {
                    "session_id": session_id,
                    "tool": tool_name,
                    "rules_credited": count,
                })

        db.close()

        # --- Hermes v2 dual-write (never crash) ---
        v2_conn = None
        try:
            import re as _re
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from v2.observer import safe_append_event, safe_record_aisthesis
            from v2.feedback import confirm_surfaced_nodes, refute_surfaced_nodes
            from v2 import db as v2db
            event_type = 'tool_error' if is_fail else 'tool_success'

            # Parse exit code from Bash output
            v2_exit_code = None
            if tool_name == "Bash":
                ec_match = _re.search(r'Exit code[:\s]+(\d+)', tool_response)
                if ec_match:
                    v2_exit_code = int(ec_match.group(1))
                elif is_fail:
                    v2_exit_code = 1  # Infer non-zero on failure

            # Capture tool output based on tool type
            v2_tool_output = ""
            if tool_name == "Bash":
                # Always store output (not just failures)
                v2_tool_output = tool_response[:2000]
            elif tool_name in ("Glob", "Grep", "Read"):
                # These were previously discarded entirely
                v2_tool_output = tool_response[:500]
            elif tool_name in ("Write", "Edit", "NotebookEdit"):
                fp = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")
                v2_tool_output = f"{tool_name}: {fp}"[:500]

            safe_append_event(agent_id, event_type, session_id=session_id,
                              tool_name=tool_name,
                              command=tool_input.get('command', '')[:500],
                              error_text=tool_response[:1000] if is_fail else '',
                              file_path=(_extract_files(data) or [''])[0],
                              exit_code=v2_exit_code,
                              tool_output=v2_tool_output)
            if session_id:
                safe_record_aisthesis(agent_id, session_id, event_type,
                                      tool_name=tool_name,
                                      command=tool_input.get('command', '')[:300],
                                      error_text=tool_response[:500] if is_fail else '',
                                      summary=v2_tool_output[:300] if not is_fail else '',
                                      files_touched=_extract_files(data))
            if is_fail and session_id:
                v2_conn = v2db.open_db(agent_id)
                refute_surfaced_nodes(v2_conn, agent_id, session_id,
                                      error_text=tool_response[:1000])
            elif tool_name in CREDIT_TOOLS and session_id:
                v2_conn = v2db.open_db(agent_id)
                # Build action context for cosine-similarity credit assignment
                action_ctx = f"{tool_name} {tool_input.get('command', '')} {tool_response[:500]}"
                confirm_surfaced_nodes(v2_conn, agent_id, session_id,
                                       action_context=action_ctx)
        except Exception:
            pass  # v2 failure must never affect v1
        finally:
            if v2_conn:
                try:
                    v2_conn.close()
                except Exception:
                    pass

    except Exception:
        pass  # Hooks must NEVER crash


if __name__ == "__main__":
    main()
