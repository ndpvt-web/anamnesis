#!/usr/bin/env python3
"""Hermes Micro-Cortex Hook: PreToolUse (Bash) - Command guardian.
Checks bash commands against agent-learned anti-patterns.
Can BLOCK dangerous commands or inject warnings.
Must be fast (<50ms) -- keyword matching only, no FTS5."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from learnings_db import (
    detect_active_agent, open_db, match_anti_patterns
)


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return

    tool_input = data.get("tool_input", {})
    command = tool_input.get("command", "")
    if not command:
        return

    agent_id = detect_active_agent()
    if not agent_id:
        return

    try:
        db = open_db(agent_id)  # Auto-creates DB + schema for new agents
        matches = match_anti_patterns(db, command)
        db.close()

        # --- Hermes v2 fallback (never crash) ---
        v2_conn = None
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from v2.retrieval import retrieve_anti_patterns
            from v2 import db as v2db
            v2_conn = v2db.open_db(agent_id)
            v2_matches = retrieve_anti_patterns(v2_conn, agent_id, command)
            if v2_matches:
                top_v2 = v2_matches[0]
                severity_v2 = top_v2.get('severity', 'medium')
                is_axiom = top_v2.get('source') == 'axiom'
                # Critical/axiom matches ALWAYS take precedence (safety net)
                # Non-critical v2 matches only used if v1 found nothing
                if is_axiom or severity_v2 == 'critical':
                    output_v2 = {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"HERMES GUARDIAN [{agent_id}]: Blocked by v2 "
                                f"anti-pattern [{severity_v2.upper()}]: {top_v2['content']}"
                            ),
                            "additionalContext": (
                                f"WARNING: This command was blocked based on a learned "
                                f"anti-pattern. {top_v2['content']}. "
                                f"Find an alternative approach."
                            )
                        }
                    }
                    print(json.dumps(output_v2))
                    return
                elif severity_v2 == 'high' and not matches:
                    output_v2 = {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"HERMES GUARDIAN [{agent_id}]: Blocked by v2 "
                                f"anti-pattern [{severity_v2.upper()}]: {top_v2['content']}"
                            ),
                            "additionalContext": (
                                f"WARNING: This command was blocked based on a learned "
                                f"anti-pattern. {top_v2['content']}. "
                                f"Find an alternative approach."
                            )
                        }
                    }
                    print(json.dumps(output_v2))
                    return
                elif not matches:
                    output_v2 = {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "additionalContext": (
                                f"HERMES NOTE [{agent_id}]: Learned caution (v2) - "
                                f"{top_v2['content']}"
                            )
                        }
                    }
                    print(json.dumps(output_v2))
                    return
        except Exception:
            pass  # v2 failure must never affect v1
        finally:
            if v2_conn:
                try:
                    v2_conn.close()
                except Exception:
                    pass

        if not matches:
            return

        top = matches[0]
        severity = top.get("severity", "medium")

        if severity in ("critical", "high"):
            # BLOCK the command
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"HERMES GUARDIAN [{agent_id}]: Blocked by learned "
                        f"anti-pattern [{severity.upper()}]: {top['content']}"
                    ),
                    "additionalContext": (
                        f"WARNING: This command was blocked based on a learned "
                        f"anti-pattern. {top['content']}. "
                        f"Find an alternative approach."
                    )
                }
            }
        else:
            # WARN but allow
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": (
                        f"HERMES NOTE [{agent_id}]: Learned caution - "
                        f"{top['content']}"
                    )
                }
            }
        print(json.dumps(output))

    except Exception:
        pass  # Hooks must NEVER crash


if __name__ == "__main__":
    main()
