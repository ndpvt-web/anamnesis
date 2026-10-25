#!/usr/bin/env python3
"""Anamnesis hook: UserPromptSubmit.

Multi-layer retrieval:
  Layer 1 (Working Memory): Recent session context -- errors, commands, files
  Layer 2 (Learned Rules):  FTS5 retrieval from agent's knowledge base
  Layer 4 (Crystallized):   Battle-tested rules get priority boost in FTS5

Must be fast (<100ms) -- FTS5 + SQLite only, no LLM calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from learnings_db import (
    detect_active_agent, open_db,
    fts5_retrieve, record_surfaced
)
from working_memory import get_recent_context

MIN_PROMPT_LENGTH = 15
TOP_K = 3


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return

    prompt = data.get("prompt", "")
    if len(prompt) < MIN_PROMPT_LENGTH:
        return

    agent_id = detect_active_agent(data)

    try:
        db = open_db(agent_id)  # Auto-creates DB + schema for new agents
        session_id = data.get("session_id", "")

        sections = []

        # Layer 1: Working Memory (session context)
        wm_context = get_recent_context(db, session_id, limit=5)
        if wm_context:
            sections.append(wm_context)

        # Layer 2+4: FTS5 retrieval (crystallized get 2x boost automatically)
        rules = fts5_retrieve(db, prompt, top_k=TOP_K)
        if rules:
            if session_id:
                record_surfaced(db, session_id, [r["id"] for r in rules])
            lines = [f"## Anamnesis Learnings ({agent_id})"]
            for r in rules:
                helpful = r.get("helpful_count") or 0
                help_mark = f" +{helpful}" if helpful else ""
                sev = r.get("severity", "")
                sev_mark = f" [{sev.upper()}]" if sev in ("high", "critical") else ""
                mat = r.get("maturity", "")
                mat_mark = " [PROVEN]" if mat == "crystallized" else ""
                lines.append(
                    f"- [{r['confidence']:.1f}|{r['category']}{help_mark}]"
                    f"{sev_mark}{mat_mark} {r['content']}"
                )
            sections.append("\n".join(lines))

        # --- v2 retrieval (never crash) ---
        v2_conn = None
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from v2.observer import safe_append_event
            from v2.retrieval import dual_retrieve, classify_error_family
            from v2 import db as v2db
            safe_append_event(agent_id, 'prompt_submit', session_id=session_id,
                              prompt_text=prompt[:2000])
            v2_conn = v2db.open_db(agent_id)
            # Session-scoped context accumulation: check for error history
            combined_query = prompt
            if session_id:
                session_errors = v2db.get_session_events(
                    v2_conn, agent_id, session_id, limit=20
                )
                error_events = [e for e in session_errors
                                if e.get("event_type") == "tool_error"]
                if error_events:
                    error_tokens = " ".join(
                        (e.get("error_text", "") or "")[:100]
                        for e in error_events[-3:]
                    )
                    combined_query = f"{prompt} {error_tokens}"
            error_family = classify_error_family(combined_query)
            v2_results = dual_retrieve(v2_conn, agent_id, combined_query,
                                        top_k=TOP_K, error_family=error_family)
            if v2_results:
                v2db.record_surfaced(v2_conn, agent_id, session_id,
                                     [r['id'] for r in v2_results])
                v2_lines = [f"## Anamnesis: Knowledge Graph ({agent_id})"]
                for r in v2_results:
                    layer_tag = f"[{r['layer'].upper()}]" if r.get('layer') else ""
                    v2_lines.append(
                        f"- [{r['confidence']:.1f}|{r['category']}] {layer_tag} {r['content']}"
                    )
                sections.append('\n'.join(v2_lines))
        except Exception:
            pass  # v2 failure must never affect v1
        finally:
            if v2_conn:
                try:
                    v2_conn.close()
                except Exception:
                    pass

        if not sections:
            db.close()
            return

        db.close()
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "\n\n".join(sections)
            }
        }
        print(json.dumps(output))

    except Exception:
        pass  # Hooks must NEVER crash


if __name__ == "__main__":
    main()
