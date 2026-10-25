#!/usr/bin/env python3
"""Anamnesis session-end learning hook (Stop hook).

Extracts learnings from the session transcript with a fast LLM and runs
the v2 reflection pipeline, writing to the agent's SQLite+FTS5 stores.
LLM provider comes from anamnesis_config; with no API key configured the
hook still runs maintenance and working-memory cleanup."""

import json
import os
import sys
from pathlib import Path

# Add parent dir so we can import learnings_db
sys.path.insert(0, str(Path(__file__).parent))
from learnings_db import open_db, add_rule, detect_active_agent, log_event
from crystallized import maintenance_cycle
from working_memory import purge_session, session_summary, purge_stale
from v2 import llm

NOISE_PATTERNS = [
    "session is being continued from a previous",
    "Base directory for this skill",
    "<teammate-message", "<task-notification>",
    "system-reminder", "<system-reminder>",
]

EXTRACT_PROMPT = """You are a learning extractor for an AI agent. Analyze this session transcript and extract learnings in 3 categories.

RULES:
- Only extract GENUINELY useful, non-obvious patterns
- Write each as a general principle, NOT tied to specific file paths
- Skip trivial things ("check if file exists", "read before edit")
- Max 3 items per category. Return FEWER if nothing quality found.
- Return valid JSON only, no markdown

Return this exact JSON structure:
{{
  "anti_patterns": [
    {{"what": "what went wrong", "why": "root cause", "instead": "what to do instead", "severity": "high|medium|low"}}
  ],
  "preferences": [
    {{"preference": "observed or stated user preference", "context": "when this applies"}}
  ],
  "principles": [
    {{"rule": "imperative principle", "derived_from": "what taught this", "applies_when": "conditions"}}
  ]
}}

If a category has nothing worth extracting, use an empty array.

SESSION TRANSCRIPT:
{transcript}"""


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return

    transcript_path = data.get("transcript_path", "")
    session_id = data.get("session_id", "")
    if not transcript_path or not Path(transcript_path).exists():
        return

    agent_id = detect_active_agent(data)

    messages = _extract_messages(transcript_path, limit=15)
    if len(messages) < 2:
        return

    transcript_text = "\n".join(
        f"[{m['role']}]: {m['text'][:500]}" for m in messages
    )
    if len(transcript_text) > 4000:
        transcript_text = transcript_text[:4000] + "\n[...truncated]"

    # --- v2 reflection pipeline first ---
    v2_succeeded = False
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from v2.reflection import run_reflection_pipeline
        from v2.feedback import record_irrelevance
        from v2.maintenance import maintenance_cycle as v2_maintenance
        from v2 import db as v2db
        v2_result = run_reflection_pipeline(agent_id, session_id)
        if v2_result and v2_result.get('status') == 'completed':
            v2_succeeded = True
            _log(f"v2 reflection: +{v2_result.get('nodes_created', 0)} nodes, "
                 f"+{v2_result.get('edges_created', 0)} edges, "
                 f"{v2_result.get('duration_ms', 0)}ms")
        # Always: mark remaining surfaced nodes as irrelevant
        v2_conn = v2db.open_db(agent_id)
        try:
            record_irrelevance(v2_conn, agent_id, session_id)
            v2_maint = v2_maintenance(v2_conn, agent_id)
        finally:
            v2_conn.close()
    except Exception as e:
        _log(f"v2 reflection failed: {e}")

    # Fall back to v1 Haiku extraction only if v2 did not succeed
    if v2_succeeded:
        learnings = None
    else:
        _log("v2 did not succeed, falling back to v1 fast-model extraction")
        learnings = _call_haiku(transcript_text)
    if not learnings:
        return

    # Open agent's SQLite database
    db = open_db(agent_id)
    added = {"anti_pattern": 0, "preference": 0, "principle": 0}

    # Anti-patterns -> rules with category='anti_pattern'
    for ap in learnings.get("anti_patterns", []):
        what = ap.get("what", "").strip()
        instead = ap.get("instead", "").strip()
        if not what:
            continue
        content = f"{what}. Fix: {instead}" if instead else what
        severity = ap.get("severity", "medium")
        rid = add_rule(db, content, category="anti_pattern", severity=severity,
                       source_session=session_id)
        if rid > 0:
            added["anti_pattern"] += 1

    # Preferences -> rules with category='preference'
    for p in learnings.get("preferences", []):
        pref = p.get("preference", "").strip()
        if not pref:
            continue
        context = p.get("context", "")
        content = f"{pref} (context: {context})" if context else pref
        rid = add_rule(db, content, category="preference",
                       applies_when=context, source_session=session_id)
        if rid > 0:
            added["preference"] += 1

    # Principles -> rules with category='principle'
    for p in learnings.get("principles", []):
        rule = p.get("rule", "").strip()
        if not rule:
            continue
        applies = p.get("applies_when", "general")
        rid = add_rule(db, rule, category="principle",
                       applies_when=applies, source_session=session_id)
        if rid > 0:
            added["principle"] += 1

    total = sum(added.values())
    if total > 0:
        log_event(db, "session_learning", {
            "agent_id": agent_id, "session_id": session_id,
            "added": added, "total": total,
        })
        _log(f"Agent '{agent_id}': +{added['anti_pattern']}ap "
             f"+{added['preference']}pref +{added['principle']}princ")

    # Layer 4: Run crystallization maintenance cycle (includes confidence decay)
    maint = maintenance_cycle(db)
    if maint.get("promoted", 0) > 0 or maint.get("demoted", 0) > 0 or maint.get("decayed", 0) > 0:
        _log(f"Maintenance: +{maint['promoted']} promoted, "
             f"-{maint['demoted']} demoted, "
             f"{maint.get('deprecated', 0)} deprecated, "
             f"{maint.get('decayed', 0)} decayed")

    # Layer 1: Purge working memory for this session + stale entries
    if session_id:
        wm_stats = session_summary(db, session_id)
        purged = purge_session(db, session_id)
        if purged > 0:
            _log(f"Working memory: purged {purged} entries "
                 f"(session had {wm_stats.get('total', 0)} events)")
    purge_stale(db)

    db.close()
    print(json.dumps({"suppressOutput": True}))


def _extract_messages(transcript_path, limit=15):
    """Extract user and assistant messages from JSONL transcript."""
    messages = []
    try:
        with open(transcript_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    role = entry.get("type", "")
                    if role not in ("user", "assistant"):
                        continue
                    msg = entry.get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = [
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        content = " ".join(parts)
                    if not isinstance(content, str) or len(content) < 10:
                        continue
                    if any(n in content for n in NOISE_PATTERNS):
                        continue
                    messages.append({"role": role, "text": content})
                except (json.JSONDecodeError, KeyError):
                    continue
        return messages[-limit:]
    except Exception:
        return []


def _call_haiku(transcript_text):
    """Extract learnings with the configured fast model. Returns JSON or None."""
    prompt = EXTRACT_PROMPT.format(transcript=transcript_text)
    return llm.chat_json("You are a learning extractor for an AI agent.",
                         prompt, purpose="fast", timeout=20, max_tokens=800)


def _log(msg):
    print(f"[anamnesis-learn] {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
