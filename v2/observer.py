#!/usr/bin/env python3
"""Safe wrappers around db.py for hook event capture.
All functions catch exceptions internally -- hooks must NEVER crash.
Import: from v2 import db as v2db"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2 import db as v2db


def safe_append_event(agent_id: str, event_type: str,
                      session_id: str = "", tool_name: str = "",
                      command: str = "", file_path: str = "",
                      error_text: str = "", prompt_text: str = "",
                      metadata: dict = None,
                      exit_code: int = None, tool_output: str = "",
                      duration_ms: int = None) -> int:
    """Safely append to empeiria. Returns event ID or -1 on failure.
    Opens and closes its own DB connection.
    Truncates: command to 500 chars, error_text to 1000 chars,
    prompt_text to 2000 chars, tool_output to 2000 chars."""
    try:
        db = v2db.open_db(agent_id)
        try:
            event_id = v2db.append_event(
                db,
                agent_id,
                event_type,
                session_id=session_id,
                tool_name=tool_name,
                command=(command or "")[:500],
                file_path=file_path or "",
                error_text=(error_text or "")[:1000],
                prompt_text=(prompt_text or "")[:2000],
                metadata=metadata,
                exit_code=exit_code,
                tool_output=(tool_output or "")[:2000],
                duration_ms=duration_ms,
            )
            return event_id
        finally:
            db.close()
    except Exception as e:
        print(f"[hermes-v2] safe_append_event failed: {e}", file=sys.stderr)
        return -1


def safe_record_aisthesis(agent_id: str, session_id: str,
                          event_type: str, tool_name: str = "",
                          summary: str = "", command: str = "",
                          error_text: str = "",
                          files_touched: list = None) -> int:
    """Safely record working memory. Returns row ID or -1 on failure.
    Opens and closes its own DB connection.
    Truncates: command to 500 chars, error_text to 500 chars, summary to 300 chars."""
    try:
        db = v2db.open_db(agent_id)
        try:
            row_id = v2db.record_aisthesis(
                db,
                agent_id,
                session_id,
                event_type,
                tool_name=tool_name,
                summary=(summary or "")[:300],
                command=(command or "")[:500],
                error_text=(error_text or "")[:500],
                files_touched=files_touched,
            )
            return row_id
        finally:
            db.close()
    except Exception as e:
        print(f"[hermes-v2] safe_record_aisthesis failed: {e}", file=sys.stderr)
        return -1
