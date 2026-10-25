#!/usr/bin/env python3
"""Shared LLM client for Anamnesis.

One client for every LLM call in the system (reflection pipeline, session
extraction, web-knowledge parsing). Provider settings come from
anamnesis_config.llm_config(); supports OpenAI-compatible chat-completions
endpoints and the native Anthropic messages API.

All functions return None on any failure -- callers degrade gracefully.
Zero external dependencies. stdlib only."""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anamnesis_config


def _log(msg: str):
    print(f"[anamnesis] llm: {msg}", file=sys.stderr)


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        if content.endswith("```"):
            content = content[:-3].rstrip()
    return content.strip()


def _post_json(url: str, payload: dict, headers: dict, timeout: float):
    """POST JSON, return decoded response body or None."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        _log(f"HTTP {e.code}: {e.reason} body={body_text}")
        return None
    except urllib.error.URLError as e:
        _log(f"URL error: {e.reason}")
        return None
    except Exception as e:
        _log(f"request error ({type(e).__name__}): {e}")
        return None


def chat_completion(system_prompt: str, user_prompt: str,
                    purpose: str = "reflection", timeout: float = 30.0,
                    max_tokens: int = 4096) -> str:
    """Call the configured LLM. Returns the text content or None.

    purpose: "reflection" (deep model) or "fast" (cheap model)."""
    cfg = anamnesis_config.llm_config(purpose)
    if not cfg["configured"]:
        _log("no LLM API key configured; set ANAMNESIS_LLM_API_KEY "
             "(or OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        return None

    if cfg["provider"] == "anthropic":
        url = f"{cfg['base_url']}/v1/messages"
        payload = {
            "model": cfg["model"],
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
            "User-Agent": "anamnesis/3.1",
        }
        data = _post_json(url, payload, headers, timeout)
        if data is None:
            return None
        try:
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks
                           if isinstance(b, dict) and b.get("type") == "text")
        except Exception as e:
            _log(f"anthropic response parse error: {e}")
            return None

    # OpenAI-compatible chat completions
    url = f"{cfg['base_url']}/chat/completions"
    payload = {
        "model": cfg["model"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
        "User-Agent": "anamnesis/3.1",
    }
    data = _post_json(url, payload, headers, timeout)
    if data is None:
        return None
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        _log(f"response structure error: {e}")
        return None


def chat_json(system_prompt: str, user_prompt: str,
              purpose: str = "reflection", timeout: float = 30.0,
              max_tokens: int = 4096) -> dict:
    """Call the LLM and parse the response as JSON. Returns dict/list or None."""
    content = chat_completion(system_prompt, user_prompt,
                              purpose=purpose, timeout=timeout,
                              max_tokens=max_tokens)
    if content is None:
        return None
    content = _strip_code_fences(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        _log(f"JSON parse error: {e}; content: {content[:300]}")
        return None
