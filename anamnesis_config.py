#!/usr/bin/env python3
"""Anamnesis configuration: data home, agent identity, LLM provider.

Single source of truth for every path and provider setting. All modules
import from here instead of hardcoding paths, URLs, model IDs, or env vars.

Resolution order (highest priority first):
  1. Environment variables (ANAMNESIS_*)
  2. Config file: <data_home>/config.json
  3. Well-known fallbacks (OPENAI_API_KEY, OPENAI_BASE_URL, ANTHROPIC_API_KEY)
  4. Built-in defaults

Zero external dependencies. stdlib only."""

import json
import os
import re
import socket
import getpass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data home
# ---------------------------------------------------------------------------

DEFAULT_HOME_NAME = ".anamnesis"
CONFIG_FILENAME = "config.json"


def data_home() -> Path:
    """Root directory for all Anamnesis data.

    $ANAMNESIS_HOME wins; otherwise ~/.anamnesis."""
    env = os.environ.get("ANAMNESIS_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / DEFAULT_HOME_NAME


def agents_dir() -> Path:
    """Directory holding one subdirectory per agent."""
    return data_home() / "agents"


def config_path() -> Path:
    return data_home() / CONFIG_FILENAME


def load_config() -> dict:
    """Load the config file. Returns {} when absent or unreadable."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(cfg: dict):
    """Write the config file (non-secret settings only)."""
    home = data_home()
    home.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(cfg, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------

# Hook payloads from different harnesses use different key names for the
# calling agent's identity. Checked in order.
_HOOK_AGENT_KEYS = ("agent_id", "agentId", "agent", "agent_name", "agentName")


def _default_agent_file() -> Path:
    return data_home() / "default_agent"


def _provision_default_agent() -> str:
    """Create and persist a stable machine-level default agent identity.

    Last-resort fallback when no project context is available either."""
    host = socket.gethostname().split(".")[0] or "host"
    try:
        user = getpass.getuser() or "user"
    except Exception:
        user = "user"
    agent_id = f"default-{user}@{host}"
    try:
        data_home().mkdir(parents=True, exist_ok=True)
        _default_agent_file().write_text(agent_id + "\n")
    except Exception:
        pass
    return agent_id


def project_agent_id(cwd: str) -> str:
    """Derive a stable per-project agent identity from a working directory.

    Deterministic: the same cwd always yields the same id, so memory stays
    isolated per project by default (no silent cross-project sharing).
    Set ANAMNESIS_AGENT_ID to share memory across projects instead."""
    import hashlib
    norm = str(Path(cwd).expanduser())
    base = Path(norm).name or "root"
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "project"
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    return f"project-{slug}-{digest}"


def _hook_cwd(hook_input: dict) -> str:
    if not isinstance(hook_input, dict):
        return ""
    for key in ("cwd", "workspace", "project_dir", "projectDir", "root"):
        value = hook_input.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_agent_id(hook_input: dict = None) -> str:
    """Resolve the active agent's identity. Never returns an empty string.

    Order:
      1. ANAMNESIS_AGENT_ID env var (explicit; use it to share memory
         across projects or to run several agents on one machine)
      2. Agent id field in the hook payload (harness-provided)
      3. agent_id in <data_home>/config.json
      4. Per-project identity derived from the hook payload's cwd
      5. Persisted machine-level default (auto-provisioned on first use)
    """
    env_agent = os.environ.get("ANAMNESIS_AGENT_ID", "").strip()
    if env_agent:
        return env_agent

    if isinstance(hook_input, dict):
        for key in _HOOK_AGENT_KEYS:
            value = hook_input.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()

    cfg_agent = str(load_config().get("agent_id", "")).strip()
    if cfg_agent:
        return cfg_agent

    cwd = _hook_cwd(hook_input)
    if cwd:
        return project_agent_id(cwd)

    path = _default_agent_file()
    try:
        if path.exists():
            saved = path.read_text().strip()
            if saved:
                return saved
    except Exception:
        pass
    return _provision_default_agent()


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------

# Purposes: "reflection" (deep, higher quality) and "fast" (cheap extraction).
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_FAST_MODEL = "gpt-4o-mini"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5"
ANTHROPIC_DEFAULT_FAST_MODEL = "claude-haiku-4-5"

_PURPOSE_ENV = {
    "reflection": "ANAMNESIS_LLM_MODEL",
    "fast": "ANAMNESIS_LLM_FAST_MODEL",
}


def _detect_provider(base_url: str) -> str:
    """Return 'anthropic' or 'openai' (OpenAI-compatible chat completions)."""
    explicit = os.environ.get("ANAMNESIS_LLM_PROVIDER", "").strip().lower()
    if explicit in ("anthropic", "openai"):
        return explicit
    if "anthropic.com" in (base_url or ""):
        return "anthropic"
    return "openai"


def llm_config(purpose: str = "reflection") -> dict:
    """Resolve LLM settings for a purpose ("reflection" or "fast").

    Returns a dict:
      {"api_key": str, "base_url": str, "model": str, "provider": str,
       "configured": bool}
    configured is False when no API key is available; callers must treat
    that as "LLM features disabled" and degrade gracefully.
    """
    cfg = load_config().get("llm", {})
    if not isinstance(cfg, dict):
        cfg = {}

    base_url = (
        os.environ.get("ANAMNESIS_LLM_BASE_URL", "").strip()
        or str(cfg.get("base_url", "")).strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
    )
    explicit_provider = os.environ.get(
        "ANAMNESIS_LLM_PROVIDER", "").strip().lower()

    api_key = os.environ.get("ANAMNESIS_LLM_API_KEY", "").strip()
    key_source = "ANAMNESIS_LLM_API_KEY" if api_key else ""
    if not api_key:
        api_key = str(cfg.get("api_key", "")).strip()
        key_source = "config.json llm.api_key" if api_key else ""

    if base_url or explicit_provider:
        # Explicit endpoint/provider decides which fallback key applies.
        provider_hint = _detect_provider(base_url)
        if not api_key:
            if provider_hint == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
                key_source = "ANTHROPIC_API_KEY" if api_key else ""
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "").strip()
                key_source = "OPENAI_API_KEY" if api_key else ""
        provider = provider_hint
    elif not api_key:
        # No explicit endpoint/provider: fall back to whichever well-known
        # key exists. OpenAI wins when both are set (documented); a lone
        # ANTHROPIC_API_KEY selects the Anthropic provider and its defaults.
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if openai_key:
            api_key, key_source, provider = openai_key, "OPENAI_API_KEY", "openai"
        elif anthropic_key:
            api_key, key_source, provider = (
                anthropic_key, "ANTHROPIC_API_KEY", "anthropic")
        else:
            provider = "openai"
    else:
        # Primary key with no endpoint hint: OpenAI-compatible is the most
        # common shape (and the shape every historical deployment used).
        provider = "openai"

    if not base_url:
        base_url = (ANTHROPIC_BASE_URL if provider == "anthropic"
                    else DEFAULT_BASE_URL)

    model_env = _PURPOSE_ENV.get(purpose, "ANAMNESIS_LLM_MODEL")
    model = (
        os.environ.get(model_env, "").strip()
        or os.environ.get("ANAMNESIS_LLM_MODEL", "").strip()
        or str(cfg.get(f"{purpose}_model", "")).strip()
        or str(cfg.get("model", "")).strip()
    )
    if not model:
        if provider == "anthropic":
            model = (ANTHROPIC_DEFAULT_FAST_MODEL if purpose == "fast"
                     else ANTHROPIC_DEFAULT_MODEL)
        else:
            model = (DEFAULT_FAST_MODEL if purpose == "fast"
                     else DEFAULT_MODEL)

    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "provider": provider,
        "key_source": key_source,
        "configured": bool(api_key),
    }


def tinyfish_api_key() -> str:
    """Optional Tinyfish web-search key (third-party, off when unset)."""
    return os.environ.get("TINYFISH_API_KEY", "").strip()
