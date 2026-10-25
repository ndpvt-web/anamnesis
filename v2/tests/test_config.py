#!/usr/bin/env python3
"""Tests for anamnesis_config: data home, agent identity, LLM provider."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import anamnesis_config as cfg


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANAMNESIS_HOME", str(tmp_path / ".anamnesis"))
    return tmp_path / ".anamnesis"


# --- data home ---

def test_data_home_env_override(tmp_home):
    assert cfg.data_home() == tmp_home


def test_data_home_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ANAMNESIS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cfg.data_home() == tmp_path / ".anamnesis"


# --- agent identity ---

def test_env_agent_wins(tmp_home, monkeypatch):
    monkeypatch.setenv("ANAMNESIS_AGENT_ID", "explicit-agent")
    assert cfg.resolve_agent_id({"agent_id": "payload", "cwd": "/x"}) == "explicit-agent"


def test_payload_agent_keys(tmp_home, monkeypatch):
    monkeypatch.delenv("ANAMNESIS_AGENT_ID", raising=False)
    for key in ("agent_id", "agentId", "agent", "agent_name"):
        assert cfg.resolve_agent_id({key: "payload-agent"}) == "payload-agent"


def test_config_file_agent(tmp_home, monkeypatch):
    monkeypatch.delenv("ANAMNESIS_AGENT_ID", raising=False)
    cfg.save_config({"agent_id": "configured-agent"})
    assert cfg.resolve_agent_id({}) == "configured-agent"


def test_per_project_identity_from_cwd(tmp_home, monkeypatch):
    monkeypatch.delenv("ANAMNESIS_AGENT_ID", raising=False)
    a = cfg.resolve_agent_id({"cwd": "/home/u/projects/alpha"})
    b = cfg.resolve_agent_id({"cwd": "/home/u/projects/beta"})
    assert a != b, "different projects must not silently share memory"
    assert a == cfg.resolve_agent_id({"cwd": "/home/u/projects/alpha"}), \
        "identity must be stable for the same project"
    assert a.startswith("project-alpha-")


def test_machine_default_persisted(tmp_home, monkeypatch):
    monkeypatch.delenv("ANAMNESIS_AGENT_ID", raising=False)
    first = cfg.resolve_agent_id({})
    assert first
    assert cfg.resolve_agent_id({}) == first, "default must be persisted"


# --- LLM config ---

def _clear_llm_env(monkeypatch):
    for var in ("ANAMNESIS_LLM_API_KEY", "ANAMNESIS_LLM_BASE_URL",
                "ANAMNESIS_LLM_MODEL", "ANAMNESIS_LLM_FAST_MODEL",
                "ANAMNESIS_LLM_PROVIDER", "OPENAI_API_KEY",
                "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_llm_unconfigured(tmp_home, monkeypatch):
    _clear_llm_env(monkeypatch)
    result = cfg.llm_config("reflection")
    assert result["configured"] is False
    assert result["api_key"] == ""


def test_llm_primary_env(tmp_home, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANAMNESIS_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ANAMNESIS_LLM_MODEL", "my-model")
    result = cfg.llm_config("reflection")
    assert result["configured"] is True
    assert result["model"] == "my-model"
    assert result["key_source"] == "ANAMNESIS_LLM_API_KEY"
    assert result["provider"] == "openai"


def test_llm_openai_fallback(tmp_home, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    result = cfg.llm_config("fast")
    assert result["configured"] is True
    assert result["key_source"] == "OPENAI_API_KEY"
    assert result["provider"] == "openai"


def test_llm_anthropic_detection(tmp_home, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("ANAMNESIS_LLM_BASE_URL", "https://api.anthropic.com")
    result = cfg.llm_config("reflection")
    assert result["provider"] == "anthropic"
    assert "claude" in result["model"], "Anthropic provider needs an Anthropic default model"


def test_llm_fast_model_distinct(tmp_home, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANAMNESIS_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ANAMNESIS_LLM_FAST_MODEL", "fast-model")
    monkeypatch.setenv("ANAMNESIS_LLM_MODEL", "deep-model")
    assert cfg.llm_config("fast")["model"] == "fast-model"
    assert cfg.llm_config("reflection")["model"] == "deep-model"


def test_llm_custom_base_url(tmp_home, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANAMNESIS_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ANAMNESIS_LLM_BASE_URL", "https://gateway.example.com/api/v1/")
    result = cfg.llm_config("reflection")
    assert result["base_url"] == "https://gateway.example.com/api/v1"


def test_llm_bare_anthropic_key(tmp_home, monkeypatch):
    """A lone ANTHROPIC_API_KEY selects the Anthropic provider, endpoint,
    and Anthropic default models - no base URL required."""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-only")
    result = cfg.llm_config("reflection")
    assert result["configured"] is True
    assert result["provider"] == "anthropic"
    assert result["key_source"] == "ANTHROPIC_API_KEY"
    assert result["base_url"] == "https://api.anthropic.com"
    assert "claude" in result["model"]


def test_llm_both_keys_prefers_openai(tmp_home, monkeypatch):
    """Ambiguity is documented: with no explicit base/provider, OPENAI_API_KEY
    wins over ANTHROPIC_API_KEY (OpenAI-compatible is the default shape)."""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    result = cfg.llm_config("reflection")
    assert result["provider"] == "openai"
    assert result["key_source"] == "OPENAI_API_KEY"


def test_llm_explicit_base_url_beats_key_ambiguity(tmp_home, monkeypatch):
    """An explicit anthropic base URL selects the Anthropic fallback key even
    when an OpenAI key is also present."""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("ANAMNESIS_LLM_BASE_URL", "https://api.anthropic.com")
    result = cfg.llm_config("reflection")
    assert result["provider"] == "anthropic"
    assert result["key_source"] == "ANTHROPIC_API_KEY"
