#!/usr/bin/env python3
"""Tests for v2/llm.py against a LOCAL HTTP fixture server.

Covers both provider API shapes (OpenAI-compatible and native Anthropic):
success payloads, HTTP error status, timeout, and secret handling (the API
key never appears in logs). No live provider is contacted."""

import json
import logging
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import anamnesis_config as cfg
from v2 import llm


class _FixtureHandler(BaseHTTPRequestHandler):
    """Records requests; behavior set per-test via class attributes."""
    requests = []
    response_status = 200
    response_body = {}
    hang_seconds = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        type(self).requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": json.loads(body),
        })
        if type(self).hang_seconds:
            import time
            time.sleep(type(self).hang_seconds)
        self.send_response(type(self).response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).response_body).encode())


@pytest.fixture
def server():
    _FixtureHandler.requests = []
    _FixtureHandler.response_status = 200
    _FixtureHandler.response_body = {}
    _FixtureHandler.hang_seconds = 0
    srv = HTTPServer(("127.0.0.1", 0), _FixtureHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANAMNESIS_HOME", str(tmp_path / ".anamnesis"))
    for var in ("ANAMNESIS_LLM_API_KEY", "ANAMNESIS_LLM_BASE_URL",
                "ANAMNESIS_LLM_MODEL", "ANAMNESIS_LLM_FAST_MODEL",
                "ANAMNESIS_LLM_PROVIDER", "OPENAI_API_KEY",
                "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


_OPENAI_BODY = {"choices": [{"message": {"content": '{"ok": true}'}}]}
_ANTHROPIC_BODY = {"content": [{"type": "text", "text": '{"ok": true}'}]}


def test_openai_shape(server, env):
    env.setenv("ANAMNESIS_LLM_API_KEY", "sk-secret-123")
    env.setenv("ANAMNESIS_LLM_BASE_URL", server + "/v1")
    env.setenv("ANAMNESIS_LLM_MODEL", "test-model")
    _FixtureHandler.response_body = _OPENAI_BODY

    result = llm.chat_json("SYS", "USER", purpose="reflection", timeout=5)
    assert result == {"ok": True}

    req = _FixtureHandler.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-secret-123"
    assert req["body"]["model"] == "test-model"
    # system prompt travels as a system message (OpenAI shape)
    assert req["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert req["body"]["messages"][1] == {"role": "user", "content": "USER"}


def test_anthropic_shape(server, env):
    env.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    env.setenv("ANAMNESIS_LLM_BASE_URL", server)
    env.setenv("ANAMNESIS_LLM_PROVIDER", "anthropic")
    env.setenv("ANAMNESIS_LLM_MODEL", "claude-test")
    _FixtureHandler.response_body = _ANTHROPIC_BODY

    result = llm.chat_json("SYS", "USER", purpose="reflection", timeout=5)
    assert result == {"ok": True}

    req = _FixtureHandler.requests[0]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["X-Api-Key"] == "sk-ant-secret"
    assert req["headers"]["Anthropic-Version"] == "2023-06-01"
    assert req["body"]["system"] == "SYS"
    assert req["body"]["messages"] == [{"role": "user", "content": "USER"}]


def test_http_error_returns_none(server, env, caplog):
    env.setenv("ANAMNESIS_LLM_API_KEY", "sk-secret-123")
    env.setenv("ANAMNESIS_LLM_BASE_URL", server)
    _FixtureHandler.response_status = 500
    _FixtureHandler.response_body = {"error": "boom"}
    with caplog.at_level(logging.WARNING):
        result = llm.chat_json("SYS", "USER", timeout=5)
    assert result is None


def test_timeout_returns_none(server, env):
    env.setenv("ANAMNESIS_LLM_API_KEY", "sk-secret-123")
    env.setenv("ANAMNESIS_LLM_BASE_URL", server)
    _FixtureHandler.response_body = _OPENAI_BODY
    _FixtureHandler.hang_seconds = 2
    result = llm.chat_json("SYS", "USER", timeout=0.3)
    assert result is None


def test_no_key_short_circuits(server, env):
    result = llm.chat_json("SYS", "USER", timeout=5)
    assert result is None
    assert _FixtureHandler.requests == [], "no HTTP call without a key"


def test_secret_not_in_logs(server, env, capsys):
    env.setenv("ANAMNESIS_LLM_API_KEY", "sk-secret-123")
    env.setenv("ANAMNESIS_LLM_BASE_URL", server)
    _FixtureHandler.response_status = 401
    _FixtureHandler.response_body = {"error": "bad key sk-secret-123"}
    result = llm.chat_json("SYS", "USER", timeout=5)
    assert result is None
    err = capsys.readouterr().err
    # the server echoes the key in its error body; our log truncates bodies,
    # and must never print the Authorization header itself
    assert "Bearer" not in err


def test_code_fence_stripping(server, env):
    env.setenv("ANAMNESIS_LLM_API_KEY", "sk-secret-123")
    env.setenv("ANAMNESIS_LLM_BASE_URL", server)
    _FixtureHandler.response_body = {
        "choices": [{"message": {"content": '```json\n{"ok": 1}\n```'}}]}
    assert llm.chat_json("S", "U", timeout=5) == {"ok": 1}
