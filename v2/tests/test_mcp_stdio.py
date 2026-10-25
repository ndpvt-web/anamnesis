#!/usr/bin/env python3
"""MCP server protocol tests over real stdio pipes."""

import json
import subprocess
import sys
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def mcp(tmp_path):
    env = dict(os.environ, ANAMNESIS_HOME=str(tmp_path / "data"))
    env.pop("ANAMNESIS_AGENT_ID", None)
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "v2" / "mcp_server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, env=env)
    yield proc
    proc.kill()


def _rpc(proc, messages):
    out, _ = proc.communicate(
        "".join(json.dumps(m) + "\n" for m in messages), timeout=20)
    return [json.loads(line) for line in out.strip().splitlines() if line.strip()]


def test_initialize_and_tools_list(mcp):
    responses = _rpc(mcp, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ])
    assert responses[0]["result"]["serverInfo"]["name"] == "anamnesis"
    names = [t["name"] for t in responses[1]["result"]["tools"]]
    assert names == ["anamnesis_recall", "anamnesis_record",
                     "anamnesis_feedback", "anamnesis_status",
                     "anamnesis_graph"]


def test_record_recall_status_roundtrip(mcp):
    responses = _rpc(mcp, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "anamnesis_record", "arguments": {
             "content": "WHEN using Flask on macOS THEN port 5000 conflict "
                        "BECAUSE AirDrop receiver occupies it. FIX: use 5001",
             "category": "error_fix", "severity": "high"}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "anamnesis_recall", "arguments": {
             "query": "Flask port 5000 AirDrop conflict"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "anamnesis_status", "arguments": {}}},
    ])
    assert "Recorded learning" in responses[1]["result"]["content"][0]["text"]
    assert "AirDrop" in responses[2]["result"]["content"][0]["text"]
    assert "Total active:" in responses[3]["result"]["content"][0]["text"]


def test_unknown_tool_errors_cleanly(mcp):
    responses = _rpc(mcp, [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}},
    ])
    assert responses[0]["result"]["isError"] is True


def test_unknown_method_jsonrpc_error(mcp):
    responses = _rpc(mcp, [
        {"jsonrpc": "2.0", "id": 1, "method": "bogus/method", "params": {}},
    ])
    assert responses[0]["error"]["code"] == -32601
