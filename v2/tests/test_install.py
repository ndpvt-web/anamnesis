#!/usr/bin/env python3
"""Tests for install.py: config writing, Claude Code settings merge,
uninstall scoping, and doctor."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
import install as install_mod
import anamnesis_config


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANAMNESIS_HOME", str(tmp_path / ".anamnesis"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANAMNESIS_AGENT_ID", raising=False)
    return tmp_path


def test_write_config_non_secret(env):
    path = install_mod.write_config("my-agent")
    cfg = json.loads(path.read_text())
    assert cfg["agent_id"] == "my-agent"
    assert "api_key" not in json.dumps(cfg).lower(), "config must never store secrets"


def test_claude_merge_preserves_unrelated(env):
    settings_path = env / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({
        "theme": "dark",
        "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [
            {"type": "command", "command": "python3 /other/tool.py"}]}]},
        "mcpServers": {"other-server": {"type": "stdio", "command": "x"}},
    }))
    install_mod.install_claude_code(settings_path)
    merged = json.loads(settings_path.read_text())
    assert merged["theme"] == "dark", "unrelated settings must be preserved"
    assert "other-server" in merged["mcpServers"]
    assert "anamnesis" in merged["mcpServers"]
    commands = [h["command"]
                for e in merged["hooks"]["PreToolUse"] for h in e["hooks"]]
    assert "python3 /other/tool.py" in commands, "other hooks preserved"
    assert any("on_pre_bash.py" in c for c in commands)
    # backup written
    assert list(settings_path.parent.glob("settings.json.bak-*"))


def test_claude_reinstall_idempotent(env):
    settings_path = env / ".claude" / "settings.json"
    install_mod.install_claude_code(settings_path)
    install_mod.install_claude_code(settings_path)
    merged = json.loads(settings_path.read_text())
    ours = [h["command"] for e in merged["hooks"]["PreToolUse"]
            for h in e["hooks"] if "on_pre_bash.py" in h.get("command", "")]
    assert len(ours) == 1, f"reinstall duplicated hooks: {ours}"


def test_uninstall_removes_only_own(env):
    settings_path = env / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({
        "theme": "dark",
        "hooks": {"PreToolUse": [
            {"matcher": "Bash", "hooks": [
                {"type": "command", "command": "python3 /other/tool.py"}]},
            {"matcher": "Bash", "hooks": [
                {"type": "command",
                 "command": f"python3 {REPO}/on_pre_bash.py"}]},
        ]},
        "mcpServers": {"anamnesis": {"type": "stdio"}, "other": {}},
    }))
    assert install_mod.uninstall_claude_code(settings_path) is True
    result = json.loads(settings_path.read_text())
    assert result["theme"] == "dark"
    assert "anamnesis" not in result["mcpServers"]
    assert "other" in result["mcpServers"]
    remaining = [h["command"] for e in result["hooks"]["PreToolUse"]
                 for h in e["hooks"]]
    assert remaining == ["python3 /other/tool.py"]


def test_doctor_passes_on_clean_home(env):
    assert install_mod.doctor() == 0


def test_doctor_cli_entrypoint(env):
    proc = subprocess.run(
        [sys.executable, str(REPO / "install.py"), "--check"],
        capture_output=True, text=True, timeout=60,
        env={"ANAMNESIS_HOME": str(env / ".anamnesis"),
             "HOME": str(env), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all checks passed" in proc.stdout
