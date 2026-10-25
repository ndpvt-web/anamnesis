#!/usr/bin/env python3
"""Anamnesis installer and doctor.

Plug-and-play setup for any AI agent harness:

  python3 install.py                  # create data home + config, print next steps
  python3 install.py --claude-code    # also register hooks + MCP in Claude Code
  python3 install.py --check          # run diagnostics (doctor)
  python3 install.py --uninstall      # remove Anamnesis' own configuration

Only ever writes:
  - <data_home>/config.json            (non-secret settings)
  - ~/.claude/settings.json            (only with --claude-code; timestamped
                                        backup first; unrelated keys preserved)
API keys are NEVER written to disk; the installer prints the environment
variables to export instead.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_config

REPO_ROOT = Path(__file__).resolve().parent
_HOOKS = [
    ("PreToolUse", "Bash", "on_pre_bash.py"),
    ("PostToolUse", "", "on_post_tool.py"),
    ("UserPromptSubmit", "", "on_prompt_submit.py"),
    ("SessionStart", "", "on_session_start.py"),
    ("Stop", "", "agent_learn.py"),
]
_MCP_SERVER_KEY = "anamnesis"


# ---------------------------------------------------------------------------
# Configuration snippets
# ---------------------------------------------------------------------------

def hook_command(script: str) -> str:
    return f"python3 {REPO_ROOT / script}"


def claude_hooks_config() -> dict:
    """Claude Code settings.json 'hooks' block for Anamnesis."""
    hooks = {}
    for event, matcher, script in _HOOKS:
        entry = {"matcher": matcher,
                 "hooks": [{"type": "command", "command": hook_command(script)}]}
        hooks.setdefault(event, []).append(entry)
    return hooks


def claude_mcp_config() -> dict:
    """Claude Code settings.json 'mcpServers' entry for Anamnesis."""
    return {_MCP_SERVER_KEY: {
        "type": "stdio",
        "command": "python3",
        "args": [str(REPO_ROOT / "v2" / "mcp_server.py")],
    }}


def generic_instructions() -> str:
    lines = [
        "Generic harness integration",
        "===========================",
        "",
        "1) Shell hooks (any harness that can run a command on lifecycle events).",
        "   Each script reads a JSON event on stdin and may emit a JSON response",
        "   on stdout. See docs/INTEGRATIONS.md for the exact shapes.",
        "",
    ]
    for event, matcher, script in _HOOKS:
        lines.append(f"   {event:16s} -> {hook_command(script)}")
    lines += [
        "",
        "2) MCP (any MCP-capable client):",
        "",
        "   " + json.dumps({"mcpServers": claude_mcp_config()}, indent=2),
        "",
        "3) Shell CLI (lowest common denominator):",
        "",
        f"   python3 {REPO_ROOT / 'anamnesis_cli.py'} briefing|record|status|doctor",
        "",
        "Environment:",
        "  ANAMNESIS_HOME          data directory (default ~/.anamnesis)",
        "  ANAMNESIS_AGENT_ID      explicit agent identity (default: per-project)",
        "  ANAMNESIS_LLM_API_KEY   enables reflection + extraction (any",
        "                          OpenAI-compatible endpoint or Anthropic)",
        "  ANAMNESIS_LLM_BASE_URL  custom endpoint (default api.openai.com)",
        "  ANAMNESIS_LLM_MODEL / ANAMNESIS_LLM_FAST_MODEL  model overrides",
        "  TINYFISH_API_KEY        optional web-search enrichment",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

def write_config(agent_id: str = None) -> Path:
    cfg = anamnesis_config.load_config()
    if agent_id:
        cfg["agent_id"] = agent_id
    cfg.setdefault("version", "3.1")
    anamnesis_config.save_config(cfg)
    return anamnesis_config.config_path()


def install_claude_code(settings_path: Path = None) -> Path:
    """Merge Anamnesis hooks + MCP server into Claude Code settings.

    Backs up the existing file first; preserves all unrelated keys."""
    settings_path = settings_path or (Path.home() / ".claude" / "settings.json")
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            if not isinstance(settings, dict):
                settings = {}
        except json.JSONDecodeError:
            backup = settings_path.with_suffix(
                f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            shutil.copyfile(str(settings_path), str(backup))
            print(f"Existing settings were not valid JSON; backed up to {backup}")
            settings = {}
    if settings_path.exists():
        backup = settings_path.with_suffix(
            f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copyfile(str(settings_path), str(backup))
        print(f"Backed up existing settings to {backup}")

    hooks = settings.setdefault("hooks", {})
    for event, entries in claude_hooks_config().items():
        existing = hooks.setdefault(event, [])
        # Remove any previous Anamnesis entries (idempotent reinstall), keep others
        ours = {e["hooks"][0]["command"] for e in entries}
        kept = [e for e in existing
                if not any(h.get("command", "") in ours
                           or str(REPO_ROOT) in h.get("command", "")
                           for h in e.get("hooks", []))]
        kept.extend(entries)
        hooks[event] = kept

    mcp = settings.setdefault("mcpServers", {})
    mcp.update(claude_mcp_config())

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return settings_path


def uninstall_claude_code(settings_path: Path = None) -> bool:
    """Remove only Anamnesis' own entries from Claude Code settings."""
    settings_path = settings_path or (Path.home() / ".claude" / "settings.json")
    if not settings_path.exists():
        return False
    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        return False
    changed = False
    hooks = settings.get("hooks", {})
    for event in list(hooks):
        kept = [e for e in hooks[event]
                if not any(str(REPO_ROOT) in h.get("command", "")
                           for h in e.get("hooks", []))]
        if kept != hooks[event]:
            changed = True
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    mcp = settings.get("mcpServers", {})
    if _MCP_SERVER_KEY in mcp:
        mcp.pop(_MCP_SERVER_KEY)
        changed = True
    if changed:
        backup = settings_path.with_suffix(
            f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copyfile(str(settings_path), str(backup))
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Removed Anamnesis entries (backup at {backup})")
    return changed


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------

def _check_ft5() -> bool:
    import sqlite3
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        db.close()


def doctor(agent: str = None, cwd: str = None, quiet: bool = False) -> int:
    """Run diagnostics. Returns 0 when healthy, 1 otherwise."""
    failures = []

    def report(ok, label, detail=""):
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    print("Anamnesis doctor")
    print("----------------")

    # 1. Data home
    home = anamnesis_config.data_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor-probe"
        probe.write_text("ok")
        probe.unlink()
        report(True, "data home writable", str(home))
    except Exception as e:
        report(False, "data home writable", f"{home}: {e}")

    # 2. FTS5
    report(_check_ft5(), "SQLite FTS5 available")

    # 3. Agent identity + source
    env_id = os.environ.get("ANAMNESIS_AGENT_ID", "").strip()
    cfg_id = str(anamnesis_config.load_config().get("agent_id", "")).strip()
    hook_input = {"cwd": cwd} if cwd else {}
    resolved = agent or anamnesis_config.resolve_agent_id(hook_input)
    if agent:
        source = "--agent flag"
    elif env_id:
        source = "ANAMNESIS_AGENT_ID env"
    elif cfg_id:
        source = "config.json"
    elif cwd:
        source = f"per-project (cwd={cwd})"
    else:
        source = "machine default"
    report(True, "agent identity", f"{resolved} (source: {source})")

    # 4. LLM configuration
    for purpose in ("reflection", "fast"):
        llm = anamnesis_config.llm_config(purpose)
        if llm["configured"]:
            report(True, f"LLM ({purpose})",
                   f"provider={llm['provider']} model={llm['model']} "
                   f"key from {llm['key_source']}")
        else:
            report(True, f"LLM ({purpose})",
                   "not configured - reflection/extraction disabled; "
                   "retrieval and observation still work "
                   "(set ANAMNESIS_LLM_API_KEY)")

    # 5. Tinyfish (optional)
    if anamnesis_config.tinyfish_api_key():
        report(True, "Tinyfish web enrichment", "key present")
    else:
        report(True, "Tinyfish web enrichment", "not configured (optional)")

    # 6. Hook smoke tests in an isolated HOME
    print("  hook smoke tests (isolated sandbox):")
    with tempfile.TemporaryDirectory(prefix="anamnesis-doctor-") as tmp:
        env = dict(os.environ, ANAMNESIS_HOME=tmp)
        smoke = [
            ("on_session_start.py", {"session_id": "doctor", "cwd": tmp}),
            ("on_prompt_submit.py", {"session_id": "doctor", "cwd": tmp,
                                     "prompt": "how do I configure sqlite wal mode?"}),
            ("on_pre_bash.py", {"session_id": "doctor", "cwd": tmp,
                                "tool_input": {"command": "git status"}}),
            ("on_post_tool.py", {"session_id": "doctor", "cwd": tmp,
                                 "tool_name": "Bash",
                                 "tool_input": {"command": "ls"},
                                 "tool_response": "ok"}),
            ("on_pre_bash.py", {"session_id": "doctor", "cwd": tmp,
                                "tool_input": {"command": "rm -rf /important"}}),
        ]
        for script, payload in smoke:
            try:
                proc = subprocess.run(
                    [sys.executable, str(REPO_ROOT / script)],
                    input=json.dumps(payload), capture_output=True, text=True,
                    timeout=15, env=env)
                ok = proc.returncode == 0
                detail = ""
                if script == "on_pre_bash.py" and "rm -rf" in json.dumps(payload):
                    out = json.loads(proc.stdout) if proc.stdout.strip() else {}
                    blocked = (out.get("hookSpecificOutput", {})
                               .get("permissionDecision") == "deny")
                    ok = ok and blocked
                    detail = "safety axiom blocked rm -rf" if blocked else "DID NOT BLOCK rm -rf"
                report(ok, f"smoke {script}", detail)
            except Exception as e:
                report(False, f"smoke {script}", str(e))

    # 7. MCP handshake
    try:
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "v2" / "mcp_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            env=dict(os.environ,
                     ANAMNESIS_HOME=tempfile.mkdtemp(prefix="anamnesis-mcp-")))
        init = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {}}) + "\n"
        tools = json.dumps({"jsonrpc": "2.0", "id": 2,
                            "method": "tools/list", "params": {}}) + "\n"
        out, _ = proc.communicate(init + tools, timeout=15)
        responses = [json.loads(line) for line in out.strip().splitlines() if line.strip()]
        names = [t["name"] for r in responses
                 for t in r.get("result", {}).get("tools", [])]
        ok = (responses and responses[0].get("result", {})
              .get("serverInfo", {}).get("name") == "anamnesis"
              and len(names) == 5)
        report(ok, "MCP handshake", f"tools: {', '.join(names)}" if ok else out[:200])
    except Exception as e:
        report(False, "MCP handshake", str(e))

    print()
    if failures:
        print(f"doctor: {len(failures)} problem(s) found")
        return 1
    print("doctor: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Anamnesis installer")
    parser.add_argument("--agent-id", help="Pin a global agent identity "
                        "(default: per-project)")
    parser.add_argument("--claude-code", action="store_true",
                        help="Register hooks + MCP in ~/.claude/settings.json")
    parser.add_argument("--check", "--doctor", dest="check",
                        action="store_true", help="Run diagnostics only")
    parser.add_argument("--uninstall", action="store_true",
                        help="Remove Anamnesis' own configuration")
    args = parser.parse_args()

    if args.check:
        return doctor()

    if args.uninstall:
        removed_cfg = False
        cfg = anamnesis_config.config_path()
        if cfg.exists():
            cfg.unlink()
            removed_cfg = True
            print(f"Removed {cfg}")
        removed_cc = uninstall_claude_code()
        if not removed_cfg and not removed_cc:
            print("Nothing to remove.")
        print("Data under " + str(anamnesis_config.data_home()) +
              " was left untouched.")
        return 0

    config_path = write_config(args.agent_id)
    print(f"Wrote {config_path}")
    anamnesis_config.agents_dir().mkdir(parents=True, exist_ok=True)

    if args.claude_code:
        path = install_claude_code()
        print(f"Registered hooks + MCP server in {path}")

    print()
    print(generic_instructions())
    print()
    print("Next: run `python3 install.py --check` to verify the setup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
