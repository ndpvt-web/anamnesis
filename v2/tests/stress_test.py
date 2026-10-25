#!/usr/bin/env python3
"""Real-world hook subprocess stress test.

Runs every hook script as a subprocess (the way Claude Code invokes them)
and checks for correctness, performance, and false positives.
"""
import subprocess, json, time, os, sys

# Repo root by default; override for testing a deployed copy.
HOOK_DIR = os.environ.get(
    'ANAMNESIS_TEST_HOOK_DIR',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Test agent id; hooks resolve identity from the payload cwd otherwise.
AGENT_ID = os.environ.get('ANAMNESIS_TEST_AGENT', 'stress-test-agent')
os.environ['ANAMNESIS_AGENT_ID'] = AGENT_ID
os.environ.setdefault('ANAMNESIS_HOME',
                      os.path.join(os.path.expanduser('~'), '.anamnesis-test'))
outcomes = []


def run_hook(script, input_data, label):
    start = time.monotonic()
    try:
        proc = subprocess.run(
            ['python3', os.path.join(HOOK_DIR, script)],
            input=json.dumps(input_data),
            capture_output=True, text=True, timeout=15
        )
        elapsed = (time.monotonic() - start) * 1000
        ok = proc.returncode == 0
        out = proc.stdout.strip()[:2000] if proc.stdout else ''
        err = proc.stderr.strip()[:500] if proc.stderr else ''
        outcomes.append((label, ok, elapsed, out, err))
        return ok, out, err
    except subprocess.TimeoutExpired:
        elapsed = (time.monotonic() - start) * 1000
        outcomes.append((label, False, elapsed, '', 'TIMEOUT'))
        return False, '', 'TIMEOUT'


def main():
    ts = int(time.time())

    # --- 1: Session Start ---
    print('=== 1: session_start ===')
    run_hook('on_session_start.py', {
        'session_id': f'stress-{ts}',
        'agent_id': AGENT_ID,
        'cwd': '/tmp/test'
    }, 'session_start')

    # --- 2: Prompt Submit (knowledge retrieval) ---
    print('=== 2: prompt_submit ===')
    prompts = [
        'How do I set up a Python virtual environment?',
        'What is the best way to handle errors in async code?',
        'Deploy a React app to production',
        'Configure nginx reverse proxy',
        'Database migration strategy for PostgreSQL',
    ]
    for i, prompt in enumerate(prompts):
        run_hook('on_prompt_submit.py', {
            'session_id': f'stress-{ts}',
            'agent_id': AGENT_ID,
            'prompt': prompt
        }, f'prompt_submit_{i}')

    # --- 3: Post Tool Use ---
    print('=== 3: post_tool_use ===')
    tools = [
        ('Read', 'success', '/tmp/test.py', 'file contents here'),
        ('Write', 'success', '/tmp/output.py', 'wrote 50 lines'),
        ('Bash', 'success', 'npm install', 'added 200 packages'),
        ('Bash', 'error', 'npm test', 'FAIL: 3 tests failed'),
        ('Read', 'error', '/nonexistent', 'FileNotFoundError'),
    ]
    for tool_name, status, tool_input, output in tools:
        run_hook('on_post_tool.py', {
            'session_id': f'stress-{ts}',
            'agent_id': AGENT_ID,
            'tool_name': tool_name,
            'tool_input': tool_input,
            'tool_output': output,
            'tool_status': status
        }, f'post_tool_{tool_name}_{status}')

    # --- 4: Bash Guardian ---
    print('=== 4: bash_guardian ===')
    commands = [
        ('ls -la', True),
        ('git status', True),
        ('curl https://api.example.com/data', True),
        ('npm install express', True),
        ('pip list', True),
        ('docker ps', True),
        ('cat /etc/hosts', True),
        ('rm -rf /var/data/production', False),
        ('git push --force origin main', False),
        ('DROP DATABASE users;', False),
    ]
    for cmd, expect_safe in commands:
        ok, out, err = run_hook('on_pre_bash.py', {
            'tool_input': {'command': cmd}
        }, f'guardian_{cmd[:30]}')
        try:
            r = json.loads(out) if out else {}
            # Check if hook output contains a deny decision
            hook_out = r.get('hookSpecificOutput', {})
            blocked = hook_out.get('permissionDecision') == 'deny'
            if expect_safe and blocked:
                print(f'  FALSE POSITIVE: {cmd} blocked! reason={r.get("reason","")}')
            elif not expect_safe and not blocked:
                print(f'  MISSED DANGER: {cmd} not blocked!')
            else:
                tag = 'allowed' if expect_safe else 'blocked'
                print(f'  OK ({tag}): {cmd}')
        except json.JSONDecodeError:
            print(f'  PARSE ERR: {cmd}: {out[:100]}')

    # --- 5: Rapid-fire sequential stress ---
    print('=== 5: rapid fire (10 calls) ===')
    for i in range(10):
        run_hook('on_prompt_submit.py', {
            'session_id': f'stress-rapid-{ts}',
            'agent_id': AGENT_ID,
            'prompt': f'iteration {i}: debug memory leaks in Node.js'
        }, f'rapid_{i}')

    # --- Summary ---
    print()
    print('=' * 60)
    total = len(outcomes)
    ok_count = sum(1 for _, ok, *_ in outcomes if ok)
    fail_count = total - ok_count
    print(f'TOTAL: {total}  OK: {ok_count}  FAIL: {fail_count}')
    print('=' * 60)
    for label, ok, elapsed, out, err in outcomes:
        tag = 'OK' if ok else 'FAIL'
        line = f'  {tag} {label:35s} {elapsed:7.1f}ms'
        if not ok:
            line += f'  err={err[:80]}'
        print(line)

    times = [e for _, _, e, _, _ in outcomes]
    print()
    print(f'Latency: min={min(times):.1f}ms  avg={sum(times)/len(times):.1f}ms  max={max(times):.1f}ms')
    print(f'Total wall time: {sum(times):.0f}ms')

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
