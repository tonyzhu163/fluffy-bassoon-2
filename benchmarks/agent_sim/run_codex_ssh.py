"""Use the signed-in local Codex client with an allocated remote simulator.

No API key or global client configuration is needed. The remote command must
launch the stdio server with a fresh output directory and qualified GPU runtime.
Keep SSH routes/keys outside this repository. Run this on macOS or Linux;
the simulator runs on the allocated Linux host.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='Existing SSH host/alias, optionally user@host')
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--remote-command', required=True, help='Trusted remote stdio launch command')
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh local client telemetry directory')
    parser.add_argument('--codex', default='codex', help='Codex executable path')
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'high'), default='medium')
    parser.add_argument('--wall-seconds', type=int, default=3600)
    args = parser.parse_args()
    if args.host.startswith('-') or not 1 <= args.port <= 65535 or args.wall_seconds < 1:
        parser.error('Require a valid SSH destination, port and positive wall limit')
    prompt = args.prompt_file.read_text()
    version = subprocess.check_output([args.codex, '--version'], text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'prompt.txt').write_text(prompt)
    metadata = dict(model_requested=args.model, reasoning_effort=args.reasoning_effort,
        client_version=version, transport='existing authenticated SSH stdio',
        model_identity_source='explicit Codex CLI selection; no independent backend attestation',
        start_utc=datetime.now(timezone.utc).isoformat(), max_wall_seconds=args.wall_seconds)
    (output / 'invocation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    # A private empty cwd prevents repository instructions becoming controller input.
    with tempfile.TemporaryDirectory(prefix='robot-controller-') as cwd:
        command = [args.codex, 'exec', '--ignore-user-config', '--ephemeral',
            '--skip-git-repo-check', '--json', '--sandbox', 'read-only', '-C', cwd,
            '-m', args.model, '-c', 'approval_policy="never"',
            '-c', 'model_reasoning_effort=' + json.dumps(args.reasoning_effort),
            '-c', 'web_search="disabled"', '-c', 'features.skip_host_skill_discovery=true']
        for feature in ('shell_tool', 'apps', 'plugins', 'browser_use', 'computer_use',
                        'multi_agent', 'image_generation'):
            command.extend(['--disable', feature])
        settings = {'mcp_servers.robot-sim.command': 'ssh',
            'mcp_servers.robot-sim.args': ['-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                '-p', str(args.port), args.host, args.remote_command],
            'mcp_servers.robot-sim.startup_timeout_sec': 120,
            'mcp_servers.robot-sim.tool_timeout_sec': 240}
        for key, value in settings.items():
            command.extend(['-c', key + '=' + json.dumps(value)])
        command.extend(['-o', str(output / 'final.txt'), '-'])
        with (output / 'events.jsonl').open('x') as events, (output / 'client.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events,
                stderr=log, text=True, start_new_session=True)
            try:
                process.communicate(prompt, timeout=args.wall_seconds)
                metadata['exit_code'] = process.returncode
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                metadata['exit_code'] = 'wall_limit' if isinstance(error, subprocess.TimeoutExpired) else 'interrupted'
                # Terminate only this invocation's process group, including SSH.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            finally:
                metadata.update(end_utc=datetime.now(timezone.utc).isoformat(),
                                wall_seconds=time.monotonic() - started)
                (output / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))
    raise SystemExit(0 if metadata['exit_code'] == 0 else 1)


if __name__ == '__main__':
    main()
