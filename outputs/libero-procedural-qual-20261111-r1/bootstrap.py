"""Deploy pinned Git objects during the notebook outage; inspect then prepare."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

BASE = '99583d3763309f13e041df379ae9ae6a9ec48d65'
OMITTED = {'ae19ddac71bf05df3fd87ca19a35089bd27175ae':
           'outputs/libero-gen-prompt-repair-20261111-r1/teacher01-failed.png'}
RUN = 'libero-procedural-qual-20261111-r1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--identity', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[a-z0-9-]+', args.identity)
    assert re.fullmatch('[0-9a-f]{40}', args.commit)
    assert re.fullmatch('[0-9a-f]{40}', args.base)
    root = args.project / 'outputs' / RUN / ('bootstrap-' + args.identity)
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = time.monotonic() + 1740
    code, phases = 1, []
    store = args.project / 'sources/libero-gen-store-20261111-r1.git'
    checkout = args.project / 'sources' / ('libero-procedural-' + args.commit[:8] + '-' + args.identity)

    def git(where, *arguments):
        return subprocess.check_output(['git', '-C', str(where), *map(str, arguments)], text=True).strip()

    def execute(name, command):
        beginning = time.time()
        log = root / (name + '.log')
        offset, timed_out = 0, False
        with log.open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    with log.open() as stream:
                        stream.seek(offset); chunk = stream.read(); offset = stream.tell()
                    if chunk:
                        print(chunk, end='', flush=True)
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
        with log.open() as stream:
            stream.seek(offset)
            print(stream.read(), end='', flush=True)
        rc = 124 if timed_out else process.returncode
        phases.append({'phase': name, 'started': beginning, 'ended': time.time(), 'exit_code': rc})
        print('QUAL_PHASE', json.dumps(phases[-1]), flush=True)
        return rc

    try:
        data = args.bundle.read_bytes()
        assert hashlib.sha256(data).hexdigest() == args.bundle_sha256
        header, pack = data.split(b'\n\n', 1)
        lines = header.decode('ascii').splitlines()
        version = lines.pop(0)
        assert version in ('# v2 git bundle', '# v3 git bundle') and pack.startswith(b'PACK')
        caps = {line for line in lines if line.startswith('@')}
        assert caps == ({'@object-format=sha1', '@filter=blob:limit=102400'} if version == '# v3 git bundle' else set())
        prerequisites = [line[1:].split(' ', 1)[0] for line in lines if line.startswith('-')]
        references = [line for line in lines if not line.startswith(('@', '-'))]
        assert prerequisites == [args.base] and references == [args.commit + ' refs/heads/main']
        assert git(store, 'cat-file', '-t', args.base) == 'commit'
        reachable = git(store, 'rev-list', '--objects', '--missing=print', args.base)
        base_missing = {line[1:].split()[0] for line in reachable.splitlines() if line.startswith('?')}
        assert base_missing <= set(OMITTED), 'unexpected incomplete base'
        # Git 2.34 understands promisor packs, but not the v3 bundle filter header.
        packed = subprocess.check_output(['git', '-C', str(store), 'index-pack', '--stdin',
                    '--fix-thin', '--promisor'], input=pack).decode().strip()
        missing = base_missing | {line[1:].split()[0] for line in git(store, 'rev-list', '--objects',
                    '--missing=print', args.commit, '^' + args.base).splitlines() if line.startswith('?')}
        assert missing <= set(OMITTED), missing
        for oid in missing:
            assert git(store, 'ls-tree', args.commit, OMITTED[oid]).split()[2] == oid
        # A separate local clone keeps active checkouts and their sparse settings
        # untouched. Git's local clone hard-links immutable object files only.
        subprocess.run(['git', 'clone', '--no-checkout', str(store), str(checkout)], check=True)
        git(checkout, 'sparse-checkout', 'init', '--cone')
        git(checkout, 'sparse-checkout', 'set', 'src', 'scripts', 'exploration', 'outputs/' + RUN)
        git(checkout, 'checkout', '--detach', args.commit)
        assert git(checkout, 'rev-parse', 'HEAD') == args.commit
        assert not git(checkout, 'status', '--porcelain')
        files = checkout / 'outputs' / RUN
        receipt = {'commit': args.commit, 'checkout': str(checkout), 'clean': True,
                   'bundle_sha256': args.bundle_sha256, 'pack': packed,
                   'missing_objects': {oid: OMITTED[oid] for oid in sorted(missing)}}
        (root / 'deployment.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('DEPLOY_VERIFIED', json.dumps(receipt), flush=True)
        code = execute('geometry-launch', ['bash', str(files / 'launch_geometry.sh'),
                       str(args.project), str(checkout), args.commit, args.identity])
        run = args.project / 'outputs' / RUN / args.identity
        for relative in ['gpu-probe.json', 'geometry/result.json', 'clock.json']:
            result = run / relative
            if result.exists():
                print('QUAL_RESULT', relative, result.read_text(), flush=True)
        if code == 0 and time.monotonic() < deadline:
            code = execute('pi05-cpu-prepare', ['bash', str(files / 'prepare_pi05.sh'),
                           str(args.project), str(checkout)])
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        clock = {'started': started, 'ended': time.time(), 'exit_code': code, 'gpus': 1,
                 'commit': args.commit, 'phases': phases}
        (root / 'clock.json').write_text(json.dumps(clock, indent=2) + '\n')
        print('QUAL_BOOTSTRAP_CLOCK', json.dumps(clock), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
