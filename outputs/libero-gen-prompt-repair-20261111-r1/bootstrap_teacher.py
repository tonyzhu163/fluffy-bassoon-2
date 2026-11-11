"""Deploy an exact Git commit and run the teacher despite a development outage.

Git 2.34 does not read filtered-bundle capabilities. Parse the pinned v3
header, then use its standard index-pack --promisor support explicitly.
This program contains no embedded bundle or authentication material.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback


OMITTED = {'ae19ddac71bf05df3fd87ca19a35089bd27175ae':
           'outputs/libero-gen-prompt-repair-20261111-r1/teacher01-failed.png'}
RUN = 'libero-gen-prompt-repair-20261111-r1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--teacher-sha256', required=True)
    parser.add_argument('--identity', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[a-z0-9-]+', args.identity)
    assert re.fullmatch('[0-9a-f]{40}', args.commit)
    root = args.project / 'outputs' / RUN / ('bootstrap-' + args.identity)
    root.mkdir(parents=True, exist_ok=False)
    started, code = time.time(), 1
    store = args.project / 'sources/libero-gen-store-20261111-r1.git'
    checkout = args.project / 'sources' / ('libero-repair-' + args.commit[:8] + '-' + args.identity)
    def git(where, *arguments):
        return subprocess.check_output(['git', '-C', str(where), *map(str, arguments)], text=True).strip()
    try:
        previous = args.project / 'outputs' / RUN / 'bootstrap-teacher02/clock.json'
        if previous.exists():
            print('PRIOR_BOOTSTRAP_CLOCK', previous.read_text(), flush=True)
        data = args.bundle.read_bytes()
        assert hashlib.sha256(data).hexdigest() == args.bundle_sha256
        header, pack = data.split(b'\n\n', 1)
        lines = header.decode('ascii').splitlines()
        assert lines.pop(0) == '# v3 git bundle' and pack.startswith(b'PACK')
        capabilities = [line for line in lines if line.startswith('@')]
        assert set(capabilities) == {'@object-format=sha1', '@filter=blob:limit=102400'}
        prerequisites, references = [], []
        for line in lines:
            if line.startswith('@'):
                continue
            if line.startswith('-'):
                oid = line[1:].split(' ', 1)[0]
                assert re.fullmatch('[0-9a-f]{40}', oid)
                prerequisites.append(oid)
            else:
                references.append(line)
        assert references == [args.commit + ' refs/heads/main']
        assert prerequisites == ['99583d3763309f13e041df379ae9ae6a9ec48d65']
        print('GIT_VERSION', subprocess.check_output(['git', '--version'], text=True).strip(), flush=True)
        for oid in prerequisites:
            assert git(store, 'cat-file', '-t', oid) == 'commit'
            reachable = git(store, 'rev-list', '--objects', '--missing=print', oid)
            assert not any(line.startswith('?') for line in reachable.splitlines()), 'Incomplete prerequisite history'
        output = subprocess.check_output(['git', '-C', str(store), 'index-pack', '--stdin', '--fix-thin', '--promisor'], input=pack).decode().strip()
        pack_id = output.split()[-1]
        assert re.fullmatch('[0-9a-f]{40}', pack_id), output
        assert (store / 'objects/pack' / ('pack-' + pack_id + '.promisor')).is_file()
        assert git(store, 'cat-file', '-t', args.commit) == 'commit'
        missing_output = git(store, 'rev-list', '--objects', '--missing=print', args.commit, '^' + prerequisites[0])
        missing = {line[1:].split()[0] for line in missing_output.splitlines() if line.startswith('?')}
        assert missing <= set(OMITTED), f'Unexpected missing Git objects: {missing}'
        for oid in missing:
            tree_entry = git(store, 'ls-tree', args.commit, OMITTED[oid])
            assert tree_entry.split()[2] == oid and tree_entry.split()[1] == 'blob'
        git(store, 'worktree', 'add', '--detach', '--no-checkout', checkout, args.commit)
        git(checkout, 'config', '--worktree', 'core.bare', 'false')
        git(checkout, 'config', '--worktree', 'core.worktree', checkout)
        git(checkout, 'sparse-checkout', 'init', '--no-cone')
        paths = ['/src/', '/scripts/', '/exploration/libero-scenes/',
                 '/outputs/libero-procedural-qual-20261111-r1/',
                 '/outputs/' + RUN + '/', '/outputs/libero-gen-custom-layouts-20261111-r1/',
                 '/outputs/libero-gen-history-pilot-20261111-r1/']
        patterns = '\n'.join(paths + ['!/' + path for path in OMITTED.values()]) + '\n'
        subprocess.run(['git', '-C', str(checkout), 'sparse-checkout', 'set', '--no-cone', '--stdin'],
                       input=patterns, text=True, check=True)
        git(checkout, 'checkout', args.commit)
        assert git(checkout, 'rev-parse', 'HEAD') == args.commit
        assert not git(checkout, 'status', '--porcelain')
        files = checkout / 'outputs' / RUN
        assert hashlib.sha256((files / 'teacher.py').read_bytes()).hexdigest() == args.teacher_sha256
        receipt = {'commit': args.commit, 'checkout': str(checkout), 'clean': True,
                   'bundle_sha256': args.bundle_sha256, 'pack': pack_id,
                   'prerequisites': prerequisites, 'capabilities': capabilities,
                   'missing_objects': {oid: OMITTED[oid] for oid in sorted(missing)},
                   'import': 'unchanged bundle PACK; git index-pack --stdin --fix-thin --promisor',
                   'teacher_sha256': args.teacher_sha256}
        (root / 'deployment.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('DEPLOY_VERIFIED', json.dumps(receipt), flush=True)
        run = args.project / 'outputs' / RUN / args.identity
        command = ['bash', str(files / 'launch_teacher.sh'), str(args.project), str(checkout), args.commit, args.identity]
        process = subprocess.Popen(command)
        offset = 0
        while True:
            log = run / 'teacher.log'
            if log.exists():
                with log.open() as stream:
                    stream.seek(offset)
                    chunk = stream.read()
                    offset = stream.tell()
                if chunk:
                    print(chunk, end='', flush=True)
            status = process.poll()
            if status is not None:
                # A final write can occur between the preceding read and poll.
                if log.exists():
                    with log.open() as stream:
                        stream.seek(offset)
                        chunk = stream.read()
                    if chunk:
                        print(chunk, end='', flush=True)
                code = status
                break
            time.sleep(2)
        for result in sorted((run / 'teacher').glob('*/result.json')):
            record = json.loads(result.read_text())
            fields = ['count', 'native_any', 'native_final', 'alignment', 'geometry_observation_hashes_equal', 'trajectory_sha256']
            print('TEACHER_RECEIPT', result.parent.name, json.dumps({key: record.get(key) for key in fields}), flush=True)
        print('TEACHER_CLOCK', (run / 'clock.json').read_text() if (run / 'clock.json').exists() else 'missing', flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        clock = {'started': started, 'ended': time.time(), 'exit_code': code, 'gpus': 1, 'commit': args.commit}
        (root / 'clock.json').write_text(json.dumps(clock, indent=2) + '\n')
        print('BOOTSTRAP_CLOCK', json.dumps(clock), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
