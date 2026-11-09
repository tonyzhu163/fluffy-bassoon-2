"""One exclusive GPU container: native server and up to four simulator clients."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

failure_root = None


def check_peers():
    if failure_root is not None:
        for path in failure_root.glob('rank-*/EXIT.json'):
            result = json.loads(path.read_text())
            if not result['completed']:
                raise RuntimeError(f'Peer failed: {path}: {result.get("error")}')


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def wait_file(path, deadline, server=None):
    while not path.exists():
        check_peers()
        if server is not None and server.poll() is not None:
            raise RuntimeError(f'Native server exited with {server.returncode}')
        if time.time() >= deadline:
            raise TimeoutError(f'Waiting for {path}')
        time.sleep(2)


def main():
    global failure_root
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    started = time.time()
    assert subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip() == a.pin
    subprocess.run(['git', '-C', str(a.source), 'diff', '--quiet'], check=True)
    plan = json.loads(a.manifest.read_text())
    m = re.fullmatch(r'(job-.+)-(master|worker)-(\d+)', socket.gethostname())
    assert m is not None
    rank = 0 if m[2] == 'master' and m[3] == '0' else int(m[3]) + 1
    world = len(plan['shards'])
    assert 0 <= rank < world and (m[2] != 'master' or rank == 0)
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output / f'rank-{rank:02d}'
    out.mkdir(exist_ok=False)
    failure_root = a.output
    deadline = started + plan['maximum_worker_seconds']
    save(out/'IDENTITY.json', dict(job=m[1], hostname=socket.gethostname(), rank=rank,
         source_commit=a.pin, started_unix=started,
         manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
         rank_environment={k: os.environ.get(k) for k in ('RANK', 'WORLD_SIZE')}))
    status = dict(completed=False, rank=rank, training_updates=0)
    server = None
    server_log = None
    clients = []
    client_logs = []
    try:
        subprocess.run([sys.executable, '-B', str(a.source/'scripts/[train]gpu_probe.py'),
                    '--expected-count', '1', '--purpose', 'evaluation', '--baige-container',
                    '--output', str(out/'GPU-PROBE.pending.json')], check=True)
        (out/'GPU-PROBE.pending.json').replace(out/'GPU-PROBE.json')
        for r in range(world):
            wait_file(a.output/f'rank-{r:02d}/GPU-PROBE.json', started+900)
        probes = [json.loads((a.output/f'rank-{r:02d}/GPU-PROBE.json').read_text()) for r in range(world)]
        identities = [json.loads((a.output/f'rank-{r:02d}/IDENTITY.json').read_text()) for r in range(world)]
        assert all(x['status'] == 'passed' for x in probes)
        assert len({x['devices'][0]['uuid'] for x in probes}) == world
        assert {x['job'] for x in identities} == {m[1]}
        assert {x['source_commit'] for x in identities} == {a.pin}
        assert len({x['manifest_sha256'] for x in identities}) == 1
        scripts = a.source/'outputs/robocasa-sft-reproduction-20261109-r1'
        checkpoint = '/mowenjie/fluffy-bassoon-2/inputs/robocasa365-public-pi05-c484-75000-20261101-r1'
        server_log = (out/'server.log').open('x')
        server = subprocess.Popen([sys.executable, '-B', str(scripts/'serve_native.py'),
        '--checkpoint', checkpoint, '--output', str(out/'server'), '--port', '18405'],
        stdout=server_log, stderr=subprocess.STDOUT)
        wait_file(out/'server/READY.json', started+1200, server)
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:18405/healthz', timeout=2) as response:
                    assert response.status == 200
                break
            except OSError:
                if time.time() > started+1200 or server.poll() is not None:
                    raise
                time.sleep(1)
        shard = plan['shards'][rank]
        for stage in plan['stages']:
            names = shard[stage]
            assert names
            stage_root = out/stage
            stage_root.mkdir()
            for offset in range(0, len(names), 4):
                for task in names[offset:offset+4]:
                    stream = (stage_root/f'{task}.log').open('x')
                    client_logs.append(stream)
                    child = subprocess.Popen([sys.executable, '-B', str(scripts/'reference_eval.py'),
                        '--checkpoint', checkpoint, '--output', str(stage_root/task), '--tasks', task,
                        '--trials', '50', '--maximum-seconds', str(max(1, int(deadline-time.time()))),
                        '--version', plan['version'], '--replan', '5', '--server-port', '18405'],
                        stdout=stream, stderr=subprocess.STDOUT)
                    clients.append(child)
                while any(child.poll() is None for child in clients):
                    check_peers()
                    if time.time() >= deadline or server.poll() is not None:
                        raise RuntimeError('Deadline reached or native server exited')
                    if any(child.poll() not in (None, 0) for child in clients):
                        raise RuntimeError(f'{stage} client failed')
                    time.sleep(2)
                failures = [child.returncode for child in clients]
                clients = []
                if any(failures):
                    raise RuntimeError(f'{stage} clients exited {failures}')
            for task in names:
                result = json.loads((stage_root/task/'RESULT.json').read_text())
                assert result['completed'] and result['episodes'] == 50 and result['training_updates'] == 0
            save(out/f'{stage}-DONE.json', dict(completed=True, tasks=names, unix=time.time()))
            if stage == 'atomic_seen':
                gate = a.output/'ATOMIC-GATE.json'
                if rank == 0:
                    for r in range(world):
                        wait_file(a.output/f'rank-{r:02d}/atomic_seen-DONE.json', deadline-180, server)
                    records = []
                    for r in range(world):
                        for task in plan['shards'][r]['atomic_seen']:
                            folder = a.output/f'rank-{r:02d}/atomic_seen'/task
                            cases = [json.loads(f.read_text()) for f in folder.glob(f'{task}-*.json')]
                            assert len(cases) == 50 and {c['episode'] for c in cases} == set(range(50))
                            records.extend(cases)
                    assert len(records) == 900
                    successes = sum(x['success'] for x in records)
                    original = sum(x['success_original_horizon'] for x in records)
                    rate = successes/900
                    mismatch = 'review_v100' if plan['version'] == '1.0.1' else 'reference_mismatch'
                    decision = 'continue_composites' if abs(rate-.396) <= .05 else mismatch
                    save(gate, dict(episodes=900, successes=successes, success_rate=rate,
                         original_horizon_successes=original, reference_rate=.396,
                         practical_band=.05, decision=decision,
                         interpretation='Practical screen; not a formal equivalence claim', unix=time.time()))
                wait_file(gate, deadline-180, server)
                if json.loads(gate.read_text())['decision'] != 'continue_composites':
                    status.update(completed=True, stopped_at_atomic_gate=True)
                    break
        else:
            status['completed'] = True
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        save(out/'EXIT.json', status)
        raise
    finally:
        for child in clients:
            if child.poll() is None:
                child.terminate()
        for child in clients:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if server is not None and server.poll() is None:
            server.terminate()
        if server is not None:
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        for stream in client_logs:
            stream.close()
        if server_log is not None:
            server_log.close()
        status.update(started_unix=started, ended_unix=time.time(),
                      server_exit=server.returncode if server is not None else None)
        save(out/'EXIT.json', status)


if __name__ == '__main__':
    main()
