"""Explicit scene-boundary migration and four fixed native collection shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
STUDY = ROOT / 'outputs' / HERE.name
PYTHON = ROOT / 'setups/robocasa365-native-20261101-r1/env/bin/python'
SEEDS = list(range(982100, 982108)) + list(range(983000, 983041))
OLD_PIN = '02d6071efe0107f43943c54911ad56f23d918c53'
PHASE_START = 1794884973


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def events(path):
    # A live writer may have a final partial append; only consume closed lines.
    text = path.read_text()
    return [json.loads(line) for line in text[:text.rfind('\n')+1].splitlines() if line.strip()]


def owned(pid, expected):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert expected in command, (pid, command)
    return command


def handoff(pin):
    assert not (STUDY / 'MIGRATION.json').exists()
    old = STUDY / 'panel01'
    parent, shell, collector = 320466, 320521, 320585
    commands = {
        parent: owned(parent, f'continue_run.py --pin {OLD_PIN}'),
        shell: owned(shell, f'run.sh {OLD_PIN} collect panel01'),
        collector: owned(collector, f'collect_panel.py {old} '),
    }
    assert os.getpgid(parent) == parent and os.getpgid(collector) == shell
    initial = [r['scene'] for r in events(old / 'events.jsonl') if r['kind'] == 'screening'][-1]
    os.kill(parent, signal.SIGSTOP)  # Prevent the old driver from launching a later stage.
    limit = time.time() + 1200
    try:
        while True:
            rows = events(old / 'events.jsonl')
            current = [r['scene'] for r in rows if r['kind'] == 'screening'][-1]
            if current != initial:
                owned(collector, f'collect_panel.py {old} ')
                os.kill(collector, signal.SIGSTOP)
                break
            if (old / 'SUMMARY.json').exists() or time.time() >= limit:
                raise RuntimeError('Handoff did not reach a fresh scene boundary; resume original driver')
            time.sleep(0.2)
        rows = events(old / 'events.jsonl')
        prefix = SEEDS[:SEEDS.index(current)]
        records = [r for r in rows if r['kind'] == 'episode' and r['scene'] in prefix]
        screened, admitted = [], []
        for scene in prefix:
            case = [r for r in records if r['scene'] == scene]
            assert case and case[0]['arm'] == 'nominal'
            nominal = case[0]
            eligible = bool(nominal['success'] and nominal['decision'] is not None)
            assert len(case) == (9 if eligible else 1)
            if eligible:
                assert len(json.loads((old / f'scene-{scene}.json').read_text())) == 9
                admitted.append(scene)
            screened.append(dict(scene=scene, eligible=eligible, **{k: nominal[k] for k in ('success', 'decision', 'first_grasp', 'controls')}))
        inventory = [dict(path=str(p.relative_to(STUDY)), bytes=p.stat().st_size, sha256=sha(p))
                     for p in sorted(old.iterdir()) if p.is_file() and not p.name.endswith('.tmp')]
        remaining = SEEDS[len(prefix):]
        assert len(admitted) < 16 and remaining
        receipt = dict(source_commit=pin, old_source_commit=OLD_PIN, commands=commands,
            stopped_at_next_scene=current, prefix_scenes=prefix, admitted=admitted, screened=screened,
            records=records, original_order=SEEDS, shards=[remaining[i::4] for i in range(4)],
            original_files=inventory, phase_start=PHASE_START, handoff_unix=time.time(),
            prior_gpu_hours=(time.time()-PHASE_START)/3600,
            maximum_parallel_gpu_hours=5, scheduler_minutes=75, worker_minutes=70,
            policy='First 16 nominally eligible scenes in original seed order; retain every extra and incomplete artifact')
        assert receipt['prior_gpu_hours'] + 5 < 8
        write(STUDY / 'MIGRATION.json', receipt)
        # Frozen policy has no optimizer; all complete scenes have their full action tapes.
        # The just-entered scene is restarted in a new output; its partial artifact is retained.
        os.killpg(shell, signal.SIGTERM)
        os.kill(collector, signal.SIGCONT)
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        print(json.dumps({k: receipt[k] for k in ('stopped_at_next_scene','admitted','shards','prior_gpu_hours')}), flush=True)
    except BaseException:
        for pid in (collector, parent):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        raise


def worker(pin):
    assert socket.gethostname().startswith('job-')
    migration = json.loads((STUDY / 'MIGRATION.json').read_text())
    assert migration['source_commit'] == pin
    started = time.time()
    slots = STUDY / 'parallel01-slots'
    slots.mkdir(exist_ok=True)
    for slot in range(4):
        receipt = slots / str(slot)
        try:
            receipt.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('All four slots already claimed; retries are forbidden')
    seeds = migration['shards'][slot]
    write(receipt / 'LAUNCH.json', dict(source_commit=pin, hostname=socket.gethostname(), pid=os.getpid(),
        started_unix=started, seeds=seeds, scheduler_minutes=75, maximum_gpu_hours=1.25))
    assert shutil.disk_usage(ROOT).free > 2 * 2**30  # Compact tapes are <1GB; preserve headroom.
    probe = receipt / '.write-probe'
    probe.write_text('native panel')
    assert probe.read_text() == 'native panel'
    probe.unlink()
    env = dict(os.environ, NCCL_P2P_DISABLE='1')
    command = ['bash', str(HERE / 'run.sh'), pin, 'collect', f'parallel01-w{slot}',
               '--seeds', ','.join(map(str,seeds)), '--target-admitted', '16', '--screen-all',
               '--bias-m', '0.04', '--window', '100']
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        rc = process.wait(timeout=70*60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        rc = process.wait(timeout=60)
        if rc == 0:
            rc = 124
    write(receipt / 'EXIT.json', dict(exit_code=rc, ended_unix=time.time(),
        gpu_hours=(time.time()-started)/3600, completed=rc == 0))
    # PyTorchJob may declare success when master exits. Keep every successful
    # instance alive until all four have flushed their complete artifacts.
    while rc == 0:
        exits = [slots / str(i) / 'EXIT.json' for i in range(4)]
        if any(p.exists() and json.loads(p.read_text())['exit_code'] != 0 for p in exits):
            rc = 1
            break
        if all(p.exists() for p in exits):
            break
        if time.time() > started + 72*60:
            rc = 124
            break
        time.sleep(5)
    raise SystemExit(rc)


def merge(pin):
    migration = json.loads((STUDY / 'MIGRATION.json').read_text())
    outputs = [STUDY / 'panel01'] + [STUDY / f'parallel01-w{i}' for i in range(4)]
    valid = [migration['prefix_scenes']] + migration['shards']
    by_scene, origins = {}, {}
    for directory, scenes in zip(outputs, valid, strict=True):
        if directory.name != 'panel01':
            summary = json.loads((directory / 'SUMMARY.json').read_text())
            assert summary['completed'] and summary['screen_all']
            assert [r['scene'] for r in summary['screened']] == scenes
        for row in events(directory / 'events.jsonl'):
            if row.get('scene') in scenes:
                by_scene.setdefault(row['scene'], []).append(row)
                assert row['scene'] not in origins or origins[row['scene']] == directory
                origins[row['scene']] = directory
    admitted, screened, records, ledger = [], [], [], []
    for scene in SEEDS:
        rows = by_scene[scene]
        case = [r for r in rows if r['kind'] == 'episode']
        nominal = case[0]
        assert nominal['arm'] == 'nominal'
        eligible = bool(nominal['success'] and nominal['decision'] is not None)
        assert len(case) == (9 if eligible else 1)
        screened.append(dict(scene=scene, eligible=eligible, **{k: nominal[k] for k in ('success','decision','first_grasp','controls')}))
        ledger.extend(rows)
        records.extend(case)
        if eligible:
            admitted.append(scene)
        if len(admitted) == 16:
            break
    assert len(admitted) == 16, f'Only {len(admitted)} eligible scenes; do not expand the seed panel silently'
    out = STUDY / 'panel02'
    out.mkdir()
    inventory = []
    for row in records:
        source = origins[row['scene']] / f"{row['scene']}-{row['arm']}.npz"
        dest = out / source.name
        os.link(source, dest)
        inventory.append(dict(path=dest.name, source=str(source), sha256=sha(source)))
    for scene in admitted:
        source = origins[scene] / f'scene-{scene}.json'
        os.link(source, out / source.name)
    (out / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ledger))
    write(out / 'SUMMARY.json', dict(completed=True, bias_m=.04, window=100, horizon=600, gate_m=.10,
        target_admitted=16, screened_count=len(screened), admitted_count=16, admitted_scenes=admitted,
        screened=screened, episodes=len(records), total_controls=sum(r['controls'] for r in records),
        records=records, source_commit=pin, migration_sha256=sha(STUDY/'MIGRATION.json')))
    write(out / 'SOURCES.json', inventory)
    return out


def finish(pin):
    state_path = STUDY / 'PARALLEL-CONTINUATION.json'
    assert not state_path.exists()
    state = dict(source_commit=pin, pid=os.getpid(), phase='waiting_for_parallel', completed=False)
    def status(**kw):
        state.update(kw, checked_unix=time.time())
        write(state_path, state)
    status()
    try:
        exits = [STUDY / 'parallel01-slots' / str(i) / 'EXIT.json' for i in range(4)]
        while not all(p.exists() for p in exits):
            if any(p.exists() and json.loads(p.read_text())['exit_code'] != 0 for p in exits):
                raise RuntimeError('Parallel worker failed; retain partial evidence and diagnose')
            if time.time() >= PHASE_START + 8*3600:
                raise TimeoutError('Original phase deadline reached; no fit started')
            time.sleep(20)
        receipts = [json.loads(p.read_text()) for p in exits]
        assert all(r['completed'] for r in receipts)
        migration = json.loads((STUDY / 'MIGRATION.json').read_text())
        # Charge the full four-instance scheduler reservation, including startup
        # and any time completed pods remain held by the multi-instance job.
        measured = sum(r['gpu_hours'] for r in receipts)
        spent = migration['prior_gpu_hours'] + migration['maximum_parallel_gpu_hours']
        status(phase='merging', collection_charged_gpu_hours=spent, worker_measured_gpu_hours=measured)
        panel = merge(pin)
        reference = STUDY / 'reference02'
        subprocess.run([str(PYTHON), str(HERE/'train_and_eval.py'), '--panel', str(panel),
                        '--out', str(reference), '--stage1-only'], check=True)
        ref = json.loads((reference/'RESULT.json').read_text())['stage1']['probed_train_cross']
        if ref['feedback_own']['attempt2_success']['point'] <= max(ref['best_fixed']['attempt2_success']['point'], ref['current_obs_ref']['attempt2_success']['point']):
            status(phase='reference_negative', completed=True)
            return
        remaining = min((8-spent)*3600, PHASE_START+8*3600-time.time())
        assert remaining > 0
        started = time.time()
        process = subprocess.Popen(['bash', str(HERE/'run.sh'), pin, 'train', 'fit02', str(panel)], start_new_session=True)
        status(phase='fit02', child_pid=process.pid, maximum_fit_seconds=remaining)
        try:
            rc = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
            raise TimeoutError('Aggregate eight GPU-hour budget reached; milestone states retained')
        assert rc == 0
        audit = json.loads((STUDY/'fit02/AUDIT.json').read_text())
        assert audit['passed']
        status(phase='complete', completed=True, total_charged_gpu_hours=spent+(time.time()-started)/3600)
    except BaseException as error:
        status(phase='stopped', error=repr(error), completed=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('handoff','worker','finish'))
    parser.add_argument('--pin', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip() == args.pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    {'handoff':handoff, 'worker':worker, 'finish':finish}[args.mode](args.pin)


if __name__ == '__main__':
    main()
