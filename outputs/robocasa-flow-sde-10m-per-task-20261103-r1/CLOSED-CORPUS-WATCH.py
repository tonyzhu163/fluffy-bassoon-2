"""Serial CPU closeout using existing pinned readout/archive admission checks.

This only acts after successful native130 and the after panel have closed.
Full reclaim additionally requires the complete numerical1..130 corpus audit.
It starts no accelerator work, simulator calls, fitting or optimizer steps.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1')
PYTHON = '/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python'
RUNTIME = '0988b5f9e9bf31ac3db673cc2860959611b9ae12'
TOOLS = {'readout': ('41ce8d78-FINAL-READOUT.py', '95295eec8a12ddbccfb62a78d06febed70bd48e16e5993d934af3256b7b43800'),
         'archive': ('1aa840ac-TIER-CORPUS.py', 'be3e1ed150a0bea4bec1b791cae385d743f974a5a55e15bf96dc5b5bdef42af4')}
TASKS = {'OpenDrawer': 'job-9j5eqjo0dfzj', 'CloseToasterOvenDoor': 'job-jpfh2j0xelgo',
         'PickPlaceDrawerToCounter': 'job-zqn304j2b9lw', 'CoffeeSetupMug': 'job-wv3zgiuj05j8',
         'TurnOffStove': 'job-z60s3hh8cn7w'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def free_gib():
    fs = os.statvfs(ROOT)
    return fs.f_bavail * fs.f_frsize / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    assert args.record.parent == ROOT and not args.record.exists()
    lock = (ROOT / '.closed-corpus-watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in TOOLS.values():
        assert sha(ROOT / 'tools' / name) == digest
    prepared_inputs = ROOT / 'offline-initial-ratio-inputs-r2/PREPARED.json'
    fixtures = read(prepared_inputs)
    assert fixtures['completed'] and len(fixtures['tasks']) == 5
    for row in fixtures['tasks']:
        assert sha(Path(row['fixture'])) == row['fixture_sha256']
    state = dict(completed=False, started_unix=time.time(), pid=os.getpid(),
                 tasks={}, CPU_workers=1, GPU_jobs=0, new_controls=0, SGD=0,
                 script_sha256=sha(Path(__file__)), pinned_tools=TOOLS,
                 scope='Native final gates plus corpus archive/reclaim only; full joins, initial-ratio GPU reporting and campaign results closeout remain required before fitting.')
    active_task = active_phase = None
    try:
        while True:
            for task, job in TASKS.items():
                active_task = task
                status = state['tasks'].setdefault(task, {})
                if status.get('full_reclaimed'):
                    continue
                production = ROOT / task / 'production-r1'
                manifest = read(production / 'manifest.json')
                assert manifest['source'] == RUNTIME and manifest['allocated_gpus'] == 8
                assert manifest['hostname'] == job + '-master-0'
                clock_path = production / 'clock.json'
                if not clock_path.exists():
                    status['state'] = 'WAITING_NATIVE_CLOSURE'
                    continue
                clock = read(clock_path)
                assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 8
                if time.time() - clock['finished_unix'] < 60:
                    continue
                progress = read(production / 'progress.json')
                after = read(production / 'after-evaluation.json')
                assert progress['completed_iterations'] == 130 and not progress['operator_checkpoint_exit']
                assert after['completed'] and after['global_step'] == 130
                audits = [ROOT / n for n in ['INDEXED-AUDIT-QUALIFICATION-R2.json',
                    'CORPUS-AUDIT-Q1-AND-021-040.json', 'CORPUS-AUDIT-041-060.json']]
                latest = ROOT / task / 'CORPUS-AUDIT-WATCH-061-130-LATEST.json'
                audits.append(latest)
                prefix = f'CORPUS-ARCHIVE-{task}-R1'
                prepared = ROOT / (prefix + '-PREPARED.json')
                readout = ROOT / task / 'FINAL-READOUT-R1.json'
                if not readout.exists():
                    active_phase = 'native-final-readout'
                    status.update(state=active_phase, started_phase_unix=time.time())
                    write(args.record, state)
                    subprocess.run([PYTHON, '-B', str(ROOT / 'tools' / TOOLS['readout'][0]),
                        '--root', str(ROOT), '--task', task, '--output', str(readout)], check=True)
                result = read(readout)
                assert result['completed'] and result['task'] == task
                status.update(readout=str(readout), readout_sha256=sha(readout))
                command = [PYTHON, '-B', str(ROOT / 'tools' / TOOLS['archive'][0]),
                           '--root', str(ROOT), '--task', task]
                if not prepared.exists():
                    active_phase = 'copy-and-full-object-readback'
                    status.update(state=active_phase, started_phase_unix=time.time())
                    write(args.record, state)
                    archive = BOS / 'corpora' / task / 'native-130-r1.tar'
                    subprocess.run(command + ['--mode', 'prepare', '--audits', *map(str, audits),
                        '--archive', str(archive), '--record', str(prepared)], check=True)
                proof = read(prepared)
                assert proof['completed'] and proof['task'] == task and not proof['source_reclaimed']
                assert proof['full_object_readback'] and proof['all_archive_members_hash_compared']
                status.update(prepared=str(prepared), prepared_sha256=sha(prepared),
                              native_closed=True, state='ARCHIVED_WAITING_AUDIT')
                current = read(latest)
                full = current['through_collection'] == 130 and current['requested_range_finished']
                reclaim = ROOT / (prefix + ('-RECLAIMED.json' if full else '-PREFIX.json'))
                if full or (free_gib() < 240 and not reclaim.exists()):
                    if not reclaim.exists():
                        active_phase = 'reclaim-full' if full else 'reclaim-audited-env-actor-prefix'
                        status.update(state=active_phase, started_phase_unix=time.time())
                        write(args.record, state)
                        subprocess.run(command + ['--mode', 'reclaim' if full else 'reclaim-prefix',
                            '--audits', *map(str, audits), '--prepared-receipt', str(prepared),
                            '--prepared-sha256', sha(prepared), '--record', str(reclaim)], check=True)
                    receipt = read(reclaim)
                    assert receipt['completed'] and receipt['reclaim_complete']
                    status.update(full_reclaimed=full, reclaim=str(reclaim),
                                  reclaim_sha256=sha(reclaim),
                                  state='CLOSED_AND_TIERED' if full else 'AUDITED_PREFIX_TIERED_SAMPLING_LOCAL')
                state.update(checked_unix=time.time(), PFS_free_GiB=free_gib())
                write(args.record, state)
            state.update(checked_unix=time.time(), PFS_free_GiB=free_gib())
            state['completed'] = all(row.get('full_reclaimed') for row in state['tasks'].values()) and len(state['tasks']) == 5
            write(args.record, state)
            if state['completed']:
                return
            time.sleep(30)
    except BaseException:
        write(args.record.with_name(args.record.stem + '-ERROR.json'),
              dict(completed=False, failed_unix=time.time(), task=active_task,
                   phase=active_phase, traceback=traceback.format_exc(), state=state))
        raise


if __name__ == '__main__':
    main()
