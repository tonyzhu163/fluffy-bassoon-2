"""Audit the four owned continuations and bind all eight fixed-final models.

Linux CPU orchestration only. This never submits accelerator work or changes
an active fit. The existing numerical audit checks the actual saved tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(16 << 20):
            h.update(block)
    return h.hexdigest()


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    here = Path(__file__).resolve().parent
    run = 'robocasa-six-task-scale-20261113-r1'
    root = Path('/mnt/bos/adgfyl/strong-plant-archives') / run
    output = root / 'final-fit-audit02'
    output.mkdir(exist_ok=False)
    audit = Path('/tmp/completed-fit-audit-source-9e714a3e/outputs') / run / 'audit_completed_fit.py'
    assert sha(audit) == '93e5d070ab5e85a9ff3d8bc5a1cc2d7d0c0a7d5ad05eccc131623d920a5c876b'
    inventory = json.loads((here / 'FINAL-FIT-CANDIDATES.json').read_text())
    assert set(inventory['models']) == {'3/ad', '3/dpt', '6/ad', '6/dpt'}
    for row in inventory['models'].values():
        for item in row.values():
            assert sha(item['path']) == item['sha256']
        assert json.loads(Path(row['audit']['path']).read_text())['passed']
    pending = {
        '3/dit': ('resume3dit01', 'job-x26sts56sm1y'),
        '3/ic_iql': ('resume3iciql01', 'job-ffzoxy4ezshi'),
        '6/dit': ('resume6dit01', 'job-8p3dcubnslyp'),
        '6/ic_iql': ('resume6iciql01', 'job-98188j8yebi8'),
    }
    state = dict(started_unix=time.time(), hostname=socket.gethostname(), pid=os.getpid(),
        script_sha256=sha(__file__), deadline_unix=args.deadline,
        audited={}, completed=False, submitted=False, GPU_hours=0)
    write(output / 'STATE.json', state)
    try:
        while pending and time.time() < args.deadline:
            for name, (attempt, job) in list(pending.items()):
                size, method = name.split('/')
                directory = root / attempt
                error = directory / 'ERROR.json'
                if error.exists():
                    raise RuntimeError(f'{name} parent error: {error}')
                terminal = directory / (method + '-EXIT.json')
                if not terminal.exists():
                    continue
                ended = json.loads(terminal.read_text())
                assert ended['exit_code'] == 0 and ended['complete'], (name, ended)
                launch = json.loads((directory / 'LAUNCH.json').read_text())
                assert launch['hostname'] == job + '-master-0'
                assert launch['fit_source_commit'] == inventory['fit_source_commit']
                complete = directory / method / 'COMPLETE.json'
                target = output / (name.replace('/', '_') + '.json')
                subprocess.run([sys.executable, '-B', str(audit), '--complete', str(complete),
                    '--method', method, '--corpus-size', size, '--output', str(target)],
                    check=True, timeout=600, stdout=subprocess.DEVNULL)
                result = json.loads(target.read_text())
                assert result['passed']
                inventory['models'][name] = {key: result[key] for key in
                    ['complete', 'candidate', 'checkpoint', 'terminal']}
                inventory['models'][name]['audit'] = reference(target)
                state['audited'][name] = reference(target)
                del pending[name]
                state['checked_unix'] = time.time()
                write(output / 'STATE.json', state)
                print(json.dumps(dict(audited=name, actor_updates=result['actor_updates'],
                    value_updates=result['value_updates'], candidate=result['candidate'])), flush=True)
            if pending:
                time.sleep(30)
        assert not pending, ('deadline with unfinished fits', sorted(pending))
        assert set(inventory['models']) == set(inventory['required_models'])
        inventory.update(ready_for_matched_evaluation=True, completed_and_audited_count=8,
            pending_models=[], checked_unix=time.time(),
            next_action='Review capacity and budget, then submit the prepared fixed paired readout.')
        inventory.pop('checked_shanghai', None)
        write(output / 'FINAL-FIT-CANDIDATES.json', inventory)
        source = Path('/mowenjie/fluffy-bassoon-2/sources/matched-final-e9fe49c4/outputs') / run
        baselines = source / 'BASELINE-CONTROLS.json'
        assert sha(baselines) == '0debb9d70bea5ebc5619aebe17f87c5089b8459a258d77288c9ab76e70167cdc'
        assert sha(source / 'EVALUATION-MIGRATION.json') == 'b44f6e7de97f5d23c352a491e65668f23a52cdae74434399fa7204c9c17f2753'
        spec = dict(models=inventory['models'], baseline_controls=reference(baselines),
            evaluation_migration=reference(source / 'EVALUATION-MIGRATION.json'),
            cases_per_task=40, shards=4, scheduler_maximum_minutes=480,
            reserved_GPU_hours=32, total_physical_episodes=13200,
            selection='All eight precommitted fixed actor10000 checkpoints; no outcome selection',
            final_audit_inventory=reference(output / 'FINAL-FIT-CANDIDATES.json'))
        write(output / 'SPEC.json', spec)
        state.update(completed=True, ended_unix=time.time(),
            inventory=reference(output / 'FINAL-FIT-CANDIDATES.json'),
            evaluation_spec=reference(output / 'SPEC.json'))
        write(output / 'STATE.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(error=repr(error), ended_unix=time.time())
        write(output / 'STATE.json', state)
        raise


if __name__ == '__main__':
    main()
