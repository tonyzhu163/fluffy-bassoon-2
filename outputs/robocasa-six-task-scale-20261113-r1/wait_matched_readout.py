"""Finish the owned matched readout on Linux CPU after all workers succeed.

This waits, audits archived cases, renders every planned figure, and verifies
cold copies. It never submits, stops, retries, or changes accelerator jobs.
Visual review and the scientific decision remain manual follow-through.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


RUN = 'robocasa-six-task-scale-20261113-r1'
PIN = '7cc56702ec6b623ae4aa4f79da211883ca4ecddd'
SPEC_SHA = '34fd790404df709796399cd845eec88788b5365488bf08f0480d9972efb0158f'
ROOT = Path('/mowenjie/fluffy-bassoon-2')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / RUN
SOURCE = ROOT / 'sources/matched-readout-7cc56702'
SCRIPTS = SOURCE / 'outputs' / RUN
NATIVE_PYTHON = ROOT / 'setups/robocasa365-native-20261101-r1/env/bin/python'
PLOT_PYTHON = ROOT / 'setups/checkpoint-dev-plot-20261104-r1/bin/python'
EXPECTED_SCRIPTS = {
    'matched_analyze.py': 'a28dfa21b8443e449d54f63ccc869ce94ae757d3a8c11d8cfd35a67a55ef7093',
    'plot_matched.py': '2757e40a1ebe2672532b99accb700b099131f5aad234533e61ec7235704d1087',
    'matched_readout.py': '9208159316e7beb7b6b4876cd9b66cf4e0361aae41d7c3e8e94a71e6b9769dfc',
    'labels.py': '6bb015cf737362d6c3acb8935c8d6bf2e6c31cd79e1971e390fb4aecd52fd469',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--orchestrator-source', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    assert not socket.gethostname().startswith('job-'), 'Use the CPU development instance'
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no'])
    for name, expected in EXPECTED_SCRIPTS.items():
        assert sha(SCRIPTS / name) == expected
    spec = COLD / 'readout-inputs01/SPEC.json'
    assert sha(spec) == SPEC_SHA
    run = COLD / 'matchedreadout01'
    launch = read(run / 'LAUNCH.json')
    assert launch['hostname'] == 'job-ayk637qgg7gw-master-0'
    assert launch['source_commit'] == PIN and launch['spec_sha256'] == SPEC_SHA
    assert launch['spec'] == read(spec)
    output = COLD / 'matched-finalize01'
    local = Path('/tmp') / RUN / 'matched-finalize01'
    local.mkdir(parents=True, exist_ok=False)
    output.mkdir(exist_ok=False)
    assert local.stat().st_dev != ROOT.stat().st_dev
    state = dict(started_unix=time.time(), hostname=socket.gethostname(), pid=os.getpid(),
        deadline_unix=args.deadline, orchestrator_source=args.orchestrator_source,
        script_sha256=sha(__file__), runtime_source=PIN, spec=reference(spec),
        launch=reference(run / 'LAUNCH.json'), GPU_hours=0, completed=False,
        stage='Waiting for the exact owned evaluation to complete',
        policy='No accelerator mutation or automatic retry; preserve all planned cases and contrasts')
    write(output / 'STATE.json', state)
    print(json.dumps(state), flush=True)
    try:
        while True:
            for path in [run / 'ERROR.json'] + [
                    run / f'shard-{i:02d}' / name for i in range(4)
                    for name in ['ERROR.json', 'EXIT.json', 'evaluation/INCOMPLETE.json']]:
                if not path.exists():
                    continue
                value = read(path)
                if path.name != 'EXIT.json' or value.get('exit_code') != 0 or not value.get('complete'):
                    raise RuntimeError(f'Evaluation needs diagnosis: {path}: {value}')
            terminal = run / 'EXIT.json'
            if terminal.exists():
                ended = read(terminal)
                assert ended['completed'] and len(ended['results']) == 4
                assert {row['shard'] for row in ended['results']} == set(range(4))
                assert all(row['exit_code'] == 0 and row['complete'] for row in ended['results'])
                for i in range(4):
                    assert read(run / f'shard-{i:02d}/evaluation/COMPLETE.json')['completed']
                state['terminal'] = reference(terminal)
                break
            if time.time() >= args.deadline:
                state.update(stage='Observation deadline reached; GPU job not changed', checked_unix=time.time())
                write(output / 'STATE.json', state)
                print(json.dumps(state), flush=True)
                return
            time.sleep(min(30, max(0, args.deadline - time.time())))
        state.update(stage='Auditing all archived cases and computing paired statistics', checked_unix=time.time())
        write(output / 'STATE.json', state)
        print(json.dumps(dict(stage=state['stage'], terminal=state['terminal'])), flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (local / 'analysis.log').open('x') as stream:
            subprocess.run([str(NATIVE_PYTHON), '-B', str(SCRIPTS / 'matched_analyze.py'),
                '--run', str(run), '--spec', str(spec), '--spec-sha256', SPEC_SHA,
                '--output', str(local / 'analysis')], env=env, stdout=stream,
                stderr=subprocess.STDOUT, check=True, timeout=2400)
        result = read(local / 'analysis/RESULT.json')
        coverage = read(local / 'analysis/COVERAGE.json')
        assert result['completed'] and result['verified_physical_episodes'] == 13200
        assert result['transformer_physical_episodes'] == 12800 and result['baseline_physical_episodes'] == 400
        assert coverage['completed'] and len(coverage['verified_banks']) == 25
        assert all(not missing for missing in coverage['missing_complete_eight_model_and_baseline_slots'].values())
        state.update(stage='Rendering all three planned comparison figures', checked_unix=time.time())
        write(output / 'STATE.json', state)
        with (local / 'figures.log').open('x') as stream:
            subprocess.run([str(PLOT_PYTHON), '-B', str(SCRIPTS / 'plot_matched.py'),
                '--result', str(local / 'analysis/RESULT.json'), '--output', str(local / 'figures'),
                '--source-commit', PIN], env=env, stdout=stream,
                stderr=subprocess.STDOUT, check=True, timeout=600)
        figures = read(local / 'figures/FIGURES.json')
        assert len(figures['files']) == 6
        for row in figures['files']:
            assert sha(row['path']) == row['sha256']
        copies = []
        for original in sorted(local.rglob('*')):
            if not original.is_file():
                continue
            target = output / original.relative_to(local)
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            shutil.copyfile(original, target)
            assert sha(original) == sha(target)
            copies.append(dict(source=reference(original), cold=reference(target), full_readback_verified=True))
        archive = dict(completed=True, checked_unix=time.time(), files=copies,
            terminal=state['terminal'], spec=state['spec'], GPU_hours=0,
            visual_review='Pending manual inspection; rendering alone does not establish visual correctness')
        write(output / 'ARCHIVE.json', archive)
        state.update(completed=True, stage='Analysis and figures archived; visual review and scientific decision pending',
            ended_unix=time.time(), archive=reference(output / 'ARCHIVE.json'),
            result=reference(output / 'analysis/RESULT.json'), coverage=reference(output / 'analysis/COVERAGE.json'),
            figures=reference(output / 'figures/FIGURES.json'))
        write(output / 'STATE.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(error=repr(error), ended_unix=time.time(), local_evidence=str(local))
        write(output / 'STATE.json', state)
        print(json.dumps(state), flush=True)
        raise


if __name__ == '__main__':
    main()
