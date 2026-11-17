"""Bounded continuation of the already-started native history qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
STUDY = ROOT / 'outputs' / HERE.name
PYTHON = ROOT / 'setups/robocasa365-native-20261101-r1/env/bin/python'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--started', required=True, type=float)
    p.add_argument('--gpu-hours', type=float, default=8.)
    a = p.parse_args()
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == a.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    assert not (STUDY / 'CONTINUATION.json').exists()
    deadline = a.started + a.gpu_hours * 3600
    state = dict(source_commit=a.pin, pid=os.getpid(), started_unix=a.started,
                 gpu_hour_budget=a.gpu_hours, deadline_unix=deadline, phase='waiting_for_gate', completed=False)

    def status(**updates):
        state.update(updates, checked_unix=time.time(), conservative_gpu_hours=(time.time()-a.started)/3600)
        write(STUDY / 'CONTINUATION.json', state)

    def stage(name, command):
        if time.time() >= deadline:
            raise TimeoutError('Phase budget exhausted before starting ' + name)
        log = STUDY / (name + '-controller.log')
        assert not log.exists()
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        with log.open('w') as f:
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            status(phase=name, child_pid=process.pid, command=command)
            last_notice = 0
            while process.poll() is None:
                if time.time() >= deadline:
                    # Only the process group created above belongs to this stage.
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=60)
                    raise TimeoutError('Phase budget reached during ' + name)
                if time.time() - last_notice >= 1200:
                    status(phase=name, child_pid=process.pid)
                    print(json.dumps(state), flush=True)
                    last_notice = time.time()
                time.sleep(20)
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}; inspect {log}')
        status(phase=name + '_complete', child_pid=None)

    try:
        status()
        clock = STUDY / 'gate01-receipt/clock.json'
        while not clock.exists():
            if time.time() >= deadline:
                raise TimeoutError('Existing gate did not complete within the phase budget; no continuation launched')
            time.sleep(20)
        receipt = json.loads(clock.read_text())
        assert receipt['exit_code'] == 0, 'Native gate failed; diagnose before retrying'
        gate = json.loads((STUDY / 'gate01/SUMMARY.json').read_text())
        assert gate['completed'] and gate['admitted_count'] == 2
        by = {(r['scene'], r['arm']): r for r in gate['records']}
        checks = []
        for scene in gate['admitted_scenes']:
            nominal, replay = by[scene, 'nominal'], by[scene, 'repeat_tape']
            assert (nominal['controls'], nominal['success'], nominal['terminal_sha256']) == (
                replay['controls'], replay['success'], replay['terminal_sha256'])
            for z in (-1, 1):
                outcomes = {u: by[scene, f'b{z:+d}-u{u:+d}']['success'] for u in (-1, 0, 1)}
                checks.append(dict(scene=scene, z=z, outcomes=outcomes,
                    qualified=bool(outcomes[-z] and not outcomes[0] and not outcomes[z])))
        qualified = all(r['qualified'] for r in checks)
        write(STUDY / 'GATE-READOUT.json', dict(completed=True, qualified=qualified, checks=checks,
            gate_summary_sha256=digest(STUDY / 'gate01/SUMMARY.json')))
        if not qualified:
            status(phase='gate_negative', completed=True,
                   conclusion='Calibration reference did not qualify; no panel or learner fit launched')
            return
        stage('panel01', ['bash', str(HERE / 'run.sh'), a.pin, 'collect', 'panel01',
            '--seeds', '982100:982108,983000:983041', '--target-admitted', '16', '--bias-m', '0.04', '--window', '100'])
        stage('reference01', [str(PYTHON), str(HERE / 'train_and_eval.py'),
            '--panel', str(STUDY / 'panel01'), '--out', str(STUDY / 'reference01'), '--stage1-only'])
        reference = json.loads((STUDY / 'reference01/RESULT.json').read_text())['stage1']['probed_train_cross']
        own = reference['feedback_own']['attempt2_success']['point']
        qualified = own > max(reference['best_fixed']['attempt2_success']['point'],
                              reference['current_obs_ref']['attempt2_success']['point'])
        # Continue on the training reference only; held-out outcomes do not choose fit settings.
        if not qualified:
            status(phase='reference_negative', completed=True,
                   conclusion='Training reference had no information advantage; native panel retained, no fit launched')
            return
        stage('fit01', ['bash', str(HERE / 'run.sh'), a.pin, 'train', 'fit01', str(STUDY / 'panel01')])
        fit = json.loads((STUDY / 'fit01/RESULT.json').read_text())
        audit = json.loads((STUDY / 'fit01/AUDIT.json').read_text())
        assert fit['completed'] and audit['passed']
        status(phase='complete', completed=True, fit_result_sha256=digest(STUDY / 'fit01/RESULT.json'))
        cold = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name / 'evidence01'
        assert not cold.exists()
        cold.mkdir(parents=True)
        inventory = []
        # The driver log remains open; exclude it and the mutable status receipt.
        for path in sorted(STUDY.rglob('*')):
            if not path.is_file() or path.name in ('continuation.log', 'CONTINUATION.json'):
                continue
            relative = path.relative_to(STUDY)
            dest = cold / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            sha = digest(path)
            assert digest(dest) == sha
            inventory.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=sha))
        write(STUDY / 'ARCHIVE.json', dict(cold_root=str(cold), files=inventory,
            all_readback_hashes_verified=True, bytes=sum(r['bytes'] for r in inventory)))
        status(phase='archived', completed=True, archive=str(cold))
    except BaseException as e:
        status(phase='stopped', completed=False, error=f'{type(e).__name__}: {e}')
        raise


if __name__ == '__main__':
    main()
