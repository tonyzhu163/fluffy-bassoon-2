"""Observe repeated native rendering; never changes a production pixel guard."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path('/mowenjie/fluffy-bassoon-2')
SOURCE = ROOT / 'outputs/robocasa-interface-screen-20261102-r1'
OUT = ROOT / 'outputs/robocasa-interface-render-qualification-20261102-r1'
CODE = Path(__file__).resolve().parents[2]
TASKS = ('OpenDrawer', 'PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def compare(reference, current):
    for key in ('format', 'case', 'physical', 'task_descriptions'):
        if reference[key] != current[key]:
            raise ValueError('Exact native reset mismatch: ' + key)
    if not np.array_equal(reference['states'], current['states']):
        raise ValueError('Exact native public-state mismatch')
    images = {}
    for key in ('main_images', 'wrist_images', 'extra_view_images'):
        a, b = reference['images'][key], current['images'][key]
        assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        images[key] = dict(max_delta=int(delta.max()), differing_components=int(np.count_nonzero(delta)),
            over_one_components=int(np.count_nonzero(delta > 1)), total_components=int(delta.size), mean_delta=float(delta.mean()))
    return dict(physical_public_camera_model_exact=True, images=images,
                observations_modified=False, original_production_bound=1,
                within_original_bound=all(row['max_delta'] <= 1 for row in images.values()))


def fixture_check():
    assert platform.system() == 'Linux' and not torch.cuda.is_available()
    from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair, MAX_RGB_DELTA
    assert MAX_RGB_DELTA == 1
    failure = json.loads((SOURCE / 'FAILURE-DIAGNOSIS.json').read_text())['failed_snapshots'][0]
    a = torch.load(failure['canonical']['path'], map_location='cpu', weights_only=False)
    b = torch.load(failure['snapshot']['path'], map_location='cpu', weights_only=False)
    assert sha(failure['canonical']['path']) == failure['canonical']['sha256']
    assert sha(failure['snapshot']['path']) == failure['snapshot']['sha256']
    result = compare(a, b)
    assert result['images']['extra_view_images']['max_delta'] == 2
    assert result['images']['extra_view_images']['over_one_components'] == 12
    try:
        compare_native_pair(a, b)
    except ValueError as error:
        assert 'render roundoff' in str(error)
    else:
        raise AssertionError('Existing production guard must still reject this fixture')
    print(json.dumps(dict(completed=True, actual_saved_fixture_verified=True,
                          unchanged_production_guard_rejects=True, comparison=result), indent=2))


def main():
    assert platform.system() == 'Linux'
    assert os.environ['RANK'] == '0' and os.environ['WORLD_SIZE'] == '1'
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_CODE_PIN']
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
    assert subprocess.check_output(['git', '-C', os.environ['RLINF_PATH'], 'rev-parse', 'HEAD'], text=True).strip() == '8a1233c3fae6f5905adfda7684c455842b0ecfb8'
    assert torch.cuda.device_count() == 1
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    from fluffy_bassoon.meta_rl.robocasa_pairing import pairing_native_factory, public_snapshot, MAX_RGB_DELTA
    from fluffy_bassoon.meta_rl.robocasa_residual import public_observation
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from hydra import compose, initialize_config_dir
    assert MAX_RGB_DELTA == 1
    assert not (OUT / 'manifest.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(OUT); assert stat.f_bavail * stat.f_frsize >= 8 << 30
    (OUT / 'write-probe').write_text('fresh qualification write probe\n')
    subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]gpu_probe.py'), '--expected-count', '1',
        '--purpose', 'training', '--baige-container', '--baige-exclusive-training', '--output', str(OUT / 'gpu-probe.json')], check=True)
    device = json.loads((OUT / 'gpu-probe.json').read_text())['devices'][0]
    os.environ.update(CUDA_VISIBLE_DEVICES=device['visibility_token'], LYCO_BAIGE_GPU_MAP=str(OUT / 'gpu-probe.json'),
        LYCO_BAIGE_ISOLATED_UUID=device['uuid'], MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
    verify_selected_cuda(torch)
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split(); assert int(cpu[0]) / int(cpu[1]) >= 32
    assert int(Path('/sys/fs/cgroup/memory.max').read_text()) >= 128 << 30
    with initialize_config_dir(config_dir=str(CODE / 'configs/native_ppo'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    cfg.env.eval.episode_horizon_source = 'max_episode_steps'
    factory = pairing_native_factory(cfg.env.eval)
    jobs, refs = [], []
    for task in TASKS:
        path = SOURCE / task / 'FAILURE-BANK.json'
        cases = json.loads(path.read_text())['selected']; assert len(cases) == 20
        refs.append(dict(path=str(path), sha256=sha(path)))
        jobs.extend((case, repeat) for case in cases for repeat in range(4))
    assert len(jobs) == 320
    deadline = int(os.environ['TASK_STARTED']) + 450
    write(OUT / 'manifest.json', dict(source_commit=os.environ['TASK_CODE_PIN'], source_banks=refs,
        GPU=device, reset_count=320, policy_controls=0, model_inference=False, PPO_updates=0,
        repeats=4, source_guard_unchanged=MAX_RGB_DELTA, deadline_unix=deadline))

    def reset(job):
        case, repeat = job
        if time.time() >= deadline:
            raise TimeoutError('Finite reset qualification deadline')
        directory = OUT / case['task'] / str(case['native_seed']) / str(repeat)
        directory.mkdir(parents=True, exist_ok=False)
        journal = ActionJournal(OUT / 'actions', rank=0, world=1,
            attempt=case['task'] + '-' + str(case['native_seed']) + '-' + str(repeat), maximum_actions=None)
        intent = journal.begin(128, phase='native_constructor_reset')
        env = None
        try:
            env = factory(case)
            observation, _ = env.reset()
            counts = env.env.get_env_attr('_flora_accounting')[0]
            journal.finish(intent, counts['reset_controls'])
            physical = env.env.get_env_attr('_native_pairing')[0]
            snapshot = public_snapshot(public_observation(observation), case, physical)
            atomic_save(directory / 'initial.pt', snapshot)
            canonical = SOURCE / 'canonical' / case['task'] / str(case['native_seed']) / 'initial.pt'
            reference = torch.load(canonical, map_location='cpu', weights_only=False)
            result = dict(case=case, repeat=repeat, comparison=compare(reference, snapshot), native_counts=counts,
                snapshot=dict(path=str(directory / 'initial.pt'), sha256=sha(directory / 'initial.pt')),
                canonical=dict(path=str(canonical), sha256=sha(canonical)))
            write(directory / 'RESULT.json', result)
            return result
        finally:
            if env is not None:
                env.close()

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(reset, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(dict(job=futures[future], error=repr(error)))
            write(OUT / 'progress.json', dict(complete_resets=len(results), errors=errors, updated_unix=time.time()))
    maximum = max((row['max_delta'] for result in results for row in result['comparison']['images'].values()), default=None)
    write(OUT / 'RESULT.json', dict(completed=len(results) == 320 and not errors, complete_resets=len(results), errors=errors,
        measured_max_RGB_delta=maximum, original_production_bound=1, bound_changed=False,
        policy_controls=0, model_inference=False, PPO_updates=0, next='Independent CPU summary and explicit fresh scientific contract required'))
    if errors or len(results) != 320:
        raise RuntimeError('Rendering qualification incomplete; preserve all snapshots and reset charges')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(); parser.add_argument('--fixture-check', action='store_true')
    args = parser.parse_args()
    fixture_check() if args.fixture_check else main()
