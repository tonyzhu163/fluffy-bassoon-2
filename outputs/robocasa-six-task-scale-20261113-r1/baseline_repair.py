"""Fill the missing OpenDrawer SFT/PPO30 comparison with 64 fresh paired cases.

One allocated GPU, evaluation only. Reuses the qualified native reference motor,
canonical reset, common-noise, action audit and bounded archive implementation.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from labels import sha, write

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')


def helper():
    path = HERE.parent / 'robocasa-dit-paired-confirmation-20261114-r1/readout.py'
    assert sha(path) == 'cffb005f07df84a7961ffe33795529574e8f4743a69b2ef6477884118752cc53'
    spec = importlib.util.spec_from_file_location('qualified_native_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args, spec, local, cold):
    sys.path[:0] = [str(CODE / 'src'), os.environ['RLINF_PATH']]
    import torch
    from hydra import compose, initialize_config_dir
    from fluffy_bassoon.baige_gpu import selected_device, verify_selected_cuda
    from fluffy_bassoon.rlinf.native_ppo import ActionJournal, tensor_hash
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowReferenceMotor, ReferencePlaceholder
    from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _load_native, _versions, native_video_writer
    from fluffy_bassoon.robocasa.steering_history_native import NativeAttemptBackend, case_key
    from rlinf.models.embodiment.openpi import get_model
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(cold / 'gpu-probe.json')
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    model = ROOT / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    norm = ROOT / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(local / 'native'), ROBOCASA_PI05_TORCH_MODEL=str(model),
                      ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, factory, signature = _load_native(dict(model_directory=str(model), norm_path=str(norm),
        source_commit=args.pin, runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1'), CODE)
    assert signature['model_sha256'] == spec['SFT_sha256'] and signature['norm_sha256'] == spec['norm_sha256']
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    codec = DirectActionCodec.from_stats(stats)
    policy = ReferencePlaceholder().to('cuda:0').eval().requires_grad_(False)
    weights = local / 'teacher/full_weights.pt'
    assert sha(weights) == spec['teacher']['sha256']
    with initialize_config_dir(config_dir=str(CODE / 'benchmarks/robocasa/configs'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    cfg.rollout.model.model_path = str(weights.parent)
    cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
    teacher = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
    motors = dict(SFT=FlowReferenceMotor(native, codec, native.model), PPO30=FlowReferenceMotor(native, codec, teacher))
    versions = [tensor_hash(_versions(model)) for model in (native.model, teacher)]
    audit = helper()
    binding = dict(format='fixed_PPO30_baseline_repair_v1', source_commit=args.pin, spec_sha256=args.spec_sha256,
        signature=signature, teacher=spec['teacher'], render_contract='bounded_two_intensity_levels',
        recording='native_rgb_frame_digests_v1', deployment_optimizer_updates=0,
        pairing='Same canonical physical state and native Gaussian draws, one physical attempt per policy')
    write(cold / 'BINDING.json', binding)
    results = []
    for first in range(0, 64, 8):
        if time.time() + 300 >= args.deadline:
            write(cold / 'INCOMPLETE.json', dict(next_first=first, results=results, reason='deadline'))
            return
        wave = local / f'bank-{first:04d}'
        assert shutil.disk_usage(local).free > 20 * 2**30
        cases = [dict(case=dict(bank=spec['case_bank'], task='OpenDrawer', split='target',
            native_seed=spec['seed_base'] + slot, slot=slot, reset_ordinal=0, counted=True,
            layout_source='native_target_constructor_seed_first_reset'), horizon=600) for slot in range(first, first + 8)]
        paired = {}
        order = ('SFT', 'PPO30') if first // 8 % 2 == 0 else ('PPO30', 'SFT')
        for arm in order:
            directory = wave / arm
            journal = ActionJournal(directory / 'accounting', rank=0, world=1, attempt=args.attempt,
                                    maximum_actions=8 * (600 + 128))
            backend = NativeAttemptBackend(output=directory / 'attempts', canonical_root=wave / 'canonical',
                motor=motors[arm], spec=codec, native_factory=factory.paired, journal=journal,
                writer_factory=native_video_writer, binding=dict(binding, arm=arm), maximum_rgb_delta=2,
                memory_mode='episode', actual_attempts=1, first_attempt=1, fixed_case_order=True, record_video=False)
            began = time.time()
            episodes = backend.run_bank(policy, cases, lanes=8, attempts=1, deadline_unix=args.deadline - 120)
            assert len(episodes) == 8 and backend.last_batch_sizes == [8] * 12
            assert [tensor_hash(_versions(model)) for model in (native.model, teacher)] == versions
            checked = audit.audit(backend.references, action_space=cfg.env.eval.action_space, reset_policy=policy)
            endpoint = {case_key(row['case']): row for row in checked}
            assert len(endpoint) == 8 and all(row['attempt'] == 1 for row in checked)
            paired[arm] = dict(endpoint=endpoint, physical_episodes=8, physical_controls=4800,
                               elapsed_seconds=time.time() - began, all_native_evidence_checked=True)
            write(directory / 'RESULT.json', paired[arm])
            write(cold / 'PROGRESS.json', dict(checked_unix=time.time(), first=first, arm=arm, completed_waves=len(results)))
        assert paired['SFT']['endpoint'].keys() == paired['PPO30']['endpoint'].keys()
        compact = dict(first=first, cases=cases, arms=paired, order=order)
        write(wave / 'PAIRED.json', compact)
        receipt = audit.bank_wave(wave, cold / f'bank-{first:04d}')
        write(cold / f'bank-{first:04d}/PAIRED.json', compact)
        results.append(dict(first=first, archive=receipt))
        write(cold / 'PROGRESS.json', dict(checked_unix=time.time(), completed_waves=len(results), results=results))
    write(cold / 'COMPLETE.json', dict(completed=True, results=results, cases=64, physical_episodes=128,
                                      binding=binding, optimizer_updates=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--deadline', type=float)
    args = p.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert args.attempt.isalnum()
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    assert sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    assert spec['task'] == 'OpenDrawer' and spec['cases'] == 64 and spec['teacher']['step'] == 30
    assert spec['scheduler_minutes'] == 90 and spec['reserved_GPU_hours'] == 1.5
    qualification = spec['native_qualification']
    assert sha(qualification['path']) == qualification['sha256'] and json.loads(Path(qualification['path']).read_text())['completed']
    local = Path('/tmp') / HERE.name / args.attempt
    cold = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name / args.attempt
    if args.worker:
        assert args.deadline and local.exists() and cold.exists()
        worker(args, spec, local, cold)
        return
    local.mkdir(parents=True, exist_ok=False); cold.mkdir(parents=True, exist_ok=False)
    assert local.stat().st_dev != ROOT.stat().st_dev and shutil.disk_usage(local).free > 40 * 2**30
    probe = cold / '.write-probe'; probe.write_text('baseline repair'); assert probe.read_text() == 'baseline repair'; probe.unlink()
    started = time.time(); deadline = started + 80 * 60
    write(cold / 'LAUNCH.json', dict(source_commit=args.pin, spec=spec, spec_sha256=args.spec_sha256,
        hostname=socket.gethostname(), started_unix=started, maximum_GPU_hours=1.5, automatic_retry=False))
    env = dict(os.environ, RLINF_PATH=str(ROOT / 'sources/critic-selected-a1229239-r1'),
        PYTHONPATH=str(CODE / 'src') + ':' + str(ROOT / 'sources/critic-selected-a1229239-r1'),
        ROBOCASA_ASSETS_PATH=str(ROOT / 'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(ROOT / 'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', NCCL_P2P_DISABLE='1', FLUFFY_GPU_EXECUTION='baige',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        assigned = env['NVIDIA_VISIBLE_DEVICES'].split(',')
        assert len(visible) == len(assigned) == 1 and set(visible) == set(assigned)
        env['CUDA_VISIBLE_DEVICES'] = '0'
    subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]gpu_probe.py'), '--expected-count', '1',
        '--purpose', 'evaluation', '--baige-container', '--output', str(cold / 'gpu-probe.json')], env=env, check=True, timeout=180)
    assert json.loads((cold / 'gpu-probe.json').read_text())['status'] == 'passed'
    weights = local / 'teacher/full_weights.pt'; weights.parent.mkdir()
    shutil.copyfile(spec['teacher']['path'], weights); assert sha(weights) == spec['teacher']['sha256']
    rc = 1
    try:
        with (cold / 'console.log').open('x') as log:
            rc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--pin', args.pin,
                '--spec', str(args.spec), '--spec-sha256', args.spec_sha256, '--attempt', args.attempt,
                '--worker', '--deadline', str(deadline)], env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, deadline + 300 - time.time())).returncode
    except BaseException as error:
        write(cold / 'ERROR.json', dict(error=repr(error), checked_unix=time.time()))
    completed = rc == 0 and (cold / 'COMPLETE.json').exists()
    if not completed:
        helper().bank_wave(local, cold / 'partial-evidence')
    write(cold / 'EXIT.json', dict(completed=completed, exit_code=rc, ended_unix=time.time(),
                                 allocation_GPU_hours=(time.time() - started) / 3600))
    raise SystemExit(0 if completed else 1)


if __name__ == '__main__':
    main()
