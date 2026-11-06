"""Qualify the shared frozen residual train/eval path on concurrent real scenes.

Four precommitted source episodes, zero optimizer updates. This is runtime and
artifact qualification, not a success-rate estimate or a teacher PPO run.
"""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from qualify_motor import BASE, MOTOR, READOUT, Memory, save, sha


def write_trace(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    with gzip.open(temporary, 'wb', compresslevel=1) as stream:
        torch.save(value, stream)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def main():
    import numpy as np
    import torch
    from PIL import Image
    from rlinf.models import get_model
    from fluffy_bassoon.meta_rl.background_io import BoundedWrites
    from fluffy_bassoon.rl.pi05_residual import ReactiveALOHAActionResidual
    from fluffy_bassoon.rl.robotwin_batch import RoboTwinBatchRuntime
    from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor, case_noise, stack_public_observations
    from fluffy_bassoon.rl.robotwin_noise import model_config, _observation_hash

    root, code = Path(sys.argv[1]), Path(os.environ['TASK_CODE'])
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1 and root.is_dir() and not (root/'RESULT.json').exists()
    torch.cuda.set_device(0)
    contract_path = Path(sys.argv[2]) if len(sys.argv) == 3 else code/'outputs/robotwin-restoration-20261106-r1/BATCH01-CONTRACT.json'
    contract = json.loads(contract_path.read_text())
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(MOTOR/'model.safetensors') == contract['model_sha256']
    assert sha(norm) == contract['norm_stats_sha256']
    prior = json.loads((READOUT/'RESULT.json').read_text())
    assert prior['status'] == 'complete' and json.loads((READOUT/'EXIT.json').read_text())['exit_code'] == 0
    seeds = list(map(int, prior['cases'][:4]))
    assert len(seeds) == len(set(seeds)) == 4
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], optimizer_updates=0,
        task='place_phone_stand', cases=seeds, episode_horizon=192, action_horizon=32, flow_steps=5,
        model_sha256=contract['model_sha256'], norm_stats_sha256=contract['norm_stats_sha256'],
        protocol='native actual-batch ODE; active lanes only, record tail batch and row; same API for residual train/eval',
        episodes=[], batches=[], initial_identities=[], started_unix=time.time(),
        scope='one shared frozen model, two owned simulator processes; no actor-update or larger scene-capacity claim')
    save(root/'RESULT.json', report)
    runtime, writer = None, None

    def deadline():
        assert time.time() < float(os.environ['TASK_DEADLINE'])-120
        used = sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
        cap = contract['maximum_output_gib']*2**30
        assert used <= cap and shutil.disk_usage(root).free >= 286*2**30+max(0, cap-used)

    def image(raw, name):
        images = np.stack((raw['main_images'][0], raw['wrist_images'][0,0], raw['wrist_images'][0,1]))
        assert all(float(v.std()) > 1 for v in images), 'Blank actual simulator camera'
        Image.fromarray(np.concatenate(images, axis=1)).save(root/name)

    try:
        model = get_model(model_config(os.environ['REPO_PATH'], MOTOR, norm, action_horizon=32, num_steps=5)).cuda()
        motor = FrozenRoboTwinMotor(model)
        head = ReactiveALOHAActionResidual(public_dim=2062, arm_joint_bound=.02, gripper_bound=.003).cuda()
        writer = BoundedWrites(maximum_jobs=2, maximum_bytes=64*2**20)
        runtime = RoboTwinBatchRuntime.create(motor, lanes=2, rlinf_root=os.environ['REPO_PATH'],
            robotwin_root=os.environ['ROBOTWIN_PATH'], output=root/'runtime',
            task_episode_limits={'place_phone_stand':192}, seed=20261107)
        pids = [p.pid for p in runtime.pool.processes]
        assert len(set(pids)) == 2 and os.getpid() not in pids
        report['worker_pids'], report['parent_pid'] = pids, os.getpid()
        motion, saved = [], []
        with Memory(torch) as memory:
            for round_index, mode in enumerate(('train_head_gradient', 'eval_head')):
                deadline()
                episodes = []
                receipts = runtime.reset('place_phone_stand', seeds[2*round_index:2*round_index+2],
                    case_ids=[f'batch01/round-{round_index}/episode-{i}' for i in range(2)])
                for lane, receipt in enumerate(receipts):
                    raw = runtime._raw[lane]  # Qualification only; private monitoring data never enters the head.
                    assert receipt['actual_seed'] == seeds[2*round_index+lane]
                    assert receipt['worker_pid'] == pids[lane] and receipt['episode_limit'] == 192
                    assert receipt['instruction'] == raw['task_descriptions']
                    assert receipt['public_observation_sha256'] == _observation_hash(raw)
                    image(raw, f'round-{round_index}-lane-{lane}-initial.jpg')
                    report['initial_identities'].append(receipt)
                    episode = dict(seed=receipt['actual_seed'], lane=lane, mode=mode,
                        case_id=runtime._cases[lane], chunks=[], complete=False)
                    episodes.append(episode); report['episodes'].append(episode)
                assert len({v['actual_render_pci'] for v in receipts}) == 1
                for decision in range(6):
                    deadline()
                    active = [i for i, v in enumerate(runtime._active) if v]
                    if not active: break
                    native = None
                    if decision == 0:
                        ids = [f'{runtime._cases[i]}/chunk-0' for i in active]
                        noise = case_noise(ids, seed=20261107, horizon=32, action_dim=32, device=model.device)
                        with torch.no_grad():
                            native, _ = model.predict_action_batch(
                                stack_public_observations([runtime._raw[i] for i in active]), mode='eval', noise=noise)
                    tick = time.monotonic()
                    base, features, metadata = runtime.infer()
                    if native is not None:
                        error = float((base-native).abs().max())
                        assert error <= 1e-5
                        report.setdefault('native_batch_max_abs_errors', []).append(error)
                        del native
                    if mode == 'train_head_gradient':
                        head.train(); correction = head(base, features)
                        assert torch.equal(correction.actions, base)
                        correction.actions.sum().backward()
                        assert torch.isfinite(head.output.bias.grad).all() and head.output.bias.grad.abs().sum() > 0
                        assert all(p.grad is None for p in model.parameters())
                        head.zero_grad(set_to_none=True)
                        report['head_gradient_isolation'] = True
                    else:
                        head.eval()
                        with torch.no_grad(): correction = head(base, features)
                        assert torch.equal(correction.actions, base)
                    inference_seconds = time.monotonic()-tick
                    # Durable request before physical dispatch; preserve returned
                    # transitions before any qualification-only assertion can fail.
                    from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree
                    intent = f'round-{round_index}-decision-{decision}-intent.pt.gz'
                    write_trace(root/intent, cpu_tree(dict(native_batch=metadata,
                        observations=runtime._pending['observations'],base_actions=base,
                        requested_actions=correction.actions,public_features=features)))
                    tick = time.monotonic(); transitions = runtime.step(correction.actions)
                    step_seconds = time.monotonic()-tick
                    paths = []
                    for row, transition in enumerate(transitions):
                        lane = metadata['lanes'][row]
                        relative = f'round-{round_index}-decision-{decision}-lane-{lane}.pt.gz'
                        writer.submit(write_trace, str(root/relative), transition)
                        saved.append(relative); paths.append(relative)
                    save(root/'TRACES.json', dict(paths=saved,complete=False))
                    report['batches'].append(dict(mode=mode, **metadata, inference_seconds=inference_seconds,
                        simulator_step_seconds=step_seconds,intent=intent))
                    save(root/'RESULT.json', report)
                    for row, transition in enumerate(transitions):
                        lane = metadata['lanes'][row]
                        assert transition['audit']['worker_pid'] == pids[lane]
                        assert transition['native_batch']['row'] == row
                        assert transition['native_batch']['batch_size'] == len(active)
                        assert np.array_equal(transition['requested_actions'], transition['base_actions'])
                        assert transition['public_features'].shape == (2062,)
                        assert transition['execution']['requested_targets'] == 32
                        assert np.array_equal(transition['measured_valid_target_mask'],
                            np.arange(32) < transition['execution']['delivered_targets'])
                        timestep = transition['execution']['timestep_seconds']
                        assert np.isclose(timestep, 1/250, rtol=1e-6, atol=0)
                        assert transition['physics_duration_seconds'] == transition['execution']['physics_steps']*timestep
                        motion.append(float(np.abs(transition['next_observation']['states']-
                                                   transition['observation']['states']).max()))
                        relative = paths[row]
                        episodes[lane]['chunks'].append(dict(path=relative, reward=transition['reward'],
                            execution=transition['execution'], terminated=transition['terminated'],
                            truncated=transition['truncated'], native_batch=transition['native_batch']))
                        if transition['terminated'] or transition['truncated']:
                            episodes[lane]['complete'] = True
                            image(transition['next_observation'], f'round-{round_index}-lane-{lane}-final.jpg')
                    save(root/'RESULT.json', report)
                    del correction, base, features, transitions
                assert all(e['complete'] for e in episodes) and not any(runtime._active)
            assert max(motion) > .001
            writer.close(); report['io'] = writer.stats(); writer = None
            motor.check_frozen()
            assert motor.prefix_calls == motor.sample_calls == runtime.native_batches == len(report['batches'])
            report.update(prefix_calls=motor.prefix_calls, sample_calls=motor.sample_calls,
                sampled_rows=motor.sampled_rows, max_state_motion=max(motion),
                frozen_parameter_and_buffer_versions_unchanged=True, traces=saved)
            runtime.close()
            assert not any(p.is_alive() for p in runtime.pool.processes)
            report['owned_workers_closed'] = True
        report['memory'] = memory.result()
        assert report['memory']['conservative_minimum_free_bytes'] >= max(8*2**30, int(.08*report['memory']['total_bytes']))
        for name in ('memory.peak', 'memory.current'):
            path = Path('/sys/fs/cgroup')/name
            if path.is_file(): report[name] = int(path.read_text())
        save(root/'TRACES.json', dict(paths=saved,complete=True))
        subprocess.run([sys.executable, str(Path(__file__).with_name('verify_batch_traces.py')), str(root)], check=True)
        report.update(status='complete', fresh_process_readback=json.loads((root/'READBACK.json').read_text()))
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        try:
            if writer is not None: writer.close()
        finally:
            if runtime is not None: runtime.close()
            report['ended_unix'] = time.time(); save(root/'RESULT.json', report)


if __name__ == '__main__': main()
