"""Paired frozen-base camera/horizon diagnostic; never a PPO learning claim.

Near-plane hook adapted from RLinf's Apache-2.0 robotwin_env.py upstream fix.
External native checkouts stay unchanged; the hook lives in owned processes.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, cpu_tree, json_metrics, sha, write
from fluffy_bassoon.rl.robotwin_batch import FastMeasuredSimulator, RoboTwinBatchRuntime, accounted_controller_source
from fluffy_bassoon.rl.robotwin_production_process import ProductionSimulatorPool
from qualify_batch_runtime import write_trace

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


class NearPlaneScene:
    def __init__(self, scene, near, requests):
        self.scene, self.near, self.requests = scene, near, requests

    def add_camera(self, *args, **kwargs):
        original = kwargs.get('near')
        if self.near is not None: kwargs['near'] = self.near
        camera = self.scene.add_camera(*args, **kwargs)
        actual = getattr(camera, 'near', None)
        self.requests.append(dict(original_near=original, selected_near=kwargs.get('near'),
                                  exposed_actual_near=float(actual) if actual is not None else None))
        return camera

    def __getattr__(self, name): return getattr(self.scene, name)


def install_camera_near(camera_cls):
    if getattr(camera_cls, '_rt_probe_near_installed', False): return
    init, load = camera_cls.__init__, camera_cls.load_camera
    @functools.wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._rt_probe_near = kwargs.get('camera_near')
    @functools.wraps(load)
    def wrapped_load(self, scene, *args, **kwargs):
        self._rt_probe_near_requests = []
        return load(self, NearPlaneScene(scene, self._rt_probe_near,
                    self._rt_probe_near_requests), *args, **kwargs)
    camera_cls.__init__, camera_cls.load_camera = wrapped_init, wrapped_load
    camera_cls._rt_probe_near_installed = True


class CameraProbeSimulator(FastMeasuredSimulator):
    def _environment_config(self, task, seed, limit):
        cfg = super()._environment_config(task, seed, limit)
        from envs.camera import Camera
        install_camera_near(Camera)
        cfg.task_config.camera_near = self.config['worker_options']['camera_near']
        return cfg

    def reset(self, task, seed):
        raw, receipt = super().reset(task, seed)
        from envs.camera import Camera
        cameras = [v for v in vars(self.env.venv.envs[0].task).values() if isinstance(v, Camera)]
        assert cameras, 'No constructed native Camera attestation'
        requests = [row for camera in cameras for row in camera._rt_probe_near_requests]
        near = self.config['worker_options']['camera_near']
        expected = .1 if near is None else near
        assert len(requests) >= 3 and all(row['selected_near'] == expected for row in requests)
        assert all(row['exposed_actual_near'] is None or np.isclose(row['exposed_actual_near'], expected, rtol=1e-6, atol=1e-8) for row in requests)
        receipt.update(camera_near_requests=requests, camera_near=expected)
        return raw, receipt


def preflight(plan):
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    class Scene:
        def add_camera(self, name, **kwargs):
            class Camera: pass
            camera = Camera(); camera.near = kwargs['near']; return camera
    class Camera:
        def __init__(self, **kwargs): pass
        def load_camera(self, scene):
            return [scene.add_camera(str(i), near=.1, far=100.) for i in range(3)]
    install_camera_near(Camera); first = Camera.load_camera; install_camera_near(Camera)
    assert Camera.load_camera is first
    for near in (None, .01, None, .01):
        camera = Camera(camera_near=near); values = camera.load_camera(Scene())
        expected = .1 if near is None else near
        assert [v.near for v in values] == [expected]*3
        assert len(camera._rt_probe_near_requests) == 3
    from fluffy_bassoon.rl.robotwin_noise import environment_config
    for horizon in (192,384):
        cfg = environment_config(os.environ['REPO_PATH'],os.environ['ROBOTWIN_PATH'],
            'place_phone_stand',plan['source_dev_cases'][0],horizon,
            task_config_name='demo_clean',final_observation_only=False)
        cfg.task_config.camera_near = .01
        assert cfg.task_config.step_lim == cfg.max_episode_steps == horizon
        assert cfg.task_config.camera_near == .01
    subprocess.run(['bash','-n',str(ROOT/'probe_motus_camera.sh')],check=True)
    for name, digest in plan['source_files'].items(): assert sha(ROOT.parents[1]/name) == digest
    assert sha(ROOT/'probe_motus_camera.py') == plan['entry_sha256']
    assert sha(ROOT/'probe_motus_camera.sh') == plan['launcher_sha256']
    output = BASE/'cpumotuscamera01'; output.mkdir()
    write(output/'RESULT.json', dict(status='passed', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, models_or_simulator_constructed=False, hook_idempotent=True,
        default_preserved=True, explicit_override_all_three_cameras=True,
        per_camera_instance_no_cross_reset_leak=True, actual_gpu_pairing_pending=True))


def main(args):
    plan = json.loads((ROOT/'MOTUSCAMERA01-CONTRACT.json').read_bytes())
    if args.preflight: return preflight(plan)
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    from PIL import Image
    from rlinf.models import get_model
    from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor
    from fluffy_bassoon.rl.robotwin_noise import model_config, _observation_hash
    from qualify_motor import MOTOR, Memory
    root, local = Path(os.environ['TASK_OUT']), Path(os.environ['TASK_LOCAL'])
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(MOTOR/'model.safetensors') == plan['model_sha256']
    assert sha(norm) == plan['norm_stats_sha256']
    cases = plan['source_dev_cases']; assert len(cases) == len(set(cases)) == 8
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], cases=cases,
        optimizer_updates=0, task='place_phone_stand', native_batch=2, action_horizon=32,
        flow_steps=5, sampler='native ODE with case-addressed initial noise',
        no_cross_batch_equivalence_claim=True, no_learning_gain_claim=True,
        profiles={}, traces=[], started_unix=time.time())
    journal = ActionJournal(root/'action-accounting', rank=0, world=1,
        attempt='camera01', maximum_actions=plan['maximum_requested_targets'])
    model = get_model(model_config(os.environ['REPO_PATH'], MOTOR, norm, action_horizon=32,num_steps=5)).cuda()
    motor = FrozenRoboTwinMotor(model)
    initial, images, prefixes = {}, {}, {}
    runtime = None
    def deadline():
        assert time.time() < float(os.environ['TASK_DEADLINE'])-60
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < plan['maximum_output_gib']*2**30
    try:
        with Memory(torch) as memory:
            for profile in plan['profiles']:
                deadline(); name, near, horizon = profile['name'], profile['camera_near'], profile['horizon']
                options = accounted_controller_source(os.environ['ROBOTWIN_PATH'], local/name/'controller')
                options['camera_near'] = near
                pool = ProductionSimulatorPool(2, rlinf_root=os.environ['REPO_PATH'],
                    robotwin_root=os.environ['ROBOTWIN_PATH'], output_dir=local/name/'simulators',
                    task_episode_limits={'place_phone_stand':horizon}, final_observation_only=False,
                    task_config_name='demo_clean', _simulator_factory=CameraProbeSimulator, worker_options=options)
                runtime = RoboTwinBatchRuntime(motor,pool,seed=plan['motor_noise_seed'])
                entry = dict(camera_near=.1 if near is None else near, horizon=horizon, episodes=[])
                report['profiles'][name] = entry; write(root/'RESULT.json', report)
                for pair in range(4):
                    deadline(); selected = cases[pair*2:pair*2+2]
                    receipts = runtime.reset('place_phone_stand', selected,
                        case_ids=[f'camera-probe/case-{seed}' for seed in selected])
                    episodes = [dict(case=seed, success=False, complete=False, decisions=0) for seed in selected]
                    entry['episodes'].extend(episodes)
                    entry.setdefault('resets',[]).extend(receipts)
                    for lane, (case, receipt) in enumerate(zip(selected,receipts)):
                        raw = runtime._raw[lane]
                        state = np.asarray(raw['states']).copy()
                        identity = (state.tobytes(), json.dumps(json_metrics(receipt['scene']),sort_keys=True), receipt['instruction'])
                        if case not in initial: initial[case] = identity
                        assert identity == initial[case], 'Paired initial physical/public identity differs'
                        pixels = np.stack([raw['main_images'][0],raw['wrist_images'][0,0],raw['wrist_images'][0,1]])
                        key = (entry['camera_near'],case)
                        if key in images: assert np.array_equal(pixels,images[key])
                        else: images[key] = pixels.copy()
                        Image.fromarray(np.concatenate(pixels,axis=1)).save(root/f'{name}-case-{case}-initial.png')
                    decision = 0
                    while any(runtime._active):
                        deadline()
                        base, features, metadata = runtime.infer(fixed_native_batch=True)
                        for lane, case in enumerate(selected):
                            key = (entry['camera_near'],case,decision)
                            signature = (_observation_hash(runtime._raw[lane]),
                                hashlib.sha256(base[lane].detach().cpu().numpy().tobytes()).hexdigest())
                            if key in prefixes: assert signature == prefixes[key], 'Same-camera horizon prefixes differ'
                            else: prefixes[key] = signature
                        label = f'{name}-pair-{pair}-decision-{decision:02d}'
                        write_trace(root/(label+'-intent.pt.gz'),cpu_tree(dict(
                            observations=runtime._pending['observations'],commands=base,native_batch=metadata)))
                        requested = len(metadata['active_lanes'])*32
                        cursor = journal.begin(requested,phase=name)
                        transitions = runtime.step(base)
                        write_trace(root/(label+'-transitions.pt.gz'),transitions)
                        journal.finish(cursor,requested)  # Charge every request, including undelivered tails.
                        report['traces'].append(dict(profile=name,pair=pair,decision=decision,
                            intent=label+'-intent.pt.gz',transitions=label+'-transitions.pt.gz'))
                        for transition in transitions:
                            lane = transition['native_batch']['row']; episode = episodes[lane]
                            episode['decisions'] += 1
                            episode['success'] |= bool(transition['audit']['success'])
                            episode['complete'] = transition['terminated'] or transition['truncated']
                        decision += 1
                        write(root/'RESULT.json', report)
                    assert all(ep['complete'] for ep in episodes)
                motor.check_frozen(); runtime.close()
                assert not any(p.is_alive() for p in runtime.pool.processes)
                runtime = None; entry['owned_workers_closed'] = True
                entry['successes'] = sum(ep['success'] for ep in entry['episodes'])
                write(root/'RESULT.json', report)
        report['memory'] = memory.result()
        assert report['memory']['conservative_minimum_free_bytes'] >= max(8*2**30,.08*report['memory']['total_bytes'])
        report['initial_pixel_difference_by_case'] = {
            str(case): [float(np.abs(images[.1,case][i].astype(float)-images[.01,case][i]).mean()) for i in range(3)]
            for case in cases}
        report.update(status='completed_paired_frozen_camera_horizon_probe', total_episodes=32,
            charged_requested_targets=journal.charged(), frozen_model_unchanged=True,
            independent_readback_pending=True, native_motor_calls=motor.sample_calls)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error)); raise
    finally:
        if runtime is not None: runtime.close()
        report['ended_unix'] = time.time(); write(root/'RESULT.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args())
