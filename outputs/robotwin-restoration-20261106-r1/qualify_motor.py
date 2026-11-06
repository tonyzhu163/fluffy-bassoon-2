"""Actual-device Motus batch capacity and native residual-motor qualification.

Uses real source observations, no replicated/dummy rows to fill device memory.
This measures model batching with one live scene, not parallel scene capacity.
"""
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import MethodType

from qualify_execution import method, REFERENCE_SHA


BASE = Path('/mowenjie/fluffy-bassoon-2')
MOTOR = BASE/'cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1'
READOUT = BASE/'outputs/robotwin-sidney-endpoint-readout-20261106-r1/readout01'


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''): digest.update(block)
    return digest.hexdigest()


class Memory:
    def __init__(self, torch):
        self.torch, self.stop = torch, threading.Event()
        self.initial_free, self.total = torch.cuda.mem_get_info()
        self.minimum_free = self.initial_free
        self.initial_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            self.minimum_free = min(self.minimum_free, self.torch.cuda.mem_get_info()[0])
            self.stop.wait(.02)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *unused):
        self.stop.set(); self.thread.join()

    def result(self):
        t = self.torch
        # The reserved-pool peak captures allocations between polling ticks.
        minimum = min(self.minimum_free,
            self.initial_free-max(0, t.cuda.max_memory_reserved()-self.initial_reserved))
        return dict(total_bytes=self.total, sampled_minimum_free_bytes=self.minimum_free,
            conservative_minimum_free_bytes=minimum, peak_allocated_bytes=t.cuda.max_memory_allocated(),
            peak_reserved_bytes=t.cuda.max_memory_reserved(), polling_seconds=.02)


def main():
    import numpy as np
    import sapien
    import torch
    from PIL import Image
    from rlinf.models import get_model
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from fluffy_bassoon.baige_gpu import pci
    from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor, case_noise, stack_public_observations
    from fluffy_bassoon.rl.pi05_residual import ReactiveALOHAActionResidual
    from fluffy_bassoon.rl.robotwin_execution import chunk_execution
    from fluffy_bassoon.rl.robotwin_noise import (model_config, environment_config,
        _install_allocated_renderer, _install_reset_retry_limit, _install_synchronous_vector_executor,
        _observation_hash)

    root, code = Path(sys.argv[1]), Path(os.environ['TASK_CODE'])
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1 and root.is_dir() and not (root/'RESULT.json').exists()
    torch.cuda.set_device(0)
    contract_path = (Path(sys.argv[2]) if len(sys.argv) == 3 else
                     code/'outputs/robotwin-restoration-20261106-r1/MOTOR01-CONTRACT.json')
    contract = json.loads(contract_path.read_text())
    native, rlinf = Path(os.environ['ROBOTWIN_PATH']), Path(os.environ['REPO_PATH'])
    for folder, expected in ((native, 'dca9ec682688821c42944eacd7f5dd9bfb15396f'),
                             (rlinf, 'c605b9d55ed93a180119b099e6f836abff293b08')):
        assert subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip() == expected
        assert not subprocess.check_output(['git','-C',str(folder),'status','--porcelain','--untracked-files=no'],text=True).strip()
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(MOTOR/'model.safetensors') == contract['model_sha256']
    assert sha(norm) == contract['norm_stats_sha256']
    assert json.loads((READOUT/'RESULT.json').read_text())['status'] == 'complete'
    assert json.loads((READOUT/'EXIT.json').read_text())['exit_code'] == 0
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], model_path=str(MOTOR),
        model_sha256=contract['model_sha256'], norm_stats_sha256=contract['norm_stats_sha256'],
        horizon=32, episode_horizon=192, flow_steps=5, sampler='native eval ODE with explicit initial noise',
        optimizer_updates=0, profile=[], episodes=[], started_unix=time.time(), warnings=[],
        profile_scope='batched motor on distinct real observations; one resident SAPIEN scene, not N parallel scenes')
    save(root/'RESULT.json', report)
    env, original_scene_step = None, sapien.Scene.step
    physical_steps = {}

    def deadline():
        assert time.time() < float(os.environ['TASK_DEADLINE'])-90
        used = sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
        assert used <= 1*2**30 and shutil.disk_usage(root).free >= 286*2**30+max(0,1*2**30-used)

    def counted_step(scene, *args, **kwargs):
        value = original_scene_step(scene, *args, **kwargs)
        physical_steps[id(scene)] = physical_steps.get(id(scene), 0)+1
        return value

    def rgb(raw):
        return np.stack([raw['main_images'][0].cpu().numpy(),
                         raw['wrist_images'][0,0].cpu().numpy(), raw['wrist_images'][0,1].cpu().numpy()])

    try:
        model = get_model(model_config(rlinf, MOTOR, norm, action_horizon=32, num_steps=5)).cuda()
        motor = FrozenRoboTwinMotor(model)
        assert motor.noise_shape == (32, 32)
        render_pci = _install_allocated_renderer(root/'renderers')
        _install_reset_retry_limit(); sapien.Scene.step = counted_step; os.chdir(native)
        readout = json.loads((READOUT/'RESULT.json').read_text())
        seeds = readout['cases'][:3]
        cfg = environment_config(rlinf, native, 'place_phone_stand', seeds[0], 192)
        cfg.skip_intermediate_renders = True; cfg.malloc_trim_reset_interval = 1
        cfg.task_config.timestep = 1/250
        cfg.task_config.camera.head_camera_type = 'Large_D435'
        cfg.task_config.camera.wrist_camera_type = 'Large_D435'
        env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
        _install_synchronous_vector_executor(env.venv)
        reference = native/'envs/_base_task.py'; assert sha(reference) == REFERENCE_SHA
        patched_root = root/'native-method'; (patched_root/'envs').mkdir(parents=True)
        patched = patched_root/'envs/_base_task.py'; shutil.copyfile(reference, patched)
        subprocess.run(['patch','--fuzz=0','-p1','-d',str(patched_root),'-i',
            str(code/'patches/robotwin-dca9ec68-execution-accounting.patch')],check=True)
        patched_code = method(patched)

        # Real episodes exercise the same frozen batched API used by residual fitting/evaluation.
        for episode_index, seed in enumerate(seeds):
            deadline(); raw, _ = env.reset(env_seeds=[int(seed)])
            task = env.venv.envs[0].task
            device = task.scene.render_system.device
            assert pci(device.pci_string.removeprefix('pci:')) == render_pci and device.can_render()
            assert int(task.ep_num) == int(seed) and raw['task_descriptions'] == [task.get_instruction()]
            images = rgb(raw); assert all(float(v.std()) > 1 for v in images)
            Image.fromarray(np.concatenate(images, axis=1)).save(root/f'episode-{episode_index:02d}-initial.jpg')
            # reset may reuse this task instance. Bind the pristine class method,
            # not the previous episode's instance-local instrumentation closure.
            original = type(task).gen_sparse_reward_data.__get__(task, type(task))
            namespace = dict(original.__func__.__globals__); exec(patched_code, namespace)
            patched_method = MethodType(namespace['gen_sparse_reward_data'], task)
            latest = []

            def measured(*args, **kwargs):
                result = patched_method(*args, **kwargs); latest[:] = [result[3]]; return result

            task.gen_sparse_reward_data = measured
            episode = dict(seed=int(seed), initial_public_sha256=_observation_hash(raw),
                instruction=raw['task_descriptions'], chunks=[], complete=False, requested_targets=0)
            report['episodes'].append(episode)
            for chunk in range(6):
                deadline(); noise = case_noise([f'motor01/{episode_index}/{chunk}'], seed=20261106,
                    horizon=32, action_dim=32, device=model.device)
                if episode_index == 0 and chunk == 0:
                    with torch.no_grad(): reference_actions, _ = model.predict_action_batch(raw,mode='eval',noise=noise)
                actions, features = motor.infer_batch(raw, noise=noise)
                if episode_index == 0 and chunk == 0:
                    torch.testing.assert_close(actions, reference_actions, atol=1e-5, rtol=1e-5)
                    report['native_singleton_max_abs_error'] = float((actions-reference_actions).abs().max())
                    head = ReactiveALOHAActionResidual(public_dim=2062, arm_joint_bound=.02,gripper_bound=.003).cuda()
                    correction = head(actions, features); assert torch.equal(correction.actions, actions)
                    correction.actions.sum().backward()
                    assert torch.isfinite(head.output.bias.grad).all() and head.output.bias.grad.abs().sum() > 0
                    assert all(p.grad is None for p in model.parameters())
                    report['residual_gradient_isolation'] = True
                    del head, correction, reference_actions
                before = physical_steps.get(id(task.scene), 0)
                observations, rewards, terms, truncs, infos = env.chunk_step(actions.float().cpu())
                assert latest
                execution = chunk_execution(latest[0], requested_targets=32,
                                            timestep_seconds=float(task.scene.get_timestep()))
                assert execution.physics_steps == physical_steps.get(id(task.scene), 0)-before
                row = dict(chunk=chunk, execution=asdict(execution), duration_seconds=execution.duration_seconds,
                    valid_target_mask=execution.valid_target_mask, reward=float(torch.as_tensor(rewards).sum()),
                    terminated=bool(torch.as_tensor(terms).any()), truncated=bool(torch.as_tensor(truncs).any()))
                episode['chunks'].append(row); episode['requested_targets'] += 32
                raw = observations[-1]
                save(root/'RESULT.json', report)
                if row['terminated'] or row['truncated']: break
            assert episode['chunks'][-1]['terminated'] or episode['chunks'][-1]['truncated']
            episode['complete'] = True
            Image.fromarray(np.concatenate(rgb(raw),axis=1)).save(root/f'episode-{episode_index:02d}-final.jpg')
            task.gen_sparse_reward_data = original
            save(root/'RESULT.json', report)
        assert sum(c['execution']['physics_steps'] for e in report['episodes'] for c in e['chunks']) > 0

        # Use distinct real observations instead of duplicated rows to fill memory.
        inputs, identities, seen = [], [], set()
        for path in sorted(READOUT.glob('case-*/*/chunk-*-observation.npz')):
            with np.load(path, allow_pickle=False) as data:
                images, state, prompt = data['images'].copy(), data['state'].copy(), str(data['instruction'].item())
            digest = hashlib.sha256(images.tobytes()+state.tobytes()+prompt.encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest); identities.append(str(path.relative_to(READOUT)))
            inputs.append(dict(main_images=torch.from_numpy(images[:1]), wrist_images=torch.from_numpy(images[None,1:]),
                states=torch.from_numpy(state[None]).float(), task_descriptions=[prompt]))
            if len(inputs) == 256: break
        assert len(inputs) >= 16
        save(root/'INPUTS.json',dict(root=str(READOUT),paths=identities,sha256=sorted(seen)))
        ids = [f'profile/{s}' for s in identities]
        draws = case_noise(ids, seed=20261106, horizon=32, action_dim=32, device=model.device)
        reference = []
        for i, raw in enumerate(inputs):
            deadline(); actions, _ = motor.infer_batch(raw,noise=draws[i:i+1]); reference.append(actions.cpu())
        reference = torch.cat(reference)
        for batch in contract['batch_candidates']:
            if batch > len(inputs): break
            deadline(); directory = root/f'batch-{batch:03d}'; directory.mkdir()
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            row = dict(batch=batch, status='starting', real_observations=batch)
            report['profile'].append(row); save(root/'RESULT.json', report)
            try:
                with Memory(torch) as memory:
                    ticks, errors = [], []
                    for repeat in range(2):
                        tick = time.monotonic(); actions, features = motor.infer_batch(
                            stack_public_observations(inputs[:batch]), noise=draws[:batch])
                        torch.cuda.synchronize(); ticks.append(time.monotonic()-tick)
                        errors.append(float((actions.cpu()-reference[:batch]).abs().max()))
                        del actions, features
                row.update(status='complete', seconds=ticks, steady_seconds=ticks[-1],
                    observations_per_second=batch/ticks[-1], max_abs_vs_singleton=max(errors), **memory.result())
                reserve = max(8*2**30, int(.08*row['total_bytes']))
                row['operating_reserve_bytes'] = reserve
                row['memory_safe'] = row['conservative_minimum_free_bytes'] >= reserve
                row['singleton_numerical_parity'] = max(errors) <= contract['batch_action_absolute_tolerance']
                row['eligible'] = row['memory_safe'] and row['singleton_numerical_parity']
            except torch.cuda.OutOfMemoryError as error:
                row.update(status='oom',error=str(error)); gc.collect(); torch.cuda.empty_cache()
            save(directory/'RESULT.json',row);save(root/'RESULT.json',report)
            print(json.dumps(row),flush=True)
            if row['status'] == 'oom' or not row['memory_safe']: break
        eligible = [r for r in report['profile'] if r.get('eligible')]
        assert eligible
        report.update(status='complete', selected_memory_batch=max(r['batch'] for r in eligible),
            fastest_batch=max(eligible,key=lambda r:r['observations_per_second'])['batch'],
            frozen_parameters_and_buffers_unchanged=True, prefix_calls=motor.prefix_calls,
            sample_calls=motor.sample_calls, prepared_rows=motor.prepared_rows, sampled_rows=motor.sampled_rows)
        motor.check_frozen()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        if env is not None: env.venv.close()
        sapien.Scene.step = original_scene_step
        report['ended_unix'] = time.time();save(root/'RESULT.json',report)


if __name__ == '__main__': main()
