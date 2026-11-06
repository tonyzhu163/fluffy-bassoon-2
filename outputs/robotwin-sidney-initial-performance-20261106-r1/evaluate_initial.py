"""Small paired frozen-policy diagnostic on allocated RoboTwin hardware."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

BASE = Path('/mowenjie/fluffy-bassoon-2')
NATIVE = BASE/'cache/robotwin-sidney-qualification-20261105-r1'
CONVERTED = BASE/'cache/robotwin-sidney-rlinf-20261105-r1'
NATIVE_PYTHON = BASE/'envs/lerobot-sidney-92f96f33-20261105-r2/bin/python'
LEROBOT = BASE/'sources/lerobot-sidney-92f96f33-20261105-r1'
MODES = ('native10', 'port10', 'port5', 'sde5_full', 'sde5_fast')


def save(path, value):
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def exposed_state(task):
    return [dict(qpos=a.get_qpos().tolist(), qvel=a.get_qvel().tolist(),
                 root_p=a.get_root_pose().p.tolist(), root_q=a.get_root_pose().q.tolist(),
                 root_linear_velocity=a.get_root_linear_velocity().tolist(),
                 root_angular_velocity=a.get_root_angular_velocity().tolist())
            for a in task.scene.get_all_articulations()]


def main():
    import numpy as np
    import torch
    from PIL import Image
    from fluffy_bassoon.rl.sidney import deterministic_reference, install_rlinf_transforms, model_config
    from fluffy_bassoon.rl.robotwin_noise import (
        environment_config, load_benchmark_episode_limits, _install_allocated_renderer,
        _install_reset_retry_limit, _install_synchronous_vector_executor, _observation_hash)
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng, restore_rng
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    import rlinf.models.embodiment.openpi as factory

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    deterministic_reference()
    manifest = json.loads(args.manifest.read_text())
    root = args.output
    assert root.is_dir() and not (root/'RESULT.json').exists()
    rlinf = Path(os.environ['REPO_PATH']); robotwin = Path(os.environ['ROBOTWIN_PATH'])
    bank = Path(manifest['bank_path'])
    assert sha(bank) == manifest['bank_sha256']
    assert load_benchmark_episode_limits(robotwin, ['place_phone_stand'])['place_phone_stand'] == 400
    cases = manifest['cases']; assert len(cases) == len(set(cases)) == 16
    assert cases == json.loads(bank.read_text())['place_phone_stand']['success_seeds'][:16]
    assert sha(NATIVE/'model.safetensors') == '9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3'
    assert sha(CONVERTED/'model.safetensors') == 'c71114641cf0819900f8b1d7b0568158a12e631b2ca08b2b19bc2d5b761b69b3'
    assert sha(CONVERTED/'sidney-processors.json') == '38ffadf70cf85cff5f8a0a4a6bf23e4886b33081b93f8bb2b20469b6cd571f5b'
    versions = {}
    for directory, expected in ((rlinf, 'c605b9d55ed93a180119b099e6f836abff293b08'),
                                (robotwin, 'dca9ec682688821c42944eacd7f5dd9bfb15396f'),
                                (LEROBOT, '92f96f33b3a842ead3a115cc4f701ba2ad8aa3b5')):
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == expected
        assert not subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        versions[str(directory)] = actual
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], runtime_commits=versions,
                  task='place_phone_stand', horizon=400, batch_size=1, cases=cases, modes=MODES,
                  optimizer_updates=0, first16_existing_eval_cases=True, episodes=[],
                  scientific_scope='Paired diagnostic, not a model-card reproduction or six-task benchmark',
                  full_native_physics_snapshot=False, started_unix=time.time())
    save(root/'RESULT.json', report)
    worker = None; envs = {}; models = {}; parameter_versions = {}
    requested_total = 0

    def deadline():
        assert time.time() < float(os.environ['TASK_DEADLINE'])-90, 'Allocated diagnostic deadline'

    def wait(path):
        limit = time.monotonic()+420
        while not path.exists():
            if worker.poll() is not None:
                raise RuntimeError(f'Native model worker exited {worker.returncode}')
            if time.monotonic() > limit:
                raise TimeoutError(str(path))
            deadline(); time.sleep(.1)

    def images(raw):
        return np.stack([raw['main_images'][0].cpu().numpy(),
                         raw['wrist_images'][0,0].cpu().numpy(),
                         raw['wrist_images'][0,1].cpu().numpy()])

    try:
        _install_allocated_renderer(root/'renderers')
        _install_reset_retry_limit()
        os.chdir(robotwin)
        for mode in MODES:
            cfg = environment_config(rlinf, robotwin, 'place_phone_stand', int(cases[0]), 400,
                                     final_observation_only=False)
            cfg.skip_intermediate_renders = mode == 'sde5_fast'
            cfg.malloc_trim_reset_interval = 1
            cfg.task_config.camera.head_camera_type = 'Large_D435'
            cfg.task_config.camera.wrist_camera_type = 'Large_D435'
            envs[mode] = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
            _install_synchronous_vector_executor(envs[mode].venv)
        worker_out = root/'native-worker'; worker_out.mkdir()
        native_code = Path(os.environ['TASK_CODE'])/'outputs/robotwin-sidney-qualification-20261105-r1/model_worker.py'
        with (root/'native-worker.log').open('w') as log:
            worker = subprocess.Popen([str(NATIVE_PYTHON), str(native_code), '--model', str(NATIVE),
                '--output', str(worker_out)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                text=True, env=dict(os.environ, PYTHONPATH=str(LEROBOT/'src')))
        wait(worker_out/'READY.json')
        report['native_load'] = json.loads((worker_out/'READY.json').read_text())
        assert report['native_load']['strict_weight_load']
        install_rlinf_transforms()
        for key, steps, task in (('port10', 10, 'eval'), ('port5', 5, 'rl')):
            cfg = model_config(rlinf, CONVERTED, flow_steps=steps, task=task)
            cfg.add_value_head = False; cfg.openpi.add_value_head = False
            if task == 'rl':
                cfg.openpi.noise_method = 'flow_sde'; cfg.openpi.noise_level = .3
                cfg.openpi.train_expert_only = True
            model = factory.get_model(cfg).cuda().eval().requires_grad_(False)
            models[key] = model
            parameter_versions[key] = {n: p._version for n,p in model.named_parameters()}
        report['models_loaded'] = True; save(root/'RESULT.json', report)
        for case_index, seed in enumerate(cases):
            used = sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
            assert used <= 12*2**30
            assert shutil.disk_usage(root).free >= 286*2**30+max(0,12*2**30-used)
            # Rotate order without changing case selection or outcome weighting.
            order = MODES[case_index % len(MODES):]+MODES[:case_index % len(MODES)]
            initial = None
            for mode in order:
                deadline()
                directory = root/f'case-{case_index:02d}'/mode; directory.mkdir(parents=True)
                env = envs[mode]; raw, _ = env.reset(env_seeds=[int(seed)])
                sub = env.venv.envs[0]; scene = sub.task.get_info()
                assert int(sub.task.ep_num) == int(sub.env_seed) == int(seed)
                assert sub.episode_info_list == [scene]
                assert sub.task.get_instruction() == sub.instruction == raw['task_descriptions'][0]
                rgb = images(raw)
                assert rgb.shape == (3,480,640,3) and rgb.dtype == np.uint8
                assert all(float(v.std()) > 1 for v in rgb)
                state = raw['states'][0].cpu().numpy().copy()
                assert state.shape == (14,) and np.isfinite(state).all()
                current = dict(seed=int(seed), scene=scene, instruction=raw['task_descriptions'],
                               public_sha256=_observation_hash(raw), exposed_state=exposed_state(sub.task))
                np.savez_compressed(directory/'initial.npz', images=rgb, state=state,
                                    instruction=np.asarray(raw['task_descriptions'][0]))
                save(directory/'initial.json', current)
                if initial is None: initial = current
                assert current == initial, 'Paired initial scene/observation/instruction mismatch'
                Image.fromarray(np.concatenate(rgb,axis=1)).save(directory/'initial.jpg')
                episode = dict(case_index=case_index, seed=int(seed), mode=mode, order=list(order),
                               initial=current, chunks=[], requested_controls=0,
                               measured_executed_controls=0 if mode!='sde5_fast' else None,
                               complete=False)
                report['episodes'].append(episode)
                for chunk in range(8):
                    deadline(); noise_seed = 1000+case_index*100+chunk
                    observation = directory/f'chunk-{chunk:02d}-observation.npz'
                    np.savez_compressed(observation, images=images(raw),
                        state=raw['states'][0].cpu().numpy().astype(np.float32),
                        instruction=np.asarray(raw['task_descriptions'][0]))
                    started = time.monotonic()
                    if mode == 'native10':
                        target = directory/f'chunk-{chunk:02d}-actions.npz'
                        worker.stdin.write(json.dumps(dict(observation=str(observation), actions=str(target),
                            noise_seed=noise_seed, check_queue_parity=False))+'\n'); worker.stdin.flush()
                        wait(target.with_suffix('.json'))
                        with np.load(target,allow_pickle=False) as record: actions = record['actions'].copy()[None]
                        model_info = json.loads(target.with_suffix('.json').read_text())
                        assert model_info['frozen_unchanged']
                    else:
                        model = models['port10' if mode=='port10' else 'port5']
                        rng = snapshot_rng()
                        try:
                            random.seed(noise_seed); np.random.seed(noise_seed)
                            torch.manual_seed(noise_seed); torch.cuda.manual_seed_all(noise_seed)
                            with torch.inference_mode():
                                noise = torch.randn((1,50,32),device='cuda',dtype=torch.float32)
                                predicted, detail = model.predict_action_batch(raw,
                                    mode='train' if mode.startswith('sde5') else 'eval',
                                    compute_values=False, noise=noise)
                            actions = predicted.float().cpu().numpy()
                            model_info = dict(denoise_indices=None if not mode.startswith('sde5') else
                                detail['forward_inputs']['denoise_inds'].cpu().tolist())
                        finally: restore_rng(rng)
                        assert not model.training and not any(p.requires_grad for p in model.parameters())
                        np.save(directory/f'chunk-{chunk:02d}-actions.npy', actions)
                    assert actions.shape == (1,50,14) and np.isfinite(actions).all()
                    inference_seconds = time.monotonic()-started
                    requested_total += 50; assert requested_total <= 32000
                    episode['requested_controls'] += 50
                    observations,rewards,terms,truncs,infos = env.chunk_step(torch.from_numpy(actions.copy()))
                    info = infos[-1]; valid = info.get('valid_action_mask')
                    executed = None if valid is None else int(torch.as_tensor(valid).sum())
                    if mode!='sde5_fast':
                        assert executed is not None and 0 < executed <= 50
                        episode['measured_executed_controls'] += executed
                    success = bool(torch.as_tensor(info['episode']['success_once']).any())
                    terminated = bool(torch.as_tensor(terms).any()); truncated = bool(torch.as_tensor(truncs).any())
                    episode['chunks'].append(dict(chunk=chunk, noise_seed=noise_seed,
                        inference_seconds=inference_seconds, reward=float(torch.as_tensor(rewards).sum()),
                        success=success, terminated=terminated, truncated=truncated,
                        measured_executed_controls=executed, sampling=model_info,
                        native_run_steps=int(sub.task.run_steps),
                        native_reward_step=int(sub.task.reward_step),
                        public_state_after=observations[-1]['states'][0].cpu().tolist(),
                        exposed_state_after=exposed_state(sub.task)))
                    raw = observations[-1]
                    report.update(status='running', requested_controls_total=requested_total,
                                  completed_episodes=sum(e['complete'] for e in report['episodes']))
                    save(root/'RESULT.json', report); save(directory/'episode.json',episode)
                    print(json.dumps(dict(case=case_index,seed=seed,mode=mode,chunk=chunk,
                                          success=success,terminated=terminated,truncated=truncated)),flush=True)
                    if terminated or truncated: break
                episode.update(complete=True, success=any(c['success'] for c in episode['chunks']))
                Image.fromarray(np.concatenate(images(raw),axis=1)).save(directory/'final.jpg')
                save(directory/'episode.json', episode); save(root/'RESULT.json', report)
            # Existing native batch-end allocator behavior, outside scene stepping.
            gc.collect()
        assert len(report['episodes']) == 80 and all(e['complete'] for e in report['episodes'])
        for key, model in models.items():
            assert parameter_versions[key] == {n:p._version for n,p in model.named_parameters()}
        report.update(status='complete', completed_episodes=80, frozen_parameter_versions_unchanged=True,
                      summaries={m:dict(episodes=16,successes=sum(e['success'] for e in report['episodes']
                          if e['mode']==m)) for m in MODES})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for env in envs.values(): env.venv.close()
        if worker is not None:
            worker.stdin.close()
            try: worker.wait(timeout=20)
            except subprocess.TimeoutExpired: worker.terminate(); worker.wait(timeout=20)
        report.update(ended_unix=time.time(), requested_controls_total=requested_total)
        save(root/'RESULT.json', report)


if __name__ == '__main__':
    main()
