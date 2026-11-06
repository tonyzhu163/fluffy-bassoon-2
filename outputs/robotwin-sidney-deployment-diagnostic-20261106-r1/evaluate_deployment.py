"""Frozen Sidney controller/prompt comparison inside one allocated GPU job."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path('/mowenjie/fluffy-bassoon-2')
NATIVE = BASE/'cache/robotwin-sidney-qualification-20261105-r1'
PYTHON = BASE/'envs/lerobot-sidney-92f96f33-20261105-r2/bin/python'
LEROBOT = BASE/'sources/lerobot-sidney-92f96f33-20261105-r1'
MODES = ('chunk_seen', 'single_seen', 'chunk_taskname', 'single_taskname')


def save(path, value):
    tmp = path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def exposed_state(task):
    return [dict(qpos=a.get_qpos().tolist(), qvel=a.get_qvel().tolist(),
        root_p=a.get_root_pose().p.tolist(), root_q=a.get_root_pose().q.tolist(),
        root_linear_velocity=a.get_root_linear_velocity().tolist(),
        root_angular_velocity=a.get_root_angular_velocity().tolist())
        for a in task.scene.get_all_articulations()]


def main():
    import numpy as np
    import sapien
    import torch
    from PIL import Image
    from fluffy_bassoon.rl.sidney import deterministic_reference
    from fluffy_bassoon.rl.robotwin_noise import (
        environment_config, load_benchmark_episode_limits, _install_allocated_renderer,
        _install_reset_retry_limit, _install_synchronous_vector_executor, _observation_hash)
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from robotwin.envs.vector_env import update_obs

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0); deterministic_reference()
    root = args.output; manifest = json.loads(args.manifest.read_text())
    assert root.is_dir() and not (root/'RESULT.json').exists()
    rlinf = Path(os.environ['REPO_PATH']); robotwin = Path(os.environ['ROBOTWIN_PATH'])
    assert sha(NATIVE/'model.safetensors') == '9a5381c3260fc58fdb4b90d17b93f9090bcbfbcd8e0287f9bf2acea2477a2bc3'
    bank = Path(manifest['bank_path']); assert sha(bank) == manifest['bank_sha256']
    cases = manifest['cases']; assert len(cases) == len(set(cases)) == 16
    assert cases == json.loads(bank.read_text())['place_phone_stand']['success_seeds'][:16]
    assert load_benchmark_episode_limits(robotwin, ['place_phone_stand'])['place_phone_stand'] == 400
    versions = {}
    for folder, expected in ((rlinf, 'c605b9d55ed93a180119b099e6f836abff293b08'),
                            (robotwin, 'dca9ec682688821c42944eacd7f5dd9bfb15396f'),
                            (LEROBOT, '92f96f33b3a842ead3a115cc4f701ba2ad8aa3b5')):
        actual = subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        assert actual == expected
        assert not subprocess.check_output(['git','-C',str(folder),'status','--porcelain','--untracked-files=no'],text=True).strip()
        versions[str(folder)] = actual
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], runtime_commits=versions,
        task='place_phone_stand', cases=cases, modes=MODES, horizon=400, chunk=50, flow_steps=10,
        batch_size=1, optimizer_updates=0, policy='original strict-loaded LeRobot Sidney', episodes=[],
        requested_action_slots=0, measured_physics_scene_steps=0, started_unix=time.time(),
        scientific_scope='Controller/prompt diagnosis, not an official benchmark reproduction',
        physics_snapshot_complete=False, physical_duration_matched=False, camera_near_changed=False)
    save(root/'RESULT.json', report)
    worker = None; envs = {}; counter = {}; requested = 0
    original_step = sapien.Scene.step

    def counted_step(scene, *a, **kw):
        result = original_step(scene, *a, **kw)
        counter[id(scene)] = counter.get(id(scene), 0)+1
        return result

    def deadline():
        assert time.time() < float(os.environ['TASK_DEADLINE'])-90, 'Allocated diagnostic deadline'

    def wait(path):
        until = time.monotonic()+420
        while not path.exists():
            assert worker.poll() is None, f'Native worker exited {worker.returncode}'
            assert time.monotonic() < until, f'Native worker timeout: {path}'
            deadline(); time.sleep(.1)

    def rgb(raw):
        return np.stack([raw['main_images'][0].cpu().numpy(),
            raw['wrist_images'][0,0].cpu().numpy(), raw['wrist_images'][0,1].cpu().numpy()])

    try:
        _install_allocated_renderer(root/'renderers'); _install_reset_retry_limit()
        sapien.Scene.step = counted_step
        os.chdir(robotwin)
        for mode in MODES:
            cfg = environment_config(rlinf, robotwin, 'place_phone_stand', int(cases[0]), 400,
                                     final_observation_only=False)
            cfg.skip_intermediate_renders = False; cfg.malloc_trim_reset_interval = 1
            cfg.task_config.timestep = 1/250
            cfg.task_config.camera.head_camera_type = 'Large_D435'
            cfg.task_config.camera.wrist_camera_type = 'Large_D435'
            envs[mode] = RoboTwinEnv(cfg,1,seed_offset=0,total_num_processes=1,worker_info=None)
            _install_synchronous_vector_executor(envs[mode].venv)
        native_out = root/'native-worker'; native_out.mkdir()
        code = Path(os.environ['TASK_CODE'])/'outputs/robotwin-sidney-qualification-20261105-r1/model_worker.py'
        with (root/'native-worker.log').open('w') as log:
            worker = subprocess.Popen([str(PYTHON),str(code),'--model',str(NATIVE),'--output',str(native_out)],
                stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,PYTHONPATH=str(LEROBOT/'src')))
        wait(native_out/'READY.json')
        report['native_load'] = json.loads((native_out/'READY.json').read_text())
        assert report['native_load']['strict_weight_load'] and counter
        report['models_loaded'] = True; save(root/'RESULT.json',report)
        for ci, seed in enumerate(cases):
            used = sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
            assert used <= 12*2**30 and shutil.disk_usage(root).free >= 298*2**30+max(0,12*2**30-used)
            order = MODES[ci % 4:]+MODES[:ci % 4]; paired = None
            for mode in order:
                deadline(); directory = root/f'case-{ci:02d}'/mode; directory.mkdir(parents=True)
                env = envs[mode]; raw,_ = env.reset(env_seeds=[int(seed)])
                sub = env.venv.envs[0]; task = sub.task
                assert int(task.ep_num) == int(sub.env_seed) == int(seed)
                assert sub.episode_info_list == [task.get_info()]
                assert raw['task_descriptions'] == [sub.instruction] == [task.get_instruction()]
                assert int(task.take_action_cnt) == 0 and not task.eval_success
                timestep = float(task.scene.get_timestep()); assert abs(timestep-.004) < 1e-9
                identity = dict(seed=int(seed),scene=task.get_info(),public_sha256=_observation_hash(raw),
                    generated_instruction=raw['task_descriptions'],exposed_state=exposed_state(task))
                save(directory/'initial-physical.json',identity)
                if paired is None: paired = identity
                assert identity == paired, 'Paired initial physical/scene/generated-prompt mismatch'
                instruction = 'place phone stand' if mode.endswith('taskname') else sub.instruction
                task.set_instruction(instruction); sub.instruction = instruction; raw['task_descriptions'] = [instruction]
                image = rgb(raw); assert image.shape == (3,480,640,3)
                assert all(float(v.std()) > 1 for v in image)
                np.savez_compressed(directory/'initial.npz',images=image,
                    state=raw['states'][0].cpu().numpy(),instruction=np.asarray(instruction))
                Image.fromarray(np.concatenate(image,axis=1)).save(directory/'initial.jpg')
                episode = dict(case_index=ci,seed=int(seed),mode=mode,instruction=instruction,
                    controller='take_action per target' if mode.startswith('single') else 'gen_sparse_reward_data per chunk',
                    initial=identity,chunks=[],complete=False,success=False,
                    requested_action_slots=0,measured_delivered_action_boundaries=0,measured_physics_scene_steps=0)
                report['episodes'].append(episode)
                for chunk in range(8):
                    deadline(); observation = directory/f'chunk-{chunk:02d}-observation.npz'
                    np.savez_compressed(observation,images=rgb(raw),state=raw['states'][0].cpu().numpy(),
                                        instruction=np.asarray(instruction))
                    target = directory/f'chunk-{chunk:02d}-actions.npz'; noise_seed = 1000+ci*100+chunk
                    worker.stdin.write(json.dumps(dict(observation=str(observation),actions=str(target),
                        noise_seed=noise_seed,check_queue_parity=False))+'\n'); worker.stdin.flush()
                    wait(target.with_suffix('.json'))
                    receipt = json.loads(target.with_suffix('.json').read_text()); assert receipt['frozen_unchanged']
                    with np.load(target,allow_pickle=False) as file: actions = file['actions'].copy()
                    assert actions.shape == (50,14) and np.isfinite(actions).all()
                    requested += 50; assert requested <= 25600
                    physics_before = counter.get(id(task.scene),0); control_before = int(task.take_action_cnt)
                    started = time.monotonic(); calls = []
                    if mode.startswith('single'):
                        for ai, action in enumerate(actions):
                            deadline(); before = counter.get(id(task.scene),0); action_before = int(task.take_action_cnt)
                            assert action_before < 400 and not task.eval_success
                            with torch.enable_grad(): task.take_action(action)
                            assert int(task.take_action_cnt) == action_before+1
                            # LeRobot obtains a public observation after each target, including the final one.
                            latest = task.get_obs()
                            success = bool(task.eval_success or task.check_success())
                            calls.append(dict(action_index=ai,physics_scene_steps=counter.get(id(task.scene),0)-before,
                                take_action_cnt=int(task.take_action_cnt),success=success))
                            if success or int(task.take_action_cnt) >= 400: break
                        obs = update_obs(latest); obs['instruction'] = instruction
                        raw = env._extract_obs_image([obs]); executed = len(calls)
                        terminated = success; truncated = int(task.take_action_cnt) >= 400
                    else:
                        observations,rewards,terms,truncs,infos = env.chunk_step(torch.from_numpy(actions[None].copy()))
                        valid = infos[-1].get('valid_action_mask'); assert valid is not None
                        executed = int(torch.as_tensor(valid).sum()); assert 0 < executed <= 50
                        success = bool(torch.as_tensor(infos[-1]['episode']['success_once']).any())
                        terminated = bool(torch.as_tensor(terms).any()); truncated = bool(torch.as_tensor(truncs).any())
                        raw = observations[-1]
                    physics = counter.get(id(task.scene),0)-physics_before; assert physics > 0
                    assert task.scene is sub.task.scene
                    record = dict(chunk=chunk,noise_seed=noise_seed,inference=receipt,
                        controller_wall_seconds=time.monotonic()-started,success=success,terminated=terminated,
                        truncated=truncated,requested_action_slots=50,measured_delivered_action_boundaries=executed,
                        native_counter_delta=int(task.take_action_cnt)-control_before,physics_scene_steps=physics,
                        physics_timestep_seconds=timestep,integrated_physics_seconds=physics*timestep,
                        single_target_calls=calls,public_state_after=raw['states'][0].cpu().tolist(),
                        exposed_state_after=exposed_state(task))
                    episode['chunks'].append(record); episode['success'] |= success
                    episode['requested_action_slots'] += 50
                    episode['measured_delivered_action_boundaries'] += executed
                    episode['measured_physics_scene_steps'] += physics
                    report.update(status='running',requested_action_slots=requested,
                        measured_physics_scene_steps=sum(e['measured_physics_scene_steps'] for e in report['episodes']))
                    save(directory/'episode.json',episode); save(root/'RESULT.json',report)
                    print(json.dumps(dict(case=ci,mode=mode,chunk=chunk,success=success,physics_steps=physics)),flush=True)
                    if terminated or truncated: break
                episode['complete'] = True
                Image.fromarray(np.concatenate(rgb(raw),axis=1)).save(directory/'final.jpg')
                save(directory/'episode.json',episode); save(root/'RESULT.json',report)
            gc.collect()
        assert len(report['episodes']) == 64 and all(e['complete'] for e in report['episodes'])
        report.update(status='complete',completed_episodes=64,
            summaries={m:dict(episodes=16,successes=sum(e['success'] for e in report['episodes'] if e['mode']==m)) for m in MODES})
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error)); raise
    finally:
        for env in envs.values(): env.venv.close()
        sapien.Scene.step = original_step
        if worker is not None:
            worker.stdin.close()
            try: worker.wait(timeout=20)
            except subprocess.TimeoutExpired: worker.terminate(); worker.wait(timeout=20)
        report['ended_unix'] = time.time(); save(root/'RESULT.json',report)


if __name__ == '__main__': main()
