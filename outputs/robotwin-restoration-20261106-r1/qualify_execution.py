"""Real SAPIEN qualification of the passive native accounting patch.

Replay a saved successful and timeout trajectory through original/patched,
fast/full controllers. This is branch coverage, not a new success-rate sample.
The original external checkout stays clean. Only this job's task instances use
the exact patched method, compiled with the original method's globals.
"""
import argparse
import ast
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType

BASE = Path('/mowenjie/fluffy-bassoon-2')
DIAGNOSTIC = BASE/'outputs/robotwin-sidney-deployment-diagnostic-20261106-r1/eval01'
REFERENCE_SHA = '6b3f74b038509ef989913b3f6f37c74e1d0b7156d201a1bf0100f9ccb6b0d46b'
MODES = ('original_fast', 'patched_fast', 'original_full', 'patched_full')


def save(path, value):
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(path):
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == 'gen_sparse_reward_data']
    assert len(functions) == 1
    return compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec')


def main():
    import numpy as np
    import sapien
    import torch
    from fluffy_bassoon.baige_gpu import pci
    from fluffy_bassoon.rl.robotwin_execution import chunk_execution
    from fluffy_bassoon.rl.robotwin_noise import (environment_config,
        _install_allocated_renderer, _install_reset_retry_limit,
        _install_synchronous_vector_executor, _observation_hash)
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.output
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    assert root.is_dir() and not (root/'RESULT.json').exists()
    code = Path(os.environ['TASK_CODE']); native = Path(os.environ['ROBOTWIN_PATH'])
    rlinf = Path(os.environ['REPO_PATH']); reference = native/'envs/_base_task.py'
    assert sha(reference) == REFERENCE_SHA
    for folder, expected in ((native, 'dca9ec682688821c42944eacd7f5dd9bfb15396f'),
                             (rlinf, 'c605b9d55ed93a180119b099e6f836abff293b08')):
        assert subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'], text=True).strip() == expected
        assert not subprocess.check_output(['git','-C',str(folder),'status','--porcelain','--untracked-files=no'], text=True).strip()
    patch = code/'patches/robotwin-dca9ec68-execution-accounting.patch'
    patched_root = root/'native-method'; (patched_root/'envs').mkdir(parents=True)
    patched = patched_root/'envs/_base_task.py'; shutil.copyfile(reference, patched)
    subprocess.run(['patch','--fuzz=0','-p1','-d',str(patched_root),'-i',str(patch)], check=True)
    patched_code = method(patched)
    previous = json.loads((DIAGNOSTIC/'RESULT.json').read_text())
    assert previous['status'] == 'complete' and len(previous['episodes']) == 64
    assert json.loads((DIAGNOSTIC/'EXIT.json').read_text())['exit_code'] == 0
    rows = [e for e in previous['episodes'] if e['mode'] == 'chunk_seen' and e['complete']]
    selected = [next(e for e in rows if e['success']),
                next(e for e in rows if not e['success'] and e['chunks'][-1]['truncated'])]
    selections = []
    for episode in selected:
        ci = int(episode['case_index']); source = DIAGNOSTIC/f'case-{ci:02d}/chunk_seen'
        paths = [source/f'chunk-{i:02d}-actions.npz' for i in range(len(episode['chunks']))]
        assert all(path.is_file() for path in paths)
        selections.append(dict(case_index=ci, seed=int(episode['seed']),
            branch='success' if episode['success'] else 'timeout',
            source_episode=str(source/'episode.json'), source_episode_sha256=sha(source/'episode.json'),
            actions=[dict(path=str(path), sha256=sha(path)) for path in paths]))
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'],
        source_unchanged=True, patch_sha256=sha(patch), reference_sha256=sha(reference),
        patched_source_sha256=sha(patched), source_diagnostic_sha256=sha(DIAGNOSTIC/'RESULT.json'),
        selections=selections, selection_reason='Explicit success/timeout branch coverage; no success-rate claim',
        optimizer_updates=0, model_sampler_calls=0, native_loss_masks_changed=False,
        episodes=[], comparisons=[], started_unix=time.time())
    save(root/'RESULT.json', report)
    counts = {}; frames = {}; tracking = set(); env = None
    original_step = sapien.Scene.step

    def physical(scene):
        values = []
        for actor in scene.get_all_actors():
            pose = actor.get_pose(); values.extend((pose.p, pose.q))
        for articulation in scene.get_all_articulations():
            pose = articulation.get_root_pose()
            values.extend((articulation.get_qpos(), articulation.get_qvel(), pose.p, pose.q,
                           articulation.get_root_linear_velocity(), articulation.get_root_angular_velocity()))
        return np.concatenate(values).astype(np.float64, copy=False)

    def counted_step(scene, *a, **kw):
        result = original_step(scene, *a, **kw); key = id(scene)
        counts[key] = counts.get(key, 0)+1
        if key in tracking: frames[key].append(physical(scene))
        return result

    def deadline():
        assert time.time() < float(os.environ['TASK_DEADLINE'])-90, 'Qualification allocation deadline'
        used = sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
        assert used < 4*2**30, 'Qualification output cap'

    try:
        render_pci = _install_allocated_renderer(root/'renderers')
        _install_reset_retry_limit(); sapien.Scene.step = counted_step; os.chdir(native)
        cfg = environment_config(rlinf, native, 'place_phone_stand', selections[0]['seed'], 400,
                                 final_observation_only=False)
        cfg.skip_intermediate_renders = False; cfg.malloc_trim_reset_interval = 1
        cfg.task_config.timestep = 1/250
        cfg.task_config.camera.head_camera_type = 'Large_D435'
        cfg.task_config.camera.wrist_camera_type = 'Large_D435'
        env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
        _install_synchronous_vector_executor(env.venv)
        for selection in selections:
            paired = None; results = {}
            actions = []
            for item in selection['actions']:
                assert sha(Path(item['path'])) == item['sha256']
                with np.load(item['path'], allow_pickle=False) as data: action = data['actions'].copy()
                assert action.shape == (50, 14) and np.isfinite(action).all(); actions.append(action)
            for mode in MODES:
                deadline(); directory = root/f"case-{selection['case_index']:02d}"/mode
                directory.mkdir(parents=True)
                raw, _ = env.reset(env_seeds=[selection['seed']]); sub = env.venv.envs[0]; task = sub.task
                renderer = task.scene.render_system.device
                assert pci(renderer.pci_string.removeprefix('pci:')) == render_pci and renderer.can_render()
                assert int(task.ep_num) == int(sub.env_seed) == selection['seed']
                assert raw['task_descriptions'] == [sub.instruction] == [task.get_instruction()]
                assert float(raw['main_images'].float().std()) > 1
                assert all(float(view.float().std()) > 1 for view in raw['wrist_images'][0])
                identity = dict(public_sha256=_observation_hash(raw), instruction=raw['task_descriptions'],
                    scene=task.get_info(), physical=physical(task.scene).tolist())
                if paired is None: paired = identity
                assert identity == paired, 'Paired native initial state differs'
                if mode.startswith('patched'):
                    original = task.gen_sparse_reward_data
                    namespace = dict(original.__func__.__globals__)
                    exec(patched_code, namespace)
                    task.gen_sparse_reward_data = MethodType(namespace['gen_sparse_reward_data'], task)
                full = mode.endswith('full'); scene_key = id(task.scene)
                timestep = float(task.scene.get_timestep()); episode = dict(mode=mode,
                    case_index=selection['case_index'], branch=selection['branch'], initial=identity,
                    chunks=[], complete=False, requested_targets=0, measured_physics_steps=0)
                report['episodes'].append(episode); results[mode] = episode
                for ci, action in enumerate(actions):
                    deadline(); before = counts.get(scene_key, 0); frames[scene_key] = []; tracking.add(scene_key)
                    try: result = task.gen_sparse_reward_data(action.copy(), collect_obs=full)
                    finally: tracking.discard(scene_key)
                    measured = counts.get(scene_key, 0)-before
                    trajectory = np.stack(frames.pop(scene_key)) if measured else np.empty((0, len(physical(task.scene))))
                    np.savez_compressed(directory/f'chunk-{ci:02d}-physics.npz', trajectory=trajectory)
                    reward, term, trunc, info = result[:4]
                    decoded = None
                    if mode.startswith('patched'):
                        decoded = asdict(chunk_execution(info, requested_targets=50, timestep_seconds=timestep))
                        assert decoded['physics_steps'] == measured
                    assert ('executed_action_count' in info) == full
                    row = dict(chunk=ci, reward=np.asarray(reward).tolist(), terminated=np.asarray(term).tolist(),
                        truncated=np.asarray(trunc).tolist(), native_info=info, execution=decoded,
                        physics_steps=measured, trajectory_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest(),
                        final_physical=physical(task.scene).tolist(), take_action_cnt=int(task.take_action_cnt))
                    episode['chunks'].append(row); episode['requested_targets'] += 50
                    episode['measured_physics_steps'] += measured
                    del result, trajectory; save(directory/'episode.json', episode); save(root/'RESULT.json', report)
                    print(json.dumps(dict(mode=mode, case=selection['case_index'], chunk=ci, physics=measured)), flush=True)
                    if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()): break
                assert (bool(episode['chunks'][-1]['terminated'][0]) if selection['branch']=='success'
                        else bool(episode['chunks'][-1]['truncated'][0]))
                # Both terminal states must return an actual zero-duration no-op.
                before = counts.get(scene_key, 0); noop = task.gen_sparse_reward_data(actions[-1].copy(), collect_obs=full)
                assert counts.get(scene_key, 0) == before
                episode['noop'] = dict(native_info=noop[3], physics_steps=0)
                if mode.startswith('patched'):
                    execution = chunk_execution(noop[3], requested_targets=50, timestep_seconds=timestep)
                    assert execution.delivered_targets == execution.physics_steps == 0
                    episode['noop']['execution'] = asdict(execution)
                    del task.gen_sparse_reward_data
                episode['complete'] = True; save(directory/'episode.json', episode); save(root/'RESULT.json', report)
                gc.collect()
            for mode in MODES:
                original_mode = 'original_full' if mode.endswith('full') else 'original_fast'
                original = results[original_mode]; candidate = results[mode]
                assert len(original['chunks']) == len(candidate['chunks'])
                for a, b in zip(original['chunks'], candidate['chunks'], strict=True):
                    for key in ('reward','terminated','truncated','physics_steps','trajectory_sha256','final_physical','take_action_cnt'):
                        assert a[key] == b[key], (selection['case_index'], mode, a['chunk'], key)
                    assert {k:v for k,v in b['native_info'].items() if k!='fb2_execution'} == original['chunks'][a['chunk']]['native_info']
                report['comparisons'].append(dict(case=selection['case_index'], reference=original_mode,
                    candidate=mode, all_exposed_physics_ticks_exact=True, reward_termination_counter_exact=True))
            full = results['original_full']; fast = results['patched_fast']
            assert len(full['chunks']) == len(fast['chunks'])
            for a, b in zip(full['chunks'], fast['chunks'], strict=True):
                assert b['execution']['delivered_targets'] == a['native_info']['executed_action_count']
                assert a['trajectory_sha256'] == b['trajectory_sha256']
            report['comparisons'].append(dict(case=selection['case_index'], reference='original_full',
                candidate='patched_fast', delivered_prefix_matches_full_reference=True, all_exposed_physics_ticks_exact=True))
        assert len(report['episodes']) == 8 and all(e['complete'] for e in report['episodes'])
        report.update(status='complete', real_sapien_accounting_qualified=True,
            original_fast_native_loss_masks_preserved=True, measured_duration_qualified=True,
            full_physics_state_recovery_claim=False, camera_pixel_equivalence_claim=False,
            total_requested_targets=sum(e['requested_targets'] for e in report['episodes']),
            total_measured_physics_steps=sum(e['measured_physics_steps'] for e in report['episodes']))
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        sapien.Scene.step = original_step
        if env is not None:
            env.venv.close(); env.venv.env_thread_pool.shutdown(wait=True)
        report['ended_unix'] = time.time(); save(root/'RESULT.json', report)


if __name__ == '__main__': main()
