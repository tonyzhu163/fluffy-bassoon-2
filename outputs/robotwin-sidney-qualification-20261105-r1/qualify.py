"""Real RoboTwin episode, with native Sidney inference in a separate runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_worker import save

MODEL = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-qualification-20261105-r1')
MODEL_PYTHON = '/mowenjie/fluffy-bassoon-2/envs/lerobot-sidney-92f96f33-20261105-r2/bin/python'
LEROBOT = Path('/mowenjie/fluffy-bassoon-2/sources/lerobot-sidney-92f96f33-20261105-r1')
LEROBOT_COMMIT = '92f96f33b3a842ead3a115cc4f701ba2ad8aa3b5'


def wait_for(path, process, seconds=420):
    deadline = time.monotonic() + seconds
    while not path.is_file():
        if process.poll() is not None:
            raise RuntimeError(f'Model worker exited {process.returncode}; inspect model-worker.log')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Model worker did not produce {path.name}')
        time.sleep(.1)


def main():
    import numpy as np
    import torch
    from PIL import Image
    from fluffy_bassoon.rl.robotwin_noise import (
        SOURCE_TASKS, HELDOUT_TASKS, RLINF_COMMIT, ROBOTWIN_COMMIT,
        environment_config, _install_allocated_renderer,
        _install_reset_retry_limit, _install_synchronous_vector_executor,
        _observation_hash,
    )
    from fluffy_bassoon.rl.robotwin_assets import validate_robotwin_assets
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='place_phone_stand', choices=SOURCE_TASKS)
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.episodes <= 16
    args.output.mkdir(parents=True, exist_ok=False)
    rlinf = Path(os.environ['REPO_PATH'])
    robotwin = Path(os.environ['ROBOTWIN_PATH'])
    for root, expected in ((rlinf, RLINF_COMMIT), (robotwin, ROBOTWIN_COMMIT), (LEROBOT, LEROBOT_COMMIT)):
        assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == expected
        assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    asset_receipt = validate_robotwin_assets(robotwin, SOURCE_TASKS + HELDOUT_TASKS)
    seeds = json.loads((rlinf / 'rlinf/envs/sim/robotwin/seeds/eval_seeds.json').read_text())
    # Missing scene banks require their own qualification; never silently substitute seeds.
    cases = seeds[args.task]['success_seeds'][:args.episodes]
    assert len(cases) == args.episodes
    started = time.time()
    report = dict(status='starting', task=args.task, episodes=[], optimizer_updates=0,
                  source_commit=os.environ['TASK_SOURCE'], lerobot_commit=LEROBOT_COMMIT,
                  model_revision='e49e2ab6c11f07511573b67261bd129e88d0a416',
                  horizon=400, chunk_size=50, flow_steps=10, assets=asset_receipt,
                  camera_types=dict(head='Large_D435', wrist='Large_D435'),
                  camera_rgb_shape=[480, 640, 3],
                  observation_capture='native RGB at every control boundary', started_unix=started)
    result = args.output / 'RESULT.json'
    save(result, report)
    worker = env = None
    try:
        model_env = dict(os.environ, PYTHONPATH=str(LEROBOT / 'src'),
                         TOKENIZERS_PARALLELISM='false')
        with (args.output / 'model-worker.log').open('w') as log:
            worker = subprocess.Popen([MODEL_PYTHON, str(Path(__file__).with_name('model_worker.py')),
                '--model', str(MODEL), '--output', str(args.output)],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                text=True, env=model_env)
        wait_for(args.output / 'READY.json', worker)
        report['model'] = json.loads((args.output / 'READY.json').read_text())
        report['status'] = 'model_loaded'
        save(result, report)
        render_pci = _install_allocated_renderer(args.output / 'renderer')
        _install_reset_retry_limit()
        os.chdir(robotwin)
        cfg = environment_config(rlinf, robotwin, args.task, int(cases[0]), 400,
                                 task_config_name='demo_clean', final_observation_only=False)
        # Same calibrated D435 field of view, using RoboTwin's existing
        # 640x480 preset to match the checkpoint's training image metadata.
        cfg.task_config.camera.head_camera_type = 'Large_D435'
        cfg.task_config.camera.wrist_camera_type = 'Large_D435'
        env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
        _install_synchronous_vector_executor(env.venv)
        for episode_index, seed in enumerate(cases):
            raw, _ = env.reset(env_seeds=[int(seed)])
            actual_seed = int(env.venv.envs[0].task.ep_num)
            episode = dict(requested_seed=int(seed), actual_seed=actual_seed,
                           instruction=list(raw['task_descriptions']),
                           reset_sha256=_observation_hash(raw), chunks=[], physical_controls=0)
            report['episodes'].append(episode)
            report['status'] = 'running'
            save(result, report)
            while episode['physical_controls'] < 400:
                index = len(episode['chunks'])
                stem = f'episode-{episode_index:02d}-chunk-{index:02d}'
                obs_file = args.output / (stem + '-obs.npz')
                act_file = args.output / (stem + '-actions.npz')
                images = np.stack([raw['main_images'][0].cpu().numpy(),
                                   raw['wrist_images'][0, 0].cpu().numpy(),
                                   raw['wrist_images'][0, 1].cpu().numpy()])
                assert images.shape == (3, 480, 640, 3) and images.dtype == np.uint8
                assert all(float(v.std()) > 1 for v in images)
                state = raw['states'][0].cpu().numpy().astype(np.float32)
                assert state.shape == (14,) and np.isfinite(state).all()
                np.savez_compressed(obs_file, images=images, state=state,
                                    instruction=np.asarray(raw['task_descriptions'][0]))
                if index == 0:
                    Image.fromarray(np.concatenate(images, axis=1)).save(args.output / f'episode-{episode_index:02d}-initial.png')
                request = dict(observation=str(obs_file), actions=str(act_file),
                    noise_seed=1000 + episode_index*100 + index,
                    check_queue_parity=(episode_index == index == 0))
                worker.stdin.write(json.dumps(request) + '\n'); worker.stdin.flush()
                wait_for(act_file.with_suffix('.json'), worker)
                with np.load(act_file, allow_pickle=False) as data:
                    actions = data['actions'].copy()
                count = min(50, 400-episode['physical_controls'])
                obs, rewards, terms, truncs, infos = env.chunk_step(torch.from_numpy(actions[None, :count]))
                info = infos[-1]
                mask = torch.as_tensor(info['valid_action_mask']).cpu().numpy()
                actual = int(mask.sum())
                assert 0 < actual <= count
                episode['physical_controls'] += actual
                done = bool(torch.as_tensor(terms).any())
                truncated = bool(torch.as_tensor(truncs).any())
                success = bool(torch.as_tensor(info['episode']['success_once']).any())
                row = dict(chunk=index, requested_controls=count, executed_controls=actual,
                    success=success, terminated=done, truncated=truncated,
                    reward=float(torch.as_tensor(rewards).sum()),
                    inference=json.loads(act_file.with_suffix('.json').read_text()))
                episode['chunks'].append(row)
                save(result, report)
                print(json.dumps(dict(task=args.task, episode=episode_index, **row)), flush=True)
                raw = obs[-1]
                if done or truncated:
                    break
            episode.update(success=any(c['success'] for c in episode['chunks']), complete=True,
                           actual_render_pci=render_pci)
            final = np.stack([raw['main_images'][0].cpu().numpy(),
                              raw['wrist_images'][0, 0].cpu().numpy(),
                              raw['wrist_images'][0, 1].cpu().numpy()])
            Image.fromarray(np.concatenate(final, axis=1)).save(args.output / f'episode-{episode_index:02d}-final.png')
            save(result, report)
        report.update(status='complete', successes=sum(e['success'] for e in report['episodes']),
                      episode_count=len(report['episodes']))
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if env is not None:
            env.venv.close()
        if worker is not None:
            worker.stdin.close()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.terminate(); worker.wait(timeout=15)
        report.update(ended_unix=time.time(), wall_seconds=time.time()-started)
        save(result, report)


if __name__ == '__main__':
    main()
