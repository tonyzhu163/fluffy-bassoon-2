"""Small development panel using the release's observation/action interface."""
from collections import deque
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

FILES = Path(__file__).resolve().parent
CODE = FILES.parents[1]
SPLIT = 'libero_spatial_selected_combinations_inverse_view'
TASKS = ['pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate',
         'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2']
BOWLS = ['akita_black_bowl_1', 'akita_black_bowl_2']


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def axis_angle(quat):
    # Same arithmetic as the release's main_parallel.py, without mutating obs.
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - quat[3] * quat[3])
    return np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(quat[3]) / den


def policy_input(obs, prompt):
    from openpi_client import image_tools
    def image(key):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs[key][::-1, ::-1]), 224, 224))
    return {'observation/image': image('agentview_image'),
            'observation/wrist_image': image('robot0_eye_in_hand_image'),
            'observation/state': np.concatenate((obs['robot0_eef_pos'],
                axis_angle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])),
            'prompt': prompt}


def positions(env):
    return np.stack([env.sim.data.body_xpos[env.env.obj_body_id[n]].copy() for n in BOWLS])


def run_episode(env, state, prompt, client, folder, max_steps,
                intended_bowl=None, reference_positions=None):
    import imageio.v2 as imageio
    from PIL import Image
    folder.mkdir()
    obs = env.set_init_state(state)
    restored_sha = sha(env.get_sim_state())
    reset_check = None
    if reference_positions is not None:
        restored = {n: env.sim.data.body_xpos[env.env.obj_body_id[n]].copy()
                    for n in reference_positions}
        # Static fixture poses may live in the model, outside the saved qpos.
        # Record their actual positions as well as the flattened state hash.
        reset_check = {'restored_world_positions': {n: v.tolist() for n, v in restored.items()},
                       'position_errors': {n: float(np.linalg.norm(v - reference_positions[n]))
                                           for n, v in restored.items()}}
    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    prompt_pixels = None
    if intended_bowl is not None:
        import admit
        admit.SIZE = obs['agentview_image'].shape[0]
        prompt_pixels = {n: admit.project(env, p) for n, p in zip(BOWLS, positions(env))}
        # Resolve wording after the exact reset and warm-up used by the policy.
        left = max(BOWLS, key=lambda n: prompt_pixels[n][0])
        prompt = f"pick up the {'left' if intended_bowl == left else 'right'} black bowl and place it on the plate"
    initial = policy_input(obs, prompt)
    Image.fromarray(initial['observation/image']).save(folder / 'policy-input.png')
    Image.fromarray(obs['agentview_image'][::-1]).save(folder / 'initial-upright.png')
    np.save(folder / 'initial-state.npy', env.get_sim_state())
    initial_true = bool(env.check_success())
    if initial_true:
        raise RuntimeError('goal_already_true_after_settling')
    plan, actions, rewards, poses, latencies, successes = deque(), [], [], [positions(env)], [], []
    error = None
    start = time.monotonic()
    client.reset()
    with imageio.get_writer(folder / 'episode.mp4', fps=4) as video:
        try:
            for step in range(max_steps):
                if step % 5 == 0:
                    video.append_data(obs['agentview_image'][::-1])
                if not plan:
                    t = time.monotonic()
                    chunk = np.asarray(client.infer(policy_input(obs, prompt))['actions'])
                    latencies.append(time.monotonic() - t)
                    if chunk.ndim != 2 or chunk.shape[0] < 5 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
                        raise RuntimeError(f'invalid_action_chunk:{chunk.shape}')
                    plan.extend(chunk[:5])
                action = plan.popleft()
                obs, reward, done, _ = env.step(action.tolist())
                actions.append(action); rewards.append(float(reward)); poses.append(positions(env))
                success = bool(env.check_success()); successes.append(success)
                if success or done:
                    break
        except Exception as exc:
            error = f'{type(exc).__name__}:{exc}'
        video.append_data(obs['agentview_image'][::-1])
    Image.fromarray(obs['agentview_image'][::-1]).save(folder / 'final-upright.png')
    poses = np.asarray(poses)
    np.savez_compressed(folder / 'trace.npz', actions=np.asarray(actions), rewards=rewards,
                        bowl_positions=poses, success=successes, inference_seconds=latencies)
    return {'prompt': prompt, 'native_success': bool(env.check_success()),
            'error': error, 'controls': len(actions), 'elapsed_seconds': time.monotonic() - start,
            'restored_state_sha256': restored_sha, 'input_state_sha256': sha(state),
            'reset_position_check': reset_check, 'prompt_upright_pixels': prompt_pixels,
            'policy_image_sha256': sha(initial['observation/image']),
            'bowl_names': BOWLS,
            'maximum_bowl_displacement': np.linalg.norm(poses - poses[0], axis=-1).max(0).tolist(),
            'final_bowl_displacement': np.linalg.norm(poses[-1] - poses[0], axis=-1).tolist(),
            'inference_seconds': latencies, 'final_bowl_positions': poses[-1].tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['native', 'generated'], required=True)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--port', type=int, default=8027)
    parser.add_argument('--max-steps', type=int, default=550)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import yaml
    base = args.source / 'behavior_prompting/train_network'
    libero = args.source / 'deps/LIBERO/libero/libero'
    config = args.output / 'libero_config'; config.mkdir()
    (config / 'config.yaml').write_text(yaml.safe_dump({'benchmark_root': str(libero),
        'assets': str(libero / 'assets'), 'bddl_files': str(base / 'env/libero/bddl_files'),
        'init_states': str(base / 'env/libero/init_files'), 'datasets': str(args.output)}))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    trials, skipped = [], []
    if args.mode == 'native':
        import torch
        for task_i, task in enumerate(TASKS):
            states = torch.load(base / 'env/libero/init_files' / SPLIT / (task + '.pruned_init'),
                                map_location='cpu', weights_only=False)
            for index in (0, 1):
                trials.append({'id': f'native-{task_i}-{index}', 'seed': 7 + task_i,
                    'bddl': str(base / 'env/libero/bddl_files' / SPLIT / (task + '.bddl')),
                    'state': np.asarray(states[index]), 'prompt': None})
    else:
        if args.geometry is None:
            raise ValueError('--geometry is required for generated scenes')
        sys.path.insert(0, str(CODE / 'exploration/libero-scenes'))
        from scenes import to_bddl
        manifest = json.loads((FILES / 'SCENES.json').read_text())
        inspection = json.loads((args.geometry / 'inspection/result.json').read_text())
        records = {r['index']: r for r in inspection['records']}
        for scene in manifest['scenes']:
            record = records[scene['index']]
            if not record['admitted']:
                skipped.append({'scene': scene['index'], 'geometry_errors': record['errors']})
                continue
            # Admission uses the upright camera. The author horizontally mirrors
            # that view for this checkpoint; resolve left/right in the model view.
            left = max(BOWLS, key=lambda n: record['pixels'][n][0])
            for bowl in BOWLS:
                target = 'plate_1'
                behaviour = next(b for b in scene['behaviours'] if b['bowl'] == bowl and b['target'] == target)
                trial_id = f"scene-{scene['index']:04d}-bowl-{BOWLS.index(bowl)+1}"
                bddl = args.output / (trial_id + '.bddl')
                bddl.write_text(to_bddl(dict(scene, behaviours=[behaviour])))
                trials.append({'id': trial_id, 'seed': scene['seed'], 'bddl': str(bddl),
                    'state': np.load(args.geometry / 'inspection' / f"scene-{scene['index']:04d}" / 'state.npy'),
                    'prompt': f"pick up the {'left' if bowl == left else 'right'} black bowl and place it on the plate",
                    'intended_bowl': bowl, 'target': target, 'scene': scene['index'],
                    'reference_positions': record['world_positions'],
                    'prompt_reference': 'left/right in horizontally mirrored policy image'})
    results = []
    (args.output / 'result.json').write_text(json.dumps({'mode': args.mode, 'episodes': [],
        'skipped': skipped, 'planned_episodes': len(trials), 'complete': not trials,
        'successes': 0, 'weight_updates': False}, indent=2) + '\n')
    for trial in trials:
        env = None
        result = {k: v for k, v in trial.items() if k != 'state'}
        try:
            env = OffScreenRenderEnv(bddl_file_name=trial['bddl'], camera_heights=256,
                                    camera_widths=256, render_gpu_device_id=0)
            env.env.hard_reset = False
            env.seed(trial['seed']); env.env.reset()
            result.update(run_episode(env, trial['state'], trial['prompt'] or env.language_instruction,
                                      client, args.output / trial['id'], args.max_steps,
                                      trial.get('intended_bowl'), trial.get('reference_positions')))
        except Exception as exc:
            result.update(native_success=False, error=f'{type(exc).__name__}:{exc}')
        finally:
            if env is not None:
                env.close()
        results.append(result)
        summary = {'mode': args.mode, 'episodes': results, 'skipped': skipped,
                   'complete': len(results) == len(trials), 'planned_episodes': len(trials),
                   'successes': sum(r['native_success'] for r in results), 'weight_updates': False}
        (args.output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('PI05_EPISODE', json.dumps({k: v for k, v in result.items() if k != 'inference_seconds'}), flush=True)


if __name__ == '__main__':
    main()
