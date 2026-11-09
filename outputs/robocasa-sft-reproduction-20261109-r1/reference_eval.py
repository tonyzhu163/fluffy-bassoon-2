"""Frozen native OpenPI evaluation using the released RoboCasa policy boundary."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def with_checkpoint_stats(train_config, stats):
    """Skip training-data fallback using the exact bundled inference normalizer."""
    import dataclasses
    from openpi.training.config import DataConfig

    base = train_config.data.base_config or DataConfig()
    return dataclasses.replace(train_config, data=dataclasses.replace(
        train_config.data, base_config=dataclasses.replace(base, norm_stats=stats)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--maximum-seconds', type=int, default=3000)
    parser.add_argument('--version', choices=['1.0.0', '1.0.1'], default='1.0.1')
    parser.add_argument('--replan', type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.maximum_seconds
    assert socket.gethostname().startswith('job-'), 'Use a scheduler allocation'
    assert args.trials > 0 and args.replan == 5
    journal = args.output / 'events.jsonl'

    def event(**value):
        value['unix'] = time.time()
        with journal.open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
        write(args.output / 'PROGRESS.json', value)
        print(json.dumps(value), flush=True)

    import numpy as np
    import jax
    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi_client import image_tools

    assert robocasa.__version__ == args.version
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu', devices
    native_config = config.get_config('pi05_pretrain_human300')
    assert native_config.model.pi05
    assert native_config.model.action_horizon == 50
    assert native_config.model.action_dim == 32
    assert native_config.model.max_token_len == 200
    norm_hash = sha(args.checkpoint / 'assets/norm_stats.json')
    norm_stats = normalize.load(args.checkpoint / 'assets')
    native_config = with_checkpoint_stats(native_config, norm_stats)
    event(phase='loading_native_original', devices=[str(d) for d in devices],
          robocasa_version=robocasa.__version__, robocasa_file=robocasa.__file__,
          config_file=config.__file__, normalization_sha256=norm_hash,
          model_backend='native JAX', training_updates=0)
    policy = policy_config.create_trained_policy(native_config, args.checkpoint,
                                               sample_kwargs={'num_steps': 10},
                                               norm_stats=norm_stats)
    event(phase='loaded_native_original', load_seconds=time.time() - started)
    results = []
    np.random.seed(7)
    for task in args.tasks:
        horizon = get_task_horizon(task)
        old_horizon = horizon if args.version == '1.0.0' else horizon * 2 // 3
        horizon = old_horizon * 3 // 2
        assert old_horizon * 3 == horizon * 2
        env = gym.make(f'robocasa/{task}', split='pretrain', seed=7)
        try:
            for episode in range(args.trials):
                if time.time() > deadline - 180:
                    raise TimeoutError('Reserved closure time reached')
                case = dict(task=task, episode=episode, split='pretrain', seed=7,
                            horizon=horizon, original_registry_horizon=old_horizon)
                event(phase='reset_intent', **case)
                obs, info = env.reset()
                prompt = obs['annotation.human.task_description']
                action_plan = collections.deque()
                controls = 0
                predictions = 0
                inference_seconds = 0.
                success_step = None
                action_digest = hashlib.sha256()
                first_inputs = args.output / f'{task}-{episode:03d}-first.npz'
                video_path = args.output / f'{task}-{episode:03d}.mp4'
                writer = imageio.get_writer(video_path, fps=20) if episode < 2 else None
                episode_started = time.time()
                try:
                    for control in range(horizon):
                        if time.time() > deadline - 90:
                            raise TimeoutError('Episode exceeded finite allocation deadline')
                        if not action_plan:
                            images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
                                np.ascontiguousarray(obs[key]), 224, 224)) for key in (
                                'video.robot0_agentview_left', 'video.robot0_eye_in_hand',
                                'video.robot0_agentview_right')]
                            state = np.concatenate([obs[key] for key in (
                                'state.end_effector_position_relative',
                                'state.end_effector_rotation_relative', 'state.base_position',
                                'state.base_rotation', 'state.gripper_qpos')], axis=0)
                            element = dict(zip(('observation/image', 'observation/wrist_image',
                                                'observation/right_image'), images))
                            element.update({'observation/state': state, 'prompt': prompt})
                            if predictions == 0:
                                np.savez_compressed(first_inputs, **element)
                            infer_started = time.time()
                            chunk = np.asarray(policy.infer(element)['actions'])
                            inference_seconds += time.time() - infer_started
                            assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                            action_plan.extend(chunk[:args.replan])
                            predictions += 1
                        action = np.asarray(action_plan.popleft())
                        action_digest.update(action.tobytes())
                        if control % 50 == 0:
                            event(phase='control_intent', controls_requested=1,
                                  control=control, predictions=predictions, **case)
                        obs, reward, done, truncated, info = env.step(convert_action(action))
                        controls += 1
                        success = bool(info['success'])
                        if writer is not None and (control % 2 == 0 or success or control == horizon - 1):
                            writer.append_data(image_tools.convert_to_uint8(
                                np.ascontiguousarray(env.render())))
                        if success:
                            success_step = controls
                            break
                finally:
                    if writer is not None:
                        writer.close()
                record = dict(**case, prompt=prompt, controls=controls,
                              success=success_step is not None, success_step=success_step,
                              success_original_horizon=success_step is not None and success_step <= old_horizon,
                              predictions=predictions, inference_seconds=inference_seconds,
                              episode_seconds=time.time() - episode_started,
                              action_sha256=action_digest.hexdigest(),
                              first_inputs_sha256=sha(first_inputs),
                              video_sha256=sha(video_path) if writer is not None else None)
                write(args.output / f'{task}-{episode:03d}.json', record)
                results.append(record)
                event(phase='episode_completed', **record)
        finally:
            env.close()
    summary = dict(completed=True, episodes=len(results), controls=sum(r['controls'] for r in results),
                   elapsed_seconds=time.time() - started, training_updates=0,
                   tasks={task: dict(episodes=sum(r['task'] == task for r in results),
                                     successes=sum(r['task'] == task and r['success'] for r in results),
                                     original_horizon_successes=sum(r['task'] == task and r['success_original_horizon'] for r in results))
                          for task in args.tasks})
    assert sha(args.checkpoint / 'assets/norm_stats.json') == norm_hash
    write(args.output / 'RESULT.json', summary)
    event(phase='completed', **summary)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
