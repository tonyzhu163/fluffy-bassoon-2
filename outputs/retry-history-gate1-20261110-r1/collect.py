"""Gate 1 collection: repeated native attempts from one exact start, with inputs and actions saved.

    collect.py OUT CKPT TASK [SCENE_START] [SCENES] [REPEATS]

Frozen native Human300 pi0.5, unchanged closed loop (predict 50, execute 5, ten denoising
steps, 600 controls). Every repeat on a scene shares one exact simulator and public start;
only the motor-noise seed differs. Each episode writes one compressed archive holding the
policy inputs at every inference step, every predicted chunk and every executed action.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time

import gymnasium as gym
import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'icrl-idea-checks-20261110-r1'))
from checks import (HORIZON, INPUT, config, convert_action, inputs, matched_start, normalize, physics,
                    policy_config, public_hash, stable_counter_regions, with_checkpoint_stats)

FIRST_SCENE = 986000          # development seeds; no earlier run uses this range
MIN_FREE_BYTES = 60 * 2**30   # about 25 MB per episode expected; stop early if the disk is short


def attempt(policy, task, scene, repeat, start, path):
    env, obs = start
    try:
        raw = env.unwrapped.env
        initial = (physics(raw), public_hash(obs))
        noise_seed = scene + 700000 + 1000 * repeat
        noise = np.random.default_rng(noise_seed).standard_normal((HORIZON // 5, 50, 32), dtype=np.float32)
        images = {key: [] for key in INPUT}
        states, chunks, actions = [], [], []
        prompt, success, chunk = None, False, None
        for t in range(HORIZON):
            if t % 5 == 0:
                x = inputs(obs)
                chunk = np.asarray(policy.infer(x, noise=noise[t // 5])['actions'])
                assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                for key in INPUT:
                    images[key].append(x[key])
                states.append(np.asarray(x['observation/state'], np.float32))
                chunks.append(chunk.astype(np.float32))
                prompt = str(x['prompt'])
            action = chunk[t % 5].copy()
            actions.append(action.astype(np.float32))
            obs, _, done, truncated, info = env.step(convert_action(action))
            success = bool(info['success'])
            if success or done or truncated:
                break
        controls = t + 1
        partial = path.with_name(path.stem + '.partial.npz')
        np.savez_compressed(
            partial, states=np.stack(states), chunks=np.stack(chunks), actions=np.stack(actions),
            success=np.bool_(success), controls=np.int32(controls), noise_seed=np.int64(noise_seed),
            prompt=np.str_(prompt), **{key.replace('/', '__'): np.stack(images[key]) for key in INPUT})
        os.replace(partial, path)
        return dict(task=task, scene=scene, repeat=repeat, noise_seed=noise_seed, success=success,
                    controls=controls, inferences=len(chunks), prompt=prompt,
                    initial_sha256=initial[0], initial_observation_sha256=initial[1],
                    terminal_sha256=physics(raw), archive=path.name, archive_bytes=path.stat().st_size)
    finally:
        env.close()


def collect(policy, log, out, task, first, scenes, repeats):
    for scene in range(first, first + scenes):
        if shutil.disk_usage(out).free < MIN_FREE_BYTES:
            raise SystemExit(f'Less than {MIN_FREE_BYTES} bytes free before scene {scene}')
        start = target = None
        for repeat in range(repeats):
            env_obs, target = matched_start(task, scene, repeat, target, log)
            row = attempt(policy, task, scene, repeat, env_obs, out / f'{task}-{scene}-{repeat}.npz')
            current = (row['initial_sha256'], row['initial_observation_sha256'])
            if start is not None and current != start:
                raise ValueError(f'Native reset mismatch for {task} {scene} repeat {repeat}')
            start = current
            log('episode', check='gate1', **row)


def main():
    out, ckpt, task, *rest = sys.argv[1:]
    out, ckpt = Path(out), Path(ckpt)
    first = int(rest[0]) if rest else FIRST_SCENE
    scenes = int(rest[1]) if len(rest) > 1 else 16
    repeats = int(rest[2]) if len(rest) > 2 else 8
    assert first >= FIRST_SCENE and scenes > 0 and repeats > 1

    def log(kind, **row):
        row = dict(kind=kind, unix=time.time(), **row)
        with (out / 'events.jsonl').open('a') as stream:
            print(json.dumps(row), file=stream, flush=True)
        print(json.dumps(row), flush=True)
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    gym.make(f'robocasa/{task}', split='pretrain', seed=first).close()   # the task must resolve before the policy loads
    stats = normalize.load(ckpt / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={'num_steps': 10}, norm_stats=stats)
    log('plan', task=task, first_scene=first, scenes=scenes, repeats=repeats, horizon=HORIZON)
    collect(policy, log, out, task, first, scenes, repeats)
    log('complete', check='gate1', task=task, first_scene=first, scenes=scenes, repeats=repeats)


if __name__ == '__main__':
    main()
