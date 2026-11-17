"""Collect a matched native OpenDrawer panel with a persistent approach calibration offset."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import gymnasium as gym
import jax
import numpy as np
import robocasa
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config
from openpi_client import image_tools
from robocasa.utils.env_utils import convert_action

CODE = Path(__file__).resolve().parents[2]
if str(CODE / 'src') not in sys.path:
    sys.path.insert(0, str(CODE / 'src'))
from fluffy_bassoon.robocasa.pairing import stable_counter_regions

CAM = ('video.robot0_agentview_left', 'video.robot0_eye_in_hand', 'video.robot0_agentview_right')
STATE = tuple('state.' + k for k in (
    'end_effector_position_relative', 'end_effector_rotation_relative',
    'base_position', 'base_rotation', 'gripper_qpos'))
INPUT = ('observation/image', 'observation/wrist_image', 'observation/right_image')
HORIZON = 600
GATE_M = 0.10


class RestoreMismatch(Exception):
    pass


def parse_seed_spec(spec):
    seeds = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            lo, hi = map(int, part.split(':', 1))
            seeds.extend(range(lo, hi))
        else:
            seeds.append(int(part))
    return seeds


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def with_checkpoint_stats(train_config, stats):
    from openpi.training.config import DataConfig
    base = train_config.data.base_config or DataConfig()
    return dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, base_config=dataclasses.replace(base, norm_stats=stats)),
    )


def digest(*values):
    h = hashlib.sha256()
    for x in values:
        h.update(x.encode() if isinstance(x, str) else np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def inputs(obs):
    images = [
        image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[k]), 224, 224))
        for k in CAM
    ]
    return {
        **dict(zip(INPUT, images)),
        'observation/state': np.concatenate([obs[k] for k in STATE]),
        'prompt': obs['annotation.human.task_description'],
    }


def physics(raw):
    return digest(raw.sim.get_state().flatten(), raw.sim.data.cam_xpos)


def public_hash(obs):
    x = inputs(obs)
    return digest(x['observation/state'], x['prompt'], *(x[k] for k in INPUT))


def handle(raw):
    model = raw.sim.model
    geoms = [(i, model.geom_id2name(i)) for i in range(model.ngeom)]
    target = [
        (i, n) for i, n in geoms
        if n and n.startswith(raw.drawer.naming_prefix)
        and 'handle' in n and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert target, ('no target collision geoms', raw.drawer.name)
    center = [i for i, n in target if 'connector' not in n and 'support' not in n]
    return [n for _, n in target], center or [i for i, _ in target]


def geometry(raw, robot, centers):
    return (
        np.asarray(raw.sim.data.site_xpos[robot.eef_site_id['right']])
        - np.asarray(raw.sim.data.geom_xpos)[centers].mean(axis=0)
    )


def reset_env(scene):
    random.seed(scene)
    np.random.seed(scene)
    env = gym.make('robocasa/OpenDrawer', split='pretrain', seed=scene)
    obs, _ = env.reset(seed=scene)
    return env, obs


def start_signature(env, obs):
    raw = env.unwrapped.env
    public = inputs(obs)
    return dict(
        sim_state=digest(raw.sim.get_state().flatten()),
        camera_pose=digest(raw.sim.data.cam_xpos, raw.sim.data.cam_xmat),
        body_geometry=digest(raw.sim.model.body_pos, raw.sim.model.geom_pos),
        public_state=digest(public['observation/state']),
        prompt=digest(public['prompt']),
        **{name: digest(public[name]) for name in INPUT},
    )


def matched_start(scene, arm, target, log):
    seen = set()
    first = None
    for attempt in range(1, 17):
        env, obs = reset_env(scene)
        try:
            fields = start_signature(env, obs)
            key = digest(*fields.values())
            match = (key == target) if target is not None else (key in seen)
            log('reset_check', scene=scene, arm=arm, attempt=attempt, signature=key, selected=match,
                differs_from_first=[] if first is None else [k for k in fields if fields[k] != first[k]])
            if match:
                return (env, obs), key
            seen.add(key)
            if first is None:
                first = fields
        except BaseException:
            env.close()
            raise
        env.close()
    raise RestoreMismatch(f'No reproducible native start for scene {scene}, arm {arm}')


def rollout(policy, out, scene, arm, *, target_start=None, tape=None, force_prefix=False,
            replay_all=False, bias=0.0, residual=0.0, window=100, log=None):
    (env, obs), start_key = matched_start(scene, arm, target_start, log)
    try:
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.part_controllers['right']
        assert controller.input_ref_frame == 'base' and controller.input_type == 'delta'
        gain = float(((controller.output_max - controller.output_min) /
                      (controller.input_max - controller.input_min))[1])
        assert 0.0 < gain < 1.0
        names, centers = handle(raw)
        initial_state = physics(raw)
        initial_obs = public_hash(obs)
        nominal_goal = controller.compute_goal_pos
        decision = tape['decision'] if (tape is not None and force_prefix) else None
        goal_calls = 0
        active = False

        def shifted(*args, **kwargs):
            nonlocal goal_calls
            goal = nominal_goal(*args, **kwargs)
            if active:
                goal_calls += 1
                if bias:
                    goal = goal.copy()
                    goal[1] += bias
            return goal

        controller.compute_goal_pos = shifted
        actions, hashes, observations, misses, realized = [], [], [], [], []
        raw_a1_post, clipped_steps = [], 0
        noise = np.random.default_rng(scene + 700000).standard_normal((HORIZON // 5, 50, 32), dtype=np.float32)
        first_grasp = None
        decision_obs = None
        decision_state = None
        post20_state = None
        post20_control = None
        first_state_diff = first_command_diff = first_command_max_abs = None
        chunk = None
        for t in range(HORIZON):
            fp = physics(raw)
            hashes.append(fp)
            if tape is not None:
                if (replay_all or (force_prefix and t <= decision)) and fp != tape['hashes'][t]:
                    raise RestoreMismatch(f'physics at control {t}')
                if t < len(tape['hashes']) and first_state_diff is None and fp != tape['hashes'][t]:
                    first_state_diff = t
            dist = geometry(raw, robot, centers)
            if not force_prefix and decision is None and first_grasp is None and t % 5 == 0 and np.linalg.norm(dist) <= GATE_M:
                decision = t
            if t == decision:
                decision_obs = public_hash(obs)
                decision_state = inputs(obs)['observation/state'].astype(float)
                if force_prefix and decision_obs != tape['decision_obs']:
                    raise RestoreMismatch(f'policy inputs at decision {t}')
            if replay_all or (force_prefix and t < decision):
                a = tape['actions'][t].copy()
            else:
                if t % 5 == 0:
                    chunk = np.asarray(policy.infer(inputs(obs), noise=noise[t // 5])['actions'])
                    assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                a = chunk[t % 5].copy()
            if decision is not None and t >= decision:
                raw_a1_post.append(float(a[1]))
            active = (decision is not None and first_grasp is None and decision <= t < decision + window)
            if active:
                old = float(a[1])
                if residual:
                    target_a1 = old + residual / gain
                    clipped_a1 = float(np.clip(target_a1, controller.input_min[1], controller.input_max[1]))
                    if abs(clipped_a1 - target_a1) > 1e-7:
                        clipped_steps += 1
                    a[1] = clipped_a1
                realized.append(float(gain * (
                    np.clip(a[1], controller.input_min[1], controller.input_max[1])
                    - np.clip(old, controller.input_min[1], controller.input_max[1])
                )))
            if tape is not None and t < len(tape['actions']) and first_command_diff is None:
                delta = np.abs(a - tape['actions'][t])
                if np.any(delta):
                    first_command_diff, first_command_max_abs = t, float(delta.max())
            actions.append(a)
            obs, reward, done, truncated, info = env.step(convert_action(a))
            success = bool(info['success'])
            if first_grasp is None and raw._check_grasp(robot.gripper['right'], names):
                first_grasp = t + 1
            if decision is not None and t - decision in (0, 4, 9, 19, 39):
                signed = float(geometry(raw, robot, centers) @ controller.origin_ori[:, 1])
                misses.append([t + 1, signed])
            if first_grasp == t + 1 and decision is not None:
                signed = float(geometry(raw, robot, centers) @ controller.origin_ori[:, 1])
                misses.append([t + 1, signed])
            if decision is not None and post20_state is None and (t + 1 == decision + 20 or success or done or truncated):
                post20_state = inputs(obs)['observation/state'].astype(float)
                post20_control = t + 1
            if t % 5 == 4 or success or done or truncated:
                observations.append([t + 1, float(reward), *np.asarray(obs['state.end_effector_position_relative']).tolist()])
            if success or done or truncated:
                break
        assert goal_calls == len(realized)
        terminal = physics(raw)
        if replay_all and (success != tape['success'] or len(actions) != len(tape['actions']) or terminal != tape['terminal']):
            raise RestoreMismatch('repeat_tape terminal state, length or success differed')
        data = dict(
            actions=np.array(actions), hashes=hashes, decision=decision, decision_obs=decision_obs,
            terminal=terminal, success=success, start_key=start_key,
        )
        np.savez_compressed(
            out / f'{scene}-{arm}.npz', actions=data['actions'], physics=hashes,
            public_eef_reward=np.asarray(observations, dtype=np.float64),
        )
        public_delta_eef_y = (
            float(post20_state[1] - decision_state[1])
            if (decision_state is not None and post20_state is not None) else None
        )
        commanded_mean_y = (
            float(np.mean(raw_a1_post[:max(1, post20_control - decision)]))
            if (decision is not None and post20_control is not None and raw_a1_post) else None
        )
        row = dict(
            scene=scene, arm=arm, start_signature=start_key,
            initial_sha256=initial_state, initial_observation_sha256=initial_obs,
            bias_m=float(bias), requested_residual_m=float(residual), window=int(window),
            active_steps=len(realized), clipped_steps=int(clipped_steps),
            applied_residual_mean_m=float(np.mean(realized)) if realized else 0.0,
            controls=t + 1, success=success, first_grasp=first_grasp,
            decision=decision, decision_observation_sha256=decision_obs,
            decision_state=decision_state.tolist() if decision_state is not None else None,
            post20_control=post20_control,
            post20_state=post20_state.tolist() if post20_state is not None else None,
            public_delta_eef_y=public_delta_eef_y,
            commanded_mean_y=commanded_mean_y,
            signed_miss_m=misses,
            matched_pre_action_states=0 if tape is None else (t + 1 if replay_all else (decision + 1 if force_prefix else 0)),
            first_state_diff=first_state_diff,
            first_command_diff=first_command_diff,
            first_command_max_abs=first_command_max_abs,
            terminal_sha256=terminal,
        )
        return row, data
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('ckpt', type=Path)
    p.add_argument('--seeds', required=True)
    p.add_argument('--target-admitted', type=int, required=True)
    p.add_argument('--bias-m', type=float, default=0.04)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--screen-all', action='store_true', help='Finish this fixed shard; admission is decided by the ordered merge')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    def log(kind, **kwargs):
        row = dict(kind=kind, unix=time.time(), **kwargs)
        with (args.out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    assert robocasa.__version__ == '1.0.1'
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(args.ckpt / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, args.ckpt, sample_kwargs={'num_steps': 10}, norm_stats=stats)

    seeds = parse_seed_spec(args.seeds)
    admitted, screened, records = [], [], []
    started = time.time()
    for scene in seeds:
        log('screening', scene=scene)
        nom_row, tape = rollout(policy, args.out, scene, 'nominal', window=args.window, log=log)
        log('episode', **nom_row)
        records.append(nom_row)
        eligible = bool(nom_row['success'] and nom_row['decision'] is not None)
        screened.append(dict(scene=scene, eligible=eligible, success=nom_row['success'],
                             decision=nom_row['decision'], first_grasp=nom_row['first_grasp'],
                             controls=nom_row['controls']))
        if not eligible:
            continue
        rep_row, _ = rollout(policy, args.out, scene, 'repeat_tape', target_start=tape['start_key'],
                             tape=tape, force_prefix=True, replay_all=True, window=args.window, log=log)
        log('episode', **rep_row)
        records.append(rep_row)
        cl_row, _ = rollout(policy, args.out, scene, 'repeat_closed_loop', target_start=tape['start_key'],
                            tape=tape, force_prefix=False, replay_all=False, window=args.window, log=log)
        log('episode', **cl_row)
        records.append(cl_row)
        scene_rows = [nom_row, rep_row, cl_row]
        for sign in (-1, 1):
            for u in (-1, 0, 1):
                arm = f'b{sign:+d}-u{u:+d}'
                row, _ = rollout(
                    policy, args.out, scene, arm, target_start=tape['start_key'],
                    tape=tape, force_prefix=True, replay_all=False,
                    bias=sign * args.bias_m, residual=u * args.bias_m, window=args.window, log=log,
                )
                log('episode', **row)
                records.append(row)
                scene_rows.append(row)
        write_json(args.out / f'scene-{scene}.json', scene_rows)
        admitted.append(scene)
        if not args.screen_all and len(admitted) >= args.target_admitted:
            break

    summary = dict(
        completed=args.screen_all or len(admitted) == args.target_admitted,
        screen_all=args.screen_all,
        bias_m=float(args.bias_m),
        window=int(args.window),
        horizon=HORIZON,
        gate_m=GATE_M,
        target_admitted=args.target_admitted,
        screened_count=len(screened),
        admitted_count=len(admitted),
        admitted_scenes=admitted,
        screened=screened,
        episodes=len(records),
        total_controls=sum(r['controls'] for r in records),
        elapsed_seconds=time.time() - started,
        records=records,
    )
    write_json(args.out / 'SUMMARY.json', summary)
    log('complete', admitted=len(admitted), screened=len(screened), episodes=len(records))
    if not args.screen_all and len(admitted) < args.target_admitted:
        raise RuntimeError(f'Only admitted {len(admitted)}/{args.target_admitted} scenes from {len(screened)} screened')


if __name__ == '__main__':
    main()
