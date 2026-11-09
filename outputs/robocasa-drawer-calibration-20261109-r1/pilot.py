"""Frozen-motor manipulation check for one unobserved calibration sign."""
import collections
import functools
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import jax
import numpy as np
import robocasa
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config
from openpi_client import image_tools
from robocasa.utils.env_utils import convert_action
from fluffy_bassoon.robocasa.pairing import stable_counter_regions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa-sft-reproduction-20261109-r1'))
from reference_eval import with_checkpoint_stats, write

CAM = ('video.robot0_agentview_left', 'video.robot0_eye_in_hand', 'video.robot0_agentview_right')
STATE = tuple('state.' + k for k in ('end_effector_position_relative', 'end_effector_rotation_relative',
                                      'base_position', 'base_rotation', 'gripper_qpos'))
INPUT = ('observation/image', 'observation/wrist_image', 'observation/right_image')
MAGNITUDE = 0.04
HORIZON = 600
NPAIRS = 8


def inputs(obs):
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[k]), 224, 224)) for k in CAM]
    return {**dict(zip(INPUT, images)), 'observation/state': np.concatenate([obs[k] for k in STATE]),
            'prompt': obs['annotation.human.task_description']}


def digest(*values):
    h = hashlib.sha256()
    for x in values:
        h.update(x.encode() if isinstance(x, str) else np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def handle(raw):
    model = raw.sim.model
    geoms = [(i, model.geom_id2name(i)) for i in range(model.ngeom)]
    target = [(i, n) for i, n in geoms if n and n.startswith(raw.drawer.naming_prefix)
              and 'handle' in n and (model.geom_contype[i] or model.geom_conaffinity[i])]
    assert target, ('no target collision geoms', raw.drawer.name)
    # Use collision geometry, excluding mounting posts when a bar is available.
    center = [i for i, n in target if 'connector' not in n and 'support' not in n]
    return [n for i, n in target], center or [i for i, n in target]


def episode(policy, out, pair, role, sign, compensation, anchors, log):
    import imageio.v2 as imageio
    scene = 910210 + 2 * pair + int(role == 'query')
    noise_seed = 1700210 + 2 * pair + int(role == 'query')
    random.seed(scene)
    np.random.seed(scene)
    motor_rng = np.random.default_rng(noise_seed)
    env = gym.make('robocasa/OpenDrawer', split='pretrain', seed=scene)
    ordinal = len(list(out.glob('episode-*.json')))
    folder = out / f'trial-{ordinal:03d}'
    folder.mkdir()
    pub = folder / 'public'
    pub.mkdir()
    priv = folder / 'evaluator'
    priv.mkdir()
    ident = dict(ordinal=ordinal, pair=pair, role=role, sign=sign, compensation=compensation,
                 scene_seed=scene, noise_seed=noise_seed)
    try:
        obs, _ = env.reset(seed=scene)
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.part_controllers['right']
        assert controller.input_ref_frame in ('world', 'base')
        geoms, centers = handle(raw)
        first = inputs(obs)
        fingerprint = digest(raw.sim.get_state().flatten(), raw.sim.model.body_pos, raw.sim.model.geom_pos,
                             raw.sim.data.cam_xpos, *[first[k] for k in INPUT], first['observation/state'], first['prompt'])
        key = (pair, role)
        assert anchors.setdefault(key, fingerprint) == fingerprint, ('unequal reset', key, fingerprint, anchors[key])
        other = anchors.get((pair, 'support' if role == 'query' else 'query'))
        assert other != fingerprint
        local_axis = np.array([0., 1., 0.])
        shift = MAGNITUDE * sign * (1 + compensation)
        active = False
        event = {}
        original = controller.compute_goal_pos

        @functools.wraps(original)
        def physical_target(*args, **kwargs):
            nominal = original(*args, **kwargs)
            if not active:
                return nominal
            assert not event, 'multiple OSC targets during the first closure'
            world = np.asarray(controller.origin_ori) @ local_axis
            axis = local_axis if controller.input_ref_frame == 'base' else world
            delivered = np.asarray(nominal) + shift * axis
            event.update(world_axis=world.tolist(), frame=controller.input_ref_frame,
                         nominal=nominal.tolist(), delivered=delivered.tolist(), net_offset_m=shift)
            return delivered

        controller.compute_goal_pos = physical_target
        plan = collections.deque()
        ring = collections.deque(maxlen=4)
        seen_open = acquired = success = False
        fired = None
        post = {}
        commands, states, rewards = [], [], []
        states.append(first['observation/state'])
        obs0 = first
        def images_at(t, element):
            for k, tag in zip(INPUT, ('left', 'wrist', 'right')):
                imageio.imwrite(pub / f'{t:04d}-{tag}.png', element[k])
        def miss(axis):
            eef = np.asarray(raw.sim.data.site_xpos[robot.eef_site_id['right']])
            center = np.asarray(raw.sim.data.geom_xpos)[centers].mean(axis=0)
            return float((eef - center) @ axis)
        for t in range(HORIZON):
            if not plan:
                x = obs0 if t == 0 else inputs(obs)
                noise = motor_rng.standard_normal((50, 32), dtype=np.float32)
                chunk = np.asarray(policy.infer(x, noise=noise)['actions'])
                assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                plan.extend(chunk[:5])
                if t == 0:
                    h = digest(chunk, noise)
                    assert anchors.setdefault((pair, role, 'draw'), h) == h, 'unmatched first motor draw'
            if t % 50 == 0:
                log('control', **ident, step=t)
            a = plan.popleft()
            close = bool(a[6] >= .5)
            active = fired is None and seen_open and close
            if not close:
                seen_open = True
            if role == 'support' and (fired is None or t <= fired + 8):
                ring.append((t, obs0 if t == 0 else inputs(obs)))
            if active:
                fired = t
                if role == 'support':
                    for ct, item in ring:
                        images_at(ct, item)
                axis = np.asarray(controller.origin_ori) @ local_axis
                post['before'] = miss(axis)
            obs, rew, done, trunc, info = env.step(convert_action(a))
            if active:
                assert event, 'OSC goal hook was not called'
                axis = np.asarray(event['world_axis'])
            if fired is not None and t - fired in (0, 1, 4, 8):
                post[str(t - fired + 1)] = miss(axis)
            if role == 'support' and fired is not None and 0 <= t - fired < 8:
                images_at(t + 1, inputs(obs))
            active = False
            commands.append(a.copy())
            states.append(np.concatenate([obs[k] for k in STATE]))
            rewards.append(float(rew))
            acquired |= bool(raw._check_grasp(robot.gripper['right'], geoms))
            success = bool(info['success'])
            if success or done or trunc:
                break
        np.savez_compressed(pub/'trajectory.npz', requested_action=commands, measured_state=states,
                            native_reward=rewards, prompt=first['prompt'])
        record = dict(**ident, paired_initial_sha256=fingerprint, controls=t+1, closure_control=fired,
                      success=success, acquired_handle=acquired, signed_handle_miss_m=post,
                      physical_target=event, handle_collision_geoms=geoms)
        write(priv/'metrics.json', record)
        write(out/f'episode-{ordinal:03d}.json', record)
        log('episode_complete', **ident, controls=t+1, closure=fired, success=success, acquired=acquired)
        return record
    finally:
        env.close()


def main():
    assert os.uname().nodename.startswith('job-')
    out, ckpt = map(Path, sys.argv[1:])
    def log(phase, **k):
        row = dict(phase=phase, unix=time.time(), **k)
        with (out/'events.jsonl').open('a') as fp:
            fp.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    assert robocasa.__version__ == '1.0.1' and len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    log('loading_frozen_motor')
    stats = normalize.load(ckpt/'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={'num_steps':10}, norm_stats=stats)
    anchors, rows = {}, []
    for pair in range(NPAIRS):
        for role, comp in [('support', 0), ('query', 0), ('query', -1), ('query', 1)]:
            for sign in (-1, 1):
                rows.append(episode(policy, out, pair, role, sign, comp, anchors, log))
        if pair == 0:
            neutral = [r for r in rows if r['compensation'] == -1]
            assert all(neutral[0][k] == neutral[1][k] for k in
                       ('closure_control', 'controls', 'success', 'acquired_handle', 'signed_handle_miss_m')), 'neutral parity'
            if not any(r['closure_control'] is not None for r in rows):
                log('stopped_no_first_closure', episodes=len(rows))
                return
        log('pair_complete', pair=pair)
    write(out/'RESULT.json', dict(episodes=len(rows), controls=sum(r['controls'] for r in rows),
                                  pairs=NPAIRS, training_updates=0))
    log('complete', episodes=len(rows))


if __name__ == '__main__':
    main()
