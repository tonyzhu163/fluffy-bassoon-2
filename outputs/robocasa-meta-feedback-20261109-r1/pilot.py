"""Collect matched, frozen-policy trials for physical calibration and goal inference."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
from openpi_client import image_tools
import robocasa
from robocasa.utils import object_utils as ou
from robocasa.utils.env_utils import convert_action

from fluffy_bassoon.robocasa.pairing import stable_counter_regions
from reference_eval import with_checkpoint_stats, write

CAMERAS = ('video.robot0_agentview_left', 'video.robot0_eye_in_hand',
           'video.robot0_agentview_right')
STATES = ('state.end_effector_position_relative', 'state.end_effector_rotation_relative',
          'state.base_position', 'state.base_rotation', 'state.gripper_qpos')
GOALS = ('left', 'right', 'either')
BIAS = 0.025


def inputs(obs, prompt):
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[key]), 224, 224)) for key in CAMERAS]
    result = dict(zip(('observation/image', 'observation/wrist_image', 'observation/right_image'), images))
    result.update({'observation/state': np.concatenate([obs[k] for k in STATES]), 'prompt': prompt})
    return result


def digest(*arrays):
    code = hashlib.sha256()
    for a in arrays:
        code.update(np.ascontiguousarray(a).tobytes())
    return code.hexdigest()


class Drawer:
    def __init__(self, env, offset):
        self.env = env
        robot = env.robots[0]
        self.robot = robot
        ctrl = robot.composite_controller.part_controllers['right']
        assert ctrl.input_ref_frame == 'base' and ctrl.interpolator_pos is None
        assert ctrl._goal_update_mode == 'achieved'
        self.controller = ctrl
        self.geoms = [g for g in env.sim.model.geom_names
                      if g and g.startswith(env.drawer.name + '_door_handle_') and
                      env.sim.model.geom_contype[env.sim.model.geom_name2id(g)]]
        assert self.geoms, env.drawer.name
        self.ids = [env.sim.model.geom_name2id(g) for g in self.geoms]
        self.site = robot.eef_site_id['right']
        self.started = False
        self.closed = False
        self.applied = 0
        self.grasped = False
        self.miss = None
        self.active = False
        self.last_open = False
        self.offset = offset
        original = ctrl.compute_goal_pos

        def goal(delta, goal_update_mode=None):
            nominal = original(delta, goal_update_mode)
            if self.active and offset:
                self.applied += 1
                return nominal + np.array([0., offset, 0.])
            return nominal

        ctrl.compute_goal_pos = goal

    def displacement(self):
        data = self.env.sim.data
        center = np.mean(data.geom_xpos[self.ids], axis=0)
        return self.controller.origin_ori.T @ (data.site_xpos[self.site] - center)

    def before(self, action):
        close = bool(action[6] >= .5)
        near = np.linalg.norm(self.displacement()) <= .16
        self.started |= near and not close
        edge = self.started and self.last_open and close and not self.closed
        self.active = self.started and not self.closed and near
        self.last_open = not close
        self.closed |= edge
        return edge

    def after(self, edge):
        if edge:
            self.miss = float(self.displacement()[1])
        self.grasped |= bool(self.env._check_grasp(self.robot.gripper['right'], self.geoms))


def receptacles(env):
    # +Y is left in the initial robot base frame. Freeze identities at reset.
    base = env.robots[0].composite_controller.part_controllers['right'].origin_ori
    data = env.sim.data
    lateral = [float((base.T @ data.body_xpos[env.obj_body_id[b]])[1]) for b in ('bowl1', 'bowl2')]
    if abs(lateral[0] - lateral[1]) < .03:
        raise ValueError(f'Ambiguous left/right bowls: {lateral}')
    return (('bowl1', 'bowl2') if lateral[0] > lateral[1] else ('bowl2', 'bowl1'))


def placed(env, bowls):
    return [any(ou.check_obj_in_receptacle(env, f'fruit{i}', b)
                and ou.gripper_obj_far(env, f'fruit{i}') for i in range(1, 5)) for b in bowls]


def episode(policy, out, task, seed, arm, sign=0, correction=0):
    ident = f'{task}-{seed}-{arm}'
    np.random.seed(seed)
    random.seed(seed)
    wrapped = gym.make('robocasa/' + task, split='pretrain', seed=seed)
    try:
        obs, _ = wrapped.reset(seed=seed)
        env = wrapped.unwrapped.env
        initial = inputs(obs, obs['annotation.human.task_description'])
        physical = digest(env.sim.get_state().flatten(), env.sim.data.cam_xpos, env.sim.data.cam_xmat,
                          env.sim.model.body_pos, env.sim.model.geom_pos)
        view = digest(*(initial[k] for k in initial if k != 'prompt'))
        drawer = Drawer(env, (sign + correction) * BIAS) if task == 'OpenDrawer' else None
        bowls = None if drawer else receptacles(env)
        prompt = initial['prompt'] if drawer else (
            'Put one fruit from the plate into a bowl.' if arm == 'either'
            else f'Put one fruit from the plate into the {arm} bowl.')
        horizon = 600 if drawer else 350
        noise = np.random.default_rng(seed + 501003)
        actions = collections.deque()
        trace = []
        images = []
        placements = [False, False]
        full_success = False
        edge_control = None
        for t in range(horizon):
            if not actions:
                model_input = inputs(obs, prompt)
                chunk = np.asarray(policy.infer(model_input, noise=noise.standard_normal(
                    (50, 32), dtype=np.float32))['actions'])
                assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                actions.extend(chunk[:5])
                images.append([np.array(model_input[k][::4, ::4]) for k in
                               ('observation/image', 'observation/wrist_image')])
            action = actions.popleft()
            edge = drawer.before(action) if drawer else False
            pre = np.concatenate([obs[k] for k in STATES])
            obs, reward, done, truncated, info = wrapped.step(convert_action(action))
            post = np.concatenate([obs[k] for k in STATES])
            trace.append(np.concatenate([action, pre, post]))
            full_success |= bool(info['success'])
            if drawer:
                drawer.after(edge)
                if edge:
                    edge_control = t
                if info['success']:
                    break
            else:
                placements = placed(env, bowls)
                # Identical stopping rule under either hidden goal.
                if any(placements):
                    break
            if done or truncated:
                break
        np.savez_compressed(out / (ident + '.npz'), action_state=np.asarray(trace), cameras=images)
        row = dict(task=task, seed=seed, arm=arm, sign=sign, correction=correction, controls=t + 1,
                   physical=physical, initial_public=view, prompt=prompt, native_success=full_success,
                   placed=placements, grasped=drawer.grasped if drawer else None,
                   closure=edge_control, miss_m=drawer.miss if drawer else None,
                   offset_target_updates=drawer.applied if drawer else None)
        write(out / (ident + '.json'), row)
        print(json.dumps(row), flush=True)
        return row
    finally:
        wrapped.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--pairs', type=int, required=True)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--task', choices=['both', 'drawer', 'goal'], default='both')
    args = p.parse_args()
    assert 0 <= args.first < 8 and args.first < args.first + args.pairs <= 8
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    import jax
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(args.checkpoint / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=stats,
                                                 sample_kwargs={'num_steps': 10})
    rows = []
    pairs = {}
    def collect(task, seed, *args):
        row = episode(policy, out, task, seed, *args)
        key = (task, seed)
        anchor = pairs.setdefault(key, (row['physical'], row['initial_public']))
        if anchor != (row['physical'], row['initial_public']):
            raise ValueError(f'Unpaired initial state or policy input on {key}, {args}')
        rows.append(row)
        write(out / 'progress.json', dict(episodes=len(rows), last=row, unix=time.time()))
    out = args.out
    for i in range(args.first, args.first + args.pairs):
        # Alternate domains so the first pair exercises all proposed mechanisms.
        if args.task in ('drawer', 'both'):
            for s in (-1, 1):
                collect('OpenDrawer', 803310 + 2*i, f'support{s}', s, 0)
                for a in (0, -s, s):
                    collect('OpenDrawer', 803311 + 2*i, f'query{s}_{a}', s, a)
        if args.task in ('goal', 'both'):
            for seed in (805610 + 2*i, 805611 + 2*i):
                for a in GOALS:
                    collect('PortionFruitBowl', seed, a)
    write(out / 'complete.json', dict(episodes=len(rows), controls=sum(r['controls'] for r in rows)))


if __name__ == '__main__':
    main()
