"""Three development checks of the constructed-history idea on frozen native Human300 pi0.5.

    checks.py OUT CKPT persistence [TASK,TASK,...] [SCENES] [REPEATS]
    checks.py OUT CKPT rescue
    checks.py OUT CKPT stiff

Every episode starts from a seeded reset and runs the unchanged closed-loop policy
(predict 50, execute 5, ten denoising steps). Only the named intervention differs.
"""
import json
from pathlib import Path
import random
import sys
import time

import gymnasium as gym
import imageio.v2 as imageio
import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'robocasa-drawer-calibration-20261109-r2'))
from replay import (GATE_M, HORIZON, INPUT, config, convert_action, digest, geometry, handle,
                    inputs, normalize, physics, policy_config, stable_counter_regions, with_checkpoint_stats)

DRAWER_SCENES = range(982100, 982108)   # screened on November 10; 982101 and 982103 opened
DEFAULT_TASKS = 'OpenDrawer,CloseDrawer,CloseFridge,TurnOnMicrowave'


def reset(task, scene):
    random.seed(scene)
    np.random.seed(scene)
    env = gym.make(f'robocasa/{task}', split='pretrain', seed=scene)
    obs, _ = env.reset(seed=scene)
    return env, obs


class Drawer:
    """OpenDrawer-only measurements and interventions; the simulator alone sees them."""

    def __init__(self, raw, *, goal_offset=0., window=100, damping=1., effort=1.):
        self.raw, self.robot = raw, raw.robots[0]
        self.controller = self.robot.part_controllers['right']
        assert self.controller.input_ref_frame == 'base' and self.controller.input_type == 'delta'
        self.names, self.centers = handle(raw)
        self.goal_offset, self.window, self.effort = goal_offset, window, effort
        self.decision = self.first_grasp = None
        self.active = False
        self.misses = []
        model = raw.sim.model
        prefix = raw.drawer.naming_prefix
        self.dofs = [int(model.jnt_dofadr[i]) for i in range(model.njnt)
                     if model.jnt_type[i] == 2 and (model.joint_id2name(i) or '').startswith(prefix)]
        self.qpos = [int(model.jnt_qposadr[i]) for i in range(model.njnt)
                     if model.jnt_type[i] == 2 and (model.joint_id2name(i) or '').startswith(prefix)]
        assert self.dofs, ('no drawer slide joint', prefix)
        self.start = raw.sim.data.qpos[self.qpos].copy()
        self.travel = 0.
        if damping != 1.:
            for dof in self.dofs:
                model.dof_damping[dof] = max(float(model.dof_damping[dof]), 1.) * damping
        self.damping = [float(model.dof_damping[dof]) for dof in self.dofs]
        nominal = self.controller.compute_goal_pos

        def shifted(*args, **kwargs):
            goal = nominal(*args, **kwargs)
            if self.active and self.goal_offset:
                goal = goal.copy()
                goal[1] += self.goal_offset
            return goal
        self.controller.compute_goal_pos = shifted

    def lateral(self):
        return float(geometry(self.raw, self.robot, self.centers) @ self.controller.origin_ori[:, 1])

    def act(self, t, a):
        if self.decision is None and t % 5 == 0 and (
                np.linalg.norm(geometry(self.raw, self.robot, self.centers)) <= GATE_M):
            self.decision = t
        self.active = (self.decision is not None and self.first_grasp is None
                       and t < self.decision + self.window)
        if self.first_grasp is not None and self.effort != 1.:
            low, high = self.controller.input_min[:3], self.controller.input_max[:3]
            a[:3] = np.clip(a[:3] * self.effort, low, high)
        return a

    def after(self, t):
        if self.first_grasp is None and self.raw._check_grasp(self.robot.gripper['right'], self.names):
            self.first_grasp = t + 1
        if self.decision is not None and (t - self.decision) % 10 == 0 and len(self.misses) < 12:
            self.misses.append([t + 1, self.lateral()])
        self.travel = max(self.travel, float(np.abs(self.raw.sim.data.qpos[self.qpos] - self.start).max()))

    def report(self):
        return dict(decision=self.decision, first_grasp=self.first_grasp, signed_miss_m=self.misses,
                    drawer_travel_m=self.travel, drawer_damping=self.damping,
                    goal_offset_m=self.goal_offset, effort=self.effort)


def public_hash(obs):
    x = inputs(obs)
    return digest(x['observation/state'], x['prompt'], *(x[k] for k in INPUT))


def episode(policy, task, scene, noise_seed, *, drawer=None, horizon=HORIZON, video=None):
    env, obs = reset(task, scene)
    writer = None
    try:
        raw = env.unwrapped.env
        hook = Drawer(raw, **drawer) if drawer is not None else None
        initial_state = physics(raw)
        initial_obs = public_hash(obs)
        decision_obs = grasp_state = None
        if video is not None:
            writer = imageio.get_writer(video, fps=10, codec='libx264', macro_block_size=1)
        noise = np.random.default_rng(noise_seed).standard_normal((horizon // 5, 50, 32), dtype=np.float32)
        success, chunk = False, None
        for t in range(horizon):
            if t % 5 == 0:
                chunk = np.asarray(policy.infer(inputs(obs), noise=noise[t // 5])['actions'])
                assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
            a = chunk[t % 5].copy()
            if hook:
                a = hook.act(t, a)
                if hook.decision == t:
                    decision_obs = public_hash(obs)
            if writer is not None and t % 5 == 0:
                frame = inputs(obs)
                writer.append_data(np.concatenate([frame['observation/image'], frame['observation/wrist_image']], axis=1))
            obs, _, done, truncated, info = env.step(convert_action(a))
            success = bool(info['success'])
            if hook:
                hook.after(t)
                if hook.first_grasp == t + 1:
                    grasp_state = physics(raw)
            if success or done or truncated:
                break
        return dict(task=task, scene=scene, noise_seed=noise_seed, success=success, controls=t + 1,
                    initial_sha256=initial_state, initial_observation_sha256=initial_obs,
                    decision_observation_sha256=decision_obs, first_grasp_sha256=grasp_state,
                    terminal_sha256=physics(raw),
                    **(hook.report() if hook else {}))
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            env.close()


def persistence(policy, log, tasks, scenes, repeats):
    """Same scene and start, different motor noise: is the outcome a property of the scene?"""
    for task in tasks:
        for index in range(scenes):
            scene = 983000 + index
            start = None
            for repeat in range(repeats):
                row = episode(policy, task, scene, scene + 700000 + 1000 * repeat)
                current = (row['initial_sha256'], row['initial_observation_sha256'])
                if start is not None and current != start:
                    raise ValueError(f'Native reset mismatch for {task} {scene} repeat {repeat}')
                start = current
                log('episode', check='persistence', repeat=repeat, **row)


def rescue(policy, log, out):
    """On scenes the frozen policy fails, does a held lateral goal offset recover the grasp?"""
    archive = HERE.parent / 'robocasa-drawer-authority-20261110-r1' / 'events.jsonl'
    records = [json.loads(line) for line in archive.read_text().splitlines()]
    nominal = {r['scene']: r for r in records if r.get('arm') == 'nominal'}
    if set(nominal) != set(DRAWER_SCENES):
        raise ValueError('Incomplete reference drawer screen')
    for scene in DRAWER_SCENES:
        base = episode(policy, 'OpenDrawer', scene, scene + 700000, drawer={},
                       video=out / f'{scene}-nominal.mp4')
        log('episode', check='rescue', arm='nominal', **base)
        fields = ('success', 'controls', 'decision', 'first_grasp', 'decision_observation_sha256', 'terminal_sha256')
        for field in fields:
            if base[field] != nominal[scene][field]:
                raise ValueError(f'Prior nominal mismatch on scene {scene}, field {field}')
    for scene in DRAWER_SCENES:
        base = nominal[scene]
        if base['success'] or base['decision'] is None:
            continue
        for cm in (-4, -2, 2, 4):
            arm = f'offset{cm:+d}cm'
            row = episode(policy, 'OpenDrawer', scene, scene + 700000, drawer=dict(goal_offset=cm / 100),
                          video=out / f'{scene}-{arm}.mp4')
            if (row['decision'], row['decision_observation_sha256']) != (base['decision'], base['decision_observation_sha256']):
                raise ValueError(f'Offset decision mismatch on scene {scene}, arm {arm}')
            log('episode', check='rescue', arm=arm, **row)


def stiff(policy, log):
    """A latent the camera cannot show before contact: does it cause failure, persist, and yield to effort?"""
    for scene in (982101, 982103):
        start = None
        for damping in (1., 8., 64.):
            for repeat in range(2):
                nominal = None
                for effort in (1., 2.):
                    row = episode(policy, 'OpenDrawer', scene, scene + 700000 + 1000 * repeat,
                                  drawer=dict(damping=damping, effort=effort))
                    current = (row['initial_sha256'], row['initial_observation_sha256'])
                    if start is not None and start != current:
                        raise ValueError(f'Stiffness changed reset on {scene}')
                    start = current
                    grasp = (row['first_grasp'], row['first_grasp_sha256'])
                    if nominal is not None and nominal != grasp:
                        raise ValueError(f'Unmatched effort branch on {scene}, {damping}, {repeat}')
                    nominal = grasp
                    log('episode', check='stiff', repeat=repeat, damping_scale=damping, **row)


def main():
    out, ckpt, check, *rest = sys.argv[1:]
    out, ckpt = Path(out), Path(ckpt)

    def log(kind, **row):
        row = dict(kind=kind, unix=time.time(), **row)
        with (out / 'events.jsonl').open('a') as stream:
            print(json.dumps(row), file=stream, flush=True)
        print(json.dumps(row), flush=True)
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(ckpt / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={'num_steps': 10}, norm_stats=stats)
    if check == 'persistence':
        tasks = (rest[0] if rest else DEFAULT_TASKS).split(',')
        persistence(policy, log, tasks, int(rest[1]) if len(rest) > 1 else 8, int(rest[2]) if len(rest) > 2 else 4)
    elif check == 'rescue':
        rescue(policy, log, out)
    elif check == 'stiff':
        stiff(policy, log)
    else:
        raise SystemExit('Choose persistence, rescue or stiff')
    log('complete', check=check)


if __name__ == '__main__':
    main()
