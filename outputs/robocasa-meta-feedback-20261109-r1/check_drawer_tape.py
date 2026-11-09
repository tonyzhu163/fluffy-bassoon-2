"""Independent CPU physics check for a finished native drawer command tape."""
import argparse
import json
from pathlib import Path
import random
import sys

import gymnasium as gym
import numpy as np
import robosuite

# Import the exact recorder's physics hash, handle geometry and input conversion.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa-drawer-calibration-20261109-r2'))
from replay import GATE_M, MAGNITUDE, convert_action, geometry, handle, physics, stable_counter_regions


def run(path, net):
    with np.load(path, allow_pickle=False) as f:
        scene, trigger = int(f['scene']), int(f['trigger'])
        actions, hashes, expected = f['actions'], f['physical_prefix'], bool(f['native_success'])
    random.seed(scene)
    np.random.seed(scene)
    env = gym.make('robocasa/OpenDrawer', split='pretrain', seed=scene, enable_render=False)
    try:
        env.reset(seed=scene)
        e = env.unwrapped.env
        robot = e.robots[0]
        c = robot.part_controllers['right']
        assert c.input_ref_frame == 'base'
        names, centers = handle(e)
        original = c.compute_goal_pos
        shifts = []
        def target(*args, **kwargs):
            nominal = original(*args, **kwargs)
            if t != trigger or not net:
                return nominal
            shifts.append(t)
            return nominal + np.array([0., net * MAGNITUDE, 0.])
        c.compute_goal_pos = target
        grasp = False
        matched = 0
        for t, a in enumerate(actions):
            if net == 0 or trigger < 0 or t <= trigger:
                if physics(e) != hashes[t]:
                    return dict(scene=scene, net=net, valid=False, first_physics_mismatch=t)
                matched += 1
            if t == trigger:
                assert np.linalg.norm(geometry(e, robot, centers)) <= GATE_M
            _, _, done, truncated, info = env.step(convert_action(a))
            grasp |= bool(e._check_grasp(robot.gripper['right'], names))
            if info['success'] or done or truncated:
                break
        assert shifts == ([trigger] if net and trigger >= 0 else [])
        if not net or trigger < 0:
            assert t + 1 == len(actions) and bool(info['success']) == expected
        return dict(scene=scene, net=net, valid=True, matched_states=matched, controls=t+1,
                    success=bool(info['success']), acquired=grasp)
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tape', type=Path)
    args = p.parse_args()
    make = robosuite.make
    def cpu_make(*args, **kwargs):
        kwargs.update(use_camera_obs=False, has_offscreen_renderer=False)
        return make(*args, **kwargs)
    robosuite.make = cpu_make
    stable_counter_regions()
    for net in (0, -1, 1, -2, 2):
        result = run(args.tape, net)
        print(json.dumps(result), flush=True)
        if not result['valid']:
            break
