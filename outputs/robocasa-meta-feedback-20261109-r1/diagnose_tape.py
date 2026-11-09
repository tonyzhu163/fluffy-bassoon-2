"""Replay a saved bowl command tape on CPU and check state before attributing failure."""
import argparse
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import robosuite
from pilot import STATES, convert_action, digest, ou, receptacles, stable_counter_regions


def replay(path):
    row = json.loads(path.read_text())
    seed = row['seed']
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
        tape = saved['action_state']
    np.random.seed(seed)
    random.seed(seed)
    wrapped = gym.make('robocasa/PortionFruitBowl', split='pretrain', seed=seed, enable_render=False)
    try:
        obs, _ = wrapped.reset(seed=seed)
        e = wrapped.unwrapped.env
        initial = digest(e.sim.get_state().flatten(), e.sim.data.cam_xpos, e.sim.data.cam_xmat,
                         e.sim.model.body_pos, e.sim.model.geom_pos)
        assert initial == row['physical'], 'CPU initial physical state disagrees with GPU rollout'
        bowls = receptacles(e)
        first_inside = [None, None]
        first_released = [None, None]
        first_grasp = None
        fruit = [f'fruit{i}' for i in range(1, 5)]
        for t, a in enumerate(tape):
            obs, *_ = wrapped.step(convert_action(a[:12]))
            np.testing.assert_array_equal(np.concatenate([obs[k] for k in STATES]), a[28:44])
            if first_grasp is None and any(e._check_grasp(e.robots[0].gripper['right'], e.objects[f])
                                           for f in fruit):
                first_grasp = t + 1
            for j, b in enumerate(bowls):
                for f in fruit:
                    if ou.check_obj_in_receptacle(e, f, b):
                        if first_inside[j] is None:
                            first_inside[j] = t + 1
                        if first_released[j] is None and ou.gripper_obj_far(e, f):
                            first_released[j] = t + 1
        assert [x is not None for x in first_released] == row['placed']
        return dict(seed=seed, arm=row['arm'], exact_public_steps=len(tape), first_grasp=first_grasp,
                    first_inside=first_inside, first_inside_and_far=first_released)
    finally:
        wrapped.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('files', type=Path, nargs='+')
    args = p.parse_args()
    original_make = robosuite.make
    def cpu_make(*args, **kwargs):
        kwargs.update(has_offscreen_renderer=False, use_camera_obs=False)
        return original_make(*args, **kwargs)
    robosuite.make = cpu_make
    stable_counter_regions()
    for path in args.files:
        print(json.dumps(replay(path)), flush=True)
