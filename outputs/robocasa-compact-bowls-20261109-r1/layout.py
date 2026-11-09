"""Place the original plate and bowls in one robot-centered row."""
import json
import random
import gymnasium as gym
import numpy as np
from pilot import convert_action, ou, receptacles, stable_counter_regions

FRUIT = tuple(f'fruit{i}' for i in range(1, 5))
DISH = ('plate', 'bowl1', 'bowl2')


def reset_scene(seed, cpu=False):
    np.random.seed(seed)
    random.seed(seed)
    w = gym.make('robocasa/PortionFruitBowl', split='pretrain', seed=seed, enable_render=not cpu)
    try:
        w.reset(seed=seed)
        e = w.unwrapped.env
        c = e.robots[0].part_controllers['right']
        origin, rot = c.origin_pos.copy(), c.origin_ori.copy()
        def pos(k):
            return e.sim.data.body_xpos[e.obj_body_id[k]].copy()
        def relative(k):
            return rot.T @ (pos(k) - origin)
        before = {k: relative(k).tolist() for k in DISH}
        plate = relative('plate')
        left, right = ('bowl1', 'bowl2') if seed % 2 == 0 else ('bowl2', 'bowl1')
        offsets = {k: rot @ [0., -plate[1], 0.] for k in ('plate', *FRUIT)}
        for bowl, side in [(left, 1), (right, -1)]:
            b = relative(bowl)
            offsets[bowl] = rot @ [plate[0] - b[0], side*.30 - b[1], 0.]
        for k, delta in offsets.items():
            j = e.objects[k].joints[0]
            q = e.sim.data.get_joint_qpos(j).copy()
            q[:2] += delta[:2]
            e.sim.data.set_joint_qpos(j, q)
        e.sim.forward()
        targets = {k: pos(k) for k in DISH}
        for _ in range(12):
            ob, _, done, trunc, _ = w.step(convert_action(np.zeros(12)))
            assert not done and not trunc
        bowls = receptacles(e)
        after = {k: relative(k).tolist() for k in DISH}
        drift = max(float(np.linalg.norm(pos(k) - targets[k])) for k in DISH)
        checks = {
            'sides': tuple(bowls) == (left, right),
            'counter_contact': all(e.check_contact(e.objects[k], e.dining_counter) for k in DISH),
            'separate_dishes': not any(e.check_contact(e.objects[k], e.objects[l])
                                       for i,k in enumerate(DISH) for l in DISH[i+1:]),
            'fruit_on_plate': all(ou.check_obj_in_receptacle(e, f, 'plate') for f in FRUIT),
            'empty_bowls': not any(ou.check_obj_in_receptacle(e, f, b) for f in FRUIT for b in bowls),
            'stable': drift < .02,
            'compact': all(np.linalg.norm(relative(k)[:2]) <= .72 for k in DISH)
        }
        report = dict(seed=seed, bowls=bowls, before=before, after=after,
                      max_dish_drift_m=drift, checks=checks)
        return w, ob, report
    except BaseException:
        w.close()
        raise


if __name__ == '__main__':
    import sys
    import robosuite
    original = robosuite.make
    def headless(*a, **kw):
        kw.update(has_offscreen_renderer=False, use_camera_obs=False)
        return original(*a, **kw)
    robosuite.make = headless
    stable_counter_regions()
    for seed in map(int, sys.argv[1:]):
        env, _, report = reset_scene(seed, cpu=True)
        print('GEOMETRY ' + json.dumps(report), flush=True)
        env.close()
