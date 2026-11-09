"""Development-only control: give the same verified VLA grasp two physical continuations."""
import argparse
import json
from pathlib import Path

import numpy as np
import robosuite
from layout import FRUIT, reset_scene
from pilot import STATES, convert_action, digest, ou, stable_counter_regions


def fork(path, side):
    row = json.loads(path.read_text())
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as f:
        tape = f['action_post']
    w, _, geo = reset_scene(row['seed'], cpu=True)
    try:
        e = w.unwrapped.env
        h = digest(e.sim.get_state().flatten(), e.sim.data.cam_xpos, e.sim.data.cam_xmat,
                   e.sim.model.body_pos, e.sim.model.geom_pos)
        assert h == row['physical'] and all(geo['checks'].values())
        robot = e.robots[0]
        gripper = robot.gripper['right']
        c = robot.part_controllers['right']
        assert c.input_ref_frame == 'base' and c.input_type == 'delta'
        def hand():
            return e.sim.data.site_xpos[robot.eef_site_id['right']].copy()
        def pos(k):
            return e.sim.data.body_xpos[e.obj_body_id[k]].copy()
        fruit = None
        for n, a in enumerate(tape, 1):
            ob, *_ = w.step(convert_action(a[:12]))
            np.testing.assert_array_equal(np.concatenate([ob[k] for k in STATES]), a[12:])
            held = [f for f in FRUIT if e._check_grasp(gripper, e.objects[f])]
            if held:
                fruit = held[0]; break
        assert (n if fruit else None) == row['first_grasp']
        audit = dict(seed=row['seed'], original_arm=row['arm'], target=side, prefix_controls=n,
                     fruit=fruit, placed=[False,False], first_event=None)
        if not fruit: return audit
        bowls = geo['bowls']
        start = hand()
        b = pos(bowls[('left','right').index(side)])
        grasp_offset = start - pos(fruit)
        high = max(start[2]+.16, b[2]+.18+grasp_offset[2])
        over = np.array([*(b[:2]+grasp_offset[:2]), high])
        low = over.copy(); low[2] = b[2]+.075+grasp_offset[2]
        away = over.copy(); away[:2] -= .12*c.origin_ori[:2,0]; away[2] += .12
        lift = start.copy(); lift[2] = high
        phases = [('secure', start, 1., 8), ('raise', lift, 1., 45),
                  ('carry', over, 1., 65), ('lower', low, 1., 40),
                  ('open', low, 0., 12), ('retreat', away, 0., 65)]
        stages = []
        done = trunc = False
        for phase, target, close, length in phases:
            for step in range(1,min(length,500-n)+1):
                a = np.zeros(12)
                delta = c.origin_ori.T @ (target-hand())
                a[:3] = np.clip(delta/.05,-.55,.55)
                a[6] = close
                _,_,done,trunc,_ = w.step(convert_action(a)); n += 1
                placed = [bool(any(ou.check_obj_in_receptacle(e, f, bowl) and ou.gripper_obj_far(e, f)
                                   for f in FRUIT)) for bowl in bowls]
                if any(placed):
                    audit.update(placed=placed, first_event=n)
                if any(placed) or done or trunc: break
                if phase in ('raise','carry','lower','retreat') and np.linalg.norm(target-hand()) < .015: break
            stages.append(dict(phase=phase,n=n,hand_error_m=float(np.linalg.norm(target-hand())),
                               fruit_z_m=float(pos(fruit)[2]),grasp=bool(e._check_grasp(gripper,e.objects[fruit]))))
            if any(audit['placed']) or done or trunc or n >= 500: break
        return dict(audit, stages=stages)
    finally:
        w.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('file',type=Path); p.add_argument('side', choices=['left','right'])
    args=p.parse_args()
    original=robosuite.make
    def headless(*a,**kw):
        kw.update(has_offscreen_renderer=False,use_camera_obs=False)
        return original(*a,**kw)
    robosuite.make=headless
    stable_counter_regions()
    print('FORK '+json.dumps(fork(args.file,args.side)),flush=True)
