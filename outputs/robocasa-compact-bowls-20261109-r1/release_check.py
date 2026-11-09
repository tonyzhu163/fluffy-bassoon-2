"""Retrospective diagnostic: at first bowl contact, give an open-and-retreat command."""
import argparse
import json
from pathlib import Path

import numpy as np
import robosuite
from layout import FRUIT, reset_scene
from pilot import STATES, convert_action, digest, ou, stable_counter_regions


def replay(path):
    d=json.loads(path.read_text())
    w,_,layout=reset_scene(d['seed'],cpu=True)
    try:
        e=w.unwrapped.env
        h=digest(e.sim.get_state().flatten(),e.sim.data.cam_xpos,e.sim.data.cam_xmat,
                 e.sim.model.body_pos,e.sim.model.geom_pos)
        assert h==d['physical']
        c=e.robots[0].part_controllers['right']
        site=e.robots[0].eef_site_id['right']
        with np.load(path.with_suffix('.npz')) as f: tape=f['action_post']
        start=min(x for x in d['first_inside'] if x is not None)
        for n in range(start):
            ob,*_=w.step(convert_action(tape[n,:12]))
            np.testing.assert_array_equal(np.concatenate([ob[k] for k in STATES]),tape[n,12:])
        bowls=layout['bowls']
        def inside(b): return [f for f in FRUIT if ou.check_obj_in_receptacle(e,f,b)]
        hit=[bool(inside(b)) for b in bowls]
        assert hit==[t==start for t in d['first_inside']]
        xyz=e.sim.data.site_xpos[site].copy()
        target=xyz+c.origin_ori @ [-.12,0.,.30]
        release=[False,False]
        t=start
        for t in range(start+1,min(500,start+100)+1):
            a=np.zeros(12)
            if t>start+12:
                delta=c.origin_ori.T @ (target-e.sim.data.site_xpos[site])
                a[:3]=np.clip(delta/.05,-.55,.55)
            ob,_,done,trunc,_=w.step(convert_action(a))
            release=[bool(any(ou.gripper_obj_far(e,f) for f in inside(b))) for b in bowls]
            if any(release) or done or trunc: break
        return dict(seed=d['seed'],arm=d['arm'],replayed_prefix=start,first_inside=hit,controls=t,
                    placed=release,original_placed=d['placed'])
    finally:
        w.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('file',type=Path,nargs='+');args=p.parse_args()
    make=robosuite.make
    def headless(*a,**kw):
        kw.update(has_offscreen_renderer=False,use_camera_obs=False)
        return make(*a,**kw)
    robosuite.make=headless;stable_counter_regions()
    for path in args.file: print('RELEASE '+json.dumps(replay(path)),flush=True)
