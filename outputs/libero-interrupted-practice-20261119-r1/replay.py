"""Replay unchanged and interrupted bowl and bottle practices in the native CPU simulator."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from libero.libero import benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest

ROOT=Path('/mowenjie/fluffy-bassoon-2')
BOWL=ROOT/'outputs/libero-matched-zero-corrections-20261119-r1/matched01/episodes'
BOTTLE=ROOT/'outputs/libero-first-attempt-20261119-r1/transfer02/episodes/bottle'
GOALS={'bowl':(4,8),'bottle':(2,9)}


@lru_cache(maxsize=2)
def setup(family):
    pro=ROOT/'sources/libero-pro2-e5500c69'
    manifest=TaskManifest.load(str(pro/'assets/task_manifest.json'))
    ids=GOALS[family]
    path=manifest.resolve_bddl('libero_goal',8,'S02') if family=='bowl' else manifest.resolve_bddl('libero_goal',2,'BASE')
    predicates=[robosuite_parse_problem(str(manifest.resolve_bddl('libero_goal',g,'BASE')))['goal_state'] for g in ids]
    bank=benchmark.get_benchmark_dict()['libero_goal']().get_task_init_states(ids[0])
    return str(path),predicates,bank


def read_tape(family,p,g,j):
    if family=='bowl':
        split='fitting' if p<40 else 'evaluation'
        block=BOWL/split/f'p{p}'
        with np.load(block/f'g{GOALS[family][g]}-practice{j}'/'tape.npz') as z:
            return z['actions'],z['states'],z['native'],json.loads((block/'initial.json').read_text())['physics']
    block=BOTTLE/f's{p-6}-BASE'/f'query-g{g}'
    rec=json.loads((block/'result.json').read_text())
    assert rec['scene']==p and rec['goal']==g
    with np.load(block/'public.npz') as z:
        return z['issued'],z['state_after'],np.load(block/'evaluator-predicates.npy'),rec['initial_physics']


def simulate(job):
    family,p,g,j,out=job
    issued,archive,native,identity=read_tape(family,p,g,j)
    path,goals,bank=setup(family)
    key=f'{family}-p{p}-g{g}-d{j}'
    objname='akita_black_bowl_1' if family=='bowl' else 'wine_bottle_1'
    records=[]
    for interrupt in (False,True):
        env=LiberoEnvironmentAdapter()
        env._env=ControlEnv(path,use_camera_obs=False,has_offscreen_renderer=False,camera_names=[])
        env._init_states=bank
        env._max_steps=300
        try:
            reset=env.reset(p,18180100+p)
            assert reset['environment_reproducibility']['initial_state_fingerprint']==identity
            world=env._env.env
            obj=world.objects_dict[objname]
            body=world.obj_body_id[objname]
            height0=world.sim.data.body_xpos[body,2]
            start=None
            states,hits,contact,heights=[],[],[],[]
            for i,original in enumerate(issued):
                a=original.copy()
                if interrupt and start is not None and start<=i<start+10:
                    a[-1]=-1.
                raw=env.step(a)['observation']
                state=np.concatenate([raw[k] for k in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')])
                h=[all(world._eval_predicate(t) for t in goal) for goal in goals]
                if not interrupt or start is None or i<start:
                    np.testing.assert_array_equal(state,archive[i],err_msg=f'{key} control {i+1}')
                    np.testing.assert_array_equal(h,native[i])
                grasp=bool(world._check_grasp(world.robots[0].gripper,obj))
                height=float(world.sim.data.body_xpos[body,2]-height0)
                states.append(state);hits.append(h);contact.append(grasp);heights.append(height)
                if start is None and grasp and height>=.03 and i+1<len(issued):
                    start=i+1
            np.savez_compressed(out/f'{key}-{int(interrupt)}.npz',states=states,issued=issued,hits=hits)
            records.append(dict(family=family,pose=p,goal=g,draw=j,interrupted=interrupt,start=start,
                                applied=min(10,300-start) if interrupt and start is not None else 0,
                                lost_contact=bool(not all(contact[start:start+10])) if start is not None else None,
                                dz10=heights[min(start+9,299)]-heights[start-1] if start is not None else None,
                                ever=np.any(hits,axis=0).tolist(),final=hits[-1]))
        finally:
            env.close()
    assert records[0]['start']==records[1]['start']
    return records


if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir()
    with (out/'branches.jsonl').open('x',buffering=1) as f, ProcessPoolExecutor(max_workers=6) as pool:
        for family,poses,draws in [('bowl',range(32,36),range(4)),('bowl',range(44,48),range(4)),('bottle',range(6,12),range(1))]:
            jobs=[(family,p,g,j,out) for p,g,j in product(poses,(0,1),draws)]
            for records in pool.map(simulate,jobs):
                for r in records:
                    print(json.dumps(r),file=f)
            print('completed',family,poses,flush=True)
