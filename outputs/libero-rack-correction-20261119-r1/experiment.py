"""Measure native placement outcomes of reference-based corrections after a verified motor prefix."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-failure-diagnosis-20261119-r1'))
from contact import ControlEnv, LiberoEnvironmentAdapter, PRO, TaskManifest, benchmark, robosuite_parse_problem
from panel import TRANSFER

CUTOFF = json.loads((HERE.parent/'libero-failure-diagnosis-20261119-r1/frozen.json').read_text())['stump']['model']['cut']
MODES = ('suffix','track')


def load(p):
    return json.loads(p.read_text())


def tape(pose, timing, paced):
    root = TRANSFER/'bottle'/f's{pose%6}-{timing}'
    if pose>=6:
        assert paced
        return root/'query-g1'
    return next(p for p in root.glob('support-g1-*') if load(p/'result.json')['paced']==paced)


def lift(actions, states):
    floor = None
    for i,(action,state) in enumerate(zip(actions,states)):
        if action[-1] <= 0:
            floor = None
            continue
        floor = state[2] if floor is None else min(floor,state[2])
        if state[2]-floor > CUTOFF:
            return i+1
    return len(actions)


@lru_cache(maxsize=1)
def references():
    refs=[]
    for pose in range(3):
        p=tape(pose,'BASE',False)
        with np.load(p/'public.npz') as d:
            a,s=d['issued'],d['state_after']
        hit=load(p/'result.json')['first'][1]
        k=lift(a,s)
        assert k<hit
        refs.append((pose,k,hit,a,s))
    return refs


@lru_cache(maxsize=1)
def reset_bank():
    return benchmark.get_benchmark_dict()['libero_goal']().get_task_init_states(2)


@lru_cache(maxsize=2)
def layout(timing):
    manifest=TaskManifest.load(str(PRO/'assets/task_manifest.json'))
    paths=[manifest.resolve_bddl('libero_goal',g,timing) for g in (2,9)]
    return str(paths[0]), [robosuite_parse_problem(str(p))['goal_state'] for p in paths]


def next_action(mode, ref, age, state):
    _,k,hit,actions,states=ref
    if mode=='suffix':
        return actions[min(k+age,len(actions)-1)].copy()
    # Aim four reference controls ahead, then hold the final reference pose for release.
    idx=min(k+age+3,hit-1)
    desired=states[idx]
    dp=desired[:3]-state[:3]
    dr=(Rotation.from_quat(desired[3:7])*Rotation.from_quat(state[3:7]).inv()).as_rotvec()
    # OSC scales the bounded action to 5 cm and 0.5 rad. Limit gain for stale observations.
    action=np.r_[np.clip(dp/.05,-1,1)*.45, np.clip(dr/.5,-1,1)*.45, 1.]
    if age>=hit-k+7:
        action[-1]=-1
    return action.astype(np.float32)


def trial(args):
    pose,timing,paced,mode=args
    p=tape(pose,timing,paced)
    record=load(p/'result.json')
    with np.load(p/'public.npz') as z:
        actions,states=z['issued'],z['state_after']
    original=np.load(p/'evaluator-predicates.npy')
    assert record['scene']==pose and record['goal']==1
    assert (np.flatnonzero(original[:,1])[0]+1 if original[:,1].any() else None)==record['first'][1]
    k=lift(actions,states)
    path,goals=layout(timing)
    env=LiberoEnvironmentAdapter()
    env._init_states=reset_bank()
    env._max_steps=300
    env._env=ControlEnv(path,use_camera_obs=False,has_offscreen_renderer=False,camera_names=[])
    try:
        reset=env.reset(pose,18180100+pose)
        assert reset['environment_reproducibility']['initial_state_fingerprint']==record['initial_physics']
        world=env._env.env
        ref=None
        hits=[]
        for i in range(300):
            if i==k:
                ref=min(references(),key=lambda r:np.linalg.norm(state[:3]-r[4][r[1]-1,:3]))
            action=actions[i] if i<k else next_action(mode,ref,i-k,state)
            raw=env.step(action)['observation']
            state=np.concatenate([raw[key] for key in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')])
            native=[all(world._eval_predicate(term) for term in g) for g in goals]
            if i<k:
                np.testing.assert_array_equal(state,states[i])
                np.testing.assert_array_equal(native,original[i])
            hits.append(bool(native[1]))
        return dict(pose=pose,timing=timing,paced=paced,mode=mode,lift=k if k<300 else None,
                    reference=ref[0] if ref else None, hit=any(hits),first=hits.index(True)+1 if any(hits) else None,
                    final=hits[-1],open_final=bool(action[-1]<0),baseline=bool(original[:,1].any()),
                    baseline_final=bool(original[-1,1]))
    finally:
        env.close()


def run(pool, jobs, path):
    rows=[]
    with path.open('x',buffering=1) as f:
        for row in pool.map(trial,jobs):
            print(json.dumps(row),file=f)
            rows.append(row)
    return rows


def main(out):
    out.mkdir()
    with ProcessPoolExecutor(max_workers=3) as pool:
        fit=run(pool,product(range(3),['S20'],(False,True),MODES),out/'fit.jsonl')
        candidates=[]
        for paced,mode in product((False,True),MODES):
            rows=[r for r in fit if (r['paced'],r['mode'])==(paced,mode)]
            candidates.append(dict(paced=paced,mode=mode,hits=sum(r['hit'] for r in rows),final=sum(r['final'] for r in rows)))
        best=max(candidates,key=lambda c:(c['hits'],c['final'],c['mode']=='suffix',not c['paced']))
        (out/'selection.json').write_text(json.dumps(dict(best=best,candidates=candidates),indent=2))
        jobs=[(i,v,(True if i>=6 else best['paced']),best['mode']) for i,v in product(range(3,12),('BASE','S20'))]
        rows=run(pool,jobs,out/'evaluation.jsonl')
        for row in rows:
            if row['pose']<6:
                other=tape(row['pose'],row['timing'],not row['paced'])
                other_rec=load(other/'result.json')
                base=load(tape(row['pose'],row['timing'],row['paced'])/'result.json')
                assert base['initial_physics']==other_rec['initial_physics'] and base['initial_public']==other_rec['initial_public']
                row['alternate_baseline']=bool(np.load(other/'evaluator-predicates.npy')[:,1].any())
        (out/'result.json').write_text(json.dumps(dict(selection=best,fit=fit,evaluation=rows),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
