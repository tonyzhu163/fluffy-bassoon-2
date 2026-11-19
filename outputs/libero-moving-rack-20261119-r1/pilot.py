"""Paired native BASE/D02 rack branches and a source-fitted reward decision tree."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product, permutations
import json
from pathlib import Path
import sys
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-rack-correction-20261119-r1'))
import experiment as old

REGIMES=('BASE','D02')
OPTIONS=('suffix','track','shift')
SHIFT=np.array([-.06,.045,0.])


def simulate(job, camera=None):
    pose,regime,option=job
    p=old.tape(pose,'BASE',pose>=6)
    with np.load(p/'public.npz') as z:
        commands,archive_states=z['issued'],z['state_after']
    archive=old.load(p/'result.json')
    manifest=old.TaskManifest.load(str(old.PRO/'assets/task_manifest.json'))
    path=manifest.resolve_bddl('libero_goal',9,regime)
    env=old.LiberoEnvironmentAdapter()
    env._init_states=old.reset_bank()
    env._env=old.ControlEnv(str(path),use_camera_obs=False,has_offscreen_renderer=False,camera_names=[])
    env._max_steps=300
    try:
        reset=env.reset(pose,18180100+pose)
        physics=reset['environment_reproducibility']['initial_state_fingerprint']
        world=env._env.env
        if camera is not None:
            camera.start(world.sim)
        body=world.obj_body_id['wine_rack_1']
        start=world.sim.data.body_xpos[body].copy()
        ref=None
        trigger=moved=diverged=first=None
        largest=0.
        floor=None
        prefix=hashlib.sha256()
        active_option=option
        for i in range(300):
            if ref is None:
                action=commands[i]
            else:
                age=i-trigger
                if camera is not None and age==5:
                    active_option=camera.choose(world.sim)
                if active_option=='shift':
                    _,k,hit,_,s=ref
                    # Moving the observed wrist the opposite way translates the target by SHIFT.
                    progress=min(1.,(age+4)/max(1,hit-k))
                    virtual=state.copy()
                    virtual[:3]-=progress*SHIFT
                    action=old.next_action('track',ref,age,virtual)
                else:
                    action=old.next_action(active_option,ref,age,state)
            step=env.step(action)
            raw=step['observation']
            state=np.concatenate([raw[x] for x in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')])
            native=bool(step['success'])
            if native and first is None:
                first=i+1
            disp=float(np.linalg.norm(world.sim.data.body_xpos[body][:2]-start[:2]))
            largest=max(largest,disp)
            if moved is None and disp>.01:
                moved=i+1
            if ref is None:
                prefix.update(state.tobytes())
                if diverged is None and not np.array_equal(state,archive_states[i]):
                    diverged=i+1
                if action[-1]<=0:
                    floor=None
                else:
                    floor=state[2] if floor is None else min(floor,state[2])
                    if state[2]-floor>old.CUTOFF:
                        trigger=i+1
                        ref=min(old.references(),key=lambda r:np.linalg.norm(state[:3]-r[4][r[1]-1,:3]))
        if regime=='BASE':
            assert physics==archive['initial_physics'] and diverged is None
        return dict(pose=pose,regime=regime,option=option,physics=physics,start=start.tolist(),prefix=prefix.hexdigest(),
                    lift=trigger,first_motion=moved,max_xy=largest,first_divergence=diverged,first=first,
                    hit=first is not None,final=native)
    finally:
        env.close()


def collect(pool,poses,out):
    data={}
    with out.open('x',buffering=1) as f:
        for r in pool.map(simulate,product(poses,REGIMES,OPTIONS)):
            print(json.dumps(r),file=f)
            data[r['pose'],r['regime'],r['option']]=r
    for p,g in product(poses,REGIMES):
        assert len({data[p,g,a]['prefix'] for a in OPTIONS})==1
    return data


def score(data,pairs,policy,swap=False):
    probe,miss,reward=policy
    rows=[]
    for (s,q),g in product(pairs,REGIMES):
        other=REGIMES[1-REGIMES.index(g)] if swap else g
        bit=data[s,other,probe]['hit']
        a=reward if bit else miss
        query=data[q,g,a]
        rows.append(dict(s=s,q=q,regime=g,bit=bit,choice=a,hit=query['hit'],final=query['final']))
    return rows


def best(data,pairs,candidates):
    def rank(tree):
        return (sum(r['hit'] for r in score(data,pairs,tree)),tree[1]==tree[2])
    return max(candidates,key=rank)


def main(out):
    out.mkdir()
    with ProcessPoolExecutor(max_workers=5) as pool:
        fit=collect(pool,range(3),out/'fitting.jsonl')
        pairs=list(permutations(range(3),2))
        tree=best(fit,pairs,product(OPTIONS,repeat=3))
        constant=best(fit,pairs,[(OPTIONS[0],a,a) for a in OPTIONS])
        oracle={g:max(OPTIONS,key=lambda a:sum(fit[p,g,a]['hit'] for p in range(3))) for g in REGIMES}
        (out/'frozen.json').write_text(json.dumps(dict(tree=tree,constant=constant,regime=oracle,
             fit_score=sum(r['hit'] for r in score(fit,pairs,tree)),fit_fixed=sum(r['hit'] for r in score(fit,pairs,constant))),indent=2))
        fresh=collect(pool,range(3,12),out/'fresh.jsonl')
        result={}
        for group in (list(range(3,6)),list(range(6,12))):
            pairs=list(zip(group,group[1:]+group[:1]))
            routed=score(fresh,pairs,tree)
            results={k:score(fresh,pairs,t) for k,t in [('history',tree),('fixed',constant)]}
            results['swapped']=score(fresh,pairs,tree,True)
            result[str(group[0])]=dict(scores={k:dict(hit=sum(r['hit'] for r in rows),final=sum(r['final'] for r in rows)) for k,rows in results.items()},
               fixed_options={a:sum(fresh[q,g,a]['hit'] for q,g in product(group,REGIMES)) for a in OPTIONS},
               regime_aware=sum(fresh[q,g,oracle[g]]['hit'] for q,g in product(group,REGIMES)),
               per_query=sum(any(fresh[q,g,a]['hit'] for a in OPTIONS) for q,g in product(group,REGIMES)),rows=routed)
        (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    main(p.parse_args().output)
