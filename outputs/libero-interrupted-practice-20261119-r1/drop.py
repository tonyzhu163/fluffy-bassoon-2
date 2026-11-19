"""Evaluate the released D18 object move on unchanged paired bottle commands."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from replay import ROOT,ControlEnv,LiberoEnvironmentAdapter,TaskManifest,setup,read_tape
from score import features,fit,load,score

LIVE=ROOT/'outputs/libero-interrupted-practice-20261119-r1/closed01/episodes'


def fork(job):
    p,g=job
    issued,states,native,physics=read_tape('live',p,g,0)
    start_pose,goals,bank=setup('live')
    manifest=TaskManifest.load(str(ROOT/'sources/libero-pro2-e5500c69/assets/task_manifest.json'))
    path=manifest.resolve_bddl('libero_goal',2,'D18')
    env=LiberoEnvironmentAdapter()
    env._init_states=bank;env._max_steps=300
    env._env=ControlEnv(str(path),use_camera_obs=False,has_offscreen_renderer=False,camera_names=[])
    try:
        reset=env.reset(p,18180100+p)
        assert reset['environment_reproducibility']['initial_state_fingerprint']==physics
        w=env._env.env
        event=None
        observed=[];hits=[];contact=[]
        for i,a in enumerate(issued):
            step=env.step(a)
            raw=step['observation']
            st=np.concatenate([raw[k] for k in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')])
            h=[all(w._eval_predicate(t) for t in terms) for terms in goals]
            assert step['success']==h[0]
            logs=w.perturbation_runtime.diagnostics()
            assert logs['case_id']=='D18'
            changes=[v for v in logs['events'] if v['event']=='dynamic_state_change']
            if changes and event is None:
                assert len(changes)==1 and changes[0]['target']=='wine_bottle_1'
                event=dict(control=i+1,details=changes[0])
            if event is None:
                np.testing.assert_array_equal(st,states[i])
                np.testing.assert_array_equal(h,native[i])
            observed.append(st);hits.append(h)
            contact.append(bool(w._check_grasp(w.robots[0].gripper,w.objects_dict['wine_bottle_1'])))
        rows=[]
        for assigned,ss,hh in [(False,states,native),(True,np.array(observed),np.array(hits))]:
            row=dict(family='live',pose=p,goal=g,draw=0,interrupted=assigned,ever=hh.any(axis=0).tolist(),
                     start=event['control']-1 if event is not None else None,lost_contact=None,final=hh[-1].tolist())
            row.update(features(issued,ss))
            rows.append(row)
        return rows,dict(pose=p,goal=g,event=event,sham=rows[0]['ever'],drop=rows[1]['ever'])
    finally:env.close()


if __name__=='__main__':
    archived,out=map(Path,sys.argv[1:])
    source=load(archived)
    model=fit([r for (f,p,g,j,b),r in source.items() if f=='bowl' and p in range(32,36)])
    data={};events=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for pair,event in pool.map(fork,list(product(range(24,30),(0,1)))):
            events.append(event)
            for r in pair:data['live',r['pose'],r['goal'],0,r['interrupted']]=r
    def query(family,p):
        vals=[]
        for g in (0,1):
            with np.load(LIVE/f'p{p}-g{g}-query/tape.npz') as z:vals.append([z['hits'].any(axis=0),z['hits'][-1]])
        return [np.stack(v) for v in zip(*vals)]
    result=score(data,model,'live',range(24,30),1,query)
    for key in ('loss_on_interrupt','loss_on_sham'):del result['physical'][key]
    result['events']=events
    (out/'drop.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    print(result['physical'],result['zero'])
