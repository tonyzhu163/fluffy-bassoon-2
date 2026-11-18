"""Fit on bowl continuations and run the frozen first-attempt policy on bottles."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from design import OLD, VARIANTS, fit, lag24, use_pacing
sys.path.insert(0,str(OLD))
import pilot


@contextmanager
def simulator(tasks):
    pipe, peer = Pipe()
    env = dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               PYTHONPATH=f'{pilot.PRO}:{pilot.SETUP}/simulator-source',
               LIBERO_CONFIG_PATH=str(pilot.SETUP/'libero-config'),LIBERO_PRO_ROOT=str(pilot.SETUP/'simulator-source'))
    worker = subprocess.Popen([pilot.SETUP/'env/bin/python','-Bu',OLD/'sim.py',str(peer.fileno()),*map(str,tasks)],
                              pass_fds=(peer.fileno(),),env=env)
    peer.close()
    try:
        yield pipe
    finally:
        pipe.close()
        worker.wait(timeout=30)


def episode(motor, pipe, directory, scene, variant, option, mode, *, model=None, bottle=False):
    directory.mkdir(parents=True)
    raw,physics,prompts = pilot.call(pipe,'reset',(scene,variant))
    hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()}
    rng = np.random.default_rng((19190000 if bottle else 18183300)+scene)
    actions,states,hits,frames,decisions = [],[],[],[],[]
    paced = True
    while len(actions)<300:
        if len(actions)==24:
            lag=lag24(actions,states)
            paced=use_pacing(model,lag) if mode=='learned' else mode=='paced'
            pilot.write(directory/'decision.json',dict(control=24,lag=lag,paced=paced,option=option))
        obs=pilot.observation(raw,prompts[option])
        decisions.append(len(actions))
        frames.append(np.stack([obs['observation/image'],obs['observation/wrist_image']]))
        noise=rng.standard_normal((10,32)).astype(np.float32)
        chunk=motor.infer(obs,noise=noise)['actions'][:10 if paced else 5].copy()
        assert chunk.shape==(10 if paced else 5,7) and np.isfinite(chunk).all()
        if paced:
            wait=np.zeros((14,7),dtype=chunk.dtype)
            wait[:,-1]=chunk[-1,-1]
            chunk=np.concatenate((chunk,wait))
        chunk=chunk[:300-len(actions)]
        raw,state,success=pilot.call(pipe,'chunk',chunk)
        actions.extend(chunk)
        states.extend(state)
        hits.extend(success)
    h=np.array(hits,dtype=bool)
    first=[int(np.flatnonzero(v)[0])+1 if v.any() else None for v in h.T]
    rec=dict(scene=scene,variant=variant,goal=option,prompt=prompts[option],mode=mode,paced=paced,lag=lag,
             controls=len(h),initial_physics=physics,initial_public=hashes,first=first,
             simultaneous=bool(h.all(axis=1).any()))
    np.save(directory/'evaluator-predicates.npy',h)
    np.savez_compressed(directory/'public.npz',issued=actions,state_after=states,frames=frames,decision_steps=decisions)
    pilot.write(directory/'result.json',rec)
    print(directory.name,first,flush=True)
    return rec


def transfer(motor,pipe,root,model):
    live=[]
    goal=model['option']
    for scene in range(6):
        for v in VARIANTS:
            block=root/f's{scene}-{v}'
            def support(option,mode):
                return episode(motor,pipe,block/f'support-g{option}-{mode}',scene,v,option,mode,model=model,bottle=True)
            selected=support(goal,'learned')
            instructions=[]
            for assigned in range(2):
                reward=selected['first'][assigned] is not None
                instructions.append(dict(assigned=assigned,reward=reward,chosen=goal if reward else 1-goal))
            pilot.write(block/'retry-decisions.json',instructions)
            query={}
            order=list(dict.fromkeys([i['chosen'] for i in instructions]+[0,1]))
            for option in order:
                query[option]=episode(motor,pipe,block/f'query-g{option}',scene+6,v,option,'paced',bottle=True)
            support(goal,'switch' if selected['paced'] else 'paced')
            for mode in ('paced','switch'):
                support(1-goal,mode)
            for i in instructions:
                hit=query[i['chosen']]['first'][i['assigned']]
                live.append(dict(scene=scene,variant=v,**i,first=hit))
            pilot.write(root/'live.json',live)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    root=p.parse_args().output
    root.mkdir()
    assert len(pilot.jax.devices())==1 and pilot.jax.devices()[0].platform=='gpu'
    for repo,pin in [(pilot.SRC,'215abfb217dbac7d5f1273282331b9b1866c0479'),(pilot.PRO,'e5500c690c1498eca1b7050350654964190cdf70')]:
        assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==pin
    motor=pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'),str(pilot.CKPT))
    pilot.write(root/'motor.json',dict(checkpoint=str(pilot.CKPT),device=str(pilot.jax.devices()[0]),
               norm_sha256=hashlib.sha256((pilot.CKPT/'assets/physical-intelligence/libero/norm_stats.json').read_bytes()).hexdigest()))
    bowl=root/'bowl'
    with simulator((4,8)) as pipe:
        for scene in range(16,20):
            for variant in VARIANTS:
                for option in range(2):
                    episode(motor,pipe,bowl/f's{scene}-{variant}-g{option}-switch',scene,variant,option,'switch')
    result=fit(bowl)
    pilot.write(root/'frozen-policy.json',result)
    print('frozen before transfer',result['learned'],flush=True)
    with simulator((2,9)) as pipe:
        transfer(motor,pipe,root/'bottle',result['learned']['model'])
    pilot.write(root/'complete.json',dict(bowl=16,bottle=72,finish=time.time()))


if __name__=='__main__':
    main()
