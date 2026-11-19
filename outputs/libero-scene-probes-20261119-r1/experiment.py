"""Collect all native goal branches in the same reset, then evaluate a frozen selector."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from selector import features,fit

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-first-attempt-20261119-r1'))
from importlib.util import module_from_spec,spec_from_file_location
spec=spec_from_file_location('qualified_first_attempt',HERE.parent/'libero-first-attempt-20261119-r1/experiment.py')
previous=module_from_spec(spec)
spec.loader.exec_module(previous)
pilot=previous.pilot

TRAIN=(4,8,2,9)
EVAL=TRAIN+(0,1)
LAYOUTS={'base':('BASE',4),'cabinet':('S02',4),'plate':('S02',8),'rack':('S02',9)}


def hash_obs(raw):
    return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()}


def prepare(motor,raw,prompts,goals,model):
    public={}
    for goal,prompt in zip(goals,prompts):
        obs=pilot.observation(raw,prompt)
        rng=np.random.default_rng(191104)
        chunks=[]
        for _ in range(3):
            noise=rng.standard_normal((10,32)).astype(np.float32)
            chunk=motor.infer(obs,noise=noise)['actions'][:10]
            assert chunk.shape==(10,7) and np.isfinite(chunk).all()
            chunks.append(chunk)
        visual,plan=features(obs,np.stack(chunks))
        public[goal]=(visual,plan,model.score(goal,visual,plan) if model is not None else None)
    return public


def rollout(motor,pipe,raw,prompt,seed,goals,output):
    rng=np.random.default_rng(seed)
    hits=[]
    while len(hits)<300:
        item=pilot.observation(raw,prompt)
        chunk=motor.infer(item,noise=rng.standard_normal((10,32)).astype(np.float32))['actions'][:5]
        assert chunk.shape==(5,7) and np.isfinite(chunk).all()
        raw,_,success=pilot.call(pipe,'chunk',chunk)
        hits.extend(success)
    hit=np.array(hits,dtype=bool)
    assert hit.shape==(300,len(goals))
    np.save(output/'native.npy',hit)
    return {str(g):int(np.flatnonzero(hit[:,i])[0])+1 if hit[:,i].any() else None for i,g in enumerate(goals)}


def collect(motor,root,scenes,goals,model):
    with previous.simulator(goals) as pipe:
        for scene in scenes:
            for layout,(variant,world) in LAYOUTS.items():
                block=root/f'p{scene}'/layout
                if block.is_symlink():
                    continue
                block.mkdir(parents=True)
                reset=(scene,variant,world)
                raw,physics,prompts=pilot.call(pipe,'reset',reset)
                hashes=hash_obs(raw)
                public=prepare(motor,raw,prompts,goals,model)
                initial=pilot.observation(raw,prompts[0])
                np.savez_compressed(block/'views.npz',agent=initial['observation/image'],wrist=initial['observation/wrist_image'])
                pilot.write(block/'choices.json',dict(scene=scene,layout=layout,physics=physics,public=hashes,
                            goals=goals,prompts=prompts,scores={str(g):public[g][2] for g in goals}))
                for i,goal in enumerate(goals):
                    if i:
                        raw,again,langs=pilot.call(pipe,'reset',reset)
                        assert again==physics and hash_obs(raw)==hashes and langs==prompts
                    path=block/f'g{goal}'
                    path.mkdir()
                    visual,plan,_=public[goal]
                    np.savez_compressed(path/'initial.npz',visual=visual,plan=plan)
                    first=rollout(motor,pipe,raw,prompts[i],19112700+scene,goals,path)
                    pilot.write(path/'result.json',dict(scene=scene,layout=layout,goal=goal,first=first))
                print(root.name,scene,layout,'complete',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--source-attempt',type=Path)
    args=parser.parse_args()
    out=args.output
    out.mkdir()
    assert len(pilot.jax.devices())==1 and pilot.jax.devices()[0].platform=='gpu'
    for repo,pin in [(pilot.SRC,'215abfb217dbac7d5f1273282331b9b1866c0479'),(pilot.PRO,'e5500c690c1498eca1b7050350654964190cdf70')]:
        assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==pin
    motor=pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'),str(pilot.CKPT))
    frozen=out/'frozen.npz'
    if args.source_attempt:
        from selector import Reachability
        import json
        prior=args.source_attempt.resolve()
        receipt=json.loads((prior/'fit.json').read_text())
        assert receipt['rows']==64 and receipt['goals']==list(TRAIN)
        assert hashlib.sha256((prior/'frozen.npz').read_bytes()).hexdigest()==receipt['sha256']
        assert len(list((prior/'fitting').glob('p*/*/g*/result.json')))==64
        (out/'fitting').symlink_to(prior/'fitting',target_is_directory=True)
        shutil.copyfile(prior/'frozen.npz',frozen)
        shutil.copyfile(prior/'fit.json',out/'fit.json')
        base=prior/'fresh/p32/base'
        for goal in EVAL:
            r=json.loads((base/f'g{goal}/result.json').read_text())
            assert r['goal']==goal and r['scene']==32 and r['layout']=='base'
            hits=np.load(base/f'g{goal}/native.npy')
            assert hits.shape==(300,len(EVAL))
        (out/'fresh/p32').mkdir(parents=True)
        (out/'fresh/p32/base').symlink_to(base,target_is_directory=True)
        pilot.write(out/'migration.json',dict(source=str(prior),fit_rows=64,fresh_rows=6,feature_hash=receipt['sha256']))
        model=Reachability(frozen)
    else:
        collect(motor,out/'fitting',range(28,32),TRAIN,None)
        model=fit(out/'fitting',frozen)
        pilot.write(out/'fit.json',dict(rows=len(model.goals),scale=model.scale.tolist(),
            sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),goals=TRAIN,finish=time.time()))
    print('frozen before fresh collection',flush=True)
    collect(motor,out/'fresh',range(32,36),EVAL,model)
    pilot.write(out/'complete.json',dict(fitting=64,fresh=96,finish=time.time()))


if __name__=='__main__':
    main()
