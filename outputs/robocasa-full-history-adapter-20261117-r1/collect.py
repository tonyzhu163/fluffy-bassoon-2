"""Predeclared nominal screening and causal three-attempt visual trials."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from native import load, rollout, write
from qualify import conditions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=range(4),required=True)
    p.add_argument('--split',choices=('train','development','test'),required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    specification=json.loads((Path(__file__).parent/'SPLITS.json').read_text())['splits'][a.split]
    def log(kind,**data):
        row=dict(kind=kind,time=time.time(),**data)
        with (a.output/'events.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    native,motor,codec,stats,identity=load(a.pin)
    write(a.output/'motor.json',identity)
    admitted=[]; screening=[]; trials=[]
    for scene in specification['candidates'][a.rank::4]:
        if len(admitted) == specification['target']//4 or time.time() >= a.deadline:
            break
        nominal=rollout(native,motor,codec,directory=a.output/f'nominal-{scene}',scene=scene,
                        offset=[0,0,0],fraction=0,save_images=False,log=log)
        eligible=nominal['success'] and nominal['onset'] is not None
        screening.append(dict(scene=scene,eligible=eligible,success=nominal['success'],onset=nominal['onset']))
        write(a.output/'screening.json',dict(candidates=specification['candidates'][a.rank::4],results=screening))
        if not eligible: continue
        ordinal=len(admitted)
        admitted.append(dict(scene=scene,start_signature=nominal['start_signature']))
        write(a.output/'admitted.json',admitted)
        if a.split == 'test': continue
        for condition in specification['condition_pairs'][a.rank*(specification['target']//4)+ordinal]:
            fraction=float(np.random.default_rng(scene+condition*1009).uniform())
            paths=[]
            for attempt,behavior in enumerate((0.,fraction,1.),1):
                if time.time() >= a.deadline:
                    write(a.output/'summary.json',dict(complete=False,reason='deadline',admitted=admitted,
                        screening=screening,trials=trials,incomplete_trial=paths))
                    return
                path=a.output/f'scene-{scene}-condition-{condition}-attempt-{attempt}'
                rollout(native,motor,codec,directory=path,scene=scene,offset=conditions()[condition],
                        fraction=behavior,target_start=nominal['start_signature'],attempt=attempt,log=log)
                paths.append(str(path))
            trials.append(paths)
            write(a.output/'corpus.json',dict(trials=trials,split=a.split,rank=a.rank))
    write(a.output/'summary.json',dict(complete=len(admitted)==specification['target']//4,
        admitted=admitted,screening=screening,trials=trials,
        note='Nominally solvable subset; all perturbed and teacher failures retained'))


if __name__=='__main__': main()
