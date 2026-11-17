"""Select one condition-independent correction using development rollouts only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from native import load, rollout, write, ROOT
from qualify import conditions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--rank',type=int,choices=range(3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    def log(kind,**data):
        row=dict(kind=kind,time=time.time(),**data)
        with (a.output/'events.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    native,motor,codec,stats,identity=load(a.pin)
    write(a.output/'motor.json',identity)
    study=ROOT/'outputs'/Path(__file__).parent.name
    candidates=[[0.,0.,0.]]+conditions()[:8]
    results=[]
    for candidate in range(a.rank,9,3):
        for condition in range(8):
            if time.time() >= a.deadline:
                write(a.output/'summary.json',dict(complete=False,results=results)); return
            rank=condition%4
            phase='qual03' if rank==0 else 'qual06' if rank==2 else 'qual04'
            reference=study/phase/f'rank{rank}'/'native'/f'condition-{condition:02d}-f0'
            original=json.loads((reference/'result.json').read_text())
            path=a.output/f'candidate-{candidate}-condition-{condition}'
            result=rollout(native,motor,codec,directory=path,scene=original['scene'],offset=conditions()[condition],
                fraction=0.,fixed_offset=candidates[candidate],target_start=original['start_signature'],
                attempt=1,save_images=False,log=log)
            if candidate==0:
                with np.load(reference/'episode.npz') as old, np.load(path/'episode.npz') as new:
                    np.testing.assert_array_equal(new['actions'],old['actions'])
                assert result['physical_hashes']==original['physical_hashes']
            results.append(dict(candidate=candidate,compensation=candidates[candidate],condition=condition,
                scene=original['scene'],success=result['success'],controls=result['controls']))
            write(a.output/'progress.json',dict(results=results))
    write(a.output/'summary.json',dict(complete=True,results=results,
        selection='Maximize development success; tie: fewer controls, then smaller norm, then candidate index.',
        access='One constant across conditions; given approach-window timing, never the hidden condition.'))


if __name__=='__main__': main()
