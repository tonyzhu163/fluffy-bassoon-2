"""Collect practice tapes, save frozen reader choices, then execute prioritized native queries."""
from contextlib import contextmanager
import hashlib
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from choices import prospective
from reader import OLD
sys.path.insert(0,str(OLD))
from live import pilot,trial

HERE=Path(__file__).resolve().parent


@contextmanager
def simulator(anchor,ids):
    pipe,peer=Pipe()
    env=dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONPATH=f'{pilot.PRO}:{pilot.SETUP}/simulator-source',
             LIBERO_CONFIG_PATH=str(pilot.SETUP/'libero-config'),LIBERO_PRO_ROOT=str(pilot.SETUP/'simulator-source'))
    proc=subprocess.Popen([pilot.SETUP/'env/bin/python','-Bu',HERE/'object_sim.py',str(peer.fileno()),str(anchor),*map(str,ids)],
                          pass_fds=(peer.fileno(),),env=env)
    peer.close()
    try:yield pipe
    finally:
        pipe.close()
        proc.wait(timeout=30)


def main(out,model,reciprocal=False):
    devices=pilot.jax.devices()
    assert len(devices)==1 and devices[0].platform=='gpu'
    for repo,pin in [(pilot.SRC,'215abfb217dbac7d5f1273282331b9b1866c0479'),(pilot.PRO,'e5500c690c1498eca1b7050350654964190cdf70')]:
        assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==pin
    motor=pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'),str(pilot.CKPT))
    out.mkdir()
    if reciprocal:
        for anchor in (2,4):
            directory=out/f'anchor{anchor}'
            directory.mkdir()
            collect(directory,model,motor,anchor,(2,4))
    else:
        collect(out,model,motor,0,(0,7))


def collect(out,model,motor,anchor,ids):
    with simulator(anchor,ids) as pipe:
        starts={}
        def execute(p,g,mode):
            raw,physics,prompts=pilot.call(pipe,'reset',(p,mode=='interrupt'))
            identity=(physics,{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()})
            assert starts.setdefault(p,identity)==identity
            seed=(19201200 if mode=='query' else 19201100)+p
            tape,start=trial(motor,pipe,raw,prompts[g],seed)
            obj=pilot.call(pipe,'event',None)
            path=out/f'p{p}-g{g}-{mode}'
            path.mkdir()
            np.savez_compressed(path/'tape.npz',**tape)
            pilot.write(path/'result.json',dict(pose=p,goal=g,prompt=prompts[g],mode=mode,seed=seed,start=start,
                        object=obj,physics=physics,anchor=anchor,task_id=ids[g],ever=tape['hits'].any(axis=0).tolist(),final=tape['hits'][-1].tolist()))
            return tape,start,obj
        for p in range(6):
            for g in (0,1):
                base,k,obj=execute(p,g,'sham')
                changed,k2,obj2=execute(p,g,'interrupt')
                assert (k,obj)==(k2,obj2)
                n=k if k is not None else 300
                for field in base:np.testing.assert_array_equal(changed[field][:n],base[field][:n])
            print('practice pose',p,flush=True)
        actual=prospective(out,model)
        for row in actual:
            for g in (row['selected'],1-row['selected']):execute(row['q'],g,'query')
            print('query pose',row['q'],flush=True)


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3:]==['--reciprocal'])
