"""Run paired closed-loop practice and independent query episodes under the original cadence."""
from contextlib import contextmanager
import hashlib
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-bowl-decisions-20261118-r1'))
import pilot


@contextmanager
def simulator():
    pipe,peer=Pipe()
    env=dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONPATH=f'{pilot.PRO}:{pilot.SETUP}/simulator-source',
             LIBERO_CONFIG_PATH=str(pilot.SETUP/'libero-config'),LIBERO_PRO_ROOT=str(pilot.SETUP/'simulator-source'))
    worker=subprocess.Popen([pilot.SETUP/'env/bin/python','-Bu',HERE/'live_sim.py',str(peer.fileno())],
                            pass_fds=(peer.fileno(),),env=env)
    peer.close()
    try:yield pipe
    finally:
        pipe.close()
        worker.wait(timeout=30)


def trial(motor,pipe,raw,prompt,seed,prefix=None):
    rng=np.random.default_rng(seed)
    issued,states,hits,contact,heights=[],[],[],[],[]
    start=None
    while len(issued)<300:
        x=pilot.observation(raw,prompt)
        noise=rng.standard_normal((10,32)).astype(np.float32)
        if prefix is not None and start is None:
            a=prefix[len(issued):len(issued)+5]
        else:
            a=motor.infer(x,noise=noise)['actions'][:5]
        assert a.shape==(5,7) and np.isfinite(a).all()
        raw,s,h,c,z,start=pilot.call(pipe,'chunk',a)
        states.extend(s);hits.extend(h);contact.extend(c);heights.extend(z);issued.extend(a)
    return dict(issued=np.array(issued),states=np.array(states),hits=np.array(hits),
                contact=np.array(contact),heights=np.array(heights)),start


def main(output,drop=False):
    devices=pilot.jax.devices()
    assert len(devices)==1 and devices[0].platform=='gpu'
    for repo,pin in [(pilot.SRC,'215abfb217dbac7d5f1273282331b9b1866c0479'),(pilot.PRO,'e5500c690c1498eca1b7050350654964190cdf70')]:
        assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==pin
    motor=pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'),str(pilot.CKPT))
    output.mkdir()
    with simulator() as pipe:
        for p in range(24,30):
            identity=None
            for g in (0,1):
                for mode in (('D18',) if drop else ('sham','interrupt','query')):
                    raw,physics,prompts=pilot.call(pipe,'reset',(p,False,'D18') if drop else (p,mode=='interrupt'))
                    fingerprint=(physics,{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()})
                    if identity is None:identity=fingerprint
                    assert identity==fingerprint
                    seed=(19195100 if mode=='query' else 19195000)+p
                    prefix=None
                    if drop:
                        base=output.parent.parent/'closed01/episodes'/f'p{p}-g{g}-sham'
                        with np.load(base/'tape.npz') as reference:prefix=reference['issued']
                    tape,start=trial(motor,pipe,raw,prompts[g],seed,prefix)
                    if mode=='sham':sham,original=tape,start
                    if mode=='interrupt':
                        assert start==original
                        n=start if start is not None else 300
                        for k in ('states','hits','contact','heights','issued'):
                            np.testing.assert_array_equal(tape[k][:n],sham[k][:n])
                    extra={}
                    if drop:
                        import json
                        base=output.parent.parent/'closed01/episodes'/f'p{p}-g{g}-sham'
                        rec=json.loads((base/'result.json').read_text())
                        assert seed==rec['seed'] and physics==rec['physics']
                        n=300 if start is None else start
                        with np.load(base/'tape.npz') as original_tape:
                            for field in ('issued','states','hits','contact','heights'):
                                np.testing.assert_array_equal(tape[field][:n],original_tape[field][:n])
                        extra['event']=pilot.call(pipe,'event',None)
                    path=output/f'p{p}-g{g}-{mode}'
                    path.mkdir()
                    np.savez_compressed(path/'tape.npz',**tape)
                    pilot.write(path/'result.json',dict(pose=p,goal=g,mode=mode,seed=seed,start=start,physics=physics,
                                ever=tape['hits'].any(axis=0).tolist(),final=tape['hits'][-1].tolist(),**extra))
            print('completed pose',p,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2:]==['--drop'])
