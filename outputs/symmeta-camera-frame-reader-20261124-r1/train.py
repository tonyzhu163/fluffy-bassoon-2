"""Train the paired models, inspect source-only validation and seal the common budget."""
import argparse
from dataclasses import asdict
import hashlib
import time
import os
from pathlib import Path
from flax import serialization
import cv2
import jax
import numpy as np
import eval_wider_family as pr7
from camera import TRAIN, VALIDATION, EXTRA_TRAIN, EXTRA_VAL
from data import SENSOR, read, write, sha, gather, load_bank, prepare, prepare_extra
from model import CFG, LABELS, OPT, init, infer, restore, decode, step


def wrap(a):
    return (a+180)%360-180


def validation(p, bank, controls, get_view=gather):
    stats, errors = [], []
    n = len(controls['valid'])
    for j, pose in enumerate(VALIDATION+EXTRA_VAL):
        yaw = []
        for start in range(0,n,32):
            idx = np.minimum(np.arange(start,start+32), n-1)
            b = {k:controls[k][idx] for k in SENSOR}
            b.update(get_view(bank,j,idx))
            yaw.extend(decode(infer(p,b))[0][:min(n-start,32)])
        err = np.abs(wrap(yaw-controls['yaw_deg']))
        stats.append(dict(pose=pose,mae=float(err.mean())))
        errors.append(err)
    return stats,np.stack(errors)


def train(out,cache,extra,arms=('B','A'),get_view=gather):
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    probe=read(out/'gpu.json')
    name=probe['devices'][0]['cuda_name'].upper()
    assert probe['status']=='passed' and probe['exclusive_training_admission'] and len(probe['devices'])==1
    assert name in ('NVIDIA GEFORCE RTX 4090','NVIDIA RPBZZZ6') or 'RTX PRO 6000' in name
    seal=read(cache/'ready.json')
    assert seal['source']==sha(cache/'source.json')
    assert read(extra/'ready.json')['source']==seal['source']
    def joined(split, arm):
        first = load_bank(cache,split,arm,False)
        more = load_bank(extra,split,arm,False)
        return {k:np.concatenate([first[k],more[k]],axis=0) for k in first}
    current = {k:v for k,v in np.load(cache/'train-control.npz').items()}
    val = {k:v for k,v in np.load(cache/'val-control.npz').items()}
    records, paths, exposure = [],{},{}
    checks = (1000,2000,2500,3500,4500,5000)
    for arm in arms:
        representation = 'B' if arm == 'C' else arm
        bank, val_bank = joined('train',representation), joined('val',representation)
        for seed in pr7.SEEDS:
            start = time.monotonic()
            rng = np.random.default_rng(seed*10007+31)
            cameras = np.random.default_rng(seed*4099+217)
            digest = hashlib.sha256()
            p = init(seed)
            state = OPT.init(p)
            params = sum(x.size for x in jax.tree.leaves(p))
            prob = np.full(len(TRAIN+EXTRA_TRAIN), .8/(len(TRAIN+EXTRA_TRAIN)-1))
            prob[0] = .2
            for i in range(1,5001):
                idx = rng.choice(len(current['valid']), size=32, replace=True)
                b = pr7.relabel({k:v[idx] for k,v in current.items()},pr7.sample_mixed(rng,('family',),32))
                labels = {k:b[k] for k in LABELS}
                pose = cameras.choice(len(TRAIN+EXTRA_TRAIN),size=32,p=prob)
                for v in (idx,pose,*labels.values()):
                    digest.update(v.tobytes())
                obs = {k:b[k] for k in SENSOR}
                obs.update(get_view(bank,pose,idx))
                p,state,loss = step(p,state,obs,labels)
                if i in checks:
                    l = float(loss)
                    assert np.isfinite(l)
                    stats, err = validation(p,val_bank,val,get_view)
                    np.save(out/f'val-{arm}-{seed}-{i}.npy',err)
                    records.append(dict(arm=arm,seed=seed,steps=i,loss=l,poses=stats,
                                        mae=float(err.mean()),seconds=time.monotonic()-start))
                    write(out/'source-curves.json',records)
                    print(arm,seed,i,l,'source validation',float(err.mean()),flush=True)
                if i in (2500,5000):
                    path=out/f'{arm}-seed{seed}-step{i}.msgpack'
                    path.write_bytes(serialization.to_bytes(p))
                    restored=restore(path,seed)
                    for x,y in zip(jax.tree.leaves(p),jax.tree.leaves(restored),strict=True):
                        np.testing.assert_array_equal(x,y)
                    paths[path.name]=sha(path)
            ex=digest.hexdigest()
            assert exposure.setdefault(seed,ex)==ex
        del bank,val_bank
    selection_arm = 'B' if 'B' in arms else arms[0]
    mean = {n:float(np.mean([r['mae'] for r in records if r['arm']==selection_arm and r['steps']==n])) for n in (2500,5000)}
    selected = min(mean,key=mean.get)
    write(out/'sealed.json',dict(steps=selected,source_B_mean=mean,params_per_arm=params,
          source=seal['source'],training_exposures=exposure,checkpoint_sha256=paths,
          cfg=asdict(CFG),revision=(out/'launch.txt').read_text().split()[0],
          source_poses=TRAIN+EXTRA_TRAIN,validation_poses=VALIDATION+EXTRA_VAL))
    print('SEALED',selected,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('prepare','train','extra'))
    ap.add_argument('cache',type=Path)
    ap.add_argument('out',type=Path,nargs='?')
    ap.add_argument('extra',type=Path,nargs='?')
    a=ap.parse_args()
    cv2.setNumThreads(1)
    if a.mode=='prepare':
        assert os.environ['CUDA_VISIBLE_DEVICES']==''
        prepare(a.cache)
    elif a.mode=='extra':
        assert os.environ['CUDA_VISIBLE_DEVICES']==''
        prepare_extra(a.cache,a.out)
    else:
        train(a.out,a.cache,a.extra)
