"""Matched fresh fits. Every draw stream is reproduced for all four estimators."""
import argparse
import gzip
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from flax import serialization
import jax
import numpy as np
import eval_wider_family as pr7
from camera import TRAIN, EXTRA_TRAIN
from data import SENSOR, VIS, read, sha, write, load_bank
from model import CFG, LABELS, OPT, init, infer, restore, decode, step
from real_bank import ANGLES, SOURCE, arrays, cases
from selection import HERE

ARMS=('B','A','digital_B','digital_A')
CHECKS=(0,1000,2500,3500,4500,5000)
BUDGETS=(2500,5000)


def wrap(x):
    return (x+180)%360-180


def join(digital,real):
    return {k:np.concatenate((x,real[k]),axis=0) for k,x in digital.items()}


def observe(weights,vis,controls,idx,pose,mode=0):
    pred=[]
    for i in range(0,len(idx),32):
        ii=idx[i:i+32]
        b={k:controls[k][ii] for k in SENSOR}
        b.update({k:v[ii,pose,mode] for k,v in vis.items()})
        pred.append(np.stack(decode(infer(weights,b)),axis=1))
    return np.concatenate(pred)


def source_curve(weights,vis,controls,indices,split):
    predictions=np.stack([observe(weights,vis,controls,indices,j) for j in range(len(ANGLES[split]))])
    err=np.abs(wrap(predictions[:,:,0]-controls['yaw_deg'][indices]))
    return predictions,dict(n=len(indices),nominal=float(err[0].mean()),orbit=float(err[1:].mean()),
            by_angle={str(a):float(err[j].mean()) for j,a in enumerate(ANGLES[split])})


def train(out,folder,primary):
    start=time.monotonic()
    probe=read(out/'gpu.json')
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    assert probe['status']=='passed' and probe['exclusive_training_admission'] and '4090' in probe['devices'][0]['cuda_name']
    base,extra=[primary/k/'bank' for k in ('prep01','offset01')]
    for directory in (base,extra):
        assert read(directory/'ready.json')['source']==sha(directory/'source.json')==SOURCE
    ready=read(folder/'ready.json')
    assert ready['digital_source']==SOURCE and ready['source_ids_sha256']==sha(HERE/'source-ids.json')
    for name,digest in ready['sha256'].items(): assert sha(folder/name)==digest,name
    keep=np.asarray([r['bank_index'] for r in read(folder/'digital-ids.json')])
    with np.load(base/'train-control.npz') as z:
        digital_controls={k:z[k][keep] for k in z}
    with np.load(folder/'train-controls.npz') as z: real_controls=dict(z)
    with np.load(folder/'val-controls.npz') as z: val_controls=dict(z)
    curves,paths,exposures=[],{},{}
    angles=len(TRAIN+EXTRA_TRAIN)
    prob=np.full(angles,.8/(angles-1)); prob[0]=.2
    assert angles==23
    for label in ARMS:
        arm=label[-1]
        first=load_bank(base,'train',arm,False)
        other=load_bank(extra,'train',arm,False)
        digital={k:np.concatenate([first[k][:,keep],other[k][:,keep]],axis=0) for k in VIS}
        real={k:np.asarray(v).copy() for k,v in arrays(folder,'train',label,'r').items()}
        val=arrays(folder,'val',arm,'r')
        train_full=arrays(folder,'train',arm,'r')
        for seed in pr7.SEEDS:
            tick=time.monotonic()
            rng=np.random.default_rng(seed*10007+31)
            cameras=np.random.default_rng(seed*4099+217)
            draw_hash=hashlib.sha256()
            draws=np.empty((5000,16,5),np.int32)
            totals=np.zeros((128,9,5),np.int32)
            estimates={}
            p=init(seed); state=OPT.init(p)
            param_count=sum(x.size for x in jax.tree.leaves(p))
            assert param_count==2124438
            for i in range(5001):
                if i:
                    di=rng.integers(len(keep),size=16)
                    ri=rng.integers(len(real_controls['valid']),size=16)
                    dp=cameras.choice(angles,size=16,p=prob)
                    rp=cameras.integers(9,size=16)
                    mask=np.r_[np.zeros(8,dtype=int),cameras.integers(1,5,size=8)]
                    cameras.shuffle(mask)
                    raw=join({k:v[di] for k,v in digital_controls.items()},{k:v[ri] for k,v in real_controls.items()})
                    relabel=pr7.relabel(raw,pr7.sample_mixed(rng,('family',),32))
                    vis=join({k:v[dp,di] for k,v in digital.items()},{k:v[ri,rp,mask] for k,v in real.items()})
                    labels={k:relabel[k] for k in LABELS}
                    sensors={k:relabel[k] for k in SENSOR}
                    draws[i-1]=np.stack((keep[di],dp,ri,rp,mask),axis=1)
                    np.add.at(totals,(ri,rp,mask),1)
                    for a in (draws[i-1],*labels.values(),*sensors.values()): draw_hash.update(a.tobytes())
                    p,state,loss=step(p,state,dict(**vis,**sensors),labels)
                if i in CHECKS:
                    train_pred,train_stat=source_curve(p,train_full,real_controls,np.arange(0,128,4),'train')
                    val_pred,val_stat=source_curve(p,val,val_controls,np.arange(64),'val')
                    estimates[f'train_{i}']=train_pred
                    estimates[f'val_{i}']=val_pred
                    err=None if not i else float(loss)
                    assert err is None or np.isfinite(err)
                    curves.append(dict(arm=label,seed=seed,step=i,train=train_stat,validation=val_stat,
                                       loss=err,seconds=time.monotonic()-tick))
                    write(out/'curves.json',curves)
                    print(label,seed,i,train_stat['orbit'],val_stat['orbit'],flush=True)
                if i in BUDGETS:
                    path=out/f'{label}-seed{seed}-step{i}.msgpack'
                    path.write_bytes(serialization.to_bytes(p))
                    revived=restore(path,seed)
                    for x,y in zip(jax.tree.leaves(p),jax.tree.leaves(revived),strict=True):
                        np.testing.assert_array_equal(x,y)
                    paths[path.name]=sha(path)
                    digest=draw_hash.hexdigest()
                    key=f'{seed}-{i}'
                    assert exposures.setdefault(key,digest)==digest
                    if label==ARMS[0]:
                        np.savez_compressed(out/f'draws-{seed}-{i}.npz',draws=draws[:i],real_counts=totals)
            np.savez_compressed(out/f'source-pred-{label}-{seed}.npz',**estimates)
        del digital,real,first,other,train_full,val
    scores={n:float(np.mean([r['validation']['orbit'] for r in curves if r['arm']=='B' and r['step']==n])) for n in BUDGETS}
    selected=min(BUDGETS,key=scores.get)
    write(out/'sealed.json',dict(steps=selected,source_B_mean=scores,cfg=asdict(CFG),params_per_arm=param_count,
        arms=ARMS,seeds=pr7.SEEDS,training_exposures=exposures,checkpoint_sha256=paths,ready_sha256=sha(folder/'ready.json'),
        source_ids_sha256=sha(HERE/'source-ids.json'),digital_ids_sha256=sha(folder/'digital-ids.json'),
        git_pin=(out/'launch.txt').read_text().split()[0],elapsed_seconds=time.monotonic()-start))
    print('SEALED',selected,scores,flush=True)


def source_controls(out,folder):
    """Sealed-weight controls on the same genuine validation observations."""
    seal=read(out/'sealed.json')
    records=[]
    with np.load(folder/'val-controls.npz') as z: controls=dict(z)
    rows=cases('val')
    for label in ARMS:
        val=arrays(folder,'val',label[-1],'r')
        for seed in pr7.SEEDS:
            path=out/f"{label}-seed{seed}-step{seal['steps']}.msgpack"
            assert sha(path)==seal['checkpoint_sha256'][path.name]
            weights=restore(path,seed)
            from real_bank import MODES
            for j,angle in enumerate(ANGLES['val']):
                for m,mode in enumerate(MODES):
                    pred=observe(weights,val,controls,np.arange(len(rows)),j,m)
                    for i,(c,(yaw,gain,sign)) in enumerate(zip(rows,pred,strict=True)):
                        records.append(dict(model=label,seed=seed,angle=angle,mode=mode,id=c['id'],body=c['arm'],
                            task=f"{c['suite']}/{c['task_id']}",yaw_deg=float(yaw),xy_scale=float(gain),gripper_sign=int(sign),
                            true_yaw_deg=float(controls['yaw_deg'][i]),true_gain=float(controls['xy_scale'][i]),
                            true_sign=int(controls['gripper_sign'][i])))
    with gzip.open(out/'source-controls.json.gz','wt') as f: json.dump(records,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path); p.add_argument('cache',type=Path); p.add_argument('primary',type=Path)
    a=p.parse_args()
    train(a.out,a.cache,a.primary)
    source_controls(a.out,a.cache)
