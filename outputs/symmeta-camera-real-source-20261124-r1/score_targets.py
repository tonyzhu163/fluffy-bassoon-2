"""Read only released target banks after a common source-only budget is sealed."""
import csv
import gzip
import json
import sys
from pathlib import Path
import cv2
import jax
import numpy as np

from camera import FULL, view_inputs
from data import ROOT, SENSOR, read, sha, write
from evaluate import physical_stream, digest
from fit_real import ARMS, BUDGETS
from model import decode, infer, restore
from selection import HERE

UR5E_SHA='8071d5d451fb625fc3cf611cfb94ff3d16bf43ae280b565f7d564cc51f494f08'
KINOVA_SHA='8639817ce59b79ecfe26564e94c0bb61352a3a5942b13a5c8edd999a6c5d6e60'
PRIMARY_PRED='f90a8505ba07239c125c9f7ee0f924af799173ddbe0dba8c9c2c625f1da3ae86'


def inputs(body):
    if body=='UR5e':
        path=ROOT/'symmeta-camera-pose-pairs-20261124-r1/cpu01/ood64/bank.json'
        assert sha(path)==UR5E_SHA
        for view,angle,_,group in physical_stream(path):
            if view in ('nominal','orbit-','orbit+'):
                yield view,angle,group
    else:
        path=ROOT/'symmeta-camera-independent-body-20261124-r1/cpu01/bank64.json'
        from score_frozen import load_bank as checked_bank, observation
        rows=checked_bank(path,KINOVA_SHA,64)
        for view,angle in (('nominal',0),('orbit-',-36),('orbit+',36)):
            group=[]
            for row in rows:
                # The independent reader loader accepts the physical view through assigned_view.
                # It uses the opposite orbit for its wrong metadata control (unused for nominal).
                obs,ctrl,h,opp=observation(dict(row,assigned_view=view))
                v=row['views'][view]
                assert v['angle_deg']==angle
                meta=dict(index=row['index'],suite=row['suite'],task_id=row['task_id'],task=f"{row['suite']}/{row['task_id']}",
                          embodiment=row['native_control_id'],regime=row['regime'],true_yaw_deg=row['control_yaw_deg'],
                          true_gain=row['xy_scale'],true_sign=row['gripper_sign'],source_sha256=row['archive_sha256'],
                          native_view_sha256=v['sha256'],physical_id=row['id'])
                group.append((obs,FULL,h,opp,ctrl,meta))
            yield view,angle,group


def score(fit,body,out):
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    cv2.setNumThreads(1)
    seal=read(fit/'sealed.json')
    paths={}
    for label in ARMS:
        for seed in (41,42,43):
            for n in BUDGETS:
                path=fit/f'{label}-seed{seed}-step{n}.msgpack'
                assert sha(path)==seal['checkpoint_sha256'][path.name]
                paths[label,seed,n]=(restore(path,seed),path)
    proto='symmeta-camera-'+('frame-native' if body=='UR5e' else 'independent-body')+'-20261124-r1'
    with (HERE.parent/proto/'cases.csv').open() as f: registry={int(r['index']):r for r in csv.DictReader(f)}
    assert len(registry)==64
    rows=[]
    cross_rows=[]
    for view,angle,group in inputs(body):
        assert len(group)==64 and {r[-1]['index'] for r in group}==set(registry)
        for cal in (('true',) if view=='nominal' else ('true','identity','opposite')):
            for arm in 'AB':
                batches,metadata=[],[]
                for obs,mask,h,opp,ctrl,m in group:
                    case=registry[m['index']]
                    assert case['support_sha256']==m['source_sha256'] and case['embodiment']==m['embodiment']
                    H=dict(true=h,identity=np.eye(3),opposite=opp)[cal]
                    f=view_inputs(obs,mask,H,arm)
                    batches.append(dict(**f,**ctrl))
                    assigned=int(case['orbit_deg' if body=='UR5e' else 'angle_deg'])
                    metadata.append(dict(**m,body=body,view=view,camera_deg=angle,assigned=angle==assigned,
                        calibration=cal,observed_sha256=digest(obs),calibration_sha256=digest(H),
                        sensor_sha256={k:digest(ctrl[k]) for k in SENSOR},coverage=float(f['pixel_valid'].mean())))
                b={k:np.stack([x[k] for x in batches]) for k in batches[0]}
                for (label,seed,n),(weights,path) in paths.items():
                    cross=label[-1]!=arm
                    if cross and (cal!='true' or n!=seal['steps']): continue
                    target_rows=cross_rows if cross else rows
                    yaw,gain,sign=decode(infer(weights,b))
                    for m,y,g,s in zip(metadata,yaw,gain,sign,strict=True):
                        target_rows.append(dict(**m,model=label,input_representation=arm,seed=seed,budget=n,selected=n==seal['steps'],yaw_deg=float(y),
                            xy_scale=float(g),gripper_sign=int(s),checkpoint_path=str(path),checkpoint_sha256=seal['checkpoint_sha256'][path.name]))
    assert len(rows)==64*7*4*3*2
    assert len(cross_rows)==64*3*4*3
    cross_path=out/f'{body}-cross-input.json.gz'
    with gzip.open(cross_path,'wt') as f: json.dump(cross_rows,f,separators=(',',':'),allow_nan=False)
    target=out/f'{body}-predictions.json.gz'
    with gzip.open(target,'wt') as f: json.dump(rows,f,separators=(',',':'),allow_nan=False)
    write(out/f'{body}-receipt.json',dict(seal_sha256=sha(fit/'sealed.json'),bank_sha256=UR5E_SHA if body=='UR5e' else KINOVA_SHA,
            rows=len(rows),cross_rows=len(cross_rows),cross_sha256=sha(cross_path),assigned_supports=64,sha256=sha(target),selected=seal['steps']))
    from reduce_scores import reduce_panel
    old=None
    if body=='UR5e':
        orig=ROOT/'symmeta-camera-frame-reader-20261124-r1/phyGpu01/physical-predictions.json.gz'
        assert sha(orig)==PRIMARY_PRED
        with gzip.open(orig,'rt') as f: old=json.load(f)
    write(out/f'{body}-summary.json',reduce_panel(rows,old))


if __name__=='__main__':
    fit,body,out=sys.argv[1:]
    assert body in ('UR5e','Kinova3')
    score(Path(fit),body,Path(out))
