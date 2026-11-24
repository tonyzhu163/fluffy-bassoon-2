"""Sealed checkpoint scoring with observed pixels and separate evaluator-only identities."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import cv2
import jax
import numpy as np
from eval_stage2a_live_ur5e import load_v2_checkpoint
from fluffy_bassoon.xemb.equiv_v2 import apply_reader_v2
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from score import official_panel, load_prefix, set_digest, SUPPORT_SET_SHA
from camera import FULL, TARGET, encoding, frames_of, homography, render, view_inputs
from data import REPO, ROOT, SENSOR, sha, read, write
from model import CFG, decode, infer, restore

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def pose_name(roll,orbit):
    if orbit:
        return f'plane{orbit:+d}'
    return f'roll{roll:+d}' if roll else 'nominal'


def digital_stream():
    cases,_=official_panel(ROOT.parent)
    raw=[]
    for case in cases:
        t,identity,_=load_prefix(case,True)
        assert np.array_equal(t['img_before'][0,0],t['frame0_img'][0])
        spec=get_embodiment_spec(case['embodiment'])
        meta = {k:case[k] for k in ('index','suite','task_id','embodiment')}
        meta.update(task=f"{case['suite']}/{case['task_id']}", regime=spec.regime,
                    source_sha256=identity['sha256'],native_view_sha256=identity['sha256'],
                    true_yaw_deg=spec.yaw_deg,true_gain=spec.xy_scale,true_sign=spec.gripper_sign)
        raw.append((frames_of(t,0),{k:t[k][0] for k in SENSOR},meta))
    assert len(raw)==96 and sum(m['regime']!='familiar' for _,_,m in raw)==64
    assert set_digest(m['source_sha256'] for _,_,m in raw)==SUPPORT_SET_SHA
    for roll,orbit in TARGET:
        group=[]
        for f,controls,meta in raw:
            obs,mask,h=render(f,roll,orbit)
            group.append((obs,mask,h,homography(-roll,-orbit),controls,meta))
        yield pose_name(roll,orbit),roll or orbit,roll,group


def physical_stream(path):
    rows=[r for r in read(path)['rows'] if r['arm']=='UR5e' and r['role']=='heldout']
    assert rows and len({r['index'] for r in rows})==len(rows)
    names=set.intersection(*(set(r['views']) for r in rows))
    for name in sorted(names,key=lambda n:(n!='nominal',n)):
        group=[]
        for row in rows:
            v=row['views'][name]
            other=name if name=='nominal' else name[:-1]+{'-':'+','+':'-'}[name[-1]]
            opp=row['views'][other]
            tp=Path(v['passive_npz'])
            assert sha(tp)==v['sha256']
            with np.load(tp) as z, np.load(row['views']['nominal']['passive_npz']) as nom:
                controls={k:z[k] for k in SENSOR}
                for k in SENSOR:
                    np.testing.assert_array_equal(controls[k],nom[k])
                assert controls['valid'].all() and list(z['chunk_steps'])==[5]*3
                np.testing.assert_array_equal(z['img_before'][0],z['frame0_img'])
                np.testing.assert_array_equal(z['img_after'][:-1],z['img_before'][1:])
                obs=np.concatenate([z['frame0_img'][None],z['img_after']])
            spec=get_embodiment_spec(row['native_control_id'])
            assert (spec.yaw_deg,spec.xy_scale,spec.gripper_sign)==(row['control_yaw_deg'],row['xy_scale'],row['gripper_sign'])
            h=np.asarray(v['H_ref_to_view'])
            inv=np.asarray(v['H_view_to_ref'])
            ident=h@inv
            np.testing.assert_allclose(ident/ident[2,2],np.eye(3),rtol=0,atol=1e-8)
            meta=dict(index=row['index'],suite=row['suite'],task_id=row['task_id'],
                      task=f"{row['suite']}/{row['task_id']}",embodiment=spec.name,regime=spec.regime,
                      true_yaw_deg=spec.yaw_deg,true_gain=spec.xy_scale,true_sign=spec.gripper_sign,
                      source_sha256=row['archive_sha256'],native_view_sha256=v['sha256'],
                      original_orbit=row['original_orbit'],bank=str(path),physical_id=row['id'])
            group.append((obs,FULL,h,np.asarray(opp['H_ref_to_view']),controls,meta))
        angle=v['angle_deg']
        yield name,angle,angle if name.startswith('roll') else 0,group


def load_models(ck):
    seal=read(ck/'sealed.json')
    models={}
    steps=seal['steps']
    lst=(REPO/'outputs/symmeta-perceiver-retry-20261123-r1/evidence/train02pro/checkpoints.sha256').read_text().splitlines()
    hist={Path(x.split()[1]).name:x.split()[0] for x in lst}
    for seed in (41,42,43):
        for arm in ('A','B','shipped'):
            if arm=='shipped':
                path=ROOT/f'symmeta-perceiver-retry-20261123-r1/train02pro/checkpoints/perceiver_cont_warm-passive_attempt1-seed{seed}.msgpack'
                expected=hist[path.name]
                assert sha(path)==expected
                p,c,m=load_v2_checkpoint(path)
                assert c==CFG and m['seed']==seed
            else:
                path=ck/f'{arm}-seed{seed}-step{steps}.msgpack'
                expected=seal['checkpoint_sha256'][path.name]
                assert sha(path)==expected
                p=restore(path,seed)
            models[arm,seed]=(p,str(path),expected)
    return models


@jax.jit
def shipped(p,b):
    return apply_reader_v2(p,CFG,b)


def evaluate(panel,ck,out,bank):
    cv2.setNumThreads(1)
    models=load_models(ck)
    pred=[]
    stream=digital_stream() if panel=='digital' else physical_stream(bank)
    for view,angle,correction,group in stream:
        for cal in ('true','opposite','identity_token'):
            if cal=='opposite' and view=='nominal':
                continue
            if cal=='identity_token' and (panel!='digital' or view not in ('roll+90','roll-90')):
                continue
            for arm in ('A','B','shipped'):
                if (cal=='identity_token' and arm!='A') or (arm=='shipped' and cal!='true'):
                    continue
                data,metadata=[],[]
                for obs,mask,correct,opp,controls,meta in group:
                    h=opp if cal=='opposite' else correct
                    visual=view_inputs(obs,mask,h,'B' if arm=='shipped' else arm)
                    coverage=float(visual['pixel_valid'].mean())
                    if cal=='identity_token':
                        visual['calibration']=visual['grid_h']=encoding(np.eye(3))
                    if arm=='shipped':
                        # Shipped code has no spatial masks. Recompute native DIS on actual neutral-filled observed bytes.
                        from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
                        visual['dis_flow']=compute_dis_flow_maps(visual['img_before'],visual['img_after'],controls['valid'])
                        for k in ('pixel_valid','calibration','grid_h'):
                            visual.pop(k)
                    data.append(dict(**visual,**controls))
                    metadata.append(dict(**meta,observed_sha256=digest(obs),calibration_sha256=digest(h),
                                         coverage=coverage))
                b={k:np.stack([d[k] for d in data]) for k in data[0]}
                for seed in (41,42,43):
                    p,path,hs=models[arm,seed]
                    yaw,gain,sign=decode((shipped if arm=='shipped' else infer)(p,b))
                    for m,y,g,s in zip(metadata,yaw,gain,sign,strict=True):
                        row=dict(**m,panel=panel,view=view,camera_deg=angle,model=arm,calibration=cal,
                                 seed=seed,yaw_deg=float(y),xy_scale=float(g),gripper_sign=int(s),
                                 checkpoint_path=path,checkpoint_sha256=hs)
                        pred.append(row)
                        if arm=='shipped' and (correction or view=='nominal'):
                            pred.append(dict(row,model='shipped_corrected',yaw_deg=float((y+correction+180)%360-180)))
        print(panel,view,len(group),'supports',flush=True)
    path=out/f'{panel}-predictions.json.gz'
    with gzip.open(path,'wt') as f:
        json.dump(pred,f,separators=(',',':'),allow_nan=False)
    from summarize import summarize
    write(out/f'{panel}-summary.json',summarize(pred))
    write(out/f'{panel}-receipt.json',dict(rows=len(pred),path=str(path),sha256=sha(path),seal=sha(ck/'sealed.json'),
           bank=None if bank is None else str(bank),bank_sha=None if bank is None else sha(bank),
           git_pin=(out/'launch.txt').read_text().split()[0]))
    return pred


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('panel',choices=('digital','physical'))
    ap.add_argument('ck',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('bank',nargs='?',type=Path)
    a=ap.parse_args()
    evaluate(a.panel,a.ck,a.out,a.bank)
