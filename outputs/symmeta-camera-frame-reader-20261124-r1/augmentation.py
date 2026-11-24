"""Third arm from the frozen B recipe with both public calibration inputs clamped."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from camera import encoding,view_inputs
from data import read,sha,write,SENSOR
from model import decode,infer,restore
import train as experiment
from evaluate import digital_stream,physical_stream,digest
from summarize import estimate,error
from collections import defaultdict

IDENTITY=encoding(np.eye(3))


def blind_gather(bank,pose,idx):
    data=experiment.gather(bank,pose,idx)
    for key in ('calibration','grid_h'):
        data[key]=np.broadcast_to(IDENTITY,data[key].shape).copy()
    return data




def fit(out,base,extra,primary):
    seal=read(primary/'sealed.json')
    # Reuse the pinned B arm schedule and save only the requested calibration-blind arm.
    experiment.train(out,base,extra,arms=('C',),get_view=blind_gather)
    raw=read(out/'sealed.json')
    assert raw['training_exposures']==seal['training_exposures'] and raw['params_per_arm']==seal['params_per_arm']
    raw.update(model='C',steps=seal['steps'],calibration='clamped_identity_global_and_local',
               source_C_mean=raw.pop('source_B_mean'),primary_seal_sha256=sha(primary/'sealed.json'))
    write(out/'sealed.json',raw)


def score(panel,out,ck,primary_pred,bank=None):
    seal=read(ck/'sealed.json')
    with gzip.open(primary_pred,'rt') as f:
        ref={(r['index'],r['seed'],r['view']):r for r in json.load(f) if r['model']=='B' and r['calibration']=='true'}
    fits={}
    for seed in (41,42,43):
        name=f"C-seed{seed}-step{seal['steps']}.msgpack"
        path=ck/name
        assert sha(path)==seal['checkpoint_sha256'][name]
        fits[seed]=(restore(path,seed),path)
    rows=[]
    stream=digital_stream() if panel=='digital' else physical_stream(bank)
    for view,angle,_,group in stream:
        data=[]
        for obs,mask,h,_,ctrl,m in group:
            v=view_inputs(obs,mask,np.eye(3),'B')
            data.append(dict(**v,**ctrl))
        b={k:np.stack([d[k] for d in data]) for k in data[0]}
        for seed,(p,path) in fits.items():
            ys,gs,ss=decode(infer(p,b))
            for (obs,mask,h,opp,ctrl,m),y,g,s in zip(group,ys,gs,ss,strict=True):
                r=ref[m['index'],seed,view]
                assert r['observed_sha256']==digest(obs) and r['source_sha256']==m['source_sha256']
                rows.append(dict(r,model='C',calibration='unsupplied',calibration_sha256=digest(np.eye(3)),
                           yaw_deg=float(y),xy_scale=float(g),gripper_sign=int(s),
                           checkpoint_path=str(path),checkpoint_sha256=seal['checkpoint_sha256'][path.name]))
    metrics={}
    for split in ('all','ood'):
        groups=defaultdict(lambda:defaultdict(list))
        for r in rows:
            if split=='ood' and r['regime']=='familiar':
                continue
            key=(r['task'],r['seed'])
            b=ref[r['index'],r['seed'],r['view']]
            for view in ([r['view'],'primary_nonquarter'] if r['view'] in ('roll-20','roll+20','roll-45','roll+45') else [r['view']]):
                groups[view,'C'][key].append(error(r))
                groups[view,'C_minus_B'][key].append(error(r)-error(b))
        metrics[split]={}
        for (v,k),x in groups.items():
            metrics[split].setdefault(v,{})[k]=estimate(x)
    path=out/f'{panel}-C-predictions.json.gz'
    with gzip.open(path,'wt') as f:
        json.dump(rows,f,separators=(',',':'),allow_nan=False)
    write(out/f'{panel}-C-summary.json',metrics)
    write(out/f'{panel}-C-receipt.json',dict(rows=len(rows),sha256=sha(path),B_reference=sha(primary_pred),seal=sha(ck/'sealed.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('fit','digital','physical'))
    ap.add_argument('paths',type=Path,nargs='+')
    args=ap.parse_args()
    if args.mode=='fit':
        fit(*args.paths)
    else:
        score(args.mode,*args.paths)
