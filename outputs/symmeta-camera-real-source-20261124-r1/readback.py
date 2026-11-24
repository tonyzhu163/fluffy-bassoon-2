"""Summaries from stored CPU-bank records and sealed prediction files. No inference."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f: return json.load(f)


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def describe(values):
    x=np.asarray(values,dtype=float)
    return dict(n=int(x.size),min=float(x.min()),median=float(np.median(x)),mean=float(x.mean()),max=float(x.max()))


def geometry(folder,out):
    geom=read(folder/'geometry.json.gz')
    drift=read(folder/'replay.json.gz')
    # Compare true RGB with its same-tape plane projection on jointly retained
    # pixels. The old digital reader erodes the footprint once before DIS.
    if (folder/'train-digital_B-img_after.npy').exists():
        true={k:np.load(folder/f'train-B-{k}.npy',mmap_mode='r') for k in ('img_before','img_after')}
        proj={k:np.load(folder/f'train-digital_B-{k}.npy',mmap_mode='r') for k in ('img_before','img_after','pixel_valid')}
        ids=[r['id'] for r in read(Path(__file__).with_name('source-ids.json'))['rows'] if r['split']=='train']
        for r in (r for r in geom if r['split']=='train'):
            i=ids.index(r['id']); j=(0,-12,12,-24,24,-42,42,-49,49).index(r['angle'])
            mask=proj['pixel_valid'][i,j,0,0].astype(bool)
            def frames(x): return np.concatenate([x['img_before'][i,j,0,:1],x['img_after'][i,j,0]])
            r['nominal_projection_overlap']=r['modes']['full']['digital_observed_valid']
            r['newly_visible_raw_pixel_fraction']=1-r['nominal_projection_overlap']
            r['real_minus_projected_rgb_mae_common']=float(np.abs(frames(true)[:,mask].astype(np.int16)-frames(proj)[:,mask].astype(np.int16)).mean())
    for r in geom:
        if r['split']=='train':
            r['nominal_projection_overlap']=r['modes']['full']['digital_observed_valid']
            r['newly_visible_raw_pixel_fraction']=1-r['nominal_projection_overlap']
    split={}
    for s in ('train','val'):
        rows=[r for r in geom if r['split']==s]
        moved=[r for r in rows if r['angle']]
        masks={str(a):sorted({v['observed_valid'] for r in moved if r['angle']==a for k,v in r['modes'].items()}) for a in sorted({r['angle'] for r in moved})}
        assert len({tuple(l) for l in masks.values()})==1
        stats={}
        for a in sorted({r['angle'] for r in rows}):
            group=[r for r in rows if r['angle']==a]
            stats[str(a)]=dict(n=len(group),unique_observed=len({r['observed_sha256'] for r in group}),
                center_shift_px=describe([np.linalg.norm(r['center_shift_px']) for r in group]),
                center_x_px=describe([r['center_shift_px'][0] for r in group]),
                center_y_px=describe([r['center_shift_px'][1] for r in group]),
                camera_displacement_m=describe([r['camera_center_m'] for r in group]),
                full_raw_B=describe([r['modes']['full']['B_coverage'] for r in group]),
                full_registered_A=describe([r['modes']['full']['A_coverage'] for r in group]),
                eef_post_registered_px=describe([x for r in group for x in r['eef_registered_px'][1:]]),
                eef_post_unregistered_px=describe([x for r in group for x in r['eef_unregistered_px'][1:]]))
            if all('nominal_projection_overlap' in r for r in group):
                for key in ('nominal_projection_overlap','real_minus_projected_rgb_mae_common','newly_visible_raw_pixel_fraction'):
                    if all(key in r for r in group): stats[str(a)][key]=describe([r[key] for r in group])
        split[s]=dict(tapes=len({r['id'] for r in rows}),tasks=sorted({r['task'] for r in rows}),
            per_body=Counter(r['arm'] for r in rows if not r['angle']),poses=len(stats),angle_summary=stats,
            distinct_observed_sequences=len({r['observed_sha256'] for r in rows}),
            genuine_observed_valid_fractions_by_angle=masks)
    result=dict(splits=split,replays=len(drift),max_mark_error_px=describe([r['max_marker_px'] for r in drift]),
            max_joint_delta_error_rad=describe([r['max_joint_delta_error_rad'] for r in drift]),
            final_nominal_rgb_mae=describe([r['final_nominal_rgb_mae'] for r in drift]),
            archive_vs_new_reset_fingerprint_matches=sum(r['archive_physics']==r['replay_physics'] for r in drift))
    dump(out/'geometry-summary.json',result)


def error(r):
    return abs((r['yaw_deg']-r['true_yaw_deg']+180)%360-180)


def clustered(groups):
    tasks=sorted({t for t,_ in groups})
    seeds=sorted({s for _,s in groups})
    a=np.array([[np.mean(groups[t,s]) for s in seeds] for t in tasks])
    means=a.mean(axis=1)
    draws=np.random.default_rng(1124).integers(len(tasks),size=(20000,len(tasks)))
    return dict(mean=float(means.mean()),ci95=np.quantile(means[draws].mean(1),[.025,.975]).tolist(),
            per_task=dict(zip(tasks,means.tolist())),per_seed=dict(zip(map(str,seeds),a.mean(axis=0).tolist())))


def fit_report(folder,out):
    seal=read(folder/'sealed.json')
    source=read(folder/'source-controls.json.gz')
    rows={(r['id'],r['seed'],r['angle'],r['model'],r['mode']):r for r in source}
    assert len(rows)==len(source)==64*5*4*7*3
    table=defaultdict(lambda:defaultdict(list))
    for r in source:
        if r['model']!='B' or r['mode']!='full': continue
        key=r['task'],r['seed']
        def get(m,t): return rows[r['id'],r['seed'],r['angle'],m,t]
        for group in (('nominal',) if not r['angle'] else ('orbits',r['body'],str(r['angle']))):
            def put(k,v): table[group,k][key].append(v)
            errs={}
            for label in seal['arms']:
                full=get(label,'full')
                e=error(full);errs[label]=e
                put(label,e)
                put(label+'_gain_abs_log',abs(np.log(full['xy_scale']/full['true_gain'])))
                put(label+'_sign_accuracy',int(full['gripper_sign']==full['true_sign']))
                for mode in ('opposite','identity'): put(label+'_'+mode+'_minus_full',error(get(label,mode))-e)
                put(label+'_edge_minus_full',np.mean([error(get(label,m)) for m in ('left','right','top','bottom')])-e)
            rg=errs['A']-errs['B'];dg=errs['digital_A']-errs['digital_B']
            put('real_gap',rg);put('digital_gap',dg);put('gap_interaction',rg-dg)
            for m in 'AB': put(m+'_real_minus_digital',errs[m]-errs['digital_'+m])
    summary={}
    for (g,k),v in table.items(): summary.setdefault(g,{})[k]=clustered(v)
    dump(out/'source-summary.json',summary)
    exposures={}
    for n in (2500,5000):
        by_seed={}
        for seed in (41,42,43):
            with np.load(folder/f'draws-{seed}-{n}.npz') as z:
                x=z['draws'].reshape(-1,5); counts=z['real_counts']
            assert counts.sum()==len(x)==n*16
            u,c=np.unique(x[:,4],return_counts=True)
            assert c[u==0].item()==len(x)//2
            nonzero=counts[counts>0]
            by_seed[seed]=dict(real_or_replacement=len(x),digital_bank=len(x),real_fraction_in_real_arms=.5,
                genuine_orbits=int((x[:,3]!=0).sum()),all_valid=int(c[u==0].item()),edge_masks=dict(zip(map(str,u),c.tolist())),
                unique_physical_tapes=int(len(np.unique(x[:,2]))),unique_digital_tapes=int(len(np.unique(x[:,0]))),
                digital_pose_counts=np.bincount(x[:,1],minlength=23).tolist(),genuine_pose_counts=np.bincount(x[:,3],minlength=9).tolist(),
                unique_physical_tape_pose_masks=int(len(nonzero)),support_size=int(counts.size),per_cell_draws=describe(counts))
        exposures[n]=by_seed
    dump(out/'draw-summary.json',exposures)
    for body in ('UR5e','Kinova3'):
        p=folder/f'{body}-predictions.json.gz'
        if not p.exists(): continue
        main=read(p);cross=read(folder/f'{body}-cross-input.json.gz')
        recip=read(folder/f'{body}-receipt.json')
        for file,digest in [(p,recip['sha256']),(folder/f'{body}-cross-input.json.gz',recip['cross_sha256'])]:
            assert hashlib.sha256(file.read_bytes()).hexdigest()==digest
        normal={(r['index'],r['view'],r['seed'],r['model']):r for r in main if r['selected'] and r['calibration']=='true'}
        by=defaultdict(lambda:defaultdict(list))
        for r in cross:
            if not r['camera_deg']: continue
            ref=normal[r['index'],r['view'],r['seed'],r['model']]
            assert r['observed_sha256']==ref['observed_sha256'] and r['checkpoint_sha256']==ref['checkpoint_sha256']
            for g in (('all','assigned') if r['assigned'] else ('all',)):
                by[g,r['model']][r['task'],r['seed']].append(error(r)-error(ref))
        dump(out/f'{body}-cross-summary.json',{g:{m:clustered(by[g,m]) for m in seal['arms']} for g in ('all','assigned')})
        native=[r for r in main if r['assigned'] and r['selected'] and r['calibration']=='true' and r['seed']==41 and r['model'] in ('A','B')]
        assert len(native)==128
        with gzip.open(out/f'{body}-native-seed41.json.gz','wt') as f: json.dump(native,f,separators=(',',':'),allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('geometry','fit'))
    p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    dest=args.out or args.directory
    dest.mkdir(parents=True,exist_ok=True)
    (geometry if args.mode=='geometry' else fit_report)(args.directory,dest)
