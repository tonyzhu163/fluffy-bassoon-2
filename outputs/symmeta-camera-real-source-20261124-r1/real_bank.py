"""CPU preprocessing of real observations; both arms use the same input bytes."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np

from selection import HERE
from camera import FULL, frames_of, view_inputs, warp
from data import SENSOR, VIS, read, sha, write

ANGLES = dict(train=(0,-12,12,-24,24,-42,42,-49,49),val=(0,-18,18,-45,45))
MODES = ('full','left','right','top','bottom','opposite','identity')
MASKS = [FULL.copy() for _ in MODES]
MASKS[1][:,:8]=0
MASKS[2][:,-8:]=0
MASKS[3][:8,:]=0
MASKS[4][-8:,:]=0
SOURCE = '68955add4586f5b50d64296de92cf188adc55bd85800e95895dc6e48c1f1d4e0'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def name(angle):
    return f'orbit{angle:+d}' if angle else 'nominal'


def cases(split):
    return [r for r in read(HERE/'source-ids.json')['rows'] if r['split']==split]


def arrays(folder,split,arm,mode):
    return {key:np.load(folder/f'{split}-{arm}-{key}.npy',mmap_mode=mode) for key in VIS}


def allocate(folder):
    example=view_inputs(np.zeros((4,64,64,3),np.uint8), FULL, np.eye(3),'A')
    for split,angles in ANGLES.items():
        shape=len(cases(split)),len(angles),5 if split=='train' else 7
        for arm in (('A','B','digital_A','digital_B') if split=='train' else ('A','B')):
            for key,values in example.items():
                np.lib.format.open_memmap(folder/f'{split}-{arm}-{key}.npy',mode='w+',
                        dtype=values.dtype,shape=(*shape,*values.shape))


def build(rendered,folder,slot):
    for split,angles in ANGLES.items():
        arms=('A','B','digital_A','digital_B') if split=='train' else ('A','B')
        inputs={arm:arrays(folder,split,arm,'r+') for arm in arms}
        rows=cases(split)
        for i in range(slot,len(rows),4):
            spec=rows[i]
            parent=rendered/spec['id']
            row=read(parent/'row.json')
            assert row['id']==spec['id'] and row['split']==split
            assert all(row[k]==spec[k] for k in ('suite','task_id','arm','init_state_id','seed'))
            assert row['archive_sha256']==sha(Path(row['archive_npz']))
            assert set(row['views'])=={name(a) for a in angles}
            nv=row['views']['nominal']
            assert sha(Path(nv['passive_npz']))==nv['sha256']
            with np.load(nv['passive_npz']) as z:
                assert z['valid'].shape==(3,) and z['valid'].all() and np.all(z['chunk_steps']==5)
                np.testing.assert_array_equal(z['frame0_img'],z['img_before'][0])
                nominal=frames_of({k:z[k][None] for k in ('img_before','img_after')},0)
                sensor={k:z[k] for k in SENSOR}
            np.savez(folder/f"{row['id']}-control.npz",**sensor,yaw_deg=np.float32(row['control_yaw_deg']),
                     xy_scale=np.float32(row['xy_scale']),gripper_sign=np.int32(row['gripper_sign']))
            records=[]
            for j,a in enumerate(angles):
                v=row['views'][name(a)]
                h=np.asarray(v['H_ref_to_view'])
                opposite=np.asarray(row['views'][name(-a)]['H_ref_to_view'])
                frames=nominal
                if a:
                    path=Path(v['frames_npz'])
                    assert sha(path)==v['sha256']
                    with np.load(path) as z:
                        frames=z['frames']
                center=h@np.array([31.5,31.5,1.])
                offset=center[:2]/center[2]-31.5
                rec=dict(id=row['id'],arm=spec['arm'],task=f"{row['suite']}/{row['task_id']}",split=split,
                         angle=a,center_shift_px=offset.tolist(),H=h.tolist(),raw_valid=1.,
                         camera_center_m=float(np.linalg.norm(np.subtract(v['center_m'],nv['center_m']))),
                         eef_registered_px=v['eef_registered_residual_px'],eef_unregistered_px=v['eef_unregistered_residual_px'],
                         observed_sha256=digest(frames),source_sha256=row['archive_sha256'],modes={})
                if split=='train':
                    synth=np.stack([warp(f,h) for f in nominal]) if a else nominal.copy()
                    footprint=(warp(FULL.astype(np.float32),h,0)>=1).astype(np.uint8) if a else FULL.copy()
                for m,mode in enumerate(MODES[:5] if split=='train' else MODES):
                    mask=MASKS[m]
                    obs=frames.copy()
                    obs[:,mask==0]=127
                    supplied=opposite if mode=='opposite' else np.eye(3) if mode=='identity' else h
                    rec['modes'][mode]=dict(observed_sha256=digest(obs),supplied_sha256=digest(supplied),
                                            observed_valid=float(mask.mean()))
                    if split=='train':
                        synthetic_mask=mask & footprint
                        synthetic=synth.copy()
                        synthetic[:,synthetic_mask==0]=127
                        rec['modes'][mode].update(digital_observed_sha256=digest(synthetic),
                                                   digital_observed_valid=float(synthetic_mask.mean()))
                    for arm in arms:
                        digital=arm.startswith('digital_')
                        vis=view_inputs(synthetic if digital else obs, synthetic_mask if digital else mask,
                                        supplied,arm[-1])
                        rec['modes'][mode][arm+'_coverage']=float(vis['pixel_valid'].mean())
                        for k in VIS:
                            inputs[arm][k][i,j,m]=vis[k]
                records.append(rec)
            write(folder/f"{row['id']}-geometry.json",records)
        for bank in inputs.values():
            for v in bank.values(): v.flush()


def seal(rendered,folder,digital):
    assert sha(digital/'source.json')==SOURCE
    source=read(digital/'source.json')['training']
    training={(x['suite'],x['task_id']) for x in cases('train')}
    validation={(x['suite'],x['task_id']) for x in cases('val')}
    assert len(training)==16 and len(validation)==8 and not training & validation
    digital_ids=[i for i,r in enumerate(source) if (r['suite'],r['task_id']) in training]
    write(folder/'digital-ids.json',[dict(bank_index=i,**source[i]) for i in digital_ids])
    geo=[]
    drift=[]
    for split,angles in ANGLES.items():
        controls=[]
        for c in cases(split):
            folder_id=rendered/c['id']
            d=read(folder_id/'diag.json')
            row=read(folder_id/'row.json')
            drift.append(dict(id=c['id'],archive_sha256=row['archive_sha256'],archive_physics=c['physics'],
                              replay_physics=row['physics'],max_joint_delta_error_rad=d['archive_joint_delta_max_error_rad'],
                              final_nominal_rgb_mae=d['archive_final_nominal_rgb_mae'],
                              max_marker_px=max(max(m['errors_px']) for m in d['landmarks'])))
            with np.load(folder/f"{c['id']}-control.npz") as z: controls.append(dict(z))
            sub=read(folder/f"{c['id']}-geometry.json")
            assert [r['angle'] for r in sub]==list(angles)
            geo.extend(sub)
        np.savez(folder/f'{split}-controls.npz',**{k:np.stack([r[k] for r in controls]) for k in controls[0]})
    by_split={s:{r['source_sha256'] for r in geo if r['split']==s} for s in ANGLES}
    assert len(by_split['train'])==128 and len(by_split['val'])==64 and not (by_split['val'] & by_split['train'])
    # Reject identical complete four-frame tapes across train and validation, both real and digital.
    tr={r['observed_sha256'] for r in geo if r['split']=='train'}
    val={r['observed_sha256'] for r in geo if r['split']=='val'}
    assert not (tr & val)
    digital_hashes=set()
    nom={k:np.load(digital/f'train-B-{k}.npy',mmap_mode='r')[0] for k in ('img_before','img_after')}
    for i in digital_ids: digital_hashes.add(digest(frames_of(nom,i)))
    assert not (digital_hashes & val)
    import gzip,json
    for filename,records in [('geometry',geo),('replay',drift)]:
        with gzip.open(folder/f'{filename}.json.gz','wt') as f: json.dump(records,f,separators=(',',':'),allow_nan=False)
    assert any(np.linalg.norm(r['center_shift_px'])>1 for r in geo if r['angle'])
    hashes={p.name:sha(p) for p in sorted(folder.iterdir()) if p.suffix=='.npy' or p.name in
            ('train-controls.npz','val-controls.npz','digital-ids.json','geometry.json.gz','replay.json.gz')}
    write(folder/'ready.json',dict(source_ids_sha256=sha(HERE/'source-ids.json'),digital_source=SOURCE,
            digital_count=len(digital_ids),training_tapes=128,validation_tapes=64,sha256=hashes,
            renderer_pin=(rendered/'launch.txt').read_text().split()[0]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('allocate','build','seal'))
    p.add_argument('rendered',type=Path)
    p.add_argument('folder',type=Path)
    p.add_argument('--slot',type=int)
    p.add_argument('--digital',type=Path)
    a=p.parse_args()
    cv2.setNumThreads(1)
    if a.stage=='allocate': allocate(a.folder)
    elif a.stage=='build': build(a.rendered,a.folder,a.slot)
    else: seal(a.rendered,a.folder,a.digital)
