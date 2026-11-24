import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import eval_wider_family as pr7
from eval_stage3_scaling import build_collection_plan
from camera import TRAIN, VALIDATION, digital, frames_of

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
SENSOR = ('issued_mean','delta_eef','delta_jaw','valid')
RAW = ('img_before','img_after',*SENSOR,*pr7.LABEL_KEYS)
VIS = ('img_before','img_after','dis_flow','pixel_valid','calibration','grid_h')
CAL = ROOT / 'symmeta-omitted-chunk-20261124-r1/data01'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def source(out):
    lock = read(REPO/'outputs/symmeta-perceiver-retry-20261123-r1/export-inputs.json')['sha256']
    plan = build_collection_plan()
    pieces, records, paths = [], [], {}
    for s in range(8):
        rel = f'xemb-icml-campaign-20261121-r1/gpu12scale/stage3/diverse_shards/shard-{s}/diverse.npz'
        path = ROOT / rel
        assert sha(path) == lock[rel]
        rows = read(path.with_name('meta.json'))
        assert rows == plan[s::8]
        with np.load(path) as z:
            assert z['valid'][:,:3].all() and not z['valid'][:,3:].any()
            part = {k: np.ascontiguousarray(z[k][:,:3]) if k not in pr7.LABEL_KEYS else z[k] for k in RAW}
            assert all(np.array_equal(part[k],np.asarray([r[k] for r in rows],dtype=part[k].dtype)) for k in pr7.LABEL_KEYS)
        pieces.append(part)
        paths[rel] = lock[rel]
        records += rows
    train = {k:np.concatenate([p[k] for p in pieces]) for k in RAW}
    assert len(records)==1466 and {r['arm'] for r in records}=={'Panda','Sawyer'}
    # Separate scale-archive source recordings reserved for validation. Check exact images against fit.
    seal = read(CAL/'source-seal.json')['sha256']
    for name in ('reader-cal.npz','labels-cal.npz'):
        assert sha(CAL/name)==seal[name]
        paths[str(CAL/name)] = seal[name]
    with np.load(CAL/'reader-cal.npz') as z, np.load(CAL/'labels-cal.npz') as lab:
        assert np.array_equal(z['case_id'],lab['case_id']) and set(z['arm'])=={'Panda','Sawyer'}
        val = {k:z[k] for k in RAW if k not in pr7.LABEL_KEYS}
        val.update({k:lab[name] for k,name in zip(pr7.LABEL_KEYS,('true_yaw','true_gain','true_sign'),strict=True)})
        val_id, val_task = z['case_id'], z['task']
    def signature(b, i):
        return hashlib.sha256(frames_of(b,i).tobytes()).digest()
    train_ids = {signature(train,i) for i in range(len(records))}
    keep = [i for i in range(len(val_id)) if signature(val,i) not in train_ids]
    if not keep:
        raise ValueError('Source validation repeats all training tapes')
    val = {k:v[keep] for k,v in val.items()}
    original = {k:val[k] for k in pr7.LABEL_KEYS}
    val = pr7.relabel(val, pr7.sample_mixed(np.random.default_rng(112417),('family',),len(keep)))
    write(out/'source.json',dict(inputs=paths, training=records, validation=list(val_id[keep]),
                                 val_task=list(val_task[keep]), excluded=len(val_id)-len(keep),
                                 native_val_labels={k:v.tolist() for k,v in original.items()}))
    return train,val


def make_bank(out, name, data, poses):
    n = len(data['valid'])
    np.savez(out/f'{name}-control.npz', **{k:data[k] for k in (*SENSOR,*pr7.LABEL_KEYS)})
    banks = {}
    for arm in ('A','B'):
        ex = digital(data,0,poses[0],arm)
        banks[arm] = {k:np.lib.format.open_memmap(out/f'{name}-{arm}-{k}.npy', mode='w+',
                      dtype=a.dtype, shape=(len(poses),n,*a.shape)) for k,a in ex.items()}
    for j,pose in enumerate(poses):
        from camera import render,view_inputs
        for i in range(n):
            obs,mask,h = render(frames_of(data,i), *pose)
            for arm in ('A','B'):
                v = view_inputs(obs,mask,h,arm)
                for k,a in v.items():
                    banks[arm][k][j,i] = a
        for bank in banks.values():
            for arr in bank.values():
                arr.flush()
        print(name, pose, 'cached', flush=True)


def load_bank(out,name,arm,ram):
    return {k:np.load(out/f'{name}-{arm}-{k}.npy',mmap_mode=None if ram else 'r') for k in VIS}


def gather(bank, pose, idx):
    return {k:v[pose,idx] for k,v in bank.items()}


def prepare(out):
    out.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(1)
    tr,val = source(out)
    make_bank(out,'train',tr,TRAIN)
    make_bank(out,'val',val,VALIDATION)
    write(out/'ready.json',dict(source=sha(out/'source.json'),train_poses=TRAIN,val_poses=VALIDATION))
