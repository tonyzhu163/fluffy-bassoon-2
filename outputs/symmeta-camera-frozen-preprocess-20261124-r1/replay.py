"""Inference-only replay using the locked camera reader observation streams."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import resource
import time

import cv2
import jax
import numpy as np
from camera import view_inputs
from data import REPO, ROOT, SENSOR, read, sha, write
from evaluate import CFG, digest, digital_stream, physical_stream
from eval_stage2a_live_ur5e import load_v2_checkpoint
from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.equiv_v2 import predict_embodiment_v2

HERE = Path(__file__).resolve().parent
SEEDS = {41, 42, 43}
HEADS = ('yaw_deg', 'xy_scale', 'gripper_sign')
ORIGINAL = {8, 9, 10, 11, 80, 81, 82, 83}
BATCH = 12   # Historical CPU reference ran batches of 12
RETRY = 'symmeta-perceiver-retry-20261123-r1'
HIST = REPO/'outputs/symmeta-perceiver-mechanism-20261124-r1/evidence'
VIS = ('img_before', 'img_after', 'dis_flow')


def load_readers():
    readers, origins = {}, {}
    for line in (REPO/'outputs'/RETRY/'evidence/train02pro/checkpoints.sha256').read_text().splitlines():
        checksum, name = line.split()
        path = ROOT/RETRY/'train02pro'/name
        assert sha(path) == checksum
        params, cfg, meta = load_v2_checkpoint(path)
        seed = meta['seed']
        assert cfg == CFG and meta['tag'] == 'perceiver_cont_warm'
        assert meta['mode'] == 'passive_attempt1' and meta['steps'] == 2500
        readers[seed] = (params, cfg)
        origins[seed] = dict(checkpoint_path=str(path), checkpoint_sha256=checksum)
    assert set(readers) == SEEDS
    return readers, origins


def prepared(obs, mask, supplied_h, controls, arm):
    """No evaluator metadata or reference pixels can enter this function."""
    assert set(controls) == set(SENSOR) and controls['valid'].all()
    view = view_inputs(obs, mask, supplied_h, arm)
    b = {k: view[k] for k in VIS} | controls
    native_flow = compute_dis_flow_maps(b['img_before'], b['img_after'], b['valid'])
    common = dict(before_sha256=digest(b['img_before']), after_sha256=digest(b['img_after']),
                  pixel_valid_sha256=digest(view['pixel_valid']), coverage=float(view['pixel_valid'].mean()))
    return b | {'dis_flow': native_flow}, b, common


def run_view(group, panel, view, angle, correction, readers, origins):
    batches, all_meta = {}, {}
    for obs, mask, correct, opposite, controls, meta in group:
        provenance = dict(panel=panel, view=view, camera_deg=angle, **meta,
                          observed_sha256=digest(obs), observed_mask_sha256=digest(mask),
                          calibration_sha256=digest(correct), H_ref_to_view=correct.tolist(),
                          sensor_sha256={k:digest(controls[k]) for k in SENSOR})
        for model, h, frame in [('shipped', correct, 'B'), ('frozen_warp', correct, 'A'),
                                ('frozen_warp_opposite', opposite, 'A')]:
            if model == 'frozen_warp_opposite' and view == 'nominal':
                continue
            native, masked, description = prepared(obs, mask, h, controls, frame)
            variants = [(model, native)]
            if model == 'frozen_warp':
                variants.append(('frozen_warp_Aflow', masked))
            for name, inputs in variants:
                assert set(inputs) == set(SENSOR) | set(VIS)
                batches.setdefault(name, []).append(inputs)
                record = dict(provenance, **description, dis_sha256=digest(inputs['dis_flow']),
                              calibration_sha256=digest(h), supplied_H=h.tolist(),
                              model=name, calibration='opposite' if name.endswith('_opposite') else 'true')
                all_meta.setdefault(name, []).append(record)
    results = []
    for model, observations in batches.items():
        for i in range(0, len(observations), BATCH):
            chunk = observations[i:i+BATCH]
            padded = chunk + [chunk[-1]]*(BATCH-len(chunk))
            batch = {k:np.stack([r[k] for r in padded]) for k in padded[0]}
            for seed, reader in readers.items():
                yaw, gain, sign = predict_embodiment_v2(*reader, batch)
                for j in range(len(chunk)):
                    row = dict(all_meta[model][i+j], seed=seed, **origins[seed], yaw_deg=float(yaw[j]),
                               xy_scale=float(gain[j]), gripper_sign=int(sign[j]))
                    results.append(row)
                    if model == 'shipped' and (view == 'nominal' or correction):
                        results.append(dict(row, model='shipped_corrected',
                                            yaw_deg=(row['yaw_deg']+correction+180)%360-180))
    return results


def audit(rows, panel):
    by = {(r['index'], r['seed'], r['model'], r['view']):r for r in rows}
    assert len(by) == len(rows)
    cpu = {(r['index'],r['seed']):r for r in json.load(gzip.open(HIST/'predictions.json.gz'))
           if r['condition'] == 'original'}
    gpu_path = ROOT/RETRY/'train02pro/predictions.json'
    assert sha(gpu_path) == 'c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b'
    refs = {(r['index'],r['seed']):r for r in read(gpu_path)['rows'] if r['panel']=='gpu09test'}
    assert len(refs) == len(cpu) == 288
    residuals = []
    for r in rows:
        i, s = r['index'], r['seed']
        prev = cpu[i,s]
        assert all(r[k] == prev[k] for k in ('regime','task_id','suite','embodiment'))
        if r['model'] == 'shipped' and r['view'] == 'nominal' and panel == 'digital':
            old = refs[i,s]
            assert r['source_sha256'] == prev['tape_sha256'] == old['tape_sha256']
            assert old['tuned'] == prev['archived']
            residuals.append(dict(index=i,seed=s,**{f'{src}_{k}':float(r[k]-ref[k])
                for src,ref in [('cpu',prev),('gpu',old['tuned'])] for k in HEADS}))
        virtual_quarter = panel == 'digital' and r['view'] in ('roll+90','roll-90')
        if r['model'] in ('frozen_warp','frozen_warp_Aflow') and (r['view']=='nominal' or virtual_quarter):
            nom = by[i,s,'shipped','nominal']
            assert all(r[k] == nom[k] for k in (*HEADS,'before_sha256','after_sha256','dis_sha256','sensor_sha256'))
    if residuals:
        # Earlier CPU mechanism replay used identical backend, arithmetic and execution batch.
        assert all(row[f'cpu_{k}'] == 0 for row in residuals for k in HEADS), residuals[:3]
    return residuals


def run(args, readers, origins, panel):
    stream = digital_stream() if panel=='digital' else physical_stream(args.bank)
    rows = []
    start = time.monotonic()
    counts = {}
    for view, angle, corr, group in stream:
        group = [entry for entry in group if entry[-1]['regime'] != 'familiar'
                 and (args.mode!='proof' or entry[-1]['index'] in ORIGINAL)]
        ids = {entry[-1]['index'] for entry in group}
        expected = ORIGINAL if args.mode=='proof' else {12*j+k for j in range(8) for k in range(4,12)}
        assert ids == expected and len(group) == len(ids), (view,sorted(ids),sorted(expected))
        rows.extend(run_view(group,panel,view,angle,corr,readers,origins))
        counts[view] = len(ids)
        print(panel, view, len(ids), len(rows), round(time.monotonic()-start,2), flush=True)
    if panel=='physical':
        assert {'orbit+','orbit-','nominal'} <= counts.keys()
        assert all(abs(r['camera_deg'])==30 for r in rows if r['view'] in ('orbit+','orbit-'))
    else:
        assert len(counts)==9
    residuals = audit(rows, panel)
    path = args.out/f'{panel}-predictions.json.gz'
    with gzip.open(path,'wt',compresslevel=4) as stream:
        json.dump(rows, stream, separators=(',',':'),allow_nan=False)
    write(args.out/f'{panel}-checks.json',dict(rows=len(rows),views=counts,
        exact_nominal_and_digital_inverse_checked=True, nominal_cpu_gpu_residuals=residuals,
        sha256=sha(path),physical_bank=str(args.bank) if panel=='physical' else None,
        bank_sha256=sha(args.bank) if panel=='physical' else None, checkpoints=origins,
        elapsed_wall_s=time.monotonic()-start, model_input_fields=sorted(set(VIS)|set(SENSOR)),
        regime_counts_for_nominal_seed41=dict(Counter(r['regime'] for r in rows if
            r['model']=='shipped' and r['view']=='nominal' and r['seed']==41))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('proof','digital','physical'))
    parser.add_argument('out',type=Path)
    parser.add_argument('bank',type=Path,nargs='?')
    args = parser.parse_args()
    assert sorted(os.sched_getaffinity(0))==[50,51] and jax.default_backend()=='cpu'
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    readers,origins = load_readers()
    for panel in (('digital','physical') if args.mode=='proof' else (args.mode,)):
        run(args,readers,origins,panel)
    print('Runtime',jax.__version__,cv2.__version__,np.__version__,resource.getrusage(resource.RUSAGE_SELF))
