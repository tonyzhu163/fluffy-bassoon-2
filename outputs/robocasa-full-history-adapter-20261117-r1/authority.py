"""Measure decoded command authority on development training queries only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from native import load, base_prediction, rollout, write
from bridge import Corpus, Queries, commands
from fluffy_bassoon.meta_rl.flow_history import load_bridge_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    native,motor,codec,stats,identity=load(a.pin)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    policy,contract=load_bridge_policy(a.checkpoint,expected_sha256=digest,device='cuda:0')
    assert contract['mode']=='privileged'
    corpus=Corpus(json.loads(a.manifest.read_text())['trials'],privileged=True)
    reader=Queries(corpus,native,motor)
    rows=[]; cache_statistics=[]
    scales=(.1,.3,1.,3.,10.,30.)
    for ti,stratum in enumerate(corpus.strata):
        for active in (True,False):
            candidates=[i for i in stratum if corpus.episode(i)[0]['meta']['rows'][i[2]]['active']==active]
            if not candidates: continue
            idx=candidates[0] if active else candidates[len(candidates)//2]
            ep,q=corpus.episode(idx)
            with torch.no_grad():
                token=torch.from_numpy(corpus.tokens(idx,'privileged')).to('cuda:0')
                context,_=policy.encode_step(token)
                prepared=reader.prepare([idx])
                base=codec.mapped_physical(base_prediction(native,prepared,ep['noise'][q]))
                if not cache_statistics:
                    for i,(key,value) in enumerate(prepared['prefix'][2]):
                        tokens=torch.cat((key.flatten(2),value.flatten(2)),-1).float()
                        pre=policy.kv_adapter.token_projections[i](tokens)+policy.kv_adapter.context_projections[i](context)[:,None]
                        cache_statistics.append(dict(layer=i,key_rms=float(key.float().square().mean().sqrt()),
                            value_rms=float(value.float().square().mean().sqrt()),
                            mean_tanh_derivative=float((1-pre.tanh().square()).mean()),
                            basis_rms=float(policy.kv_adapter.output_bases[i].weight.square().mean().sqrt())))
                n=int(ep['action_masks'][q,:,0].sum())
                target=ep['teacher_actions'][q]
                for scale in scales:
                    policy.kv_adapter.scale=scale
                    predicted=commands(native,policy,prepared,context,ep['noise'][q][None])[0]
                    rows.append(dict(index=idx,active=active,scale=scale,
                        correction_xyz=(predicted[:n,:3]-base[:n,:3]).mean(0).tolist(),
                        needed_xyz=(target[:n,:3]-base[:n,:3]).mean(0).tolist(),
                        command_rms=float(np.mean((predicted[:n,:3]-base[:n,:3])**2)**.5),
                        needed_rms=float(np.mean((target[:n,:3]-base[:n,:3])**2)**.5),
                        teacher_mse=float(np.mean((predicted[:n,:3]-target[:n,:3])**2)),
                        other_control_rms=float(np.mean((predicted[:n,3:]-base[:n,3:])**2)**.5)))
    aggregates=[]
    for scale in scales:
        active=[r for r in rows if r['scale']==scale and r['active']]
        quiet=[r for r in rows if r['scale']==scale and not r['active']]
        aggregates.append(dict(scale=scale,active_command_rms=float(np.mean([r['command_rms'] for r in active])),
            active_needed_rms=float(np.mean([r['needed_rms'] for r in active])),
            active_teacher_mse=float(np.mean([r['teacher_mse'] for r in active])),
            inactive_command_rms=float(np.mean([r['command_rms'] for r in quiet]))))
    write(a.output/'authority.json',dict(checkpoint=str(a.checkpoint),sha256=digest,contract=contract,
        scales=aggregates,cache_statistics=cache_statistics,rows=rows,
        note='Scaling a fitted adapter measures available command authority; it is not a newly trained model or a held-out result.'))
    print(json.dumps(aggregates),flush=True)
    meta=corpus.episodes[0]['meta']
    result=rollout(native,motor,codec,directory=a.output/'native-nominal',scene=meta['scene'],offset=[0,0,0],
        fraction=0.,target_start=meta['start_signature'],save_images=False)
    write(a.output/'summary.json',dict(complete=True,scales=aggregates,native_nominal_success=result['success']))


if __name__=='__main__': main()
