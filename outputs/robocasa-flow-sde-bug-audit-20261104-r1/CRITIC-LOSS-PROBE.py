"""Isolate value-clipping behavior through the deployed embodied loss registry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rlinf',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.rlinf))
    import rlinf.algorithms.losses  # Registers the actual actor_critic loss.
    from rlinf.algorithms.registry import policy_loss
    cases=[]
    for predicted,target in [(.9,1.),(.1,0.)]:
        for reference in [.55,predicted]:
            value=torch.full((2,),predicted,requires_grad=True)
            loss,metrics=policy_loss(task_type='embodied',loss_type='actor_critic',
                reward_type='chunk_level',logprob_type='chunk_level',single_action_dim=12,
                logprobs=torch.zeros(2,50,12),old_logprobs=torch.zeros(2,50,12),
                advantages=torch.zeros(2,1),values=value,
                prev_values=torch.full((2,1),reference),returns=torch.full((2,1),target),
                loss_mask=torch.ones(2,1,dtype=torch.bool),clip_ratio_low=.2,
                clip_ratio_high=.2,value_clip=.2,huber_delta=10.)
            loss.backward()
            cases.append(dict(predicted=predicted,return_target=target,
                old_value_reference=reference,critic_loss=float(metrics['critic/value_loss']),
                value_clip_ratio=float(metrics['critic/value_clip_ratio']),
                gradient_wrt_predictions=value.grad.tolist()))
    assert cases[0]['value_clip_ratio']==cases[2]['value_clip_ratio']==1
    assert cases[0]['gradient_wrt_predictions']==cases[2]['gradient_wrt_predictions']==[0.,0.]
    assert all(abs(v)>0 for i in [1,3] for v in cases[i]['gradient_wrt_predictions'])
    summaries={}
    for path in sorted(args.metrics.glob('*.csv')):
        rows=list(csv.DictReader(path.open()))
        keys=['critic/value_clip_ratio','critic/value_loss','critic/explained_variance']
        summaries[path.stem]=dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(rows),
            statistics={key:dict(mean=statistics.mean(float(r[key]) for r in rows),
                minimum=min(float(r[key]) for r in rows),maximum=max(float(r[key]) for r in rows),
                last_ten_updates_mean=statistics.mean(float(r[key]) for r in rows if int(r['iteration'])>=121))
                for key in keys})
    paths=[args.rlinf/'rlinf/algorithms'/name for name in ['registry.py','losses.py','utils.py']]
    report=dict(completed=True,device='cpu',torch_version=torch.__version__,
        scope='Synthetic values, actual deployed policy_loss registry/preprocessing/loss; not measured model errors',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=cases,production_update_metrics=summaries,
        metrics_limit='Epoch/update-averaged metrics mix pre-SGD mismatch and real parameter updates; cannot attribute clipping to this defect')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    main()
