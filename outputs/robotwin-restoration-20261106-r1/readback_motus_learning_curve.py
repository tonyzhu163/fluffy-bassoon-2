"""Linux CPU comparison of independently verified checkpoint1/2 DEV readouts."""
import json
import math
import os
from pathlib import Path

import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    identities = ['motus-source-grpo-dev03','motus-source-grpo-dev02']
    reports = []
    receipts = []
    for identity,step in zip(identities,(1,2),strict=True):
        path=BASE/(identity+'-readback01')/'RESULT.json'
        report=json.loads(path.read_bytes())
        assert report['status']=='passed_independent_paired_source_trace_noise_readback'
        assert report['algorithm']=='grpo' and report['bank']=='dev'
        assert report.get('selected_training_iteration',2)==step
        assert report['episodes']==128 and report['optimizer_updates']==0
        assert report['all192_paired_batch_draws_tensor_exact']
        assert report['all32_initial_B2_scene_observations_exact']
        reports.append(report)
        receipts.append(dict(identity=identity,selected_iteration=step,result=str(path),sha256=sha(path)))
    # Compare actual cross-run evidence, not just the declared common seed.
    for policy_ordinal,(left_step,right_step) in enumerate(((0,0),(1,2))):
        for panel in range(4):
            for rank in range(8):
                states=[BASE/v/'state' for v in identities]
                rows=[json.loads((state/'actual-eval-noise'/
                    f'policy-{step:04d}-panel-{panel:02d}-rank-{rank:02d}.json').read_bytes())
                    for state,step in zip(states,(left_step,right_step),strict=True)]
                for left,right in zip(rows[0]['samples'],rows[1]['samples'],strict=True):
                    assert left['initial_gaussian_sha256']==right['initial_gaussian_sha256']
                    assert left['shape']==right['shape']==[2,32,32]
                    assert left['extra_random_draws']==right['extra_random_draws']==0
                ordinal=policy_ordinal*24+panel*6
                first=[read_tensor(state/'sampling'/f'rank-{rank:02d}'/f'{ordinal:05d}.pt.gz')
                    for state in states]
                assert first[0]['public_observation_sha256']==first[1]['public_observation_sha256']
    maps=[{row['seed']:row for row in report['paired']} for report in reports]
    assert len(maps[0])==64 and maps[0].keys()==maps[1].keys()
    paired=[]
    for seed,left in maps[0].items():
        right=maps[1][seed]
        assert left['before']==right['before']
        paired.append(dict(seed=seed,SFT=left['before'],checkpoint1=left['after'],checkpoint2=right['after']))
    gained=sum(not row['checkpoint1'] and row['checkpoint2'] for row in paired)
    lost=sum(row['checkpoint1'] and not row['checkpoint2'] for row in paired)
    n=gained+lost
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(gained,lost)+1))/2**n) if n else 1.
    out=BASE/'motus-source-grpo-curve01';out.mkdir()
    result=dict(status='passed_independent_cross_run_existing_DEV_learning_curve',
        source_commit=os.environ['TASK_SOURCE'],GPU_seconds=0,inputs=receipts,
        cross_run_all384_actual_Gaussian_draw_hashes_match=True,
        cross_run_all64_initial_B2_observation_hashes_match=True,
        all64_SFT_outcomes_match=True,SFT_successes=sum(r['SFT'] for r in paired),
        checkpoint1_successes=sum(r['checkpoint1'] for r in paired),
        checkpoint2_successes=sum(r['checkpoint2'] for r in paired),
        checkpoint2_gained_vs1=gained,checkpoint2_lost_vs1=lost,
        paired_gain_checkpoint2_vs1=(gained-lost)/64,exact_two_sided_paired_p=p,
        exploratory_existing_DEV_only=True,new_effectiveness_confirmation=False,
        checkpoint2_negative_confirmation_unchanged=True,paired=paired)
    write(out/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='paired'},indent=2),flush=True)
    print('CURVE_RESULT_SHA256',sha(out/'RESULT.json'),flush=True)


if __name__=='__main__':main()
