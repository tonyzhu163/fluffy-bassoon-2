"""Allocated CPU readout of saved canonical rewards; never changes training."""
import json
from pathlib import Path
import sys

import torch

root=Path(sys.argv[1]);rows=[]
for iteration in (1,2):
    for rank in range(4):
        receipt=root/'training-rollouts'/f'iteration-{iteration:04d}-policy-{iteration-1:04d}-rank-{rank:02d}.json'
        ref=json.loads(receipt.read_text());path=Path(ref['path']);before=path.stat()
        assert ref['completed'] and ref['producing_policy_iteration']==iteration-1
        # Native checkpoint capture already hashes the full canonical file.
        # mmap touches only small reward/mask tensors, not tens of GB of RGB.
        record=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        batch=record['rollout_batch'];reward=batch['rewards'].double()
        assert reward.shape==(8,256,50) and torch.isfinite(reward).all()
        returns=reward.sum(dim=(0,2)).reshape(4,8,8)
        spread=returns.std(dim=-1,correction=1)
        assert torch.isfinite(spread).all()
        mask=batch['loss_mask'];assert mask.shape==(8,256,1)
        rows.append(dict(collection_iteration=iteration,rank=rank,
            canonical_path=str(path),runtime_verified_canonical_sha256=ref['sha256'],
            groups=32,replicas_per_group=8,trajectory_reward_sums=returns.tolist(),
            sample_std_by_group=spread.tolist(),constant_return_groups=int((spread==0).sum()),
            positive_reward_sum_episodes=int((returns>0).sum()),
            native_valid_chunk_slots=int(mask.sum()),native_total_chunk_slots=mask.numel(),
            measured_executed_motor_mask=False,student_corpus_admitted=False))
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        del record,batch,reward,returns,spread,mask
(root/'GROUP-RETURN-SPREAD.json').write_text(json.dumps(dict(status='complete',
    native_training_unchanged=True,post_training_readout=True,extra_optimizer_updates=0,
    extra_simulator_controls=0,measured_physical_control_matching=False,rows=rows),indent=2)+'\n')
