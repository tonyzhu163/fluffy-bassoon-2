"""Independent learner references; execute only inside an allocated Linux step."""
import json
import math
from pathlib import Path
import socket
import sys

import torch
from rlinf.algorithms.registry import calculate_adv_and_returns


assert sys.platform=='linux' and socket.gethostname().startswith('job-')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
steps,batch,chunk,group=3,16,50,8
reward=torch.zeros(steps,batch,chunk)
done=torch.zeros(steps+1,batch,chunk,dtype=torch.bool)
mask=torch.ones(steps,batch,chunk,dtype=torch.bool)
first=[1,0,1,0,0,1,0,0]+[1]*8
second=[0,1,0,0,0,0,0,0]+[0]*8
for b in range(batch):
    reward[0,b,0]=first[b];reward[1,b,0]=second[b]
    if first[b]:done[1,b]=True;mask[1:,b]=False
    elif second[b]:done[2,b]=True;mask[2:,b]=False
common=dict(task_type='embodied',rewards=reward,dones=done,loss_mask=mask,
    loss_mask_sum=None,reward_type='chunk_level',num_action_chunks=chunk,
    gamma=.99,gae_lambda=.95)
actual=calculate_adv_and_returns(adv_type='grpo',group_size=group,values=None,**common)
# Scalar loops independently reproduce the documented trajectory score and
# unbiased group standard deviation; no native preprocessing helper is used.
scores=[]
for b in range(batch):
    score=0.
    for t in reversed(range(steps)):
        if bool(done[t+1,b].any()):score=0.
        score+=sum(float(v) for v in reward[t,b])
    scores.append(score)
expected=torch.zeros(steps,batch,1)
for start in range(0,batch,group):
    values=scores[start:start+group];mean=sum(values)/group
    std=math.sqrt(sum((v-mean)**2 for v in values)/(group-1))
    for b in range(start,start+group):
        advantage=(scores[b]-mean)/(std+1e-6)
        for t in range(steps):
            if bool(mask[t,b].any()):expected[t,b,0]=advantage
error=float((actual['advantages']-expected).abs().max())
assert error<=1e-6 and bool(torch.isfinite(actual['advantages']).all())
assert 'returns' not in actual and torch.equal(actual['advantages'][:,8:],torch.zeros_like(expected[:,8:]))
value=torch.linspace(-.2,.2,(steps+1)*batch).reshape(steps+1,batch,1)
one=calculate_adv_and_returns(adv_type='gae',group_size=1,values=value.clone(),**common)
eight=calculate_adv_and_returns(adv_type='gae',group_size=8,values=value.clone(),**common)
for key in ('advantages','returns'):
    assert torch.equal(one[key],eight[key]) and bool(torch.isfinite(eight[key]).all())
report=dict(status='complete',host=socket.gethostname(),native_grpo_reference_max_abs=error,
    constant_return_group_zero=True,critic_returns_absent=True,
    gae_group1_vs8_advantages_and_returns_bit_identity=True,
    source_reward_first_success_fixture=True,numerical_device='cpu',
    model_or_gpu_training_qualified_by_this_receipt=False)
(out/'learner-numerics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
