"""Measure CPU/GPU differences without changing the failed replay's tolerances."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from evaluate_decoded_students import BASE, ROOT, policies
from fit_decoded_student import write
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha


def read(path):
    with gzip.open(path,'rb') as stream:
        return torch.load(stream,map_location='cpu',weights_only=False)


def main():
    assert sys.platform=='linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    out=BASE/'studentonline01-memorydiagnostic01';out.mkdir()
    root=BASE/'studentonline01-signals'
    plan=json.loads((ROOT/'STUDENTONLINE01-CONTRACT.json').read_bytes())
    result=json.loads((root/'RESULT.json').read_bytes())
    loaded=policies(plan)
    report=dict(status='diagnostic_only_not_qualification',source_commit=os.environ['TASK_SOURCE'],
        gpu_result_sha256=sha(root/'RESULT.json'),original_tolerances=plan['cpu_replay_tolerances'],
        gpu_seconds=0,optimizer_updates=0,test_tensor_loaded=False,arms={})
    started=time.monotonic()
    try:
        with torch.no_grad():
            for arm in plan['arms']:
                method='static_behavior_cloning' if arm=='frozen_base' else arm
                policy=copy.deepcopy(loaded[method]) if arm=='frozen_base' else loaded[method]
                if arm=='frozen_base':policy.actor.weight.zero_();policy.actor.bias.zero_()
                static=method=='static_behavior_cloning'
                feedback=np.zeros((2,policy.config.layout.feedback_dim),np.float32)
                state=policy.core.initial_state(2)
                first=np.ones(2,np.bool_)
                row=dict(max_command_absolute_error=0.,max_command_error_over_radius=0.,boundaries=[])
                report['arms'][arm]=row
                for episode in range(2):
                    for entry in result['traces']:
                        if entry['arm']!=arm or entry['episode']!=episode:continue
                        intent=read(root/entry['intent']);transitions=read(root/entry['transitions'])
                        features,bases,commands=(torch.as_tensor(intent[k],dtype=torch.float32)
                                                for k in ('features','bases','commands'))
                        active=np.zeros(2,np.bool_);active[intent['native_batch']['active_lanes']]=True
                        previous=np.zeros_like(feedback) if static else feedback
                        token=torch.cat((features,bases.flatten(1),torch.from_numpy(previous),
                                         torch.from_numpy(intent['episode_starts'][:,None]).float()),-1)
                        residual,_,after=policy.step(token,state,reset=torch.from_numpy((first|static)&active),
                                                    deterministic=True)
                        mask=torch.from_numpy(active)
                        state={k:torch.where(mask.reshape(-1,*([1]*(v.ndim-1))),v,state[k]) for k,v in after.items()}
                        error=(bases+torch.where(mask[:,None,None],residual,0.)-commands).abs()
                        row['max_command_absolute_error']=max(row['max_command_absolute_error'],float(error.max()))
                        row['max_command_error_over_radius']=max(row['max_command_error_over_radius'],float((error/policy.residual_radius).max()))
                        for t in transitions:
                            lane=t['native_batch']['row'];first[lane]=False
                            feedback[lane]=np.concatenate((t['requested_actions'].reshape(-1),
                                t['measured_valid_target_mask'].astype(np.float32),
                                np.array([t['reward'],t['terminated'],t['truncated'],t['physics_duration_seconds']],np.float32)))
                    saved=read(root/f'{arm}-episode-{episode}-controller.pt.gz')['state']
                    assert torch.equal(saved['length'],state['length'])
                    a,b=state['memory'],saved['memory'];delta=a-b
                    layers=[]
                    for i in range(a.shape[1]):
                        x,y,z=a[:,i],b[:,i],delta[:,i]
                        layers.append(dict(layer=i,max_abs=float(z.abs().max()),
                            relative_l2=float(z.norm()/y.norm().clamp_min(1e-20)),
                            cpu_max_magnitude=float(x.abs().max()),gpu_max_magnitude=float(y.abs().max())))
                    row['boundaries'].append(dict(episode=episode,lengths=state['length'].tolist(),
                        memory_allclose_original_tolerance=torch.allclose(a,b,atol=1e-4,rtol=1e-4),layers=layers))
                    write(out/'RESULT.json',report)
        assert not torch.cuda.is_initialized()
    finally:
        report['seconds']=time.monotonic()-started;write(out/'RESULT.json',report)


if __name__=='__main__':main()
