"""Isolate precision on recorded physical inputs; no new robot dispatch or SGD."""
import copy
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_decoded_students import BASE, ROOT, policies
from fit_decoded_student import write
from fluffy_bassoon.meta_rl.chunk_controller import FrozenChunkController
from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, sha
from qualify_motor import Memory


def main():
    assert os.uname().nodename.startswith('job-') and torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    out=Path(os.environ['TASK_OUT']);source=BASE/'studentonline01-signals'
    plan=json.loads((ROOT/os.environ.get('TASK_PRECISION_CONTRACT','STUDENTPRECISIONGPU01-CONTRACT.json')).read_bytes())
    student=json.loads((ROOT/'STUDENTONLINE01-CONTRACT.json').read_bytes())
    assert sha(source/'RESULT.json')==plan['physical_result_sha256']
    physical=json.loads((source/'RESULT.json').read_bytes())
    assert physical['status']=='complete_trained_student_runtime_qualification'
    loaded=policies(student)
    report=dict(status='checking',source_commit=os.environ['TASK_SOURCE'],gpu_seconds_charged_separately=True,
        optimizer_updates=0,new_robot_episodes=0,test_tensor_loaded=False,arms={},input_sha256={})
    write(out/'RESULT.json',report)

    cache={}
    def read(name):
        if name in cache:return cache[name]
        path=source/name
        assert path.resolve().parent==source.resolve()
        report['input_sha256'][name]=sha(path)
        with gzip.open(path,'rb') as stream:cache[name]=torch.load(stream,map_location='cpu',weights_only=False)
        return cache[name]

    def replay(policy,arm,device,precision):
        policy=copy.deepcopy(policy).to(device).eval().requires_grad_(False)
        signature={k:(id(v),v._version) for k,v in (*policy.named_parameters(),*policy.named_buffers())}
        static=arm in ('frozen_base','static_behavior_cloning')
        feedback=np.zeros((2,policy.config.layout.feedback_dim),np.float32)
        state=policy.core.initial_state(2);first=np.ones(2,np.bool_)
        commands,states={},{}
        production_error=None
        with torch.no_grad():
            for episode in range(2):
                for entry in physical['traces']:
                    if entry['arm']!=arm or entry['episode']!=episode:continue
                    assert time.time()<float(os.environ['TASK_DEADLINE'])-15
                    intent=read(entry['intent']);transitions=read(entry['transitions'])
                    features,bases=(torch.as_tensor(intent[k],dtype=torch.float32,device=device)
                                    for k in ('features','bases'))
                    active=np.zeros(2,np.bool_);active[intent['native_batch']['active_lanes']]=True
                    previous=np.zeros_like(feedback) if static else feedback
                    token=torch.cat((features,bases.flatten(1),torch.as_tensor(previous,device=device),
                                     torch.as_tensor(intent['episode_starts'][:,None],device=device).float()),-1)
                    ambient=torch.get_float32_matmul_precision()
                    try:
                        torch.set_float32_matmul_precision(precision)
                        residual,_,after=policy.step(token,state,
                            reset=torch.as_tensor((first|static)&active,device=device),deterministic=True)
                    finally:torch.set_float32_matmul_precision(ambient)
                    mask=torch.as_tensor(active,device=device)
                    state={k:torch.where(mask.reshape(-1,*([1]*(v.ndim-1))),v,state[k]) for k,v in after.items()}
                    issued=bases+torch.where(mask[:,None,None],residual,0.)
                    commands[entry['intent']]=issued.cpu()
                    if device=='cuda' and precision=='highest' and production_error is None:
                        # A real production-controller call on the first recorded
                        # observation also exercises the scoped precision guard.
                        control=FrozenChunkController(policy,native_batch=2,mode='static' if static else 'full')
                        assert torch.get_float32_matmul_precision()=='high'
                        actual=control.propose(features,bases,episode_starts=intent['episode_starts'])
                        assert torch.get_float32_matmul_precision()=='high'
                        production_error=float((actual-issued).abs().max())
                        assert production_error==0.
                        del control
                    for row in transitions:
                        lane=row['native_batch']['row'];first[lane]=False
                        feedback[lane]=np.concatenate((row['requested_actions'].reshape(-1),
                            row['measured_valid_target_mask'].astype(np.float32),
                            np.array([row['reward'],row['terminated'],row['truncated'],
                                      row['physics_duration_seconds']],np.float32)))
                states[episode]=cpu_tree(state)
        assert signature=={k:(id(v),v._version) for k,v in (*policy.named_parameters(),*policy.named_buffers())}
        del policy
        return commands,states,production_error

    try:
        with Memory(torch) as memory:
            selected=plan.get('arms',student['arms'])
            assert selected and len(set(selected))==len(selected) and set(selected)<=set(student['arms'])
            for arm in selected:
                method='static_behavior_cloning' if arm=='frozen_base' else arm
                policy=copy.deepcopy(loaded[method]) if arm=='frozen_base' else loaded[method]
                if arm=='frozen_base':
                    with torch.no_grad():policy.actor.weight.zero_();policy.actor.bias.zero_()
                cpu,cpu_states,_=replay(policy,arm,'cpu','highest')
                torch.set_float32_matmul_precision('high')
                high,high_states,_=replay(policy,arm,'cuda','high')
                fixed,fixed_states,production_error=replay(policy,arm,'cuda','highest')
                high_error=fixed_error=normalized_error=memory_error=0.
                for entry in physical['traces']:
                    if entry['arm']!=arm:continue
                    name=entry['intent'];recorded=read(name)['commands']
                    high_error=max(high_error,float((high[name]-recorded).abs().max()))
                    error=(fixed[name]-cpu[name]).abs()
                    fixed_error=max(fixed_error,float(error.max()))
                    normalized_error=max(normalized_error,float((error/policy.residual_radius).max()))
                for episode in range(2):
                    original=read(f'{arm}-episode-{episode}-controller.pt.gz')['state']
                    for key in original:
                        assert torch.allclose(high_states[episode][key],original[key],atol=1e-5,rtol=1e-5)
                        a,b=fixed_states[episode][key],cpu_states[episode][key]
                        if a.is_floating_point():
                            memory_error=max(memory_error,float((a-b).abs().max()))
                            assert torch.allclose(a,b,atol=1e-4,rtol=1e-4)
                        else:assert torch.equal(a,b)
                row=dict(original_high_command_error=high_error,fixed_cpu_command_error=fixed_error,
                    fixed_cpu_command_error_over_radius=normalized_error,fixed_cpu_memory_max_error=memory_error,
                    production_guard_first_query_error=production_error)
                def compact(states):
                    result={}
                    for episode,state in states.items():
                        values=[]
                        for lane,length in enumerate(state['length'].tolist()):
                            assert length>0 and not state['memory'][lane,:,:-length].any()
                            values.append(state['memory'][lane,:,-length:].clone())
                        result[episode]=dict(length=state['length'],valid_memory=values)
                    return result
                fixture=out/(arm+'-precision.pt.gz')
                with gzip.open(fixture,'xb',compresslevel=1) as stream:
                    torch.save(dict(cpu=cpu,high=high,highest=fixed,radius=policy.residual_radius.cpu(),
                        cpu_states=compact(cpu_states),high_states=compact(high_states),
                        highest_states=compact(fixed_states)),stream)
                row.update(fixture=fixture.name,fixture_sha256=sha(fixture))
                report['arms'][arm]=row;write(out/'RESULT.json',report)
                assert high_error<=1e-5 and fixed_error<=1e-5 and normalized_error<=1e-3
                del cpu,cpu_states,high,high_states,fixed,fixed_states
                torch.cuda.empty_cache()
        measured=memory.result()
        assert measured['conservative_minimum_free_bytes']>=max(8*2**30,.08*measured['total_bytes'])
        report.update(status='passed',memory=measured,high_reproduces_original_records=True,
            highest_matches_cpu_at_original_tolerances=True,production_precision_guard_exercised_on_gpu=True,
            fresh_robot_episodes_with_fixed_precision_still_required=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:write(out/'RESULT.json',report)


if __name__=='__main__':main()
