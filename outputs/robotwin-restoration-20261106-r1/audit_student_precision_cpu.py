"""Recompute recorded student decisions independently on a cold CPU process."""
import argparse
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
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    assert sys.platform=='linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    out=Path(args.out);out.mkdir()
    started=time.monotonic()
    report=dict(status='checking',source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,
        optimizer_updates=0,new_robot_episodes=0,test_tensor_loaded=False,arms={})
    write(out/'RESULT.json',report)
    try:
        physical_root=BASE/'studentonline01-signals'
        plan=json.loads((ROOT/'STUDENTPRECISIONGPU02-CONTRACT.json').read_bytes())
        assert sha(physical_root/'RESULT.json')==plan['physical_result_sha256']
        for name,digest in plan['source_files'].items():assert sha(ROOT.parents[1]/name)==digest
        assert sha(ROOT/'qualify_student_precision_gpu.py')==plan['entry_sha256']
        assert sha(ROOT/'qualify_student_precision_gpu02.sh')==plan['launcher_sha256']
        physical=json.loads((physical_root/'RESULT.json').read_bytes())
        student=json.loads((ROOT/'STUDENTONLINE01-CONTRACT.json').read_bytes())
        loaded=policies(student)
        torch.set_float32_matmul_precision('highest')
        if args.preflight:
            targets=[('ic_iql',None)]
        else:
            targets=[]
            for producer in ('studentprecisiongpu01-signals','studentprecisiongpu02-signals'):
                source=BASE/producer
                if not (source/'RESULT.json').exists():continue
                result=json.loads((source/'RESULT.json').read_bytes())
                if producer.endswith('01-signals'):assert sha(source/'RESULT.json')==plan['prior_partial_result_sha256']
                else:assert result['status']=='passed'
                for arm,row in result['arms'].items():
                    assert sha(source/row['fixture'])==row['fixture_sha256']
                    targets.append((arm,(source,row)))
            assert len({arm for arm,_ in targets})==len(targets)
            assert set(arm for arm,_ in targets)==set(student['arms'])
        with torch.no_grad():
            for arm,producer in targets:
                method='static_behavior_cloning' if arm=='frozen_base' else arm
                policy=copy.deepcopy(loaded[method]) if arm=='frozen_base' else loaded[method]
                if arm=='frozen_base':policy.actor.weight.zero_();policy.actor.bias.zero_()
                static=arm in ('frozen_base','static_behavior_cloning')
                state=policy.core.initial_state(2);first=np.ones(2,np.bool_)
                feedback=np.zeros((2,policy.config.layout.feedback_dim),np.float32)
                fixture=read(producer[0]/producer[1]['fixture']) if producer else None
                row=dict(decisions=0,max_gpu_command_error=0.,max_gpu_memory_error=0.)
                for episode in range(2):
                    for entry in physical['traces']:
                        if entry['arm']!=arm or entry['episode']!=episode:continue
                        intent=read(physical_root/entry['intent']);transitions=read(physical_root/entry['transitions'])
                        features,bases=(torch.as_tensor(intent[k],dtype=torch.float32) for k in ('features','bases'))
                        active=np.zeros(2,np.bool_);active[intent['native_batch']['active_lanes']]=True
                        previous=np.zeros_like(feedback) if static else feedback
                        token=torch.cat((features,bases.flatten(1),torch.from_numpy(previous),
                            torch.as_tensor(intent['episode_starts'][:,None]).float()),-1)
                        residual,_,after=policy.step(token,state,reset=torch.from_numpy((first|static)&active),deterministic=True)
                        mask=torch.from_numpy(active)
                        state={k:torch.where(mask.reshape(-1,*([1]*(v.ndim-1))),v,state[k]) for k,v in after.items()}
                        issued=bases+torch.where(mask[:,None,None],residual,0.)
                        if fixture:
                            name=entry['intent'];assert torch.equal(issued,fixture['cpu'][name])
                            error=(issued-fixture['highest'][name]).abs()
                            row['max_gpu_command_error']=max(row['max_gpu_command_error'],float(error.max()))
                            assert float(error.max())<=1e-5 and float((error/policy.residual_radius).max())<=1e-3
                            assert float((fixture['high'][name]-intent['commands']).abs().max())<=1e-5
                        row['decisions']+=1
                        for transition in transitions:
                            lane=transition['native_batch']['row'];first[lane]=False
                            feedback[lane]=np.concatenate((transition['requested_actions'].reshape(-1),
                                transition['measured_valid_target_mask'].astype(np.float32),
                                np.array([transition['reward'],transition['terminated'],transition['truncated'],
                                    transition['physics_duration_seconds']],np.float32)))
                    if fixture:
                        assert torch.equal(state['length'],fixture['cpu_states'][episode]['length'])
                        assert torch.equal(state['length'],fixture['highest_states'][episode]['length'])
                        for lane,length in enumerate(state['length'].tolist()):
                            memory=state['memory'][lane,:,-length:]
                            assert torch.equal(memory,fixture['cpu_states'][episode]['valid_memory'][lane])
                            gpu=fixture['highest_states'][episode]['valid_memory'][lane]
                            row['max_gpu_memory_error']=max(row['max_gpu_memory_error'],float((memory-gpu).abs().max()))
                            assert torch.allclose(memory,gpu,atol=1e-4,rtol=1e-4)
                        original=read(physical_root/f'{arm}-episode-{episode}-controller.pt.gz')['state']
                        assert torch.equal(original['length'],fixture['high_states'][episode]['length'])
                        for lane,length in enumerate(original['length'].tolist()):
                            assert torch.allclose(original['memory'][lane,:,-length:],fixture['high_states'][episode]['valid_memory'][lane],atol=1e-5,rtol=1e-5)
                report['arms'][arm]=row;write(out/'RESULT.json',report)
        assert not torch.cuda.is_initialized()
        report.update(status='passed',preflight_only=args.preflight,
            all_four_gpu_precision_qualified=not args.preflight,fresh_robot_episodes_still_required=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['seconds']=time.monotonic()-started;write(out/'RESULT.json',report)


if __name__=='__main__':main()
