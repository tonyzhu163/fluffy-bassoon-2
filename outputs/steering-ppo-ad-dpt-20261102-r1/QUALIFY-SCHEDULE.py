"""Allocated-CPU numerical fixture; never an admitted scientific source."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.frozen_steering_history import (
    SteeringSource, load_episode, save_corpus, split_streams,
)
from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
from fluffy_bassoon.meta_rl.joint_steering_ppo import boundary_state
from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import collect_bank, _save
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset, trial_tokens


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux' or torch.cuda.is_available():
        raise ValueError('Use the allocated Linux CPU with CUDA masked')
    code=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(code/'tests'))
    from test_joint_steering_ppo_worker import Environment,Motor
    check=runpy.run_path(str(Path(__file__).with_name('PREPARE.py')))['prospective_schedule']
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    rows=[]
    for arm in ('velocity','kv_all_layers'):
        root=a.output/('SYNTHETIC-'+arm);root.mkdir()
        arguments=dict(kind=arm,steering_dim=32 if arm=='velocity' else 1152,fraction=.05)
        if arm=='kv_all_layers':arguments['kv_layers']=18
        interface=InterfaceSpec(**arguments)
        source=SteeringSource(arm,pin,'SYNTHETIC-runtime','SYNTHETIC-motor','SYNTHETIC-normalization',
            ('A','B'),interface.record(),None,feature_dim=2)
        actor=ResidualActorCritic(ResidualPPOConfig(2,interface.steering_dim,width=8,layers=1))
        entries,requests,episodes,counters=[],[],[],dict(A=0,B=0)
        for version in range(4):
            cases=[]
            for task in source.tasks:
                count=3 if (version+(task=='B'))%2==0 else 2
                for slot in range(count):
                    case=dict(task=task,slot=counters[task],native_seed=(1000 if task=='A' else 10000)+counters[task],
                              horizon=100,first_success=25 if slot==0 else None)
                    counters[task]+=1;cases.append(case)
                    entries.append(dict(case=case,behavior_version=version,stream_slot=slot))
            with torch.no_grad():actor.actor[-1].bias.fill_(version/10)
            request=dict(sequence=version+1,phase='train',iteration=version+1,policy_iteration=version,
                cases=cases,spec=arguments,lanes=2,source_commit=pin,runtime_commit=source.runtime_commit,
                policy_config=asdict(actor.config),policy=actor.state_dict())
            reference=_save(root/f'request-{version}.pt',request);requests.append(reference)
            actual=dict(request,request_path=reference['path'],request_sha256=reference['sha256'])
            work=root/f'round-{version}'
            environments=[Environment(cases,work,lane) for lane in range(2)]
            result=collect_bank(actor,Motor(),environments,request=actual,output=work/'history',
                                rank=0,deadline_unix=time.time()+60)
            episodes.extend(result['episodes'])
        document=dict(format='frozen_steering_episode_schedule_v1',source_commit=pin,
            runtime_commit=source.runtime_commit,entries=entries,producer_requests=requests)
        path=root/'SYNTHETIC-schedule.json';path.write_text(json.dumps(document))
        reference=dict(path=str(path),sha256=file_sha256(path))
        recipe=dict(iterations=4,envs_per_gpu=2,episodes_per_update=5,collection_shards=1)
        schedule,requested=check(reference,source,recipe)
        loaded=[load_episode(item['checkpoint'],source,schedule=schedule) for item in episodes]
        for episode in loaded:
            item=episode['request_reference']
            from fluffy_bassoon.meta_rl.frozen_steering_history import case_identity
            assert (item['path'],item['sha256'])==requested[(episode['update'],case_identity(episode['physical_episode']['case']))]
        streams=split_streams(loaded,source,validation_slots=dict(A=[1],B=[1]),
                             expected_updates=range(4),schedule=schedule)
        with torch.no_grad():actor.actor[-1].bias.fill_(.75)
        teacher=_save(root/'SYNTHETIC-teacher.pt',boundary_state(actor,torch.optim.Adam(actor.parameters()),
            torch.Generator().manual_seed(7),arm=arm,iteration=4,sequence=4,
            worker_states={},escrow={},interface_spec=arguments))
        manifests=save_corpus(root/'SYNTHETIC-corpus',streams,source,teacher_reference=teacher,
            provenance=dict(scope='SYNTHETIC numerical fixture; no human authority or source admission'))
        for partition in ('train','validation'):
            ad,_=load_dataset(root/'SYNTHETIC-corpus'/f'ad-{partition}.npz')
            dpt,_=load_dataset(root/'SYNTHETIC-corpus'/f'dpt-{partition}.npz')
            for actual,labeled in zip(ad,dpt,strict=True):
                np.testing.assert_array_equal(trial_tokens(actual),trial_tokens(labeled))
                np.testing.assert_allclose(labeled['target_noise'],.75)
        assert len(loaded)==20 and sum(manifests[key]['transitions'] for key in ('ad_train','ad_validation'))==40
        rows.append(dict(arm=arm,synthetic_archived_episodes=20,physical_reference_transitions=40,
            actual_producer_format=True,task_and_version_schedule_exact=True,
            prospective_request_case_join_exact=True,all_non_target_history_identical=True,
            final_teacher_targets_reference_exact=True))
    result=dict(passed=True,source=pin,kind='SYNTHETIC allocated-Linux CPU qualification',arms=rows,
        scientific_source_admitted=False,actual_simulator_episodes=0,actual_policy_controls=0,
        PPO_or_student_training_launched=False,human_launch_hold_unchanged=True)
    (a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
