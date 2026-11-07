"""Bounded real support/query qualification of completed decoded students.

Two source DEV cases, five controls, no SGD and no success-rate/adaptation
claim. B2 deployment is an explicit numerical condition distinct from the
source corpus's B64, not a cross-batch equivalence claim.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.chunk_controller import FrozenChunkController
from fluffy_bassoon.meta_rl.chunk_robotwin import RoboTwinChunkSession
from fluffy_bassoon.meta_rl.chunk_training import load_chunk_policy
from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree,sha
from fit_decoded_student import write
from fluffy_bassoon.rl.robotwin_batch import RoboTwinBatchRuntime
from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor
from fluffy_bassoon.rl.robotwin_noise import model_config
from qualify_batch_runtime import write_trace
from qualify_motor import MOTOR,READOUT,Memory


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def policies(plan):
    audit_path = BASE/'decodedfit02-readback01/RESULT.json'
    assert sha(audit_path)==plan['fit_readback_sha256']
    audit = json.loads(audit_path.read_bytes())
    assert audit['status']=='passed' and audit['all_complete_recoveries_passed']
    assert not audit['test_tensor_loaded']
    history_path=BASE/'decodedhistorybc01-readback01/RESULT.json'
    assert sha(history_path)==plan['history_readback_sha256']
    history=json.loads(history_path.read_bytes())
    assert history['status']=='passed' and history['full_learner_optimizer_sampler_rng_restored']
    assert history['actor_window_stream_matches_all3_previous_methods']
    precision=BASE/'studentprecisionall01-readback01/RESULT.json'
    assert sha(precision)==plan['precision_readback_sha256']
    assert json.loads(precision.read_bytes())['all_four_gpu_precision_qualified']
    result = {}
    for method,binding in plan['policy_exports'].items():
        path = Path(binding['output_dir'])
        assert path.resolve().is_relative_to(BASE.resolve())
        report = json.loads((path/'RESULT.json').read_bytes())
        reference=history if method=='history_behavior_cloning' else audit['methods'][method]
        assert sha(path/'policy.pt')==binding['sha256']==reference['policy_sha256']
        policy,_ = load_chunk_policy(path/'policy.pt',binding['sha256'],
                                    expected_contract=report['contract'],device='cpu')
        assert policy.config.layout.command_shape==(32,14)
        result[method] = policy
    assert set(result)=={'ic_iql','dit','static_behavior_cloning','history_behavior_cloning'}
    return result


def main(args):
    assert sys.platform=='linux'
    plan = json.loads((ROOT/'STUDENTONLINE02-CONTRACT.json').read_bytes())
    loaded = policies(plan)
    cases = list(map(int,json.loads((READOUT/'RESULT.json').read_bytes())['cases'][:2]))
    assert cases==plan['source_dev_cases'] and len(set(cases))==2
    if args.preflight:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
        import gzip
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha as file_sha
        for name,digest in plan['source_files'].items():assert file_sha(ROOT.parents[1]/name)==digest
        assert file_sha(ROOT/'evaluate_decoded_students02.py')==plan['entry_sha256']
        assert file_sha(ROOT/'evaluate_decoded_students02.sh')==plan['launcher_sha256']
        fixture=BASE/'studentonline01-signals/frozen_base-episode-0-batch-01-intent.pt.gz'
        with gzip.open(fixture,'rb') as stream:intent=torch.load(stream,map_location='cpu',weights_only=False)
        features,bases=(torch.as_tensor(intent[k],dtype=torch.float32) for k in ('features','bases'))
        torch.set_float32_matmul_precision('high')
        for method,policy in loaded.items():
            controller=FrozenChunkController(policy,native_batch=2,mode='static' if method=='static_behavior_cloning' else 'full')
            commands=controller.propose(features,bases,episode_starts=intent['episode_starts'])
            assert commands.shape==(2,32,14) and torch.isfinite(commands).all()
            assert torch.get_float32_matmul_precision()=='high'
            controller._check()
        root = BASE/'cpustudentonline03';root.mkdir()
        write(root/'RESULT.json',dict(status='passed',source_commit=os.environ['TASK_SOURCE'],
            policy_exports=plan['policy_exports'],source_dev_cases=cases,gpu_seconds=0,
            native_motus_or_simulator_constructed=False,robot_success_or_adaptation_qualified=False,
            matched_full_history_imitation_loaded=True,scoped_precision_guard_exercised_on_all4_cpu_exports=True,
            native_batch=2,chronological_ad_qualified=False))
        return
    del loaded;gc.collect()
    assert plan['status']=='admitted' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    root = Path(os.environ['TASK_OUT'])
    local = Path(os.environ['TASK_LOCAL'])
    assert root==BASE/'studentonline02-signals' and local.is_dir()
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(MOTOR/'model.safetensors')==plan['model_sha256'] and sha(norm)==plan['norm_stats_sha256']
    report = dict(status='starting',source_commit=os.environ['TASK_SOURCE'],cases=cases,
        contract_sha256=sha(ROOT/'STUDENTONLINE02-CONTRACT.json'),
        source_native_batch=64,deployment_native_batch=2,cross_batch_equivalence_claim=False,
        optimizer_updates=0,test_tensor_loaded=False,learning_improvement_claim=False,
        arms={},traces=[],started_unix=time.time())
    write(root/'RESULT.json',report)
    from rlinf.models import get_model
    model = get_model(model_config(os.environ['REPO_PATH'],MOTOR,norm,action_horizon=32,num_steps=5)).cuda()
    motor = FrozenRoboTwinMotor(model)
    initial = {}
    runtime = None

    def deadline():
        assert time.time()<float(os.environ['TASK_DEADLINE'])-45,'Preserve time for owned-worker cleanup'
        used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        assert used<plan['maximum_output_gib']*2**30

    try:
        with Memory(torch) as memory:
            for arm in plan['arms']:
                deadline()
                method = 'static_behavior_cloning' if arm=='frozen_base' else arm
                policy_root = Path(plan['policy_exports'][method]['output_dir'])
                policy,_ = load_chunk_policy(policy_root/'policy.pt',plan['policy_exports'][method]['sha256'],
                    expected_contract=json.loads((policy_root/'RESULT.json').read_bytes())['contract'],device='cuda')
                if arm=='frozen_base':
                    with torch.no_grad():
                        policy.actor.weight.zero_();policy.actor.bias.zero_()
                mode = 'static' if method=='static_behavior_cloning' else 'full'
                control = FrozenChunkController(policy,native_batch=2,mode=mode)
                runtime = RoboTwinBatchRuntime.create(motor,lanes=2,rlinf_root=os.environ['REPO_PATH'],
                    robotwin_root=os.environ['ROBOTWIN_PATH'],output=local/arm,
                    task_episode_limits={'place_phone_stand':192},seed=plan['motor_noise_seed'])
                session = RoboTwinChunkSession(runtime,control)
                record = dict(mode=mode,policy_export=plan['policy_exports'][method],episodes=[],
                    zero_output_head=arm=='frozen_base')
                report['arms'][arm] = record
                for episode in range(2):
                    deadline()
                    # Separate runtimes permit identical noise addresses across
                    # controls without reusing an episode inside one runtime.
                    receipts = session.reset('place_phone_stand',cases,
                        case_ids=[f'case-{seed}/attempt-{episode}' for seed in cases],new_trials=episode==0)
                    for lane,receipt in enumerate(receipts):
                        key = episode,lane
                        identity = receipt['public_observation_sha256']
                        if arm=='frozen_base':initial[key]=identity
                        assert identity==initial[key],'Paired initial observations differ'
                    episodes = [dict(case=seed,role='support' if episode==0 else 'query',
                                     success=False,complete=False,transitions=[]) for seed in cases]
                    record['episodes'].append(dict(index=episode,resets=receipts,lanes=episodes))
                    while any(runtime._active):
                        deadline()
                        commands,features,metadata = session.propose()
                        base = runtime._pending['base_actions']
                        if arm=='frozen_base':assert torch.equal(commands,base)
                        name = f'{arm}-episode-{episode}-batch-{metadata["native_batch"]:02d}'
                        intent = name+'-intent.pt.gz'
                        write_trace(root/intent,cpu_tree(dict(commands=commands,features=features,bases=base,
                            native_batch=metadata,episode_starts=session.starts.copy())))
                        transitions = session.step(commands)
                        path = name+'-transitions.pt.gz'
                        write_trace(root/path,transitions)
                        report['traces'].append(dict(arm=arm,episode=episode,intent=intent,transitions=path))
                        for transition in transitions:
                            lane = transition['native_batch']['row']
                            value = episodes[lane]
                            value['transitions'].append(dict(chunk=transition['chunk_index'],
                                duration=transition['physics_duration_seconds'],execution=transition['execution'],
                                terminated=transition['terminated'],truncated=transition['truncated']))
                            value['success'] |= bool(transition['audit']['success'])
                            value['complete'] = transition['terminated'] or transition['truncated']
                        write(root/'RESULT.json',report)
                    assert all(value['complete'] for value in episodes)
                    write_trace(root/f'{arm}-episode-{episode}-controller.pt.gz',cpu_tree(dict(
                        state=control.state,encoders=[vars(encoder) for encoder in control.encoders],
                        policy_export=plan['policy_exports'][method],mode=mode,zero_output_head=arm=='frozen_base')))
                control._check();motor.check_frozen()
                session.close()
                assert not any(process.is_alive() for process in runtime.pool.processes)
                record.update(owned_workers_closed=True,frozen_weights_and_statistics_unchanged=True)
                del session,control,policy,runtime
                runtime = None;gc.collect();torch.cuda.empty_cache()
                write(root/'RESULT.json',report)
        report['memory'] = memory.result()
        assert report['memory']['conservative_minimum_free_bytes']>=max(8*2**30,.08*report['memory']['total_bytes'])
        assert len(report['arms'])==5 and sum(len(row['episodes']) for row in report['arms'].values())==10
        report.update(status='complete_trained_student_runtime_qualification',real_episodes=20,
            native_motor_calls=motor.sample_calls,frozen_online_control_executed=True,
            independent_cpu_replay_pending=True,chronological_ad_qualified=False,scoped_highest_precision=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if runtime is not None:runtime.close()
        report['ended_unix'] = time.time();write(root/'RESULT.json',report)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args())
