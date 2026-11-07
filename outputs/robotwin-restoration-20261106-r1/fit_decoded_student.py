"""Fit the shared decoded student on the verified one-policy physical pilot.

TRAIN fits; DEV monitors without selecting a checkpoint or fitting statistics.
TEST is never opened. This pilot does not admit chronological AD or DPT.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout, trial_tokens
from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig
from fluffy_bassoon.meta_rl.chunk_training import ChunkFitConfig, ChunkTrainer, load_chunk_policy
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from fluffy_bassoon.meta_rl.warmstart import checked_load


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def physical_data():
    exported = json.loads((ROOT/'DECODEDHISTORY01-COMPLETE.json').read_bytes())
    bounds = json.loads((ROOT/'DECODED-FIT-BOUNDS02.json').read_bytes())
    for path, expected in ((Path(exported['export_result']),exported['export_result_sha256']),
            (Path(exported['independent_audit_result']),exported['independent_audit_sha256']),
            (Path(bounds['result']),bounds['result_sha256']),
            (Path(bounds['statistics']),bounds['statistics_sha256'])):
        if sha(path) != expected:
            raise ValueError('Physical dataset or fitting profile changed')
    original = json.loads(Path(exported['export_result']).read_bytes())
    qualified = json.loads(Path(bounds['result']).read_bytes())
    if (original['available_teacher_policy_versions'] != [0]
            or qualified['status'] != 'train_bounds_qualified_on_train_dev'
            or qualified['global_margin'] != 2 or qualified['test_tensor_loaded']):
        raise ValueError('Require the qualified one-policy source fitting profile')
    layout = ChunkHistoryLayout(2062,32,14)
    statistics = read_tensor(bounds['statistics'])
    data = {}
    for partition in ('TRAIN','DEV'):
        name = partition.lower()+'.pt.gz'
        path = Path(exported['export_result']).parent/name
        binding = original['files'][name]
        if path.stat().st_size != binding['bytes'] or sha(path) != binding['sha256']:
            raise ValueError('Physical partition changed')
        packet = read_tensor(path)
        if packet['format'] != 'decoded_chunk_physical_trials_v1':
            raise ValueError('Require complete decoded physical trials')
        data[partition] = packet['trials']
        if len(data[partition]) != exported['partitions'][partition]['trials']:
            raise ValueError('Trial coverage differs')
        if any(row['partition'] != partition or row['policy_versions'] != [0] for row in packet['metadata']):
            raise ValueError('Partition or teacher version differs')
        count = episodes = 0
        for trial in data[partition]:
            trial_tokens(layout,**trial)
            scaled = (trial['issued']-trial['bases'])/statistics['residual_radius'].numpy()
            if not np.isfinite(scaled).all() or not (np.abs(scaled) < 1).all():
                raise ValueError('Label outside the fixed common bounds; never clip')
            count += len(trial['features']);episodes += int(trial['episode_starts'].sum())
        if (count != exported['partitions'][partition]['physical_decisions']
                or episodes != exported['partitions'][partition]['episodes']):
            raise ValueError('Physical decision/episode coverage differs')
    if (statistics['source_train_sha256'] != original['files']['train.pt.gz']['sha256']
            or statistics['split_sha256'] != original['split_sha256']):
        raise ValueError('Statistics belong to a different physical corpus')
    contract = dict(format='decoded_chunk_physical_trials_v1',task='place_phone_stand',
        eligible_methods=original['eligible_data_uses'],discount=original['bellman_discount'],
        chronological_learning_histories=False,teacher_policy_versions=[0],
        export_sha256=exported['export_result_sha256'],audit_sha256=exported['independent_audit_sha256'],
        train_sha256=original['files']['train.pt.gz']['sha256'],dev_sha256=original['files']['dev.pt.gz']['sha256'],
        statistics_file_sha256=bounds['statistics_sha256'],split_sha256=original['split_sha256'],
        motor_sha256=original['motor_sha256'],norm_stats_sha256=original['norm_stats_sha256'],
        native_batch=64,feature_contract=original['feature_contract'],residual_contract=original['residual_contract'],
        bellman_clock=original['bellman_clock'],global_margin_selected_on_dev=2,
        student_dev_was_teacher_train=True,test_tensor_loaded=False,full_source_campaign_admitted=False,
        student_implementation_sha256={name:sha(ROOT.parents[1]/'src/fluffy_bassoon/meta_rl'/name)
            for name in ('causal_decoder.py','chunk_history.py','chunk_policy.py','chunk_value.py','chunk_training.py')})
    return layout,statistics,data,contract


def checkpoint_identity(path, digest):
    saved = checked_load(path,digest,weights_only=False)
    # Separate scientific/recovery state from container serialization metadata.
    return {name:tensor_hash(saved[name]) for name in
            ('contract','statistics','learner','optimizers','updates','valid_positions','rng','value_sampler')}


def run(args):
    if sys.platform != 'linux':
        raise RuntimeError('Run model computation on allocated Linux')
    code = ROOT.parents[1]
    source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    if source != os.environ['TASK_SOURCE'] or subprocess.check_output(
            ['git','-C',str(code),'status','--porcelain','--untracked-files=no']):
        raise ValueError('Require the exact clean recorded source')
    if args.device == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise RuntimeError('CPU qualification requires explicit empty GPU visibility')
    else:
        from fluffy_bassoon.baige_gpu import mapping
        if not os.uname().nodename.startswith('job-') or not os.environ.get('LYCO_BAIGE_GPU_MAP'):
            raise RuntimeError('CUDA fitting requires actual scheduler GPU attestation')
        if len(mapping()) != 1 or torch.cuda.device_count() != 1:
            raise RuntimeError('Require the declared single allocated fitting device')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    plan = json.loads(Path(args.contract).read_bytes())
    if plan['intent'] not in ('runtime_qualification','student_fitting'):
        raise ValueError('Declare fitting intent')
    output = Path(args.output).resolve()
    if not output.is_relative_to(BASE.resolve()):
        raise ValueError('Keep fitting outputs on the project archive')
    output.mkdir()
    started = time.monotonic()
    report = dict(status='starting',source_commit=source,method=args.method,device=args.device,
        intent=plan['intent'],test_tensor_loaded=False,native_motus_or_simulator_constructed=False,
        learning_improvement_claim=False,online_deployment_qualified=False,checkpoints=[])
    write(output/'RESULT.json',report)
    try:
        layout,statistics,data,data_contract = physical_data()
        model = ChunkModelConfig(layout,**plan['model'])
        fitting = dict(plan['fitting'],method=args.method,
            mode='static' if args.method == 'static_behavior_cloning' else plan['mode'])
        if args.method == 'static_behavior_cloning':
            fitting['value_updates'] = 0
        trainer = ChunkTrainer(model,ChunkFitConfig(**fitting),statistics,data_contract,device=args.device)
        expected_contract = trainer.contract
        if args.resume:
            if not args.resume_sha256:
                raise ValueError('Declare the complete resume checkpoint checksum')
            del trainer
            gc.collect()
            trainer = ChunkTrainer.restore(args.resume,args.resume_sha256,
                expected_contract=expected_contract,device=args.device)
            report['restored'] = dict(path=args.resume,sha256=args.resume_sha256,updates=dict(trainer.updates))
        else:
            digest = trainer.save(output/'initial.pt')
            report['checkpoints'].append(dict(update=0,file='initial.pt',sha256=digest))
        report.update(status='fitting',contract=expected_contract,model_configuration=asdict(model),
            full_planned_architecture=(model.layers,model.width,model.heads,model.context_length)==(6,512,8,2048),
            starting_updates=dict(trainer.updates))
        write(output/'RESULT.json',report)
        metrics_path = output/'updates.jsonl'
        with metrics_path.open('x') as stream:
            while trainer.phase != 'complete':
                stamp = time.monotonic()
                metrics = trainer.step(data['TRAIN'],execution_batch_size=plan['execution_batch_size'])
                metrics.update(seconds=time.monotonic()-stamp,host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                stream.write(json.dumps(metrics,allow_nan=False)+'\n');stream.flush()
                count = sum(trainer.updates.values())
                boundary = count in plan['retained_update_boundaries']
                if boundary or count % plan['checkpoint_interval'] == 0 or trainer.phase == 'complete':
                    name = f'boundary-{count:06d}.pt' if boundary else 'latest.pt'
                    digest = trainer.save(output/name)
                    report['checkpoints'].append(dict(update=count,file=name,sha256=digest))
                report.update(updates=dict(trainer.updates),valid_positions=dict(trainer.valid_positions),last_metrics=metrics)
                write(output/'RESULT.json',report)
                print(json.dumps(dict(method=args.method,**metrics)),flush=True)
        last = report['checkpoints'][-1]
        report['final_checkpoint_identity'] = checkpoint_identity(output/last['file'],last['sha256'])
        report['dev_actor_loss'] = trainer.validate(data['DEV'],phase='actor',batches=1)
        if fitting['value_updates']:
            report['dev_value_loss'] = trainer.validate(data['DEV'],phase='value',batches=1)
        policy_path = output/'policy.pt'
        policy_sha = trainer.export_policy(policy_path)
        policy,_ = load_chunk_policy(policy_path,policy_sha,expected_contract=trainer.contract,device=args.device)
        for name,value in trainer.learner.actor.state_dict().items():
            if not torch.equal(value,policy.state_dict()[name]):
                raise AssertionError('Frozen export weights/statistics differ')
        report.update(status='complete_offline_'+plan['intent'],policy_sha256=policy_sha,
            frozen_export_exact=True,updates=dict(trainer.updates),valid_positions=dict(trainer.valid_positions),
            host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cuda_initialized=torch.cuda.is_initialized(),optimizer_updates_this_process=
                sum(trainer.updates.values())-sum(report['starting_updates'].values()))
        if args.device == 'cuda':
            report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--method',required=True,choices=['dit','ic_iql','static_behavior_cloning'])
    parser.add_argument('--device',required=True,choices=['cpu','cuda'])
    parser.add_argument('--resume')
    parser.add_argument('--resume-sha256')
    run(parser.parse_args())
