"""Immutable DPT VAL targets from preselected matching TRAIN actors.

load_binding is metadata-only. SAC/JAX inference is imported only by the
allocated CPU writer. No VLA, simulator, optimizer or source-ledger mutation.
"""
import copy
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import os
import socket
import subprocess
import sys
import time

import numpy as np


FORMAT='source40_dpt_same_train_teacher_labels_v1'
METRIC='heldout_state_same_train_teacher'
NON_TARGET=('observations','executed_noise','rewards','dones','episode_starts','teacher_steps')
TRAIN_LABEL_ATOL=1e-5


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def array_sha(value):
    value=np.asarray(value)
    h=hashlib.sha256();h.update(str(value.dtype).encode()+str(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def non_target_hashes(trial):return {key:array_sha(trial[key]) for key in NON_TARGET}


def groups(admission):
    suites=('libero_spatial','libero_object','libero_goal','libero_10')
    result={split:{r['slot'].split('/')[0]:r for r in admission['source_indices'][split]} for split in ('train','validation')}
    if (admission.get('source_only') is not True or admission.get('source_data_from_targets') is not False
            or not 0<admission['cumulative_source_action_charge']<=1000000
            or len(admission['source_indices']['train'])!=40 or len(admission['source_indices']['validation'])!=8
            or set(result['train'])!={f'{suite}:{i:02d}' for suite in suites for i in range(10)}
            or set(result['validation'])!={f'{suite}:{i:02d}' for suite in suites for i in (0,1)}):
        raise ValueError('Require immutable all40/fixed8 source-only admission')
    return result


def within(deadline):
    if time.time()+5>=deadline:raise TimeoutError('Original CPU label-preparation deadline reached; retain partial files')


def member(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Source member must remain inside the selected prefix')
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Source symlink escapes its prefix')
    return path


def source_spec(row):
    path=Path(row['index_path'])
    if sha(path)!=row['index_sha256']:raise ValueError('Selected source index changed')
    index=read(path);pub=index['publications'][0];dataset=member(path.parent,pub['datasets']['dpt']['path'])
    metadata=read(dataset.with_suffix('.json'))['metadata']
    checkpoint=member(path.parent,pub['source_checkpoint'])
    if (metadata['source_actor_checkpoint_sha256']!=pub['source_checkpoint_sha256']
            or metadata['source_updates']!=row['selected_teacher_updates']):
        raise ValueError('DPT teacher differs from the selected complete source publication')
    return dict(index_path=str(path),index_sha256=row['index_sha256'],index=index,dataset=str(dataset),
        dataset_sha256=pub['datasets']['dpt']['sha256'],metadata=metadata,checkpoint_path=str(checkpoint),
        checkpoint_sha256=pub['source_checkpoint_sha256'])


def recorded_rows(spec,*,limit=None):
    """Read only selected complete episode traces; preserve actual chronology."""
    index=spec['index'];origin=Path(index['complete_prefix_snapshot']['selected_origin'])
    count=0
    for receipt in index['episode_receipts']:
        path=member(origin,receipt['trace'])
        if sha(path)!=receipt['trace_sha256']:raise ValueError('Original raw episode changed')
        with gzip.open(path,'rb') as stream:episode=pickle.load(stream)
        rows=episode['rows']
        if len(rows)!=receipt['chunks']:raise ValueError('Raw episode decision count differs')
        for row in rows:
            yield row
            count+=1
            if limit is not None and count>=limit:return


def sac_input(row):
    obs=row['sac_observation']
    if (set(obs)!={'pixels','state'} or np.asarray(obs['pixels']).shape!=(64,64,3,1)
            or np.asarray(obs['pixels']).dtype!=np.uint8 or np.asarray(obs['state']).shape!=(8,1)
            or np.asarray(obs['state']).dtype!=np.float32 or not np.isfinite(obs['state']).all()):
        raise ValueError('Require the recorded canonical pixel64/proprio8 SAC input')
    return obs


def relabel_trial(trial,rows,evaluate,*,deadline,batch_size=64):
    """Pure array/row checks with injectable deterministic actor for unit tests."""
    result=np.empty_like(trial['target_noise']);pending=[];offset=0;seen=0
    def flush():
        nonlocal offset
        if not pending:return
        within(deadline)
        obs={key:np.stack([value[key] for value in pending]) for key in ('pixels','state')}
        values=np.asarray(evaluate(obs),np.float32).reshape(len(pending),32)
        if values.shape!=result[offset:offset+len(pending)].shape or not np.isfinite(values).all():
            raise ValueError('TRAIN actor returned invalid deterministic labels')
        result[offset:offset+len(pending)]=values;offset+=len(pending);pending.clear()
    for ordinal,row in enumerate(rows):
        if ordinal>=len(result):raise ValueError('Raw episode selection contains extra decisions')
        if (not np.array_equal(row['public_features'],trial['observations'][ordinal])
                or not np.array_equal(row['executed_noise'],trial['executed_noise'][ordinal])
                or np.float32(row['reward'])!=trial['rewards'][ordinal]
                or bool(row['done'])!=bool(trial['dones'][ordinal])
                or int(row['source_updates_before'])!=int(trial['teacher_steps'][ordinal])):
            raise ValueError('Raw behavior rows do not match the immutable VAL history/order')
        pending.append(sac_input(row));seen+=1
        if len(pending)==batch_size:flush()
    flush()
    if seen!=len(result):raise ValueError('Raw episode selection omitted decisions')
    derived={key:value.copy() for key,value in trial.items()};derived['target_noise']=result
    if non_target_hashes(derived)!=non_target_hashes(trial):raise RuntimeError('Relabeling changed actual history')
    return derived


def verify_teacher_reference(teacher,train):
    if (teacher['checkpoint_path']!=train['checkpoint_path'] or teacher['checkpoint_sha256']!=train['checkpoint_sha256']
            or teacher['train_dpt_sha256']!=train['dataset_sha256']
            or teacher['source_updates']!=train['metadata']['source_updates']
            or len(teacher.get('pure_actor_sha256',''))!=64):
        raise ValueError('Label teacher must be the exact preselected TRAIN publication, not VAL actor')


def load_binding(path,expected_sha256,*,admission_path,admission_sha256):
    """Metadata/file-hash admission, without SAC/JAX imports or inference."""
    if sha(path)!=expected_sha256 or sha(admission_path)!=admission_sha256:raise ValueError('Binding/admission checksum differs')
    value,admission=read(path),read(admission_path)
    if (value.get('format')!=FORMAT or value.get('completed') is not True or value.get('metric_name')!=METRIC
            or Path(value['admission_path']).resolve()!=Path(admission_path).resolve()
            or value.get('admission_sha256')!=admission_sha256
            or value.get('source_action_charge')!=admission['cumulative_source_action_charge']
            or value.get('simulator_actions')!=0 or value.get('optimizer_updates')!=0 or value.get('VLA_inferences')!=0):
        raise ValueError('Require a complete source-only same-TRAIN-teacher binding')
    selected=groups(admission)
    expected=[r['slot'].split('/')[0] for r in admission['source_indices']['validation']]
    if len(expected)!=8 or [r['task_key'] for r in value['rows']]!=expected:raise ValueError('Binding omitted/reordered fixed8 VAL goals')
    for row in value['rows']:
        key=row['task_key'];train,val=(source_spec(selected[split][key]) for split in ('train','validation'))
        if (row['train_index_path']!=train['index_path'] or row['train_index_sha256']!=train['index_sha256']
                or row['validation_index_path']!=val['index_path'] or row['validation_index_sha256']!=val['index_sha256']
                or row['original_validation_dpt']!=dict(path=val['dataset'],sha256=val['dataset_sha256'])
                or sha(train['checkpoint_path'])!=train['checkpoint_sha256']
                or sha(train['dataset'])!=train['dataset_sha256'] or sha(val['dataset'])!=val['dataset_sha256']):
            raise ValueError('Binding source/behavior/teacher identity differs')
        verify_teacher_reference(row['label_teacher'],train)
        derived=row['derived_validation_dpt'];sidecar=read(Path(derived['path']).with_suffix('.json'))
        if (sha(derived['path'])!=derived['sha256'] or sidecar['sha256']!=derived['sha256']
                or Path(derived['sidecar_path']).resolve()!=Path(derived['path']).with_suffix('.json').resolve()
                or sha(derived['sidecar_path'])!=derived['sidecar_sha256']
                or sidecar['metadata'].get('label_teacher')!=row['label_teacher']
                or sidecar['metadata'].get('metric_name')!=METRIC
                or any(sidecar['metadata'].get(k)!=v for k,v in val['metadata'].items())
                or set(row['non_target_field_sha256'])!=set(NON_TARGET)
                or row['actor_params_unchanged'] is not True or row['train_label_replay']['passed'] is not True):
            raise ValueError('Derived labels altered behavior provenance or lack exact actor/replay proof')
    return value


def generate(admission_path,admission_sha256,output,dsrl_root,*,deadline,source_commit):
    """Actual allocated CPU execution only; caller has attested the allocation."""
    import jax
    from flax import serialization
    from .vla_history import load_dataset,save_dataset
    from .vla_libero import OfficialSAC,sac_observation,structure_digest
    if jax.default_backend()!='cpu' or any(device.platform!='cpu' for device in jax.devices()):
        raise ValueError('CPU actor backend unsupported; no GPU fallback')
    if sha(admission_path)!=admission_sha256:raise ValueError('Immutable CPU snapshot admission checksum differs')
    admission=read(admission_path);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    selected=groups(admission)
    result=dict(format=FORMAT,completed=False,metric_name=METRIC,admission_path=str(Path(admission_path).resolve()),
        admission_sha256=admission_sha256,source_action_charge=admission['cumulative_source_action_charge'],
        source_commit=source_commit,backend='cpu',jax_devices=list(map(str,jax.devices())),rows=[],
        simulator_actions=0,optimizer_updates=0,VLA_inferences=0,
        interpretation='Same TRAIN label actor, different recorded behavior/state distributions; no robot quality claim')
    write(output/'progress.json',result)
    for val_row in admission['source_indices']['validation']:
        within(deadline);started=time.monotonic();key=val_row['slot'].split('/')[0]
        train,val=source_spec(selected['train'][key]),source_spec(val_row)
        train_trials,_=load_dataset(train['dataset'],expected_sha256=train['dataset_sha256'])
        val_trials,_=load_dataset(val['dataset'],expected_sha256=val['dataset_sha256'])
        if len(train_trials)!=1 or len(val_trials)!=1 or sha(train['checkpoint_path'])!=train['checkpoint_sha256']:
            raise ValueError('Require exact single-history published source checkpoints')
        contract=train['metadata']['source_contract']
        if contract['partition']!='train' or contract['task']['task_key']!=key or contract.get('source_data_from_targets') is not False:
            raise ValueError('Label teacher must be source TRAIN only')
        learner=OfficialSAC(dsrl_root,contract['learner_seed'])
        learner.restore(train['checkpoint_path'],np.random.default_rng(0),contract)
        with Path(train['checkpoint_path']).open('rb') as stream:payload=pickle.load(stream)
        expected=structure_digest(serialization.msgpack_restore(payload['agent'])['actor']['params'])
        before=structure_digest(serialization.to_state_dict(learner.agent._actor.params))
        if expected!=before or learner.updates!=train['metadata']['source_updates']:
            raise ValueError('Loaded pure actor parameters/clock differ from the TRAIN checkpoint')
        # One actual TRAIN-row replay bounds cross-backend numerical differences.
        first=next(recorded_rows(train,limit=1));obs=sac_input(first)
        canonical=sac_observation(first['motor_observation'])
        if any(not np.array_equal(canonical[k],obs[k]) for k in obs):raise ValueError('Recorded canonical PIL preprocessing changed')
        replay=np.asarray(learner.agent.eval_actions({k:v[None] for k,v in obs.items()}),np.float32).reshape(32)
        delta=float(np.max(np.abs(replay-train_trials[0]['target_noise'][0])))
        if delta>TRAIN_LABEL_ATOL:raise ValueError(f'TRAIN label replay differs by{delta}; no silent backend equivalence claim')
        derived=relabel_trial(val_trials[0],recorded_rows(val),learner.agent.eval_actions,deadline=deadline)
        after=structure_digest(serialization.to_state_dict(learner.agent._actor.params))
        if after!=before or learner.updates!=train['metadata']['source_updates']:raise RuntimeError('Label inference changed actor/optimizer clock')
        teacher=dict(checkpoint_path=train['checkpoint_path'],checkpoint_sha256=train['checkpoint_sha256'],
            pure_actor_sha256=before,train_dpt_sha256=train['dataset_sha256'],source_updates=learner.updates,
            parameter_scope='actor.params including pixel encoder; excludes optimizer/critic/replay/RNG',digest_algorithm='vla_libero.structure_digest')
        metadata=copy.deepcopy(val['metadata']);metadata.update(label_teacher=teacher,metric_name=METRIC)
        destination=output/key.replace(':','-')/'validation-dpt.npz'
        manifest=save_dataset(destination,[derived],objective='dpt',metadata=metadata)
        reloaded,_=load_dataset(destination,expected_sha256=manifest['sha256'])
        equality=non_target_hashes(reloaded[0])==non_target_hashes(val_trials[0])
        if not equality:raise RuntimeError('Derived export changed non-target fields')
        difference=derived['target_noise']-val_trials[0]['target_noise']
        row=dict(task_key=key,train_index_path=train['index_path'],train_index_sha256=train['index_sha256'],
            validation_index_path=val['index_path'],validation_index_sha256=val['index_sha256'],
            original_validation_dpt=dict(path=val['dataset'],sha256=val['dataset_sha256']),
            derived_validation_dpt=dict(path=str(destination.resolve()),sha256=manifest['sha256'],
                sidecar_path=str(destination.with_suffix('.json').resolve()),sidecar_sha256=sha(destination.with_suffix('.json'))),label_teacher=teacher,
            non_target_field_sha256=non_target_hashes(derived),non_target_fields_exact=equality,
            actor_params_unchanged=True,train_label_replay=dict(passed=True,max_absolute_delta=delta,tolerance=TRAIN_LABEL_ATOL),
            label_seconds=time.monotonic()-started,validation_decisions=len(derived['target_noise']),
            behavior_cost_ref=val_row['slot'],teacher_full_cost_ref=selected['train'][key]['slot'],
            legacy_target_comparison=dict(max_absolute_delta=float(np.abs(difference).max()),
                mean_absolute_delta=float(np.abs(difference).mean()),rms_delta=float(np.sqrt(np.mean(difference.astype(np.float64)**2))),
                interpretation='Descriptive same recorded inputs; teacher identity plus backend floor, no robot quality inference'))
        if (sha(train['dataset'])!=train['dataset_sha256'] or sha(val['dataset'])!=val['dataset_sha256']
                or sha(train['checkpoint_path'])!=train['checkpoint_sha256']):
            raise ValueError('Original source dataset/teacher changed during read-only label preparation')
        result['rows'].append(row);write(output/'progress.json',result)
        print(json.dumps(dict(task=key,derived_rows=row['validation_decisions'],seconds=row['label_seconds'],
            pure_actor_sha256=before,train_replay_max_delta=delta)),flush=True)
        del learner,payload,train_trials,val_trials,derived,reloaded
    result['completed']=True;write(output/'binding.json',result)
    return load_binding(output/'binding.json',sha(output/'binding.json'),admission_path=admission_path,admission_sha256=admission_sha256)


def assemble_and_verify(admission_path,admission_sha256,binding_path,output,*,deadline):
    """Exercise the opt-in assembly, checking untouched AD and TRAIN arrays."""
    from .vla_history import FIELDS,load_dataset
    from .vla_source40_topup_assembly import assemble_topup
    from .vla_source40_training import load_assembly
    within(deadline)
    manifest=assemble_topup(admission_path,admission_sha256,output,
        dpt_validation_binding_path=binding_path,dpt_validation_binding_sha256=sha(binding_path))
    assembly=load_assembly(Path(output)/'assembly.json');admission=read(admission_path)
    if (manifest['cumulative_source_action_charge']!=admission['cumulative_source_action_charge']
            or manifest['additional_simulator_actions']!=0 or manifest['dpt_validation_metric']!=METRIC):
        raise ValueError('Derived assembly changed source cost or label metric')
    checked=[]
    for split,objective in (('train','ad'),('train','dpt'),('validation','ad')):
        within(deadline);artifact=assembly['datasets'][split][objective]
        trials,_=load_dataset(Path(output)/artifact['path'],expected_sha256=artifact['sha256'])
        selected=sorted(admission['source_indices'][split],key=lambda row:row['slot'].split('/')[0])
        if len(trials)!=len(selected):raise ValueError('Derived assembly changed source-history count')
        for trial,row in zip(trials,selected,strict=True):
            within(deadline);index=read(row['index_path']);item=index['publications'][0]['datasets'][objective]
            original,_=load_dataset(member(Path(row['index_path']).parent,item['path']),expected_sha256=item['sha256'])
            if len(original)!=1 or any(array_sha(trial[k])!=array_sha(original[0][k]) for k in FIELDS):
                raise ValueError('Derived assembly changed AD or TRAIN arrays')
        checked.append(dict(split=split,objective=objective,histories=len(trials),all_fields_exact=True))
    return dict(assembly_path=str(Path(output)/'assembly.json'),assembly_file_sha256=sha(Path(output)/'assembly.json'),
        assembly_sha256=assembly['assembly_sha256'],unchanged_arrays=checked,
        source_action_charge=manifest['cumulative_source_action_charge'],metric_name=METRIC)


def allocated_cpu(args):
    """Base Python3.12 attests/tests; qualified Python3.11 executes JAX CPU only."""
    from .vla_source40_cpu_qualification import CPU_ENV,cgroup_resources,require_zero_gpu
    from .vla_source40_production_plan_cpu import finish
    import sysconfig
    code=Path(__file__).resolve().parents[3]
    if (sys.platform!='linux' or sys.version_info[:2]!=(3,12) or Path(sys.prefix).resolve()!=CPU_ENV.resolve()
            or socket.gethostname()!=args.expected_hostname or not socket.gethostname().startswith('job-')
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()!=args.source
            or subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
            or any(bad in os.environ.get('PYTHONPATH','') for bad in ('cu128-overlay','cp311','python3.11'))
            or args.output.name!=args.run or sha(args.admission)!=args.admission_sha256
            or not 0<=time.time()-args.started_unix<1140):
        raise ValueError('Require fresh exact-source identified allocated CPU runtime and immutable admission')
    args.output.mkdir(parents=True,exist_ok=False)
    clock=dict(format=FORMAT,source_commit=args.source,run_id=args.run,started_unix=args.started_unix,
        status='started',maximum_cpu_worker_seconds=1200,zero_gpu_attested=False,charged_gpu_seconds=None)
    write(args.output/'cpu-worker-clock.json',clock);rc=1
    try:
        import torch
        attestation=require_zero_gpu(torch);attestation.update(hostname=args.expected_hostname,
            python=sys.executable,python_abi=sysconfig.get_config_var('SOABI'),resources=cgroup_resources())
        write(args.output/'zero-gpu-attestation.json',attestation)
        if any(v is not True for v in attestation['resources']['cgroup_matches_request'].values()):
            raise ValueError('Require actual0GPU/4CPU/16GiB allocation')
        clock.update(zero_gpu_attested=True,charged_gpu_seconds=0);write(args.output/'cpu-worker-clock.json',clock)
        remaining=args.started_unix+1100-time.time()
        tests=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            'tests/test_vla_source40_dpt_same_teacher.py','tests/test_vla_source40_dpt_label_admission.py',
            'tests/test_vla_source40_topup_assembly.py','tests/test_vla_source40_training.py']
        with (args.output/'focused-tests.log').open('x') as log:
            subprocess.run(tests,cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        model=Path('/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python')
        dsrl=Path('/mowenjie/fluffy-bassoon-2/deps/dsrl-pi0-7f48937-20261029-r1/repo')
        overlay='/mowenjie/fluffy-bassoon-2/envs/dsrl-pi0-overlay-20261029-r1'
        env=dict(os.environ,JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',CUDA_VISIBLE_DEVICES='',
            PYTHONPATH=f'{overlay}:{code}/src:{dsrl}')
        attestation_path=args.output/'zero-gpu-attestation.json'
        command=[str(model),'-m',__package__+'.vla_source40_dpt_same_teacher','generate',
            '--admission',str(args.admission),'--admission-sha256',args.admission_sha256,
            '--output',str(args.output/'generation'),'--dsrl-root',str(dsrl),'--source',args.source,
            '--attestation',str(attestation_path),'--attestation-sha256',sha(attestation_path),
            '--expected-hostname',args.expected_hostname,'--stop-unix',str(args.started_unix+1100)]
        write(args.output/'command.json',dict(command=command,model_runtime='Qualified OpenPI3.11+DSRL, JAX_PLATFORMS=cpu',
            VLA_constructed=False,simulator_actions=0,optimizer_updates=0,GPU_fallback=False))
        with (args.output/'labels.log').open('x') as log:
            subprocess.run(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                timeout=max(1,args.started_unix+1100-time.time()))
        path=args.output/'generation/binding.json'
        value=load_binding(path,sha(path),admission_path=args.admission,admission_sha256=args.admission_sha256)
        assembly=assemble_and_verify(args.admission,args.admission_sha256,path,args.output/'assembly',deadline=args.started_unix+1100)
        write(args.output/'index.json',dict(completed=True,format=FORMAT,binding_path=str(path),binding_sha256=sha(path),
            source_action_charge=value['source_action_charge'],derived_goals=len(value['rows']),metric_name=METRIC,
            derived_assembly=assembly,
            zero_gpu_attestation_sha256=sha(attestation_path),allocated_GPU_count=0,
            tests_log_sha256=sha(args.output/'focused-tests.log'),simulator_actions=0,optimizer_updates=0,VLA_inferences=0,
            label_models='Eight frozen TRAIN PixelSAC actors on CPU only; no VLA'))
        rc=0;return 0
    except BaseException as error:
        write(args.output/'failure.json',dict(completed=False,error=repr(error),partial_outputs_preserved=True,
            charged_gpu_seconds=0 if clock['zero_gpu_attested'] else None,GPU_fallback=False))
        raise
    finally:finish(args.output,rc)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    outer=sub.add_parser('run');inner=sub.add_parser('generate')
    for p in (outer,inner):
        for name in ('admission','output'):p.add_argument('--'+name,type=Path,required=True)
        for name in ('admission-sha256','source','expected-hostname'):p.add_argument('--'+name,required=True)
    outer.add_argument('--run',required=True);outer.add_argument('--started-unix',type=float,required=True)
    inner.add_argument('--dsrl-root',type=Path,required=True);inner.add_argument('--stop-unix',type=float,required=True)
    inner.add_argument('--attestation',type=Path,required=True);inner.add_argument('--attestation-sha256',required=True)
    args=parser.parse_args(argv)
    if args.operation=='run':return allocated_cpu(args)
    if (sys.platform!='linux' or sys.version_info[:2]!=(3,11)
            or Path(sys.prefix).resolve()!=Path('/mowenjie/fluffy-bassoon/envs/openpi-215abfb').resolve()
            or socket.gethostname()!=args.expected_hostname or os.environ.get('JAX_PLATFORMS')!='cpu'
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or sha(args.attestation)!=args.attestation_sha256):
        raise ValueError('Label inference requires attested allocated OpenPI3.11 CPU backend; no GPU fallback')
    evidence=read(args.attestation)
    if evidence.get('hostname')!=args.expected_hostname or evidence.get('device_count_before_mask')!=0 or evidence.get('passed') is not True:
        raise ValueError('Actual zero-GPU parent attestation differs')
    generate(args.admission,args.admission_sha256,args.output,args.dsrl_root,deadline=args.stop_unix,source_commit=args.source)
    return 0


if __name__=='__main__':raise SystemExit(main())
