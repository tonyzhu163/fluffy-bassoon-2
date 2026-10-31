"""Prospective fixed12 learning curves using the existing reactive losses/runtime.

No allocator, simulator loop, teacher relabeling or best-validation selection.
The ended capacity100 path remains unchanged. Each invocation owns one motor
context and one independently fitted head; the parent owns admission and costs.
"""
import argparse
from copy import deepcopy
import json
import math
import inspect
from pathlib import Path
import socket
import subprocess
import time
import threading
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual
from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual
from fluffy_bassoon.rl.robotwin_noise import _RoboTwinBackend
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json, _fsync_directory
from .checkpoint import rng_state, restore_rng
from .warmstart import atomic_save
from .robotwin_demo_capacity import file_sha, masked_mse, reactive_flow_loss
from .robotwin_reactive_fit import cpu, batch_from_cache, action_loss, ARM_INDICES
from .robotwin_replay_diagnosis import Audit
from . import robotwin_teacher_bank as bank_api

_MOTOR_CONTEXTS=set()
_CONTEXT_LOCK=threading.Lock()
FORMAT='robotwin_fixed12_reactive_learning_curve_v1'
WIDE_SHA='0fa57d4fa9e380353e304cc66d4f842c8ff0c6b08e0df4820e2c998b64ddfdde'
MOTOR_SHA='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb'
NORM_SHA='1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb'
PROFILE=dict(format=FORMAT,seed=20261031,updates=5000,global_batch=2,learning_rate=1e-3,
    milestones=[0,100,500,2000,5000],primary_endpoint=5000,selection='fixed_final_endpoint',
    optimizer='AdamW',weight_decay=0.,gradient_clip=1.,execution_microbatch=1,
    rank=4,kv_scale=.1,kv_layers=[-1],wide_contract_sha256=WIDE_SHA,
    whole_split={f'phone-{i:02d}':('train' if i<8 else 'validation') for i in range(12)},
    labels='captured full submitted plans; native executed masks are physical costs only',
    maximum_new_robot_actions=0,maximum_fit_output_bytes=16<<30)


def validate_profile(profile):
    if profile!=PROFILE:raise ValueError('Exact predeclared fixed5000 curve profile required')


def fit_records(sealed,*,_api=None):
    api=bank_api if _api is None else _api
    expected_format='robotwin_fixed12_complete_cache_v1' if _api is None else api.COMPLETE_FORMAT
    if _api is None:
        if (sealed.get('format')!=expected_format or sealed.get('complete') is not True
                or sealed.get('fixed12_fit_eligible') is not True or sealed.get('resource_fit_admitted') is not False):
            raise ValueError('Full12 data eligibility required; partial11 never admits a fit')
    else:api.validate_complete_cache(sealed)
    api.validate_bank(sealed['bank'],complete=True)
    rows=deepcopy(sealed['records']);seen=set();ids=set()
    for row in rows:
        context=row['context'];identity=context['teacher_id'];key=(identity,row['anchor'])
        if (identity not in PROFILE['whole_split'] or context['split']!=PROFILE['whole_split'][identity]
                or key in seen or type(row['valid_actions']) is not int or not 1<=row['valid_actions']<=32
                or type(row['native_actual_actions']) is not int
                or not 0<=row['native_actual_actions']<=row['valid_actions']):
            raise ValueError('Whole-case split, full-plan mask or native cost changed')
        seen.add(key);ids.add(identity);row['split']=context['split']
    if ids!=set(PROFILE['whole_split']):raise ValueError('Missing fixed12 identities')
    return sorted(rows,key=lambda r:(r['context']['teacher_id'],r['anchor']))


def make_head(kind,backend,wide_configuration):
    device=torch.device(backend.model.device)
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]
                               if device.type=='cuda' else []):
        torch.manual_seed(PROFILE['seed'])
        if kind=='action':
            head=WideActionResidual.from_config(wide_configuration)
            if head.contract_sha256()!=WIDE_SHA:raise ValueError('Exact prospective2rad/1.1 head required')
        elif kind=='kv':
            head=ReactiveKVResidual(public_dim=2062,kv_heads=1,head_dim=256,rank=4,layers=(-1,),scale=.1)
        else:raise ValueError('Select action or kv, never a shared joint optimizer')
    return head.to(device).eval()


def _bits_equal(a,b):
    return (a.dtype==b.dtype and a.shape==b.shape
            and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))


def parity_check(kind,head,backend,records,*,guard):
    """All actual cached native bases; no new draw, physics or teacher labels."""
    saved_rng=rng_state();checked=0
    try:
        for row in records:
            guard();batch=batch_from_cache([row],[0],backend.model.device,guard=guard)
            packet=torch.load(row['path'],map_location='cpu',weights_only=False)
            noise=packet['proposal_noise'].to(device=backend.model.device,dtype=torch.float32)[None]
            observation=SimpleNamespace(state=batch['state'])
            with torch.no_grad():
                cache=batch['cache']
                if kind=='kv':
                    edited=head(cache,batch['mask'],batch['features'])
                    if any(not _bits_equal(a,b) for pair,old in zip(edited,cache) for a,b in zip(pair,old)):
                        raise ValueError('Zero KV head changed actual native cache bits')
                    cache=edited
                targets=backend.model._sample_actions_from_prefix_cache(observation,batch['mask'],cache,noise=noise)
                base=backend.model.decode_actions(targets,observation.state)
                # Match the unchanged backend's public float32 command boundary.
                if not _bits_equal(base.float(),batch['base14']):raise ValueError('Actual cached native base parity failed')
                if kind=='action':
                    if base.dtype!=batch['base14'].dtype:
                        raise ValueError('Action-head runtime dtype differs from captured training base')
                    if not _bits_equal(head(base,batch['features']).actions,base):
                        raise ValueError('Wide zero head changed raw native command bits')
            backend._check_frozen();checked+=1
        if backend.env is not None:raise RuntimeError('Model-only parity created physics')
    finally:restore_rng(saved_rng)
    return dict(actual_cached_rows=checked,all_native_base_bits_equal=True,new_robot_actions=0)


def evaluate(kind,head,backend,records,indices,*,guard,seed):
    # Deterministic private corruption for each whole-case readout; validation
    # never consumes training sampler/noise streams or selects a checkpoint.
    vrng=np.random.default_rng(seed+3000)
    vgen=torch.Generator(device=backend.model.device).manual_seed(seed+4000)
    totals={};weight=0
    for index in indices:
        guard();batch=batch_from_cache(records,[index],backend.model.device,guard=guard)
        with torch.enable_grad():
            if kind=='kv':
                loss,flow=reactive_flow_loss(backend.model,head,batch,rng=vrng,generator=vgen,diagnostics=True)
                details={'flow_velocity_'+k:v for k,v in flow.items()}
            else:
                prediction=head(batch['base14'],batch['features']).actions
                loss=masked_mse(prediction,batch['teacher14'],batch['valid']);details={}
        with torch.no_grad():
            if kind=='kv':
                packet=torch.load(records[index]['path'],map_location='cpu',weights_only=False)
                noise=packet['proposal_noise'].to(device=backend.model.device,dtype=torch.float32)[None]
                edited=head(batch['cache'],batch['mask'],batch['features'])
                target=backend.model._sample_actions_from_prefix_cache(SimpleNamespace(state=batch['state']),
                    batch['mask'],edited,noise=noise)
                prediction=backend.model.decode_actions(target,batch['state'])
            details.update({name:float(masked_mse(prediction[...,columns],batch['teacher14'][...,columns],
                                                  batch['valid']))
                            for name,columns in [('decoded_physical14_MSE',list(range(14))),
                                                 ('decoded_arm12_MSE',ARM_INDICES),('decoded_gripper2_MSE',[6,13])]})
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite curve readout')
        n=int(batch['valid'].sum());weight+=n
        for key,value in dict(MSE=float(loss.detach()),**details).items():totals[key]=totals.get(key,0.)+value*n
    backend._check_frozen()
    return dict(valid_submitted_slots=weight,**{k:v/weight for k,v in totals.items()})


def publish(output,state,head,optimizer,rng,generator):
    """Immutable full recovery, then atomic pointer; interruption preserves old SHA."""
    directory=Path(output)/'recoveries';directory.mkdir(exist_ok=True)
    path=directory/f"update-{state['iteration']:06d}-{uuid.uuid4().hex}.pt"
    digest=atomic_save(path,dict(format=FORMAT,state=cpu(state),policy=cpu(head.state_dict()),
        optimizer=cpu(optimizer.state_dict()),sampler_numpy=deepcopy(rng.bit_generator.state),
        noise_torch=generator.get_state().cpu(),global_rng=rng_state()))
    _fsync_directory(directory);_fsync_directory(Path(output))
    reference=dict(path=str(path),sha256=digest,iteration=state['iteration'])
    _atomic_json(Path(output)/'progress.json',dict(format=FORMAT,iteration=state['iteration'],
        complete=state.get('complete',False),recovery=reference,milestones=state['milestones'],
        primary_endpoint=state.get('primary_endpoint'),new_robot_actions=0),replace=True)
    return reference


def _fit(kind,head,backend,records,output,*,guard,config,contract,resume=None):
    """Loop over existing loss/batch APIs; public caller fixes5000 and real gates."""
    output=Path(output);output.mkdir(exist_ok=False)
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=0.)
    rng=np.random.default_rng(config['seed']+(0 if kind=='kv' else 1000))
    generator=torch.Generator(device=backend.model.device).manual_seed(config['seed']+2000)
    state=dict(format=FORMAT,kind=kind,contract=deepcopy(contract),config=deepcopy(config),
        head_configuration=head.config(),records=deepcopy(records),iteration=0,milestones={},complete=False)
    if resume is not None:
        if file_sha(resume['path'])!=resume['sha256']:raise ValueError('Resume recovery bytes changed')
        saved=torch.load(resume['path'],map_location='cpu',weights_only=False);old=saved['state']
        for key in ('format','kind','contract','config','head_configuration','records'):
            if old[key]!=state[key]:raise ValueError('Resume scientific/source/cache contract changed: '+key)
        if (type(old['iteration']) is not int or not 0<=old['iteration']<config['updates']
                or old.get('complete') is not False):raise ValueError('Resume only an unfinished complete-update boundary')
        if set(old['milestones'])!={str(i) for i in config['milestones'] if i<=old['iteration']}:
            raise ValueError('Resume milestone schedule is incomplete or changed')
        for milestone in old['milestones'].values():
            ref=milestone['policy']
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('Retained milestone policy bytes changed')
        head.load_state_dict(saved['policy']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['sampler_numpy'];generator.set_state(saved['noise_torch'])
        restore_rng(saved['global_rng']);state=deepcopy(old)
    train=[i for i,r in enumerate(records) if r['split']=='train']
    validation=[i for i,r in enumerate(records) if r['split']=='validation']
    if not train or not validation:raise ValueError('Both whole-case partitions required')
    def record_milestone():
        update=state['iteration'];head.eval()
        metrics={name:evaluate(kind,head,backend,records,indices,guard=guard,seed=config['seed'])
                 for name,indices in [('train',train),('validation',validation)]}
        # Immutable policy artifact remains directly compatible with existing
        # reactive deployment's kind/config/policy loader. No best-VAL branch.
        path=output/f'endpoint-{update:06d}.pt'
        sha=atomic_save(path,dict(kind=kind,iteration=update,contract=contract,config=head.config(),
                                 policy=cpu(head.state_dict()),diagnostics=metrics))
        _fsync_directory(output)
        state['milestones'][str(update)]=dict(policy=dict(path=str(path),sha256=sha),diagnostics=metrics)
        head.train()
    if resume is None:
        record_milestone();publish(output,state,head,optimizer,rng,generator)
    head.train()
    for update in range(state['iteration']+1,config['updates']+1):
        guard();indices=rng.choice(train,size=config['global_batch'],replace=True)
        optimizer.zero_grad(set_to_none=True)
        denominator=sum(records[int(i)]['valid_actions'] for i in indices)
        total=0.
        for index in indices:
            batch=batch_from_cache(records,[index],backend.model.device,guard=guard)
            loss=(reactive_flow_loss(backend.model,head,batch,rng=rng,generator=generator)
                  if kind=='kv' else action_loss(head,batch))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fit loss; previous recovery retained')
            scaled=loss*int(batch['valid'].sum())/denominator
            scaled.backward();total+=float(scaled.detach())
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in backend.model.parameters()):raise RuntimeError('Frozen motor received gradients')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in head.parameters()):raise RuntimeError('Nonfinite head update')
        backend._check_frozen();state.update(iteration=update,last_training_loss=total,gradient_norm=float(norm))
        if update in config['milestones']:record_milestone()
        if update==config['updates']:
            state.update(complete=True,primary_endpoint=state['milestones'][str(update)]['policy'])
        guard();publish(output,state,head,optimizer,rng,generator)
    head.eval();return state


def _qualified_curve(kind,backend,bank,cache_receipts,wide_configuration,output,*,guard,deadline,contract,resume=None,_api=None,_profile=None,_validate=None):
    """Public scientific gate. Resource admission is the parent's responsibility."""
    api=bank_api if _api is None else _api
    profile=PROFILE if _profile is None else _profile
    validator=validate_profile if _validate is None else _validate
    validator(contract['profile']);guard()
    if getattr(backend,'_reactive_curve_active',False):raise ValueError('Never share one motor context across fits')
    if (backend.env is not None or backend.model.training
            or any(p.requires_grad or p.grad is not None for p in backend.model.parameters())):
        raise ValueError('Require a private lazy eval/frozen motor without gradients or physics')
    if (backend.frozen_fingerprint()!=bank_api.FROZEN
            or backend.contract['norm_stats_sha256']!=NORM_SHA):
        raise ValueError('Actual frozen parameter/normalization hash differs')
    sealed=api.seal_full_cache(bank,cache_receipts,deadline=deadline)
    if bank_api.digest(sealed)!=contract['complete_cache_sha256']:raise ValueError('Full12 cache identity differs')
    records=fit_records(sealed,_api=_api);head=make_head(kind,backend,wide_configuration)
    # Always qualify the fresh zero head, including before loading a resume.
    parity=parity_check(kind,head,backend,records,guard=guard)
    if _api is not None:
        contract=dict(contract,diagnostic_purpose=api.PURPOSE,source_eligible=False,
            outcome_evidence=deepcopy(bank['outcome_evidence']),
            command_geometry=api.geometric_floor(records,wide_configuration,guard=guard))
    backend._reactive_curve_active=True
    try:
        state=_fit(kind,head,backend,records,output,guard=guard,config=profile,contract=contract,resume=resume)
        if backend.env is not None or backend.frozen_fingerprint()!=bank_api.FROZEN:
            raise RuntimeError('Fit changed frozen motor or created physics')
        _atomic_json(Path(output)/'result.json',dict(state,parity=parity,new_robot_actions=0,
            cumulative_head_optimizer_steps=state['iteration'],target_optimizer_steps=0,closed_loop_success_measured=False,
            interpretation='Action losses use physical14 commands; KV diagnostics use normalized flow velocity, not robot success.'))
        return state
    finally:backend._reactive_curve_active=False


def fit_curve(kind,backend,bank,cache_receipts,wide_configuration,output,*,guard,deadline,contract,resume=None,
              _api=None,_profile=None,_validate=None):
    # Two wrappers around one motor are still one autograd context. Independent
    # contexts can proceed concurrently only after the parent's capacity gate.
    identity=id(backend.model)
    with _CONTEXT_LOCK:
        if identity in _MOTOR_CONTEXTS:raise ValueError('Motor context is already fitting')
        _MOTOR_CONTEXTS.add(identity)
    try:
        return _qualified_curve(kind,backend,bank,cache_receipts,wide_configuration,output,
            guard=guard,deadline=deadline,contract=contract,resume=resume,_api=_api,_profile=_profile,_validate=_validate)
    finally:
        with _CONTEXT_LOCK:_MOTOR_CONTEXTS.remove(identity)


def validate_ticket(ticket,clock,*,source,request_sha,kind,output,now,_format=FORMAT):
    """Read an already registered owner attempt; never begin/finish a clock."""
    if (ticket.get('format')!=_format or ticket.get('admitted') is not True or ticket['source_commit']!=source
            or ticket['request_sha256']!=request_sha or ticket['kind']!=kind
            or ticket['hostname']!=socket.gethostname() or ticket['gpu_count']!=1
            or ticket['private_motor_context'] is not True or ticket['maximum_robot_actions']!=0
            or Path(ticket['output']).resolve()!=Path(output).resolve()
            or clock['contract']!=ticket['clock_contract']):raise ValueError('Actual owner/source/host/clock admission required')
    expected_root=Path(ticket['phase_output_root']).resolve()/'reactive-curve-v1'/kind
    if Path(output).resolve().parent!=expected_root:
        raise ValueError('Fresh child inside the fixed per-head aggregate output root required')
    rows=[r for r in clock['attempts'] if r['path']==ticket['resource_attempt']]
    if len(rows)!=1:raise ValueError('Exactly one registered owner attempt required')
    row=rows[0];end=ticket['driver_deadline_unix'];start=row['started_unix']
    seconds=row['reserved_seconds']
    if (row['status']!='started' or type(seconds) is not int or not 180<=seconds<8*3600
            or not isinstance(end,(int,float)) or not math.isfinite(end)
            or not start<=now<end<=start+seconds-120
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in clock['attempts'])>clock['worker_seconds']):
        raise ValueError('Finite inclusive reservation/deadline/aggregate clock mismatch')
    # Parent must predeclare the concrete phase, visible closed ancestry and
    # any linked continuation. No arbitrary8h default or old-cap borrowing.
    if not ticket.get('authority_receipt') or not ticket.get('resource_plan_receipt'):
        raise ValueError('Owner must bind concrete authority and finite resource plan')


def validate_runtime_blobs(declared,root,*,_api=None,_core_bank_sha=None,_extra_runtime=()):
    # Bind loaded implementations, not merely similarly named files elsewhere
    # on PYTHONPATH. The caller's hash-bound request also pins this wrapper.
    objects=[fit_curve,WideActionResidual,(bank_api if _api is None else _api).seal_full_cache,batch_from_cache,
             reactive_flow_loss,rng_state,atomic_save,_RoboTwinBackend,ReactiveKVResidual]+list(_extra_runtime)
    for obj in objects:
        path=Path(inspect.getfile(obj)).resolve()
        relative=str(path.relative_to(root))
        if relative not in declared or file_sha(path)!=declared[relative]:
            raise ValueError('Loaded runtime blob differs from source-bound request: '+relative)
    expected={
        'src/fluffy_bassoon/meta_rl/robotwin_teacher_bank.py':('ddde178ef9a11e8eab1e886d4094daeb762aaa620cd749439ee61688309d8b4a'
            if _core_bank_sha is None else _core_bank_sha),
        'src/fluffy_bassoon/rl/robotwin_wide_action.py':'8594349bc1e417db8b6539a0bdaada26a7be5928c54482faa647e2d92055cab4'}
    if any(declared.get(k)!=v for k,v in expected.items()):
        raise ValueError('This version carries exact061 bank and de667 wide; no silent adapter migration')


def main(argv=None,*,_api=None,_profile=None,_validate=None,_core_bank_sha=None,_operation='fit',_validate_resource=None):
    profile=PROFILE if _profile is None else _profile
    validator=validate_profile if _validate is None else _validate
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','ticket','gpu-probe','model-path','norm-stats','robotwin-root','rlinf-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--ticket-sha256',required=True)
    p.add_argument('--kind',choices=['action','kv'] if _operation=='fit' else ['cache04'],required=True)
    a=p.parse_args(argv);audit=Audit(time.time()+60)
    request=audit.json(dict(path=str(a.request),sha256=a.request_sha256))
    ticket=audit.json(dict(path=str(a.ticket),sha256=a.ticket_sha256))
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Exact clean pinned runtime required')
    clock=json.loads((Path(ticket['clock_root'])/'worker-time.json').read_text())
    validate_ticket(ticket,clock,source=source,request_sha=a.request_sha256,kind=a.kind,output=a.output,now=time.time(),_format=profile['format'])
    validator(request['profile']);audit=Audit(ticket['driver_deadline_unix']-60)
    if _api is not None:
        if request.get('operation')!=_operation or ticket.get('operation')!=_operation:
            raise ValueError('Explicit diagnostic cache04/fit operation must match request and actual ticket')
        if _validate_resource is None:raise ValueError('Explicit diagnostic resource validator required')
    for key in ('authority_receipt','resource_plan_receipt'):audit.json(ticket[key])
    if _api is not None:
        actual_plan=audit.json(ticket['resource_plan_receipt'])
        for ref in actual_plan['prior_cost_receipts']:audit.json(ref)
        audit.json(actual_plan['storage_admission'])
        others=[audit.json(ref) for ref in actual_plan.get('other_fit_ticket_receipts',[])]
        _validate_resource(ticket,clock,operation=_operation,plan=actual_plan,other_fit_tickets=others)
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve()
    extra=() if _validate is None else (_validate,)
    if _api is not None:
        from .robotwin_cache04_recovery import cache04,publish_cursor,committed_receipt
        extra+= (cache04,publish_cursor,committed_receipt)
    validate_runtime_blobs(request['runtime_blob_sha256'],root,_api=_api,_core_bank_sha=_core_bank_sha,
        _extra_runtime=extra)
    for path,sha in request['runtime_blob_sha256'].items():audit.bound(root/path,sha)
    if file_sha(a.model_path/'model.safetensors')!=MOTOR_SHA or file_sha(a.norm_stats)!=NORM_SHA:
        raise ValueError('Original motor/normalization files required')
    probe=audit.json(dict(path=str(a.gpu_probe),sha256=file_sha(a.gpu_probe)))
    if probe.get('status')!='passed' or probe['hostname']!=socket.gethostname() or len(probe['devices'])!=1:
        raise ValueError('Actual one-GPU allocation probe required')
    nvml=verify_selected_cuda(torch);selected=selected_device()
    if selected['uuid']!=nvml or selected['cuda_uuid']!=probe['devices'][0]['cuda_uuid']:
        raise ValueError('Current selected GPU differs from saved allocation proof')
    parent=a.output.parent;parent.mkdir(parents=True,exist_ok=True)
    if request.get('resume') is not None:
        Path(request['resume']['path']).resolve().relative_to(parent.resolve())
    import os
    if os.statvfs(parent).f_bavail*os.statvfs(parent).f_frsize<16<<30:
        raise ValueError('Actual shared free16GiB required')
    path=parent/(a.output.name+'.write-probe')
    with path.open('xb') as stream:stream.write(b'curve');stream.flush();os.fsync(stream.fileno())
    path.unlink();_fsync_directory(parent)
    def guard():
        if time.time()+60>=ticket['driver_deadline_unix']:raise TimeoutError('Original inclusive fit clock')
        limit=(profile.get('maximum_cache04_output_bytes',16<<30) if _operation=='cache04' else 16<<30)
        if sum(f.stat().st_size for f in parent.rglob('*') if f.is_file())+(256<<20)>limit:
            raise RuntimeError('Predeclared fit output bound')
    guard();backend=None
    if _api is not None:
        # Admission is staged: start head A through this receipt BEFORE booking
        # head B. B can reference A's immutable ticket without a cyclic pair of
        # ticket->plan->other-ticket hashes, while both fits still overlap.
        _atomic_json(parent/(a.output.name+'-resource-admission.json'),dict(
            status='resource_ticket_admitted_before_model_load',operation=_operation,kind=a.kind,
            hostname=socket.gethostname(),resource_attempt=ticket['resource_attempt'],source_commit=source,
            ticket=dict(path=str(a.ticket),sha256=a.ticket_sha256),request_sha256=a.request_sha256,
            data_or_model_qualified=False))
    try:
        backend=_RoboTwinBackend(a.model_path,a.norm_stats,a.robotwin_root,a.rlinf_root,device='cuda',
            episode_limit=400,action_horizon=32,num_steps=5,output_dir=parent/(a.output.name+'-runtime'))
        contract=dict(profile=profile,source_commit=source,complete_cache_sha256=request['complete_cache_sha256'],
            bank=request['bank'],cache_receipts=request['cache_receipts'],wide=request['wide_configuration'],
            runtime_blob_sha256=request['runtime_blob_sha256'])
        if _operation=='cache04':
            if _api is None or backend.env is not None or backend.frozen_fingerprint()!=bank_api.FROZEN:
                raise ValueError('Actual lazy original frozen model required for diagnostic cache04')
            qualification=dict(format='robotwin_model_only_cache_qualification_v1',status='passed',
                hostname=socket.gethostname(),gpu_probe=dict(path=str(a.gpu_probe),sha256=file_sha(a.gpu_probe)),
                cuda_uuid=selected['cuda_uuid'],motor_sha256=MOTOR_SHA,normalization_sha256=NORM_SHA,
                frozen_parameter_sha256=backend.frozen_fingerprint())
            qpath=parent/(a.output.name+'-runtime')/'model-qualification.json'
            _atomic_json(qpath,qualification)
            cache04(_api,backend,audit.json(request['bank']),request,ticket,clock,a.output,
                dict(path=str(qpath),sha256=file_sha(qpath)),source=source,guard=guard,
                deadline=ticket['driver_deadline_unix']-60)
        else:
            fit_curve(a.kind,backend,audit.json(request['bank']),request['cache_receipts'],
                audit.json(request['wide_configuration']),a.output,guard=guard,deadline=ticket['driver_deadline_unix']-60,
                contract=contract,resume=request.get('resume'),_api=_api,_profile=_profile,_validate=_validate)
    finally:
        if backend is not None:backend.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
