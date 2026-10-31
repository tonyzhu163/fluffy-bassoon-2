"""One explicit linked1800/head continuation; original400 science stays exact.

No allocator, replacement ledger, learner, checkpoint writer or optimization.
Existing owner books/closes the two new families in the ORIGINAL global clock.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from .robotwin_replay_diagnosis import Audit
from .robotwin_demo_capacity import file_sha
from .robotwin_teacher_bank import digest
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json

SCIENCE='400eed69ead8b8672686c62f33ed5531f6f60558'
PROFILE_SHA='efcf704d5844ea1a4efaad7f4862d98cb5b5b8982822bf7673056eab0d9d0891'
CACHE_SHA='5f8ada37185fb7d5add69f9eb17cb032fc63bbc90f562fea825d161ca2a9d7dd'
REQUEST_SHA='9514b69420a1389eb76a377d2c44a32fc2bbc7e14f8d708a5cd6c9382ef2341d'
TICKET_SHA={'action':'2e48bcba05d7a6a529c7dd45e54456ff8fef4f8206c8e6c4ec267efa69d1ac5a',
            'kv':'47b946353021a31084051cb1cf6f53f25a616ada4a7b163851981f9f6a74f240'}
FUNDING_SHA='35b57693ecaaa38c7ed1a0f99c097339ab3deb2ac08d08e0e2e4e8e7f40d6841'
FORMAT='robotwin_command_fit_linked_resource_v1'
PURPOSE='fixed12_original_plan_command_capacity_diagnostic_only'
FAMILY='fixed12-command-diagnostic/'
LINKED={k:FAMILY+'fit-'+k+'-linked-v1' for k in ('action','kv')}
RECOVERY={'action':(3399,'efa5e2a42ab2c55dc8abd442e205c9253964e4d1bf3247be4e9cd901ccf1470f'),
          'kv':(2655,'f839ee47ddad56399675a872ad7f9fe00c3b42d5e78b0c21bed340dfd12ab367')}
CLOCK_CONTRACT=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',
    scope_sha256='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447',
    maximum_gpu_seconds=28800,maximum_actions=300000,
    motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')


def charge(row):
    value=row.get('charged_seconds',row['reserved_seconds'])
    if type(value) is not int or value<0:raise ValueError('Nonnegative actual/reserved charge required')
    return value


def validate_funding(authority,audit):
    ref=authority['funding_authority'];fund=audit.json(ref)
    if (ref['sha256']!=FUNDING_SHA or fund['format']!='prospective_linked_command_fit_resource_extension_v1'
            or fund['maximum_additional_GPU_seconds']!=3600 or fund['per_head_new_lineage_GPU_seconds']!=1800
            or fund['new_allowed_families']!={v:1800 for v in LINKED.values()}
            or fund['global_maximum_GPU_seconds']!=28800 or fund['maximum_new_actions']!=0
            or fund['original_global_gpu_clock_root']!=authority['clock_root']
            or fund['old_lineages_caps_and_charges_immutable']!={k:{'cap':900,'actual':782} for k in LINKED}
            or fund['scientific_source']!=SCIENCE or fund['scientific_profile_sha256']!=PROFILE_SHA
            or fund['same_fixed_endpoint']!=5000 or fund['source_eligible'] is not False
            or fund['original_A1_accounting_separate']['original_ceiling']!=7200
            or fund['original_A1_accounting_separate']['old_actual']!=5189
            or fund['original_A1_accounting_separate']['original_readout_reserved']!=1500
            or fund['original_global_clock_contract_unchanged'] is not True
            or fund['original_GRANT_SCOPE_action_header_unchanged'] is not True):
        raise ValueError('Bind the exact published prospective funding; no inferred or sealed credit')


def predecessors(authority,audit):
    """Bind exact ended900 attempts, requests and durable3399/2655 checkpoints."""
    snapshot=audit.json(authority['parent_clock_snapshot']);rows=snapshot['attempts']
    if (snapshot['contract']!=CLOCK_CONTRACT or snapshot['worker_seconds']!=28800
            or any(r['status']!='finished' for r in rows) or sum(map(charge,rows))!=5189):
        raise ValueError('Immutable all-closed original5189/global28800 snapshot required')
    result={}
    if set(authority['predecessors'])!={'action','kv'}:raise ValueError('Both original heads required')
    for kind,refs in authority['predecessors'].items():
        request=audit.json(refs['request']);ticket=audit.json(refs['ticket'])
        plan=audit.json(refs['plan']);close=audit.json(refs['closeout']);progress=audit.json(refs['progress'])
        row=close['attempt'];iteration,sha=RECOVERY[kind];recovery=progress['recovery']
        if (refs['ticket']['sha256']!=TICKET_SHA[kind] or refs['request']['sha256']!=REQUEST_SHA or ticket['request_sha256']!=REQUEST_SHA
                or ticket['source_commit']!=SCIENCE or ticket['kind']!=kind or close['source_commit']!=SCIENCE
                or close['kind']!=kind or close.get('error') is not None or close['exit_code']!=row['exit_code']
                or row['path']!=ticket['resource_attempt'] or row['status']!='finished'
                or row.get('slice_overrun') is not False or row['lineage']!=FAMILY+'fit-'+kind
                or row['lineage_seconds']!=900 or row['reserved_seconds']!=900 or charge(row)!=782
                or close['charged_gpu_seconds']!=782 or [r for r in rows if r['path']==row['path']]!=[row]
                or plan['fit_head_gpu_limits']!={'action':900,'kv':900} or plan['stage_gpu_limits']['fit']!=1800
                or plan['clock_root']!=authority['clock_root'] or plan['clock_contract']!=CLOCK_CONTRACT
                or ticket['clock_root']!=authority['clock_root'] or ticket['clock_contract']!=CLOCK_CONTRACT
                or request['complete_cache_sha256']!=CACHE_SHA or progress['complete'] is not False
                or progress['iteration']!=iteration or recovery['iteration']!=iteration or recovery['sha256']!=sha
                or set(progress['milestones'])!={'0','100','500','2000'}
                or Path(recovery['path']).parent!=Path(ticket['output'])/'recoveries'):
            raise ValueError('Original request/science/cache/cap/cost/closed checkpoint binding differs')
        audit.bound(recovery['path'],sha)
        result[kind]=dict(request=request,ticket=ticket,progress=progress,
            resume={k:recovery[k] for k in ('path','sha256')})
    return snapshot,result


def validate_resource(ticket,clock,*,operation,plan,other_fit_tickets=(),_audit=None):
    """Additional3600 is separate from original7200; all share the hard28800."""
    audit=Audit(ticket['driver_deadline_unix']) if _audit is None else _audit
    ref=plan['linked_fit_continuation'];authority=audit.json(ref)
    validate_funding(authority,audit)
    if (operation!='fit' or ticket['kind'] not in LINKED or ticket['authority_receipt']!=ref
            or authority.get('format')!=FORMAT or authority['execution_source_commit']!=ticket['source_commit']
            or authority['scientific_source_commit']!=SCIENCE or authority['scientific_profile_sha256']!=PROFILE_SHA
            or authority['complete_cache_sha256']!=CACHE_SHA or authority['additional_gpu_seconds']!=3600
            or authority['additional_head_gpu_seconds']!={'action':1800,'kv':1800}
            or authority['original_A1_gpu_seconds']!=7200 or authority['readout_hold_gpu_seconds']!=1500
            or authority['maximum_global_gpu_seconds']!=28800 or authority['maximum_new_robot_actions']!=0
            or authority['source_eligible'] is not False or plan['diagnostic_purpose']!=PURPOSE
            or plan['source_eligible'] is not False or plan['admitted'] is not True
            or plan['stage_gpu_limits']['fit']!=3600 or plan['fit_head_gpu_limits']!={'action':1800,'kv':1800}
            or plan['fit_topology']!='separate_one_gpu_physical_allocations'
            or not plan['prior_cost_receipts'] or not plan['storage_admission']):
        raise ValueError('Exact prospective3600/1800-per-head linked authority required')
    expected=Path(ticket['phase_output_root'])/'gpu-clock'
    if (Path(ticket['clock_root']).resolve()!=expected.resolve() or plan['clock_root']!=ticket['clock_root']
            or authority['clock_root']!=ticket['clock_root'] or clock['contract']!=CLOCK_CONTRACT
            or plan['clock_contract']!=CLOCK_CONTRACT or clock['worker_seconds']!=28800):
        raise ValueError('Original global clock/header only; no unanchored or replacement ledger')
    snapshot,old=predecessors(authority,audit);prefix=snapshot['attempts'];rows=clock['attempts']
    if rows[:len(prefix)]!=prefix:raise ValueError('Original rows/caps/costs must remain verbatim')
    if any(r.get('lineage') not in {*LINKED.values(),FAMILY+'readout'} for r in rows[len(prefix):]):
        raise ValueError('Only the two declared linked families and conditional readout may follow')
    linked=[r for r in rows if r.get('lineage') in LINKED.values()]
    original=[r for r in rows if r not in linked]
    readout=sum(charge(r) for r in original if r.get('lineage')==FAMILY+'readout')
    if (any(r['lineage_seconds']!=1800 for r in linked) or sum(map(charge,linked))>3600
            or readout>1500 or sum(map(charge,original))+max(0,1500-readout)>7200
            or sum(map(charge,rows))>28800):
        raise ValueError('Preserve original7200/readout1500 and separate linked3600/global28800 exactly once')
    for kind,lineage in LINKED.items():
        related=[r for r in linked if r['lineage']==lineage]
        if len(related)>1 or sum(map(charge,related))>1800:
            raise ValueError('One continuation checkpoint start/head; no refill or cross-head borrowing')
    current=[r for r in linked if r['path']==ticket['resource_attempt'] and r['lineage']==LINKED[ticket['kind']]]
    if len(current)!=1 or current[0]['status']!='started':raise ValueError('Actual current linked head reservation required')
    tickets=[ticket,*other_fit_tickets];by_attempt={};hosts=set()
    for item in tickets:
        if (item['resource_attempt'] in by_attempt or item['hostname'] in hosts or item['gpu_count']!=1
                or item['kind'] not in LINKED or item['operation']!='fit'
                or item['source_commit']!=ticket['source_commit'] or item['authority_receipt']!=ref):
            raise ValueError('One actual exclusive physical allocation per head/ticket')
        by_attempt[item['resource_attempt']]=item;hosts.add(item['hostname'])
    if set(by_attempt)!={r['path'] for r in linked}:raise ValueError('Bind all already-booked linked tickets')
    if any(LINKED[by_attempt[r['path']]['kind']]!=r['lineage'] for r in linked):
        raise ValueError('Do not relabel a resource family to bypass admission')
    return authority,old


def _ast_without(source,names):
    tree=ast.parse(source);tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in names]
    return ast.dump(tree)


def validate_execution(root,old_blobs,new_blobs):
    """Actual new hashes stay honest; only CLI/resource seams differ from400."""
    own='src/fluffy_bassoon/meta_rl/robotwin_command_fit_linked.py'
    common='src/fluffy_bassoon/meta_rl/robotwin_reactive_curve.py'
    adapter='src/fluffy_bassoon/meta_rl/robotwin_command_diagnostic_curve.py'
    if set(new_blobs)!=set(old_blobs)|{own}:raise ValueError('Exact original scientific modules plus this admission module')
    for name,sha in old_blobs.items():
        before=subprocess.check_output(['git','show',SCIENCE+':'+name],cwd=root)
        import hashlib
        if hashlib.sha256(before).hexdigest()!=sha:raise ValueError('Original400 runtime byte proof differs')
        after=(Path(root)/name).read_bytes()
        if name==common:
            if _ast_without(before,{'main'})!=_ast_without(after,{'main'}):raise ValueError('400 mathematical bodies changed')
            def guard(src):
                main=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='main')
                return ast.dump(next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='guard'))
            if guard(before)!=guard(after):raise ValueError('Original time/storage guard changed')
        elif name==adapter:
            if _ast_without(before,{'main','validate_resource'})!=_ast_without(after,{'main','validate_resource'}):
                raise ValueError('Original scientific adapter/profile/fit gate changed')
            old=next(n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name=='validate_resource')
            new=next(n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef) and n.name=='validate_resource')
            new.body=new.body[1:]
            if ast.dump(old)!=ast.dump(new):raise ValueError('Existing resource admission tail changed')
        elif after!=before:raise ValueError('Protected400 scientific module bytes changed: '+name)
    for name,sha in new_blobs.items():
        if file_sha(Path(root)/name)!=sha:raise ValueError('Actual new execution runtime hash differs')


def bind_contract(contract,request,ticket,source,root,*,deadline):
    """Operational migration only; unchanged _fit still compares WHOLE contract."""
    ref=request.get('linked_fit_continuation')
    if ref is None:return contract
    audit=Audit(deadline);plan=audit.json(ticket['resource_plan_receipt'])
    if ref!=plan.get('linked_fit_continuation'):raise ValueError('Request/ticket/plan migration binding differs')
    clock=json.loads((Path(ticket['clock_root'])/'worker-time.json').read_text())
    others=[audit.json(r) for r in plan.get('other_fit_ticket_receipts',[])]
    authority,old=validate_resource(ticket,clock,operation='fit',plan=plan,other_fit_tickets=others,_audit=audit)
    previous=old[ticket['kind']]['request']
    if (source!=ticket['source_commit'] or source==SCIENCE
            or {k:v for k,v in request.items() if k not in ('resume','runtime_blob_sha256','linked_fit_continuation')}!=
               {k:v for k,v in previous.items() if k not in ('resume','runtime_blob_sha256')}
            or request['resume']!=old[ticket['kind']]['resume']):
        raise ValueError('Only exact checkpoint and operational execution hashes may change')
    validate_execution(root,previous['runtime_blob_sha256'],request['runtime_blob_sha256'])
    if file_sha(Path(root)/'outputs/robotwin-command-diagnostic-20261101-r1/PROFILE.json')!=PROFILE_SHA:
        raise ValueError('Originalefcf scientific profile bytes required')
    result=deepcopy(contract);result['source_commit']=SCIENCE;result['runtime_blob_sha256']=deepcopy(previous['runtime_blob_sha256'])
    # Exclusive fixed marker also refuses a second invocation of the same booking.
    marker=Path(ticket['output']).parent/'linked-v1-checkpoint-start.json'
    _atomic_json(marker,dict(format=FORMAT,execution_source_commit=source,scientific_source_commit=SCIENCE,
        execution_runtime_blob_sha256=request['runtime_blob_sha256'],scientific_runtime_blob_sha256=previous['runtime_blob_sha256'],
        resource_attempt=ticket['resource_attempt'],authority=ref,resume=request['resume'],output=ticket['output']))
    return result


def inspect_predecessors(authority_ref,*,deadline):
    """Optional allocated-CPU inspection of only the two small owned recoveries."""
    import torch
    from . import robotwin_command_diagnostic_bank as bank
    from . import robotwin_reactive_curve as curve
    audit=Audit(deadline);authority=audit.json(authority_ref);validate_funding(authority,audit);_,old=predecessors(authority,audit);result={}
    for kind,proof in old.items():
        request=proof['request'];saved=torch.load(proof['resume']['path'],map_location='cpu',weights_only=False);state=saved['state']
        original_plan=audit.json(authority['predecessors'][kind]['plan']);complete=audit.json(original_plan['complete_cache_file'])
        expected=dict(profile=request['profile'],source_commit=SCIENCE,complete_cache_sha256=CACHE_SHA,
            bank=request['bank'],cache_receipts=request['cache_receipts'],wide=request['wide_configuration'],
            runtime_blob_sha256=request['runtime_blob_sha256'])
        if (digest(complete)!=CACHE_SHA or state['format']!=curve.FORMAT or state['kind']!=kind
                or state['config']!=request['profile'] or state['iteration']!=RECOVERY[kind][0] or state['complete'] is not False
                or any(state['contract'].get(k)!=v for k,v in expected.items())
                or state['contract']['diagnostic_purpose']!=PURPOSE or state['contract']['source_eligible'] is not False
                or state['contract']['outcome_evidence']!=complete['bank']['outcome_evidence']
                or not isinstance(state['contract']['command_geometry'],dict)
                or state['records']!=curve.fit_records(complete,_api=bank)
                or not saved['optimizer']['state'] or saved['sampler_numpy']['bit_generator']!='PCG64'
                or not isinstance(saved['noise_torch'],torch.Tensor)
                or set(saved['global_rng'])!={'python','numpy','torch','cuda'}
                or set(state['milestones'])!={'0','100','500','2000'}):
            raise ValueError('Exact owned optimizer/sampler/private/global-RNG scientific recovery required')
        for milestone in state['milestones'].values():audit.bound(milestone['policy']['path'],milestone['policy']['sha256'])
        result[kind]=dict(iteration=state['iteration'],recovery=proof['resume'],complete=False,
            contract=digest(state['contract']),records=len(state['records']),RNG='contained, not restored on CPU')
    return dict(passed=True,heads=result,geometry_recomputed=False,new_robot_actions=0,new_SGD=0)
