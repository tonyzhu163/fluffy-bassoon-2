"""Allocated CPU metadata/source-proof deltas only; no model/resource execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from fluffy_bassoon.meta_rl import robotwin_command_fit_linked as linked
from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_curve as adapter


class MetadataAudit:
    """Synthetic metadata map, NOT evidence of actual file/tensor admission."""
    def __init__(self,values):self.values=values
    def json(self,ref):return deepcopy(self.values[ref['path']])
    def bound(self,*args):pass


def fixture(root='/phase'):
    root=Path(root);values={};clock_root=str(root/'gpu-clock');refs={};oldrows=[]
    for kind in ('action','kv'):
        row=dict(path='old-'+kind,status='finished',reserved_seconds=900,charged_seconds=782,exit_code=1,
            slice_overrun=False,lineage=linked.FAMILY+'fit-'+kind,lineage_seconds=900)
        oldrows.append(row);output=str(root/'reactive-curve-v1'/kind/'old')
        request=dict(status='original',operation='fit',profile=deepcopy(adapter.PROFILE),bank={'old':'bank'},
            cache_receipts=[{'paid':77},{'closed04':13}],complete_cache_sha256=linked.CACHE_SHA,
            wide_configuration={'same':'literal-path'},runtime_blob_sha256={'original':'oldhash'},resume=None)
        ticket=dict(source_commit=linked.SCIENCE,request_sha256=linked.REQUEST_SHA,kind=kind,
            resource_attempt=row['path'],clock_root=clock_root,clock_contract=linked.CLOCK_CONTRACT,output=output)
        plan=dict(fit_head_gpu_limits={'action':900,'kv':900},stage_gpu_limits={'fit':1800},
                  clock_root=clock_root,clock_contract=linked.CLOCK_CONTRACT)
        close=dict(attempt=row,source_commit=linked.SCIENCE,kind=kind,exit_code=1,error=None,charged_gpu_seconds=782)
        n,sha=linked.RECOVERY[kind]
        progress=dict(complete=False,iteration=n,recovery=dict(iteration=n,path=output+'/recoveries/own.pt',sha256=sha),
            milestones={str(i):{} for i in (0,100,500,2000)})
        refs[kind]={}
        for name,value in [('request',request),('ticket',ticket),('plan',plan),('closeout',close),('progress',progress)]:
            path=kind+'-'+name;values[path]=value
            refs[kind][name]=dict(path=path,sha256=linked.REQUEST_SHA if name=='request' else linked.TICKET_SHA[kind] if name=='ticket' else 'f'*64)
    snapshot=dict(contract=deepcopy(linked.CLOCK_CONTRACT),worker_seconds=28800,
        attempts=[dict(path='earlier',status='finished',charged_seconds=3625,reserved_seconds=4000),*deepcopy(oldrows)])
    values['snapshot']=snapshot
    authority=dict(format=linked.FORMAT,execution_source_commit='e'*40,scientific_source_commit=linked.SCIENCE,
        scientific_profile_sha256=linked.PROFILE_SHA,complete_cache_sha256=linked.CACHE_SHA,
        additional_gpu_seconds=3600,additional_head_gpu_seconds={'action':1800,'kv':1800},
        original_A1_gpu_seconds=7200,readout_hold_gpu_seconds=1500,maximum_global_gpu_seconds=28800,
        maximum_new_robot_actions=0,source_eligible=False,clock_root=clock_root,
        parent_clock_snapshot={'path':'snapshot','sha256':'f'*64},predecessors=refs)
    values['funding']=dict(format='prospective_linked_command_fit_resource_extension_v1',maximum_additional_GPU_seconds=3600,
        per_head_new_lineage_GPU_seconds=1800,new_allowed_families={v:1800 for v in linked.LINKED.values()},
        global_maximum_GPU_seconds=28800,maximum_new_actions=0,original_global_gpu_clock_root=clock_root,
        old_lineages_caps_and_charges_immutable={k:{'cap':900,'actual':782} for k in linked.LINKED},
        scientific_source=linked.SCIENCE,scientific_profile_sha256=linked.PROFILE_SHA,same_fixed_endpoint=5000,source_eligible=False,
        original_A1_accounting_separate=dict(original_ceiling=7200,old_actual=5189,original_readout_reserved=1500),
        original_global_clock_contract_unchanged=True,original_GRANT_SCOPE_action_header_unchanged=True)
    authority['funding_authority']={'path':'funding','sha256':linked.FUNDING_SHA}
    values['authority']=authority;authority_ref={'path':'authority','sha256':'f'*64}
    current=dict(path='new-action',status='started',reserved_seconds=1800,lineage=linked.LINKED['action'],lineage_seconds=1800)
    clock=dict(contract=deepcopy(linked.CLOCK_CONTRACT),worker_seconds=28800,attempts=deepcopy(snapshot['attempts'])+[current])
    ticket=dict(kind='action',operation='fit',source_commit='e'*40,authority_receipt=authority_ref,phase_output_root=str(root),
        clock_root=clock_root,resource_attempt='new-action',hostname='own-A',gpu_count=1,driver_deadline_unix=1e20,
        output=str(root/'reactive-curve-v1/action/fresh'),resource_plan_receipt={'path':'newplan','sha256':'f'*64})
    plan=dict(linked_fit_continuation=authority_ref,diagnostic_purpose=linked.PURPOSE,source_eligible=False,admitted=True,
        stage_gpu_limits={'fit':3600},fit_head_gpu_limits={'action':1800,'kv':1800},
        fit_topology='separate_one_gpu_physical_allocations',prior_cost_receipts=[refs['action']['closeout'],refs['kv']['closeout']],
        storage_admission={'fresh':'proof'},clock_root=clock_root,clock_contract=deepcopy(linked.CLOCK_CONTRACT))
    values['newplan']=plan
    return values,ticket,clock,plan


def test_linked_admission_retains_parent900_costs_and_separate_credit():
    values,ticket,clock,plan=fixture();before=deepcopy(clock)
    authority,old=linked.validate_resource(ticket,clock,operation='fit',plan=plan,_audit=MetadataAudit(values))
    assert clock==before and sum(linked.charge(r) for r in clock['attempts'])==5189+1800
    assert [r['lineage_seconds'] for r in clock['attempts'][1:3]]==[900,900]
    assert old['action']['resume']['sha256']==linked.RECOVERY['action'][1] and authority['additional_gpu_seconds']==3600


@pytest.mark.parametrize('fault',['old_cap','old_cost','wrong_global_clock','prefix_rewrite','unclosed_parent','credit',
    'headcap','unknown_family','wrong_family','duplicate_start','source','cache','old_request_sha','missing_other_ticket','readout_borrow','funding','old_ticket_sha'])
def test_linked_admission_refuses_drift_hidden_costs_borrow_and_duplicate_start(fault):
    values,ticket,clock,plan=fixture();a=values['authority']
    if fault=='old_cap':values['action-closeout']['attempt']['lineage_seconds']=1800
    if fault=='old_cost':values['action-closeout']['charged_gpu_seconds']=0
    if fault=='wrong_global_clock':ticket['clock_root']='/fresh/ledger'
    if fault=='prefix_rewrite':clock['attempts'][1]['charged_seconds']=0
    if fault=='unclosed_parent':values['snapshot']['attempts'][0]['status']='started'
    if fault=='credit':a['additional_gpu_seconds']=7200
    if fault=='funding':values['funding']['maximum_additional_GPU_seconds']=7200
    if fault=='old_ticket_sha':a['predecessors']['action']['ticket']['sha256']='x'*64
    if fault=='headcap':clock['attempts'][-1]['lineage_seconds']=3600
    if fault=='unknown_family':clock['attempts'].append(dict(path='unknown',lineage='new-hidden-credit',reserved_seconds=1))
    if fault=='wrong_family':clock['attempts'][-1]['lineage']=linked.FAMILY+'fit-action'
    if fault=='duplicate_start':
        second=deepcopy(clock['attempts'][-1]);second['path']='duplicate';clock['attempts'].append(second)
    if fault=='source':a['scientific_source_commit']='x'*40
    if fault=='cache':a['complete_cache_sha256']='x'*64
    if fault=='old_request_sha':a['predecessors']['action']['request']['sha256']='x'*64
    if fault=='missing_other_ticket':clock['attempts'].append(dict(path='new-kv',status='started',reserved_seconds=1800,lineage=linked.LINKED['kv'],lineage_seconds=1800))
    if fault=='readout_borrow':clock['attempts'].append(dict(path='readout',status='finished',reserved_seconds=1510,charged_seconds=1510,lineage=linked.FAMILY+'readout'))
    with pytest.raises(ValueError):linked.validate_resource(ticket,clock,operation='fit',plan=plan,_audit=MetadataAudit(values))


def test_contract_migration_changes_only_source_and_runtime_origin_and_starts_once(tmp_path,monkeypatch):
    values,ticket,clock,plan=fixture(tmp_path);root=Path(tmp_path);(root/'gpu-clock').mkdir()
    (root/'gpu-clock/worker-time.json').write_text(json.dumps(clock))
    monkeypatch.setattr(linked,'Audit',lambda deadline:MetadataAudit(values))
    monkeypatch.setattr(linked,'validate_execution',lambda *args:None)
    monkeypatch.setattr(linked,'file_sha',lambda path:linked.PROFILE_SHA)
    previous=deepcopy(values['action-request']);request=deepcopy(previous)
    request.update(linked_fit_continuation=plan['linked_fit_continuation'],runtime_blob_sha256={'actual':'newhash'},
        resume={'path':values['action-progress']['recovery']['path'],'sha256':linked.RECOVERY['action'][1]})
    base=dict(profile=request['profile'],source_commit='e'*40,complete_cache_sha256=linked.CACHE_SHA,
        bank=request['bank'],cache_receipts=request['cache_receipts'],wide=request['wide_configuration'],runtime_blob_sha256=request['runtime_blob_sha256'])
    unchanged=deepcopy(base)
    result=linked.bind_contract(base,request,ticket,'e'*40,root,deadline=1e20)
    expected=deepcopy(base);expected.update(source_commit=linked.SCIENCE,runtime_blob_sha256=previous['runtime_blob_sha256'])
    assert result==expected and base==unchanged
    marker=json.loads((root/'reactive-curve-v1/action/linked-v1-checkpoint-start.json').read_text())
    assert marker['execution_source_commit']=='e'*40 and marker['scientific_source_commit']==linked.SCIENCE
    with pytest.raises(FileExistsError):linked.bind_contract(base,request,ticket,'e'*40,root,deadline=1e20)


@pytest.mark.parametrize('field',['bank','profile','complete_cache_sha256','resume','execution_source'])
def test_contract_migration_refuses_any_scientific_or_checkpoint_drift(tmp_path,monkeypatch,field):
    values,ticket,clock,plan=fixture(tmp_path);(tmp_path/'gpu-clock').mkdir();(tmp_path/'gpu-clock/worker-time.json').write_text(json.dumps(clock))
    monkeypatch.setattr(linked,'Audit',lambda deadline:MetadataAudit(values))
    request=deepcopy(values['action-request']);request.update(linked_fit_continuation=plan['linked_fit_continuation'],
        resume={'path':values['action-progress']['recovery']['path'],'sha256':linked.RECOVERY['action'][1]})
    source='e'*40
    if field=='execution_source':source=linked.SCIENCE
    else:request[field]={'changed':True}
    with pytest.raises(ValueError):linked.bind_contract({},request,ticket,source,tmp_path,deadline=1e20)
    assert not (tmp_path/'reactive-curve-v1/action/linked-v1-checkpoint-start.json').exists()


def source_maps(root):
    old=json.loads(subprocess.check_output(['git','show',linked.SCIENCE+':outputs/robotwin-command-diagnostic-20261101-r1/RUNTIME-BLOBS.json'],cwd=root))
    names=[*old,'src/fluffy_bassoon/meta_rl/robotwin_command_fit_linked.py']
    return old,{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def test_actual_source_migration_keeps400_mathematics_and_guard():
    root=Path(__file__).resolve().parents[1];old,new=source_maps(root)
    linked.validate_execution(root,old,new)


@pytest.mark.parametrize('fault',['math','guard','adapter_profile','protected_blob'])
def test_source_migration_refuses_code_drift_even_with_updated_execution_hashes(tmp_path,monkeypatch,fault):
    root=Path(__file__).resolve().parents[1];old,new=source_maps(root)
    for name in new:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/name).read_bytes())
    paths={'math':'robotwin_reactive_curve.py','guard':'robotwin_reactive_curve.py',
           'adapter_profile':'robotwin_command_diagnostic_curve.py','protected_blob':'robotwin_teacher_bank.py'}
    name='src/fluffy_bassoon/meta_rl/'+paths[fault];path=tmp_path/name;text=path.read_text()
    before,after={'math':("error_if_nonfinite=True)","error_if_nonfinite=False)"),
                  'guard':("Original inclusive fit clock","changed clock"),
                  'adapter_profile':("maximum_cache04_output_bytes=2<<30","maximum_cache04_output_bytes=3<<30"),
                  'protected_blob':("shape!=(1,32)","shape!=(1,14)")}[fault]
    assert before in text;path.write_text(text.replace(before,after,1));new[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    actual=subprocess.check_output
    monkeypatch.setattr(linked.subprocess,'check_output',lambda args,**kw:actual(args,cwd=root))
    with pytest.raises(ValueError):linked.validate_execution(tmp_path,old,new)
