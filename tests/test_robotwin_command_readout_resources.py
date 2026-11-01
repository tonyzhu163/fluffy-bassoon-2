"""Only new/affected admission cases; synthetic receipts, allocated Linux CPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys

import pytest

from fluffy_bassoon.meta_rl import robotwin_command_readout_resources as gate

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Actual allocated Linux CPU only')
DATA=Path(__file__).resolve().parents[1]/'outputs/archive/robotwin-command-readout-linked-20261101-r1'


class SyntheticAudit:
    """Receipt lookup only. The existing Audit owns real file SHA verification."""
    def __init__(self):self.values={};self.digests={};self.counter=0
    def add(self,name,value,sha=None,path=None):
        path=path or '/synthetic-receipts/'+name+'.json'
        digest=sha or hashlib.sha256(name.encode()).hexdigest()
        self.values[path]=value;self.digests[path]=digest
        return dict(path=path,sha256=digest)
    def json(self,ref):
        if self.digests.get(ref['path'])!=ref['sha256']:raise ValueError('Synthetic immutable receipt ref changed')
        return deepcopy(self.values[ref['path']])
    def get(self,ref):return self.values[ref['path']]


def fixture():
    audit=SyntheticAudit();fund=json.loads((DATA/'FUNDING-AUTHORITY-OWNER-COPY.json').read_text())
    root=fund['original_global_gpu_clock_root'];phase=str(Path(root).parent);execution='d'*40
    historical=dict(path=root+'/historical',status='finished',reserved_seconds=3700,charged_seconds=3625,exit_code=0,slice_overrun=False)
    old_rows={kind:dict(path=root+'/old-'+kind,lineage=gate.FAMILY+'fit-'+kind,status='finished',
        lineage_seconds=900,reserved_seconds=900,charged_seconds=782,exit_code=0,slice_overrun=False) for kind in gate.LINKED}
    prefix=[historical,*old_rows.values()];parent=dict(contract=deepcopy(gate.CLOCK),worker_seconds=28800,attempts=deepcopy(prefix))
    old={};old_requests={};old_resumes={}
    for kind,row in old_rows.items():
        old_requests[kind]=dict(profile={'fixed5000':True},complete_cache_sha256=gate.CACHE_CONTENT,
            runtime_blob_sha256={'old/scientific.py':'a'*64},bank={'original12':True},resume=None)
        request=audit.add('old-request-'+kind,old_requests[kind],gate.OLD_REQUEST)
        ticket=audit.add('old-ticket-'+kind,dict(request_sha256=gate.OLD_REQUEST,source_commit=gate.SCIENCE,kind=kind,
            resource_attempt=row['path'],clock_root=root,clock_contract=deepcopy(gate.CLOCK)),gate.OLD_TICKET[kind])
        plan=audit.add('old-plan-'+kind,dict(clock_root=root,clock_contract=deepcopy(gate.CLOCK),
            fit_head_gpu_limits={'action':900,'kv':900},stage_gpu_limits={'fit':1800},complete_cache_file={'sha256':gate.CACHE_FILE}))
        close=audit.add('old-close-'+kind,dict(attempt=deepcopy(row),source_commit=gate.SCIENCE,kind=kind,error=None,
            exit_code=0,charged_gpu_seconds=782))
        resume={k:fund['resume'][kind][k] for k in ('path','sha256')};old_resumes[kind]=resume
        progress=audit.add('old-progress-'+kind,dict(complete=False,iteration=fund['resume'][kind]['complete_updates'],
            recovery=dict(resume,iteration=fund['resume'][kind]['complete_updates'])))
        old[kind]=dict(request=request,ticket=ticket,plan=plan,closeout=close,progress=progress)
    authority=dict(format=gate.LINKED_FORMAT,execution_source_commit=execution,
        funding_authority=audit.add('funding',fund,gate.FUNDING),scientific_source_commit=gate.SCIENCE,
        scientific_profile_sha256=gate.PROFILE,complete_cache_sha256=gate.CACHE_CONTENT,additional_gpu_seconds=3600,
        additional_head_gpu_seconds={'action':1800,'kv':1800},original_A1_gpu_seconds=7200,
        readout_hold_gpu_seconds=1500,maximum_global_gpu_seconds=28800,maximum_new_robot_actions=0,
        source_eligible=False,clock_root=root,parent_clock_snapshot=audit.add('parent-clock',parent),predecessors=old)
    authority_ref=audit.add('authority',authority);proof=dict(format='robotwin_command_readout_closed_linked_v1',authority=authority_ref,heads={})
    linked=[]
    for kind,actual in [('action',1100),('kv',1400)]:
        row=dict(path=root+'/linked-'+kind,lineage=gate.LINKED[kind],status='finished',lineage_seconds=1800,
            reserved_seconds=1800,charged_seconds=actual,exit_code=0,slice_overrun=False);linked.append(row)
        output=Path(phase)/'reactive-curve-v1'/kind/'linked-final'
        request_value=dict(deepcopy(old_requests[kind]),resume=deepcopy(old_resumes[kind]),
            runtime_blob_sha256={'actual/execution.py':'b'*64},linked_fit_continuation=authority_ref)
        request=audit.add('new-request-'+kind,request_value)
        plan=audit.add('new-plan-'+kind,dict(linked_fit_continuation=authority_ref,admitted=True,
            diagnostic_purpose=gate.PURPOSE,source_eligible=False,
            clock_root=root,clock_contract=deepcopy(gate.CLOCK),stage_gpu_limits={'fit':3600},
            fit_head_gpu_limits={'action':1800,'kv':1800},fit_topology='separate_one_gpu_physical_allocations'))
        ticket=audit.add('new-ticket-'+kind,dict(kind=kind,operation='fit',gpu_count=1,hostname='actual-host-'+kind,
            source_commit=execution,resource_attempt=row['path'],clock_root=root,clock_contract=deepcopy(gate.CLOCK),
            phase_output_root=phase,
            authority_receipt=authority_ref,resource_plan_receipt=plan,request_sha256=request['sha256'],output=str(output)))
        close=audit.add('new-close-'+kind,dict(attempt=deepcopy(row),exit_code=0,error=None,
            charged_gpu_seconds=actual,source_commit=execution,kind=kind))
        migration=audit.add('migration-'+kind,dict(format=gate.LINKED_FORMAT,execution_source_commit=execution,
            scientific_source_commit=gate.SCIENCE,resource_attempt=row['path'],authority=authority_ref,resume=old_resumes[kind],
            output=str(output),execution_runtime_blob_sha256=request_value['runtime_blob_sha256'],
            scientific_runtime_blob_sha256=old_requests[kind]['runtime_blob_sha256']),path=str(output.parent/'linked-v1-checkpoint-start.json'))
        admission=audit.add('resource-admission-'+kind,dict(source_commit=execution,execution_source_commit=execution,
            status='resource_ticket_admitted_before_model_load',hostname='actual-host-'+kind,
            scientific_source_commit=gate.SCIENCE,resource_attempt=row['path'],ticket=ticket,request_sha256=request['sha256'],
            execution_runtime_blob_sha256=request_value['runtime_blob_sha256'],migration=authority_ref),
            path=str(output.parent/(output.name+'-resource-admission.json')))
        proof['heads'][kind]=dict(request=request,ticket=ticket,plan=plan,closeout=close,migration=migration,resource_admission=admission)
    current=dict(path=root+'/readout',lineage=gate.FAMILY+'readout',lineage_seconds=1500,status='started',
        started_unix=100.,slice_seconds=1440,reserved_seconds=1500)
    clock=dict(contract=deepcopy(gate.CLOCK),worker_seconds=28800,attempts=deepcopy(prefix)+linked+[current])
    ticket=dict(clock_root=root,phase_output_root=phase,driver_deadline_unix=1420.)
    plan=dict(stage_gpu_limits={'readout':1500},closed_linked_fit_receipts=proof)
    return ticket,clock,plan,audit


def validate(data):
    ticket,clock,plan,audit=data;gate.validate_lineages(ticket,clock,plan,audit=audit)


def test_funded_closed_linked_families_are_admitted_once_with_distinct_execution_source():
    data=fixture();validate(data)
    assert sum(gate.charge(row) for row in data[1]['attempts'][:3])==5189
    assert sum(gate.charge(row) for row in data[1]['attempts'][3:5])==2500


@pytest.mark.parametrize('fault',['missing_proof','missing_head','funding_ref','funding_cap','parent_open','parent_charge',
    'prefix_mutation','old900cap','old782actual','old_request_ref','old_science','old_resume',
    'file_as_content','content_as_file','foreign_clock','global_drift','readout_cap','unknown_family','unknown_suffix',
    'open_linked','missing_actual_charge','linked_overrun','linked_cap_drift','duplicate_linked','missing_linked',
    'foreign_authority','wrong_head','host_alias','closeout_mismatch','failed_fit','request_rewrite','unbound_migration',
    'migration_runtime','science_relabel','missing_marker','marker_ticket','foreign_migration_path','foreign_output',
    'foreign_purpose','marker_host'])
def test_unfunded_open_unanchored_relabelled_or_cap_drift_cannot_pass(fault):
    ticket,clock,plan,audit=fixture();proof=plan['closed_linked_fit_receipts'];authority=audit.get(proof['authority'])
    refs=proof['heads']['action'];old=authority['predecessors']['action'];parent=audit.get(authority['parent_clock_snapshot'])
    if fault=='missing_proof':plan.pop('closed_linked_fit_receipts')
    elif fault=='missing_head':proof['heads'].pop('kv')
    elif fault=='funding_ref':authority['funding_authority']['sha256']='0'*64
    elif fault=='funding_cap':audit.get(authority['funding_authority'])['maximum_additional_GPU_seconds']=3601
    elif fault=='parent_open':parent['attempts'][0]['status']='started'
    elif fault=='parent_charge':parent['attempts'][0]['charged_seconds']+=1
    elif fault=='prefix_mutation':clock['attempts'][0]['reserved_seconds']+=1
    elif fault=='old900cap':audit.get(old['closeout'])['attempt']['lineage_seconds']=901
    elif fault=='old782actual':audit.get(old['closeout'])['charged_gpu_seconds']=781
    elif fault=='old_request_ref':old['request']['sha256']='0'*64
    elif fault=='old_science':audit.get(old['ticket'])['source_commit']='e'*40
    elif fault=='old_resume':audit.get(old['progress'])['recovery']['sha256']='0'*64
    elif fault=='file_as_content':authority['complete_cache_sha256']=gate.CACHE_FILE
    elif fault=='content_as_file':audit.get(old['plan'])['complete_cache_file']['sha256']=gate.CACHE_CONTENT
    elif fault=='foreign_clock':ticket['clock_root']='/replacement/gpu-clock'
    elif fault=='global_drift':clock['worker_seconds']=28801
    elif fault=='readout_cap':plan['stage_gpu_limits']['readout']=1501
    elif fault=='unknown_family':clock['attempts'].append(dict(clock['attempts'][3],lineage=gate.FAMILY+'fit-action-linked-v2'))
    elif fault=='unknown_suffix':clock['attempts'].append(dict(clock['attempts'][3],lineage='foreign/future-source'))
    elif fault=='open_linked':clock['attempts'][3]['status']='started'
    elif fault=='missing_actual_charge':clock['attempts'][3].pop('charged_seconds')
    elif fault=='linked_overrun':clock['attempts'][3]['charged_seconds']=1801
    elif fault=='linked_cap_drift':clock['attempts'][3]['lineage_seconds']=1801
    elif fault=='duplicate_linked':clock['attempts'].append(deepcopy(clock['attempts'][3]))
    elif fault=='missing_linked':clock['attempts'].pop(4)
    elif fault=='foreign_authority':audit.get(refs['ticket'])['authority_receipt']={'path':'/foreign','sha256':'0'*64}
    elif fault=='wrong_head':audit.get(refs['ticket'])['kind']='kv'
    elif fault=='host_alias':audit.get(proof['heads']['kv']['ticket'])['hostname']='actual-host-action'
    elif fault=='closeout_mismatch':audit.get(refs['closeout'])['charged_gpu_seconds']+=1
    elif fault=='failed_fit':audit.get(refs['closeout'])['exit_code']=1
    elif fault=='request_rewrite':audit.get(refs['request'])['profile']={'changed':True}
    elif fault=='unbound_migration':audit.get(refs['migration'])['authority']={'path':'/foreign','sha256':'0'*64}
    elif fault=='migration_runtime':audit.get(refs['migration'])['execution_runtime_blob_sha256']={}
    elif fault=='science_relabel':audit.get(refs['migration'])['scientific_source_commit']=authority['execution_source_commit']
    elif fault=='missing_marker':refs['resource_admission']['sha256']='0'*64
    elif fault=='marker_ticket':audit.get(refs['resource_admission'])['ticket']={'path':'/foreign','sha256':'0'*64}
    elif fault=='foreign_migration_path':refs['migration']=audit.add('foreign-migration',audit.get(refs['migration']))
    elif fault=='foreign_output':audit.get(refs['ticket'])['output']='/foreign/phase/experiment'
    elif fault=='foreign_purpose':audit.get(refs['plan'])['diagnostic_purpose']='foreign-source'
    elif fault=='marker_host':audit.get(refs['resource_admission'])['hostname']='foreign-host'
    with pytest.raises(ValueError):validate((ticket,clock,plan,audit))


def test_original_known_route_is_unchanged_and_unknown_linked_still_rejected():
    ticket,clock,plan,audit=fixture();clock['attempts']=clock['attempts'][:3]+clock['attempts'][-1:]
    plan.pop('closed_linked_fit_receipts');validate((ticket,clock,plan,audit))
    clock['attempts'].append(dict(clock['attempts'][-1],lineage=gate.FAMILY+'readout-v2'))
    with pytest.raises(ValueError):validate((ticket,clock,plan,audit))


def test_affected_full_ticket_accepts_only_after_closed_linked_proof():
    from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_readout as readout
    ticket,clock,plan,audit=fixture();phase=ticket['phase_output_root'];output=Path(phase)/'readout-final'/'experiment'
    ticket.update(format=readout.FORMAT,admitted=True,source_commit='f'*40,request_sha256='c'*64,
        hostname=socket.gethostname(),gpu_count=1,maximum_robot_actions=6000,output=str(output),
        aggregate_output_root=phase,maximum_output_bytes=64<<30,clock_contract=deepcopy(gate.CLOCK),
        resource_attempt=clock['attempts'][-1]['path'],cleanup_seconds=60,minimum_episode_admission_seconds=120)
    plan.update(clock_root=ticket['clock_root'],clock_contract=deepcopy(gate.CLOCK),admitted=True,
        source_eligible=False,diagnostic_purpose=readout.DIAGNOSTIC,prior_cost_receipts=[{'closed':True}],
        storage_admission={'actual':True},readout_proposal_sha256=readout.PROPOSAL_SHA,maximum_readout_controller_actions=6000)
    readout.validate_ticket(ticket,clock,plan,source='f'*40,request_sha='c'*64,output=output,now=150.,_audit=audit)
    clock['attempts'][3]['status']='started'
    with pytest.raises(ValueError):
        readout.validate_ticket(ticket,clock,plan,source='f'*40,request_sha='c'*64,output=output,now=150.,_audit=audit)
