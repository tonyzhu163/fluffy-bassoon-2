"""Read-only admission of the two explicitly funded, CLOSED linked fit rows.

No booking, ledger writer, model, policy or scientific-contract migration.
"""
import re
from pathlib import Path

from .robotwin_replay_diagnosis import Audit

FAMILY='fixed12-command-diagnostic/'
LINKED={kind:FAMILY+'fit-'+kind+'-linked-v1' for kind in ('action','kv')}
ORIGINAL={FAMILY+kind for kind in ('cache04','fit-action','fit-kv','readout')}
SCIENCE='400eed69ead8b8672686c62f33ed5531f6f60558'
PROFILE='efcf704d5844ea1a4efaad7f4862d98cb5b5b8982822bf7673056eab0d9d0891'
CACHE_CONTENT='5f8ada37185fb7d5add69f9eb17cb032fc63bbc90f562fea825d161ca2a9d7dd'
CACHE_FILE='65c708bf3b2c85509ed6f35150768149569558def96dd981e5b548f10a673cf8'
FUNDING='35b57693ecaaa38c7ed1a0f99c097339ab3deb2ac08d08e0e2e4e8e7f40d6841'
OLD_REQUEST='9514b69420a1389eb76a377d2c44a32fc2bbc7e14f8d708a5cd6c9382ef2341d'
OLD_TICKET={'action':'2e48bcba05d7a6a529c7dd45e54456ff8fef4f8206c8e6c4ec267efa69d1ac5a',
    'kv':'47b946353021a31084051cb1cf6f53f25a616ada4a7b163851981f9f6a74f240'}
LINKED_FORMAT='robotwin_command_fit_linked_resource_v1'
PURPOSE='fixed12_original_plan_command_capacity_diagnostic_only'
CLOCK=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',
    scope_sha256='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447',
    maximum_gpu_seconds=28800,maximum_actions=300000,
    motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
CLOCK_ROOT='/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/gpu-clock'


def charge(row):
    value=row.get('charged_seconds',row['reserved_seconds'])
    if type(value) is not int or value<0:raise ValueError('Nonnegative actual/reserved clock charge required')
    if row['status']=='finished' and 'charged_seconds' not in row:raise ValueError('Closed clock row requires actual charge')
    return value


def same_ref(a,b):
    return all(a[k]==b[k] for k in ('path','sha256'))


def validate_lineages(ticket,clock,plan,*,audit=None):
    """Original route unchanged; linked rows require this exact funded closeout."""
    rows=clock['attempts'];linked=[row for row in rows if row.get('lineage') in LINKED.values()]
    if (ticket['clock_root']!=CLOCK_ROOT or Path(ticket['phase_output_root'])!=Path(CLOCK_ROOT).parent
            or clock['contract']!=CLOCK or clock['worker_seconds']!=28800
            or any(row['status']!='finished' and row.get('lineage')!=FAMILY+'readout' for row in rows)):
        raise ValueError('Only the original28800 clock and closed non-readout work may admit physics')
    if any(row.get('lineage','').startswith(FAMILY) and row['lineage'] not in ORIGINAL|set(LINKED.values()) for row in rows):
        raise ValueError('Unknown diagnostic family is not funded')
    if not linked:
        if plan.get('closed_linked_fit_receipts') is not None:raise ValueError('Linked proof without its actual clock rows')
        return
    proof=plan.get('closed_linked_fit_receipts')
    if (not isinstance(proof,dict) or proof.get('format')!='robotwin_command_readout_closed_linked_v1'
            or set(proof.get('heads',{}))!={'action','kv'}):
        raise ValueError('Both actual closed linked fits and funding/migration receipts required')
    audit=Audit(ticket['driver_deadline_unix']) if audit is None else audit
    ref=proof['authority'];authority=audit.json(ref);fund_ref=authority['funding_authority'];fund=audit.json(fund_ref)
    root=ticket['clock_root'];execution=authority['execution_source_commit']
    if (fund_ref['sha256']!=FUNDING or fund['format']!='prospective_linked_command_fit_resource_extension_v1'
            or fund['maximum_additional_GPU_seconds']!=3600 or fund['per_head_new_lineage_GPU_seconds']!=1800
            or fund['new_allowed_families']!={v:1800 for v in LINKED.values()}
            or fund['old_lineages_caps_and_charges_immutable']!={k:{'cap':900,'actual':782} for k in LINKED}
            or fund['global_maximum_GPU_seconds']!=28800 or fund['maximum_new_actions']!=0
            or fund['scientific_source']!=SCIENCE or fund['scientific_profile_sha256']!=PROFILE
            or fund['same_fixed_endpoint']!=5000 or fund['source_eligible'] is not False
            or fund['original_A1_accounting_separate']['original_ceiling']!=7200
            or fund['original_A1_accounting_separate']['old_actual']!=5189
            or fund['original_A1_accounting_separate']['original_readout_reserved']!=1500
            or fund['original_global_clock_contract_unchanged'] is not True
            or fund['original_GRANT_SCOPE_action_header_unchanged'] is not True
            or authority['format']!=LINKED_FORMAT or not re.fullmatch('[0-9a-f]{40}',execution) or execution==SCIENCE
            or authority['scientific_source_commit']!=SCIENCE or authority['scientific_profile_sha256']!=PROFILE
            or authority['complete_cache_sha256']!=CACHE_CONTENT or authority['source_eligible'] is not False
            or authority['additional_gpu_seconds']!=3600 or authority['additional_head_gpu_seconds']!={'action':1800,'kv':1800}
            or authority['original_A1_gpu_seconds']!=7200 or authority['readout_hold_gpu_seconds']!=1500
            or authority['maximum_global_gpu_seconds']!=28800 or authority['maximum_new_robot_actions']!=0
            or authority['clock_root']!=root or fund['original_global_gpu_clock_root']!=root
            or Path(root).resolve()!= (Path(ticket['phase_output_root'])/'gpu-clock').resolve()
            or clock['contract']!=CLOCK or clock['worker_seconds']!=28800 or plan['stage_gpu_limits']['readout']!=1500):
        raise ValueError('Exact linked funding/original clock/scientific origin/readout1500 required')
    parent=audit.json(authority['parent_clock_snapshot']);prefix=parent['attempts']
    if (parent['contract']!=CLOCK or parent['worker_seconds']!=28800 or any(r['status']!='finished' for r in prefix)
            or sum(map(charge,prefix))!=5189 or rows[:len(prefix)]!=prefix
            or any(r.get('lineage') not in {*LINKED.values(),FAMILY+'readout'} for r in rows[len(prefix):])):
        raise ValueError('All-closed immutable5189 prefix only; no changed old charge or replacement lineage')
    if any(r.get('lineage') in LINKED.values() for r in prefix):raise ValueError('Linked credit cannot be hidden in old5189')
    if set(authority['predecessors'])!={'action','kv'}:raise ValueError('Both original900 predecessors required')
    hosts=set();attempts=set()
    for kind,lineage in LINKED.items():
        prior=authority['predecessors'][kind]
        old_request=audit.json(prior['request']);old_ticket=audit.json(prior['ticket'])
        old_plan=audit.json(prior['plan']);old_close=audit.json(prior['closeout']);old_progress=audit.json(prior['progress'])
        old_row=old_close['attempt'];resume={k:old_progress['recovery'][k] for k in ('path','sha256')}
        funded_resume={k:fund['resume'][kind][k] for k in ('path','sha256')}
        if (prior['request']['sha256']!=OLD_REQUEST or prior['ticket']['sha256']!=OLD_TICKET[kind]
                or old_ticket['request_sha256']!=OLD_REQUEST or old_ticket['source_commit']!=SCIENCE or old_ticket['kind']!=kind
                or old_close['source_commit']!=SCIENCE or old_close['kind']!=kind or old_close.get('error') is not None
                or old_row['lineage']!=FAMILY+'fit-'+kind or old_row['lineage_seconds']!=900 or old_row['reserved_seconds']!=900
                or old_row['status']!='finished' or old_row.get('slice_overrun') is not False or charge(old_row)!=782
                or old_close['charged_gpu_seconds']!=782 or old_close['exit_code']!=old_row['exit_code']
                or old_row['path']!=old_ticket['resource_attempt'] or [r for r in prefix if r['path']==old_row['path']]!=[old_row]
                or old_ticket['clock_root']!=root or old_ticket['clock_contract']!=CLOCK
                or old_plan['clock_root']!=root or old_plan['clock_contract']!=CLOCK
                or old_plan['fit_head_gpu_limits']!={'action':900,'kv':900} or old_plan['stage_gpu_limits']['fit']!=1800
                or old_plan['complete_cache_file']['sha256']!=CACHE_FILE or old_request['complete_cache_sha256']!=CACHE_CONTENT
                or old_progress['complete'] is not False or old_progress['iteration']!=fund['resume'][kind]['complete_updates']
                or old_progress['recovery']['iteration']!=old_progress['iteration'] or resume!=funded_resume):
            raise ValueError('Original900/782 closed predecessor, typed cache or checkpoint origin changed')
        refs=proof['heads'][kind];fit_ticket=audit.json(refs['ticket']);fit_plan=audit.json(refs['plan'])
        request=audit.json(refs['request']);close=audit.json(refs['closeout']);migration=audit.json(refs['migration'])
        admission=audit.json(refs['resource_admission']);related=[r for r in linked if r['lineage']==lineage]
        if len(related)!=1:raise ValueError('Exactly one funded closed continuation/head; no retry refill')
        row=related[0];output=Path(fit_ticket['output'])
        if (row['status']!='finished' or row.get('slice_overrun') is not False or row.get('exit_code')!=0
                or row['lineage_seconds']!=1800 or row['reserved_seconds']!=1800 or charge(row)>1800
                or close['attempt']!=row or close['exit_code']!=0 or close.get('error') is not None
                or close['charged_gpu_seconds']!=charge(row) or close['source_commit']!=execution or close['kind']!=kind
                or fit_ticket['kind']!=kind or fit_ticket['operation']!='fit' or fit_ticket['gpu_count']!=1
                or fit_ticket['source_commit']!=execution or fit_ticket['resource_attempt']!=row['path']
                or fit_ticket['phase_output_root']!=ticket['phase_output_root']
                or not output.resolve().is_relative_to(Path(ticket['phase_output_root']).resolve())
                or fit_ticket['hostname'] in hosts or row['path'] in attempts
                or fit_ticket['clock_root']!=root or fit_ticket['clock_contract']!=CLOCK
                or not same_ref(fit_ticket['authority_receipt'],ref) or not same_ref(fit_ticket['resource_plan_receipt'],refs['plan'])
                or fit_ticket['request_sha256']!=refs['request']['sha256']
                or not same_ref(fit_plan['linked_fit_continuation'],ref) or fit_plan['admitted'] is not True
                or fit_plan['diagnostic_purpose']!=PURPOSE or fit_plan['source_eligible'] is not False
                or fit_plan['clock_root']!=root or fit_plan['clock_contract']!=CLOCK
                or fit_plan['stage_gpu_limits']['fit']!=3600 or fit_plan['fit_head_gpu_limits']!={'action':1800,'kv':1800}
                or fit_plan['fit_topology']!='separate_one_gpu_physical_allocations'
                or not same_ref(request['linked_fit_continuation'],ref) or request['resume']!=resume
                or {k:v for k,v in request.items() if k not in ('resume','runtime_blob_sha256','linked_fit_continuation')}!=
                   {k:v for k,v in old_request.items() if k not in ('resume','runtime_blob_sha256')}
                or migration['format']!=LINKED_FORMAT or migration['execution_source_commit']!=execution
                or migration['scientific_source_commit']!=SCIENCE or migration['resource_attempt']!=row['path']
                or not same_ref(migration['authority'],ref) or migration['resume']!=resume or Path(migration['output'])!=output
                or migration['execution_runtime_blob_sha256']!=request['runtime_blob_sha256']
                or migration['scientific_runtime_blob_sha256']!=old_request['runtime_blob_sha256']
                or Path(refs['migration']['path'])!=output.parent/'linked-v1-checkpoint-start.json'
                or admission['source_commit']!=execution or admission['execution_source_commit']!=execution
                or admission['status']!='resource_ticket_admitted_before_model_load' or admission['hostname']!=fit_ticket['hostname']
                or admission['scientific_source_commit']!=SCIENCE or admission['resource_attempt']!=row['path']
                or not same_ref(admission['ticket'],refs['ticket']) or admission['request_sha256']!=refs['request']['sha256']
                or admission['execution_runtime_blob_sha256']!=request['runtime_blob_sha256']
                or not same_ref(admission['migration'],ref)
                or Path(refs['resource_admission']['path'])!=output.parent/(output.name+'-resource-admission.json')):
            raise ValueError('Actual linked closeout/ticket/request/migration/runtime binding differs or fit remains open')
        hosts.add(fit_ticket['hostname']);attempts.add(row['path'])
    original=[row for row in rows if row not in linked]
    readout=sum(charge(row) for row in original if row.get('lineage')==FAMILY+'readout')
    if (sum(map(charge,linked))>3600 or readout>1500
            or sum(map(charge,original))+max(0,1500-readout)>7200 or sum(map(charge,rows))>28800):
        raise ValueError('Original7200/readout1500, separate linked3600 and global28800 stay charged once')
