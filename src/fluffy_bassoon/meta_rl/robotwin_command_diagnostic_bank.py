"""All fixed12 command diagnostic, explicitly including original failed04.

This never upgrades failed behavior into a successful teacher/source bank.
Known submitted source plans are diagnostic labels on actual original inputs;
executed masks, physical costs and task outcomes remain separate and unchanged.
"""
from copy import deepcopy
from pathlib import Path
import sys

import torch

from . import robotwin_teacher_bank as core
from .robotwin_replay_diagnosis import Audit
from .robotwin_reactive_fit import batch_from_cache, ARM_INDICES
from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual

PURPOSE='fixed12_original_plan_command_capacity_diagnostic_only'
FORMAT='robotwin_fixed12_command_diagnostic_bank_v1'
COMPLETE_FORMAT='robotwin_fixed12_complete_command_diagnostic_cache_v1'
FROZEN=core.FROZEN
digest=core.digest
captured_plan=core.captured_plan


def annotations(bank):
    return dict(diagnostic_purpose=PURPOSE,source_eligible=False,fixed12_fit_eligible=False,
        source_student_admitted=False,outcome_evidence=deepcopy(bank['outcome_evidence']))


def cache_annotations(bank):
    return dict(annotations(bank),diagnostic12_fit_eligible=False,
        meaning='Diagnostic preparation cache only; original failed04 outcome stays failed; join all12 separately')


def seal_annotations(bank):
    return dict(annotations(bank),diagnostic12_fit_eligible=True,
        meaning='All12 command diagnostic data only; not a successful teacher/source/student admission')


def validate_bank(bank,*,complete=False):
    if (bank.get('format')!=FORMAT or bank.get('diagnostic_purpose')!=PURPOSE
            or bank.get('source_eligible') is not False or bank.get('fixed12_fit_eligible') is not False
            or bank.get('source_student_admitted') is not False or bank.get('diagnostic12_data_eligible') is not True
            or [e['id'] for e in bank['entries']]!=core.IDS
            or [e['split'] for e in bank['entries']]!=['train']*8+['validation']*4):
        raise ValueError('Explicit full12/8TRAIN4VAL diagnostic-only identity required')
    outcomes={}
    for entry in bank['entries']:
        row=entry['record']
        if (entry['status']!='closed_command_diagnostic' or row.get('complete') is not True
                or type(row.get('success')) is not bool or entry['plan_convention']!='original_event_h32_v1'):
            raise ValueError('Only actual closed original source-plan records; no missing/F4/subset relabeling')
        outcomes[entry['id']]=dict(success=row['success'],primitive_steps=row['primitive_steps'],chunks=row['chunks'],
            record_id=row['spec']['id'],producer_result=entry['producer_result'],producer_commit=entry['producer_commit'],
            episode_manifest_ref=row['episode_manifest_ref'])
    if outcomes['phone-04']['success'] is not False or outcomes!=bank['outcome_evidence']:
        raise ValueError('Original failed04/task outcome evidence must remain verbatim')
    return core.IDS.copy()


def assemble_bank(demo_ref,interface_ref,result_refs,*,audit):
    # Existing assembler validates all eleven successes and retains the entire
    # original event result, including its failed04 and failed controls.
    bank=core.assemble_bank(demo_ref,interface_ref,result_refs,audit=audit)
    demos=audit.json(demo_ref);episode=demos['episodes'][4];event=audit.json(result_refs['event'])
    matches=[r for r in event['records'] if r['spec']['id']=='event-teacher-phone-04']
    if len(matches)!=1:raise ValueError('Exactly original be24704 record required')
    row=matches[0];core._validate_record_spec(row,episode,require_success=False)
    if (row['success'] is not False or row['spec']['stage']!='event_aligned_teacher_execution'
            or row['spec']['method']!='event_aligned_teacher'):
        raise ValueError('Do not substitute a later04 recipe or manufacture original success')
    core._closed_manifest(row,audit)
    bank['entries'][4]=dict(id='phone-04',split='train',status='closed_command_diagnostic',record=deepcopy(row),
        original_source=deepcopy(demos['actual_whole_files']['phone-04']),producer_result=deepcopy(result_refs['event']),
        producer_commit=core.PRODUCERS['event'][0],
        profile=dict(kind='original_event_h32_v1',producer_commit=core.PRODUCERS['event'][0]),
        plan_convention='original_event_h32_v1')
    outcomes={}
    for entry in bank['entries']:
        entry['status']='closed_command_diagnostic';row=entry['record']
        outcomes[entry['id']]=dict(success=row['success'],primitive_steps=row['primitive_steps'],chunks=row['chunks'],
            record_id=row['spec']['id'],producer_result=deepcopy(entry['producer_result']),producer_commit=entry['producer_commit'],
            episode_manifest_ref=deepcopy(row['episode_manifest_ref']))
    bank.update(format=FORMAT,diagnostic_purpose=PURPOSE,source_eligible=False,fixed12_fit_eligible=False,
        source_student_admitted=False,diagnostic12_data_eligible=True,outcome_evidence=outcomes,
        scope='Original all12 command-capacity diagnostic; actual failed04 and all failure/tail provenance retained',
        label_selection='Original captured submitted source-plan labels for EVERY case; no laterF4 substitution')
    validate_bank(bank,complete=True);return bank


def _recheck_bank(bank,deadline):
    rebuilt=assemble_bank(bank['demos'],bank['interface'],bank['retained_producer_results'],audit=Audit(deadline))
    if digest(rebuilt)!=digest(bank):raise ValueError('Diagnostic bank differs from exact original source evidence')
    return rebuilt


def cache_bank(bank,backend,output,*,deadline,guard,model_qualification_ref):
    # Do not repeat the paid successful11 encoding. Existing canonical partial11
    # receipts are joined later; only failed04 is newly encoded from real inputs.
    return core.cache_bank(bank,backend,output,deadline=deadline,guard=guard,
        model_qualification_ref=model_qualification_ref,_api=sys.modules[__name__],
        _only_ids=['phone-04'],_require_success=False)


def validate_complete_cache(value):
    if (value.get('format')!=COMPLETE_FORMAT or value.get('complete') is not True
            or value.get('diagnostic_purpose')!=PURPOSE or value.get('diagnostic12_fit_eligible') is not True
            or value.get('fixed12_fit_eligible') is not False or value.get('source_eligible') is not False
            or value.get('source_student_admitted') is not False or value.get('resource_fit_admitted') is not False
            or value.get('outcome_evidence')!=value['bank']['outcome_evidence']):
        raise ValueError('Explicit complete diagnostic cache only; no source/teacher success promotion')
    validate_bank(value['bank'],complete=True)


def seal_full_cache(bank,cache_receipts,*,deadline):
    value=core.seal_full_cache(bank,cache_receipts,deadline=deadline,_api=sys.modules[__name__])
    validate_complete_cache(value);return value


def geometric_floor(records,wide_configuration,*,guard):
    """Raw physical-command box floor, not task success or actuator dynamics."""
    with torch.random.fork_rng(devices=[]):head=WideActionResidual.from_config(wide_configuration)
    if head.contract_sha256()!='0fa57d4fa9e380353e304cc66d4f842c8ff0c6b08e0df4820e2c998b64ddfdde':
        raise ValueError('Exact prospective wide2/1.1 geometry required')
    totals={s:dict(squared=0.,valid_slots=0,outside=0) for s in ('train','validation')}
    for row in records:
        guard();batch=batch_from_cache([row],[0],torch.device('cpu'),guard=guard)
        base=batch['base14'];target=batch['teacher14'];valid=batch['valid']
        if (not torch.isfinite(base).all() or not torch.isfinite(target).all()
                or (base[...,ARM_INDICES]<head.arm_lower).any() or (base[...,ARM_INDICES]>head.arm_upper).any()):
            raise ValueError('Actual finite native base/target and source arm headroom required')
        lower=base-head.bounds;upper=base+head.bounds
        lower[...,ARM_INDICES]=torch.maximum(lower[...,ARM_INDICES],head.arm_lower)
        upper[...,ARM_INDICES]=torch.minimum(upper[...,ARM_INDICES],head.arm_upper)
        if (lower>upper).any():raise ValueError('Native base violates actual arm headroom')
        closest=torch.maximum(lower,torch.minimum(upper,target));error=(target-closest).square()[valid]
        t=totals[row['split']];t['squared']+=float(error.sum());t['valid_slots']+=int(valid.sum())
        t['outside']+=int((error>0).sum())
    return dict(kind='wide2rad_1p1_raw_gripper_with_actual_arm_headroom_box_floor',
        units='decoded physical14 command squared units',native_actuator_clip_applied=False,
        partitions={s:dict(valid_submitted_slots=t['valid_slots'],outside_coordinate_count=t['outside'],
            attainable_command_MSE_lower_bound=t['squared']/(14*t['valid_slots'])) for s,t in totals.items()},
        meaning='Diagnostic geometric reachability bound only; neither expertise nor closed-loop success')
