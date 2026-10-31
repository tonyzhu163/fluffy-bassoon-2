"""One cache04 closed-case boundary using qualified7eef publication unchanged.

The cache replays captured Gaussian values; its private/motor generator is an
explicit canonical-source replay cursor, never a replacement-noise sampler.
No per-sample resume, allocator, clock reset, simulator or paid11 encoding.
"""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import torch

from .checkpoint import rng_state,restore_rng
from .robotwin_demo_capacity import file_sha
from .robotwin_replay_diagnosis import Audit
from .robotwin_partial11_cache import publish_cursor
from .robotwin_teacher_bank import digest
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json,_fsync_directory

FORMAT='robotwin_cache04_closed_case_recovery_v1'
ROLE='private aliases motor_rng: canonical PCG64 replay anchor; no replacement draws'
CACHE_LIMIT=600
PUBLISHER_SHA='4850b43b6272ba6e8e1f13b4b017f8856f469a45ac2bc91042ff2d406c5107f8'


def capture_rng(motor):
    return dict(rng_state(),motor_rng=deepcopy(motor.bit_generator.state),private_rng=ROLE)


def restore_snapshot(saved,motor):
    if set(saved)!={'python','numpy','torch','cuda','motor_rng','private_rng'} or saved['private_rng']!=ROLE:
        raise ValueError('Complete global/private/motor replay RNG state required')
    motor.bit_generator.state=deepcopy(saved['motor_rng'])
    restore_rng(saved)


def motor_anchors(bank,*,deadline):
    entry=bank['entries'][4];row=entry['record'];audit=Audit(deadline)
    root=Path(row['episode_manifest_ref']['root']);path=audit.nested(root,row['episode_manifest_ref'])
    manifest=audit.json(dict(path=str(path),sha256=row['episode_manifest_ref']['sha256']))
    chain=[];previous=None
    for ref in manifest['chunks']:
        p=audit.nested(path.parent,ref);chunk=audit.json(dict(path=str(p),sha256=ref['sha256']))
        meta=chunk['decision_metadata'];before=meta['capacity_motor_rng_before'];after=meta['capacity_motor_rng_after']
        if previous is not None and before!=previous:raise ValueError('Canonical source motor RNG chain changed')
        for value in (before,after):
            if value.get('bit_generator')!='PCG64':raise ValueError('Exact source PCG64 replay state required')
            probe=np.random.default_rng(0);probe.bit_generator.state=deepcopy(value)
        chain.append(dict(chunk=dict(path=str(p),sha256=ref['sha256']),before=before,after=after));previous=after
    if len(chain)!=row['chunks'] or not chain:raise ValueError('Complete canonical04 motor-state ancestry required')
    return dict(chunks=chain,initial=chain[0]['before'],final=chain[-1]['after'],mode=ROLE)


def partial11_binding(request,bank,*,deadline):
    audit=Audit(deadline);cursor_ref=request['partial11_cursor'];cursor=audit.json(cursor_ref)
    refs=request['cache_receipts'];records=[]
    if not refs:raise ValueError('Bind the actual paid11 receipts; never encode them here')
    for ref in refs:
        receipt=audit.json(ref)
        if receipt.get('format')!='robotwin_fixed12_cache_receipt_v1' or receipt.get('fit_eligible') is not False:
            raise ValueError('Exact partial preparation receipts required')
        records.extend(receipt['records'])
    ids=[e['id'] for e in bank['entries'] if e['id']!='phone-04']
    if (cursor.get('format')!='robotwin_partial11_model_cache_v1' or cursor['complete_cases']!=ids
            or cursor['fit_eligible'] is not False or len(records)!=77
            or cursor.get('status')!='complete_partial11' or 'cleanup_error' not in cursor
            or cursor['cleanup_error'] is not None or cursor.get('active_case') is not None
            or cursor.get('new_robot_actions')!=0 or cursor.get('full12_eligible') is not False
            or cursor.get('frozen_after')!=bank['frozen_parameter_sha256']):raise ValueError('Actual completed PARTIAL11 cursor required')
    key=lambda r:(r['context']['teacher_id'],r['anchor'])
    if digest(sorted(records,key=key))!=digest(sorted(cursor['records'],key=key)):
        raise ValueError('Paid11 receipt records differ from their bound closed cursor')
    for identity in ids:
        rows=[r for r in records if r['context']['teacher_id']==identity];entry=next(e for e in bank['entries'] if e['id']==identity)
        expected=dict(teacher_id=identity,split=entry['split'],producer_result=entry['producer_result'],
            producer_commit=entry['producer_commit'],profile=entry['profile'],record_id=entry['record']['spec']['id'],
            episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
        if (sorted(r['anchor'] for r in rows)!=list(range(7)) or sum(r['native_actual_actions'] for r in rows)!=193
                or any(r['context']!=expected for r in rows)):raise ValueError('Paid11 identity/provenance/cost differs')
    recovery=cursor['recovery'];audit.bound(recovery['path'],recovery['sha256'])
    close_ref=request['partial11_closeout'];close=audit.json(close_ref);row=close['attempt']
    if (row['path']!=cursor['resource_attempt'] or row['status']!='finished' or row.get('slice_overrun') is not False
            or close['source_commit']!=cursor['source_commit'] or row.get('exit_code')!=0
            or close.get('exit_code')!=0 or close.get('error') is not None
            or row['charged_seconds']!=close['charged_gpu_seconds']):
        raise ValueError('Bind the actual completed paid11 allocation closure; no active writer')
    return dict(cursor=deepcopy(cursor_ref),checkpoint=deepcopy(recovery),cache_receipts=deepcopy(refs),closeout=deepcopy(close_ref))


def read_snapshot(cursor_ref,*,deadline):
    audit=Audit(deadline);state=audit.json(cursor_ref);ref=state['recovery'];audit.bound(ref['path'],ref['sha256'])
    saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
    if (state.get('format')!=FORMAT or saved.get('optimizer') is not None or saved.get('control') is not None
            or saved['state']!={k:v for k,v in state.items() if k!='recovery'}):
        raise ValueError('Exact closed-case cursor/checkpoint pair required')
    if (set(saved['global_rng'])!={'python','numpy','torch','cuda','motor_rng','private_rng'}
            or saved['global_rng'].get('private_rng')!=ROLE):
        raise ValueError('Complete global/private/motor recovery bundle required')
    return state,saved['global_rng']


def validate_lineage_head(cache_root,resume,*,deadline):
    """A completed child anywhere in this fixed lineage forbids more encoding."""
    root=Path(cache_root).resolve();audit=Audit(deadline)
    cursors=list(root.glob('*/cursor.json'))
    for path in cursors:
        state=audit.json(dict(path=str(path),sha256=file_sha(path)))
        if (path.is_symlink() or path.parent.is_symlink() or state.get('format')!=FORMAT
                or Path(state['output']).resolve()!=path.parent.resolve()
                or Path(state['binding']['cache_root']).resolve()!=root
                or state.get('boundary') not in ('before_phone04','after_phone04')):
            raise ValueError('Exact fixed-root closed-case lineage required')
        if state['boundary']=='after_phone04':
            raise ValueError('Completed cache04 lineage requires CPU reconciliation, never another GPU resume')
    if cursors and resume is None:
        raise ValueError('Existing cache04 boundary requires explicit resume/reconciliation, not a fresh600 reset')


def resume_initial(resume,binding,clock,output,*,deadline):
    validate_lineage_head(binding['cache_root'],resume,deadline=deadline)
    state,rng=read_snapshot(resume['cursor'],deadline=deadline);audit=Audit(deadline)
    if (state['binding']!=binding or state['boundary']!='before_phone04' or state['complete_cases']!=[] or state['records']!=[]
            or rng['motor_rng']!=binding['motor_anchors']['initial']):
        raise ValueError('GPU resume only the exact initial closed boundary, never a sample prefix or completed case')
    if Path(output).resolve()==Path(state['output']).resolve():raise ValueError('Fresh child identity required')
    latest_path=Path(state['output'])/'cursor.json'
    latest=audit.json(dict(path=str(latest_path),sha256=file_sha(latest_path)))
    if latest!=state:raise ValueError('Resume must use the last published closed boundary, not a stale initial copy')
    close=audit.json(resume['closeout']);row=close['attempt']
    if (row['path']!=state['resource_attempt'] or row['status']!='finished' or row.get('slice_overrun') is not False
            or close['source_commit']!=binding['source_commit'] or close.get('error') is not None
            or type(close.get('exit_code')) is not int or row.get('exit_code')!=close['exit_code']
            or type(row['charged_seconds']) is not int
            or row['charged_seconds']!=close['charged_gpu_seconds']
            or [r for r in clock['attempts'] if r['path']==row['path']]!=[row]):
        raise ValueError('Actual same-clock predecessor cost/clean closure required before resuming')
    diagnosis=audit.json(resume['diagnosis'])
    if diagnosis.get('cursor_sha256')!=resume['cursor']['sha256'] or not diagnosis.get('reason','').strip():
        raise ValueError('Bind the diagnosed interruption; no blind fresh-budget retry')
    return state,rng


def committed_receipt(cursor_ref,*,deadline):
    """CPU-only reconciliation of a completed boundary; never a reporting GPU retry."""
    state,rng=read_snapshot(cursor_ref,deadline=deadline);audit=Audit(deadline)
    if (state['boundary']!='after_phone04' or state['complete_cases']!=['phone-04']
            or rng['motor_rng']!=state['binding']['motor_anchors']['final']):raise ValueError('Complete04 boundary required')
    latest=Path(state['output'])/'cursor.json'
    if audit.json(dict(path=str(latest),sha256=file_sha(latest)))!=state:
        raise ValueError('Completed recovery must be the actually published closed boundary, not an orphan snapshot')
    raw=audit.json(state['payload_receipt'])
    if raw['records']!=state['records'] or digest(raw['bank'])!=state['binding']['bank_sha256']:
        raise ValueError('Completed cache receipt/bank differs from closed boundary')
    wanted=state['binding']['record_count']
    if (len(raw['records'])!=wanted or sorted(r['anchor'] for r in raw['records'])!=list(range(wanted))
            or any(r['context']['teacher_id']!='phone-04' for r in raw['records'])
            or sum(r['native_actual_actions'] for r in raw['records'])!=state['binding']['native_actions']):
        raise ValueError('Whole04 payload only; no prefix completion')
    root=Path(state['output']).resolve()
    for row in raw['records']:
        if not Path(row['path']).resolve().is_relative_to(root):raise ValueError('Completed payload escaped its immutable child')
        Audit(deadline).bound(row['path'],row['sha256'])
    return dict(raw,cache04_closed_case=dict(format=FORMAT,cursor=deepcopy(cursor_ref),checkpoint=deepcopy(state['recovery'])))


def durable_payload(raw,output,*,deadline):
    # A durable recovery must never point to lost child-directory entries.
    root=Path(output).resolve();directories={root,root/'payload'}
    for row in raw['records']:
        path=Path(row['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root/'payload'):
            raise ValueError('Only this fresh child owns the completed payload')
        Audit(deadline).bound(path,row['sha256'])
        with path.open('rb') as stream:os.fsync(stream.fileno())
        parent=path.parent
        while parent.is_relative_to(root):
            directories.add(parent)
            if parent==root:break
            parent=parent.parent
    for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):_fsync_directory(directory)


def run_boundary(output,binding,resource_attempt,qualification,cache_one,*,guard,deadline,resume_state=None,resume_rng=None):
    """One case only. A failure keeps the initial pointer/RNG and all orphan bytes."""
    output=Path(output);output.mkdir(exist_ok=False)
    motor=np.random.default_rng(0);motor.bit_generator.state=deepcopy(binding['motor_anchors']['initial'])
    if resume_rng is not None:restore_snapshot(resume_rng,motor)
    state=dict(format=FORMAT,boundary='before_phone04',binding=deepcopy(binding),output=str(output),
        resource_attempt=resource_attempt,model_qualification=deepcopy(qualification),complete_cases=[],records=[],
        predecessor_cursor=None if resume_state is None else deepcopy(resume_state['cursor_ref']),
        new_robot_actions=0,source_eligible=False,per_sample_prefix_resume=False)
    # Must succeed BEFORE cache_one can encode even a single input.
    publish_cursor(output,state,capture_rng(motor))
    initial_path=output/'boundary-before-phone04.json'
    _atomic_json(initial_path,deepcopy(state))
    initial_ref=dict(path=str(initial_path),sha256=file_sha(initial_path))
    try:
        guard();raw=cache_one(output/'payload')
        records=raw['records']
        if (len(records)!=binding['record_count'] or any(r['context']['teacher_id']!='phone-04' for r in records)
                or sorted(r['anchor'] for r in records)!=list(range(binding['record_count']))
                or sum(r['native_actual_actions'] for r in records)!=binding['native_actions']):
            raise ValueError('Commit only a complete original04 case')
        durable_payload(raw,output,deadline=deadline)
        payload=output/'payload'/'partial-cache.json'
        state.update(boundary='after_phone04',complete_cases=['phone-04'],records=deepcopy(records),
                     payload_receipt=dict(path=str(payload),sha256=file_sha(payload)))
        motor.bit_generator.state=deepcopy(binding['motor_anchors']['final'])
        guard();publish_cursor(output,state,capture_rng(motor))
        complete_path=output/'boundary-after-phone04.json'
        _atomic_json(complete_path,deepcopy(state))
        ref=dict(path=str(complete_path),sha256=file_sha(complete_path))
        receipt=committed_receipt(ref,deadline=deadline)
        _atomic_json(output/'partial-cache.json',receipt)
        return receipt
    except BaseException as error:
        # Never overwrite the published closed initial RNG with advanced partial
        # model state. The same publisher preserves an orphan complete snapshot
        # if its pointer publication was interrupted.
        _atomic_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            initial_cursor=initial_ref,published_cursor=dict(path=str(output/'cursor.json'),sha256=file_sha(output/'cursor.json')),
            partial_payloads_preserved=True,new_robot_actions=0))
        raise


def validate_cache_budget(clock,attempt):
    rows=[r for r in clock['attempts'] if r.get('lineage')=='fixed12-command-diagnostic/cache04']
    if (len([r for r in rows if r['path']==attempt and r['status']=='started'])!=1
            or any(r.get('lineage_seconds')!=CACHE_LIMIT for r in rows)
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in rows)>CACHE_LIMIT
            or any(r['path']!=attempt and r['status']=='started' for r in rows)):
        raise ValueError('Same original aggregate600 cache04 lineage; no sibling or budget reset')


def cache04(api,backend,bank,request,ticket,clock,output,qualification,*,source,guard,deadline):
    import inspect
    if file_sha(Path(inspect.getfile(publish_cursor)))!=PUBLISHER_SHA:
        raise ValueError('Byte-exact qualified7eef publication API required')
    validate_cache_budget(clock,ticket['resource_attempt'])
    validate_lineage_head(Path(output).parent,request.get('cache04_resume'),deadline=deadline)
    bank=api._recheck_bank(bank,deadline);api.validate_bank(bank,complete=True)
    partial=partial11_binding(request,bank,deadline=deadline);anchors=motor_anchors(bank,deadline=deadline)
    binding=dict(source_commit=source,profile_sha256=digest(request['profile']),bank_ref=deepcopy(request['bank']),
        bank_sha256=digest(bank),partial11=partial,motor_anchors=anchors,record_count=bank['entries'][4]['record']['chunks'],
        native_actions=bank['entries'][4]['record']['primitive_steps'],cache_root=str(Path(output).parent.resolve()),
        runtime_blob_sha256=deepcopy(request['runtime_blob_sha256']),recovery_contract=FORMAT,
        clock_root=ticket['clock_root'],clock_contract=deepcopy(clock['contract']),aggregate_cache_gpu_seconds=CACHE_LIMIT,
        # Operational deadlines change on continuation and are deliberately not
        # part of the scientific/resume identity; assigned below for this child.
        )
    resume=request.get('cache04_resume');old=None;saved=None
    if resume is not None:
        old,saved=resume_initial(resume,binding,clock,output,deadline=deadline)
        old['cursor_ref']=deepcopy(resume['cursor'])
    # Pass a child deadline separately so it cannot silently alter the stable
    # identity stored at either boundary.
    def case(destination):
        return api.cache_bank(bank,backend,destination,deadline=deadline,guard=guard,model_qualification_ref=qualification)
    return run_boundary(output,binding,ticket['resource_attempt'],qualification,case,guard=guard,
                        resume_state=old,resume_rng=saved,deadline=deadline)
