"""Fixed12 teacher-bank assembly and captured-plan cache, never a launcher.

The original event helpers remain immutable. This prospective interface permits
partial cache preparation, but only the unchanged full8TRAIN/4VAL bank can become
fit-eligible. It never schedules, resets, steps, fits or owns an action ledger.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .robotwin_demo_capacity import file_sha, validate_demo_manifest, read_episode, transformed_target
from .robotwin_replay_diagnosis import Audit, validate_reset_identity
from .robotwin_event_schedule import build_schedule, scheduled_request
from .robotwin_reactive_fit import cpu
from .warmstart import atomic_save
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device

FORMAT = 'robotwin_fixed12_captured_teacher_bank_v1'
DEMOS_SHA = 'a80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4'
INTERFACE_SHA = 'c7a6dbf1bee96f5984f9a708c732fd81d4f21c0bbccbb5b2dca4922d6c728ceb'
PRODUCERS = {
    'event': ('be247676c1217d9e046091c52cac2b7d0f2d3014',
              'ba775b4fcbb3029120927ecd59461fabb1cccd7e95326f8d9f3a6f731ea77d3f'),
    'suffix': ('aefd5563b128cb7ea8e017d4664e1df0ba22a160',
               '73bd6483cd5abf8093fc46d8eb5fb116b745d2010e1e922b9f7d658d4625cea6'),
}
IDS = [f'phone-{i:02d}' for i in range(12)]
ARM = [0,1,2,3,4,5,7,8,9,10,11,12]
E_FORMAT='robotwin_phone04_prelift_closure_pair_v1'
E_SOURCE='374ee8958f7cabcd54d4401057a604c6cc6ae51a'
E_PROFILE_SHA='0072b8c2c8239fb28d2cee7999f6614c770cd4c11ade6404c6fd285751a77ba1'
FROZEN='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _closed(result):
    actions=result['actions']
    if (result.get('pending') is not None or result.get('cleanup_error') is not None
            or actions['actual_actions']!=actions['charged_actions']
            or actions['pending_actions'] or actions['uncertain_actions'] or actions['unresolved_decisions']):
        raise ValueError('Only closed canonical producer costs; never refund or hide an attempt')


def _validate_record_spec(row,episode,*,require_success):
    spec=row['spec'];ordinal=episode['canonical_index']
    if (row.get('complete') is not True or type(row.get('success')) is not bool
            or (require_success and row['success'] is not True)
            or spec['teacher_id']!=episode['id'] or spec['task']!='place_phone_stand' or spec['horizon']!=400
            or spec['case']!=ordinal or spec['reset_seed']!=episode['reset_seed']
            or spec['expected_teacher_seed']!=episode['reset_seed']
            or spec['paired_instruction']!=[episode['instruction']]
            or spec['motor_seed']!=20261201+ordinal or spec['arm']!='action' or spec['coefficients']!=[0.]*4
            or row['reset_info']['actual_seed']!=episode['reset_seed']
            or row['reset_info']['instruction']!=[episode['instruction']]
            or not row.get('episode_manifest_ref') or type(row['primitive_steps']) is not int
            or not 0<row['primitive_steps']<=400):
        raise ValueError('Failed/foreign teacher task/seed/text/motor/profile cannot enter fixed12')


def _closed_manifest(row,audit):
    ref=row['episode_manifest_ref'];path=audit.nested(Path(ref['root']),ref)
    manifest=audit.json(dict(path=str(path),sha256=ref['sha256']))
    reset_path=path.with_name('reset.json')
    reset=audit.json(dict(path=str(reset_path),sha256=file_sha(reset_path)))
    validate_reset_identity(reset,row['reset_info'],manifest)
    if (manifest['pending_decision'] is not None or manifest['completed_chunks']!=row['chunks']
            or len(manifest['chunks'])!=row['chunks'] or not manifest['chunks']
            or manifest['executed_primitive_steps']!=row['primitive_steps']):
        raise ValueError('Teacher closed-manifest counters/pending state differ')
    last_ref=manifest['chunks'][-1];last_path=audit.nested(path.parent,last_ref)
    last=audit.json(dict(path=str(last_path),sha256=last_ref['sha256']))
    if (last['status']!='complete' or last['episode_id']!=manifest['episode_id']
            or last['context']['trial_id']!=row['spec']['id']
            or last['chunk_index']!=row['chunks'] or last['learning_result']['budget_truncated']
            or not (last['learning_result']['terminated'] or last['learning_result']['truncated'])
            or last['native_result']['success'] is not row['success']):
        raise ValueError('Teacher actual final native chunk is not the declared closed outcome')
    return path,manifest


def _bind_current_cuda(qualification,probe):
    # Existing Baige check joins actual CUDA PCI/UUID to the current scheduler
    # selection. Saved same-host receipts alone are not allocation proof.
    current_nvml=verify_selected_cuda(torch);current=selected_device()
    current_cuda=str(torch.cuda.get_device_properties(0).uuid)
    if (current['uuid']!=current_nvml or current['cuda_uuid']!=current_cuda
            or qualification['cuda_uuid']!=current_cuda
            or probe['devices'][0]['cuda_uuid']!=current_cuda
            or probe['devices'][0]['uuid']!=current_nvml):
        raise ValueError('Saved model/probe GPU differs from current selected CUDA allocation')
    return current_cuda


def assemble_bank(demo_ref, interface_ref, result_refs, *, audit, accepted04_ref=None):
    """Read sealed producer records; absent04 stays explicitly PENDING.

    accepted04_ref is the source owner's separately bound prospective selection
    contract, not a success flag. It binds exact E result/source/spec/profile and
    captured-plan convention. Its wire schema must match the E producer.
    """
    if demo_ref['sha256']!=DEMOS_SHA or interface_ref['sha256']!=INTERFACE_SHA:
        raise ValueError('Keep exact a80 demos and c7a6 native motor/norm/source conventions')
    demos=validate_demo_manifest(audit.json(demo_ref));interface=audit.json(interface_ref)
    if ([r['id'] for r in demos['episodes']]!=IDS
            or [r['split'] for r in demos['episodes']]!=['train']*8+['validation']*4
            or demos['motor_sha256']!=interface['motor_sha256']
            or demos['normalization_sha256']!=interface['normalization_sha256']):
        raise ValueError('Original12 identities/8TRAIN4VAL/motor/norm differ')
    if set(result_refs)!={'event','suffix'}:raise ValueError('Bind both original sealed producer results, including failed04')
    producers={}
    for name,(source,sha) in PRODUCERS.items():
        ref=result_refs[name]
        if ref['sha256']!=sha:raise ValueError('Foreign original producer result')
        value=audit.json(ref);_closed(value)
        if value['source_commit']!=source:raise ValueError('Foreign source producer')
        producers[name]=value
    if producers['event']['demos_sha256']!=DEMOS_SHA or producers['suffix']['plan']['predecessor_result_sha256']!=PRODUCERS['event'][1]:
        raise ValueError('Producer ancestry/demo binding changed')
    accepted=None;profile=None;selected04=None
    if (producers['event']['frozen_motor_sha256']!=FROZEN or producers['suffix']['frozen_after']!=FROZEN):
        raise ValueError('Original frozen parameter identity differs')
    if accepted04_ref is not None:
        accepted=audit.json(accepted04_ref)
        if (accepted.get('format')!='robotwin_fixed12_accepted04_profile_v1'
                or accepted.get('source_commit')!=E_SOURCE or accepted['profile']['sha256']!=E_PROFILE_SHA):
            raise ValueError('Explicit source-owner04 profile/result/source binding required')
        profile=audit.json(accepted['profile'])
        if (profile['format']!=E_FORMAT or profile['demos_sha256']!=DEMOS_SHA
                or profile['conditions']!=['hold_current','close_endpoint']
                or profile['bank_candidate_priority']!=['close_endpoint','hold_current']
                or profile['whole_split']!={r['id']:r['split'] for r in demos['episodes']}
                or profile['historical_result']['sha256']!=PRODUCERS['event'][1]
                or profile['maximum_complete_episodes']!=2 or profile['teacher_id']!='phone-04'):
            raise ValueError('Foreign04 prospective condition/profile/selection rule')
        value=audit.json(accepted['result']);_closed(value);producers['04']=value
        if (value['source_commit']!=accepted['source_commit'] or value['format']!=E_FORMAT
                or value['profile_sha256']!=accepted['profile']['sha256'] or value['plan']!=profile
                or value['status']!='complete_diagnostic_pair' or value['frozen_after']!=FROZEN
                or len(value['records'])!=2
                or [r['spec']['condition'] for r in value['records']]!=profile['conditions']
                or any(r.get('complete') is not True for r in value['records'])):
            raise ValueError('Both complete source-bound04 conditions required before selection')
        for r in value['records']:
            _validate_record_spec(r,demos['episodes'][4],require_success=False)
            spec=r['spec']
            if (spec['id']!='prelift04-'+spec['condition']
                    or spec['stage']!='prelift_closure_diagnostic' or spec['method']!='prelift_'+spec['condition']):
                raise ValueError('04 pair contains a foreign case/spec/text/profile')
            _closed_manifest(r,audit)
        selected04=next((r for c in profile['bank_candidate_priority'] for r in value['records']
                         if r['spec']['condition']==c and r['success'] is True),None)
        if selected04 is None:raise ValueError('Neither04 condition succeeded; no complete fitting bank')
    entries=[]
    for episode in demos['episodes']:
        identity=episode['id'];ordinal=episode['canonical_index']
        if identity=='phone-04' and accepted is None:
            entries.append(dict(id=identity,split=episode['split'],status='PENDING04',record=None));continue
        producer='04' if identity=='phone-04' else 'event' if ordinal<4 else 'suffix'
        value=producers[producer]
        record_id=selected04['spec']['id'] if producer=='04' else 'event-teacher-'+identity
        matches=[r for r in value['records'] if r['spec']['id']==record_id]
        if len(matches)!=1:raise ValueError('Missing/duplicate original teacher record')
        row=matches[0];spec=row['spec']
        _validate_record_spec(row,episode,require_success=True)
        if producer=='04':
            profile_ref=accepted['profile'];convention='prelift_profile_v1'
            result_ref=accepted['result']
        else:
            if spec['stage']!='event_aligned_teacher_execution' or spec['method']!='event_aligned_teacher':
                raise ValueError('Only original aligned teacher profiles')
            profile_ref=dict(kind='original_event_h32_v1',producer_commit=PRODUCERS[producer][0])
            convention='original_event_h32_v1';result_ref=result_refs[producer]
        entries.append(dict(id=identity,split=episode['split'],status='successful_closed',record=deepcopy(row),
            original_source=deepcopy(demos['actual_whole_files'][identity]),producer_result=deepcopy(result_ref),
            producer_commit=value['source_commit'],profile=deepcopy(profile_ref),plan_convention=convention))
    bank=dict(format=FORMAT,demos=deepcopy(demo_ref),interface=deepcopy(interface_ref),entries=entries,
        retained_producer_results=deepcopy(result_refs),accepted04_profile=deepcopy(accepted04_ref),
        retained_cost_snapshots={name:deepcopy(value['actions']) for name,value in producers.items()},
        cost_meaning='Bound canonical cumulative snapshots, not additive charges; no new actions or ledger',
        fixed12_fit_eligible=all(e['status']=='successful_closed' for e in entries),
        frozen_parameter_sha256=FROZEN,scope='Full original12 only; partial cache is not a reduced fitting bank')
    validate_bank(bank)
    return bank


def validate_bank(bank, *, complete=False):
    if (bank.get('format')!=FORMAT or [e['id'] for e in bank['entries']]!=IDS
            or [e['split'] for e in bank['entries']]!=['train']*8+['validation']*4):
        raise ValueError('No missing, duplicate, reordered or changed-split teacher bank')
    available=[]
    for e in bank['entries']:
        if e['status']=='PENDING04':
            if e['id']!='phone-04' or e['record'] is not None:raise ValueError('Only04 can remain pending')
        elif e['status']=='successful_closed' and e['record'].get('complete') is True and e['record'].get('success') is True:
            available.append(e['id'])
        else:raise ValueError('Failed or unclosed teacher bank entry')
    if bank['fixed12_fit_eligible']!=(len(available)==12) or (complete and len(available)!=12):
        raise ValueError('All12 successful teachers required; never train on11')
    return available


def captured_plan(chunk, decision, native, *, targets, source, convention, profile_sha, cursor, condition=None):
    """Validate captured full H32 labels; never derive E labels by call ordinal."""
    residual=chunk['decision_metadata']['frozen_pi05_residual']
    full=decision[residual['actions']['numeric_array']]
    base=decision[residual['base_actions']['numeric_array']]
    noise=decision['sampled_noise'];issued=decision['issued_motor_joint_targets']
    if (full.shape!=(1,32,14) or base.shape!=(1,32,14) or noise.shape!=(32,32)
            or not 1<=len(issued)<=32 or issued.shape[1:]!=(14,)
            or not np.array_equal(issued,full[0,:len(issued)])
            or any(v.dtype.kind!='f' or not np.isfinite(v).all() for v in (full,base,noise,issued))):
        raise ValueError('Keep captured full proposed/submitted H32/native Gaussian/base14 separately')
    if convention=='original_event_h32_v1':
        metadata=chunk['decision_metadata']['event_schedule']
        if (metadata['teacher_reference']['sha256']!=source['sha256']
                or metadata['source_cursor_before']!=cursor['source']
                or metadata['physical_slots_before']!=cursor['physical']
                or metadata['chunk_index']!=cursor['logical_chunk']):
            raise ValueError('Original captured teacher/cursor/physical prefix differs')
        _,indices,new=scheduled_request(targets,build_schedule(targets),metadata['chunk_index'])
        target=targets[indices];kind='reference' if new.any() else 'tail_hold'
    elif convention=='prelift_profile_v1':
        metadata=chunk['decision_metadata']['prelift_probe'];entry=metadata['entry']
        if (metadata['format']!=E_FORMAT or metadata['profile']['sha256']!=profile_sha
                or metadata['source_cursor_before']!=cursor['source']
                or metadata['execution_index']!=cursor['execution']
                or metadata['physical_slots_before']!=cursor['physical']
                or metadata['teacher_reference']['sha256']!=source['sha256']
                or metadata['condition']!=condition or condition not in ('hold_current','close_endpoint')):
            raise ValueError('Prelift actual profile/execution/source cursor differs')
        schedule=build_schedule(targets);index=cursor['execution']
        expected_source=index if index<3 else index-1
        if index==3:
            if entry!=dict(kind='probe',start=70,end=70,source_chunk=None) or cursor['source']!=70:
                raise ValueError('Exactly one actual current-command H32 hold afterC3/source70')
            indices=np.full(32,-1,dtype=np.int64);new=np.zeros(32,dtype=bool);kind='inserted_hold'
            anchor_ref=metadata['probe_anchor_public14'];target_ref=metadata['explicit_probe_target']
            anchor=decision[anchor_ref['numeric_array']].reshape(14)
            current=decision['_canonical_current_state14'].reshape(14)
            if not np.array_equal(anchor,current):raise ValueError('Probe target must use actual current public14, not a future/source pose')
            target=np.repeat(anchor[None],32,axis=0)
            if metadata['condition']=='close_endpoint':target[:,6]=0.
            explicit=decision[target_ref['numeric_array']]
            if explicit.shape!=(32,14) or not np.array_equal(explicit,target):
                raise ValueError('Actual explicit probe target changed arm/other-gripper/condition')
        elif expected_source<len(schedule['chunks']):
            source_chunk=schedule['chunks'][expected_source]
            if entry!=dict(kind='source',source_chunk=expected_source,start=source_chunk['start'],end=source_chunk['end']):
                raise ValueError('Source chunk changed after inserted hold')
            if cursor['source']!=source_chunk['start']:raise ValueError('Original source frame skipped or repeated')
            target,indices,new=scheduled_request(targets,schedule,expected_source);kind='reference'
        else:
            if entry!=dict(kind='tail',start=len(targets),end=len(targets),source_chunk=None) or cursor['source']!=len(targets):
                raise ValueError('Tail hold before complete original reference')
            target,indices,new=scheduled_request(targets,schedule,len(schedule['chunks']));kind='tail_hold'
    else:raise ValueError('Unaccepted source plan convention')
    if not np.array_equal(full[0],target.astype(full.dtype)):
        raise ValueError('Captured full proposal differs from actual source/probe plan')
    mask=native['valid_action_mask'].reshape(-1);actual=chunk['native_result']['executed_primitive_steps']
    closed=bool(chunk['learning_result']['terminated'] or chunk['learning_result']['truncated'])
    if (type(actual) is not int or mask.size!=len(issued) or not np.isin(mask,(0,1)).all()
            or not np.array_equal(mask,np.arange(len(issued))<actual) or not 0<actual<=len(issued)
            or (not closed and (actual!=32 or len(issued)!=32))):
        raise ValueError('Native valid/cost prefix differs; no fabricated executed tail')
    submitted=np.arange(32)<len(issued);executed=np.zeros(32,dtype=bool);executed[:len(issued)]=mask.astype(bool)
    following=dict(source=cursor['source']+int((new&executed).sum()),physical=cursor['physical']+actual,
                   logical_chunk=cursor['logical_chunk']+(kind!='inserted_hold'),execution=cursor.get('execution',0)+1)
    return dict(target=full[0].copy(),base=base[0].copy(),noise=noise.copy(),indices=indices.copy(),new=new.copy(),
                submitted=submitted,executed=executed,kind=kind,following=following)


def _recheck_bank(bank,deadline):
    rebuilt=assemble_bank(bank['demos'],bank['interface'],bank['retained_producer_results'],
        audit=Audit(deadline),accepted04_ref=bank['accepted04_profile'])
    if digest(rebuilt)!=digest(bank):raise ValueError('Bank manifest differs from its sealed producer/profile records')
    return rebuilt


def _cache_episode(entry,episode,demos,backend,output,*,deadline,guard):
    """Cache one successful episode through existing typed Audit/transforms."""
    audit=Audit(deadline);row=entry['record'];ref=row['episode_manifest_ref'];root=Path(ref['root'])
    contract_path=root/'contract.json'
    contract=audit.json(dict(path=str(contract_path),sha256=file_sha(contract_path)))
    if (contract['source_commit']!=entry['producer_commit'] or contract['dataset_sha256']!=DEMOS_SHA
            or contract['motor_sha256']!=demos['motor_sha256']
            or contract['normalization_sha256']!=demos['normalization_sha256']):
        raise ValueError('Canonical corpus contract belongs to foreign producer/motor/norm/source')
    path=audit.nested(root,ref);manifest=audit.json(dict(path=str(path),sha256=ref['sha256']))
    reset_path=path.with_name('reset.json')
    reset=audit.json(dict(path=str(reset_path),sha256=file_sha(reset_path)))
    validate_reset_identity(reset,row['reset_info'],manifest)
    if (manifest['pending_decision'] is not None or manifest['completed_chunks']!=row['chunks']
            or len(manifest['chunks'])!=row['chunks'] or manifest['executed_primitive_steps']!=row['primitive_steps']):
        raise ValueError('Incomplete canonical episode cannot be cached')
    source=entry['original_source'];audit.bound(source['path'],source['sha256'])
    _,targets=read_episode(source['path'],episode,demos['schema'])  # Command provenance only, no HDF5 RGB.
    cursor=dict(source=0,physical=0,logical_chunk=0,execution=0);previous=reset['initial_observation'];cached=[]
    output=Path(output);output.mkdir(exist_ok=False)
    for ordinal,chunk_ref in enumerate(manifest['chunks']):
        guard();audit.guard();chunk_path=audit.nested(path.parent,chunk_ref)
        chunk=audit.json(dict(path=str(chunk_path),sha256=chunk_ref['sha256']))
        closed=bool(chunk['learning_result']['terminated'] or chunk['learning_result']['truncated'])
        if (chunk['status']!='complete' or chunk['learning_result']['budget_truncated']
                or chunk['context']['trial_id']!=row['spec']['id'] or chunk['episode_id']!=manifest['episode_id']
                or chunk['chunk_index']!=ordinal+1 or chunk['observation']!=previous
                or closed!=(ordinal==len(manifest['chunks'])-1)
                or (closed and chunk['native_result']['success'] is not True)):
            raise ValueError('Canonical observation/chunk/closed-success chain differs')
        decision=audit.arrays(root,chunk['decision_arrays']);native=audit.arrays(root,chunk['native_result']['arrays'])
        frame=chunk['observation'];arrays=audit.arrays(root,frame)
        native_noise_ref=chunk['decision_metadata']['capacity_native_noise']
        if not np.array_equal(decision[native_noise_ref['numeric_array']],decision['sampled_noise']):
            raise ValueError('Captured native private Gaussian differs from actual submitted noise')
        decision['_canonical_current_state14']=arrays['states'].reshape(14)
        plan=captured_plan(chunk,decision,native,targets=targets,source=source,
            convention=entry['plan_convention'],profile_sha=entry['profile'].get('sha256'),cursor=cursor,
            condition=row['spec'].get('condition'))
        raw={key:arrays[key] for key in ('main_images','wrist_images','states')}
        raw['task_descriptions']=frame['language']
        if raw['task_descriptions']!=[episode['instruction']]:raise ValueError('Actual current-frame language differs')
        public=backend._encode(raw)
        if not np.array_equal(np.asarray(public),arrays['features']):raise ValueError('Captured public features fail exact replay')
        observation,target32,roundtrip=transformed_target(backend.model,raw,plan['target'])
        if not torch.equal(observation.state,backend._prepared.state):raise ValueError('Official normalized current state differs')
        if not torch.isfinite(target32).all():raise ValueError('Nonfinite official padded32 target')
        # Never draw replacement noise or build a new native base proposal.
        proposal=backend.decode(plan['noise'])
        if not np.array_equal(proposal,plan['base']):raise ValueError('Captured Gaussian/base14 fail exact replay')
        data=cpu(dict(state=backend._prepared.state,features=torch.as_tensor(public,device=backend.model.device)[None],
            cache=backend._kv_cache,mask=backend._prefix_mask,target32=target32,
            teacher14=torch.as_tensor(plan['target'],device=backend.model.device)[None],
            base14=torch.as_tensor(plan['base'],device=backend.model.device)[None],
            valid=torch.as_tensor(plan['submitted'],device=backend.model.device)[None]))
        diagnostics=dict(physical14_roundtrip=roundtrip,
            normalized32_shape=list(target32.shape),padded18_mean_square=float(target32[...,14:].float().square().mean()),
            padded18_max_abs=float(target32[...,14:].float().abs().max()),
            meaning='Retained normalized32 including18 padding coordinates; not clipped or replaced labels')
        dest=output/f'sample-{ordinal:04d}.pt'
        context=dict(teacher_id=entry['id'],split=entry['split'],producer_result=entry['producer_result'],
            producer_commit=entry['producer_commit'],profile=entry['profile'],record_id=row['spec']['id'],
            episode_manifest_ref=ref,original_source=source)
        sha=atomic_save(dest,dict(format='robotwin_captured_plan_cache_v1',context=context,data=data,
            canonical_chunk=dict(path=str(chunk_path),sha256=chunk_ref['sha256']),
            source_indices=plan['indices'].tolist(),source_new_reference_mask=torch.from_numpy(plan['new']),
            full_known_proposed_plan14=torch.from_numpy(plan['target']),submitted_plan_mask=torch.from_numpy(plan['submitted']),
            native_executed_mask=torch.from_numpy(plan['executed']),proposal_noise=torch.from_numpy(plan['noise']),
            plan_kind=plan['kind'],diagnostics=diagnostics,
            meaning='Actual current input and known submitted plan; native executed mask is separate physical cost'))
        cached.append(dict(path=str(dest),sha256=sha,context=context,anchor=ordinal,
            valid_actions=int(plan['submitted'].sum()),native_actual_actions=int(plan['executed'].sum()),
            submitted_hold_slots=int((~plan['new']&plan['submitted']).sum()),
            native_valid_hold_slots=int((~plan['new']&plan['executed']).sum()),diagnostics=diagnostics))
        cursor=plan['following'];previous=chunk['next_observation']
    if cursor['physical']!=row['primitive_steps']:raise ValueError('Canonical episode action sum differs')
    return cached


def cache_bank(bank,backend,output,*,deadline,guard,model_qualification_ref,allow_partial=False):
    """Model-only lazy backend, never reset/step/extra qualification episodes.

    Existing resource owner binds actual CUDA/host/frozen motor/norm to the
    supplied saved qualification. No physical work or scheduler is owned here.
    """
    import socket
    if backend.model.device.type!='cuda' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Actual one-GPU model-only allocation required')
    if backend.env is not None:raise ValueError('Use lazy model-only backend before any physical scene/reset')
    bank=_recheck_bank(bank,deadline);available=validate_bank(bank,complete=not allow_partial)
    audit=Audit(deadline);demos=audit.json(bank['demos']);interface=audit.json(bank['interface'])
    qualification=audit.json(model_qualification_ref);probe=audit.json(qualification['gpu_probe'])
    if len(probe['devices'])!=1:raise ValueError('One actual probed GPU required')
    _bind_current_cuda(qualification,probe)  # Before any encode/decode.
    frozen=backend.frozen_fingerprint()
    if (qualification.get('format')!='robotwin_model_only_cache_qualification_v1'
            or qualification.get('status')!='passed' or qualification['hostname']!=socket.gethostname()
            or probe.get('status')!='passed' or probe['hostname']!=socket.gethostname() or len(probe['devices'])!=1
            or qualification['cuda_uuid']!=probe['devices'][0]['cuda_uuid']
            or qualification['motor_sha256']!=interface['motor_sha256']
            or qualification['normalization_sha256']!=interface['normalization_sha256']
            or qualification['frozen_parameter_sha256']!=frozen or frozen!=FROZEN
            or backend.contract['norm_stats_sha256']!=interface['normalization_sha256']):
        raise ValueError('Actual frozen model/norm/hardware qualification binding differs')
    backend._check_frozen()
    if backend.kv_adapter is not None or backend.action_residual is not None:
        raise ValueError('Captured native-base replay requires both residual hooks disabled')
    backend._paired_instruction=None
    output=Path(output);output.mkdir(parents=True,exist_ok=False);cached=[]
    for entry,episode in zip(bank['entries'],demos['episodes']):
        if entry['id'] not in available:continue
        cached.extend(_cache_episode(entry,episode,demos,backend,output/entry['id'],deadline=deadline,guard=guard))
    if backend.env is not None or backend.frozen_fingerprint()!=frozen:
        raise ValueError('Model-only cache created physics or changed frozen parameters')
    backend._check_frozen()
    result=dict(format='robotwin_fixed12_cache_receipt_v1',bank=deepcopy(bank),records=cached,
        model_qualification=deepcopy(model_qualification_ref),complete=False,fit_eligible=False,
        cached_teacher_ids=available,meaning='Prospective partial/preparation cache; seal full12 separately, no normalizer/fit here')
    _atomic_json(output/'partial-cache.json',result)
    return result


def seal_full_cache(bank,cache_receipts,*,deadline):
    """Join sealed per-case caches only once the full fixed12 bank qualifies."""
    bank=_recheck_bank(bank,deadline);validate_bank(bank,complete=True)
    expected={e['id']:e for e in bank['entries']};actual={};records=[];canonical_seen=set();manifests={}
    for receipt in cache_receipts:
        value=Audit(deadline).json(receipt)
        if value.get('format')!='robotwin_fixed12_cache_receipt_v1' or value.get('fit_eligible') is not False:
            raise ValueError('Only explicit partial/preparation cache receipts can be joined')
        for row in value['records']:
            context=row['context'];identity=context['teacher_id'];entry=expected.get(identity)
            if entry is None:raise ValueError('Foreign cached teacher')
            wanted=dict(teacher_id=identity,split=entry['split'],producer_result=entry['producer_result'],
                producer_commit=entry['producer_commit'],profile=entry['profile'],record_id=entry['record']['spec']['id'],
                episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
            if context!=wanted:raise ValueError('Cache profile/spec/producer/source does not match complete bank')
            key=(identity,row['anchor'])
            if key in actual:raise ValueError('Duplicate cache chunk')
            audit=Audit(deadline);audit.bound(row['path'],row['sha256'])
            saved=torch.load(row['path'],map_location='cpu',weights_only=False)
            if identity not in manifests:
                manifests[identity]=_closed_manifest(entry['record'],Audit(deadline))
            manifest_path,manifest=manifests[identity];anchor=row['anchor']
            if type(anchor) is not int or not 0<=anchor<len(manifest['chunks']):
                raise ValueError('Cache anchor is outside the canonical episode manifest')
            original=manifest['chunks'][anchor]
            canonical_path=audit.nested(manifest_path.parent,original)
            canonical=dict(path=str(canonical_path),sha256=original['sha256'])
            if saved.get('canonical_chunk')!=canonical:
                raise ValueError('Saved cache canonical chunk differs from its exact episode anchor')
            canonical_key=(str(canonical_path.resolve()),original['sha256'])
            if canonical_key in canonical_seen:raise ValueError('Canonical chunk reused under another cache anchor')
            canonical_seen.add(canonical_key)
            native=audit.json(canonical)
            if (native['status']!='complete' or native['episode_id']!=manifest['episode_id']
                    or native['context']['trial_id']!=entry['record']['spec']['id']
                    or native['chunk_index']!=anchor+1
                    or native['native_result']['executed_primitive_steps']!=row['native_actual_actions']):
                raise ValueError('Canonical chunk identity/native cost differs from cached anchor')
            submitted=saved['submitted_plan_mask'];executed=saved['native_executed_mask']
            if (saved['format']!='robotwin_captured_plan_cache_v1' or saved['context']!=wanted
                    or submitted.shape!=(32,) or executed.shape!=(32,)
                    or submitted.dtype!=torch.bool or executed.dtype!=torch.bool
                    or not torch.equal(submitted,torch.arange(32)<int(submitted.sum()))
                    or not torch.equal(executed,torch.arange(32)<int(executed.sum()))
                    or (executed & ~submitted).any()
                    or int(executed.sum())!=row['native_actual_actions'] or int(submitted.sum())!=row['valid_actions']
                    or not torch.equal(saved['data']['valid'],submitted[None])
                    or not torch.equal(saved['data']['teacher14'],saved['full_known_proposed_plan14'][None])
                    or set(saved['data'])!={'state','features','cache','mask','target32','teacher14','base14','valid'}
                    or saved['data']['features'].shape!=(1,2062) or saved['data']['state'].shape!=(1,14)
                    or saved['data']['base14'].shape!=(1,32,14)
                    or saved['data']['target32'].shape!=(1,32,32)):
                raise ValueError('Saved cache masks/labels/context differ from full-plan/native-cost receipt')
            actual[key]=True;records.append(deepcopy(row))
    for identity,entry in expected.items():
        rows=[r for r in records if r['context']['teacher_id']==identity]
        if (sorted(r['anchor'] for r in rows)!=list(range(entry['record']['chunks']))
                or sum(r['native_actual_actions'] for r in rows)!=entry['record']['primitive_steps']):
            raise ValueError('Full12 cache is missing/duplicating chunks or changing native costs')
    return dict(format='robotwin_fixed12_complete_cache_v1',bank=deepcopy(bank),cache_receipts=deepcopy(cache_receipts),
        records=records,complete=True,fixed12_fit_eligible=True,
        resource_fit_admitted=False,meaning='Full12 data eligibility only; actual predeclared fit profile/resources still required')
