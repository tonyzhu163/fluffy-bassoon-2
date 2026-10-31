"""Bounded read-only diagnosis of fixed02/aligned02/aligned04, no model imports.

Only the parent's admitted Linux CPU unit calls array/image helpers. This module
never constructs an environment, ledger, planner, RNG or checkpoint loader.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

from .robotwin_replay_diagnosis import Audit,json_write,validate_reset_identity,panel

FORMAT='robotwin_event_teacher_diagnosis_v1'
PRODUCER='be247676c1217d9e046091c52cac2b7d0f2d3014'
RESULT_SHA='ba775b4fcbb3029120927ecd59461fabb1cccd7e95326f8d9f3a6f731ea77d3f'
RECOVERY_SHA='0403db7c2bd0c4e7566cfdbb5a6404409a0d45a0eb85df4c0972a2bbcd58a153'
IDS=('event-control-fixed-phone-02','event-teacher-phone-02','event-teacher-phone-04')
EXPECTED=((False,400),(True,193),(False,400))


def validate_closed_metadata(result,terminal,closeout):
    """No partial panel, trained result, changed failed case or refunded cost."""
    ordered=['event-control-fixed-phone-02','event-teacher-phone-02','event-teacher-phone-00',
             'event-teacher-phone-01','event-teacher-phone-03','event-teacher-phone-04']
    records=result['records']
    if (result['source_commit']!=PRODUCER or result['status']!='blocked_failure'
            or result['stage']!='teacher_executability' or result['pending'] is not None
            or result['failure']!=dict(type='RuntimeError',message='Fixed aligned teacher failed; preserve all demos and stop before fitting')
            or [r['spec']['id'] for r in records]!=ordered
            or [(r['success'],r['primitive_steps']) for r in records]!=
               [(False,400),(True,193),(True,193),(True,193),(True,193),(False,400)]
            or any(r.get('complete') is not True for r in records)
            or result['actions']['actual_actions']!=20171 or result['actions']['charged_actions']!=20171
            or result['actions']['pending_actions'] or result['actions']['uncertain_actions']
            or result['actions']['unresolved_decisions']):raise ValueError('Exact six closed cases/failed04/no uncertain cost required')
    if (closeout.get('exit_code')!=1 or closeout.get('error') is not None
            or closeout.get('source_commit')!=PRODUCER or closeout.get('charged_gpu_seconds')!=298
            or closeout['attempt']['status']!='finished' or closeout['attempt'].get('slice_overrun') is not False
            or terminal.get('format')!='original_a1_event_capacity_terminal_receipt_v1'
            or terminal.get('job_id')!='job-2rkimbvypowd' or terminal.get('scheduler_status')!='Failed'
            or terminal.get('source_commit')!=PRODUCER or terminal.get('exit_code')!=1
            or terminal.get('charged_gpu_seconds')!=298 or terminal.get('cumulative_a1_gpu_seconds')!=2334
            or terminal.get('original_twelve_demos_retained') is not True
            or terminal.get('all_twelve_teacher_gate_passed') is not False or terminal.get('cache_created') is not False
            or terminal.get('kv_updates')!=0 or terminal.get('action_updates')!=0
            or terminal.get('unrun_aligned_teacher_suffix')!=[f'phone-{i:02d}' for i in range(5,12)]
            or any(result['actions'][k]!=v for k,v in terminal['canonical_costs'].items())):
        raise ValueError('Exact terminal failure/retained cost/unrun suffix/no cache or fit required')


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def compare_plan(targets,decision,mask,*,ordinal,physical_before,aligned):
    """Expert proposed/submitted plans are distinct from native cost indices."""
    import numpy as np
    from .robotwin_event_schedule import build_schedule,scheduled_request
    targets=np.asarray(targets)
    if aligned:
        schedule=build_schedule(targets)
        expected,indices,new=scheduled_request(targets,schedule,ordinal)
    else:
        indices=np.minimum(physical_before+np.arange(32),len(targets)-1)
        expected=targets[indices];new=physical_before+np.arange(32)<len(targets)
    proposed=decision['proposed'];issued=decision['issued'];mask=np.asarray(mask).reshape(-1)
    if (proposed.shape!=(32,14) or issued.ndim!=2 or issued.shape[1]!=14
            or not 1<=len(issued)<=32 or mask.size!=len(issued)
            or proposed.dtype!=targets.dtype or issued.dtype!=targets.dtype
            or not np.isfinite(proposed).all() or not np.isfinite(issued).all()
            or not np.isin(mask,(0,1)).all()):raise ValueError('Full proposedH32/submitted14/binary native mask required')
    count=int(mask.sum())
    if not np.array_equal(mask,np.arange(len(mask))<count):raise ValueError('Native cost mask is not a prefix')
    return dict(full_proposed_reference_exact=bool(np.array_equal(proposed,expected)),
        submitted_reference_exact=bool(np.array_equal(issued,expected[:len(issued)])),
        submitted_is_proposed_prefix=bool(np.array_equal(issued,proposed[:len(issued)])),
        proposed_slots=32,submitted_slots=len(issued),native_valid_slots=count,
        terminal_unexecuted_submitted_slots=len(issued)-count,
        proposed_reference_indices=indices.tolist(),
        submitted_reference_slots=int(new[:len(issued)].sum()),submitted_declared_hold_slots=int((~new[:len(issued)]).sum()),
        native_valid_reference_indices=indices[:len(issued)][mask.astype(bool)].tolist(),
        meaning='Request/plan and native cost evidence; prefix indices are not physical waypoint/time observations')


def summarize_physical(value):
    """Keep recorded quantities and names; never substitute drives for qpos."""
    import numpy as np
    out={k:value.get(k) for k in ('task_info','actors','grippers','conventions','native_counters')}
    out['articulations']={}
    for side in ('left','right'):
        entry=value.get('articulations',{}).get(side,dict(available=False,reason='not captured'))
        if not entry.get('available'):
            out['articulations'][side]=entry;continue
        source=entry['value'];names=source['joint_names']
        qpos=np.asarray(source['qpos'],dtype=float);drives=np.asarray(source['drive_targets'],dtype=float)
        if (len(names)!=len(set(names)) or qpos.shape!=(len(names),) or drives.shape!=qpos.shape
                or not np.isfinite(qpos).all() or not np.isfinite(drives).all()):
            raise ValueError('Measured/drive native coordinates cannot be aligned by named1DOF joints')
        joints={name:dict(measured_qpos=float(qpos[i]),drive_target=float(drives[i]),
            measured_minus_drive=float(qpos[i]-drives[i])) for i,name in enumerate(names)}
        grip=value.get('grippers',{}).get(side,{})
        fingers=[]
        if grip.get('available'):
            for finger in grip['value']['named_drive_targets']:
                if finger['name'] not in joints:raise ValueError('Named gripper joint absent from captured articulation')
                fingers.append(dict(name=finger['name'],**joints[finger['name']],
                    separately_recorded_drive_target=finger['target']))
        out['articulations'][side]=dict(available=True,named_joints=joints,gripper_joints=fingers)
    out['camera_calibration_available']=sorted(value.get('already_available_camera_calibration',{}))
    return out


def physical_change(before,after):
    """Native within-chunk changes, never source-time or success geometry."""
    import numpy as np
    result=dict(actors={},articulations={})
    for name in ('phone','stand'):
        left=before.get('actors',{}).get(name,{});right=after.get('actors',{}).get(name,{})
        if left.get('available') and right.get('available'):
            a,b=left['value'],right['value'];dp=np.asarray(b['p'])-np.asarray(a['p'])
            qa,qb=np.asarray(a['q']),np.asarray(b['q'])
            result['actors'][name]=dict(available=True,world_translation_delta=dp.tolist(),
                world_translation_L2=float(np.linalg.norm(dp)),
                quaternion_component_delta_up_to_sign=float(min(np.max(np.abs(qb-qa)),np.max(np.abs(qb+qa)))),
                variant_unchanged=a['variant_id']==b['variant_id'])
        else:result['actors'][name]=dict(available=False,reason='pose missing at one boundary')
    for side in ('left','right'):
        a,b=before.get('articulations',{}).get(side,{}),after.get('articulations',{}).get(side,{})
        if a.get('available') and b.get('available'):
            aa,bb=a['value'],b['value']
            if aa['joint_names']!=bb['joint_names']:raise ValueError('Native articulation names changed within chunk')
            delta=np.asarray(bb['qpos'])-np.asarray(aa['qpos'])
            result['articulations'][side]=dict(available=True,
                named_measured_qpos_delta=dict(zip(aa['joint_names'],delta.tolist())))
        else:result['articulations'][side]=dict(available=False,reason='measured qpos missing at one boundary')
    return result


def compare_source_scene(row,calibration,physical):
    import numpy as np
    expected=row['scene_metadata']['info'];native=physical.get('task_info',{})
    variants={key:dict(source=value,native=(native.get('value') or {}).get(key),
        exact=native.get('available') is True and (native.get('value') or {}).get(key)==value)
        for key,value in expected.items()}
    checks=[];actual=physical.get('already_available_camera_calibration',{})
    for path,value in calibration.items():
        _,view,field=path.split('/')
        found=actual.get(view,{}).get(field)
        check=dict(source_dataset=path,native_view=view,field=field,available=found is not None)
        if found is not None:
            aa,bb=np.asarray(value),np.asarray(found)
            check.update(same_shape=aa.shape==bb.shape,exact=bool(np.array_equal(aa,bb)),
                max_abs_difference=float(np.abs(aa-bb).max()) if aa.shape==bb.shape else None)
        checks.append(check)
    return dict(source_scene_metadata=row['scene_metadata'],native_task_info=native,
        variant_and_arm_tag_checks=variants,reset_camera_calibration_checks=checks,
        source_measured_articulation_qpos_available=False,source_actor_worldposes_available=False,
        synchronized_source_physics_timestamps_available=False,
        limits='Source joint_action/vector is qualified DRIVE targets, not measured qpos. No qualified source actor/worldpose or dense-time field is bound for comparison.')


def inspect(packet,output,audit):
    import numpy as np
    import h5py
    from .robotwin_demo_capacity import validate_demo_manifest,read_episode,raw_observation
    from .robotwin_event_schedule import build_schedule,advance_cursor
    if (packet.get('format')!='robotwin_event_teacher_diagnosis_inputs_v1'
            or packet['result']['sha256']!=RESULT_SHA or packet['closed_recovery']['sha256']!=RECOVERY_SHA
            or set(packet['resets'])!=set(IDS)):raise ValueError('Exact three-case terminal packet required')
    result=audit.json(packet['result']);terminal=audit.json(packet['terminal'])
    closeout=audit.json(packet['allocation_closeout']);demos=validate_demo_manifest(audit.json(packet['demos']))
    recovery=audit.bound(packet['closed_recovery']['path'],RECOVERY_SHA)
    validate_closed_metadata(result,terminal,closeout)
    if (recovery.stat().st_size!=223589 or packet['demos']['sha256']!=result['demos_sha256']
            or terminal['full_recovery']['path']!=str(recovery)
            or terminal['full_recovery']['sha256']!=RECOVERY_SHA
            or terminal['full_recovery']['bytes']!=223589):raise ValueError('Exact opaque recovery/demos required')
    all_records=result['records'];selected={r['spec']['id']:r for r in all_records if r['spec']['id'] in IDS}
    if (set(selected)!=set(IDS) or len(all_records)!=6
            or [(selected[i]['success'],selected[i]['primitive_steps']) for i in IDS]!=list(EXPECTED)
            or any(r.get('complete') is not True for r in all_records)
            or sum(r['primitive_steps'] for r in all_records)!=1572
            or result['actions']['actual_actions']!=20171 or result['actions']['charged_actions']!=20171
            or result['actions']['pending_actions'] or result['actions']['uncertain_actions']
            or result['actions']['unresolved_decisions']):raise ValueError('Fixed complete cases/canonical costs changed')
    producer=Path(packet['result']['path']).parent;corpus=producer/'corpus';output=Path(output)
    if output.resolve().is_relative_to(producer.resolve()):raise ValueError('Never write into immutable producer')
    absent={name:not (producer/name).exists() for name in ('cache','kv','action')}
    if not all(absent.values()):raise ValueError('Unexpected cache/fit; not this diagnosed failure')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(format=FORMAT,status='running',producer=PRODUCER,episodes=[],
        all_original_demos_retained=[dict(id=r['id'],split=r['split']) for r in demos['episodes']],
        unrun_suffix=['phone-'+f'{i:02d}' for i in range(5,12)],training_outputs_absent=absent,
        opaque_recovery=dict(packet['closed_recovery'],deserialized=False),terminal=packet['terminal'],
        closed_cost_snapshot=result['actions'],new_robot_actions=0,model_forwards=0,SGD=0,automatic_retry_authorized=False)
    rows={r['id']:r for r in demos['episodes']};physical_initial={};phase_rows=[]
    for ident in IDS:
        audit.guard();record=selected[ident];aligned=ident!='event-control-fixed-phone-02'
        source_id='phone-04' if ident.endswith('04') else 'phone-02';row=rows[source_id]
        source=demos['actual_whole_files'][source_id];h5=audit.bound(source['path'],source['sha256'])
        state,targets=read_episode(h5,row,demos['schema']);schedule=build_schedule(targets) if aligned else None
        mr=record['episode_manifest_ref']
        if Path(mr['root']).resolve()!=corpus.resolve():raise ValueError('Foreign canonical root')
        mp=audit.nested(corpus,mr);manifest=json.loads(mp.read_text())
        reset_ref=packet['resets'][ident]['canonical_reset'];reset=audit.json(reset_ref)
        if Path(reset_ref['path']).resolve()!=(mp.parent/manifest['reset']).resolve():raise ValueError('Foreign canonical reset')
        validate_reset_identity(reset,record['reset_info'],manifest)
        pr=packet['resets'][ident]['physical_reset']
        if pr not in result['physical_reset_audits']:raise ValueError('Unbound reset physical audit')
        physical_reset=audit.json(pr)
        if physical_reset['corpus_episode_id']!=manifest['episode_id']:raise ValueError('Physical reset belongs to another episode')
        physical_initial[ident]=physical_reset['physical']
        current=reset['initial_observation'];total=0;cursor=dict(chunk_index=0,physical_slots=0,source_cursor=0)
        current_arrays=audit.arrays(corpus,current)
        chunks=[];previous_rng=None;last_physical=physical_reset['physical']
        if len(manifest['chunks'])!=len(record['corpus_refs']):raise ValueError('Canonical chunk chain differs')
        refs={Path(ref['path']).resolve():ref for ref in result['chunk_audits']}
        for ordinal,(cr,mcr) in enumerate(zip(record['corpus_refs'],manifest['chunks'])):
            audit.guard();cp=audit.nested(corpus,cr);chunk=json.loads(cp.read_text())
            if (Path(cr['root']).resolve()!=corpus.resolve() or cp.resolve()!=(mp.parent/mcr['path']).resolve()
                    or cr['sha256']!=mcr['sha256'] or chunk['status']!='complete'
                    or chunk['observation']!=current or chunk['context']['trial_id']!=ident
                    or chunk['episode_id']!=manifest['episode_id'] or chunk['chunk_index']!=ordinal+1
                    or chunk['learning_result']['budget_truncated']):raise ValueError('Wrong/incomplete canonical chain')
            data=audit.arrays(corpus,chunk['decision_arrays']);native=audit.arrays(corpus,chunk['native_result']['arrays'])
            next_arrays=audit.arrays(corpus,chunk['next_observation'])
            metadata=chunk['decision_metadata'];mode=metadata['frozen_pi05_residual']['configuration']
            if (chunk['observation']['language']!=[row['instruction']]
                    or chunk['next_observation']['language']!=[row['instruction']]
                    or not np.array_equal(data['sampled_noise'],data[metadata['capacity_native_noise']['numeric_array']])):
                raise ValueError('Recorded observation language/private VLA draw binding differs')
            kind='original_gripper_event_h32_replay_v1' if aligned else 'qualified_source_demo_command_replay_v1'
            if mode['kind']!=kind or mode['reference']['sha256']!=source['sha256']:raise ValueError('Foreign expert plan source')
            if aligned and mode['schedule']!=schedule:raise ValueError('Captured aligned schedule differs')
            proposed=data[metadata['frozen_pi05_residual']['actions']['numeric_array']][0]
            comparison=compare_plan(targets,dict(proposed=proposed,issued=data['issued_motor_joint_targets']),
                native['valid_action_mask'],ordinal=ordinal,physical_before=total,aligned=aligned)
            if not all(comparison[k] for k in ('full_proposed_reference_exact','submitted_reference_exact','submitted_is_proposed_prefix')):
                raise ValueError('Captured full expert/submitted plan differs from fixed reference')
            n=comparison['native_valid_slots'];closed=bool(chunk['learning_result']['terminated'] or chunk['learning_result']['truncated'])
            if n!=chunk['native_result']['executed_primitive_steps'] or closed!=(ordinal==len(manifest['chunks'])-1):
                raise ValueError('Native cost/complete episode boundary differs')
            if aligned:cursor=advance_cursor(schedule,ordinal,total,cursor['source_cursor'],actual=n,
                issued=comparison['submitted_slots'],closed=closed)
            arpath=(producer/'chunk-audits'/manifest['episode_id']/cp.name).resolve()
            if arpath not in refs:raise ValueError('Missing result-bound physical/RNG chunk audit')
            ar=refs[arpath];observed=audit.json(ar)
            if observed['corpus_chunk']!=cr:raise ValueError('Physical audit not bound to canonical chunk SHA')
            if observed['native_global_numpy_before']!=metadata['native_global_numpy_before']:
                raise ValueError('Captured before-global RNG differs across bound records')
            if observed['physical_before_audit']!=metadata['physical_before_audit']:
                raise ValueError('Captured before-physical fields differ across bound records')
            if aligned and (observed['cursor_before']['source_cursor']!=metadata['event_schedule']['source_cursor_before']
                    or observed['cursor_after']!=cursor):raise ValueError('Aligned source cursor/audit differs')
            before=observed['physical_before_audit'];after=observed['physical_after_audit']
            summary=dict(chunk=ordinal+1,canonical_chunk=cr,physical_rng_audit=ar,request=comparison,
                physical_before=summarize_physical(before),physical_after=summarize_physical(after),
                public_drive_target14_before=current_arrays['states'].tolist(),
                public_drive_target14_after=next_arrays['states'].tolist(),
                captured_native_noise_array=chunk['decision_arrays']['arrays']['sampled_noise'],
                within_native_change=physical_change(before,after),
                global_numpy=dict(before_sha256=json_sha(observed['native_global_numpy_before']),
                    after_sha256=json_sha(observed['native_global_numpy_after']),
                    previous_after_matches_before=None if previous_rng is None else previous_rng==observed['native_global_numpy_before']),
                native_wall_seconds=chunk['native_result']['duration_seconds'],physics_duration_available=False,
                terminated=chunk['learning_result']['terminated'],truncated=chunk['learning_result']['truncated'])
            chunks.append(summary);previous_rng=observed['native_global_numpy_after'];last_physical=after
            phone=summary['within_native_change']['actors']['phone']
            phase_rows.append(dict(episode=ident,chunk=ordinal+1,physical_slots_before=total,native_valid_slots=n,
                submitted_slots=comparison['submitted_slots'],reference_min=min(comparison['proposed_reference_indices']),
                reference_max=max(comparison['proposed_reference_indices']),
                phone_world_translation_L2=phone.get('world_translation_L2'),
                left_nominal_before=(before.get('grippers',{}).get('left',{}).get('value') or {}).get('nominal'),
                left_nominal_after=(after.get('grippers',{}).get('left',{}).get('value') or {}).get('nominal')))
            total+=n;current=chunk['next_observation'];current_arrays=next_arrays
        if (total!=record['primitive_steps'] or manifest['executed_primitive_steps']!=total
                or len(chunks)!=record['chunks'] or manifest['pending_decision'] is not None):raise ValueError('Closed episode totals differ')
        calibration={}
        with h5py.File(h5,'r') as data:
            for view in ('head_camera','left_camera','right_camera'):
                for key in ('intrinsic_cv','extrinsic_cv','cam2world_gl'):
                    name=f'observation/{view}/{key}'
                    if name in data:calibration[name]=np.asarray(data[name][0]).tolist()
        images=[]
        for label,frame,index in [('reset',reset['initial_observation'],0),('final',current,len(targets)-1)]:
            audit.guard();arrays=audit.arrays(corpus,frame);original=raw_observation(h5,row,demos['schema'],index,state)
            dest=output/(ident+'-'+label+'.png')
            metrics=panel(original,arrays,f'{ident} {label}: sourceframe{index}; UNSYNCHRONIZED physical times',dest)
            images.append(dict(path=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                actual_frame=frame,source_frame=index,physical_times_matched=False,camera_metrics=metrics))
        initial=audit.arrays(corpus,reset['initial_observation'])
        report['episodes'].append(dict(id=ident,success=record['success'],native_actual_actions=total,
            expected_seed=row['reset_seed'],actual_seed=record['reset_info']['actual_seed'],
            instruction_exact=record['reset_info']['instruction']==[row['instruction']],source=source,
            initial_public_drive_exact=bool(np.array_equal(initial['states'][0],state[0])),
            source_scene_comparison=compare_source_scene(row,calibration,physical_reset['physical']),
            physical_reset_ref=pr,physical_reset=summarize_physical(physical_reset['physical']),
            physical_final=summarize_physical(last_physical),
            reset_to_final_native_change=physical_change(physical_reset['physical'],last_physical),
            chunks=chunks,images=images))
    report['two02_reset_native_physical_comparison']=physical_change(physical_initial[IDS[0]],physical_initial[IDS[1]])
    with (output/'native-phase-summary.csv').open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(phase_rows[0]));writer.writeheader();writer.writerows(phase_rows)
    for path,ref in list(audit.files.items()):audit.bound(path,ref['sha256'])
    report.update(status='audit_complete',full_proposed_and_submitted_plans_exact=True,
        input_files=audit.files,input_bytes=audit.bytes,findings=[
            'Native measured qpos/drive/nominal grip/actor poses are separate recorded quantities.',
            'All submitted chunk parameters can affect native planning before execution; valid masks are cost boundaries, not physical waypoint traces.',
            'Source seed/variant/calibration matches are reported where available; source measured articulation/actor poses and synchronized dense physics timing are not established.',
            'Before/after global NumPy state hashes are opaque audit references only; no RNG replay or reseeding was performed.',
            'No failed demo filtering, all12 gate relaxation, fitting, class rejection or GPU retry follows automatically.'])
    if sum(p.stat().st_size for p in output.iterdir())>16<<20:raise ValueError('Small evidence exceeds16MiB')
    json_write(output/'report.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','allocation','zero-gpu-proof'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);args=p.parse_args(argv)
    if sys.platform!='linux':raise RuntimeError('Allocated Linux only')
    if not math.isfinite(args.deadline_unix):raise ValueError('Original finite absolute CPU clock required')
    audit=Audit(args.deadline_unix)
    allocation=audit.json(dict(path=str(args.allocation),sha256=args.allocation_sha256))
    proof=audit.json(dict(path=str(args.zero_gpu_proof),sha256=args.zero_gpu_proof_sha256))
    def git(*values):
        audit.guard();return subprocess.check_output(['git',*values],text=True,
            timeout=max(.1,min(10,args.deadline_unix-time.time()))).strip()
    if (git('rev-parse','HEAD')!=args.source_commit or git('status','--porcelain')
            or allocation.get('source_commit')!=args.source_commit or allocation.get('hostname')!=socket.gethostname()
            or allocation.get('gpu_count')!=0 or allocation.get('phase_id')!='frozen-pi05-endtoend-20261031-r1'
            or allocation.get('prior_phase_cpu_charge')<978 or allocation.get('maximum_phase_cpu_worker_seconds')!=1800
            or not 1<=allocation['maximum_cpu_worker_seconds']<=180
            or allocation['prior_phase_cpu_charge']+allocation['maximum_cpu_worker_seconds']>1800
            or allocation['deadline_unix']!=args.deadline_unix
            or not allocation['started_unix']<=time.time()<args.deadline_unix<=allocation['started_unix']+allocation['maximum_cpu_worker_seconds']
            or Path(allocation['inspection_output']).resolve()!=args.output.resolve()
            or args.output.resolve()!=Path(allocation['output']).resolve()/'inspection'
            or proof.get('passed') is not True or proof.get('hostname')!=socket.gethostname()
            or proof.get('device_count_before_mask')!=0 or proof.get('cuda_initialized_before_mask') is not False
            or proof.get('actual_cpu_cores',0)<2 or proof.get('actual_memory_bytes',0)<16<<30
            or allocation.get('zero_gpu_attestation_sha256')!=args.zero_gpu_proof_sha256):
        raise ValueError('Bound actual0GPU2CPU16GiB/original<=180CPU-s setup+tests+inspection clock required')
    packet=audit.json(dict(path=str(args.inputs),sha256=args.inputs_sha256))
    try:report=inspect(packet,args.output,audit)
    except BaseException as error:
        if args.output.is_dir():json_write(args.output/'failure.json',dict(type=type(error).__name__,message=str(error),input_files=audit.files))
        raise
    if any(k.split('.')[0] in ('torch','jax','rlinf','openpi','sapien') for k in sys.modules):raise RuntimeError('Forbidden model/simulator import')
    print(json.dumps(dict(status=report['status'],selected_episodes=len(report['episodes']),new_robot_actions=0,SGD=0)))


if __name__=='__main__':main()
