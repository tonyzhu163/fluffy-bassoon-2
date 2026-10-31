"""Read-only CPU diagnosis of the three immutable source-demo replays.

No Torch, model, simulator, scheduler or campaign ledger is imported/constructed.
Array/image work is called only by the parent's allocated Linux CPU gate.
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
import zipfile

FORMAT='robotwin_teacher_replay_diagnosis_v1'
PRODUCER='8fb991c8d5a3b9f86a81e2a4f2cf55b185cec81f'
ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
GRIP=[6,13]
IDS=[f'capacity-teacher-phone-{i:02d}' for i in range(3)]


def json_write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


class Audit:
    """Bound immutable file reads and the parent's shared absolute CPU clock."""
    def __init__(self,deadline):
        self.deadline=deadline;self.files={};self.bytes=0
    def guard(self):
        if time.time()>=self.deadline:raise TimeoutError('Original shared CPU deadline')
    def bound(self,path,digest):
        self.guard();path=Path(path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError('Require an existing absolute immutable regular file')
        if str(path) not in self.files:
            size=path.stat().st_size
            if size>100<<20 or self.bytes+size>1<<30 or len(self.files)>=256:
                raise ValueError('Bounded three-episode input inventory exceeded')
            value=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''):self.guard();value.update(block)
            self.files[str(path)]=dict(bytes=size,sha256=value.hexdigest(),mtime_ns=path.stat().st_mtime_ns);self.bytes+=size
        if self.files[str(path)]['sha256']!=digest:raise ValueError('Immutable input SHA mismatch: '+str(path))
        if path.stat().st_size!=self.files[str(path)]['bytes'] or path.stat().st_mtime_ns!=self.files[str(path)]['mtime_ns']:
            raise ValueError('Immutable input changed during inspection')
        return path
    def json(self,ref):
        path=self.bound(ref['path'],ref['sha256'])
        if path.stat().st_size>4<<20:raise ValueError('JSON metadata exceeds bound')
        return json.loads(path.read_text())
    def nested(self,root,ref):
        root=Path(root).resolve();relative=Path(ref['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Corpus path escapes its bound root')
        path=root/relative
        if not path.resolve().is_relative_to(root):raise ValueError('Corpus alias escapes its bound root')
        path=self.bound(path,ref['sha256'])
        if 'bytes' in ref and path.stat().st_size!=ref['bytes']:raise ValueError('Captured byte count differs')
        return path
    def arrays(self,root,ref):
        import numpy as np
        path=self.nested(root,ref)
        with zipfile.ZipFile(path) as archive:
            entries=archive.infolist()
            if (len(entries)>128 or len({x.filename for x in entries})!=len(entries)
                    or sum(x.file_size for x in entries)>128<<20 or archive.testzip() is not None):
                raise ValueError('Invalid/bounded NPZ CRC inventory')
        with np.load(path,allow_pickle=False) as source:
            if set(source.files)!=set(ref['arrays']):raise ValueError('NPZ keys differ from captured descriptors')
            data={}
            for key in source.files:
                self.guard();value=source[key];descriptor=ref['arrays'][key]
                typed=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
                if (value.dtype.hasobject or value.dtype.kind not in 'buifc' or str(value.dtype)!=descriptor['dtype']
                        or list(value.shape)!=descriptor['shape'] or typed!=descriptor['sha256']
                        or (value.dtype.kind in 'fc' and not np.isfinite(value).all())):
                    raise ValueError('Captured typed array mismatch: '+key)
                data[key]=value.copy()
        return data


def compare_commands(targets,issued,mask,cursor,*,terminal):
    import numpy as np
    mask=np.asarray(mask)
    if (targets.ndim!=2 or targets.shape[1]!=14 or len(targets)<1 or targets.dtype.kind!='f' or not np.isfinite(targets).all()
            or issued.ndim!=2 or issued.shape[1]!=14 or not 1<=len(issued)<=32
            or issued.dtype.kind!='f' or not np.isfinite(issued).all()
            or mask.size!=len(issued) or not np.isin(mask,(0,1)).all()):
        raise ValueError('One native H32/14D stream and binary actual-valid mask required')
    mask=mask.reshape(-1).astype(bool);count=int(mask.sum())
    if not np.array_equal(mask,np.arange(len(mask))<count):raise ValueError('Native valid mask is not a prefix')
    indices=np.minimum(cursor+np.arange(len(issued)),len(targets)-1)
    expected=targets[indices].astype(issued.dtype)
    error=np.abs(issued[mask]-expected[mask])
    maximum=np.zeros(14) if not count else error.max(0)
    clipped=issued.copy();clipped[:,GRIP]=np.clip(clipped[:,GRIP],0,1)
    return dict(cursor_before=cursor,cursor_after=cursor+count,issued=len(issued),valid=count,
        reference_indices=indices.tolist(),valid_mask=mask.tolist(),
        valid_reference_exact=bool(np.array_equal(issued[mask],expected[mask])),
        all_issued_reference_exact=bool(np.array_equal(issued,expected)),
        maximum_valid_error14=maximum.tolist(),arm_error=float(maximum[ARM].max()),
        left_arm_error=float(maximum[:6].max()),right_arm_error=float(maximum[7:13].max()),
        gripper_error=float(maximum[GRIP].max()),
        issued_gripper_would_clip_count=((issued[:,GRIP]<0)|(issued[:,GRIP]>1)).sum(0).tolist(),
        valid_gripper_would_clip_count=((issued[mask][:,GRIP]<0)|(issued[mask][:,GRIP]>1)).sum(0).tolist(),
        terminal_last_action_may_be_partial=bool(terminal and count>0)),expected,clipped


def drive_delivery(states,last_request,*,terminal):
    import numpy as np
    actual=np.asarray(states,dtype=np.float32).reshape(-1,14)
    if actual.shape!=(1,14):raise ValueError('Expected one public physical14 drive-target state')
    expected=np.asarray(last_request,dtype=np.float32).copy();expected[GRIP]=np.clip(expected[GRIP],0,1)
    delta=actual[0]-expected
    return dict(actual_drive14=actual[0].tolist(),last_request_after_native_gripper_clip14=expected.tolist(),
        delta14=delta.tolist(),arm_max_abs=float(np.abs(delta[ARM]).max()),
        left_arm_max_abs=float(np.abs(delta[:6]).max()),right_arm_max_abs=float(np.abs(delta[7:13]).max()),
        gripper_max_abs=float(np.abs(delta[GRIP]).max()),
        terminal_partial_possible=bool(terminal),
        meaning='Public arm drive targets and nominal gripper commands, not measured physical joint/object poses')


def jump_metrics(targets):
    import numpy as np
    difference=np.diff(targets.astype(np.float64),axis=0)
    return {name:dict(max_abs_per_joint=np.abs(difference[:,indices]).max(0).tolist(),
                     max_L2=float(np.linalg.norm(difference[:,indices],axis=1).max()))
            for name,indices in [('left_arm',list(range(6))),('right_arm',list(range(7,13))),('grippers',GRIP)]}


def ranges(mask):
    """Transition indices i mean recorded frame i-1 -> i."""
    result=[]
    for index,active in enumerate(mask,1):
        if active:
            if result and result[-1][1]==index-1:result[-1][1]=index
            else:result.append([index,index])
    return result


def gripper_events(targets):
    import numpy as np
    delta=np.diff(targets,axis=0);out={}
    for name,arm,grip in [('left',list(range(6)),6),('right',list(range(7,13)),13)]:
        still=np.array([np.allclose(a,b,atol=1e-5,rtol=0)
                        for a,b in zip(targets[1:,arm],targets[:-1,arm])])
        closing=delta[:,grip]<-1e-5;opening=delta[:,grip]>1e-5
        out[name]=dict(arm_stationary=ranges(still),gripper_closing=ranges(closing),gripper_opening=ranges(opening),
            stationary_arm_closing=ranges(still&closing),stationary_arm_opening=ranges(still&opening),
            arm_rule='np.allclose adjacent reference arm positions, atol1e-5/rtol0; chunk compression compares last retained',
            interpretation='Recorded transition ranges, not recovered dense timestamps or a causal failure attribution')
    return out


def compression_geometry(start,issued):
    """Exact deterministic compress_path selection; NEVER draw its jitter."""
    import numpy as np
    result={}
    for name,indices in [('left',list(range(6))),('right',list(range(7,13)))]:
        path=np.vstack((np.asarray(start).reshape(14)[indices],issued[:,indices]));kept=[0]
        for i in range(1,len(path)):
            if not np.allclose(path[i],path[kept[-1]],atol=1e-5,rtol=0):kept.append(i)
        result[name]=dict(input_points=len(path),retained_before_jitter=len(kept),
            retained_command_slots=[i-1 for i in kept if i>0],
            stationary_jitter_branch=len(kept)==1,
            meaning='Geometry reconstructed from captured starting drive and issued requests; no TOPP or RNG executed')
    return result


def validate_reset_identity(reset,record_info,manifest):
    """Backend adds only corpus_episode_id after the writer snapshots reset."""
    episode=manifest['episode_id']
    if (not isinstance(episode,str) or not episode
            or record_info.get('corpus_episode_id')!=episode or reset.get('episode_id')!=episode):
        raise ValueError('Reset/closed summary/manifest episode identities differ')
    remaining={key:value for key,value in record_info.items() if key!='corpus_episode_id'}
    if reset['reset_info']!=remaining:
        raise ValueError('Reset identity differs from closed summary beyond corpus_episode_id')


def panel(source_raw,native,label,path):
    """Display native RGB correctly in a PNG; never modify source JPEG bytes."""
    import cv2
    import numpy as np
    source=[source_raw['main_images'][0],*source_raw['wrist_images'][0]]
    actual=[native['main_images'][0],*native['wrist_images'][0]]
    rows=[];metrics=[]
    for name,left,right in zip(('head','left_wrist','right_wrist'),source,actual):
        for image in (left,right):
            if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:raise ValueError('Native RGB view contract differs')
        metrics.append(dict(camera=name,source_shape=list(left.shape),actual_shape=list(right.shape),
            same_shape=left.shape==right.shape,
            mean_absolute_pixel_difference=(float(np.abs(left.astype(float)-right.astype(float)).mean())
                                           if left.shape==right.shape else None)))
        cells=[]
        for prefix,image in [('source',left),('actual',right)]:
            image=cv2.resize(image,(240,180),interpolation=cv2.INTER_AREA)
            canvas=np.zeros((210,240,3),np.uint8);canvas[30:]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.putText(canvas,f'{prefix} {name}',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cells.append(canvas)
        rows.append(np.concatenate(cells,axis=1))
    canvas=np.concatenate(rows,axis=0)
    cv2.putText(canvas,label,(8,canvas.shape[0]-5),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
    if not cv2.imwrite(str(path),canvas):raise ValueError('PNG evidence write failed')
    return metrics


def inspect_replays(result,result_path,demos,output,audit,*,reset_refs):
    import numpy as np
    import h5py
    from .robotwin_demo_capacity import read_episode,raw_observation
    if (result.get('source_commit')!=PRODUCER or result.get('stage')!='teacher_executability'
            or result.get('status')!='blocked_failure' or result.get('pending') is not None):
        raise ValueError('Bind the actual stopped-before-fitting producer')
    corpus=Path(result_path).parent/'corpus'
    records={r['spec']['id']:r for r in result['records'] if r['spec']['id'] in IDS}
    if set(records)!=set(IDS) or set(reset_refs)!=set(IDS):raise ValueError('Keep exactly teacher00/01/02, including the failed02')
    if [(records[k]['success'],records[k]['primitive_steps']) for k in IDS]!=[(True,104),(True,104),(False,400)]:
        raise ValueError('This narrow diagnosis binds the actual closed three-episode outcomes')
    if result['actions']['pending_actions'] or result['actions']['uncertain_actions']:raise ValueError('Unresolved actual cost')
    absent={name:not (Path(result_path).parent/name).exists() for name in ('cache','kv','action')}
    if not all(absent.values()):raise ValueError('Unexpected cache/fit output; do not assume pre-fit failure')
    output=Path(output)
    if output.resolve().is_relative_to(Path(result_path).parent.resolve()):
        raise ValueError('Never write diagnosis into the immutable producer output')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(format=FORMAT,status='running',producer=PRODUCER,episodes=[],new_robot_actions=0,
        model_forwards=0,SGD=0,scene_identity_proven=False,automatic_retry_authorized=False)
    report['training_outputs_absent']=absent
    report['closed_producer_action_snapshot']=result['actions']
    slots=[]
    for ordinal,ident in enumerate(IDS):
        audit.guard();record=records[ident];row=demos['episodes'][ordinal]
        if row['id']!=f'phone-{ordinal:02d}':raise ValueError('Canonical whole episode identity differs')
        ref=demos['actual_whole_files'][row['id']];h5=audit.bound(ref['path'],ref['sha256'])
        state,targets=read_episode(h5,row,demos['schema'])
        mr=record['episode_manifest_ref']
        if Path(mr['root']).resolve()!=corpus.resolve():raise ValueError('Foreign corpus root')
        manifest_path=audit.nested(corpus,mr);manifest=json.loads(manifest_path.read_text())
        if manifest['pending_decision'] is not None:raise ValueError('Unclosed episode')
        reset_path=manifest_path.parent/manifest['reset']
        if Path(reset_refs[ident]['path']).resolve()!=reset_path.resolve():raise ValueError('Different bound reset path')
        reset=audit.json(reset_refs[ident])
        validate_reset_identity(reset,record['reset_info'],manifest)
        cursor=0;chunks=[];frames=[(0,reset['initial_observation'])]
        current_frame=audit.arrays(corpus,reset['initial_observation'])
        if len(manifest['chunks'])!=len(record['corpus_refs']):raise ValueError('Manifest chunk list differs')
        for number,(cref,mref) in enumerate(zip(record['corpus_refs'],manifest['chunks']),1):
            audit.guard()
            if Path(cref['root']).resolve()!=corpus.resolve():raise ValueError('Foreign chunk corpus')
            chunk=json.loads(audit.nested(corpus,cref).read_text())
            if ((manifest_path.parent/mref['path']).resolve()!=(corpus/cref['path']).resolve()
                    or mref['sha256']!=cref['sha256'] or chunk['chunk_index']!=number
                    or chunk['episode_id']!=manifest['episode_id']
                    or chunk['context']['trial_id']!=ident or chunk['observation']!=frames[-1][1]):
                raise ValueError('Captured ordered chunk/observation chain differs')
            if chunk['status']!='complete':raise ValueError('Unclosed native chunk')
            decision=audit.arrays(corpus,chunk['decision_arrays'])
            native=audit.arrays(corpus,chunk['native_result']['arrays'])
            mode=chunk['decision_metadata']['frozen_pi05_residual']['configuration']
            if (mode['kind']!='qualified_source_demo_command_replay_v1'
                    or mode['reference']['sha256']!=ref['sha256']):raise ValueError('Different actual replay source/head')
            terminal=chunk['native_result']['terminated'];truncated=chunk['native_result']['truncated']
            comparison,expected,clipped=compare_commands(targets,decision['issued_motor_joint_targets'],
                native['valid_action_mask'],cursor,terminal=terminal)
            comparison['compressed_arm_geometry']=compression_geometry(current_frame['states'],decision['issued_motor_joint_targets'])
            if (comparison['valid']!=chunk['native_result']['executed_primitive_steps']
                    or chunk['learning_result']['budget_truncated']):raise ValueError('Native count/budget boundary differs')
            next_frame=audit.arrays(corpus,chunk['next_observation'])
            if comparison['valid']:
                last=decision['issued_motor_joint_targets'][comparison['valid']-1]
                comparison['drive_delivery']=drive_delivery(next_frame['states'],last,terminal=terminal)
            comparison.update(chunk=chunk['chunk_index'],terminated=terminal,truncated=truncated,
                native_wall_seconds=chunk['native_result']['duration_seconds'],
                actual_physics_duration_available=False,source_of_physics_time='not persisted by native capture')
            for slot,valid in enumerate(comparison['valid_mask']):
                if valid:
                    delta=np.abs(decision['issued_motor_joint_targets'][slot]-expected[slot])
                    slots.append(dict(episode=ident,chunk=chunk['chunk_index'],slot=slot,
                        reference_index=comparison['reference_indices'][slot],arm_max_error=float(delta[ARM].max()),
                        gripper_max_error=float(delta[GRIP].max()),terminal_partial_possible=terminal and slot==comparison['valid']-1))
            cursor=comparison['cursor_after'];chunks.append(comparison);frames.append((cursor,chunk['next_observation']))
            current_frame=next_frame
        if (cursor!=record['primitive_steps'] or len(chunks)!=record['chunks']
                or manifest['executed_primitive_steps']!=cursor or manifest['completed_chunks']!=len(chunks)):
            raise ValueError('Episode/summary/manifest counts disagree')
        initial=audit.arrays(corpus,reset['initial_observation'])
        source_calibration={}
        with h5py.File(h5,'r') as source:
            for view in ('head_camera','left_camera','right_camera'):
                for key in ('intrinsic_cv','extrinsic_cv','cam2world_gl'):
                    name=f'observation/{view}/{key}'
                    if name in source:source_calibration[name]=np.asarray(source[name][0]).tolist()
        observations=[]
        # Source-reference midpoint and actual final point remain explicit;
        # after the recorded end, the declared replay holds its final target.
        middle=min(cursor,len(targets))//2
        selected=[frames[0],min(frames,key=lambda pair:abs(pair[0]-middle)),
                  min(frames,key=lambda pair:abs(pair[0]-cursor//2)),frames[-1]]
        for label,(position,frame_ref) in zip(('reset','reference_mid','actual_mid','final'),selected):
            index=min(max(position-1,0),len(targets)-1)
            native_frame=audit.arrays(corpus,frame_ref)
            original=raw_observation(h5,row,demos['schema'],index,state)
            path=output/f'{row["id"]}-{label}.png'
            metrics=panel(original,native_frame,f'{ident} cursor={position} sourceframe={index}',path)
            observations.append(dict(label=label,actual_cursor=position,source_frame=index,
                native_frame_sha256=frame_ref['sha256'],image=path.name,camera_metrics=metrics))
        delta=np.asarray(initial['states']).reshape(1,14)[0]-state[0]
        report['episodes'].append(dict(id=ident,success=record['success'],actual_actions=cursor,
            reference_frame_count=len(targets),expected_seed=row['reset_seed'],actual_seed=record['reset_info']['actual_seed'],
            seed_exact=record['reset_info']['actual_seed']==row['reset_seed'],
            instruction_exact=record['reset_info']['instruction']==[row['instruction']],
            initial_drive_delta14=delta.tolist(),initial_drive_max_abs=float(np.abs(delta).max()),
            source_scene_metadata=row['scene_metadata'],native_actor_pose_or_variant_metadata_available=False,
            source_camera_calibration=source_calibration,native_camera_calibration_available=False,
            reference_adjacent_jumps=jump_metrics(targets),reference_gripper_events=gripper_events(targets),
            chunks=chunks,images=observations,
            valid_reference_exact=all(c['valid_reference_exact'] for c in chunks)))
    with (output/'valid-command-comparisons.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(slots[0]));writer.writeheader();writer.writerows(slots)
    report.update(status='audit_complete',actual_valid_commands_compared=len(slots),
        all_valid_reference_exact=all(e['valid_reference_exact'] for e in report['episodes']),
        input_files=audit.files,input_bytes=audit.bytes,
        findings=['Exact reference requests, drive-target end states and physical trajectories are different claims.',
            'Terminal success may occur during the last valid command; that end-state discrepancy is not automatically a delivery bug.',
            'Native capture persists wall durations and valid command counts, not per-arm TOPP/physics timing or measured object/joint poses.',
            'Author dense execution/save_freq15 plus boundary captures differs from native chunk compression/TOPP/downsampling and gripper retiming.',
            'Pinned compress_path adds Gaussian jitter with std2e-4 when an arm path compresses to one point. This is a standard deviation, not a hard error tolerance.',
            'Producer reset restores global NumPy state afterward; private motor_rng covers only the VLA Gaussian. Equal motor seeds do not establish paired native planner randomness.',
            'Small drive-target residuals may include that native planner jitter; neither jitter nor cadence is established as the cause of teacher02 failure by source inspection.',
            'Saved image differences diagnose possible scene/camera compatibility; expert commands do not depend on visual policy features.',
            'No demonstration exclusion, trained-class rejection, refit or GPU retry follows automatically.'])
    size=sum(p.stat().st_size for p in output.iterdir())
    if size>16<<20:raise ValueError('Small diagnostic evidence exceeds16MiB')
    for path,ref in audit.files.items():audit.bound(path,ref['sha256'])
    json_write(output/'report.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('result','demos','source-evidence','allocation','zero-gpu-proof'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    args=p.parse_args(argv)
    if sys.platform!='linux':raise RuntimeError('Actual allocated Linux CPU only')
    if not math.isfinite(args.deadline_unix):raise ValueError('Original finite absolute deadline required')
    boot=Audit(args.deadline_unix)
    ticket=boot.json(dict(path=str(args.allocation),sha256=args.allocation_sha256))
    proof=boot.json(dict(path=str(args.zero_gpu_proof),sha256=args.zero_gpu_proof_sha256))
    def git(*values):
        boot.guard()
        return subprocess.check_output(['git',*values],text=True,timeout=max(.1,min(10,args.deadline_unix-time.time()))).strip()
    source=git('rev-parse','HEAD')
    if source!=args.source_commit or git('status','--porcelain'):raise ValueError('Exact clean source required')
    if (ticket.get('gpu_count')!=0 or ticket.get('hostname')!=socket.gethostname()
            or ticket.get('source_commit')!=source or ticket.get('phase_id')!='frozen-pi05-endtoend-20261031-r1'
            or not 1<=ticket['maximum_cpu_worker_seconds']<=300
            or ticket['prior_phase_cpu_charge']<746 or ticket['maximum_phase_cpu_worker_seconds']!=1800
            or ticket['prior_phase_cpu_charge']+ticket['maximum_cpu_worker_seconds']>1800
            or Path(ticket['inspection_output']).resolve()!=args.output.resolve()
            or args.output.resolve()!=Path(ticket['output']).resolve()/'inspection'
            or proof.get('passed') is not True or proof.get('hostname')!=socket.gethostname()
            or proof.get('device_count_before_mask')!=0 or proof.get('cuda_initialized_before_mask') is not False
            or proof.get('actual_cpu_cores',0)<2 or proof.get('actual_memory_bytes',0)<(16<<30)
            or ticket.get('zero_gpu_attestation_sha256')!=args.zero_gpu_proof_sha256):
        raise ValueError('Actual CPU-only allocation/proof/input/output binding required')
    start=ticket['started_unix'];deadline=ticket['deadline_unix']
    if deadline!=args.deadline_unix or not start<=time.time()<deadline<=start+ticket['maximum_cpu_worker_seconds']:
        raise ValueError('Reuse original CPU clock including setup/tests; no reset/extension')
    audit=Audit(deadline)
    result=audit.json(dict(path=str(args.result),sha256=args.result_sha256))
    demos=audit.json(dict(path=str(args.demos),sha256=args.demos_sha256))
    evidence=audit.json(dict(path=str(args.source_evidence),sha256=args.source_evidence_sha256))
    if (evidence.get('format')!='capacity_replay_static_source_handoff_v1'
            or evidence.get('robotwin_commit')!='dca9ec682688821c42944eacd7f5dd9bfb15396f'
            or evidence.get('source_sha256')!='6b3f74b038509ef989913b3f6f37c74e1d0b7156d201a1bf0100f9ccb6b0d46b'):
        raise ValueError('Pinned native cadence/reset-source evidence required')
    reset_refs={r['episode']:dict(path=r['reset_path'],sha256=r['reset_sha256'])
                for r in evidence['already_saved_teacher_reset_metadata']}
    if result['demos_sha256']!=args.demos_sha256:raise ValueError('Different admitted demo manifest')
    try:
        report=inspect_replays(result,args.result,demos,args.output,audit,reset_refs=reset_refs)
    except BaseException as error:
        if args.output.is_dir():json_write(args.output/'failure.json',dict(type=type(error).__name__,error=str(error),input_files=audit.files))
        raise
    if any(k.split('.')[0] in ('jax','rlinf','openpi','sapien') for k in sys.modules):raise RuntimeError('Forbidden model/simulator imports')
    print(json.dumps(dict(status=report['status'],valid_commands=report['actual_valid_commands_compared'],
        all_valid_reference_exact=report['all_valid_reference_exact'],new_robot_actions=0,model_forwards=0,SGD=0)))


if __name__=='__main__':main()
