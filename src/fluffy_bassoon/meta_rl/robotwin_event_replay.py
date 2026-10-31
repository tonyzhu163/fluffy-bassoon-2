"""Native-H32 event replay and canonical-rollout SL cache; no scheduler.

Only the admitted Linux GPU driver calls these model/simulator helpers.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from fluffy_bassoon.rl.pi05_residual import DecodedActionResidual
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
from .robotwin_event_schedule import build_schedule, scheduled_request, advance_cursor, validate_teacher_bank
from .robotwin_reactive_fit import ReactiveEpisodeControl, ReplayTeacherHead, ARM_INDICES, cpu
from .robotwin_demo_capacity import file_sha, read_episode, transformed_target
from .robotwin_replay_diagnosis import Audit, validate_reset_identity
from .warmstart import atomic_save


def numpy_state():
    value=np.random.get_state()
    return dict(kind=value[0],keys=value[1].tolist(),position=value[2],has_gaussian=value[3],cached_gaussian=value[4])


def physical_audit(task):
    """Read existing physics objects only; no get_obs/render/step/check_success.

Missing optional native fields remain explicitly unavailable, not a fabricated
scene-equality prerequisite. None of this enters public inputs/reset identity.
    """
    before=numpy_state()
    def plain(value):
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if value is None or isinstance(value,(str,int,float,bool)):return value
        raise ValueError('Non-JSON native audit field')
    def read(function):
        try:return dict(available=True,value=plain(function()))
        except (AttributeError,TypeError,ValueError,RuntimeError) as error:
            return dict(available=False,reason=type(error).__name__+': '+str(error))
    actors={}
    for name in ('phone','stand'):
        def actor(name=name):
            pose=getattr(task,name).get_pose()
            return dict(variant_id=getattr(task,name+'_id'),p=pose.p,q=pose.q)
        actors[name]=read(actor)
    robot=task.robot;articulations={};grippers={}
    for side in ('left','right'):
        def articulation(side=side):
            entity=getattr(robot,side+'_entity');joints=entity.get_active_joints()
            return dict(qpos=entity.get_qpos(),joint_names=[j.get_name() for j in joints],
                        drive_targets=[float(j.get_drive_target()[0]) for j in joints])
        articulations[side]=read(articulation)
        grippers[side]=read(lambda side=side:dict(nominal=getattr(robot,side+'_gripper_val'),
            named_drive_targets=[dict(name=j[0].get_name(),target=float(j[0].get_drive_target()[0]))
                                for j in getattr(robot,side+'_gripper') if j is not None]))
    # Calibration is copied only if the last existing native observation
    # already contains it. No camera getter or additional observation call.
    cameras={}
    for name,camera in getattr(task,'now_obs',{}).get('observation',{}).items():
        if isinstance(camera,dict):
            calibration={k:camera[k] for k in ('intrinsic_cv','extrinsic_cv','cam2world_gl') if k in camera}
            if calibration:cameras[name]=plain(calibration)
    result=dict(task_info=read(lambda:task.get_info()),actors=actors,articulations=articulations,grippers=grippers,
        already_available_camera_calibration=cameras,
        conventions=dict(actor_position='world meters',actor_quaternion='native SAPIEN wxyz',
            qpos='named native articulation coordinates, revolute rad/prismatic meters',
            gripper_nominal='normalized command variable, separate from measured qpos',
            drive_targets='native joint coordinates, named joints'),
        native_counters={k:plain(getattr(task,k,None)) for k in ('take_action_cnt','run_steps','FRAME_IDX')},
        meaning='Audit only; measured articulation qpos separate from nominal/drive targets; no physics-time alignment claim')
    if numpy_state()!=before:raise RuntimeError('Read-only physical audit consumed global NumPy randomness')
    return result


class EventReplayHead(ReplayTeacherHead):
    def __init__(self):
        super().__init__();self.schedule=None;self.chunk_index=0
    def config(self):
        return dict(kind='original_gripper_event_h32_replay_v1',public_dim=2062,
            reference=self.reference,schedule=self.schedule,
            policy_inputs='current public2062 only; schedule/cursor are expert audit, never learned-head inputs')
    def forward(self,base,features):
        if self.targets is None or self.schedule is None:raise ValueError('No complete aligned reference')
        # Schedule contains indices only. Targets retain their original bytes.
        if self.chunk_index<len(self.schedule['chunks']):
            chunk=self.schedule['chunks'][self.chunk_index]
            indices=(torch.arange(base.shape[1],device=base.device)+chunk['start']).clamp(max=chunk['end']-1)
        else:
            indices=torch.full((base.shape[1],),len(self.targets)-1,device=base.device,dtype=torch.long)
        targets=self.targets.to(base.device)[indices][None].to(base.dtype)
        delta=targets-base
        return DecodedActionResidual(base.detach(),delta,delta,torch.zeros_like(base,dtype=torch.bool),targets)


class EventEpisodeControl(ReactiveEpisodeControl):
    """Reuse the common complete-episode loop and unchanged env.step(noise)."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.schedule=None;self.cursor=dict(chunk_index=0,physical_slots=0,source_cursor=0)
        self.last_audit=None
    def select_teacher(self,targets,reference,*,aligned):
        if self.active:raise ValueError('Never change teacher inside an active episode')
        self.replay=EventReplayHead().eval() if aligned else ReplayTeacherHead().eval()
        self.replay.targets=torch.as_tensor(targets,device=self.env.backend.model.device)
        self.replay.reference=deepcopy(reference)
        self.schedule=build_schedule(targets) if aligned else None
        if self.schedule is not None:
            if not self.schedule['admissible']:raise ValueError('Schedule cannot issue every original frame within400')
            self.replay.schedule=self.schedule
    def clear_teacher(self):
        if self.active:raise ValueError('Never clear an active teacher')
        self.replay.targets=None;self.replay.reference=None;self.schedule=None
    def reset(self,*args,**kwargs):
        self.cursor=dict(chunk_index=0,physical_slots=0,source_cursor=0)
        self.last_audit=None
        result=super().reset(*args,**kwargs)
        self.reset_physical_audit=physical_audit(self.env.backend.env.venv.envs[0].task)
        return result
    def step(self,values):
        before=numpy_state();cursor=deepcopy(self.cursor)
        physical_before=physical_audit(self.env.backend.env.venv.envs[0].task)
        if self.schedule is not None:
            self.replay.chunk_index=cursor['chunk_index']
            self._pending_metadata=dict(self._pending_metadata,event_schedule=dict(
                format=self.schedule['format'],teacher_reference=self.replay.reference,
                chunk_index=cursor['chunk_index'],source_cursor_before=cursor['source_cursor'],
                physical_slots_before=cursor['physical_slots']))
        self._pending_metadata=dict(self._pending_metadata,native_global_numpy_before=before,
                                    physical_before_audit=physical_before)
        result=super().step(values)  # H32; never max_primitive_steps or a second simulator loop.
        _,_,terminated,truncated,info=result
        if self.schedule is not None:
            self.cursor=advance_cursor(self.schedule,cursor['chunk_index'],cursor['physical_slots'],
                cursor['source_cursor'],actual=info['primitive_steps'],issued=info['issued_primitive_steps'],
                closed=terminated or truncated)
        relative=info.get('corpus_chunk')
        if relative is None:raise RuntimeError('Actual native result lacks canonical chunk')
        path=self.env.corpus_writer.root/relative
        self.last_audit=dict(corpus_chunk=dict(root=str(self.env.corpus_writer.root),path=relative,sha256=file_sha(path)),
            native_global_numpy_before=before,native_global_numpy_after=numpy_state(),
            physical_before_audit=physical_before,
            physical_after_audit=physical_audit(self.env.backend.env.venv.envs[0].task),
            cursor_before=cursor,cursor_after=deepcopy(self.cursor),
            meaning='Global native planner RNG audit only; no reseeding/private physics generator')
        return result
    def snapshot(self):
        return dict(super().snapshot(),source_schedule=deepcopy(self.schedule),
            source_cursor=deepcopy(self.cursor),teacher_reference=deepcopy(self.replay.reference),
            last_native_rng_audit=deepcopy(self.last_audit))
    def publish_reset_audit(self,output):
        path=Path(output)/self.last_reset_info['corpus_episode_id']/'reset.json'
        _atomic_json(path,dict(corpus_episode_id=self.last_reset_info['corpus_episode_id'],
            physical=self.reset_physical_audit,native_global_numpy=numpy_state()))
        return dict(path=str(path),sha256=file_sha(path))
    def publish_chunk_audit(self,output):
        if self.last_audit is None:raise RuntimeError('Missing completed native audit')
        relative=Path(self.last_audit['corpus_chunk']['path'])
        path=Path(output)/relative.parts[1]/relative.name
        _atomic_json(path,self.last_audit)
        return dict(path=str(path),sha256=file_sha(path))


def cache_rollouts(backend,demos,paths,records,output,*,guard,deadline,action_head):
    """Expert PLAN imitation on new aligned successes and actual current inputs.

No HDF5 images, future event/cursor fields, or control/old-rollout rows become
policy inputs. Original files supply reference command provenance only. Native
plans use the full submitted chunk before physics: the executed mask is cost
accounting, not a causal label mask. Loss covers all submitted plan parameters,
including terminal-unexecuted ones; unsubmitted natural-horizon tail is masked.
    """
    selected=validate_teacher_bank(records,demos)
    output=Path(output);output.mkdir(exist_ok=False)
    backend.kv_adapter=backend.action_residual=None
    backend._paired_instruction=None
    cached=[]
    for episode in demos['episodes']:
        guard();row=selected[episode['id']];source=paths[episode['id']]
        if file_sha(source['path'])!=source['sha256']:raise ValueError('Original demo bytes changed')
        _,targets=read_episode(source['path'],episode,demos['schema'])
        schedule=build_schedule(targets)
        if not schedule['admissible']:raise ValueError('Rejected schedule cannot enter cache')
        audit=Audit(deadline);ref=row['episode_manifest_ref'];root=Path(ref['root'])
        path=audit.nested(root,ref);manifest=audit.json(dict(path=str(path),sha256=ref['sha256']))
        reset_path=path.with_name('reset.json')
        reset=audit.json(dict(path=str(reset_path),sha256=file_sha(reset_path)))
        validate_reset_identity(reset,row['reset_info'],manifest)
        if (manifest['pending_decision'] is not None or len(manifest['chunks'])!=row['chunks']
                or manifest['executed_primitive_steps']!=row['primitive_steps']):
            raise ValueError('Only complete canonical teacher episodes enter SL')
        cursor=dict(chunk_index=0,physical_slots=0,source_cursor=0)
        for ordinal,chunk_ref in enumerate(manifest['chunks']):
            guard();chunk_path=audit.nested(path.parent,chunk_ref)
            chunk=audit.json(dict(path=str(chunk_path),sha256=chunk_ref['sha256']))
            if (chunk['status']!='complete' or chunk['learning_result']['budget_truncated']
                    or chunk['context']['trial_id']!=row['spec']['id']):
                raise ValueError('Wrong/incomplete canonical teacher chunk')
            decision=audit.arrays(root,chunk['decision_arrays'])
            native=audit.arrays(root,chunk['native_result']['arrays'])
            frame=chunk['observation'];arrays=audit.arrays(root,frame)
            target,indices,new=scheduled_request(targets,schedule,ordinal)
            issued=decision['issued_motor_joint_targets'];n=len(issued)
            mask=native['valid_action_mask'].reshape(-1)
            actual=chunk['native_result']['executed_primitive_steps']
            if (not np.array_equal(issued,target[:n]) or mask.size!=n
                    or not np.array_equal(mask,np.arange(n)<actual)):
                raise ValueError('Actual issued commands/mask differ from the declared event schedule')
            expert_ref=chunk['decision_metadata']['frozen_pi05_residual']['actions']
            if not np.array_equal(decision[expert_ref['numeric_array']][0],target):
                raise ValueError('Full proposed expert H32 plan differs from its fixed reference')
            metadata=chunk['decision_metadata']['event_schedule']
            if (metadata['chunk_index']!=ordinal or metadata['source_cursor_before']!=cursor['source_cursor']
                    or metadata['physical_slots_before']!=cursor['physical_slots']
                    or metadata['teacher_reference']['sha256']!=source['sha256']):
                raise ValueError('Expert audit cursor/reference differs from actual canonical request')
            closed=bool(chunk['learning_result']['terminated'] or chunk['learning_result']['truncated'])
            cursor=advance_cursor(schedule,ordinal,cursor['physical_slots'],cursor['source_cursor'],
                                  actual=actual,issued=n,closed=closed)
            if closed!=(ordinal==len(manifest['chunks'])-1):raise ValueError('Teacher has an unclosed/early-closed chunk chain')
            if closed and chunk['native_result']['success'] is not True:
                raise ValueError('Closed canonical aligned teacher did not succeed')
            if actual<1:raise ValueError('Cannot cache an empty native command')
            raw={key:arrays[key] for key in ('main_images','wrist_images','states')}
            raw['task_descriptions']=frame['language']
            if raw['task_descriptions']!=[episode['instruction']]:raise ValueError('Actual rollout language differs')
            public=backend._encode(raw)
            if not np.array_equal(np.asarray(public),arrays['features']):raise ValueError('Canonical public features do not reproduce exactly')
            observation,target32,roundtrip=transformed_target(backend.model,raw,target)
            if not torch.equal(observation.state,backend._prepared.state):raise ValueError('Actual input normalized state changed')
            # Reuse the EXACT recorded proposal draw. Hooks are disabled, and
            # no teacher action/noise/future schedule enters the learned head.
            noise=decision['sampled_noise'];proposal=backend.decode(noise)
            base_ref=chunk['decision_metadata']['frozen_pi05_residual']['base_actions']
            base=decision[base_ref['numeric_array']][0]
            if not np.array_equal(proposal,base):raise ValueError('Recorded native proposal does not reproduce exactly')
            # Known submitted plan labels and actual native execution/cost
            # masks deliberately differ on early success. No executed tail is
            # invented and no later-waypoint causal independence is assumed.
            valid=np.arange(32)<n
            executed=np.zeros(32,dtype=bool);executed[:n]=mask.astype(bool)
            lower=proposal-action_head.bounds.cpu().numpy();upper=proposal+action_head.bounds.cpu().numpy()
            lower[:,ARM_INDICES]=np.maximum(lower[:,ARM_INDICES],action_head.arm_lower.cpu().numpy())
            upper[:,ARM_INDICES]=np.minimum(upper[:,ARM_INDICES],action_head.arm_upper.cpu().numpy())
            if np.any(lower>upper):raise ValueError('Actual native proposal exceeds arm headroom')
            outside=np.abs(target-np.maximum(lower,np.minimum(upper,target)))
            support=dict(valid_actions=int(valid.sum()),outside_counts=(outside[valid]>0).sum(0).tolist(),
                maximum_absolute_delta=np.abs(target-proposal)[valid].max(0).tolist(),
                attainable_MSE_floor=float((outside[valid]**2).mean()))
            data=cpu(dict(state=backend._prepared.state,features=torch.as_tensor(public,device=backend.model.device)[None],
                cache=backend._kv_cache,mask=backend._prefix_mask,target32=target32,
                teacher14=torch.as_tensor(target,device=backend.model.device)[None],
                base14=torch.as_tensor(proposal,device=backend.model.device)[None],valid=torch.as_tensor(valid)[None]))
            dest=output/f'sample-{len(cached):04d}.pt'
            digest=atomic_save(dest,dict(format='reactive_canonical_event_cache_v1',episode=episode,
                canonical_chunk=dict(path=str(chunk_path),sha256=chunk_ref['sha256']),data=data,
                original_reference=source,reference_indices=indices.tolist(),roundtrip=roundtrip,
                full_known_proposed_plan14=torch.from_numpy(target.copy()),
                submitted_plan_mask=torch.from_numpy(valid.copy()),
                native_executed_mask=torch.from_numpy(executed.copy()),
                native_valid_reference_slots=int((new&executed).sum()),native_valid_hold_slots=int((~new&executed).sum()),
                submitted_reference_slots=int((new&valid).sum()),submitted_hold_slots=int((~new&valid).sum()),
                proposal_noise=torch.from_numpy(noise.copy()),
                interpretation='Expert submitted-plan imitation; native executed mask is separately preserved cost accounting'))
            cached.append(dict(path=str(dest),sha256=digest,split=episode['split'],episode_id=episode['id'],
                anchor=ordinal,valid_actions=int(valid.sum()),roundtrip=roundtrip,
                physical_residual_support=support,native_actual_actions=int(executed.sum()),
                loss_mask_semantics='all known submitted plan slots, not native executed prefix',
                actual_source_cursor_after=cursor['source_cursor']))
        if cursor['physical_slots']!=row['primitive_steps']:raise ValueError('Canonical cache physical totals differ')
    return cached
