"""One matched-length phone04 closure-lead pair; no scheduler or fitter."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import math

import numpy as np
import torch

from fluffy_bassoon.rl.pi05_residual import DecodedActionResidual
from .checkpoint import restore_rng,rng_state
from .robotwin_event_replay import EventEpisodeControl
from .robotwin_event_schedule import build_schedule
from .robotwin_reactive_fit import ReplayTeacherHead

FORMAT='robotwin_phone04_prelift_closure_pair_v1'
FACTORIAL_FORMAT='robotwin_phone04_prelift_dose_grip_factorial_v1'
CONDITIONS=('hold_current','close_endpoint')
FACTORIAL_CASES=((1,'hold_current'),(1,'close_endpoint'),(5,'hold_current'),(5,'close_endpoint'))
FACTORIAL_PRIORITY=('dose1-close_endpoint','dose1-hold_current','dose5-close_endpoint','dose5-hold_current')
JOINT=6


def sha_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def probe_schedule(targets,probe_holds=1,*,protocol=FORMAT):
    """Keep the qualified one-hold default; new factorial admits only1 or5."""
    if (type(probe_holds) is not int or probe_holds not in (1,5)
            or protocol not in (FORMAT,FACTORIAL_FORMAT) or (protocol==FORMAT and probe_holds!=1)):
        raise ValueError('Exactly the old one hold or prospectively declared factorial1/5 holds')
    schedule=build_schedule(targets)
    closing=[e for e in schedule['events'] if e['joint']==JOINT and e['direction']==-1]
    if (len(targets)!=140 or not closing or closing[0]['start']!=56
            or closing[0]['end_exclusive']!=70 or targets[69,JOINT]!=0.
            or len(schedule['chunks'])!=7 or schedule['chunks'][2]['end']!=70
            or schedule['minimum_slots_to_issue_every_original_frame']!=198):
        raise ValueError('Exact source04 first-closing phase/reference schedule required')
    entries=[]
    for index,chunk in enumerate(schedule['chunks']):
        entries.append(dict(kind='source',source_chunk=index,start=chunk['start'],end=chunk['end']))
        if index==2:
            for repeat in range(probe_holds):
                entry=dict(kind='probe',start=70,end=70,source_chunk=None)
                if protocol==FACTORIAL_FORMAT:entry.update(probe_repetition_index=repeat,probe_holds=probe_holds)
                entries.append(entry)
    return dict(format=protocol,source_schedule=schedule,entries=entries,reference_frames=140,
        insertion_after_source_chunk=2,original_full_slots=224,full_slots_with_probe=224+32*probe_holds,
        minimum_slots_to_issue_every_original_frame=198+32*probe_holds,native_episode_limit=400,
        target_kind='current public12 arm drives+2 nominal grippers; change only active grip for treatment')


def probe_request(current14,close_value,condition):
    current=np.asarray(current14,dtype=np.float32).reshape(-1)
    if current.shape!=(14,) or not np.isfinite(current).all() or condition not in CONDITIONS or close_value!=0.:
        raise ValueError('Current finite public14/original closeendpoint0/declared condition required')
    request=np.repeat(current[None],32,axis=0)
    if condition=='close_endpoint':request[:,JOINT]=close_value
    return request


def preflight04(profile,demos,*,guard):
    """Called by the owner's already attested allocated-CPU gate, no models."""
    from pathlib import Path
    from .robotwin_demo_capacity import file_sha,read_episode,validate_demo_manifest
    guard();validate_demo_manifest(demos)
    row=next(r for r in demos['episodes'] if r['id']=='phone-04')
    ref=demos['actual_whole_files']['phone-04'];path=Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or file_sha(path)!=ref['sha256']:
        raise ValueError('Exact original04 file required, no new teacher')
    guard();_,targets=read_episode(path,row,demos['schema'])
    if profile['format']==FACTORIAL_FORMAT:
        plan={str(n):probe_schedule(targets,n,protocol=FACTORIAL_FORMAT) for n in (1,5)}
        if plan!=profile['expected_schedules']:raise ValueError('Fixed factorial real04 schedules differ')
    else:
        plan=probe_schedule(targets)
        if plan!=profile['expected_schedule']:raise ValueError('Fixed real04 profile schedule differs')
    guard()
    return dict(format=profile['format'],status='schedule_pass_only',source=ref,schedule=plan,
        close_endpoint=float(targets[69,6]),whole_original_bank_retained=True,
        new_robot_actions=0,model_forwards=0,SGD=0,
        interpretation='Plan qualification only; no physical/mechanism success or GPU admission')


@contextmanager
def observe_native_plans(task):
    """Optional metadata on existing calls; no repeat/replan/scene/RNG call.

Record unavailable instrumentation and preserve native behavior when instance
methods cannot be shadowed. Original results/exceptions pass through unchanged.
    """
    record=dict(available=False,topp=[],downsample=[],unavailable=[])
    restored=[];position_sides=[]
    def count(value):
        try:return len(value)
        except (TypeError,AttributeError):return None
    def scalar(value):
        try:
            if not isinstance(value,(int,float,np.integer,np.floating)):return None
            result=float(value)
            return result if math.isfinite(result) else None
        except (TypeError,ValueError,OverflowError):return None
    def install(obj,name,wrapper):
        original=getattr(obj,name)
        namespace=getattr(obj,'__dict__',None)
        descriptor=inspect.getattr_static(type(obj),name,None)
        if not isinstance(namespace,dict) or hasattr(descriptor,'__set__'):
            raise AttributeError('No safely shadowable instance method '+name)
        existed=name in namespace
        setattr(obj,name,wrapper(original))
        restored.append((obj,name,original,existed))
    try:
        robot=task.robot
        left,right=robot.left_mplib_planner,robot.right_mplib_planner
        if left is right:raise AttributeError('Shared planner object cannot identify arm without altering calls')
        for side,planner in [('left',left),('right',right)]:
            def make_topp(original,side=side):
                def wrapped(*args,**kwargs):
                    row=dict(side=side,input_points=count(args[0]) if args else None,status='called')
                    record['topp'].append(row)
                    try:result=original(*args,**kwargs)
                    except BaseException as error:
                        row.update(status='raised',exception=type(error).__name__);raise
                    row['status']='returned'
                    try:
                        position=result[1]
                        position_sides.append((position,side))
                        row.update(raw_position_steps=count(position),duration=scalar(result[-1]))
                    except (TypeError,IndexError,AttributeError):row['metadata_unavailable']=True
                    return result
                return wrapped
            install(planner,'TOPP',make_topp)
        def make_downsample(original):
            def wrapped(*args,**kwargs):
                result=original(*args,**kwargs)
                sides={side for position,side in position_sides if args and position is args[0]}
                row=dict(side=next(iter(sides)) if len(sides)==1 else None,
                    input_position_steps=count(args[0]) if args else None)
                if len(sides)!=1:row['side_unavailable']='ambiguous_position_identity' if sides else 'unknown_position_identity'
                try:row['output_position_steps']=count(result[0])
                except (TypeError,IndexError,AttributeError):row['metadata_unavailable']=True
                record['downsample'].append(row)
                return result
            return wrapped
        install(task,'downsample_trajectory',make_downsample)
        record['available']=True
    except (AttributeError,TypeError) as error:
        record['unavailable'].append(type(error).__name__+': '+str(error))
        for obj,name,original,existed in reversed(restored):
            if existed:setattr(obj,name,original)
            else:delattr(obj,name)
        restored=[]
    try:yield record
    finally:
        for obj,name,original,existed in reversed(restored):
            if existed:setattr(obj,name,original)
            else:delattr(obj,name)


class PreliftHead(ReplayTeacherHead):
    def __init__(self):
        super().__init__();self.plan=None;self.execution_index=0;self.probe_target=None;self.condition=None
    def config(self):
        return dict(kind=FORMAT if self.plan is None else self.plan['format'],reference=self.reference,plan=self.plan,condition=self.condition,
            policy_inputs='expert diagnostic only; no learned policy/cache/fit admission')
    def forward(self,base,features):
        if self.targets is None or self.plan is None:raise ValueError('No complete source04 reference')
        entries=self.plan['entries']
        entry=entries[self.execution_index] if self.execution_index<len(entries) else None
        if entry is not None and entry['kind']=='probe':
            if self.probe_target is None:raise ValueError('Probe target must be captured before native execution')
            target=self.probe_target.to(base.device)[None].to(base.dtype)
        else:
            if entry is None:indices=torch.full((32,),len(self.targets)-1,dtype=torch.long,device=base.device)
            else:indices=(torch.arange(32,device=base.device)+entry['start']).clamp(max=entry['end']-1)
            target=self.targets.to(base.device)[indices][None].to(base.dtype)
        delta=target-base
        return DecodedActionResidual(base.detach(),delta,delta,torch.zeros_like(base,dtype=torch.bool),target)


class PreliftControl(EventEpisodeControl):
    """Same full native step/physical audit, explicit inserted-command cursor."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.probe_plan=None;self.condition=None;self.probe_current14=None;self.prelude=[];self.probe_profile_ref=None
    def select_probe(self,targets,reference,condition,*,probe_holds=1,protocol=FORMAT):
        if self.active or condition not in CONDITIONS:raise ValueError('Closed control/declared condition required')
        self.probe_plan=probe_schedule(targets,probe_holds,protocol=protocol);self.schedule=None
        self.replay=PreliftHead().eval();self.replay.targets=torch.as_tensor(targets,device=self.env.backend.model.device)
        self.replay.plan=self.probe_plan;self.replay.reference=deepcopy(reference);self.replay.condition=condition
        self.condition=condition;self.probe_current14=None;self.prelude=[]
    def reset(self,*args,**kwargs):
        result=super().reset(*args,**kwargs)
        self.cursor=dict(chunk_index=0,physical_slots=0,source_cursor=0,probe_submitted=False)
        self.prelude=[];self.probe_current14=None;self.replay.probe_target=None
        return result
    def step(self,values):
        before=deepcopy(self.cursor);index=before['chunk_index'];entries=self.probe_plan['entries']
        entry=entries[index] if index<len(entries) else dict(kind='tail',start=140,end=140,source_chunk=None)
        self.replay.execution_index=index
        if entry['kind']=='probe' and self.probe_current14 is None:
            raw=self.env.backend._raw_obs['states']
            current=(raw.detach().cpu().numpy() if isinstance(raw,torch.Tensor) else np.asarray(raw)).reshape(14)
            self.probe_current14=current.astype(np.float32).copy()
            request=probe_request(current,0.,self.condition)
            self.replay.probe_target=torch.as_tensor(request,device=self.env.backend.model.device)
        self._pending_metadata=dict(self._pending_metadata,prelift_probe=dict(format=self.probe_plan['format'],condition=self.condition,
            profile=deepcopy(self.probe_profile_ref),
            entry=entry,execution_index=index,source_cursor_before=before['source_cursor'],
            physical_slots_before=before['physical_slots'],teacher_reference=deepcopy(self.replay.reference),
            probe_anchor_public14=None if self.probe_current14 is None else self.probe_current14.copy(),
            explicit_probe_target=None if entry['kind']!='probe' else self.replay.probe_target.detach().cpu()))
        with observe_native_plans(self.env.backend.env.venv.envs[0].task) as plans:
            result=super().step(values)
        _,_,terminated,truncated,info=result
        n=info['primitive_steps'];issued=info['issued_primitive_steps'];closed=bool(terminated or truncated)
        if not 0<n<=issued<=32 or (not closed and (n!=32 or issued!=32)):
            raise RuntimeError('No shortened nonterminal/native-budget-truncated commands')
        if entry['kind']=='source':
            if before['source_cursor']!=entry['start']:raise RuntimeError('Source frame skipped/repeated')
            consumed=min(n,entry['end']-entry['start'])
        else:consumed=0
        self.cursor=dict(chunk_index=index+1,physical_slots=before['physical_slots']+n,
            source_cursor=before['source_cursor']+consumed,
            probe_submitted=before['probe_submitted'] or entry['kind']=='probe',closed=closed)
        self.last_audit.update(cursor_before=before,cursor_after=deepcopy(self.cursor),
            native_plan_observation=plans,probe_entry=entry,condition=self.condition)
        for observation in plans['downsample']:
            steps=observation.get('output_position_steps')
            if type(steps) is int and steps>0:
                observation.update(submitted_gripper_parameters=issued,
                    derived_zero_duration_trailing_gripper_slots=max(0,issued-steps),
                    allocation_rule='Pinned native quotient/remainder interpolation; not an observed physics waypoint trace')
        if entry.get('source_chunk') in (0,1,2) and entry['kind']=='source':
            chunk=json.loads((self.env.corpus_writer.root/info['corpus_chunk']).read_text())
            self.prelude.append(dict(source_chunk=entry['source_chunk'],
                observation=chunk['observation']['content_sha256'],next_observation=chunk['next_observation']['content_sha256'],
                issued_actions=chunk['decision_arrays']['arrays']['issued_motor_joint_targets']['sha256'],
                sampled_noise=chunk['decision_arrays']['arrays']['sampled_noise']['sha256'],
                private_motor_rng_before=sha_json(chunk['decision_metadata']['capacity_motor_rng_before']),
                private_motor_rng_after=sha_json(chunk['decision_metadata']['capacity_motor_rng_after']),
                native_mask=chunk['native_result']['arrays']['arrays']['valid_action_mask']['sha256'],
                physical_before=sha_json(self.last_audit['physical_before_audit']),
                physical_after=sha_json(self.last_audit['physical_after_audit']),
                global_numpy_before=sha_json(self.last_audit['native_global_numpy_before']),
                global_numpy_after=sha_json(self.last_audit['native_global_numpy_after'])))
        return result
    def snapshot(self):
        return dict(super().snapshot(),probe_plan=deepcopy(getattr(self,'probe_plan',None)),
            condition=getattr(self,'condition',None),probe_current14=None if self.probe_current14 is None else self.probe_current14.tolist(),
            probe_target14=None if getattr(self.replay,'probe_target',None) is None else self.replay.probe_target.detach().cpu().tolist(),
            replay_execution_index=getattr(self.replay,'execution_index',None),
            profile=deepcopy(self.probe_profile_ref),
            prelude_signatures=deepcopy(getattr(self,'prelude',[])))


def assert_restored_rng(expected):
    actual=rng_state()
    if (actual['python']!=expected['python'] or actual['numpy'][0]!=expected['numpy'][0]
            or not np.array_equal(actual['numpy'][1],expected['numpy'][1])
            or actual['numpy'][2:]!=expected['numpy'][2:] or not torch.equal(actual['torch'],expected['torch'])
            or ((actual['cuda'] is None)!=(expected['cuda'] is None))
            or (actual['cuda'] is not None and (len(actual['cuda'])!=len(expected['cuda'])
                or any(not torch.equal(a,b) for a,b in zip(actual['cuda'],expected['cuda']))))):
        raise ValueError('Prospective global RNG fork did not restore all four existing domains exactly')


@contextmanager
def preserve_global_rng():
    """Keep observational qualification outside the paired physics RNG path."""
    saved=rng_state()
    try:yield
    finally:
        restore_rng(saved);assert_restored_rng(saved)


def run_pair(specs,episode,global_rng,*,restore=restore_rng,verify=assert_restored_rng):
    """Prospective paired RNG fork before TWO fresh complete resets, not retry."""
    if [s['condition'] for s in specs]!=list(CONDITIONS) or len({s['id'] for s in specs})!=2:
        raise ValueError('Exactly one predeclared control/treatment; no outcome-based selection')
    results=[]
    for index,spec in enumerate(specs):
        restore(deepcopy(global_rng))
        verify(global_rng)
        results.append(episode(deepcopy(spec),qualify=(index==0)))
    return results


def compare_pair(records,priority):
    """Apply the prespecified falsifiers/nomination after both closed records."""
    if ([r['spec']['condition'] for r in records]!=list(CONDITIONS)
            or any(r.get('complete') is not True for r in records)
            or priority!=['close_endpoint','hold_current']):
        raise ValueError('Both closed predeclared conditions and fixed priority required')
    left,right=records;other_equal=False;contrast=None
    if left['probe_target14'] is not None and right['probe_target14'] is not None:
        indices=[j for j in range(14) if j!=JOINT]
        other_equal=bool(np.array_equal(np.asarray(left['probe_target14'])[:,indices],np.asarray(right['probe_target14'])[:,indices]))
        contrast=float(np.abs(np.asarray(left['probe_target14'])[:,JOINT]-np.asarray(right['probe_target14'])[:,JOINT]).max())
    prelude_equal=len(left['prelude_signatures'])==len(right['prelude_signatures'])==3 and left['prelude_signatures']==right['prelude_signatures']
    selected=next((r for condition in priority for r in records if r['spec']['condition']==condition and r['success']),None)
    comparison=dict(prelude_three_chunks_exact=prelude_equal,probe_other13_parameters_exact=other_equal,
        actual_successes={r['spec']['condition']:r['success'] for r in records},
        nominal_grip_contrast_max_abs=contrast,contrast_threshold=1e-5,
        closure_contrast_nondegenerate=contrast is not None and contrast>1e-5,
        mechanism_interpretation_eligible=prelude_equal and other_equal and contrast is not None and contrast>1e-5,
        interpretation='One original case/one pair; equal/negligible grip commands or unequal preludes make closure interpretation inconclusive; no significance/class/fit/all12bank claim')
    candidate=None if selected is None else dict(condition=selected['spec']['condition'],
        record_id=selected['spec']['id'],episode_manifest_ref=selected['episode_manifest_ref'],priority=priority,admitted=False,
        meaning='Predeclared candidate only; independent exact12-bank/protocol/cache qualification required')
    return comparison,candidate


def run_factorial(specs,episode,global_rng,*,restore=restore_rng,verify=assert_restored_rng):
    """Four fixed fresh episodes share ONE prospective existing RNG fork."""
    if ([(s['probe_holds'],s['condition']) for s in specs]!=list(FACTORIAL_CASES)
            or len({s['id'] for s in specs})!=4):
        raise ValueError('Exactly1current/1close/5current/5close; no old-run dose baseline')
    records=[]
    for index,spec in enumerate(specs):
        restore(deepcopy(global_rng));verify(global_rng)
        records.append(episode(deepcopy(spec),qualify=(index==0)))
    return records


def compare_factorial(records,priority):
    if ([(r['spec']['probe_holds'],r['spec']['condition']) for r in records]!=list(FACTORIAL_CASES)
            or priority!=list(FACTORIAL_PRIORITY) or any(r.get('complete') is not True for r in records)):
        raise ValueError('All FOUR closed prospective cases and fixed smallest-dose/close-first priority required')
    pairs={str(n):compare_pair(records[i:i+2],['close_endpoint','hold_current'])[0] for n,i in ((1,0),(5,2))}
    common=len(records[0]['prelude_signatures'])==3 and all(r['prelude_signatures']==records[0]['prelude_signatures'] for r in records)
    first_targets=all(r['probe_target14'] is not None for r in records)
    same_dose_anchor=first_targets and all(
        np.array_equal(np.asarray(records[i]['probe_target14']),np.asarray(records[i+2]['probe_target14'])) for i in (0,1))
    selected=next((r for key in priority for r in records if r['spec']['case_key']==key and r['success']),None)
    comparison=dict(all_four_preludes_exact=common,same_grip_cross_dose_probe_targets_exact=bool(same_dose_anchor),
        within_dose=pairs,actual_successes={r['spec']['case_key']:r['success'] for r in records},
        recipe_comparison_eligible=bool(common and same_dose_anchor and all(p['mechanism_interpretation_eligible'] for p in pairs.values())),
        interpretation='Matched prospective1vs5 repeated-native-hold recipes; extra TOPP micro-motion/RNG consumption precludes pure elapsed-time causality. OldE is historical only, no significance/class/fit/all12bank claim.')
    candidate=None if selected is None else dict(case_key=selected['spec']['case_key'],probe_holds=selected['spec']['probe_holds'],
        condition=selected['spec']['condition'],record_id=selected['spec']['id'],episode_manifest_ref=selected['episode_manifest_ref'],
        priority=priority,admitted=False,meaning='Nomination after all four closed; explicit new-profile fixed12 assembler admission still required')
    return comparison,candidate
