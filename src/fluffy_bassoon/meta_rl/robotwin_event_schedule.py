"""Deterministic reference scheduling; no simulator or model imports.

Executed only on allocated Linux. A source frame is a reference command, not a
physical time sample. Every chunk still contains32 native command slots.
"""
from copy import deepcopy

FORMAT = 'robotwin_gripper_event_h32_v1'
HORIZON = 32
THRESHOLD = 1e-5


def build_schedule(targets, *, limit=400):
    """Split at start/end of monotone nonconstant gripper phases, then at32.

For each gripper, transition i is frame i-1 -> i. A maximal same-sign run
[l,r] with abs(delta)>1e-5 contributes boundaries l and r+1. Union both
grippers' boundaries with0/N. A changing sample is not its own boundary.
Each resulting <=32-frame chunk repeats its final target to fill H32.
    """
    import numpy as np
    targets=np.asarray(targets)
    if (targets.ndim!=2 or targets.shape[1]!=14 or not len(targets)
            or targets.dtype.kind!='f' or not np.isfinite(targets).all()):
        raise ValueError('Complete finite floating physical14 reference required')
    boundaries={0,len(targets)};events=[]
    for joint in (6,13):
        delta=np.diff(targets[:,joint].astype(np.float64))
        signs=np.where(delta>THRESHOLD,1,np.where(delta<-THRESHOLD,-1,0))
        index=0
        while index<len(signs):
            if signs[index]==0:
                index+=1;continue
            start=index;direction=int(signs[index]);index+=1
            while index<len(signs) and signs[index]==direction:index+=1
            # signs[start:index] are transitions start+1 through index.
            left,right=start+1,index+1
            boundaries.update((left,right))
            events.append(dict(joint=joint,direction=direction,start=left,end_exclusive=right))
    chunks=[];edges=sorted(boundaries)
    for left,right in zip(edges,edges[1:]):
        for start in range(left,right,HORIZON):
            end=min(start+HORIZON,right)
            chunks.append(dict(start=start,end=end,new_frames=end-start,
                issued_slots=HORIZON,added_holds=HORIZON-(end-start)))
    minimum=HORIZON*(len(chunks)-1)+chunks[-1]['new_frames']
    issued=len(chunks)*HORIZON
    return dict(format=FORMAT,reference_frames=len(targets),threshold=THRESHOLD,
        boundaries=edges,events=events,chunks=chunks,full_schedule_slots=issued,
        full_schedule_added_holds=issued-len(targets),
        minimum_slots_to_issue_every_original_frame=minimum,
        native_episode_limit=limit,admissible=minimum<=limit,
        interpretation='Schedule retains every source frame; successful physics may stop before all are consumed')


def scheduled_request(targets, schedule, chunk_index):
    """Return full H32 command, original indices, and newly consumed slots."""
    import numpy as np
    targets=np.asarray(targets)
    if schedule['format']!=FORMAT or len(targets)!=schedule['reference_frames'] or chunk_index<0:
        raise ValueError('Wrong schedule/reference/cursor')
    if chunk_index<len(schedule['chunks']):
        chunk=schedule['chunks'][chunk_index]
        indices=np.minimum(np.arange(HORIZON)+chunk['start'],chunk['end']-1)
        new=np.arange(HORIZON)<chunk['new_frames']
    else:
        indices=np.full(HORIZON,len(targets)-1);new=np.zeros(HORIZON,dtype=bool)
    return targets[indices].copy(),indices,new


def advance_cursor(schedule, chunk_index, physical_before, source_before, *, actual, issued, closed):
    """No shortened nonterminal chunks; terminal slots never consume a frame."""
    if (type(actual) is not int or type(issued) is not int or not 0<actual<=issued<=HORIZON
            or (not closed and (actual!=HORIZON or issued!=HORIZON))):
        raise ValueError('Native H32 valid prefix/complete episode boundary required')
    if chunk_index<len(schedule['chunks']):
        chunk=schedule['chunks'][chunk_index]
        if source_before!=chunk['start']:raise ValueError('Source cursor skipped or repeated a reference frame')
        consumed=min(actual,chunk['new_frames'])
    else:
        if source_before!=schedule['reference_frames']:raise ValueError('Tail hold before complete reference consumption')
        consumed=0
    return dict(chunk_index=chunk_index+1,physical_slots=physical_before+actual,
        source_cursor=source_before+consumed,native_valid_reference_slots=consumed,
        native_valid_hold_slots=actual-consumed,issued_slots=issued,
        terminal_unexecuted_slots=issued-actual,closed=bool(closed))


def validate_teacher_bank(records, demos):
    """No control/old rollout, duplicate, reordered split or failed row enters SL."""
    wanted={row['id']:row for row in demos['episodes']}
    if len(wanted)!=12 or len(records)!=12:raise ValueError('All12 fixed aligned teacher executions required')
    actual={row['spec'].get('teacher_id'):row for row in records}
    if set(actual)!=set(wanted) or len(actual)!=len(records):raise ValueError('Duplicate/missing/foreign teacher')
    for identity,row in actual.items():
        if (row['spec'].get('stage')!='event_aligned_teacher_execution'
                or not row.get('complete') or not row.get('success')
                or row['spec']['reset_seed']!=wanted[identity]['reset_seed']
                or row['spec']['paired_instruction']!=[wanted[identity]['instruction']]):
            raise ValueError('Only declared successful complete aligned rollouts enter SL')
    return {key:deepcopy(value) for key,value in actual.items()}
