"""Native evaluation case routing for the bounded LIBERO90 search.

No model, simulator, learner or resource allocation implementation lives here.
"""
from .rlinf_flora_ppo import digest

PHASES=('screen','confirm')
FIRST_TASKS=(0,8,16,24,32,40,48,56,64,72,80,89)


def candidate_batches():
    order=list(FIRST_TASKS)+[i for i in range(90) if i!=44 and i not in FIRST_TASKS]
    return [order[i:i+12] for i in range(0,len(order),12)]


def headroom_schedule(tasks,phase):
    tasks=list(tasks)
    if (phase not in PHASES or not tasks or len(tasks)>4 or len(set(tasks))!=len(tasks)
            or any(type(t) is not int or not 0<=t<90 or t==44 for t in tasks)
            or phase=='confirm' and len(tasks)!=1):
        raise ValueError('Concrete native-only screen wave or single confirmation required')
    if phase=='screen':
        rows=[dict(task_id=t,nominal_state=i,draw=0,counted=True) for t in tasks for i in range(8)]
    else:rows=[dict(task_id=tasks[0],nominal_state=i%40,draw=i//40,counted=True) for i in range(64)]
    return [dict(r,slot=i) for i,r in enumerate(rows)]


def reset_ids_for_cases(cases,trial_bins):
    """Map actual per-task nominal placements into the native global ID space."""
    offsets=[];total=0
    for n in trial_bins:offsets.append(total);total+=int(n)
    ids=[]
    for row in cases:
        task=row['task_id'];state=row['nominal_state']
        if (type(task) is not int or type(state) is not int or not 0<=task<len(trial_bins)
                or not 0<=state<int(trial_bins[task])):raise ValueError('Invalid native task/placement index')
        ids.append(offsets[task]+state)
    return ids


def evaluation_seed(tasks,phase):
    # Frozen before outcomes; batching is fixed and confirmation uses a disjoint
    # seed namespace. Each worker uses this base+rank after model initialization.
    return int(digest(dict(namespace='libero90_native_headroom_20261101_v1',tasks=list(tasks),phase=phase))[:8],16)%(2**31-8)
