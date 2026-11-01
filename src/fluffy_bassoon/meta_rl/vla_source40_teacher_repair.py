"""One diagnosed source40 reset-guard migration; no allocator or physical retry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

from .vla_history import file_sha256

FORMAT='source40_teacher_assessment_reset_guard_repair_v1'
SOURCE='7ef8a62058773201e85c8b2ec26bf3847ff15026'
ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1')
DATA=ROOT/'data'
CHILD=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r2')
DECISION_SHA='73c28c6f0a47a7fff01de3554e5359628d45f544340d0153eb335ed293f3ae1e'
CPU_GRANT_SHA='22dd73a9b75873e1302e6ffd99e73d2e5d2a803bbf88e222c4cd5c1f4d3a8e42'
PROFILE_SHA='222e98b6dd8762944d504b9af033cbda52b7b1d1e3625b8c0b12fda3ceb508a0'
PROGRESS=dict(path=str(DATA/'assessment-recovery.pkl'),sha256='c96dc68b18f95f136550eb824fa5bdf7fcad23950257500f446d28e3a1b94b8c',bytes=62745)
BINDING=dict(path=str(DATA/'loaded-actor-bindings.json'),sha256='e7f860f6fc7390afd3193630c3c3a15639b2e0f19251513157b8666cba75031c')
FAILED_RAW=dict(path=str(DATA/'episode-161/raw-rpc/1793504049893719572-reset.pkl.gz'),sha256='1361cdf53df3ae1422990628c3513502bd86bc33dfda8f9be67a4509090a3e47')
FAILED_JOURNAL=dict(path=str(DATA/'episode-161/physical-journal/1793504048251783216-reset.json'),sha256='865954945faaf3c45cf659dd3f0832904b4296592ba96c6d5945aeddba0aa6f3')
FIRST_PATH=str(ROOT/'clock/attempts/1793502780000000-0000')
PRIOR_COSTS=dict(all_in_actions=26096,physical_episodes=162,uncertain_actions=0)
CLOSED_COSTS=dict(all_in_actions=26086,physical_episodes=161,uncertain_actions=0)


def expected_profile():
    return dict(format=FORMAT,original_source=SOURCE,original_clock_root=str(ROOT/'clock'),
        original_progress=PROGRESS,original_actor_binding=BINDING,failed_raw=FAILED_RAW,failed_journal=FAILED_JOURNAL,
        decision_sha256=DECISION_SHA,CPU_grant_sha256=CPU_GRANT_SHA,child_output=str(CHILD/'data'),
        original_charge=1340,proposed_slice=9360,proposed_total=9420,unrequested=40,
        prior_costs=PRIOR_COSTS,failed_ordinal=161,next_ordinal=162,remaining_roles=158,
        camera_max_uint8_lsb=1,failed_role_replayed=False,source_pretraining_admitted=False)


def check_profile(value):
    if value!=expected_profile():raise ValueError('Only the exact diagnosed reset161 repair is authorized')


def repair_admission(ticket,clock,*,source,output,profile):
    from .vla_source40_teacher_assessment import worker_contract,clock_identity
    check_profile(profile)
    expected=worker_contract(SOURCE,'libero40-teacher-assessment-20261101-r1')
    old=clock['attempts'][0]
    if (ticket['source_commit']!=source or ticket['run_id']!='libero40-teacher-assessment-20261101-r1'
            or ticket['clock_contract']!=expected or clock['contract']!=expected
            or Path(ticket['clock_root']).resolve()!=ROOT/'clock' or Path(output).resolve()!=CHILD/'data'
            or clock_identity(ticket['clock_root'],clock)['origin']!=dict(path=FIRST_PATH,started_unix=1793502780.,reserved_seconds=10800)
            or old!=dict(path=FIRST_PATH,started_unix=1793502780.,slice_seconds=10740,reserved_seconds=10800,
                status='finished',scheduler_allowance_seconds=60,ended_unix=1793504059.1853201,
                exit_code=1,charged_seconds=1340,slice_overrun=False)
            or len(clock['attempts'])!=2 or clock['attempts'][1]['slice_seconds']!=9360
            or clock['attempts'][1]['reserved_seconds']!=9420
            or ticket['repair_profile']['sha256']!=PROFILE_SHA):
        raise ValueError('Repair must retain original7ef header/root/first-attempt1340 and finite residual child')
    return expected


def old_provenance(ticket,closed_clock):
    from .vla_source40_teacher_assessment import clock_identity
    return dict(worker_contract=closed_clock['contract'],original_clock=clock_identity(str(ROOT/'clock'),closed_clock),
        runtime_files=ticket['runtime_files'],loaded_actors=BINDING)


def numerical_digest(progress):
    # Preserve *all* non-disposition fields, not a selected subset of RNGs.
    fields={k:v for k,v in progress.items() if k not in ('contract','cursor','active','costs','aborted','migration')}
    return hashlib.sha256(pickle.dumps(fields,protocol=5)).hexdigest()


def aborted_role(table):
    from .vla_source40_teacher_assessment import panel
    spec=panel(table)[161]
    if spec['task_key']!='libero_spatial:05' or spec['state']!=40 or spec['role']!='exact_source_tied_Gaussian_warmup':
        raise ValueError('Failed role mapping differs')
    return dict(ordinal=161,spec=spec,status='ABORTED_RESET_GUARD',completed=False,success=None,
        performance_score=None,primitive_steps=0,settling_steps=10,all_in_actions=10,
        original_progress=PROGRESS,failed_raw=FAILED_RAW,failed_journal=FAILED_JOURNAL)


def validate_disposition(progress,table):
    if (progress.get('aborted')!=[aborted_role(table)] or not 162<=progress['cursor']<=320
            or len(progress['receipts'])!=progress['cursor']-1
            or progress.get('migration',{}).get('original_progress')!=PROGRESS):
        raise ValueError('Only the single diagnosed aborted161 slot may be skipped')


def migrate_progress(progress,*,table,old_contract,new_contract):
    if (progress['contract']!=old_contract or progress['active']!=161 or progress['cursor']!=161
            or len(progress['receipts'])!=161 or progress['costs']!=CLOSED_COSTS
            or progress.get('aborted') or progress.get('migration') or progress['qualification_complete'] is not True
            or progress['case_key']!='libero_spatial:05/state-40'
            or len(progress['references'])!=41):
        raise ValueError('Require exact active161 checkpoint and unchanged161 closed prefix')
    before=numerical_digest(progress);result=dict(progress)
    result.update(contract=new_contract,cursor=162,active=None,costs=deepcopy(PRIOR_COSTS),aborted=[aborted_role(table)],
        migration=dict(format=FORMAT,original_progress=PROGRESS,original_contract=old_contract,
            numerical_digest=before,original_source=SOURCE,closed_prefix_preserved=161,failed_ordinal_never_replayed=161))
    if numerical_digest(result)!=before:raise RuntimeError('Numerical RNG/case fork/receipt state changed')
    validate_disposition(result,table)
    return result


def verify_failed_role():
    """Only the sole failed reset directory, never the old RPC forest."""
    from .vla_source40_teacher_assessment import read_ref
    episode=DATA/'episode-161'
    if (sorted(p.name for p in (episode/'physical-journal').iterdir())!=[Path(FAILED_JOURNAL['path']).name]
            or sorted(p.name for p in (episode/'raw-rpc').iterdir())!=[Path(FAILED_RAW['path']).name]
            or any((episode/name).exists() for name in ('receipt.json','trace.pkl.gz','rng-context.pkl.gz'))
            or file_sha256(FAILED_RAW['path'])!=FAILED_RAW['sha256']):
        raise ValueError('Failed161 evidence is not one returned reset and zero steering')
    expected=dict(kind='reset',role='query',episode_id='libero_spatial:05/state-40/exact_source_tied_Gaussian_warmup',
        returned=True,maximum_steps=10,actual_steps=10)
    if read_ref(FAILED_JOURNAL)!=expected:raise ValueError('Failed reset cost/identity differs')
    return dict(failed_reset=FAILED_JOURNAL,raw=FAILED_RAW,physical_actions=10,steering_chunks=0)


def initialize_recovery(output,*,table,provenance,original_ticket,closed_clock):
    from .vla_source40_teacher_assessment import recovery_contract
    from .vla_source40_evaluate import _save_recovery
    output=Path(output);path=output/'assessment-recovery.pkl'
    if path.exists():return
    if any(output.glob('episode-*')):raise ValueError('No recovery may be manufactured over physical output')
    if file_sha256(PROGRESS['path'])!=PROGRESS['sha256'] or Path(PROGRESS['path']).stat().st_size!=62745:
        raise ValueError('Original active161 recovery changed')
    with Path(PROGRESS['path']).open('rb') as stream:old=pickle.load(stream)
    proof=verify_failed_role()
    new=migrate_progress(old,table=table,old_contract=recovery_contract(table,old_provenance(original_ticket,closed_clock)),
        new_contract=recovery_contract(table,provenance))
    if file_sha256(PROGRESS['path'])!=PROGRESS['sha256']:raise ValueError('Original progress changed during migration')
    _save_recovery(path,new)
    from .vla_libero import write_json
    write_json(output/'ABORTED-161-RESET-GUARD.json',new['aborted'][0])
    write_json(output/'migration-receipt.json',dict(new['migration'],failed_reset_proof=proof,
        new_recovery_sha256=file_sha256(path),metadata_changes=['contract','cursor161_to162','active161_toNone','costs_add_returned_reset10','aborted','migration']))


def main(argv=None):
    """Allocated software CPU only: tiny state dryrun, no actor/sim/physical work."""
    import argparse
    from .vla_source40_teacher_assessment import read_ref,recovery_contract
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--progress',type=Path,required=True)
    p.add_argument('--table',type=Path,required=True);p.add_argument('--old-ticket',type=Path,required=True)
    p.add_argument('--closed-clock',type=Path,required=True);a=p.parse_args(argv)
    from .vla_source40_teacher_assessment import TABLE_SHA
    if file_sha256(a.progress)!=PROGRESS['sha256'] or a.progress.stat().st_size!=62745:raise ValueError('Exact tiny progress required')
    table=read_ref(dict(path=str(a.table),sha256=TABLE_SHA))
    old_ticket=read_ref(dict(path=str(a.old_ticket),sha256='ccef81ddfe47679d6476657163c0e0bc0572a67f16ed657346ef07d4892bb570'))
    closed=read_ref(dict(path=str(a.closed_clock),sha256='7501d3a7695c5281cc05b795f36eafa8a73a90b1ec909902649a6f7ea8932310'))
    with a.progress.open('rb') as stream:old=pickle.load(stream)
    new=migrate_progress(old,table=table,old_contract=recovery_contract(table,old_provenance(old_ticket,closed)),
        new_contract='CPU_DRYRUN_NOT_GPU_ADMISSION')
    if file_sha256(a.progress)!=PROGRESS['sha256']:raise ValueError('Original input mutated')
    from .vla_libero import write_json
    write_json(a.output,dict(status='TINY_METADATA_MIGRATION_DRYRUN_PASS_NOT_ADMISSION',old_cursor=161,new_cursor=162,
        old_receipts=len(old['receipts']),new_receipts=len(new['receipts']),aborted=new['aborted'],
        numerical_before=numerical_digest(old),numerical_after=numerical_digest(new),physical_actions=0,
        model_or_actor_loads=0,source_pretraining_admitted=False))


if __name__=='__main__':main()
