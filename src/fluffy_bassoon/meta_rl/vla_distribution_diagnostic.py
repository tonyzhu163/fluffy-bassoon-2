"""Fixed9 source500 mean/std intervention screen; no parameter updates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

from .vla_distribution_control import DistributionController,qualify_original
from .vla_distribution_plan import FORMAT as PLAN_FORMAT,LEDGER_FORMAT,OPERATIONS,build_plan
from .vla_common_runtime_evaluate import run_common
from .vla_history import file_sha256
from .vla_libero import FrozenOpenPI,write_json
from .vla_motor_replay import parameter_signature
from .vla_source40_corpus import PI05_RECEIPT,save_motor_rng
from .vla_source40_evaluate import load_candidates,_policy_signature
from .vla_source40_memory_replication import trace_rows
from .vla_source40_noise_controls import graceful_seconds
from .vla_source40_schedule import content_hash
from .vla_source40_target_firstwave import PlannedServer,audit_firstwave,fixed_matrix

FORMAT='source40_distribution_diagnostic_v1'


def load_plan(path,digest,source):
    if file_sha256(path)!=digest:raise ValueError('Distribution diagnostic plan file checksum differs')
    p=json.loads(Path(path).read_text())
    if (p.get('format')!=PLAN_FORMAT or p.get('diagnostic_sha256')!=content_hash({k:v for k,v in p.items() if k!='diagnostic_sha256'})
        or p!=build_plan(worktree=p['contract']['worktree'],source_commit=source,inputs=p['contract']['inputs'])):
        raise ValueError('Exact fixed9 distribution diagnostic differs')
    return p


def reservation(path,plan,run,source):
    value=json.loads(Path(path).read_text());record=value.get('reservations',{}).get(run)
    if (value.get('format')!=LEDGER_FORMAT or value.get('diagnostic_sha256')!=plan['diagnostic_sha256']
        or value.get('maximum_total_actions')!=27000 or value.get('maximum_gpu_seconds')!=3600
        or value.get('source_campaign_immutable') is not True or not value.get('scope_amendment')
        or len(value.get('reservations',{}))!=1 or record is None or record.get('run_id')!=run
        or record.get('source_commit')!=source or record.get('maximum_total_actions')!=27000 or record.get('worker_seconds')!=3600
        or record.get('diagnostic_sha256')!=plan['diagnostic_sha256'] or record.get('purpose')!='target_evaluation'
        or record.get('source_pretraining_data') is not False or record.get('fresh_ppo_training_data') is not False):
        raise ValueError('Require the existing exact27k/one-hour distribution diagnostic reservation')
    return dict(ledger_path=str(Path(path).resolve()),record=record,record_sha256=content_hash(record))


def specs(candidates):
    return {method:dict(candidates[method.split('_')[0]+'_full'],mode='full',distribution_operation=operation,
        diagnostic_method=method) for method,operation in OPERATIONS.items()}


def audit_distribution(path,result,group):
    identity=audit_firstwave(path,result,group)
    for episode in result['episodes']:
        for row in trace_rows(path,episode):
            if group['method']=='native':continue # mean0/std1; epsilon is native_initial_noise.
            d=row.get('steering_distribution',{});op=OPERATIONS[group['method']]
            if (d.get('format')!='frozen_steering_distribution_draw_v1' or d.get('mean_operation')!=op['mean']
                or d.get('std_operation')!=op['std'] or d.get('random_draws')!=1 or d.get('parameter_updates')!=0):
                raise ValueError('Missing actual distribution operation/draw receipt')
            for name in ('base_mean','learned_std','applied_mean','applied_std','epsilon','proposed_noise','actual_executed_noise'):
                a=np.asarray(d[name])
                if a.shape!=(32,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Distribution receipt shape/dtype differs')
            expected_mean=np.zeros(32,np.float32) if op['mean']=='zero' else d['base_mean']
            expected_std=np.ones(32,np.float32) if op['std']=='unit' else d['learned_std']
            if (not np.array_equal(d['applied_mean'],expected_mean) or not np.array_equal(d['applied_std'],expected_std)
                or (np.asarray(d['learned_std'])<=0).any()
                or not np.allclose(d['proposed_noise'],d['applied_mean']+d['applied_std']*d['epsilon'],rtol=0,atol=2e-7)
                or not np.array_equal(d['proposed_noise'],row['proposed_noise'])
                or not np.array_equal(d['actual_executed_noise'],row['executed_noise'])
                or not np.array_equal(np.clip(d['proposed_noise'],-1.,1.),row['executed_noise'])
                or not np.array_equal(np.repeat(row['executed_noise'][None],10,axis=0),row['native_initial_noise'])):
                raise ValueError('Actual mean/std/sample/clip differs from diagnostic operation')
            if file_sha256(d['before_rpc_path'])!=d['before_rpc_sha256']:raise ValueError('Before-RPC distribution receipt checksum differs')
            with Path(d['before_rpc_path']).open('rb') as f:before=pickle.load(f)['distribution']
            if any(not np.array_equal(before[k],d[k]) for k in ('base_mean','learned_std','epsilon','proposed_noise')):
                raise ValueError('Before-RPC distribution differs from actual final trace')
    identity['actual_distribution_records_verified']=True
    return identity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','plan','ledger','checkpoint','openpi-root'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('source-commit','run-id','expected-hostname','plan-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--stop-before-unix',type=float,required=True);p.add_argument('--port',type=int,default=8776)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true');a=p.parse_args(argv)
    root=Path(__file__).resolve().parents[3]
    if (sys.platform!='linux' or socket.gethostname()!=a.expected_hostname or a.output.name!='experiment' or a.output.parent.name!=a.run_id
        or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=a.source_commit
        or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):raise ValueError('Require this exact clean source in the identified Linux allocation')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION','baige'),os.environ)
    probe=json.loads(Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
    if probe.get('status')!='passed' or len(probe.get('devices',[]))!=1:raise ValueError('Require actual allocated one-GPU probe')
    plan=load_plan(a.plan,a.plan_sha256,a.source_commit);budget=reservation(a.ledger,plan,a.run_id,a.source_commit)
    seconds=graceful_seconds(a.output.parent,budget['record_sha256'],a.stop_before_unix)
    if seconds<60 or file_sha256(a.checkpoint/'READY.json')!=PI05_RECEIPT:raise ValueError('No authorized time remains or motor receipt differs')
    deadline=time.monotonic()+seconds;runtime=f'{socket.gethostname()}-{os.getpid()}-{time.time_ns()}'
    attempt=a.output.parent/'runtime-attempts'/runtime;attempt.mkdir(parents=True)
    write_json(attempt/'invocation-deadline.json',dict(stop_before_unix=a.stop_before_unix,driver_seconds=seconds,shutdown_margin_seconds=60))
    inputs=plan['contract']['inputs']
    matrix,registry=fixed_matrix(inputs['matrix'],inputs['matrix_sha256'],inputs['assembly'],inputs['assembly_sha256'])
    contract=dict(format=FORMAT,source_commit=a.source_commit,diagnostic_sha256=plan['diagnostic_sha256'],plan_file_sha256=a.plan_sha256,
        groups=plan['groups'],budget_reservation=budget,maximum_total_actions=27000,maximum_seconds_per_invocation=3600.,fixed500_models=registry,
        operations=OPERATIONS,parameter_updates=0,qualification_extra_resets=0,
        native_distribution=plan['contract']['native_distribution'],canonical_motor_key='legacy jax.random.PRNGKey(seed), shared by all comparators',
        purpose=plan['contract']['purpose'])
    lock=(a.output.parent/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.model_config,a.feature_interface='pi05_libero','vla-prefix';motor=FrozenOpenPI(SimpleNamespace(**dict(vars(a),output=attempt)))
    loaded,source=load_candidates(inputs['matrix'],inputs['matrix_sha256'],inputs['assembly'],motor,
        seed=matrix['controllers'][0]['seed'],device=a.device,allow_small_pilot=False)
    selected=specs(loaded);before=parameter_signature(motor.policy._model)
    policies_before={o:_policy_signature(loaded[o+'_full']['policy']) for o in ('ad','dpt')}
    write_json(attempt/'parameters-before.json',dict(motor=before,controllers=policies_before))
    entries={e['case']['case_id']:e for e in plan['cases']}
    def reset_rng(seed):
        import jax
        motor.policy._rng=jax.random.PRNGKey(seed)
    def qualify(observation,server):
        zero=np.zeros((10,32),np.float32);x=motor.infer(observation,server.metadata['instruction'],zero);y=motor.infer(observation,server.metadata['instruction'],zero.copy())
        if not np.allclose(x,y,rtol=0,atol=1e-5):raise RuntimeError('Frozen motor same-noise parity failed')
        features=motor.feature_encoder(observation,server.metadata['instruction'])
        checks={o:qualify_original(loaded[o+'_full']['policy'],features) for o in ('ad','dpt')}
        write_json(attempt/'original-equivalence.json',dict(controllers=checks,extra_physical_resets=0,same_noise_max_delta=float(np.abs(x-y).max())))
    def controller(spec,server):
        op=spec['distribution_operation']
        return DistributionController(spec['policy'],mean=op['mean'],std=op['std'],
            record_output=attempt/'distribution-draws'/f'{server.metadata["task_id"]:02d}-{spec["diagnostic_method"]}-{time.time_ns()}')
    result=None
    try:
        result=run_common(motor,{'source500':selected},{},output=a.output,contract=contract,
            server_factory=lambda case:PlannedServer(case,entries[case.case_id],worktree=root,
                output=attempt/f'server-{case.task_id:02d}-{time.time_ns()}',port=a.port),
            provenance={'source500':dict(source,diagnostic_operations=OPERATIONS), 'native':dict(source_training_actions=0,distribution=plan['contract']['native_distribution'])},
            controller_factory=controller,runtime_id=runtime,resume=a.resume,reset_motor_rng=reset_rng,
            motor_rng_state=lambda:save_motor_rng(motor.policy._rng),qualify=qualify,deadline=deadline,
            action_ceiling=27000,audit_fn=audit_distribution,strict_initial_observations=True,graceful_deadline=True)
        result.update(format=FORMAT,diagnostic_sha256=plan['diagnostic_sha256'],source_provenance=source,
            parameter_updates=0,interpretation=plan['contract']['purpose'])
    finally:
        after=parameter_signature(motor.policy._model);policies_after={o:_policy_signature(loaded[o+'_full']['policy']) for o in ('ad','dpt')}
        unchanged=before==after and policies_before==policies_after
        write_json(attempt/'parameters-after.json',dict(motor_before=before,motor_after=after,controllers_before=policies_before,controllers_after=policies_after,unchanged=unchanged))
        if result is not None:
            result.update(all_base_parameter_bytes_unchanged=unchanged,motor_parameters_before_sha256=before['sha256'],motor_parameters_after_sha256=after['sha256'])
            result['strict_common_runtime_primary_eligible'] &= unchanged;write_json(a.output/'index.json',result)
        if not unchanged:raise RuntimeError('Base motor/controller parameter bytes changed during distribution diagnostic')

if __name__=='__main__':main()
