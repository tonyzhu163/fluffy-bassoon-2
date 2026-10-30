"""CPU-only terminal checkpoint/receipt audit; no model construction or forward.

Usage on the development host: CUDA_VISIBLE_DEVICES='' python FILE RUN_ROOT
Prints a compact receipt; never writes to the original experiment directory.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def fingerprint(value):
    # Exact recursive leaf algorithm used by the immutable migration receipt.
    digest=hashlib.sha256()
    def walk(item):
        if torch.is_tensor(item):
            array=item.detach().cpu().numpy()
            digest.update(str(('tensor',array.shape,array.dtype)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        elif isinstance(item,np.ndarray):
            digest.update(str(('array',item.shape,item.dtype)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict')
            for key in sorted(item,key=repr):walk(key);walk(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(str((type(item).__name__,len(item))).encode())
            for entry in item:walk(entry)
        else:digest.update(str((type(item).__name__,item)).encode())
    walk(value)
    return digest.hexdigest()


root=Path(sys.argv[1]).resolve();exp=root/'experiment'
clock=read(root/'worker-time.json');attempt=Path(clock['attempts'][-1]['path'])
index=read(exp/'index.json');summary=index['summary'];source=index['source_contract']
recovery=exp/'learner/latest.pt';candidate=exp/'learner/candidate.pt';initial=exp/'learner/warm-start-initial.pt'
artifacts={str(path.relative_to(root)):dict(sha256=sha(path),bytes=path.stat().st_size)
    for path in [root/'worker-time.json',root/'exit-code.txt',exp/'index.json',exp/'source-contract.json',
        exp/'learner/summary.json',exp/'learner/metrics.jsonl',recovery,candidate,initial,
        exp/'learner/warm-start-migration.json',attempt/'gpu-probe.json',attempt/'focused-tests.log',
        attempt/'motor-rng-tests.log',attempt/'qualification-receipt.json']}
saved=torch.load(recovery,map_location='cpu',weights_only=False)
export=torch.load(candidate,map_location='cpu',weights_only=False)
start=torch.load(initial,map_location='cpu',weights_only=False)
progress=saved['progress'];inherited=progress['inheritance'];old=inherited['legacy_recovery']
migration=read(exp/'learner/warm-start-migration.json')
restore_paths=sorted((exp/'learner').glob('warm-restore-*.json'))
restore=read(restore_paths[0])
old_digest=fingerprint({key:value for key,value in old.items() if key!='contract'})
initial_digest=fingerprint({key:value for key,value in start['progress']['inheritance']['legacy_recovery'].items() if key!='contract'})
candidate_equal=(export['policy'].keys()==saved['policy'].keys()
    and all(torch.equal(value,saved['policy'][key]) for key,value in export['policy'].items()))
new_cycle=saved['runtime']['task_cycle'];old_cycle=old['runtime']['task_cycle']
old_steps=sorted({float(row['step']) for row in old['optimizer']['state'].values() if 'step' in row})
new_steps=sorted({float(row['step']) for row in saved['optimizer']['state'].values() if 'step' in row})
sampler=np.random.default_rng();sampler.bit_generator.state=old['environment_rng']
planned_seeds=[int(sampler.integers(2**31)) for _ in range(14)]
sampler_equal=fingerprint(sampler.bit_generator.state)==fingerprint(saved['sampler'])
motor=saved['runtime']['motor_rng'];key=np.asarray(motor['key_data'])
update=progress['update_receipts'][0];stats=update['source_trials'];metrics=update['metrics'];replay=update['replay']
qual=read(attempt/'qualification-receipt.json');probe=read(attempt/'gpu-probe.json')
parameter_paths=sorted((exp/'attempts').glob('*/parameters-after.json'))
parameters=read(parameter_paths[-1])
collection=stats['committed_collection_seconds'];steady=collection+replay['seconds']+metrics['update_seconds']
row=clock['attempts'][0];elapsed=row['ended_unix']-row['started_unix']
new_actions=index['physical_cost']['total_action_charge'];prior=inherited['history']['prior_actions']
training_actions=progress['counters']['environment_steps']+progress['counters']['settling_steps']
checks=dict(terminal_exit_zero=(root/'exit-code.txt').read_text().strip()=='0',
    all_attempts_finished=all(row['status']=='finished' for row in clock['attempts']),
    exact_source_pin=source['source_commit']==clock['contract']['source_commit']=='376415e6915399436cb3fea214c50bd3e41f32d5',
    gpu_probe_passed=probe['status']=='passed',
    focused_tests_passed='50 passed' in (attempt/'focused-tests.log').read_text(),
    motor_rng_tests_passed='2 passed' in (attempt/'motor-rng-tests.log').read_text(),
    candidate_weights_equal_full_recovery=candidate_equal,
    export_manifest_matches_recovery=(export['manifest']['contract']==saved['contract']
        and export['manifest']['counters']==progress['counters']
        and export['manifest']['full_recovery_sha256']==sha(recovery)),
    inherited_noncontract_leaves_exact=(old_digest==initial_digest==inherited['legacy_non_contract_sha256']
        ==migration['old_non_contract_sha256']==migration['new_inherited_non_contract_sha256']),
    restore_operational_fingerprints_equal=restore['passed'] and restore['expected_operational_sha256']==restore['restored_operational_sha256'],
    zero_restore_sgd=migration['restore_optimizer_steps']==restore['restore_optimizer_steps']==0,
    old_counters_preserved=old['iteration']==238 and old['counters']['optimizer_steps']==476 and old_steps==[476.],
    new_counters_and_adam=progress['iteration']==1 and progress['counters']['optimizer_steps']==2 and new_steps==[478.],
    cursor_complete_no_pending=new_cycle['cursor']==progress['completed_trials']==14 and new_cycle['remaining']==0
        and progress['plan'] is None and progress['phase']=='collect' and progress['pending_rollouts']==progress['pending_receipts']==[],
    actual_exposure_preserved=len([v for v in old_cycle['exposure'].values() if v['episodes']>0])==40
        and sum(v['episodes'] for v in old_cycle['exposure'].values())==952
        and len([v for v in new_cycle['exposure'].values() if v['episodes']>0])==14
        and sum(v['episodes'] for v in new_cycle['exposure'].values())==56,
    inherited_sampler_draws_exact=sampler_equal,
    initial_sampler_exact=fingerprint(start['sampler'])==fingerprint(old['environment_rng']),
    global_rng_complete=all({'python','numpy','torch','cuda'}<=set(value['rng']) for value in [old,start,saved]),
    initial_global_rng_exact=fingerprint(start['rng'])==fingerprint(old['rng']),
    initial_typed_motor_rng_exact=fingerprint(start['runtime']['motor_rng'])==fingerprint(old['runtime']['motor_rng']),
    final_typed_motor_rng=motor['format']=='jax_motor_key_v1' and motor['implementation']=='threefry2x32'
        and key.dtype==np.uint32 and key.shape==(2,),
    frozen_motor_exact=parameters['unchanged'] is True and parameters['motor_before']==parameters['motor_after']
        and index['frozen_motor_parameters_unchanged'] is True
        and parameters['motor_before']['sha256']==qual['motor_parameters_before_sha256']==qual['motor_parameters_after_sha256'],
    exact_behavior_replay=all(replay[name]==0. for name in replay if name.endswith('max_error')),
    finite_optimizer_metrics=all(np.isfinite(value) for value in metrics.values()),
    physical_and_cumulative_cost=new_actions==9272 and training_actions==9262 and prior==195964
        and summary['cumulative_source_actions']==prior+new_actions==205236,
    action_gpu_caps=new_actions<=20000 and sum(row['charged_seconds'] for row in clock['attempts'])==910<=1400,
    no_artifact_changed_during_read=all(sha(root/name)==value['sha256'] for name,value in artifacts.items()))
receipt=dict(format='source40_warm_batch14_terminal_cpu_audit_v1',checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    run_id=root.name,job_id='job-pqje6dlyz0of',source_commit=source['source_commit'],root=str(root),
    operation='Read-only CPU checkpoint/metadata inspection; no policy/model construction, forward pass, optimizer step, simulator or test',
    checks=checks,passed=all(checks.values()),artifacts=artifacts,
    global_rng=dict(python_present=True,numpy_present=True,torch_bytes=int(saved['rng']['torch'].numel()),
        cuda_generators=None if saved['rng']['cuda'] is None else len(saved['rng']['cuda']),
        final_sha256=fingerprint(saved['rng'])),
    motor_rng=dict(format=motor['format'],implementation=motor['implementation'],key_shape=list(key.shape),
        key_dtype=str(key.dtype),final_sha256=fingerprint(motor)),
    sampler=dict(exact_next14_seed_draws=True,scene_seeds=planned_seeds,final_sha256=fingerprint(saved['sampler'])),
    inherited=dict(updates=old['iteration'],counters=old['counters'],adam_steps=old_steps,
        noncontract_sha256=old_digest,actual_source_goals=40,complete_K4_trials=old_cycle['cursor'],episodes=952),
    new_phase=dict(updates=progress['iteration'],counters=progress['counters'],adam_steps=new_steps,
        complete_K4_trials=progress['completed_trials'],pending_trials=0,task_cursor=new_cycle['cursor'],actual_source_goals=14,
        exposure={name:value for name,value in new_cycle['exposure'].items() if value['episodes']},
        episodes=stats['episodes'],successes=stats['successes'],failures=stats['failures'],patterns=stats['patterns'],
        decisions=replay['valid_tokens'],replay={key:value for key,value in replay.items() if key!='trial_boundaries'},metrics=metrics),
    frozen_motor=dict(before=parameters['motor_before'],after=parameters['motor_after'],unchanged=parameters['unchanged'],
        receipt_path=str(parameter_paths[-1]),receipt_sha256=sha(parameter_paths[-1])),
    migration_receipt_sha256=sha(exp/'learner/warm-start-migration.json'),
    operational_restore_receipts=[dict(path=str(path),sha256=sha(path),**read(path)) for path in restore_paths],
    physical_cost=dict(**index['physical_cost'],training_episode_actions=training_actions,qualification_settling_actions=new_actions-training_actions,
        prior_source_resource_actions=prior,cumulative_source_resource_actions=prior+new_actions,
        interpretation='Resource charge includes prior source training/repairs, source validation and qualifications; not all optimization data'),
    measured_time=dict(collection_seconds=collection,replay_seconds=replay['seconds'],update_seconds=metrics['update_seconds'],
        steady_batch_seconds=steady,actual_cold_worker_seconds=elapsed,other_cold_setup_shutdown_seconds=elapsed-steady,
        scheduler_allowance_seconds=row['scheduler_allowance_seconds'],charged_gpu_seconds=910,
        training_actions_per_steady_second=training_actions/steady,source_resource_actions_per_charged_second=new_actions/910),
    memory=dict(sampled_device_peak_mib=qual['sampled_whole_device_peak_memory_mib'],
        torch_peak_allocated_bytes=metrics['peak_cuda_allocated_bytes'],torch_peak_reserved_bytes=metrics['peak_cuda_reserved_bytes']),
    interpretation='One warm-started source runtime/restore/update qualification. SFSS occurred under the inherited frozen behavior before new-phase SGD. No heldout, memory-benefit, convergence or robot-performance comparison.')
print(json.dumps(receipt,indent=2,allow_nan=False))
if not receipt['passed']:raise SystemExit(1)
