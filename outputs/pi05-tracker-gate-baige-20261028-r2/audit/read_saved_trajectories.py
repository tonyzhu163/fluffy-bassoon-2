"""Read-only stdlib arithmetic over saved JSON; no Torch/model imports."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import struct

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/pi05-tracker-gate-baige-20261028-r2')
SOURCE='c2c47152303ccfc1ac469fd3e1ad1e1344201e42'
assert socket.gethostname()=='aibox-rffe453e3884-5f96c78b88-rtvhb'
def read(path): return json.loads(path.read_text())
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def flat_shape(value):
    if not isinstance(value,list):
        if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Non-finite tensor scalar')
        return [],[value]
    if not value:return [0],[]
    children=[flat_shape(item) for item in value]
    if not all(c[0]==children[0][0] for c in children):raise ValueError('Ragged tensor')
    return [len(value)]+children[0][0],[v for c in children for v in c[1]]
def float_tensor_hash(value):
    shape,flat=flat_shape(value)
    raw=struct.pack('<'+str(len(flat))+'f',*flat)
    return digest(dict(kind='tensor',dtype='torch.float32',shape=shape,data_sha256=hashlib.sha256(raw).hexdigest()))
summary=read(ROOT/'tracker/summary.json');config=read(ROOT/'tracker/config.json');checks={}
def check(name,value): checks[name]=bool(value)
check('exact_source',summary['source_commit']==SOURCE)
check('complete_phase',summary['phase']=='finished' and not summary['bounded_stop'])
check('exit_zero',int((ROOT/'exit-code.txt').read_text())==0)
check('frozen_motor_unchanged',summary['frozen_motor_before']==summary['frozen_motor_after']==config['contract']['frozen_motor_fingerprint'])
check('expected_frozen_motor',summary['frozen_motor_after']=='ed862a89a39052dad591d6542959ef0a72866b2c85eac2c84767bf55f52205dc')
expected=set([f'calibrate-{int(theta):+d}-{gain}' for theta in (-45,45) for gain in (.25,.5,1.)]+[f'heldout-{int(theta):+d}-{condition}' for theta in (-45,0,45) for condition in ('bridge','zero_bridge','zero_bridge-repeat1','external_oracle')])
check('all18_expected_runs',set(summary['trajectory_artifacts'])==expected and len(expected)==18)
trajectories={};files={};lane_chunks=0
for name,artifact in summary['trajectory_artifacts'].items():
    path=ROOT/'tracker'/artifact['file'];d=read(path);rows=d['chunks'];plan=d['plan'];counts={}
    counts['source_and_completion']=d['source_commit']==SOURCE and d['completed'] is True and d['name']==name
    counts['artifact_sha']=sha(path)==artifact['sha256']
    counts['complete16_chunks']=len(rows)==16 and [(r['attempt'],r['chunk']) for r in rows]==[(0,i) for i in range(16)]
    counts['one_attempt_two_lanes']=len(plan['scenes'])==1 and len(plan['scenes'][0])==2 and len(plan['theta_degrees'])==2
    counts['native_seed']=d['native_noise_seed']==810000
    counts['reset_seed_all_chunks']=all(r['reset_seed']==plan['reset_seeds'][0] for r in rows)
    counts['original_input_tree_hashes']=all(digest(r['observation_fields'])==r['observation_sha256'] for r in rows)
    counts['prepared_input_tree_hashes']=all(digest(r['prepared_observation_fields'])==r['prepared_observation_sha256'] for r in rows)
    counts['actual_tensor_snapshot_hashes']=all(float_tensor_hash(r[field])==r[key] for r in rows for field,key in [('state','state_sha256'),('prepared_state','prepared_state_sha256'),('feedback','feedback_sha256'),('decoded_actions','decoded_actions_sha256'),('prepared_commands','prepared_commands_sha256'),('issued_commands','issued_commands_sha256')])
    counts['decode_uses_original_state']=all(r['decode_state_sha256']==r['state_sha256'] for r in rows)
    counts['actual_command_shapes']=all(flat_shape(r[field])[0]==[2,5,7] for r in rows for field in ('decoded_actions','prepared_commands','issued_commands'))
    counts['nonXY_commands_unchanged']=all(a[2:]==b[2:] for r in rows for ac,bc in zip(r['prepared_commands'],r['issued_commands']) for a,b in zip(ac,bc))
    counts['native_generator_continuity']=all(rows[i]['generator_after_sha256']==rows[i+1]['generator_before_sha256'] for i in range(len(rows)-1))
    success=[False,False];consistent=True
    for row in rows:
        if len(row['success_per_step'])!=5 or any(len(v)!=2 for v in row['success_per_step']):consistent=False
        for perstep in row['success_per_step']:
            success=[a or bool(b) for a,b in zip(success,perstep)]
        consistent=consistent and success==row['success_once']
    counts['outcomes_recomputed_all80_steps']=consistent and d['outcomes']==[success] and artifact['outcomes']==[success]
    for key,value in counts.items():check(name+'/'+key,value)
    lane_chunks+=len(rows)*2
    trajectories[name]=d
    files[name]=dict(file=artifact['file'],sha256=sha(path),bytes=path.stat().st_size,chunks=len(rows),outcomes=d['outcomes'],checks=counts,first_native_prefix_sha256=rows[0]['native_kv_sha256'],first_native_noise_sha256=rows[0]['initial_noise_sha256'],trajectory_sha256=digest(rows))
zero={};raw={}
for theta in (-45.,0.,45.):
    prefix=f'heldout-{int(theta):+d}-';a=trajectories[prefix+'zero_bridge'];b=trajectories[prefix+'zero_bridge-repeat1']
    zero[str(theta)]=dict(exact_all_chunk_equality=a['chunks']==b['chunks'],same_plan=a['plan']==b['plan'],same_initial_state=a['initial_state']==b['initial_state'],same_noise_seed=a['native_noise_seed']==b['native_noise_seed'],same_outcomes=a['outcomes']==b['outcomes'],outcomes=a['outcomes'])
    check('zero_repeat_exact/'+str(theta),all(v for k,v in zero[str(theta)].items() if k!='outcomes'))
    check('zero_saved_stability_matches/'+str(theta),summary['zero_stability'][str(theta)]['stable'] is True and summary['zero_stability'][str(theta)]['complete'] is True)
    raw[str(theta)]={condition:trajectories[prefix+condition]['outcomes'] for condition in ('bridge','zero_bridge','external_oracle')}
check('raw_comparisons_match_summary',raw==summary['comparisons'])
selected={str(theta):max((.25,.5,1.),key=lambda gain:(sum(sum(row) for row in trajectories[f'calibrate-{int(theta):+d}-{gain}']['outcomes']),gain)) for theta in (-45.,45.)};selected['0.0']=1.
check('declared_train_only_gain_argmax',selected==summary['selected_gains'])
residuals={};extra_tracker=0
for name,rows in summary['diagnostics'].items():
    check('diagnostic_complete/'+name,len(rows)==16)
    check('actual_bounded_sampler/'+name,all(r['actual_sampler']=='StochasticPi05BridgePolicy.sample_chunk' and r['num_steps']==4 and r['active_dims']==7 and r['residual_scale']==1. and r['actual_executed_action_steps']==5 and not r['training_eligible'] and 0<=r['residual_mean_abs_max']<=1 and 0<=r['residual_bound_fraction']<=1 and math.isfinite(r['target_executed_model_action_mse']) for r in rows))
    extra_tracker+=sum(r['extra_native_prefix_queries'] for r in rows)
    check('tracker_extra_query_factor/'+name,all(r['extra_native_suffix_queries']==4*r['extra_native_prefix_queries'] for r in rows))
    residuals[name]=dict(chunks=len(rows),max_abs_residual=max(r['residual_mean_abs_max'] for r in rows),mean_clipped_fraction=sum(r['residual_bound_fraction'] for r in rows)/len(rows),max_clipped_fraction=max(r['residual_bound_fraction'] for r in rows),mean_executed_target_model_MSE=sum(r['target_executed_model_action_mse'] for r in rows)/len(rows),max_executed_target_model_MSE=max(r['target_executed_model_action_mse'] for r in rows))
external=sum(len(d['chunks'])*2 for name,d in trajectories.items() if name.endswith('external_oracle'))
counters=dict(simulator_steps=5*lane_chunks,policy_chunks=lane_chunks,native_prefix_queries=lane_chunks+external+extra_tracker,native_suffix_queries=4*(lane_chunks+external+extra_tracker),extra_label_motor_calls=external,extra_tracker_target_motor_calls=extra_tracker)
check('all_costs_independently_recomputed',counters==summary['counters'])
check('fixed_full_budget',counters==dict(simulator_steps=2880,policy_chunks=576,native_prefix_queries=928,native_suffix_queries=3712,extra_label_motor_calls=96,extra_tracker_target_motor_calls=256))
pertheta={theta:{c:sum(values[0]) for c,values in comparisons.items()} for theta,comparisons in raw.items()}
totals={c:sum(row[c] for row in pertheta.values()) for c in ('bridge','zero_bridge','external_oracle')}
no_degradation=all(v['bridge']>=v['zero_bridge'] for v in pertheta.values());each_perturbed=all(pertheta[str(theta)]['bridge']>=1 for theta in (-45.,45.));improvement=totals['bridge']>totals['zero_bridge'];passed=no_degradation and each_perturbed and improvement
check('gate_rule_recomputed',summary['review']['passed']==passed==False and summary['review']['status']=='unresolved' and summary['review']['perturbed_success_each']==each_perturbed and totals==summary['review']['totals'])
admission=read(ROOT/'followup-admission.json');check('crossed_and_PPO_not_started',admission['crossed']=='not_run' and admission['ppo']=='not_launched')
result=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),job_id='job-o16q3eimo8bq',source_commit=SOURCE,method='Read-only stdlib JSON/struct/SHA256 over all saved actual command streams; no model computation',checks=checks,all_checks_passed=all(checks.values()),trajectory_files=files,zero_repeats=zero,selected_gains=selected,per_theta=pertheta,totals=totals,counters=counters,residual_diagnostics=residuals,elapsed_seconds=summary['elapsed_seconds'],frozen_motor_before=summary['frozen_motor_before'],frozen_motor_after=summary['frozen_motor_after'],review=summary['review'],followup_admission=admission,qualification_text=(ROOT/'qualification.log').read_text(),exit_code=int((ROOT/'exit-code.txt').read_text()),artifact_hashes={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (ROOT/'tracker/summary.json',ROOT/'tracker/config.json',ROOT/'followup-admission.json',ROOT/'qualification.log',ROOT/'exit-code.txt')})
print(json.dumps(result,indent=2,allow_nan=False))
