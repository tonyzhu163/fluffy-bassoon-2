"""Independent CPU readback of the completed native expert positive control."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

base=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
raw=base/'motus-curobo-control02/RESULT.json';signals=base/'motus-curobo-control02-signals'
h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert h(raw)=='6020ff8f8251aaa538c675a1e08ebadf45a4e2f54608f686a6e8fbe747415f13'
r=json.loads(raw.read_bytes());reference=json.loads((base/'motus-planner-control01/RESULT.json').read_bytes())
old={x['seed']:x for x in reference['records'] if x['profile']=='full_state'}
assert r['status']=='qualified_four_scene_expert_control' and r['expert_successes']==4
assert r['source_commit']=='2a14c88d410e5f80abcf82ed4d0904643378dde8'
assert r['cleanup_complete'] and r['full_native_API_import_passed'] and r['initial_observations_paired']
assert r['VLA_calls']==r['optimizer_updates']==0 and not r['fresh_bank_published']
assert not r['learning_effectiveness_tested'] and r['native_seed_inventory_unchanged']
assert r['collision_configuration']=='unmodified_native_table_only_curobo'
assert [x['seed'] for x in r['records']]==[100100002,100100003,100100005,100100006]
for x in r['records']:
    assert x['seed']==x['physical_seed'] and x['setup_demo_success'] and x['play_once_success']
    assert x['paired_with_preserved_control'] and x['initial_observation_sha256']==old[x['seed']]['initial_observation_sha256']
    assert not old[x['seed']]['play_once_success']
assert len(r['planner_calls'])==24 and all('status' in x and x['seconds']>=0 for x in r['planner_calls'])
assert len(r['resource_snapshots'])==len(r['path_repairs'])==16
for x in r['resource_snapshots']:assert x['cuda_free_bytes']>=max(8*2**30,int(.08*x['cuda_total_bytes']))
code=Path(os.environ['TASK_CODE']);native=code/'outputs/robotwin-restoration-20261106-r1'
p=json.loads((native/'MOTUS-CUROBO-CONTROL02-CONTRACT.json').read_bytes())
paths=json.loads(Path(p['path_result']).read_bytes());expected={x['native_yaml']:x for x in paths['rows']}
for x in r['path_repairs']:
    e=expected[x['native_yaml']]
    assert x['native_yaml_unchanged'] and x['only_asset_paths_changed']
    for k in ('native_yaml_sha256','actual_urdf_sha256','actual_collision_yaml_sha256','repaired_asset_paths'):assert x[k]==e[k]
    assert h(x['native_yaml'])==x['native_yaml_sha256']
for key in ('referenced_assets','referenced_meshes'):
    for f,d in paths[key].items():assert h(f)==d
exit_data=json.loads((signals/'EXIT.json').read_bytes());assert exit_data['exit_code']==0 and exit_data['worker_gpu_seconds']==157
probe=json.loads((signals/'gpu-probe.json').read_bytes());assert probe['status']=='passed'
assert probe['hostname'].startswith('job-yacelskvbzo0') and len(probe['devices'])==1
assert probe['purpose']=='evaluation' and probe['admission_policy']=='exclusive_scheduler_evaluation'
device=probe['devices'][0];assert device['allocation_ordinal']==0 and device['allocation_check']['own_pid_gpu_uuids']==[device['uuid']]
log=(signals/'qualify.log').read_text();assert 'FINAL_NATIVE_CUROBO qualified_four_scene_expert_control 4 '+h(raw) in log
out=base/'motus-curobo-control02-readback01';assert not out.exists();out.mkdir()
report=dict(status='passed_independent_four_scene_native_expert_control',audit_source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),control_source_commit=r['source_commit'],raw_result=str(raw),raw_result_sha256=h(raw),files={str(f):h(f) for f in [raw,*signals.glob('*')] if f.is_file()},seeds=[x['seed'] for x in r['records']],MPLib_expert_successes=0,cuRobo_expert_successes=4,initial_observations_paired=True,native_collision_configuration_unchanged=True,numerical_configuration_unchanged=True,completed_planner_calls=len(r['planner_calls']),minimum_free_gpu_bytes=min(x['cuda_free_bytes'] for x in r['resource_snapshots']),device=device,cleanup_complete=True,exit=exit_data,model_calls=0,optimizer_updates=0,fresh_bank_published=False,learning_effectiveness_tested=False,GPU_seconds=0)
(out/'RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+chr(10))
d=h(out/'RESULT.json');print('CONTROL_READBACK_PASSED',report['cuRobo_expert_successes'],report['completed_planner_calls'],report['minimum_free_gpu_bytes']/2**30,flush=True);print('READBACK_SHA_PARTS',d[:32],d[32:],flush=True)
