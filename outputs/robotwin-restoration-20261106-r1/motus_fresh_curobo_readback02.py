"""Independent CPU audit after the fixed native expert scene scan closes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from motus_fresh_bank import declaration,readback,sha

base=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
raw=base/'motus-fresh-scenes05';signals=base/'motus-fresh-scenes05-signals'
code=Path(os.environ['TASK_CODE']);driver=code/'outputs/robotwin-restoration-20261106-r1'
contract=json.loads((driver/'MOTUS-FRESH-SCENES05-CONTRACT.json').read_bytes())
assert contract['status']=='admitted' and contract['expert_backend']=='curobo'
for f,d in contract['source_files'].items():assert sha(code/f)==d
verified=readback(raw)
r=json.loads((raw/'RESULT.json').read_bytes());bank=json.loads((raw/'BANK.json').read_bytes())
assert r['source_commit']==os.environ['TASK_SCAN_SOURCE']
assert r['expert_backend']=='native_curobo'
assert r['native_collision_configuration']=='unmodified_native_table_only_curobo'
assert r['expert_control_readback_sha256']==contract['expert_control_readback_sha256']
proof=Path(contract['expert_control_readback']);assert sha(proof)==contract['expert_control_readback_sha256']
control=json.loads(proof.read_bytes());assert control['cuRobo_expert_successes']==4 and control['initial_observations_paired']
path=Path(contract['path_result']);assert sha(path)==contract['path_result_sha256']
geometry=json.loads(path.read_bytes());native_yamls={x['native_yaml_sha256'] for x in geometry['rows']}
for group in ('referenced_meshes','referenced_assets'):
    for f,d in geometry[group].items():assert sha(f)==d
resources=r['backend_resources'];assert resources
for x in resources:
    assert x['native_yaml_sha256'] in native_yamls
    assert x['cuda_free_bytes']>=max(8*2**30,int(.08*x['cuda_total_bytes']))
probe=json.loads((signals/'gpu-probe.json').read_bytes())
assert probe['status']=='passed' and probe['purpose']=='evaluation'
assert probe['hostname']==os.environ['TASK_AUDIT_JOB']+'-master-0' and len(probe['devices'])==1
assert probe['admission_policy']=='exclusive_scheduler_evaluation'
device=probe['devices'][0];assert device['allocation_check']['own_pid_gpu_uuids']==[device['uuid']]
assert all(x['cuda_total_bytes']==device['cuda_total_bytes'] for x in resources)
exit_data=json.loads((signals/'EXIT.json').read_bytes())
assert exit_data['exit_code']==0 and exit_data['worker_gpu_seconds']<=contract['worker_timeout_seconds']
worker_readback=json.loads((signals/'READBACK.json').read_bytes());assert worker_readback==verified
scan,candidates=declaration();assert len(verified['seeds'])==64
from motus_fresh_prefix import load
prefix,provenance=load(contract,candidates)
assert r['prefix_provenance']==provenance and r['records'][:83]==prefix
assert len(r['records'])>83 and verified['seeds'][:63]==[x['seed'] for x in prefix if x['play_once_success'] and x['initial_reset_exact']]
assert set(verified['seeds'])<=set(candidates)
assert not set(verified['seeds'])&set(scan['excluded_native_eval_seeds'])
train=json.loads((driver/scan['excluded_train_file']).read_bytes())['success_seeds']
assert not set(train)&set(verified['seeds'])
assert r['model_calls']==r['optimizer_updates']==0 and not r['policy_outcomes_observed']
files=[x for root in (raw,signals) for x in root.rglob('*') if x.is_file()]
assert sum(x.stat().st_size for x in files)<=64*2**20
out=base/'motus-fresh-scenes05-readback01';assert not out.exists();out.mkdir()
report=dict(verified,status='passed_independent_native_curobo_fresh_confirmation_bank',
    audit_source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
    scan_source_commit=r['source_commit'],record_count=len(r['records']),qualified_scene_count=64,
    native_planner_initializations=len(resources),minimum_free_gpu_bytes=min(provenance['prior_minimum_free_gpu_bytes'],min(x['cuda_free_bytes'] for x in resources)),prefix_provenance=provenance,
    old_eval_exclusion_count=len(scan['excluded_native_eval_seeds']),train_exclusion_count=len(train),
    native_collision_configuration_unchanged=True,native_success_seed_inventory_unchanged=True,
    cleanup_complete=True,expert_control_readback_sha256=sha(proof),device=device,
    files={str(x):sha(x) for x in files},output_bytes=sum(x.stat().st_size for x in files),exit=exit_data,
    model_calls=0,optimizer_updates=0,GPU_seconds=0)
(out/'RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+chr(10))
for label,f in [('AUDIT',out/'RESULT.json'),('BANK',raw/'BANK.json'),('QUALIFICATION',raw/'RESULT.json')]:
    d=sha(f);print(label+'_SHA_GROUPS',' '.join(d[i:i+16] for i in range(0,64,16)),flush=True)
print('FRESH_BANK_INDEPENDENT_AUDIT_PASSED',len(r['records']),len(verified['seeds']),report['minimum_free_gpu_bytes']/2**30,report['output_bytes'],flush=True)
