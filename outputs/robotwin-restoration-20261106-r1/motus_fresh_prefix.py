"""Keep a completed policy-free qualification prefix without rerunning candidates."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

from motus_fresh_bank import accepted,declaration,sha,write


def audit(directory,output):
    directory,output=map(Path,(directory,output))
    r=json.loads((directory/'RESULT.json').read_bytes());plan,candidates=declaration()
    assert r['status']=='failed' and r['error_type']=='AssertionError'
    assert r['error']=='Stop before the cleanup/release margin' and r['cleanup_complete']
    assert r['source_commit']=='b66c80b9cbf6b77d9746685143914c158f50398f'
    assert r['model_calls']==r['optimizer_updates']==0 and not r['policy_outcomes_observed']
    assert r['native_success_seed_inventory_unchanged'] and not (directory/'BANK.json').exists()
    assert r['expert_backend']=='native_curobo'
    assert r['native_collision_configuration']=='unmodified_native_table_only_curobo'
    assert r['native_vector_env_sha256']==plan['native_vector_env_sha256']
    assert r['scan_sha256']==sha(Path(__file__).with_name('MOTUS-FRESH-CONFIRMATION-SCAN.json'))
    rows=r['records'];assert len(rows)==83 and [x['seed'] for x in rows]==candidates[:83]
    assert sum(accepted(x) for x in rows)==63
    resources=r['backend_resources'];assert len(resources)==292
    for x in resources:assert x['cuda_free_bytes']>=max(8*2**30,int(.08*x['cuda_total_bytes']))
    signals=directory.with_name(directory.name+'-signals')
    exit_data=json.loads((signals/'EXIT.json').read_bytes())
    assert exit_data['exit_code']==1 and exit_data['worker_gpu_seconds']==1640
    probe=json.loads((signals/'gpu-probe.json').read_bytes())
    assert probe['status']=='passed' and probe['hostname']=='job-tdmed4t504sh-master-0' and len(probe['devices'])==1
    assert probe['devices'][0]['allocation_check']['own_pid_gpu_uuids']==[probe['devices'][0]['uuid']]
    files=[x for root in (directory,signals) for x in root.rglob('*') if x.is_file()]
    assert sum(x.stat().st_size for x in files)<=64*2**20
    assert not output.exists();output.mkdir()
    report=dict(status='passed_complete_ordered_qualification_prefix',raw_result=str(directory/'RESULT.json'),raw_result_sha256=sha(directory/'RESULT.json'),scan_sha256=r['scan_sha256'],source_commit=r['source_commit'],record_count=83,qualified_scene_count=63,next_candidate=candidates[83],minimum_free_gpu_bytes=min(x['cuda_free_bytes'] for x in resources),native_collision_configuration_unchanged=True,cleanup_complete=True,policy_outcomes_observed=False,model_calls=0,optimizer_updates=0,files={str(x):sha(x) for x in files},exit=exit_data,GPU_seconds=0)
    write(output/'RESULT.json',report)
    for label,f in [('PREFIX_AUDIT',output/'RESULT.json'),('PREFIX_RESULT',directory/'RESULT.json')]:
        d=sha(f);print(label+'_SHA_GROUPS',' '.join(d[i:i+16] for i in range(0,64,16)),flush=True)
    print('PREFIX_AUDIT_PASSED',83,63,candidates[83],flush=True)


def load(contract,candidates):
    p=Path(contract['prefix_audit_result']);assert sha(p)==contract['prefix_audit_sha256']
    evidence=json.loads(p.read_bytes());assert evidence['status']=='passed_complete_ordered_qualification_prefix'
    raw=Path(contract['prefix_result']);assert sha(raw)==contract['prefix_result_sha256']==evidence['raw_result_sha256']
    r=json.loads(raw.read_bytes());assert r['cleanup_complete'] and r['model_calls']==r['optimizer_updates']==0
    assert not r['policy_outcomes_observed'] and r['native_success_seed_inventory_unchanged']
    assert r['scan_sha256']==contract['scan_sha256']==evidence['scan_sha256']
    assert r['expert_control_readback_sha256']==contract['expert_control_readback_sha256']
    rows=deepcopy(r['records']);assert [x['seed'] for x in rows]==candidates[:len(rows)]
    assert len(rows)==83 and sum(accepted(x) for x in rows)==63
    assert candidates[len(rows)]==evidence['next_candidate']==100100325
    return rows,dict(audit_result=str(p),audit_sha256=sha(p),raw_result=str(raw),raw_result_sha256=sha(raw),record_count=len(rows),qualified_scene_count=63,next_candidate=candidates[len(rows)],prior_minimum_free_gpu_bytes=evidence['minimum_free_gpu_bytes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();audit(args.audit,args.output)
