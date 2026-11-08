"""Run-local CPU asset dependency qualification and prior control readback."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from motus_curobo_paths import prepare

code = Path(os.environ['TASK_CODE'])
source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
base = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
root = code/'outputs/robotwin-restoration-20261106-r1'
p = json.loads((root/'MOTUS-CUROBO-CONTROL02-CONTRACT.json').read_bytes())
h = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
write = lambda f,d: Path(f).write_text(json.dumps(d,indent=2,sort_keys=True)+chr(10))
assert p['status']=='prepared_not_admitted' and not p['cpu_preflight_passed']
assert p['all_worst_gpu_seconds']==357635+1800==359435<=360000
assert p['seeds']==[100100002,100100003,100100005,100100006]
for f,d in p['source_files'].items(): assert h(code/f)==d
for k in ('backend_readback','reference_control','original_configuration'):assert h(p[k])==p[k+'_sha256']
# Independently preserve the first failed control's terminal evidence.
prior=json.loads((base/'motus-curobo-control01/RESULT.json').read_bytes())
assert h(base/'motus-curobo-control01/RESULT.json')=='b73946cf34a1b7cf6f7755a6071bc5f73fecd010019d971995e279def52eff5d'
assert prior['full_native_API_import_passed'] and prior['cleanup_complete']
assert prior['records']==prior['planner_calls']==[] and prior['VLA_calls']==prior['optimizer_updates']==0
assert 'collision_aloha_left.yml' in prior['error'] and prior['error_type']=='RuntimeError'
exit_data=json.loads((base/'motus-curobo-control01-signals/EXIT.json').read_bytes())
assert exit_data['exit_code']==1 and exit_data['worker_gpu_seconds']==73
readback=base/'motus-curobo-control01-readback01';assert not readback.exists();readback.mkdir()
files={str(f):h(f) for f in [base/'motus-curobo-control01/RESULT.json',*sorted((base/'motus-curobo-control01-signals').glob('*'))] if f.is_file()}
write(readback/'RESULT.json',dict(status='verified_pre_scene_missing_collision_asset',source_commit=source,files=files,full_native_API_import_passed=True,expert_episodes=0,planner_calls=0,model_calls=0,optimizer_updates=0,exit=exit_data,cleanup_complete=True))
robotwin=Path(os.environ['ROBOTWIN_PATH']);local=Path('/tmp/rt-curobo-control-cpu02');assert not local.exists();local.mkdir()
rows=[];meshes={};assets={}
for name in ('curobo_left.yml','curobo_right.yml'):
    row=prepare(robotwin/'assets/embodiments/aloha-agilex'/name,robotwin,local/name);rows.append(row)
    for item in row['repaired_asset_paths'].values(): assets[item['actual']]=item['sha256']
    urdf=Path(row['actual_urdf'])
    for mesh in ET.parse(urdf).findall('.//mesh'):
        filename=mesh.attrib['filename'];assert not filename.startswith(('package:','file:'))
        actual=(urdf.parent/filename).resolve(strict=True);assert actual.is_file()
        meshes[str(actual)]=h(actual)
    import yaml
    spheres=yaml.safe_load(Path(row['actual_collision_yaml']).read_bytes())['collision_spheres']
    count=0
    for link,entries in spheres.items():
        assert isinstance(link,str) and isinstance(entries,list)
        for sphere in entries:
            assert len(sphere['center'])==3 and all(isinstance(v,(int,float)) for v in sphere['center'])
            assert isinstance(sphere['radius'],(int,float)) and sphere['radius']>0
            count+=1
    row['collision_sphere_count']=count;assert count>0
assert len(meshes)==30
path_out=Path(p['path_result']).parent;assert not path_out.exists();path_out.mkdir()
write(path_out/'RESULT.json',dict(status='passed_CPU_all_native_asset_paths',source_commit=source,rows=rows,referenced_meshes=meshes,referenced_assets=assets,native_yaml_unchanged=True,numerical_configuration_unchanged=True,CUDA_initialized=False,GPU_expert_qualified=False))
# No simulator/API imports on CPU; only the five independently compiled ABI modules.
import importlib,torch
backend=json.loads(Path(p['backend_readback']).read_bytes())
for name,row in backend['extension_modules'].items():
    assert h(row['file'])==row['sha256'];importlib.import_module(name)
assert not torch.cuda.is_initialized()
for f in ('motus_curobo_paths.py','motus_curobo_control.py'): ast.parse((root/f).read_text())
subprocess.run(['bash','-n',str(root/'motus_curobo_control02.sh')],check=True)
out=Path(p['CPU_result']).parent;assert not out.exists();out.mkdir()
result=dict(status='passed_Linux_CPU_all_asset_dependency_preflight',source_commit=source,path_result=str(path_out/'RESULT.json'),path_result_sha256=h(path_out/'RESULT.json'),prior_control_readback=str(readback/'RESULT.json'),prior_control_readback_sha256=h(readback/'RESULT.json'),verified_meshes=len(meshes),verified_assets=assets,collision_sphere_counts=[r['collision_sphere_count'] for r in rows],native_yaml_unchanged=True,numerical_configuration_unchanged=True,CUDA_initialized=False,GPU_seconds=0,expert_qualified=False,closed_gpu_seconds=357635,reserved_gpu_seconds=1800,worst_gpu_seconds=359435)
write(out/'RESULT.json',result)
print('CPU02_PASSED',json.dumps(result,sort_keys=True),flush=True)
for label,f in [('CPU',out/'RESULT.json'),('PATH',path_out/'RESULT.json'),('PRIOR',readback/'RESULT.json')]:
    d=h(f);print(label+'_SHA_PARTS',d[:32],d[32:],flush=True)
