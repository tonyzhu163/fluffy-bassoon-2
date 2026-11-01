"""One finite eight-engine calibration job; no learner or robot computation."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS

root, code, out = map(Path,(os.environ['TASK_ROOT'],os.environ['TASK_META'],os.environ['TASK_OUT']))
exchange = out.parent/'exchange'
node_rank = int(os.environ['RANK'])
started = int(os.environ['TASK_STARTED'])
deadline = started+3420
inputs = json.loads((Path(os.environ['TASK_INPUT'])/'INPUTS.json').read_text())
children, streams = [], []


def wait_for(paths):
    while True:
        if time.time() >= deadline:
            raise TimeoutError('Finite eight-engine calibration clock expired')
        if list((exchange/'errors').glob('*.json')):
            raise RuntimeError('A calibration node failed; no automatic retry or partial quality result')
        if all(path.exists() for path in paths):return
        time.sleep(2)


def carry(receipt, xml, files, names):
    value=read_ref(receipt)
    assert value['completed'] and value['failures']==value['errors']==value['skipped']==0
    assert value['xml']==xml
    read_ref_xml=Path(xml['path'])
    assert sha(read_ref_xml)==xml['sha256']
    cases=ET.parse(read_ref_xml).getroot().findall('.//testcase')
    for name in names:
        rows=[case for case in cases if case.attrib['name']==name]
        assert len(rows)==1 and not any(rows[0].find(tag) is not None for tag in ('error','failure','skipped'))
    for path,expected in files.items():
        assert value['files'][path]==expected==sha(code/path)


def main():
    head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    assert int(os.environ['WORLD_SIZE'])==4 and 0<=node_rank<4 and int(os.environ['NPROC_PER_NODE'])==2
    assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==2
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert int(cpu[0])/int(cpu[1])==32 and int(Path('/sys/fs/cgroup/memory.max').read_text())==256<<30
    assert head(code)==os.environ['TASK_CODE_PIN']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
    assert [row['task'] for row in inputs['cohorts']]==list(PPO_TASKS)
    space=os.statvfs(out)
    assert space.f_bavail*space.f_frsize>=inputs['output_cap_bytes']==128<<30
    probe=out/'write-probe'
    with probe.open('x') as stream:
        stream.write('fresh allocated calibration output\n');stream.flush();os.fsync(stream.fileno())
    probe.unlink()
    write_new(out/'resource.json',dict(hostname=socket.gethostname(),local_gpu_count=2,
        cpus=32,memory_gib=256,node_rank=node_rank,node_count=4,requested_gpus=8,
        free_bytes=space.f_bavail*space.f_frsize,write_probe_pass=True,quota='UNVERIFIED',
        deadline_unix=deadline,robot_actions=0))
    subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),
        '--expected-count','2','--purpose','evaluation','--baige-container',
        '--output',str(out/'gpu-probe.json')],check=True,timeout=min(120,max(1,deadline-time.time())))
    devices=json.loads((out/'gpu-probe.json').read_text())['devices']
    assert len(devices)==2 and len({d['uuid'] for d in devices})==2
    assert all(d['inventory']['memory_total_mib']>=90000 for d in devices)
    for qualification in inputs['carried_qualifications']:
        carry(qualification['receipt'],qualification['xml'],qualification['exact_current_files'],qualification['required_cases'])
    worker=inputs['carried_worker_qualification']
    carry({k:worker[k] for k in ('path','sha256')},worker['xml'],worker['exact_current_files'],[worker['required_case']])
    native=inputs['native_provenance']
    deployment,bindings,submission=(read_ref(native[key]) for key in ('deployment','bindings','submission'))
    assert deployment['source_commit']==submission['source_commit']==native['source_commit']
    assert submission['job_id']==native['job_id']=='job-0vf7klnif9ot'
    assert [v['task'] for v in bindings['tasks']]==list(PPO_TASKS)
    write_new(out/'native-and-core-binding.json',dict(completed=True,native_provenance=native,
        qualifications=inputs['carried_qualifications'],worker_qualification=worker,tests_rerun=0))
    assert head(inputs['official_code'])==inputs['official_code_commit']
    model=read_ref(inputs['model_receipt'])
    assert model['completed'] and model['model_revision']==inputs['model_revision']
    assert Path(model['model_directory']).resolve()==Path(inputs['model']).resolve()
    geometry=read_ref(inputs['three_view_geometry'])
    assert geometry['completed'] and geometry['model_constructed'] is False and geometry['context_limit']==16384
    assert all(row['sampled_frames']==64 and row['total_input_tokens']+row['completion_reserve']<=16384 for row in geometry['requests'])
    write_new(out/'carried-three-view-geometry.json',dict(completed=True,geometry=inputs['three_view_geometry'],
        scope='Qualified fixed three-view native processor geometry; each real inference retains the official context guard, no truncation'))
    write_new(out/'node-ready.json',dict(completed=True,node_rank=node_rank,source_commit=os.environ['TASK_CODE_PIN'],
        gpu_uuids=[d['uuid'] for d in devices],gpu_probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json')),
        inputs_sha256=sha(Path(os.environ['TASK_INPUT'])/'INPUTS.json'),started_unix=started))
    nodes=[out.parent/f'node-{r:02d}/node-ready.json' for r in range(4)]
    wait_for(nodes)
    rows=[json.loads(p.read_text()) for p in nodes]
    assert [row['node_rank'] for row in rows]==list(range(4))
    assert len({uuid for row in rows for uuid in row['gpu_uuids']})==8
    assert all(row['completed'] and row['source_commit']==os.environ['TASK_CODE_PIN']
        and row['inputs_sha256']==sha(Path(os.environ['TASK_INPUT'])/'INPUTS.json') for row in rows)
    for row in rows:assert sha(row['gpu_probe']['path'])==row['gpu_probe']['sha256']
    shared_deadline=min(row['started_unix'] for row in rows)+3420
    python=root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
    toolkit=Path(inputs['toolkit'])
    last_band=0.
    completed=False
    try:
        for local,device in enumerate(devices):
            rank=node_rank*2+local
            directory=out/f'rank-{rank:02d}'
            profile=dict(mode='fifty_case_baseline_only',annotation_world=8,annotation_rank=rank,
                source_commit=os.environ['TASK_CODE_PIN'],output=str(directory),exchange=str(exchange),
                deadline_unix=shared_deadline,closure_seconds=30,batch_size=8,gpu_memory_utilization=.78,
                model=inputs['model'],official_code=inputs['official_code'],model_receipt=inputs['model_receipt'],
                cohorts=inputs['cohorts'],actual_gpu_probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json')),
                expected_gpu_uuid=device['uuid'])
            path=out/f'rank-{rank:02d}-profile.json';write_new(path,profile)
            temporary=Path(f'/tmp/fb-50-{node_rank}-{local}');temporary.mkdir(exist_ok=True)
            cache=out/f'cache-rank-{rank:02d}'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_GPU_MAP=str(out/'gpu-probe.json'),LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
                PYTHONPATH=str(code/'src')+':'+inputs['official_code'],CUDA_HOME=str(toolkit),CUDA_PATH=str(toolkit),
                CUDACXX=str(toolkit/'bin/nvcc'),PATH=str(toolkit/'bin')+':'+str(python.parent)+':'+os.environ['PATH'],
                LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
                LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),TMPDIR=str(temporary),
                XDG_CACHE_HOME=str(cache),TORCHINDUCTOR_CACHE_DIR=str(cache/'torch'),FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
                VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',MAX_JOBS='4',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
            log=(out/f'rank-{rank:02d}-console.log').open('x');streams.append(log)
            children.append(subprocess.Popen([str(python),'-B',str(code/'scripts/diagnostics/robocasa_flora_fifty_score.py'),
                '--profile',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(child.poll() is None for child in children):
            if time.time()>=shared_deadline or list((exchange/'errors').glob('*.json')):
                raise RuntimeError('Finite calibration boundary or peer failure; preserve every existing row')
            if any(child.poll() not in (None,0) for child in children):
                raise RuntimeError('A local scorer failed; no automatic fallback')
            if time.monotonic()-last_band>=60:
                size=int(subprocess.check_output(['du','-sk',str(out)],text=True,timeout=10).split()[0])*1024
                assert size<=31<<30,'Fresh node output band exceeded; no deletion/expansion'
                if node_rank==0 and exchange.exists():
                    shared=int(subprocess.check_output(['du','-sk',str(exchange)],text=True,timeout=10).split()[0])*1024
                    assert shared<=4<<30,'Fresh shared output band exceeded; no deletion/expansion'
                print(json.dumps(dict(node_rank=node_rank,output_allocated_bytes=size,cap_bytes=31<<30)),flush=True)
                last_band=time.monotonic()
            time.sleep(2)
        assert all(child.returncode==0 for child in children)
        wait_for([exchange/'ALL-FIVE.json'])
        final=json.loads((exchange/'ALL-FIVE.json').read_text())
        assert final['completed'] and len(final['task_results'])==5 and final['training_authorization'] is False
        completed=True
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in streams:log.close()
        write_new(out/'summary.json',dict(completed=completed,
            node_rank=node_rank,return_codes=[c.returncode for c in children],elapsed_wall_seconds=time.time()-started,
            robot_actions=0,scope='All five fixed native50 cohorts; no learner or training authorization'))


try:main()
except BaseException as error:
    path=exchange/'errors'/f'node-{node_rank:02d}.json'
    if not path.exists():write_new(path,dict(error=f'{type(error).__name__}: {error}',node_rank=node_rank))
    raise
