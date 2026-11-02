"""Finite four-GPU video replay, independent from all robot jobs."""
import json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from REPLAY import OUT,ROOT,sha,read,write,ref,checked

code=Path(os.environ['TASK_META']);out=Path(os.environ['TASK_OUT']);node=int(os.environ['TASK_NODE_RANK'])
started=int(os.environ['TASK_STARTED']);deadline=started+3180;children=[];streams=[]
def wait_for(paths):
    while not all(p.exists() for p in paths):
        if time.time()>=deadline or list(OUT.glob('node-*/ERROR.json')):raise RuntimeError('Finite replay boundary or peer failure')
        time.sleep(2)
def disk_bytes(path):
    # Files can disappear during an atomic cache rename; count retained files only.
    total=0
    for d,_,names in os.walk(path):
        for name in names:
            try:total+=(Path(d)/name).stat().st_blocks*512
            except FileNotFoundError:pass
    return total
def main():
    assert int(os.environ['WORLD_SIZE'])==2 and int(os.environ['NPROC_PER_NODE'])==2 and node in (0,1)
    assert socket.gethostname().startswith('job-')
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
    cgroup=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert int(cgroup[0])/int(cgroup[1])>=32
    assert int(Path('/sys/fs/cgroup/memory.max').read_text())>=256<<30
    space=os.statvfs(out);assert space.f_bavail*space.f_frsize>=16<<30
    p=out/'write-probe';p.write_text('replay output probe\n');p.unlink()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','2',
        '--purpose','evaluation','--baige-container','--output',str(out/'gpu-probe.json')],check=True,timeout=120)
    devices=read(out/'gpu-probe.json')['devices'];assert len(devices)==2
    m=read(OUT/'MANIFEST.json');assert m['video_count']==634 and m['label_count']==1268
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    write(out/'NODE-READY.json',dict(node=node,source_commit=os.environ['TASK_CODE_PIN'],started_unix=started,
        gpu_uuids=[d['uuid'] for d in devices],probe=ref(out/'gpu-probe.json'),manifest=ref(OUT/'MANIFEST.json'),
        free_bytes=space.f_bavail*space.f_frsize,quota='UNVERIFIED',write_probe=True))
    paths=[OUT/f'node-{i:02d}/NODE-READY.json' for i in range(2)];wait_for(paths);nodes=[read(p) for p in paths]
    assert len({u for n in nodes for u in n['gpu_uuids']})==4
    assert all(n['source_commit']==os.environ['TASK_CODE_PIN'] and n['manifest']==ref(OUT/'MANIFEST.json') for n in nodes)
    shared_deadline=min(n['started_unix'] for n in nodes)+3180
    py=ROOT/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python';toolkit=Path(inputs['toolkit'])
    for local,d in enumerate(devices):
        rank=node*2+local;profile=dict(rank=rank,code=str(code),source_commit=os.environ['TASK_CODE_PIN'],gpu_uuid=d['uuid'],
            manifest=ref(OUT/'MANIFEST.json'),deadline_unix=shared_deadline)
        path=out/f'rank-{rank:02d}-profile.json';write(path,profile)
        cache=out/f'cache-{rank:02d}';tmp=Path(f'/tmp/rubric-{rank:02d}');tmp.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=d['visibility_token'],LYCO_BAIGE_GPU_MAP=str(out/'gpu-probe.json'),
            LYCO_BAIGE_ISOLATED_UUID=d['uuid'],PYTHONPATH=str(code/'src')+':'+inputs['official_code'],CUDA_HOME=str(toolkit),CUDA_PATH=str(toolkit),
            CUDACXX=str(toolkit/'bin/nvcc'),PATH=str(toolkit/'bin')+':'+str(py.parent)+':'+os.environ['PATH'],
            LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),
            TMPDIR=str(tmp),XDG_CACHE_HOME=str(cache),TORCHINDUCTOR_CACHE_DIR=str(cache/'torch'),FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
            VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MAX_JOBS='4')
        stream=(out/f'rank-{rank:02d}.log').open('x');streams.append(stream)
        children.append(subprocess.Popen([str(py),'-B',str(code/'outputs'/os.environ['TASK_RUN']/'REPLAY.py'),'worker','--profile',str(path)],
            env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    checked_at=0
    while any(c.poll() is None for c in children):
        if time.time()>=shared_deadline or list(OUT.glob('node-*/ERROR.json')):raise RuntimeError('Finite clock or peer failure; preserve rows')
        if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Scorer failed; no automatic retry')
        if time.time()-checked_at>60:
            size=disk_bytes(OUT);assert size<=16<<30,'Replay output cap exceeded'
            print(json.dumps(dict(node=node,output_bytes=size,checked_unix=time.time())),flush=True);checked_at=time.time()
        time.sleep(2)
    assert all(c.returncode==0 for c in children)
    write(out/'COMPLETED.json',dict(completed=True,node=node,source_commit=os.environ['TASK_CODE_PIN'],return_codes=[c.returncode for c in children],robot_actions=0))
    # Each node releases independently; the CPU summary runs after both exits.

try:main()
except BaseException as e:
    if not (out/'ERROR.json').exists():write(out/'ERROR.json',dict(error=f'{type(e).__name__}: {e}',node=node))
    raise
finally:
    for c in children:
        if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
    for c in children:
        try:c.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    for stream in streams:stream.close()
