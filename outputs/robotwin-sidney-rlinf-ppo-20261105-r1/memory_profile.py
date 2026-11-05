"""Model-free host-memory attribution on the actual RoboTwin simulator."""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import tracemalloc
import threading

import numpy as np
import torch
from omegaconf import OmegaConf
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal,write,tensor_hash,snapshot_rng
from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer,_install_reset_retry_limit


root=Path(os.environ['TASK_OUT']);source=root.parent/'train06';scenes=int(os.environ.get('TASK_SCENES','16'))
os.chdir(os.environ['ROBOTWIN_PATH'])
_install_allocated_renderer(root/'renderer');_install_reset_retry_limit()
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

report=dict(status='running',optimizer_updates=0,model_loaded=False,scenes=scenes,snapshots=[],events=[])
tracemalloc.start(5)
peak={'rss_bytes':0,'cgroup_bytes':0};stop=threading.Event()
def watch():
    while not stop.wait(.2):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):peak['rss_bytes']=max(peak['rss_bytes'],int(line.split()[1])*1024)
        path=Path('/sys/fs/cgroup/memory.current')
        if path.exists():peak['cgroup_bytes']=max(peak['cgroup_bytes'],int(path.read_text()))
monitor=threading.Thread(target=watch,daemon=True);monitor.start()
libc=ctypes.CDLL('libc.so.6')
class Mallinfo(ctypes.Structure):
    _fields_=[(name,ctypes.c_size_t) for name in ('arena','ordblks','smblks','hblks','hblkhd','usmblks','fsmblks','uordblks','fordblks','keepcost')]
libc.mallinfo2.restype=Mallinfo
def sample(stage,**extra):
    status={}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        if ':' in line:
            key,value=line.split(':',1)
            if value.strip().endswith('kB'):status[key]=int(value.split()[0])*1024
    m=libc.mallinfo2();base=Path('/sys/fs/cgroup')
    row=dict(stage=stage,at_unix=time.time(),smaps=status,
        mallinfo={name:int(getattr(m,name)) for name,_ in m._fields_},**extra)
    row['python_traced_current_peak_bytes']=list(tracemalloc.get_traced_memory())
    for name in ('memory.current','memory.peak','memory.max'):
        if (base/name).exists():row[name]=(base/name).read_text().strip()
    if (base/'memory.stat').exists():row['cgroup_stat']=dict((k,int(v)) for k,v in (l.split() for l in (base/'memory.stat').read_text().splitlines()))
    report['snapshots'].append(row);write(root/'RESULT.json',report)
    print(json.dumps(dict(stage=stage,rss_gib=status.get('Rss',0)/2**30,pss_gib=status.get('Pss',0)/2**30,
        live_glibc_gib=m.uordblks/2**30,free_glibc_gib=m.fordblks/2**30)),flush=True)

def digest(obs):
    for key in ('main_images','wrist_images'):
        image=torch.as_tensor(obs[key]);assert image.dtype==torch.uint8
        a=image.reshape(-1,*image.shape[-3:])[:,::16,::16].float().flatten(1)
        assert bool((a.std(1)>1).all()),f'Blank {key}'
    assert bool(torch.isfinite(obs['states']).all())
    return tensor_hash(obs)

journal=ActionJournal(source.parent/'train03/action-accounting',rank=0,world=4,
    attempt=root.name,maximum_actions=1200000,previous_maximum_actions=1000000)
cfg=OmegaConf.create(json.loads((source/'resolved-native-config.json').read_text())['env']['train'])
from robotwin.envs import vector_env
original_setup=vector_env.SubEnv.setup_task
constructed=0
def measured_setup(self):
    global constructed
    result=original_setup(self);constructed+=1
    if constructed in (1,4,8,16,32,64):sample(f'constructed_scene_{constructed}')
    return result
vector_env.SubEnv.setup_task=measured_setup
original_factory=vector_env.class_decorator
instrumented=set();counts={}
def measured_factory(name):
    task=original_factory(name)
    for method in ('setup_scene','load_robot','load_camera'):
        cls=next((c for c in type(task).__mro__ if method in c.__dict__),None)
        if cls is None or (cls,method) in instrumented:continue
        original=getattr(cls,method);instrumented.add((cls,method))
        def measured(self,*args,_original=original,_method=method,**kwargs):
            count=counts.get(_method,0);counts[_method]=count+1
            if count==0:sample(f'first/{_method}/before')
            result=_original(self,*args,**kwargs)
            if count==0:sample(f'first/{_method}/after')
            return result
        setattr(cls,method,measured)
    return task
vector_env.class_decorator=measured_factory
sample('imports_before_constructor')
slot=journal.begin(scenes*10,phase='memory-diagnostic/constructor-allowance')
env=RoboTwinEnv(cfg,scenes,seed_offset=0,total_num_processes=4,worker_info=None)
journal.finish(slot,scenes*10);sample(f'constructor_{scenes}_scenes')
successful=False
paired=None
try:
    for round_index in range(3 if scenes==64 else 2):
        if scenes==64 and round_index==1:cfg.malloc_trim_reset_interval=1
        if scenes==64 and round_index==2:
            for subenv in env.venv.envs:subenv._fb2_trim_after_scene_close=True
        peak.update(rss_bytes=0,cgroup_bytes=0)
        slot=journal.begin(scenes*10,phase='memory-diagnostic/reset-allowance')
        obs,_=env.reset();journal.finish(slot,scenes*10)
        reset_peak=dict(peak)
        seen=set()
        def array_bytes(value):
            if id(value) in seen:return 0
            seen.add(id(value))
            if isinstance(value,np.ndarray):return value.nbytes
            if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
            if isinstance(value,dict):return sum(array_bytes(v) for v in value.values())
            if isinstance(value,(list,tuple)):return sum(array_bytes(v) for v in value)
            return 0
        native_fields=[]
        for sub in env.venv.envs:
            native_fields.append({key:array_bytes(value) for key,value in vars(sub.task).items()
                if isinstance(value,(dict,list,tuple,np.ndarray,torch.Tensor))})
        del sub
        before=digest(obs);sample(f'round{round_index}/after_reset',observation_sha256=before,
            reset_peak=reset_peak,native_malloc_trim_reset_interval=int(cfg.get('malloc_trim_reset_interval',0)),
            trim_after_each_scene_close=round_index==2,
            task_reachable_array_bytes=native_fields,
            largest_python_traces=[dict(bytes=s.size,count=s.count,trace=str(s.traceback))
                for s in tracemalloc.take_snapshot().statistics('traceback')[:8]])
        # Distinguish Python/native live objects from freed glibc arenas.
        collected=gc.collect();sample(f'round{round_index}/after_gc',gc_collected=collected)
        rng_before=tensor_hash(snapshot_rng());scene_before=[s.task.get_info() for s in env.venv.envs]
        motor_before=tensor_hash(obs['states'][:,None,:].repeat(1,50,1))
        trimmed=libc.malloc_trim(0);sample(f'round{round_index}/after_malloc_trim',trim_return=int(trimmed))
        after=digest(obs)
        assert before==after,'Allocator diagnostic changed retained observation bytes'
        assert rng_before==tensor_hash(snapshot_rng())
        assert scene_before==[s.task.get_info() for s in env.venv.envs]
        assert motor_before==tensor_hash(obs['states'][:,None,:].repeat(1,50,1))
        report['events'].append(dict(round=round_index,allocator_observation_identity=True,
            python_numpy_torch_rng_identity=True,scene_identity=True,motor_input_identity=True))
        if scenes==64 and round_index==1:
            paired={key:obs[key].clone() for key in ('main_images','wrist_images','states')}
        if scenes==64 and round_index==2:
            changes={key:float((obs[key].float()-paired[key].float()).abs().max()) for key in paired}
            assert changes['states']<=1e-6,'Per-scene trim changed matched reset joint state'
            report['paired_reset_max_abs']=changes
            report['paired_reset_rgb_bit_identity']=all(changes[key]==0 for key in ('main_images','wrist_images'))
        # Replay only a fixed absolute-joint hold, never model inference.
        count=8 if scenes==16 and round_index==0 else 1
        for chunk in range(count):
            actions=obs['states'].detach().cpu()[:,None,:].repeat(1,50,1)
            slot=journal.begin(scenes*50,phase='memory-diagnostic/physical-hold')
            result=env.chunk_step(actions)
            mask=result[-1][-1].get('valid_action_mask');executed=int(mask.sum()) if mask is not None else scenes*50
            journal.finish(slot,executed);obs=result[0][-1]
            if scenes==64 and round_index==1:
                paired['after_hold_states']=obs['states'].clone()
                paired['after_hold_rewards']=result[1].clone()
            if scenes==64 and round_index==2:
                error=float((obs['states']-paired['after_hold_states']).abs().max())
                assert error<=1e-6 and torch.equal(result[1],paired['after_hold_rewards'])
                report['paired_motor_prefix_state_max_abs']=error
                report['paired_motor_prefix_reward_identity']=True
            sample(f'round{round_index}/chunk{chunk}',physical_controls=executed)
            del result,actions
        sample(f'round{round_index}/round_end')
        if round_index==0:env.update_reset_state_ids()
    successful=True
except BaseException as error:
    report.update(error_type=type(error).__name__,error=str(error));raise
finally:
    env.venv.close(clear_cache=True);sample('after_vector_close')
    del env;gc.collect();sample('after_delete_gc')
    libc.malloc_trim(0);sample('after_delete_trim')
    stop.set();monitor.join()
    report.update(status='complete' if successful else 'failed',total_shared_rank0_charge=journal.charged(),
        physical_controls=sum(r.get('executed_or_conservative',r['requested']) for r in journal.rows if r['phase']=='memory-diagnostic/physical-hold'),
        active_worktrees_changed=False,scientific_source_admission=False)
    write(root/'RESULT.json',report)
