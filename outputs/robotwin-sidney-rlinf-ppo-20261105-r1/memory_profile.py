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
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal,write,tensor_hash,snapshot_rng,restore_rng
from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer,_install_reset_retry_limit


root=Path(os.environ['TASK_OUT']);source=root.parent/'train06';scenes=int(os.environ.get('TASK_SCENES','16'))
profile_mode=os.environ.get('TASK_PROFILE_MODE','full')
native_repeat=profile_mode=='repeat'
paired_only=profile_mode in ('paired','repeat')
accounting_rank=int(os.environ.get('TASK_ACCOUNTING_RANK','0'))
assert not paired_only or scenes==64
os.chdir(os.environ['ROBOTWIN_PATH'])
_install_allocated_renderer(root/'renderer');_install_reset_retry_limit()
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

report=dict(status='running',optimizer_updates=0,model_loaded=False,scenes=scenes,snapshots=[],events=[])
report.update(profile_mode=profile_mode,accounting_rank=accounting_rank,source_seed_offset=0,
    per_close_round_index=3 if native_repeat else 2,motor_comparisons=[],hold_rng=[])
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

journal=ActionJournal(source.parent/'train03/action-accounting',rank=accounting_rank,world=4,
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
reference_hold_rng=None
if native_repeat:
    from physics_probe import snapshot as physics_snapshot
try:
    if paired_only:env.update_reset_state_ids()
    rounds=(1,2,3) if native_repeat else ((1,2) if paired_only else range(3 if scenes==64 else 2))
    for round_index in rounds:
        if scenes==64 and round_index==1:cfg.malloc_trim_reset_interval=1
        per_close=scenes==64 and round_index==report['per_close_round_index']
        if scenes==64:
            for subenv in env.venv.envs:subenv._fb2_trim_after_scene_close=per_close
        peak.update(rss_bytes=0,cgroup_bytes=0)
        slot=journal.begin(scenes*10,phase='memory-diagnostic/reset-allowance')
        obs,_=env.reset();journal.finish(slot,scenes*10)
        reset_peak=dict(peak)
        if native_repeat:write(root/f'round{round_index}-reset-physics.json',physics_snapshot(env))
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
            trim_after_each_scene_close=per_close,
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
            write(root/'paired-reference.json',dict(reset_observation_sha256=before,
                states=obs['states'].tolist()))
        if scenes==64 and round_index>1:
            changes={key:float((obs[key].float()-paired[key].float()).abs().max()) for key in ('main_images','wrist_images','states')}
            report.setdefault('reset_comparisons',[]).append(dict(round=round_index,per_close=per_close,max_abs=changes))
            if per_close:
                report['paired_reset_max_abs']=changes
                report['paired_reset_rgb_bit_identity']=all(changes[key]==0 for key in ('main_images','wrist_images'))
            write(root/'RESULT.json',report)
            assert changes['states']<=1e-6,'Matched reset joint state differs'
        # Replay only a fixed absolute-joint hold, never model inference.
        count=8 if scenes==16 and round_index==0 else 1
        for chunk in range(count):
            actions=obs['states'].detach().cpu()[:,None,:].repeat(1,50,1)
            reset_states=obs['states'].detach().cpu().clone()
            before_rng=tensor_hash(snapshot_rng())
            if native_repeat:
                if reference_hold_rng is None:reference_hold_rng=snapshot_rng()
                restore_rng(reference_hold_rng)
            hold_rng=snapshot_rng()
            action_hash=tensor_hash(actions)
            submitted_actions=actions.clone()
            report['hold_rng'].append(dict(round=round_index,after_reset_sha256=before_rng,
                actual_hold_sha256=tensor_hash(hold_rng),action_sha256=action_hash,
                diagnostic_only_reference_restore=native_repeat))
            write(root/'RESULT.json',report)
            slot=journal.begin(scenes*50,phase='memory-diagnostic/physical-hold')
            result=env.chunk_step(actions)
            mask=result[-1][-1].get('valid_action_mask');executed=int(mask.sum()) if mask is not None else scenes*50
            journal.finish(slot,executed);obs=result[0][-1]
            torch.save(dict(reset_states=reset_states,actual_actions=submitted_actions,
                actions_after_call=actions.clone(),rng_before_hold=hold_rng,
                states_after=obs['states'].detach().cpu(),rewards=result[1].detach().cpu(),
                terminations=result[2].detach().cpu(),truncations=result[3].detach().cpu(),
                valid_action_mask=None if mask is None else torch.as_tensor(mask).cpu()),root/f'round{round_index}-hold{chunk}.pt')
            if native_repeat:write(root/f'round{round_index}-after-hold-physics.json',physics_snapshot(env))
            if scenes==64 and round_index==1:
                paired['after_hold_states']=obs['states'].clone()
                paired['after_hold_rewards']=result[1].clone()
                paired['after_hold_terminations']=result[2].clone()
                paired['after_hold_truncations']=result[3].clone()
                paired['action_sha256']=action_hash
                paired['valid_mask']=None if mask is None else torch.as_tensor(mask).clone()
                reference=json.loads((root/'paired-reference.json').read_text())
                reference.update(after_hold_states=obs['states'].tolist(),after_hold_rewards=result[1].tolist())
                write(root/'paired-reference.json',reference)
            if scenes==64 and round_index>1:
                difference=(obs['states']-paired['after_hold_states']).abs().detach().cpu()
                reward_difference=(result[1]-paired['after_hold_rewards']).abs().detach().cpu()
                comparison=dict(round=round_index,per_close=per_close,state_max_abs=float(difference.max()),
                    per_environment_max_abs=difference.reshape(scenes,-1).amax(1).tolist(),
                    per_joint_max_abs=difference.amax(0).tolist(),
                    state_offending_indices=(difference>1e-6).nonzero().tolist(),
                    reward_max_abs=float(reward_difference.max()),
                    reward_offending_indices=(reward_difference!=0).nonzero().tolist(),
                    reward_identity=torch.equal(result[1],paired['after_hold_rewards']),
                    termination_identity=torch.equal(result[2],paired['after_hold_terminations']),
                    truncation_identity=torch.equal(result[3],paired['after_hold_truncations']),
                    valid_mask_identity=(mask is None and paired['valid_mask'] is None) or
                        (mask is not None and paired['valid_mask'] is not None and
                         torch.equal(torch.as_tensor(mask),paired['valid_mask'])),
                    action_identity=action_hash==paired['action_sha256'])
                report['motor_comparisons'].append(comparison)
                if per_close:
                    report['paired_motor_prefix_state_max_abs']=comparison['state_max_abs']
                    report['paired_motor_prefix_reward_identity']=comparison['reward_identity']
                write(root/'RESULT.json',report)
            sample(f'round{round_index}/chunk{chunk}',physical_controls=executed)
            del result,actions,submitted_actions
        sample(f'round{round_index}/round_end')
        if round_index==0:env.update_reset_state_ids()
    for comparison in report['motor_comparisons']:
        assert comparison['state_max_abs']<=1e-6 and all(comparison[k] for k in
            ('action_identity','reward_identity','termination_identity','truncation_identity','valid_mask_identity')),comparison
    successful=True
except BaseException as error:
    report.update(error_type=type(error).__name__,error=str(error));raise
finally:
    env.venv.close(clear_cache=True);sample('after_vector_close')
    del env;gc.collect();sample('after_delete_gc')
    libc.malloc_trim(0);sample('after_delete_trim')
    stop.set();monitor.join()
    report.update(status='complete' if successful else 'failed',total_shared_accounting_rank_charge=journal.charged(),
        physical_controls=sum(r.get('executed_or_conservative',r['requested']) for r in journal.rows if r['phase']=='memory-diagnostic/physical-hold'),
        active_worktrees_changed=False,scientific_source_admission=False)
    write(root/'RESULT.json',report)
