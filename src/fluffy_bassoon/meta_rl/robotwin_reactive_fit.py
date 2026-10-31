"""Small supervised reactive fit, reusing frozen RoboTwin runtime/checkpoint APIs.

Only an admitted GPU caller constructs the runtime. These helpers contain no
scheduler, new simulator loop or replacement action normalization.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual, ReactiveALOHAActionResidual, DecodedActionResidual
from .checkpoint import rng_state
from .warmstart import atomic_save
from .robotwin_demo_capacity import file_sha, read_episode, raw_observation, whole_episode_chunks, transformed_target, masked_mse, reactive_flow_loss


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(cpu(v) for v in value)
    return deepcopy(value)


ARM_INDICES=[0,1,2,3,4,5,7,8,9,10,11,12]


class CapacityActionResidual(ReactiveALOHAActionResidual):
    """Existing head with12-arm headroom enforcement; raw grippers untouched."""
    def __init__(self,*,joint_lower,joint_upper,**kwargs):
        super().__init__(**kwargs)
        lower=torch.as_tensor(joint_lower,dtype=torch.float32)
        upper=torch.as_tensor(joint_upper,dtype=torch.float32)
        if (lower.shape!=(14,) or upper.shape!=(14,) or not torch.isfinite(lower).all()
                or not torch.isfinite(upper).all() or not (lower<upper).all()):
            raise ValueError('Require exact source joint limits14 before extracting12 arms')
        self.register_buffer('arm_lower',lower[ARM_INDICES].clone())
        self.register_buffer('arm_upper',upper[ARM_INDICES].clone())
        with torch.no_grad():self.bounds[[6,13]]=0.
    def config(self):
        return dict(super().config(),arm_indices=ARM_INDICES,arm_lower=self.arm_lower.cpu().tolist(),
            arm_upper=self.arm_upper.cpu().tolist(),arm_limit_enforcement='clip correction to actual remaining headroom',
            raw_native_grippers='bitwise preserved, never clamped/rejected for range overshoot')
    def forward(self,base,features):
        arms=base[...,ARM_INDICES]
        if ((arms<self.arm_lower).any() or (arms>self.arm_upper).any()):
            raise ValueError('Native base ARM exceeds source joint limits; cannot preserve safe zero parity')
        result=super().forward(base,features)
        actions=result.actions.clone()
        actions[...,ARM_INDICES]=result.actions[...,ARM_INDICES].clamp(min=self.arm_lower,max=self.arm_upper)
        actions[...,[6,13]]=result.base_actions[...,[6,13]]
        return DecodedActionResidual(result.base_actions,result.requested_delta,actions-result.base_actions,
                                     actions!=result.actions,actions)


def make_heads(backend,interface,config):
    device=torch.device(backend.model.device)
    with torch.random.fork_rng(devices=[device.index or torch.cuda.current_device()] if device.type=='cuda' else []):
        torch.manual_seed(config['seed'])
        kv=ReactiveKVResidual(public_dim=2062,kv_heads=interface['kv_heads'],head_dim=interface['head_dim'],
            rank=config['rank'],layers=(-1,),scale=config['kv_scale']).to(backend.model.device).eval()
        action=CapacityActionResidual(public_dim=2062,width=64,joint_lower=interface['joint_lower'],
            joint_upper=interface['joint_upper'],arm_joint_bound=interface['arm_joint_bound'],
            gripper_bound=1.).to(backend.model.device).eval()
        if config['gripper_bound']!=0.:raise ValueError('Capacity action head must preserve raw native grippers')
        # Reuse the existing bounded head, with an explicit zero intervention
        # mask. No policy-level clamp repairs native pre-actuator overshoot.
        with torch.no_grad():action.bounds[[6,13]]=0.
    return dict(kv=kv,action=action)


def cache_demos(backend,manifest,paths,output,*,guard,seed,action_bounds):
    """Frozen actual prefix + official targets; full trajectories stay retained."""
    output=Path(output);output.mkdir(exist_ok=False)
    model=backend.model
    backend.kv_adapter=backend.action_residual=None
    backend._paired_instruction=None
    records=[]
    for episode in manifest['episodes']:
        guard();source=paths[episode['id']]
        if file_sha(source['path'])!=source['sha256']: raise ValueError('Whole demo bytes changed')
        state,targets=read_episode(source['path'],episode,manifest['schema'])
        for index,target,valid in whole_episode_chunks(targets):
            guard()
            raw=raw_observation(source['path'],episode,manifest['schema'],index,state)
            public=backend._encode(raw)
            observation,target32,roundtrip=transformed_target(model,raw,target)
            if not torch.equal(observation.state,backend._prepared.state):
                raise ValueError('Teacher input transform changed the actual normalized public state')
            # Actual H32 native proposal uses the same source input and fixed
            # noise as the action head's declared deployment convention.
            noise=np.random.default_rng(np.random.SeedSequence([seed,episode['canonical_index'],index])).standard_normal((32,32)).astype(np.float32)
            proposal=backend.decode(noise)
            error=np.abs(target-proposal);outside=np.maximum(error-action_bounds,0.)
            support=dict(valid_actions=int(valid.sum()),outside_counts=(outside[valid]>0).sum(0).tolist(),
                maximum_absolute_delta=error[valid].max(0).tolist(),
                attainable_MSE_floor=float((outside[valid]**2).mean()))
            data=cpu(dict(state=backend._prepared.state,features=torch.as_tensor(public,device=model.device)[None],
                cache=backend._kv_cache,mask=backend._prefix_mask,target32=target32,
                teacher14=torch.as_tensor(target,device=model.device)[None],
                base14=torch.as_tensor(proposal,device=model.device)[None],valid=torch.as_tensor(valid)[None]))
            path=output/f'sample-{len(records):04d}.pt'
            digest=atomic_save(path,dict(format='reactive_demo_cache_v1',episode=episode,index=index,
                source=source,roundtrip=roundtrip,data=data))
            records.append(dict(path=str(path),sha256=digest,split=episode['split'],episode_id=episode['id'],
                anchor=index,valid_actions=int(valid.sum()),roundtrip=roundtrip))
            records[-1]['physical_residual_support']=support
    return records


def batch_from_cache(records,indices,device,*,guard):
    rows=[]
    for index in indices:
        guard();ref=records[int(index)]
        if file_sha(ref['path'])!=ref['sha256']: raise ValueError('Cached frozen observation/target changed')
        saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
        rows.append(saved['data'])
    batch={key:torch.cat([r[key] for r in rows],0).to(device) for key in rows[0] if key!='cache'}
    batch['cache']=tuple(tuple(torch.cat([r['cache'][i][j] for r in rows],0).to(device)
                              for j in range(2)) for i in range(len(rows[0]['cache'])))
    return batch


def action_loss(head,batch):
    result=head(batch['base14'],batch['features'])
    return masked_mse(result.actions,batch['teacher14'],batch['valid'])


def fit_reactive_head(kind,head,backend,records,output,*,guard,config,contract):
    """One fixed seed/global-batch/update schedule; complete state at each update."""
    if kind not in ('kv','action'): raise ValueError('Unknown reactive head')
    output=Path(output);output.mkdir(exist_ok=False)
    started=time.monotonic()
    if backend.model.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    train=[i for i,r in enumerate(records) if r['split']=='train']
    validation=[i for i,r in enumerate(records) if r['split']=='validation']
    if not train or not validation: raise ValueError('No complete train/validation cache split')
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=0.)
    rng=np.random.default_rng(config['seed']+(0 if kind=='kv' else 1000))
    generator=torch.Generator(device=backend.model.device).manual_seed(config['seed']+2000)
    state=dict(format='robotwin_reactive_fit_v1',kind=kind,contract=contract,configuration=head.config(),
               config=deepcopy(config),records=deepcopy(records),iteration=0,best=None,metrics=[],
               parameters=sum(p.numel() for p in head.parameters()),
               fixed_training_monitor_indices=train[:4],
               interpretation='fixed small capacity fit; underfit/time-limited negatives cannot reject the full class')
    def persist():
        guard()
        return atomic_save(output/'latest.pt',dict(state=cpu(state),policy=cpu(head.state_dict()),
            optimizer=cpu(optimizer.state_dict()),sampler_numpy=deepcopy(rng.bit_generator.state),
            noise_torch=generator.get_state().cpu(),global_rng=rng_state()))
    def evaluate(indices):
        vrng=np.random.default_rng(config['seed']+3000)
        vgen=torch.Generator(device=backend.model.device).manual_seed(config['seed']+4000)
        totals={};weight=0.
        for index in indices:
            guard();batch=batch_from_cache(records,[index],backend.model.device,guard=guard)
            # KV primitive intentionally rejects no_grad: validation computes
            # the same differentiable suffix, detaches its scalar, never SGD.
            with torch.enable_grad():
                if kind=='kv':
                    loss,details=reactive_flow_loss(backend.model,head,batch,rng=vrng,generator=vgen,diagnostics=True)
                else:
                    prediction=head(batch['base14'],batch['features']).actions
                    loss=masked_mse(prediction,batch['teacher14'],batch['valid'])
                    details=dict(arm12_MSE=float(masked_mse(prediction[...,[0,1,2,3,4,5,7,8,9,10,11,12]],
                        batch['teacher14'][...,[0,1,2,3,4,5,7,8,9,10,11,12]],batch['valid']).detach()),
                        gripper2_MSE=float(masked_mse(prediction[...,[6,13]],batch['teacher14'][...,[6,13]],batch['valid']).detach()))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite reactive validation')
            n=int(batch['valid'].sum());weight+=n
            for key,value in dict(MSE=float(loss.detach()),**details).items():totals[key]=totals.get(key,0.)+value*n
            del loss,batch
        return {key:value/weight for key,value in totals.items()}
    initial=evaluate(validation)
    state['metrics'].append(dict(iteration=0,validation=initial,validation_loss=initial['MSE'],
                               training_monitor=evaluate(train[:4])))
    best_path=output/'best.pt'
    digest=atomic_save(best_path,dict(kind=kind,iteration=0,contract=contract,config=head.config(),policy=cpu(head.state_dict())))
    state['best']=dict(path=str(best_path),sha256=digest,iteration=0,validation_loss=initial['MSE'])
    persist()
    for update in range(1,config['updates']+1):
        guard();indices=rng.choice(train,size=config['global_batch'],replace=True)
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        # Execution microbatch1 preserves the declared global mean, including
        # exact valid-time weighting of short terminal chunks.
        denominator=sum(records[int(i)]['valid_actions'] for i in indices)
        for index in indices:
            batch=batch_from_cache(records,[index],backend.model.device,guard=guard)
            loss=(reactive_flow_loss(backend.model,head,batch,rng=rng,generator=generator)
                  if kind=='kv' else action_loss(head,batch))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite reactive training loss; preserve prior latest')
            (loss*int(batch['valid'].sum())/denominator).backward()
            losses.append(float(loss.detach())*int(batch['valid'].sum())/denominator)
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in backend.model.parameters()):
            raise RuntimeError('Frozen motor unexpectedly received gradients')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise RuntimeError('Nonfinite head step; preserve last complete finite recovery')
        backend._check_frozen()
        state['iteration']=update
        row=dict(iteration=update,loss=float(sum(losses)),gradient_norm=float(norm),
            elapsed_seconds=time.monotonic()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if backend.model.device.type=='cuda' else 0)
        if update%config['validate_every']==0 or update==config['updates']:
            diagnostics=evaluate(validation);score=diagnostics['MSE']
            row.update(validation_loss=score,validation=diagnostics,training_monitor=evaluate(train[:4]))
            if score<state['best']['validation_loss']:
                digest=atomic_save(best_path,dict(kind=kind,iteration=update,contract=contract,config=head.config(),policy=cpu(head.state_dict())))
                state['best']=dict(path=str(best_path),sha256=digest,iteration=update,validation_loss=score)
        state['metrics'].append(row);persist()
    state['complete']=True
    state['fit_diagnostics']=dict(initial=state['metrics'][0],final=state['metrics'][-1],
        physical_error_must_be_considered=True,full_class_rejection_authorized=False)
    persist()
    best=torch.load(best_path,map_location='cpu',weights_only=False)
    head.load_state_dict(best['policy']);head.eval()
    return state


class ReplayTeacherHead(ReactiveALOHAActionResidual):
    """Source command replay through the SAME decoded-action hook and recorder."""
    def __init__(self):
        torch.nn.Module.__init__(self);self.public_dim=2062
        self.targets=None;self.offset=0;self.reference=None
    def config(self):
        return dict(kind='qualified_source_demo_command_replay_v1',public_dim=2062,
                    reference=self.reference,tail='hold final recorded target until native termination/horizon')
    def forward(self,base,features):
        if self.targets is None: raise ValueError('No qualified complete teacher trajectory selected')
        indices=torch.arange(base.shape[1],device=base.device)+self.offset
        targets=self.targets.to(base.device)[indices.clamp(max=len(self.targets)-1)][None].to(base.dtype)
        delta=targets-base
        return DecodedActionResidual(base.detach(),delta,delta,torch.zeros_like(base,dtype=torch.bool),targets)


class ReactiveEpisodeControl:
    """Minimal protocol adapter to the existing complete-episode runner."""
    def __init__(self,env,heads,writer,interface):
        self.env,self.heads=env,heads;self.replay=ReplayTeacherHead().eval();self.mode='native'
        self.interface=deepcopy(interface)
        self.coefficients=np.zeros(4,np.float32);self.motor_rng=None;self.references={}
        self.kv,self.action=heads['kv'],heads['action']
        env.corpus_writer=env.backend.corpus_writer=writer
        writer.register_runtime(dict(env.contract,reactive_capacity_hooks={k:h.config() for k,h in heads.items()}))
    @property
    def native_horizon(self):return self.env.native_horizon
    @property
    def active(self):return self.env.active
    @property
    def elapsed_steps(self):return self.env.elapsed_steps
    @property
    def last_reset_info(self):return self.env.last_reset_info
    def episode_limit_for_task(self,task):return self.env.episode_limit_for_task(task)
    def set_recording_context(self,value):self.env.set_recording_context(value)
    def record_decision_metadata(self,value):
        reference=self.references.get(self.mode)
        self.env.record_decision_metadata(dict(value,capacity_head=self.mode,capacity_checkpoint=reference,
            optimizer_updates=0 if reference is None else reference['iteration'],
            interpretation='reactive capacity diagnosis; not RL/AD/DPT histories'))
    def set_coefficients(self,arm,values):
        if np.asarray(values).shape!=(4,) or np.any(values):raise ValueError('Common-runner placeholder coefficients must be zero')
        # The mode is selected only by the immutable capacity spec, never by
        # a learned4D mapping or an unrecorded action replacement.
        self.mode=arm
        backend=self.env.backend
        backend.kv_adapter=self.heads['kv'] if arm=='kv' else None
        backend.action_residual=(self.replay if arm=='action' and self.replay.targets is not None
                                 else self.heads['action'] if arm=='action' else None)
    def reset(self,task,seed,*,motor_seed,paired_instruction=None):
        self.motor_rng=np.random.default_rng(motor_seed)
        features=self.env.reset(task,seed,paired_instruction=paired_instruction)
        cache=self.env.backend._kv_cache
        if len(cache)!=self.interface['kv_layers'] or any(
                any(t.shape[-2:]!=(self.interface['kv_heads'],self.interface['head_dim']) for t in pair) for pair in cache):
            raise ValueError('Actual native KV cache differs from source receipt')
        return features
    def step(self,values):
        self.replay.offset=self.env.elapsed_steps
        noise=self.motor_rng.standard_normal((32,32)).astype(np.float32)
        return self.env.step(noise)
    def snapshot(self):
        return dict(mode=self.mode,elapsed_steps=self.elapsed_steps,
                    motor_rng=None if self.motor_rng is None else deepcopy(self.motor_rng.bit_generator.state))
    def close(self):self.env.close()
