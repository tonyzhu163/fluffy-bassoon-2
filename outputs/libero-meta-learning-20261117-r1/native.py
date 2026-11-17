"""LIBERO public inputs and frozen RLinf Pi0.5 inference for qualification."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUNTIME = ROOT/'sources/critic-selected-a1229239-r1'
RUNTIME_PIN = '8f9bfa7661928ef8c17494b04f4d7bf3125634e1'
PRO = ROOT/'sources/libero-pro2-e5500c69'
PRO_PIN = 'e5500c690c1498eca1b7050350654964190cdf70'
SETUP = ROOT/'setups/libero-meta-learning-20261117-r1'
MODEL = ROOT/'inputs/rlinf-pi05-libero40-sft-45ccfcc4-20261101-r1'
NORM = MODEL/'physical-intelligence/libero/norm_stats.json'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def public(raw, prompt):
    from openpi_client import image_tools
    from robosuite.utils.transform_utils import quat2axisangle
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(raw[k][::-1,::-1]),224,224))
        for k in ('agentview_image','robot0_eye_in_hand_image')]
    state = np.concatenate((raw['robot0_eef_pos'],
        quat2axisangle(raw['robot0_eef_quat_xyzw'].copy()),raw['robot0_gripper_qpos'])).astype(np.float32)
    assert state.shape == (8,) and np.isfinite(state).all()
    return dict(main_images=images[0][None],wrist_images=images[1][None],
        states=state[None],task_descriptions=[prompt])


class PrefixReady(Exception):
    pass


class Motor:
    def __init__(self):
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.openpi import get_model
        self.config = dict(model_type='openpi',pi05=True,precision='bf16',
            model_path=str(MODEL),num_action_chunks=10,action_dim=7,num_steps=5,
            add_value_head=False,openpi=dict(task='eval',config_name='pi05_libero',
                action_horizon=10,action_chunk=10,model_action_dim=32,action_env_dim=7,
                num_steps=5,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',
                max_token_len=200,discrete_state_input=False,torch_compile=False),
            openpi_data=dict(norm_stats_path=str(NORM)))
        self.model=get_model(OmegaConf.create(self.config),torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
        assert (self.model.action_horizon,self.model.action_env_dim,self.model.action_chunk)==(10,7,10)

    @torch.no_grad()
    def prepare(self, item):
        original=self.model.build_prefix_cache
        captured=[]
        def capture(observation,*args,**kwargs):
            captured.append((observation,original(observation,*args,**kwargs)))
            raise PrefixReady()
        self.model.build_prefix_cache=capture
        try:
            self.model.predict_action_batch(item,mode='eval',noise=torch.zeros(1,10,32,device='cuda'))
        except PrefixReady:
            pass
        finally:
            self.model.build_prefix_cache=original
        assert len(captured)==1
        observation,prefix=captured[0]
        hidden,mask,cache=prefix
        pooled=hidden.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
        pooled=pooled/pooled.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        features=torch.cat((observation.state[:,:8].float(),pooled),-1)
        assert features.shape==(1,2056) and torch.isfinite(features).all()
        return dict(public=item,observation=observation,prefix=prefix,features=features)

    @torch.no_grad()
    def actions(self, prepared, noise, *, policy=None, context=None):
        hidden,mask,cache=prepared['prefix']
        if policy is not None:
            cache=policy.kv_adapter(cache,mask,context)
        original=self.model.build_prefix_cache
        try:
            self.model.build_prefix_cache=lambda *a,**kw:(hidden,mask,cache)
            actions,_=self.model.predict_action_batch(prepared['public'],mode='eval',
                noise=torch.as_tensor(noise,device='cuda'))
        finally:
            self.model.build_prefix_cache=original
        result=actions.detach().float().cpu().numpy()[0]
        assert result.shape==(10,7) and np.isfinite(result).all()
        return result

    def targets(self, item, commands):
        # Invoke the checkpoint's real transform, including its quantile/
        # padding conventions. No RoboCasa twelve-coordinate codec is reused.
        transformed=self.model.input_transform(dict(
            **{'observation/image':item['main_images'],
               'observation/wrist_image':item['wrist_images'],
               'observation/state':item['states']},
            prompt=item['task_descriptions'],actions=np.asarray(commands)[None]))
        return transformed['actions'].to(device='cuda',dtype=torch.float32)


def make_adapter():
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeConfig, HistoryFlowSteeringPolicy
    # History retains10 executed7D controls, padded to the model's32D width.
    # Padding is always zero; raw controls and delivered-action masks are saved.
    cfg=FlowBridgeConfig(observation_dim=2056,history_action_dim=320,
        width=256,layers=6,heads=8,context_length=160,kv_heads=1,head_dim=256,
        kv_rank=4,kv_layers=tuple(range(18)),scale=1.,action_horizon=10,
        model_action_dim=32,active_action_dim=7)
    return HistoryFlowSteeringPolicy(cfg).to('cuda')


def qualify_gradient(motor, prepared, noise, output):
    """Numerical wiring check on one native query, not an adaptation result."""
    torch.manual_seed(171117)
    policy=make_adapter()
    token=torch.cat((prepared['features'],torch.zeros(1,322,device='cuda'),
        torch.ones(1,1,device='cuda')),-1)
    context,_=policy.encode_step(token)
    reference=motor.actions(prepared,noise)
    zero=motor.actions(prepared,noise,policy=policy,context=context)
    assert np.array_equal(reference,zero), 'Zero KV adapter changed native actions'
    target=motor.targets(prepared['public'],reference)
    assert target.shape==(1,10,32)
    versions=[p._version for p in motor.model.parameters()]
    optimizer=torch.optim.AdamW(policy.parameters(),lr=3e-4,weight_decay=0)
    rows=[]
    for step in range(3):
        started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        context,_=policy.encode_step(token)
        loss=policy.flow_matching_loss(motor.model,prepared['observation'],target,
            torch.randn_like(target),torch.tensor([.5],device='cuda'),
            prepared['prefix'][2],prepared['prefix'][1],context)
        loss.backward()
        def grad(module):
            values=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
            return float(torch.stack(values).sum().sqrt()) if values else 0.
        row=dict(step=step,loss=float(loss.detach()),decoder_gradient=grad(policy.core),
            adapter_gradient=grad(policy.kv_adapter))
        assert all(np.isfinite(v) for v in row.values()) and row['adapter_gradient']>0
        optimizer.step(); torch.cuda.synchronize()
        row['seconds']=time.monotonic()-started; rows.append(row)
    assert any(r['decoder_gradient']>0 for r in rows[1:])
    assert versions==[p._version for p in motor.model.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in motor.model.parameters())
    with torch.no_grad():
        context,_=policy.encode_step(token)
        updated=motor.actions(prepared,noise,policy=policy,context=context)
    result=dict(zero_adapter_exact=True,steps=rows,config=asdict(policy.config),
        decoded_action_change_rms=float(np.sqrt(np.mean((updated-reference)**2))),
        motor_parameters_unchanged=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        role='native-query gradient wiring check only; no teacher or history-benefit claim')
    write(output,result)
    return result
