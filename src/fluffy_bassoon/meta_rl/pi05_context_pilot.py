"""Bounded native-attention qualification and warm-start/trial-PPO pilot.

Run only on an allocated Linux GPU. The gate checks a real native observation,
base parity, history-conditioned gradients, tiny-set fitting and base integrity.
This execution pilot is not evidence of generalizable in-context RL.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from fluffy_bassoon.rl import maniskill_pi05 as ms
from fluffy_bassoon.rl.pi05_context_policy import NativeContextPi05, PublicHistory
from .bridge_distillation import corrected_action_label
from .checkpoint import rng_state
from .pi05_bridge_train import admit_gpu, feedback_batch, sample_trial_plan
from .warmstart import atomic_save


DATA_FORMAT = "native_pi05_prefix_inputs_history_v2"


def cpu(t):
    return t.detach().to('cpu', copy=True)


def materialize(record, device):
    return (SimpleNamespace(state=record['state'].to(device)),
            tuple(t.to(device) for t in record['prefix_inputs']))


def history_at(records, index, device, lesion='normal'):
    start = 0
    if lesion == 'memory_wiped':
        start = max(i for i in range(index+1) if i == 0 or bool(records[i]['feedback'][0,8]))
    values = torch.stack([r['feedback'] for r in records[start:index+1]], 1).to(device)
    if lesion == 'reward_blind':
        values = values.clone()
        values[...,7] = 0
    return values, torch.ones(values.shape[:2], dtype=torch.bool, device=device)


@torch.no_grad()
def collect_trial(policy, env, plan, generator, *, behavior='base', lesion='normal', labels=False):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
    from rlinf.envs.action_utils import prepare_actions
    motor = policy.motor
    attempts = len(plan['scenes'])
    records, rewards, successes = [], [], []
    history = PublicHistory(1,attempts*16,device='cuda')
    previous, feedback_reward = torch.zeros(1,7,device='cuda'), torch.zeros(1)
    for attempt, scene in enumerate(plan['scenes']):
        obs,_ = env.reset(seed=int(plan['reset_seeds'][attempt]),
                          options={'episode_id':torch.tensor([scene[0]['episode_id']],device=env.device)})
        if env.env.unwrapped.select_carrot_ids.cpu().tolist() != [scene[0]['object']]:
            raise RuntimeError('Scene reset differs from the private plan')
        success = torch.zeros(1,dtype=torch.bool)
        if lesion == 'memory_wiped':
            history.reset()
        for chunk in range(16):
            feedback = feedback_batch(previous,feedback_reward,attempt,chunk,attempts,
                                      reward_blind=lesion == 'reward_blind')
            history.append(feedback)
            values,mask = history.snapshot()
            observation = motor.env_obs_to_observation(obs)
            prepared = preprocess_observation(observation,train=False)
            inputs = policy.embed_observation(prepared)
            if behavior in ('base','oracle'):
                public,public_mask = values[:,:0],mask[:,:0]
            else:
                public,public_mask = values,mask
            action,stochastic = policy.sample_chunk(prepared,inputs,public,public_mask,generator=generator,
                                                   deterministic=behavior != 'stochastic')
            target = corrected_action_label(motor,observation.state,action,plan['theta_degrees']) if labels or behavior == 'oracle' else None
            if behavior == 'oracle':
                action = target
            command = prepare_actions(raw_chunk_actions=motor.decode_actions(action,observation.state),
                env_type='maniskill',model_type='openpi',num_action_chunks=5,action_dim=7,policy='widowx_bridge')
            if command.shape != (1,5,7) or not command.isfinite().all():
                raise RuntimeError('Invalid decoded native commands')
            issued = ms.rotate_translation(command,plan['theta_degrees'])
            observations,_,_,_,infos = env.chunk_step(issued)
            for info in infos:
                success |= info['success'].detach().cpu().bool()
            obs = observations[-1]
            reward = success.float() if chunk == 15 else torch.zeros(1)
            records.append(dict(prefix_inputs=tuple(cpu(t) for t in inputs), state=cpu(observation.state),
                feedback=cpu(feedback), requested_commands=cpu(command),
                target_actions=None if target is None else cpu(target), stochastic=stochastic))
            rewards.append(reward)
            previous = command[:,-1].detach()
            feedback_reward = reward
        successes.append(float(success[0]))
    return dict(format=DATA_FORMAT,records=records,rewards=torch.stack(rewards),successes=successes,
                simulator_steps=attempts*80,behavior=behavior,lesion=lesion)


def fit_queries(policy, optimizer, trials, sampler, generator, *, queries=2):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for _ in range(queries):
        trial = trials[int(sampler.integers(len(trials)))]
        index = int(sampler.integers(len(trial['records'])))
        record = trial['records'][index]
        prepared,inputs = materialize(record,'cuda')
        history,mask = history_at(trial['records'],index,'cuda')
        target = record['target_actions'].to('cuda')
        noise = torch.randn(target.shape,device='cuda',generator=generator)
        t = torch.rand((target.shape[0],),device='cuda',generator=generator)*.998+.001
        loss = policy.flow_loss(prepared,inputs,history,mask,target,noise,t)
        (loss/queries).backward()
        loss_sum += float(loss.detach())/queries
    norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(flow_loss=loss_sum,gradient_norm=float(norm))


@torch.no_grad()
def validation_loss(policy, trials):
    generator = torch.Generator(device='cuda').manual_seed(814)
    losses = []
    for trial in trials:
        for index in (0,len(trial['records'])//2,len(trial['records'])-1):
            record = trial['records'][index]
            prepared,inputs = materialize(record,'cuda')
            history,mask = history_at(trial['records'],index,'cuda')
            target = record['target_actions'].to('cuda')
            noise = torch.randn(target.shape,device='cuda',generator=generator)
            t = torch.rand((1,),device='cuda',generator=generator)*.998+.001
            losses.append(float(policy.flow_loss(prepared,inputs,history,mask,target,noise,t)))
    return float(np.mean(losses))


def ppo_update(policy,optimizer,trial,*,replay_tolerance=.002):
    # Exact full-trial Monte Carlo returns; no break at physical attempt resets.
    returns = trial['rewards'].flip(0).cumsum(0).flip(0).to('cuda')
    values = torch.stack([r['stochastic']['old_value'] for r in trial['records']]).to('cuda')
    advantages = returns-values
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
    optimizer.zero_grad(set_to_none=True)
    errors,losses = [],[]
    for index,record in enumerate(trial['records']):
        prepared,inputs = materialize(record,'cuda')
        history,mask = history_at(trial['records'],index,'cuda')
        log_prob,value = policy.replay_chunk(prepared,inputs,history,mask,record['stochastic'])
        old = record['stochastic']['old_log_prob'].to('cuda')
        error = float((log_prob.detach()-old).abs().max())
        errors.append(error)
        if error > replay_tolerance or not log_prob.isfinite().all():
            raise RuntimeError(f'Native context likelihood replay failed: {error}')
        ratio = (log_prob-old).exp()
        a = advantages[index]
        actor = -torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
        loss = actor+.01*(value-returns[index]).square().mean()
        (loss/len(trial['records'])).backward()
        losses.append(float(loss.detach()))
    norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(np.mean(losses)),gradient_norm=float(norm),max_replay_error=max(errors),
                optimizer_steps=1,policy_chunks=len(trial['records']))


def gate(policy,env,plan,generator,output,*,fit_steps=12):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
    motor = policy.motor
    observation,_ = env.reset(seed=int(plan['reset_seeds'][0]),options={
        'episode_id':torch.tensor([plan['scenes'][0][0]['episode_id']],device=env.device)})
    observation = motor.env_obs_to_observation(observation)
    prepared = preprocess_observation(observation,train=False)
    inputs = policy.embed_observation(prepared)
    initial = policy.trainable_state()
    frozen = policy.base_fingerprint()
    history = torch.zeros(1,2,11,device='cuda')
    history[:,1,7] = 1
    mask = torch.ones(1,2,dtype=torch.bool,device='cuda')
    noise = torch.randn(1,8,32,device='cuda',generator=generator)
    with torch.no_grad():
        native = motor.sample_actions(observation,num_steps=4,noise=noise)
        action,_ = policy.sample_chunk(prepared,inputs,history[:,:0],mask[:,:0],generator=generator,
                                      deterministic=True,initial_noise=noise)
    parity = float((native-action).abs().max())
    if not torch.equal(native,action):
        raise RuntimeError(f'Empty-history native parity failed: {parity}')
    target = corrected_action_label(motor,observation.state,native,[45.])
    time_sample = torch.full((1,),.5,device='cuda')
    fm_noise = torch.randn(target.shape,device='cuda',generator=generator)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=1e-4,weight_decay=0)
    losses=[]
    started=time.monotonic()
    for _ in range(fit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss=policy.flow_loss(prepared,inputs,history,mask,target,fm_noise,time_sample)
        loss.backward()
        grad=float(policy.feedback.weight.grad.norm())
        invalid=[name for name,p in policy.named_parameters() if p.grad is not None and not p.grad.isfinite().all()]
        write_json(output/'gate-gradient.json',dict(update=len(losses),loss=float(loss.detach()),
            feedback_gradient_norm=grad if np.isfinite(grad) else None,nonfinite_parameters=invalid))
        norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        final=float(policy.flow_loss(prepared,inputs,history,mask,target,fm_noise,time_sample))
    state_changed = any(not torch.equal(initial[n],p.cpu()) for n,p in policy.trainable_state().items())
    if not grad > 0 or not final < losses[0] or not state_changed or policy.base_fingerprint()!=frozen:
        raise RuntimeError('Native prefix gradient, fitting or frozen-base gate failed')
    policy.load_trainable_state(initial)
    _,record=policy.sample_chunk(prepared,inputs,history,mask,generator=generator)
    with torch.no_grad():
        log_prob,value=policy.replay_chunk(prepared,inputs,history,mask,record)
    replay=float((log_prob.cpu()-record['old_log_prob']).abs().max())
    if replay > .002:
        raise RuntimeError(f'Native stochastic replay gate failed: {replay}')
    result=dict(passed=True,empty_history_max_error=parity,feedback_gradient_norm=grad,
        fit_initial_loss=losses[0],fit_final_loss=final,fit_updates=fit_steps,fit_elapsed_seconds=time.monotonic()-started,
        max_replay_error=replay,frozen_base_sha256=frozen,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),config=policy.config(),
        interpretation='real-observation interface/gradient/likelihood qualification; no adaptation result')
    write_json(output/'gate.json',result)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-commit','expected-model-sha256'):
        p.add_argument('--'+name,required=True)
    for name in ('output-dir','rlinf-root','model-path','norm-stats'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allocation-probe',type=Path)
    p.add_argument('--expected-gpu-uuid')
    p.add_argument('--gate-only',action='store_true')
    p.add_argument('--rank',type=int,default=4)
    p.add_argument('--updates',type=int,default=20)
    p.add_argument('--gate-fit-steps',type=int,default=12)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--noise-scale',type=float,default=.05)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-elapsed-seconds',type=float,default=2700)
    p.add_argument('--resume',type=Path)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if sys.platform!='linux':
        raise ValueError('Native computation requires allocated Linux')
    if (min(args.updates,args.rank,args.threads,args.gate_fit_steps)<1 or not 0<=args.seed<2**32
            or any(not np.isfinite(v) or v<=0 for v in (args.learning_rate,args.noise_scale,args.max_elapsed_seconds))):
        raise ValueError('Invalid bounded pilot settings')
    root=Path(__file__).resolve().parents[3]
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if source!=args.source_commit or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Require clean exact reviewed source')
    if ms.sha256_file(args.model_path/'model.safetensors')!=args.expected_model_sha256:
        raise ValueError('Native motor checkpoint differs')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    env,policy=None,None
    started=time.monotonic()
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,stop)
    def budget():
        if stopped or time.monotonic()-started>args.max_elapsed_seconds:
            raise TimeoutError('Bounded pilot stopped at a complete phase/update boundary')
    try:
        admit_gpu(args,root,args.output_dir)
        sys.path.insert(0,str(args.rlinf_root))
        from rlinf.models import get_model
        from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
        import rlinf.envs.sim.maniskill.tasks  # noqa: F401
        torch.set_num_threads(args.threads)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        sampler=np.random.default_rng(args.seed)
        generator=torch.Generator(device='cuda').manual_seed(args.seed+1)
        env=ManiskillEnv(ms.env_config(args.rlinf_root,1,args.seed),1,seed_offset=0,
                         total_num_processes=1,worker_info=None,record_metrics=True)
        ms.record_renderer_binding(env,args.output_dir)
        motor=get_model(ms.model_config(args.model_path,args.norm_stats)).cuda().eval().requires_grad_(False)
        if (motor.action_horizon,motor.action_dim,motor.num_steps)!=(8,32,4):
            raise ValueError('Require qualified native 8x32 four-pass motor')
        policy=NativeContextPi05(motor,rank=args.rank,noise_scale=args.noise_scale)
        config=dict(source_commit=source,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            policy=policy.config(),norm_stats_sha256=ms.sha256_file(args.norm_stats),
            rlinf_sources={f:ms.sha256_file(args.rlinf_root/f'rlinf/models/embodiment/openpi/{f}')
                           for f in ('pi0.py','modules/gemma.py','env_io.py')})
        write_json(args.output_dir/'config.json',config)
        infos=env.env.unwrapped.basic_obj_infos
        plan=lambda evaluation=False:sample_trial_plan(infos,[0,1],4,1,sampler,evaluation=evaluation)
        result=gate(policy,env,plan(),generator,args.output_dir,fit_steps=args.gate_fit_steps)
        print(json.dumps(dict(stage='gate',**result)),flush=True)
        if args.gate_only:
            write_json(args.output_dir/'summary.json',dict(status='qualified',gate=result))
            return
        frozen=result['frozen_base_sha256']
        optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0)
        trial_paths=[]
        complete=0
        ppo_done=False
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu',weights_only=False)
            if saved['source_commit']!=source or saved['policy_config']!=policy.config():
                raise ValueError('Resume source/policy differs')
            policy.load_trainable_state(saved['weights'])
            optimizer.load_state_dict(saved['optimizer'])
            trial_paths=[Path(p) for p in saved['trial_paths']]
            for p,digest in zip(trial_paths,saved['trial_hashes'],strict=True):
                if ms.sha256_file(p)!=digest: raise ValueError('Resume history differs')
            sampler.bit_generator.state=saved['sampler']
            generator.set_state(saved['generator'])
            from .checkpoint import restore_rng
            restore_rng(saved['rng'])
            complete=saved['complete_updates']; ppo_done=saved['ppo_done']
        def checkpoint():
            atomic_save(args.output_dir/'latest.pt',dict(source_commit=source,policy_config=policy.config(),
                config=config,weights=policy.trainable_state(),optimizer=optimizer.state_dict(),
                sampler=sampler.bit_generator.state,generator=generator.get_state(),rng=rng_state(),
                trial_paths=list(map(str,trial_paths)),trial_hashes=[ms.sha256_file(p) for p in trial_paths],
                complete_updates=complete,ppo_done=ppo_done))
        if not trial_paths:
            for index,theta in enumerate((-45.,0.,45.,-30.,30.)):
                budget()
                private=plan(evaluation=index>=3); private['theta_degrees']=[theta]
                paired_generator=torch.Generator(device='cuda').set_state(generator.get_state()) if index==2 else None
                trial=collect_trial(policy,env,private,generator,labels=True)
                path=args.output_dir/f'history-{index}.pt'
                atomic_save(path,trial); trial_paths.append(path)
                write_json(args.output_dir/f'private-trial-{index}.json',private)
                print(json.dumps(dict(stage='collection',index=index,theta=theta,successes=trial['successes'])),flush=True)
                if index==2:
                    budget()
                    oracle=collect_trial(policy,env,private,paired_generator,behavior='oracle')
                    baseline=float(np.mean(trial['successes']))
                    oracle_rate=float(np.mean(oracle['successes']))
                    competence=dict(passed=oracle_rate>=.25 and oracle_rate>=baseline,
                                    baseline=baseline,oracle=oracle_rate,theta=theta,trials=1,attempts=4,
                                    interpretation='paired execution gate; not statistical evidence')
                    write_json(args.output_dir/'label-competence.json',competence)
                    if not competence['passed']:
                        write_json(args.output_dir/'summary.json',dict(status='label_competence_failed',gate=result,competence=competence))
                        return
            checkpoint()
        trials=[torch.load(p,map_location='cpu',weights_only=False) for p in trial_paths]
        initial_validation=validation_loss(policy,trials[3:])
        while complete<args.updates:
            budget()
            metrics=fit_queries(policy,optimizer,trials[:3],sampler,generator)
            complete+=1
            checkpoint()
            if complete%5==0:
                print(json.dumps(dict(stage='warmstart',update=complete,**metrics)),flush=True)
        final_validation=validation_loss(policy,trials[3:])
        if not ppo_done:
            budget()
            rollout=collect_trial(policy,env,plan(),generator,behavior='stochastic')
            atomic_save(args.output_dir/'ppo-rollout.pt',rollout)
            metrics=ppo_update(policy,optimizer,rollout)
            ppo_done=True
            checkpoint()
            write_json(args.output_dir/'ppo-smoke.json',metrics)
            print(json.dumps(dict(stage='ppo_smoke',**metrics)),flush=True)
        evaluation={}
        for lesion in ('normal','memory_wiped','reward_blind'):
            results=[]
            for index,theta in enumerate((-30.,30.)):
                budget()
                local=np.random.default_rng(904+index)
                private=sample_trial_plan(infos,[0,1],4,1,local,evaluation=True)
                private['theta_degrees']=[theta]
                eval_generator=torch.Generator(device='cuda').manual_seed(1904+index)
                trial=collect_trial(policy,env,private,eval_generator,behavior='mean',lesion=lesion)
                results.append(trial['successes'])
                write_json(args.output_dir/f'evaluation-{lesion}-{index}.json',dict(successes=trial['successes'],theta=theta))
            evaluation[lesion]=results
        if policy.base_fingerprint()!=frozen:
            raise RuntimeError('Frozen native base changed')
        write_json(args.output_dir/'summary.json',dict(status='completed',warmstart_updates=complete,
            initial_validation_flow_loss=initial_validation,final_validation_flow_loss=final_validation,
            ppo_smoke=ppo_done,evaluation=evaluation,elapsed_seconds=time.monotonic()-started,
            interpretation='one-seed bounded execution pilot; not a generalization or architecture result'))
    except BaseException as exc:
        write_json(args.output_dir/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if env is not None: env.close()


if __name__=='__main__':
    main()
