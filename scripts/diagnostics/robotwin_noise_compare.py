#!/usr/bin/env python3
"""Allocated-Linux experiment: frozen RoboTwin motor, GTrXL/PPO and noise SAC.

Source learners receive the same *actual primitive-step* ceiling. Source
collection can cut its final trial/chunk at that ceiling; PPO bootstraps the
external cut. Target trials have equal episode/horizon allowances and record
actual costs. Tiny source-subset pilots are qualification, never held-out
generalization results. No checkpoint is downloaded by this command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO, observation_hash, capture_rng, restore_rng
from fluffy_bassoon.meta_rl.vla_noise_sac import SACConfig, NoiseSAC


SOURCE_TASKS = ("handover_mic", "pick_diverse_bottles", "place_a2b_left",
                "place_object_stand", "place_phone_stand", "scan_object")
HELDOUT_TASKS = ("adjust_bottle", "move_stapler_pad", "place_a2b_right",
                 "place_dual_shoes", "place_empty_cup", "stamp_seal")


class WallBudgetReached(RuntimeError):
    pass


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def append_json(path,value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value,allow_nan=False)+"\n")


class BalancedSchedule:
    """Uniform task blocks; actual task interactions are reported separately."""
    def __init__(self,tasks,seed):
        self.tasks=tuple(tasks); self.rng=np.random.default_rng(seed)
        self.pending=[]; self.trials=0

    def sample(self,episodes):
        if not self.pending:
            self.pending=self.rng.permutation(self.tasks).tolist()
        task=self.pending.pop()
        seeds=self.rng.integers(0,2**31-1,size=episodes).tolist()
        self.trials+=1
        return task,seeds

    def state_dict(self):
        return dict(tasks=self.tasks,rng=self.rng.bit_generator.state,pending=self.pending,trials=self.trials)

    def load_state_dict(self,state):
        if tuple(state["tasks"])!=self.tasks: raise ValueError("Task schedule changed")
        self.rng.bit_generator.state=state["rng"]; self.pending=list(state["pending"])
        self.trials=int(state["trials"])


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit",required=True)
    p.add_argument("--model-path",type=Path,required=True)
    p.add_argument("--norm-stats",type=Path,required=True)
    p.add_argument("--robotwin-root",type=Path,required=True)
    p.add_argument("--rlinf-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source-tasks",nargs="+",default=list(SOURCE_TASKS))
    p.add_argument("--eval-tasks",nargs="+",default=list(HELDOUT_TASKS))
    p.add_argument("--methods",nargs="+",choices=("ppo","sac","ppo_reset"),default=["ppo","sac"])
    p.add_argument("--source-steps",type=int,default=10_240_000)
    p.add_argument("--ppo-update-steps",type=int,default=0)
    p.add_argument("--ppo-trials-per-update",type=int,default=256)
    p.add_argument("--checkpoint-steps",type=int,default=20_000)
    p.add_argument("--train-episodes",type=int,default=4)
    p.add_argument("--eval-episodes",type=int,default=8)
    p.add_argument("--episode-limit",type=int,default=400)
    p.add_argument("--native-horizon",type=int,default=32)
    p.add_argument("--task-config-name",choices=("demo_clean","demo_randomized"),default="demo_clean")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--eval-trials",type=int,default=1)
    p.add_argument("--native-episodes",type=int,default=2)
    p.add_argument("--ppo-epochs",type=int,default=5)
    p.add_argument("--ppo-minibatch-trials",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--sac-warmup",type=int,default=500)
    p.add_argument("--sac-batch-size",type=int,default=256)
    p.add_argument("--sac-utd",type=int,default=20)
    p.add_argument("--action-limit",type=float,default=1.5)
    p.add_argument("--wall-seconds",type=float,default=2400)
    p.add_argument("--boundary-reserve-seconds",type=float,default=180)
    p.add_argument("--qualification-smoke",action="store_true")
    p.add_argument("--resume",type=Path)
    args=p.parse_args(argv)
    if (any(getattr(args,k)<1 for k in ("source_steps","ppo_trials_per_update","checkpoint_steps","train_episodes",
        "eval_episodes","episode_limit","native_horizon","eval_trials","ppo_epochs",
        "ppo_minibatch_trials","sac_batch_size","sac_utd")) or args.native_episodes<0
        or args.sac_warmup<0 or args.ppo_update_steps<0 or args.wall_seconds<=args.boundary_reserve_seconds):
        p.error("Invalid budget or batch settings")
    if not set(args.source_tasks)<=set(SOURCE_TASKS):
        p.error("Source tasks must remain inside the six VLRLA source tasks")
    if not args.qualification_smoke and set(args.source_tasks)!=set(SOURCE_TASKS):
        p.error("A source subset must be labeled --qualification-smoke")
    if not set(args.eval_tasks)<=set(SOURCE_TASKS+HELDOUT_TASKS):
        p.error("Unknown evaluation task")
    if len(set(args.methods))!=len(args.methods): p.error("Duplicate method")
    return args


def main(argv=None):
    args=parse_args(argv)
    if sys.platform!="linux" or not torch.cuda.is_available():
        raise RuntimeError("Run only inside a qualified allocated Linux CUDA job")
    if (args.output/"contract.json").exists():
        raise FileExistsError("Use a fresh output identity, including for resumed allocations")
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv
    env=RobotwinNoiseEnv(model_path=args.model_path,norm_stats=args.norm_stats,
                        robotwin_root=args.robotwin_root,rlinf_root=args.rlinf_root,
                        device="cuda",episode_limit=args.episode_limit,
                        action_horizon=args.native_horizon,output_dir=args.output/"runtime",
                        task_config_name=args.task_config_name)
    observation_dim=int(env.observation_dim); noise_shape=tuple(env.noise_shape)
    contract=dict(source_commit=args.source_commit,source_tasks=args.source_tasks,
                  eval_tasks=args.eval_tasks,methods=args.methods,source_steps_per_learner=args.source_steps,
                  train_episodes=args.train_episodes,eval_episodes=args.eval_episodes,
                  reported_episode_indices=[x for x in (1,2,4,8) if x<=args.eval_episodes],
                  episode_limit=args.episode_limit,noise_shape=noise_shape,
                  observation_dim=observation_dim,native_horizon=env.native_horizon,
                  task_config_name=args.task_config_name,
                  action_limit=args.action_limit,source_sampling="uniform shuffled task blocks",
                  qualification_smoke=args.qualification_smoke,
                  scope="source-subset qualification; no generalization claim" if args.qualification_smoke
                  else "six-task source pretraining / declared target evaluation",
                  source_budget="exact executed primitive actions; external final collection cuts recorded",
                  target_budget="equal episode and horizon allowance; actual interactions reported",
                  clock="episode index clamped at training maximum during longer evaluation",
                  source_checkpoint_transfer="complete source learner independently restored; online target replay is fresh with replacement sampling from first target row",
                  model_path=str(args.model_path),norm_stats=str(args.norm_stats))
    contract["ppo"] = asdict(PPOConfig(learning_rate=args.learning_rate,epochs=args.ppo_epochs,
        minibatch_trials=args.ppo_minibatch_trials,train_episodes=args.train_episodes,
        episode_limit=args.episode_limit,action_limit=args.action_limit))
    contract["ppo_trials_per_update"] = args.ppo_trials_per_update
    contract["ppo_update_primitive_threshold"] = args.ppo_update_steps
    contract["sac"] = asdict(SACConfig(warmup_chunks=args.sac_warmup,
        batch_size=args.sac_batch_size,utd=args.sac_utd))
    if hasattr(env,"contract"):
        value=env.contract
        runtime=value() if callable(value) else dict(value)
        # GPU identity is re-attested in every allocation. A physical PCI
        # address is evidence, not part of the resumable learning contract.
        write_json(args.output/"runtime-allocation.json",runtime)
        contract["runtime"]={k:v for k,v in runtime.items() if k!="actual_render_pci"}
    write_json(args.output/"contract.json",contract)
    episodes_path=args.output/"episodes.jsonl"
    updates_path=args.output/"updates.jsonl"
    history=[]; trained={}; summaries={}
    def log(stage,**values):
        row=dict(stage=stage,elapsed_seconds=time.monotonic()-started,**values)
        print(json.dumps(row,allow_nan=False),flush=True)
        append_json(args.output/"progress.jsonl",row)
    def guard():
        if time.monotonic()-started>=args.wall_seconds-args.boundary_reserve_seconds:
            raise WallBudgetReached("Stopped before a new trial to preserve checkpoint reserve")
    def record_episode(method,phase,**metadata):
        def callback(row):
            append_json(episodes_path,dict(method=method,phase=phase,**metadata,**row))
        return callback
    def ppo_config(reset=False):
        return PPOConfig(learning_rate=args.learning_rate,epochs=args.ppo_epochs,
                         minibatch_trials=args.ppo_minibatch_trials,
                         train_episodes=args.train_episodes,episode_limit=args.episode_limit,
                         action_limit=args.action_limit,reset_each_episode=reset)
    def new_ppo(reset=False):
        return RecurrentNoisePPO(observation_dim,noise_shape,config=ppo_config(reset),device="cuda",seed=args.seed)
    def new_sac():
        return NoiseSAC(observation_dim,math.prod(noise_shape),device="cuda",seed=args.seed,
                        action_limit=args.action_limit,config=SACConfig(
                            warmup_chunks=args.sac_warmup,batch_size=args.sac_batch_size,
                            utd=args.sac_utd))
    def checkpoint(method,learner,schedule,steps,task_steps,pending=None):
        path=args.output/(method+"-recovery.pt")
        temporary=path.with_suffix(".tmp")
        torch.save(dict(contract=contract,method=method,learner=learner.state_dict(),
                        schedule=schedule.state_dict(),source_steps=steps,task_steps=task_steps,
                        pending_rollouts=pending or [],boundary="complete trial or declared budget cut",
                        resume_stage="source",elapsed_seconds=time.monotonic()-started),temporary)
        temporary.replace(path)
    def native(phase):
        rng=np.random.default_rng(args.seed+907)
        for task in args.eval_tasks:
            for index in range(args.native_episodes):
                guard(); features=env.reset(task,args.seed+900_000+index)
                steps=0; chunks=0; success=False; reward_sum=0.; done=False
                initial_hash=observation_hash(features)
                while not done:
                    _,reward,end,trunc,info=env.step(rng.normal(size=math.prod(noise_shape)).astype(np.float32))
                    steps+=int(info["primitive_steps"]); chunks+=1
                    success|=bool(info.get("success",False)); reward_sum+=float(reward); done=bool(end or trunc)
                row=dict(method="native",phase=phase,task=task,episode=index+1,
                         seed=args.seed+900_000+index,success=success,return_=reward_sum,
                         primitive_steps=steps,chunks=chunks,initial_observation_sha256=initial_hash)
                append_json(episodes_path,row); log("native_episode",**row)
    def sac_trial(learner,task,seeds,*,phase,source=False,online=False,primitive_budget=None,method="sac"):
        total=0; rows=[]
        for episode,seed in enumerate(seeds):
            observation=np.asarray(env.reset(task,int(seed)),np.float32)
            initial_hash=observation_hash(observation); done=False; steps=chunks=0; success=False; reward_sum=0.
            while not done:
                action=(learner.sample_warmup(observation) if source and learner.warming_up
                        else learner.act(observation,deterministic=False))
                if primitive_budget is None:
                    following,reward,end,trunc,info=env.step(action)
                else:
                    following,reward,end,trunc,info=env.step(action,max_primitive_steps=primitive_budget-total)
                count=int(info["primitive_steps"])
                if count<1 or count>env.native_horizon:
                    raise RuntimeError("Invalid exact primitive-step increment")
                steps+=count; total+=count; chunks+=1
                if primitive_budget is not None and total>primitive_budget:
                    raise RuntimeError("Source primitive budget exceeded")
                success|=bool(info.get("success",False)); reward_sum+=float(reward)
                budget_end=primitive_budget is not None and total>=primitive_budget
                if source or online:
                    learner.observe(observation,action,float(reward),following,bool(end),bool(trunc),primitive_steps=count)
                    metrics=learner.update(gradient_steps=args.sac_utd)
                    if metrics and learner.environment_steps%25==0:
                        append_json(updates_path,dict(method=method,phase=phase,**metrics))
                        log("sac_updates",method=method,phase=phase,source_primitive_steps=learner.primitive_steps,**metrics)
                observation=np.asarray(following,np.float32); done=bool(end or trunc or budget_end)
            row=dict(task=task,episode=episode+1,seed=int(seed),success=success,return_=reward_sum,
                     primitive_steps=steps,chunks=chunks,initial_observation_sha256=initial_hash,
                     budget_truncated=bool(budget_end))
            append_json(episodes_path,dict(method=method,phase=phase,**row)); rows.append(row)
            if primitive_budget is not None and total>=primitive_budget: break
        return dict(episodes=rows,primitive_steps=total)
    source_states={}
    accepted_resume_steps={}
    next_method_index=0
    last_joint_steps=0
    resume_joint=None
    if args.resume is not None:
        resume_path=(args.resume/"coordinator-recovery.pt" if args.resume.is_dir() else args.resume)
        resume_joint=torch.load(resume_path,map_location="cpu",weights_only=False)
        if resume_joint["contract"]!=contract:
            raise ValueError("Resume scientific/source contract changed")
    def initialize_sources():
        nonlocal next_method_index,last_joint_steps
        for method in args.methods:
            learner=new_sac() if method=="sac" else new_ppo(method=="ppo_reset")
            schedule=BalancedSchedule(args.source_tasks,args.seed+101)
            state=dict(learner=learner,schedule=schedule,steps=0,
                       task_steps={task:0 for task in args.source_tasks},pending=[],last_checkpoint_steps=0)
            if resume_joint is not None:
                saved=resume_joint["states"][method]
                learner.load_state_dict(saved["learner"])
                schedule.load_state_dict(saved["schedule"])
                state.update(steps=int(saved["source_steps"]),task_steps=saved["task_steps"],
                             pending=saved.get("pending_rollouts",[]),
                             last_checkpoint_steps=int(saved["source_steps"]))
            source_states[method]=state
            accepted_resume_steps[method]=state["steps"]
        if resume_joint is not None:
            next_method_index=int(resume_joint["next_method_index"])
            last_joint_steps=sum(s["steps"] for s in source_states.values())
            # Per-learner loading may restore older global RNG snapshots; the
            # coordinator is the final authority after every learner is loaded.
            restore_rng(resume_joint["rng"])
            log("resume_accepted",checkpoint=str(args.resume),source_steps=accepted_resume_steps,
                next_method_index=next_method_index,
                note="Only checkpoint counters are resumed; later uncheckpointed predecessor logs are excluded")
    def save_joint(force=False):
        nonlocal last_joint_steps
        total=sum(s["steps"] for s in source_states.values())
        if not force and total-last_joint_steps<args.checkpoint_steps: return
        states={}
        for method,state in source_states.items():
            states[method]=dict(learner=state["learner"].state_dict(),
                schedule=state["schedule"].state_dict(),source_steps=state["steps"],
                task_steps=state["task_steps"],pending_rollouts=state["pending"])
        path=args.output/"coordinator-recovery.pt"
        temporary=path.with_suffix(".tmp")
        torch.save(dict(contract=contract,states=states,rng=capture_rng(),
                        next_method_index=next_method_index,boundary="completed source trial/update",
                        elapsed_seconds=time.monotonic()-started),temporary)
        temporary.replace(path); last_joint_steps=total
    def save_state(method,force=False):
        state=source_states[method]
        if force or state["steps"]-state["last_checkpoint_steps"]>=args.checkpoint_steps:
            checkpoint(method,state["learner"],state["schedule"],state["steps"],
                       state["task_steps"],state["pending"])
            state["last_checkpoint_steps"]=state["steps"]
    def refresh_summaries():
        for method,state in source_states.items():
            learner=state["learner"]
            summaries[method]=dict(source_steps=state["steps"],task_steps=state["task_steps"],
                                   allocation_source_steps=state["steps"]-accepted_resume_steps[method],
                                   updates=learner.updates,optimizer_steps=learner.optimizer_steps,
                                   pending_trials=len(state["pending"]),
                                   pending_primitive_steps=sum(r["primitive_steps"] for _,r in state["pending"]))
            if method=="sac": summaries[method]["contract"]=learner.contract()
    def maybe_ppo_update(method):
        state=source_states[method]; pending=state["pending"]
        if not pending: return
        pending_steps=sum(r["primitive_steps"] for _,r in pending)
        reached=(pending_steps>=args.ppo_update_steps if args.ppo_update_steps
                 else len(pending)>=args.ppo_trials_per_update)
        if not reached and state["steps"]<args.source_steps: return
        log("ppo_update_start",method=method,source_steps=state["steps"],
            trials=len(pending),primitive_steps=pending_steps,
            chunks=sum(r["chunks"] for _,r in pending))
        metrics=state["learner"].update([r for r,_ in pending]); state["pending"]=[]
        append_json(updates_path,dict(method=method,source_steps=state["steps"],**metrics))
        save_state(method,force=True)
        log("ppo_update",method=method,source_steps=state["steps"],**metrics)
    def source_round(method):
        state=source_states[method]; learner=state["learner"]
        if method!="sac": maybe_ppo_update(method)
        if state["steps"]>=args.source_steps: return
        guard(); task,seeds=state["schedule"].sample(args.train_episodes)
        trial_start=time.monotonic()
        if method=="sac":
            result=sac_trial(learner,task,seeds,phase="source",source=True,
                             primitive_budget=args.source_steps-state["steps"])
            errors={}
        else:
            rollout,result=learner.collect_trial(env,task,seeds,
                primitive_budget=args.source_steps-state["steps"],
                episode_callback=record_episode(method,"source"))
            errors=learner.replay_error(rollout)
            if errors["log_prob_max_error"]>2e-4 or errors["value_max_error"]>1e-5:
                raise RuntimeError(f"Old-policy replay failed: {errors}")
            state["pending"].append((rollout,result))
        state["steps"]+=result["primitive_steps"]
        state["task_steps"][task]+=result["primitive_steps"]
        if method!="sac": learner.primitive_steps=state["steps"]
        save_state(method)
        log("source_trial",method=method,source_steps=state["steps"],task=task,
            episodes=len(result["episodes"]),trial_seconds=time.monotonic()-trial_start,
            **errors)
        if method!="sac": maybe_ppo_update(method)
        refresh_summaries()
    status="complete"
    try:
        log("runtime_ready",observation_dim=observation_dim,noise_shape=noise_shape)
        # Preflight one declared source task; held-out/native comparison is
        # deferred until source training is complete, avoiding first-allocation
        # benchmark cost before learning starts.
        preflight_task=args.source_tasks[0]
        env.reset(preflight_task,args.seed+777_000)
        parity=env.check_noise_parity(np.zeros(math.prod(noise_shape),np.float32))
        log("motor_noise_parity",task=preflight_task,**parity)
        write_json(args.output/"motor-parity.json",parity)
        frozen_before=env.frozen_fingerprint()
        initialize_sources()
        for method in args.methods: save_state(method,force=True)
        # Interleave complete K4 trials so both learning arms advance in the
        # first allocation. Pending PPO trials retain sampled old-policy data.
        save_joint(force=True)
        while any(s["steps"]<args.source_steps for s in source_states.values()):
            method=args.methods[next_method_index]
            previous_updates=source_states[method]["learner"].updates
            source_round(method)
            next_method_index=(next_method_index+1)%len(args.methods)
            save_joint(force=method!="sac" and source_states[method]["learner"].updates>previous_updates)
        for method,state in source_states.items():
            if method!="sac": maybe_ppo_update(method)
            save_state(method,force=True)
            trained[method]=deepcopy(state["learner"].state_dict())
        refresh_summaries()
        if len({summaries[m]["source_steps"] for m in args.methods})!=1:
            raise RuntimeError("Source interaction budgets do not match")
        native("native_before_target")
        for task in args.eval_tasks:
            for repetition in range(args.eval_trials):
                seeds=[args.seed+2_000_000+repetition*100_000+i*997 for i in range(args.eval_episodes)]
                for method in args.methods:
                    if method=="sac":
                        for online in (False,True):
                            guard(); learner=new_sac(); learner.load_state_dict(deepcopy(trained["sac"]))
                            label="sac_online" if online else "sac_frozen"
                            convention=(learner.begin_target_adaptation(replay_mode="fresh_target")
                                        if online else None)
                            result=sac_trial(learner,task,seeds,phase="target",online=online,method=label)
                            history.append(dict(method=label,task=task,repetition=repetition,
                                target_adaptation=convention,
                                target_update_costs=learner.target_counters() if online else None,**result))
                            log("target_trial",method=label,task=task,successes=[r["success"] for r in result["episodes"]])
                    else:
                        for lesion in ("normal","memory_reset","reward_blind"):
                            guard(); learner=new_ppo(method=="ppo_reset")
                            learner.load_state_dict(deepcopy(trained[method]))
                            _,result=learner.collect_trial(env,task,seeds,
                                reset_memory=method=="ppo_reset" or lesion=="memory_reset",
                                reward_blind=lesion=="reward_blind",
                                episode_callback=record_episode(method,"target",lesion=lesion,repetition=repetition))
                            history.append(dict(method=method,lesion=lesion,repetition=repetition,**result))
                            log("target_trial",method=method,lesion=lesion,task=task,
                                successes=[r["success"] for r in result["episodes"]])
        native("native_after")
    except WallBudgetReached as exc:
        status="wall_budget_stop"; log(status,reason=str(exc))
    except Exception as exc:
        status="failed"; log(status,error_type=type(exc).__name__,reason=str(exc))
        raise
    finally:
        if source_states: save_joint(force=True)
        for method in source_states: save_state(method,force=True)
        refresh_summaries()
        if "frozen_before" in locals():
            frozen_after=env.frozen_fingerprint()
            write_json(args.output/"frozen-motor.json",dict(before=frozen_before,after=frozen_after,
                unchanged=frozen_before==frozen_after))
            if frozen_before!=frozen_after:
                status="frozen_motor_changed"
        summary=dict(status=status,elapsed_seconds=time.monotonic()-started,
                     source=summaries,target=history,contract=contract,
                     source_budget_match=len(summaries)==len(args.methods) and
                     all(x["source_steps"]==args.source_steps for x in summaries.values()))
        if hasattr(env,"counters"):
            value=env.counters
            summary["runtime_counters"]=value() if callable(value) else value
        write_json(args.output/"summary.json",summary)
        if hasattr(env,"close"): env.close()
    return 0 if status in ("complete","wall_budget_stop") else 3


if __name__=="__main__":
    raise SystemExit(main())
