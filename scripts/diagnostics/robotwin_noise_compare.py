#!/usr/bin/env python3
"""Allocated-Linux experiment: frozen RoboTwin motor, GTrXL/PPO and noise SAC.

Source learners receive the same *actual primitive-step* ceiling. Source
collection can cut its final trial/chunk at that ceiling; PPO bootstraps the
external cut. Optional fixed-iteration training instead matches complete K4
trial counts, reserves full task-native horizons, and reports differing actual
action totals without cutting episodes to meet an action target.
Target trials have equal episode/horizon allowances and record
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
import os
from pathlib import Path
import random
import sys
import time
import weakref

import numpy as np
import torch

from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO, observation_hash, capture_rng, restore_rng
from fluffy_bassoon.meta_rl.vla_noise_sac import SACConfig, NoiseSAC
from fluffy_bassoon.meta_rl.collection_admission import (
    admit_collected_group, assert_resume_allowed, quarantine_collection_failure,
    campaign_checkpoint_fence, assert_campaign_checkpoint_fence,
)


SOURCE_TASKS = ("handover_mic", "pick_diverse_bottles", "place_a2b_left",
                "place_object_stand", "place_phone_stand", "scan_object")
HELDOUT_TASKS = ("adjust_bottle", "move_stapler_pad", "place_a2b_right",
                 "place_dual_shoes", "place_empty_cup", "stamp_seal")


class WallBudgetReached(RuntimeError):
    pass


class CampaignBudgetReached(RuntimeError):
    pass


class CollectorTimeAdmission:
    """Reviewed first-group bound, then measured complete-group wall times."""
    def __init__(self, initial_seconds, checkpoint_reserve):
        if (not math.isfinite(initial_seconds) or initial_seconds <= 0
                or not math.isfinite(checkpoint_reserve) or checkpoint_reserve < 0):
            raise ValueError("Parallel collection needs a finite reviewed group-time bound")
        self.initial_seconds=float(initial_seconds)
        self.checkpoint_reserve=float(checkpoint_reserve)
        self.maximum_observed=0.

    def require(self, remaining_seconds):
        required=max(self.initial_seconds,1.25*self.maximum_observed)+self.checkpoint_reserve
        if not math.isfinite(remaining_seconds) or remaining_seconds <= required:
            raise WallBudgetReached(f"Complete parallel group plus checkpoint reserve requires {required:.3f}s")

    def observe(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid measured complete-group wall time")
        self.maximum_observed=max(self.maximum_observed,float(seconds))

    def state_dict(self):
        return dict(initial_seconds=self.initial_seconds,checkpoint_reserve=self.checkpoint_reserve,
                    maximum_observed=self.maximum_observed)

    def load_state_dict(self, state):
        if (state["initial_seconds"]!=self.initial_seconds
                or state["checkpoint_reserve"]!=self.checkpoint_reserve
                or not math.isfinite(state["maximum_observed"]) or state["maximum_observed"]<0):
            raise ValueError("Parallel collection timing recovery changed")
        self.maximum_observed=float(state["maximum_observed"])


def restore_unstarted_source_group(state, plan):
    """Only for groups with zero simulator RPCs; never refund physical work."""
    state["schedule"].load_state_dict(deepcopy(plan["schedule_before"]))
    state["learner"].sampler.bit_generator.state=deepcopy(plan["sampler_before"])
    restore_rng(plan["rng_before"])


def plan_parallel_source_group(state, *, lanes, episodes, episode_limit_for_task,
                               remaining_actions, admit_time):
    """Plan within one six-trial round, rolling back every prelaunch stream."""
    if lanes not in (2,3,6) or episodes!=4 or len(state["pending"])%lanes:
        raise ValueError("Parallel groups require aligned two/three/six-lane K4 execution")
    if len(state["pending"])+lanes>6:
        raise ValueError("Parallel group would cross the fixed six-trial PPO update")
    plan=dict(schedule_before=deepcopy(state["schedule"].state_dict()),
        sampler_before=deepcopy(state["learner"].sampler.bit_generator.state),
        rng_before=deepcopy(capture_rng()))
    try:
        plans=[]
        for _ in range(lanes):
            task,seeds=state["schedule"].sample(episodes)
            limit=episode_limit_for_task(task)
            if type(limit) is not int or limit<1:
                raise ValueError("Native source task horizon must be a positive integer")
            plans.append(dict(task=task,seeds=seeds,episode_limit=limit,
                              schedule_trial=state["schedule"].trials))
        required=sum(len(item["seeds"])*item["episode_limit"] for item in plans)
        if remaining_actions<required:
            raise CampaignBudgetReached("Insufficient reservation for complete parallel K4 trials; no RPC dispatched")
        admit_time()
        plan.update(plans=plans,required_actions=required)
        return plan
    except BaseException:
        restore_unstarted_source_group(state,plan)
        raise


class PPOUpdateRejected(RuntimeError):
    """An attempted update was rolled back; rejected state is forensic only."""
    def __init__(self,reason,*,metrics,rejected_state,rejected_rng,cause_type):
        super().__init__(reason)
        self.metrics=metrics
        self.rejected_state=rejected_state
        self.rejected_rng=rejected_rng
        self.cause_type=cause_type


def guarded_ppo_update(learner,rollouts,*,max_kl=None):
    """Optionally halt unsafe updates without consuming their on-policy data."""
    if max_kl is None:
        return learner.update(rollouts)
    if not math.isfinite(max_kl) or max_kl<=0:
        raise ValueError("PPO update KL guard must be finite and positive")
    before=deepcopy(learner.state_dict())
    before_rng=deepcopy(capture_rng())
    metrics=None
    try:
        metrics=learner.update(rollouts)
        finite=all(math.isfinite(float(value)) for value in metrics.values())
        approximate_kl=float(metrics.get("approximate_kl",float("nan")))
        if not finite or not math.isfinite(approximate_kl) or approximate_kl>max_kl:
            raise ValueError(f"PPO update reported approximate_kl={approximate_kl}, limit={max_kl}, finite={finite}")
        return metrics
    except BaseException as cause:
        # Capture the rejected optimizer/policy/RNG before restoring the valid
        # learner. The caller saves this separately from resumable training.
        rejected=None; rejected_rng=None
        try:
            rejected=deepcopy(learner.state_dict())
            rejected_rng=deepcopy(capture_rng())
        except Exception as capture_error:
            rejected=dict(forensic_capture_error=type(capture_error).__name__,
                          reason=str(capture_error))
        finally:
            learner.load_state_dict(before)
            restore_rng(before_rng)
        raise PPOUpdateRejected(str(cause),metrics=deepcopy(metrics),
            rejected_state=rejected,rejected_rng=rejected_rng,
            cause_type=type(cause).__name__) from cause


def validate_capture_qualification(path, *, model_path, norm_stats):
    """Only a saved successful real comparison can enable final-only capture."""
    receipt=json.loads(Path(path).read_text())
    if not (receipt.get("passed") is True and receipt.get("status")=="passed"
            and receipt.get("timing_comparison_eligible") is True
            and receipt.get("same_unchanged_frozen_motor") is True):
        raise ValueError("Final-only capture requires a passed allocated-runtime qualification")
    modes=receipt.get("modes",{})
    final=modes.get("final_only",{})
    runtime=final.get("runtime_contract",{})
    if (final.get("final_observation_only") is not True
            or not final.get("native_noise_parity",{}).get("passed")
            or Path(runtime.get("model_path","")).resolve()!=Path(model_path).resolve()
            or Path(runtime.get("norm_stats_path","")).resolve()!=Path(norm_stats).resolve()):
        raise ValueError("Capture qualification does not match the requested motor/runtime")
    comparisons=receipt.get("comparisons",{})
    if len(comparisons)<4 or not all(item.get("passed") is True for item in comparisons.values()):
        raise ValueError("Incomplete capture qualification comparisons")
    return dict(path=str(Path(path).resolve()),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                source_commit=receipt.get("source_commit"),runtime_contract=runtime,
                frozen_fingerprint=final.get("frozen_before"))


def validate_loaded_capture_runtime(qualification,runtime,frozen_fingerprint=None):
    """Paths alone cannot admit a checkpoint that changed in place."""
    expected=qualification["runtime_contract"]
    for name in ("norm_stats_sha256","rlinf_commit","robotwin_commit"):
        if not expected.get(name) or runtime.get(name)!=expected[name]:
            raise ValueError(f"Loaded capture runtime differs from qualification: {name}")
    if frozen_fingerprint is not None:
        if (not qualification.get("frozen_fingerprint")
                or frozen_fingerprint!=qualification["frozen_fingerprint"]):
            raise ValueError("Loaded frozen motor differs from capture qualification")


class CampaignEnv:
    """Budget every real action before execution, without owning learner state."""
    def __init__(self, env, budget, *, run_id):
        self.env=env; self.budget=budget; self.run_id=run_id
        self.context={}; self.episode=0; self.chunk=0; self.sequence=0

    def __getattr__(self,name):
        return getattr(self.env,name)

    def set_recording_context(self, metadata):
        self.context.update(metadata)
        if hasattr(self.env,"set_recording_context"):
            self.env.set_recording_context(metadata)

    def record_decision_metadata(self, metadata):
        if hasattr(self.env,"record_decision_metadata"):
            self.env.record_decision_metadata(metadata)

    def reset(self,task_name,seed):
        if not all(self.context.get(k) for k in ("phase","method","trial_id","policy_version")):
            raise RuntimeError("Campaign reset requires complete behavior context")
        self.task=task_name; self.episode+=1; self.chunk=0
        # The qualified adapter issues no policy motor commands during reset.
        # A runtime that introduces settling commands needs a pre-reset bound
        # and reservation; never charge unbounded work after it happened.
        if getattr(self.env,"reset_policy_action_bound",0)!=0:
            raise RuntimeError("Campaign runtime requires reserved reset-action accounting")
        result=self.env.reset(task_name,seed)
        if self.env.last_reset_info.get("reset_policy_actions",0)!=0:
            raise RuntimeError("Unexpected unreserved reset motor actions")
        return result

    def step(self,noise,*,max_primitive_steps=None):
        ordinary=min(self.env.native_horizon,self.env.episode_limit-self.env.elapsed_steps)
        requested=ordinary
        if max_primitive_steps is not None: requested=min(requested,max_primitive_steps)
        if requested<1: raise CampaignBudgetReached("No episode/source action allowance remains")
        decision_id=f"{self.run_id}:action:{self.sequence}"
        self.sequence+=1
        remaining=self.budget.remaining(phase=self.context["phase"],method=self.context["method"])
        reservation=self.budget.reserve(decision_id,int(requested),
            phase=self.context["phase"],method=self.context["method"],task=self.task,
            episode=f"{self.context['trial_id']}:{self.episode}",chunk=self.chunk,
            policy_version=self.context["policy_version"])
        if not reservation.execute:
            raise CampaignBudgetReached("A recorded/uncertain decision must not execute again")
        if reservation.granted<1:
            raise CampaignBudgetReached("The cumulative physical-action allowance is exhausted")
        self.set_recording_context(dict(campaign_decision_id=decision_id,
                                       campaign_reserved_actions=reservation.granted))
        # Passing H32 as an external budget on every chunk would truncate every
        # episode after one decision. Preserve the caller's full allowance,
        # adding a smaller cap only at an actual campaign boundary.
        actual_cap=(int(reservation.granted) if remaining<=ordinary
                    or reservation.granted<requested else max_primitive_steps)
        following,reward,end,trunc,info=(self.env.step(noise) if actual_cap is None
            else self.env.step(noise,max_primitive_steps=actual_cap))
        actual=int(info["primitive_steps"])
        self.budget.resolve(decision_id,actual)
        self.chunk+=1
        info=dict(info,campaign_decision_id=decision_id,
                  campaign_reserved_actions=reservation.granted)
        return following,reward,end,trunc,info


class BehaviorPolicyArchive:
    """Immutable acting weights; full optimizer/replay recovery stays separate."""
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.index={}; self.owners={}; self.bytes_written=0; self.seconds=0.

    def identify(self,learner,method):
        if method=="sac" and learner.warming_up:
            return dict(policy_version="bounded_uniform_warmup",behavior="bounded uniform",
                        policy_checkpoint=None,policy_sha256=None,policy_checkpoint_sha256=None)
        key=(id(learner),method,int(learner.updates))
        if key in self.index and self.owners[id(learner)]() is learner: return self.index[key]
        started=time.monotonic()
        network=learner.actor if method=="sac" else learner.policy
        state={name:value.detach().cpu().contiguous().clone()
               for name,value in network.state_dict().items()}
        digest=hashlib.sha256()
        for name,value in sorted(state.items()):
            digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        sha=digest.hexdigest(); path=self.root/(sha+".pt")
        if not path.exists():
            temporary=path.with_suffix(".tmp")
            torch.save(dict(kind="sac_actor" if method=="sac" else "ppo_actor_value",
                            state_dict=state,parameter_sha256=sha),temporary)
            temporary.replace(path); self.bytes_written+=path.stat().st_size
        descriptor=dict(policy_version=f"{method}:update:{learner.updates}:{sha}",
            policy_checkpoint=str(path.resolve()),policy_sha256=sha,
            policy_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.owners[id(learner)]=weakref.ref(learner)
        self.index[key]=descriptor; self.seconds+=time.monotonic()-started
        return descriptor


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def append_json(path,value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value,allow_nan=False)+"\n")


def durable_torch_replace(payload, path):
    """The native group fence cannot clear before file and directory fsync."""
    temporary=path.with_suffix(".tmp")
    torch.save(payload,temporary)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


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
    p.add_argument("--source-steps",type=int)
    p.add_argument("--source-iterations",type=int)
    p.add_argument("--source-eval-iterations",type=int,nargs="+",default=[])
    p.add_argument("--benchmark-episode-limits",action="store_true")
    p.add_argument("--ppo-update-steps",type=int,default=0)
    p.add_argument("--ppo-trials-per-update",type=int)
    p.add_argument("--checkpoint-steps",type=int,default=20_000)
    p.add_argument("--train-episodes",type=int,default=4)
    p.add_argument("--eval-episodes",type=int,default=8)
    p.add_argument("--episode-limit",type=int,default=400)
    p.add_argument("--native-horizon",type=int,default=32)
    p.add_argument("--task-config-name",choices=("demo_clean","demo_randomized"),default="demo_clean")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--eval-trials",type=int,default=1)
    p.add_argument("--native-episodes",type=int)
    p.add_argument("--source-validation-episodes",type=int)
    p.add_argument("--source-milestones",type=int,nargs="+",default=[])
    p.add_argument("--stage-source-validation-only",action="store_true")
    p.add_argument("--campaign-root",type=Path)
    p.add_argument("--campaign-id",default="robotwin-source-1m")
    p.add_argument("--campaign-max-actions",type=int,default=1_000_000)
    p.add_argument("--campaign-prior-receipts",type=Path)
    p.add_argument("--record-corpus",action="store_true")
    p.add_argument("--final-observation-only",action="store_true")
    p.add_argument("--capture-qualification",type=Path)
    p.add_argument("--collector-lanes",type=int,default=1)
    p.add_argument("--collector-group-seconds",type=float)
    p.add_argument("--ppo-epochs",type=int,default=5)
    p.add_argument("--ppo-minibatch-trials",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--ppo-max-update-kl",type=float)
    p.add_argument("--sac-warmup",type=int,default=500)
    p.add_argument("--sac-batch-size",type=int,default=256)
    p.add_argument("--sac-utd",type=int,default=20)
    p.add_argument("--action-limit",type=float,default=1.5)
    p.add_argument("--wall-seconds",type=float,default=2400)
    p.add_argument("--boundary-reserve-seconds",type=float,default=180)
    p.add_argument("--qualification-smoke",action="store_true")
    p.add_argument("--resume",type=Path)
    args=p.parse_args(argv)
    if args.source_steps is None: args.source_steps=272_000 if args.campaign_root else 10_240_000
    if args.ppo_trials_per_update is None: args.ppo_trials_per_update=8 if args.campaign_root else 256
    if args.native_episodes is None: args.native_episodes=1 if args.campaign_root else 2
    if args.source_validation_episodes is None: args.source_validation_episodes=4 if args.campaign_root else 0
    if args.ppo_max_update_kl is not None and (
            not math.isfinite(args.ppo_max_update_kl) or args.ppo_max_update_kl<=0):
        p.error("--ppo-max-update-kl must be finite and positive")
    if (any(getattr(args,k)<1 for k in ("source_steps","ppo_trials_per_update","checkpoint_steps","train_episodes",
        "eval_episodes","episode_limit","native_horizon","eval_trials","ppo_epochs",
        "ppo_minibatch_trials","sac_batch_size","sac_utd")) or args.native_episodes<0
        or args.sac_warmup<0 or args.ppo_update_steps<0 or args.wall_seconds<=args.boundary_reserve_seconds
        or args.source_validation_episodes<0 or args.campaign_max_actions<1):
        p.error("Invalid budget or batch settings")
    if not set(args.source_tasks)<=set(SOURCE_TASKS):
        p.error("Source tasks must remain inside the six VLRLA source tasks")
    if not args.qualification_smoke and set(args.source_tasks)!=set(SOURCE_TASKS):
        p.error("A source subset must be labeled --qualification-smoke")
    if not set(args.eval_tasks)<=set(SOURCE_TASKS+HELDOUT_TASKS):
        p.error("Unknown evaluation task")
    if len(set(args.methods))!=len(args.methods): p.error("Duplicate method")
    if args.collector_lanes not in (1,2,3,6):
        p.error("collector-lanes must be one, two, three or six")
    if args.collector_lanes>1:
        if (not args.source_iterations or args.ppo_trials_per_update!=6
                or args.train_episodes!=4 or args.ppo_update_steps or args.native_horizon!=32
                or not args.benchmark_episode_limits or not args.campaign_root or not args.record_corpus
                or args.ppo_max_update_kl is None or not set(args.methods)&{"ppo","ppo_reset"}
                or len(args.source_tasks)!=6 or set(args.source_tasks)!=set(SOURCE_TASKS)):
            p.error("Parallel PPO requires the native six-task, six-complete-K4 fixed-round campaign with corpus and KL guard")
        if (args.collector_group_seconds is None or not math.isfinite(args.collector_group_seconds)
                or args.collector_group_seconds<=0):
            p.error("Parallel PPO needs an explicit reviewed --collector-group-seconds bound")
    elif args.collector_group_seconds is not None:
        p.error("collector-group-seconds requires parallel collection")
    if args.campaign_root and not args.record_corpus:
        p.error("Campaign collection requires --record-corpus")
    if args.campaign_root and args.source_steps*len(args.methods)>args.campaign_max_actions:
        p.error("Source allocations exceed the entire physical campaign allowance")
    if args.campaign_prior_receipts and not args.campaign_root:
        p.error("Prior receipts require --campaign-root")
    if args.final_observation_only and not args.capture_qualification:
        p.error("Final-only capture requires --capture-qualification")
    if args.source_iterations is not None:
        if args.source_iterations<1 or args.train_episodes!=4 or args.ppo_update_steps:
            p.error("Fixed source iterations require positive count, complete K4 trials and no primitive update threshold")
        if not args.campaign_root or not args.record_corpus:
            p.error("Fixed source iterations require campaign accounting and raw corpus recording")
        if args.source_milestones:
            p.error("Whole-trial iterations cannot use exact action milestones")
        if not args.source_eval_iterations: args.source_eval_iterations=[args.source_iterations]
        if (args.source_eval_iterations!=sorted(set(args.source_eval_iterations))
                or args.source_eval_iterations[0]<1
                or args.source_eval_iterations[-1]!=args.source_iterations):
            p.error("Evaluation iterations must increase strictly and finish at --source-iterations")
    elif args.source_eval_iterations:
        p.error("Evaluation iteration schedule requires --source-iterations")
    if args.source_milestones:
        if not args.campaign_root or not args.record_corpus:
            p.error("Source milestones require campaign accounting and raw corpus recording")
        if (args.source_milestones!=sorted(set(args.source_milestones))
                or args.source_milestones[0]<1 or args.source_milestones[-1]!=args.source_steps):
            p.error("Source milestones must increase strictly and finish at --source-steps")
        if args.train_episodes!=4 or args.source_validation_episodes!=4:
            p.error("Staged source training and validation require the declared K4 trials")
        if set(args.source_tasks)!=set(SOURCE_TASKS):
            p.error("Staged comparison requires all six source tasks")
        if not args.stage_source_validation_only and args.eval_tasks!=list(HELDOUT_TASKS[:2]):
            p.error("Stage --eval-tasks must be the predeclared adjust_bottle move_stapler_pad pair")
        if not args.stage_source_validation_only and args.eval_episodes!=8:
            p.error("Stage target adaptation uses the declared K8 trials")
    elif args.stage_source_validation_only and not args.source_iterations:
        p.error("Stage validation mode requires action milestones or fixed source iterations")
    if args.resume:
        try: assert_resume_allowed(args.resume)
        except RuntimeError as error: p.error(str(error))
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.collector_lanes>1:
        for name in ("model_path","norm_stats","robotwin_root","rlinf_root","output",
                     "campaign_root","campaign_prior_receipts","capture_qualification","resume"):
            value=getattr(args,name)
            if value is not None:setattr(args,name,value.resolve())
    resume_joint=None
    if args.resume is not None:
        resume_path=(args.resume/"coordinator-recovery.pt" if args.resume.is_dir() else args.resume)
        resume_joint=torch.load(resume_path,map_location="cpu",weights_only=False)
        assert_resume_allowed(args.resume,checkpoint=resume_joint)
        parallel=resume_joint.get("contract",{}).get("parallel_source_collection")
        if parallel or resume_joint.get("campaign_journal_fence"):
            if (not parallel or args.collector_lanes!=parallel["lanes"]
                    or args.collector_group_seconds!=parallel["group_seconds_bound"]
                    or args.campaign_root is None):
                raise RuntimeError("Parallel recovery requires the same explicit collector and canonical campaign configuration before GPU setup")
            assert_campaign_checkpoint_fence(args.campaign_root,resume_joint)
    if sys.platform!="linux" or not torch.cuda.is_available():
        raise RuntimeError("Run only inside a qualified allocated Linux CUDA job")
    if (args.output/"contract.json").exists():
        raise FileExistsError("Use a fresh output identity, including for resumed allocations")
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    capture_qualification=None
    if args.final_observation_only:
        capture_qualification=validate_capture_qualification(
            args.capture_qualification,model_path=args.model_path,norm_stats=args.norm_stats)
    corpus_writer=None
    if args.record_corpus:
        from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter
        corpus_writer=RobotwinCorpusWriter(args.output/"corpus",dict(
            source_commit=args.source_commit,model_path=str(args.model_path),
            norm_stats=str(args.norm_stats),source_tasks=args.source_tasks,
            train_episodes=args.train_episodes,native_horizon=args.native_horizon))
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv
    runtime_options={}
    task_episode_limits=None
    if args.benchmark_episode_limits:
        from fluffy_bassoon.rl.robotwin_noise import load_benchmark_episode_limits
        task_episode_limits=load_benchmark_episode_limits(args.robotwin_root,SOURCE_TASKS+HELDOUT_TASKS)
        args.episode_limit=max(task_episode_limits.values())
        runtime_options["task_episode_limits"]=task_episode_limits
    if args.source_iterations:
        # BalancedSchedule visits every task once per block. Reserve full
        # benchmark episodes before starting a trial; never cut to hit a count.
        limits=[(task_episode_limits or {}).get(task,args.episode_limit) for task in args.source_tasks]
        blocks,remainder=divmod(args.source_iterations*args.ppo_trials_per_update,len(limits))
        planned_source_max=args.train_episodes*(blocks*sum(limits)+remainder*max(limits))
        if args.source_steps<planned_source_max:
            raise ValueError("Source safety cap cannot fit the fixed complete-trial schedule")
    if args.final_observation_only: runtime_options["final_observation_only"]=True
    if corpus_writer is not None: runtime_options["corpus_writer"]=corpus_writer
    env=RobotwinNoiseEnv(model_path=args.model_path,norm_stats=args.norm_stats,
                        robotwin_root=args.robotwin_root,rlinf_root=args.rlinf_root,
                        device="cuda",episode_limit=args.episode_limit,
                        action_horizon=args.native_horizon,output_dir=args.output/"runtime",
                        task_config_name=args.task_config_name,**runtime_options)
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
    if args.collector_lanes>1:
        contract["parallel_source_collection"]=dict(lanes=args.collector_lanes,
            controller_microbatch=1,motor_microbatch=1,
            group_seconds_bound=args.collector_group_seconds,
            native_task_horizons=True,raw_corpus=True,
            scope="source PPO only; scalar SAC and evaluations unchanged")
    if task_episode_limits is not None:
        contract["task_episode_limits"]=task_episode_limits
    if args.source_iterations:
        contract["source_budget"]="fixed complete K4 trials; actual actions reported; source steps are a safety reservation"
        contract["source_iterations"]=dict(count=args.source_iterations,
            trials_per_iteration=args.ppo_trials_per_update,episodes_per_trial=args.train_episodes,
            evaluation_iterations=args.source_eval_iterations,planned_max_actions_per_learner=planned_source_max,
            ppo_update_rule="one update after each complete iteration; no partial-trial action cuts",
            sac_update_rule="online UTD unchanged; iterations count complete trial groups")
    if args.ppo_max_update_kl is not None:
        contract["ppo_max_update_kl"] = args.ppo_max_update_kl
    contract["sac"] = asdict(SACConfig(warmup_chunks=args.sac_warmup,
        batch_size=args.sac_batch_size,utd=args.sac_utd))
    if args.campaign_root:
        contract["campaign"]=dict(id=args.campaign_id,max_actions=args.campaign_max_actions,
            source_caps={method:args.source_steps for method in args.methods},
            source_validation_episodes=args.source_validation_episodes,
            trained_reset_target_lesions=["normal"],
            recipe="bounded three-arm pilot; smaller complete-trial PPO batch, not the 256-trial recipe",
            raw_public_corpus=True,behavior_checkpoints="immutable acting weights at every used version")
        if args.source_iterations:
            contract["campaign"]["recipe"]="fixed complete-trial source iterations; physical-action safety caps"
    if args.record_corpus: contract["corpus"]="lossless public RGB/proprio/language and actual decision/action/outcome records"
    if args.source_milestones:
        contract["stages"]=dict(source_action_milestones=args.source_milestones,
            source_validation_episodes=args.source_validation_episodes,
            target_tasks=[] if args.stage_source_validation_only else args.eval_tasks,
            boundary="exact external source-stage budget cut with bootstrap; pending PPO rollouts retained",
            ppo_update_rule="ordinary declared batch trigger; never flush at intermediate stages",
            evaluation_rng="restore source RNG after frozen cloned-stage evaluation",
            final_evaluation="last stage replaces the standard final evaluation")
    if capture_qualification: contract["capture_qualification"]=capture_qualification
    if hasattr(env,"contract"):
        value=env.contract
        runtime=value() if callable(value) else dict(value)
        # GPU identity is re-attested in every allocation. A physical PCI
        # address is evidence, not part of the resumable learning contract.
        write_json(args.output/"runtime-allocation.json",runtime)
        contract["runtime"]={k:v for k,v in runtime.items() if k!="actual_render_pci"}
        if capture_qualification:
            validate_loaded_capture_runtime(capture_qualification,runtime)
    write_json(args.output/"contract.json",contract)
    budget=None
    if args.campaign_root:
        from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
        budget=CampaignBudget(args.campaign_root,campaign_id=args.campaign_id,
            total_actions=args.campaign_max_actions,
            source_caps={method:args.source_steps for method in args.methods},
            contract=contract,run_id=args.output.name,source_commit=args.source_commit)
        budget.__enter__()
        if args.campaign_prior_receipts:
            for receipt in json.loads(args.campaign_prior_receipts.read_text()):
                budget.import_receipt(**receipt)
        if args.collector_lanes>1 and resume_joint is not None:
            assert_campaign_checkpoint_fence(args.campaign_root,resume_joint,live_budget=budget)
        env=CampaignEnv(env,budget,run_id=args.output.name)
    policy_archive=BehaviorPolicyArchive(args.output/"corpus"/"policies") if args.record_corpus else None
    vector_env=None
    collector_time=(CollectorTimeAdmission(args.collector_group_seconds,args.boundary_reserve_seconds)
                    if args.collector_lanes>1 else None)
    if args.collector_lanes>1:
        from fluffy_bassoon.rl.robotwin_production_vector import ProductionRobotwinVectorEnv
        vector_env=ProductionRobotwinVectorEnv(env.env if isinstance(env,CampaignEnv) else env,
            budget,run_id=args.output.name,lanes=args.collector_lanes,
            corpus_writer=corpus_writer,output_dir=args.output)
    def set_context(method,phase,trial_id,learner=None,**metadata):
        behavior=(policy_archive.identify(learner,"sac" if isinstance(learner,NoiseSAC) else method)
                  if policy_archive is not None and learner is not None else
                  dict(policy_version="native_gaussian" if method=="native" else "preflight_no_action"))
        context=dict(method=method,phase=phase,trial_id=trial_id,
                     **dict(dict(policy_checkpoint=None,policy_sha256=None,
                                 policy_checkpoint_sha256=None),**behavior),**metadata)
        if hasattr(env,"set_recording_context"): env.set_recording_context(context)
        return context
    def decision_metadata(metadata):
        if hasattr(env,"record_decision_metadata"): env.record_decision_metadata(metadata)
    episodes_path=args.output/"episodes.jsonl"
    updates_path=args.output/"updates.jsonl"
    history=[]; trained={}; summaries={}; completed_units=set()
    stage_reports={}; active_milestone=None; checkpoint_rng_override=None
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
    def checkpoint(method,learner,schedule,steps,task_steps,pending=None,*,iteration_state=None):
        path=args.output/(method+"-recovery.pt")
        temporary=path.with_suffix(".tmp")
        payload=dict(contract=contract,method=method,learner=learner.state_dict(),
                     schedule=schedule.state_dict(),source_steps=steps,task_steps=task_steps,
                     pending_rollouts=pending or [],boundary="complete trial or declared budget cut",
                     resume_stage="source",elapsed_seconds=time.monotonic()-started)
        rejected=source_states.get(method,{}).get("rejected_group")
        if rejected: payload.update(rejected_group=rejected,resume_blocked=True)
        if vector_env is not None:
            payload["campaign_journal_fence"]=campaign_checkpoint_fence(budget)
            payload["parallel_vector_root"]=str(vector_env.root.resolve())
            group=vector_env.group_recovery_metadata()
            if group and group.get("status") not in ("checkpointed","cancelled_before_dispatch"):
                payload["parallel_collection_group"]=group
        if iteration_state is not None: payload.update(iteration_state)
        if vector_env is not None:durable_torch_replace(payload,path)
        else:
            torch.save(payload,temporary)
            temporary.replace(path)
    def native(phase,*,unit_prefix="",stage=None,tasks=None,stage_iteration=None):
        rng=np.random.default_rng(args.seed+907)
        for task in args.eval_tasks if tasks is None else tasks:
            for index in range(args.native_episodes):
                unit=f"{unit_prefix}{phase}:native:{task}:{index}"
                if args.campaign_root and unit in completed_units: continue
                unit_rng=(np.random.default_rng(np.random.SeedSequence([
                    args.seed+907,(SOURCE_TASKS+HELDOUT_TASKS).index(task),index]))
                    if stage is not None or stage_iteration is not None else rng)
                guard(); set_context("native",phase,f"{unit_prefix}{phase}:{task}:{index}",
                                     stage_source_actions=stage,stage_source_iteration=stage_iteration)
                features=env.reset(task,args.seed+900_000+index)
                steps=0; chunks=0; success=False; reward_sum=0.; done=False
                initial_hash=observation_hash(features)
                while not done:
                    decision_metadata(dict(noise_distribution="standard_gaussian",history_index=chunks))
                    _,reward,end,trunc,info=env.step(unit_rng.normal(size=math.prod(noise_shape)).astype(np.float32))
                    steps+=int(info["primitive_steps"]); chunks+=1
                    success|=bool(info.get("success",False)); reward_sum+=float(reward); done=bool(end or trunc)
                row=dict(method="native",phase=phase,task=task,episode=index+1,
                         seed=args.seed+900_000+index,success=success,return_=reward_sum,
                         primitive_steps=steps,chunks=chunks,initial_observation_sha256=initial_hash)
                if stage is not None: row["stage_source_actions"]=stage
                if stage_iteration is not None: row["stage_source_iteration"]=stage_iteration
                append_json(episodes_path,row); log("native_episode",**row)
                if args.campaign_root:
                    history.append(row); completed_units.add(unit); save_joint(force=True)
    def sac_trial(learner,task,seeds,*,phase,source=False,online=False,primitive_budget=None,method="sac",stage=None,
                  source_iteration=None,stage_iteration=None):
        total=0; rows=[]; external_cut=False
        for episode,seed in enumerate(seeds):
            observation=np.asarray(env.reset(task,int(seed)),np.float32)
            initial_hash=observation_hash(observation); done=False; steps=chunks=0; success=False; reward_sum=0.
            while not done:
                if policy_archive is not None:
                    behavior=policy_archive.identify(learner,"sac")
                    env.set_recording_context(behavior)
                # Save the interaction RNG before consuming the actual sample.
                decision_metadata(dict(episode_index=episode,chunk_index=chunks,
                    history_index=learner.environment_steps,
                    source_warmup=bool(source and learner.warming_up),
                    interaction_numpy_rng=deepcopy(learner.rng.bit_generator.state),
                    interaction_torch_rng=learner.torch_rng.get_state().cpu().numpy().copy(),
                    optimizer_updates=int(learner.updates),online_updates=bool(source or online)))
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
                external_cut=bool(info.get("budget_truncated",False)
                    or (budget_end and episode+1<len(seeds)))
                if source or online:
                    learner.observe(observation,action,float(reward),following,bool(end),bool(trunc),primitive_steps=count)
                    metrics=learner.update(gradient_steps=args.sac_utd)
                    if metrics and learner.environment_steps%25==0:
                        append_json(updates_path,dict(method=method,phase=phase,**metrics))
                        log("sac_updates",method=method,phase=phase,source_primitive_steps=learner.primitive_steps,**metrics)
                observation=np.asarray(following,np.float32); done=bool(end or trunc or budget_end)
            row=dict(task=task,episode=episode+1,seed=int(seed),success=success,return_=reward_sum,
                     primitive_steps=steps,chunks=chunks,initial_observation_sha256=initial_hash,
                     budget_truncated=external_cut)
            if stage is not None: row["stage_source_actions"]=stage
            if source_iteration is not None: row["source_iteration"]=source_iteration
            if stage_iteration is not None: row["stage_source_iteration"]=stage_iteration
            append_json(episodes_path,dict(method=method,phase=phase,**row)); rows.append(row)
            if primitive_budget is not None and total>=primitive_budget: break
        return dict(episodes=rows,primitive_steps=total,chunks=sum(row["chunks"] for row in rows),
                    budget_truncated=external_cut,requested_episodes=len(seeds))
    source_states={}
    accepted_resume_steps={}
    next_method_index=0
    last_joint_steps=0
    if resume_joint is not None:
        if resume_joint["contract"]!=contract:
            raise ValueError("Resume scientific/source contract changed")
        if args.campaign_root:
            completed_units.update(resume_joint.get("completed_units",[]))
            history[:]=resume_joint.get("history",[])
            stage_reports.update(resume_joint.get("stages",{}))
        if collector_time is not None:
            collector_time.load_state_dict(resume_joint["collector_time_admission"])
    def initialize_sources():
        nonlocal next_method_index,last_joint_steps
        for method in args.methods:
            learner=new_sac() if method=="sac" else new_ppo(method=="ppo_reset")
            schedule=BalancedSchedule(args.source_tasks,args.seed+101)
            state=dict(learner=learner,schedule=schedule,steps=0,
                       task_steps={task:0 for task in args.source_tasks},pending=[],last_checkpoint_steps=0,
                       completed_trials=0,iterations=0)
            if resume_joint is not None:
                saved=resume_joint["states"][method]
                learner.load_state_dict(saved["learner"])
                schedule.load_state_dict(saved["schedule"])
                state.update(steps=int(saved["source_steps"]),task_steps=saved["task_steps"],
                             pending=saved.get("pending_rollouts",[]),
                             last_checkpoint_steps=int(saved["source_steps"]))
                if args.source_iterations:
                    state.update(completed_trials=int(saved["completed_source_trials"]),
                                 iterations=int(saved["completed_source_iterations"]))
                    partial=state["completed_trials"]-state["iterations"]*args.ppo_trials_per_update
                    if (not 0<=state["iterations"]<=args.source_iterations
                            or not 0<=partial<=args.ppo_trials_per_update
                            or state["steps"]>args.source_steps
                            or (method!="sac" and (learner.updates!=state["iterations"]
                                or len(state["pending"])!=partial))):
                        raise ValueError("Fixed-iteration recovery counters are inconsistent")
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
            if state.get("rejected_group"):
                states[method]["rejected_group"]=state["rejected_group"]
            if args.source_iterations:
                states[method].update(completed_source_trials=state["completed_trials"],
                                      completed_source_iterations=state["iterations"])
        path=args.output/"coordinator-recovery.pt"
        temporary=path.with_suffix(".tmp")
        recovery=dict(contract=contract,states=states,
                        rng=capture_rng() if checkpoint_rng_override is None else deepcopy(checkpoint_rng_override),
                        next_method_index=next_method_index,boundary="completed source trial/update",
                        completed_units=sorted(completed_units),history=history,
                        stages=stage_reports,
                        resume_stage="source_or_completed_evaluation_unit",
                        elapsed_seconds=time.monotonic()-started)
        if collector_time is not None:
            recovery["collector_time_admission"]=collector_time.state_dict()
        if vector_env is not None:
            recovery["campaign_journal_fence"]=campaign_checkpoint_fence(budget)
            recovery["parallel_vector_root"]=str(vector_env.root.resolve())
            group=vector_env.group_recovery_metadata()
            if group and group.get("status") not in ("checkpointed","cancelled_before_dispatch"):
                recovery["parallel_collection_group"]=group
        if any(s.get("rejected_group") for s in source_states.values()): recovery["resume_blocked"]=True
        if vector_env is not None:durable_torch_replace(recovery,path)
        else:
            torch.save(recovery,temporary)
            temporary.replace(path)
        last_joint_steps=total
    def save_state(method,force=False):
        state=source_states[method]
        if force or state["steps"]-state["last_checkpoint_steps"]>=args.checkpoint_steps:
            checkpoint(method,state["learner"],state["schedule"],state["steps"],
                       state["task_steps"],state["pending"],iteration_state=(
                           dict(completed_source_trials=state["completed_trials"],
                                completed_source_iterations=state["iterations"])
                           if args.source_iterations else None))
            state["last_checkpoint_steps"]=state["steps"]
    def refresh_summaries():
        for method,state in source_states.items():
            learner=state["learner"]
            summaries[method]=dict(source_steps=state["steps"],task_steps=state["task_steps"],
                                   allocation_source_steps=state["steps"]-accepted_resume_steps[method],
                                   updates=learner.updates,optimizer_steps=learner.optimizer_steps,
                                   pending_trials=len(state["pending"]),
                                   pending_primitive_steps=sum(r["primitive_steps"] for _,r in state["pending"]))
            if state.get("rejected_group"): summaries[method]["rejected_group"]=state["rejected_group"]
            if method=="sac": summaries[method]["contract"]=learner.contract()
            if args.source_iterations:
                summaries[method].update(completed_source_trials=state["completed_trials"],
                    completed_source_iterations=state["iterations"],
                    source_action_safety_cap=args.source_steps)
            if budget is not None:
                physical=budget.snapshot()["by_method"].get(f"source/{method}",0)
                summaries[method].update(source_physical_charge=physical,
                    retained_learning_actions=state["steps"],
                    unretained_or_uncertain_source_charge=max(0,physical-state["steps"]))
    def maybe_ppo_update(method):
        state=source_states[method]; pending=state["pending"]
        if not pending: return
        pending_steps=sum(r["primitive_steps"] for _,r in pending)
        if args.source_iterations:
            trials=state["completed_trials"]-state["iterations"]*args.ppo_trials_per_update
            if trials!=len(pending) or trials>args.ppo_trials_per_update:
                raise RuntimeError("Fixed source iteration has inconsistent complete-trial data")
            if trials<args.ppo_trials_per_update: return
        else:
            reached=(pending_steps>=args.ppo_update_steps if args.ppo_update_steps
                     else len(pending)>=args.ppo_trials_per_update)
            if not reached and state["steps"]<args.source_steps: return
        log("ppo_update_start",method=method,source_steps=state["steps"],
            trials=len(pending),primitive_steps=pending_steps,
            chunks=sum(r["chunks"] for _,r in pending))
        if args.ppo_max_update_kl is not None:
            # Also retain a disk boundary before CUDA work, so an external
            # interruption can restart the entire update from its old policy.
            save_state(method,force=True)
            save_joint(force=True)
        try:
            metrics=guarded_ppo_update(state["learner"],[r for r,_ in pending],
                                       max_kl=args.ppo_max_update_kl)
        except PPOUpdateRejected as exc:
            directory=args.output/"rejected-ppo-updates"
            directory.mkdir(exist_ok=True)
            path=directory/f"{method}-source-{state['steps']}-rejected.pt"
            temporary=path.with_suffix(".tmp")
            torch.save(dict(contract=contract,method=method,source_steps=state["steps"],
                max_update_kl=args.ppo_max_update_kl,rejection_reason=str(exc),
                cause_type=exc.cause_type,metrics=exc.metrics,
                rejected_learner=exc.rejected_state,rejected_rng=exc.rejected_rng,
                pending_rollouts=pending,task_steps=state["task_steps"],
                schedule=state["schedule"].state_dict(),
                resumable_training=False,valid_learner_restored=True),temporary)
            temporary.replace(path)
            log("ppo_update_rejected",method=method,source_steps=state["steps"],
                reason=str(exc),forensics=str(path),pending_trials=len(pending),
                valid_learner_restored=True)
            raise
        state["pending"]=[]
        if args.source_iterations: state["iterations"]+=1
        append_json(updates_path,dict(method=method,source_steps=state["steps"],**metrics))
        save_state(method,force=True)
        log("ppo_update",method=method,source_steps=state["steps"],**metrics)
        if args.source_iterations: record_iteration_complete(method)
    def record_iteration_complete(method):
        state=source_states[method]
        log("source_iteration_complete",method=method,source_iteration=state["iterations"],
            completed_source_trials=state["completed_trials"],source_steps=state["steps"],
            source_optimizer_updates=state["learner"].updates,
            optimizer_steps=state["learner"].optimizer_steps)
        save_state(method,force=True); save_joint(force=True)
    def maybe_sac_iteration(method):
        state=source_states[method]
        if not args.source_iterations: return
        trials=state["completed_trials"]-state["iterations"]*args.ppo_trials_per_update
        if trials>args.ppo_trials_per_update:
            raise RuntimeError("SAC fixed source iteration passed its complete-trial boundary")
        if trials==args.ppo_trials_per_update:
            state["iterations"]+=1
            record_iteration_complete(method)
    def parallel_source_round(method):
        from fluffy_bassoon.meta_rl.vla_noise_batched_collection import (
            CollectionNotStarted,collect_trials_batched,
        )
        from fluffy_bassoon.rl.robotwin_production_vector import ProductionAdmissionError
        state=source_states[method]; learner=state["learner"]
        remaining=min(args.source_steps-state["steps"],budget.remaining(phase="source",method=method))
        plan=plan_parallel_source_group(state,lanes=args.collector_lanes,
            episodes=args.train_episodes,episode_limit_for_task=vector_env.episode_limit_for_task,
            remaining_actions=remaining,
            admit_time=lambda: collector_time.require(args.wall_seconds-(time.monotonic()-started)))
        plans=plan["plans"]
        lane_iteration=[dict(source_iteration=state["iterations"]+1,
            trial_in_iteration=state["completed_trials"]%6+lane+1) for lane in range(len(plans))]
        admitted=False; collected=None
        trial_start=time.monotonic()
        try:
            behavior=policy_archive.identify(learner,method)
            for lane,item in enumerate(plans):
                vector_env.set_lane_context(lane,dict(phase="source",method=method,
                    trial_id=f"source:{method}:{item['schedule_trial']}",**behavior,
                    stage_source_actions=None,stage_source_iteration=None,**lane_iteration[lane]))
            vector_env.admit_group([item["task"] for item in plans],[item["seeds"] for item in plans])
            admitted=True
            log("source_execution_group_start",method=method,source_steps=state["steps"],
                lanes=len(plans),plans=plans,required_actions=plan["required_actions"],
                source_iteration=state["iterations"]+1,optimizer_steps=learner.optimizer_steps)
            collected=collect_trials_batched(learner,vector_env,
                [item["task"] for item in plans],[item["seeds"] for item in plans],
                primitive_budget=remaining,controller_microbatch=1,
                should_stop=lambda: time.monotonic()-started>=args.wall_seconds-args.boundary_reserve_seconds,
                decision_callback=vector_env.record_decision_metadata,
                episode_callback=lambda lane,row: record_episode(method,"source",collector_lane=lane,
                    **lane_iteration[lane])(row))
            group_receipt=vector_env.finish_group()
        except CollectionNotStarted as error:
            try: vector_env.cancel_group(str(error))
            finally: restore_unstarted_source_group(state,plan)
            raise WallBudgetReached(str(error)) from error
        except ProductionAdmissionError as error:
            proof=vector_env.group_recovery_metadata()
            if vector_env.can_restore_unstarted_group():
                restore_unstarted_source_group(state,plan)
            else:
                quarantine_collection_failure(state,output_dir=args.output,method=method,error=error,
                    capture_rng=capture_rng,save_payload=torch.save,collected=collected,
                    details=dict(plan=plan,vector=proof,
                                 vector_resume_blocker=getattr(vector_env,"resume_blocker_metadata",None),
                                 campaign_budget=budget.snapshot()))
            raise
        except BaseException as error:
            if not admitted:
                restore_unstarted_source_group(state,plan)
            else:
                quarantine_collection_failure(state,output_dir=args.output,method=method,error=error,
                    capture_rng=capture_rng,save_payload=torch.save,collected=collected,
                    details=dict(plan=plan,vector=vector_env.group_recovery_metadata(),
                                 vector_resume_blocker=getattr(vector_env,"resume_blocker_metadata",None),
                                 campaign_budget=budget.snapshot()))
            raise
        elapsed=time.monotonic()-trial_start
        collector_time.observe(elapsed)
        # Every physical lane is charged before replay. A rejected lane blocks
        # the whole group and future ordinary resume, retaining prior pending.
        errors=admit_collected_group(state,collected,output_dir=args.output,method=method,
            capture_rng=capture_rng,save_payload=torch.save)
        state["completed_trials"]+=len(collected)
        save_state(method,force=True)
        save_joint(force=True)
        vector_env.acknowledge_checkpoint(args.output/"coordinator-recovery.pt")
        log("source_execution_group_complete",method=method,source_steps=state["steps"],
            lanes=len(collected),trial_seconds=elapsed,replay_errors=errors,
            completed_source_trials=state["completed_trials"],group_receipt=group_receipt)
        maybe_ppo_update(method)
        refresh_summaries()
    def source_round(method):
        state=source_states[method]; learner=state["learner"]
        previous_iteration=state["iterations"]
        if method!="sac": maybe_ppo_update(method)
        else: maybe_sac_iteration(method)
        if args.source_iterations:
            # Recovery may restore the six completed trials immediately before
            # their update. Finish that round without collecting the next one
            # ahead of the outer barrier and its frozen evaluation.
            if state["iterations"]!=previous_iteration:
                refresh_summaries()
                return
            if state["iterations"]>=args.source_iterations: return
        elif state["steps"]>=args.source_steps: return
        guard()
        if method!="sac" and vector_env is not None:
            parallel_source_round(method)
            return
        allowance=None if args.source_iterations else args.source_steps-state["steps"]
        if active_milestone is not None and not args.source_iterations:
            allowance=min(allowance,active_milestone-state["steps"])
            if allowance<1: return
        if budget is not None and not args.source_iterations:
            allowance=min(allowance,budget.remaining(phase="source",method=method))
            if allowance<1: raise CampaignBudgetReached(f"No source allowance remains for {method}")
        schedule_before=deepcopy(state["schedule"].state_dict()) if args.source_iterations else None
        task,seeds=state["schedule"].sample(args.train_episodes)
        iteration_metadata={}
        if args.source_iterations:
            horizon=(env.episode_limit_for_task(task) if hasattr(env,"episode_limit_for_task") else args.episode_limit)
            required=args.train_episodes*horizon
            remaining=args.source_steps-state["steps"]
            if budget is not None: remaining=min(remaining,budget.remaining(phase="source",method=method))
            if remaining<required:
                state["schedule"].load_state_dict(schedule_before)
                raise CampaignBudgetReached("Insufficient reservation for a complete source K4 trial; no partial trial executed")
            iteration_metadata=dict(source_iteration=state["iterations"]+1,
                trial_in_iteration=state["completed_trials"]%args.ppo_trials_per_update+1)
        set_context(method,"source",f"source:{method}:{state['schedule'].trials}",learner,
                    stage_source_actions=active_milestone,stage_source_iteration=None,**iteration_metadata)
        trial_start=time.monotonic()
        log("source_trial_start",method=method,task=task,seeds=seeds,
            source_steps=state["steps"],optimizer_steps=learner.optimizer_steps,**iteration_metadata)
        if method=="sac":
            result=sac_trial(learner,task,seeds,phase="source",source=True,
                             primitive_budget=allowance,stage=active_milestone,
                             source_iteration=iteration_metadata.get("source_iteration"))
            errors={}
        else:
            rollout,result=learner.collect_trial(env,task,seeds,
                primitive_budget=allowance,
                episode_callback=record_episode(method,"source",**iteration_metadata),
                decision_callback=decision_metadata if args.record_corpus else None)
            errors=learner.replay_error(rollout)
            if errors["log_prob_max_error"]>2e-4 or errors["value_max_error"]>1e-5:
                raise RuntimeError(f"Old-policy replay failed: {errors}")
        if args.source_iterations:
            if len(result["episodes"])!=args.train_episodes or result.get("budget_truncated",False):
                raise RuntimeError("Fixed source iterations require complete K4 trials without external cuts")
            state["completed_trials"]+=1
        if method!="sac": state["pending"].append((rollout,result))
        state["steps"]+=result["primitive_steps"]
        if active_milestone is not None:
            result["stage_source_actions"]=active_milestone
            result["external_stage_budget_cut"]=bool(result.get("budget_truncated",False)
                and active_milestone<args.source_steps)
        state["task_steps"][task]+=result["primitive_steps"]
        if method!="sac": learner.primitive_steps=state["steps"]
        save_state(method)
        log("source_trial",method=method,source_steps=state["steps"],task=task,
            episodes=len(result["episodes"]),trial_seconds=time.monotonic()-trial_start,
            stage_source_actions=active_milestone,
            external_stage_budget_cut=result.get("external_stage_budget_cut",False),
            **iteration_metadata,**errors)
        if method!="sac": maybe_ppo_update(method)
        else: maybe_sac_iteration(method)
        refresh_summaries()
    def evaluate_frozen_sources(trained,*,stage=None,unit_prefix="",tasks=None,stage_iteration=None):
        targets=args.eval_tasks if tasks is None else tasks
        stage_metadata=dict(stage_source_actions=stage)
        if stage_iteration is not None: stage_metadata["stage_source_iteration"]=stage_iteration
        if args.source_validation_episodes:
            for task_index,task in enumerate(args.source_tasks):
                seeds=[3_000_000_000+task_index*10_000+i*997
                       for i in range(args.source_validation_episodes)]
                for method in args.methods:
                    unit=f"{unit_prefix}source_validation:{method}:{task}"
                    if args.campaign_root and unit in completed_units: continue
                    guard()
                    learner=new_sac() if method=="sac" else new_ppo(method=="ppo_reset")
                    learner.load_state_dict(deepcopy(trained[method]))
                    set_context(method,"source_validation",f"{unit_prefix}source-validation:{method}:{task}",learner,
                                **stage_metadata)
                    if method=="sac":
                        result=sac_trial(learner,task,seeds,phase="source_validation",method=method,stage=stage,
                                         stage_iteration=stage_iteration)
                    else:
                        _,result=learner.collect_trial(env,task,seeds,
                            episode_callback=record_episode(method,"source_validation",**stage_metadata),
                            decision_callback=decision_metadata if args.record_corpus else None)
                    history.append(dict(result,method=method,phase="source_validation",task=task,
                                        **stage_metadata))
                    log("source_validation_trial",method=method,task=task,
                        successes=[row["success"] for row in result["episodes"]],updates=learner.updates)
                    if args.campaign_root:
                        completed_units.add(unit); save_joint(force=True)
        native("native_before_target",unit_prefix=unit_prefix,stage=stage,tasks=targets,stage_iteration=stage_iteration)
        for task in targets:
            for repetition in range(args.eval_trials):
                seeds=[args.seed+2_000_000+repetition*100_000+i*997 for i in range(args.eval_episodes)]
                for method in args.methods:
                    if method=="sac":
                        for online in (False,True):
                            label="sac_online" if online else "sac_frozen"
                            unit=f"{unit_prefix}target:{label}:{task}:{repetition}"
                            if args.campaign_root and unit in completed_units: continue
                            guard(); learner=new_sac(); learner.load_state_dict(deepcopy(trained["sac"]))
                            convention=(learner.begin_target_adaptation(replay_mode="fresh_target")
                                        if online else None)
                            set_context(label,"target",f"{unit_prefix}target:{label}:{task}:{repetition}",learner,
                                        **stage_metadata)
                            result=sac_trial(learner,task,seeds,phase="target",online=online,method=label,stage=stage,
                                             stage_iteration=stage_iteration)
                            history.append(dict(method=label,phase="target",task=task,repetition=repetition,
                                **stage_metadata,
                                target_adaptation=convention,
                                target_update_costs=learner.target_counters() if online else None,**result))
                            log("target_trial",method=label,task=task,successes=[r["success"] for r in result["episodes"]])
                            if args.campaign_root:
                                recovery=args.output/("target-"+hashlib.sha256(unit.encode()).hexdigest()+"-recovery.pt")
                                temporary=recovery.with_suffix(".tmp")
                                torch.save(dict(unit=unit,contract=contract,learner=learner.state_dict(),
                                    result=result,rng=capture_rng(),boundary="completed target trial"),temporary)
                                temporary.replace(recovery)
                                history[-1]["target_recovery"]=str(recovery.resolve())
                                history[-1]["target_recovery_sha256"]=hashlib.sha256(recovery.read_bytes()).hexdigest()
                                completed_units.add(unit); save_joint(force=True)
                    else:
                        lesions=(("normal",) if args.campaign_root and method=="ppo_reset"
                                 else ("normal","memory_reset","reward_blind"))
                        for lesion in lesions:
                            unit=f"{unit_prefix}target:{method}:{lesion}:{task}:{repetition}"
                            if args.campaign_root and unit in completed_units: continue
                            guard(); learner=new_ppo(method=="ppo_reset")
                            learner.load_state_dict(deepcopy(trained[method]))
                            set_context(method,"target",f"{unit_prefix}target:{method}:{lesion}:{task}:{repetition}",
                                        learner,lesion=lesion,**stage_metadata)
                            _,result=learner.collect_trial(env,task,seeds,
                                reset_memory=method=="ppo_reset" or lesion=="memory_reset",
                                reward_blind=lesion=="reward_blind",
                                episode_callback=record_episode(method,"target",lesion=lesion,repetition=repetition,
                                                                **stage_metadata),
                                decision_callback=decision_metadata if args.record_corpus else None)
                            history.append(dict(method=method,phase="target",lesion=lesion,repetition=repetition,
                                                **stage_metadata,**result))
                            log("target_trial",method=method,lesion=lesion,task=task,
                                successes=[r["success"] for r in result["episodes"]])
                            if args.campaign_root:
                                completed_units.add(unit); save_joint(force=True)
        native("native_after",unit_prefix=unit_prefix,stage=stage,tasks=targets,stage_iteration=stage_iteration)

    def evaluate_stage(milestone=None,*,source_iteration=None):
        nonlocal checkpoint_rng_override
        stage_key=str(milestone) if source_iteration is None else f"iteration:{source_iteration}"
        prefix=f"stage:{stage_key}:"
        complete_key=prefix+"complete"
        if complete_key in completed_units: return
        if source_iteration is None:
            if any(state["steps"]!=milestone for state in source_states.values()):
                raise RuntimeError("Every source arm must reach the exact stage action barrier")
        elif any(state["iterations"]!=source_iteration for state in source_states.values()):
            raise RuntimeError("Every source arm must complete the declared iteration before evaluation")
        guard()
        stage_rng=capture_rng()
        directory=args.output/"stages"/(str(milestone) if source_iteration is None else f"iteration-{source_iteration}")
        directory.mkdir(parents=True,exist_ok=True)
        report=stage_reports.get(stage_key)
        if report is None:
            report=dict(source_action_milestone=milestone,status="evaluating",
                        snapshots={},source_steps={},source_updates={},pending_trials={})
            if source_iteration is not None: report["source_iteration"]=source_iteration
            for method,state in source_states.items():
                learner=state["learner"]
                path=directory/(method+"-recovery.pt")
                if path.exists(): raise FileExistsError("Stage source snapshots are immutable")
                temporary=path.with_suffix(".tmp")
                torch.save(dict(contract=contract,method=method,source_stage=milestone,source_iteration=source_iteration,
                    source_steps=state["steps"],task_steps=state["task_steps"],
                    learner=learner.state_dict(),schedule=state["schedule"].state_dict(),
                    pending_rollouts=state["pending"],rng=deepcopy(stage_rng),
                    boundary=("completed whole-trial source iteration" if source_iteration is not None
                              else "exact matched source milestone; pending PPO data retained")),temporary)
                temporary.replace(path)
                report["snapshots"][method]=dict(path=str(path.resolve()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                report["source_steps"][method]=state["steps"]
                report["source_updates"][method]=learner.updates
                report["pending_trials"][method]=len(state["pending"])
            stage_reports[stage_key]=report
        else:
            for snapshot in report["snapshots"].values():
                if hashlib.sha256(Path(snapshot["path"]).read_bytes()).hexdigest()!=snapshot["sha256"]:
                    raise RuntimeError("A saved stage source snapshot changed")
        report["manifest_path"]=str((directory/"stage.json").resolve())
        write_json(directory/"stage.json",report)
        checkpoint_rng_override=deepcopy(stage_rng)
        save_joint(force=True)
        frozen={method:deepcopy(state["learner"].state_dict())
                for method,state in source_states.items()}
        log("source_stage_evaluation_start",source_action_milestone=milestone,
            source_iteration=source_iteration,
            source_steps=report["source_steps"],source_updates=report["source_updates"],
            pending_trials=report["pending_trials"])
        try:
            evaluate_frozen_sources(frozen,stage=milestone,unit_prefix=prefix,
                tasks=[] if args.stage_source_validation_only else args.eval_tasks,stage_iteration=source_iteration)
        finally:
            # Cloning/evaluating controllers must not consume the source action
            # stream. Per-unit checkpoint RNG also uses this same source state.
            restore_rng(stage_rng)
            checkpoint_rng_override=None
        if any(state["learner"].updates!=report["source_updates"][method]
               or len(state["pending"])!=report["pending_trials"][method]
               for method,state in source_states.items()):
            raise RuntimeError("Frozen stage evaluation altered source learning state")
        report["status"]="complete"
        report["evaluation_units"]=sorted(unit for unit in completed_units if unit.startswith(prefix))
        report["source_rng_restored"]=True
        report["reported_without_checkpoint_selection"]=True
        write_json(directory/"stage.json",report)
        write_json(directory/"evaluation.json",[row for row in history
            if (row.get("stage_source_actions")==milestone if source_iteration is None
                else row.get("stage_source_iteration")==source_iteration)])
        completed_units.add(complete_key)
        save_joint(force=True)
        log("source_stage_complete",source_action_milestone=milestone,
            source_iteration=source_iteration,
            source_steps=report["source_steps"],source_updates=report["source_updates"],
            pending_trials=report["pending_trials"],manifest=report["manifest_path"])

    status="complete"
    try:
        log("runtime_ready",observation_dim=observation_dim,noise_shape=noise_shape)
        # Preflight one declared source task; held-out/native comparison is
        # deferred until source training is complete, avoiding first-allocation
        # benchmark cost before learning starts.
        preflight_task=args.source_tasks[0]
        set_context("preflight","qualification","motor_noise_parity")
        env.reset(preflight_task,args.seed+777_000)
        parity=env.check_noise_parity(np.zeros(math.prod(noise_shape),np.float32))
        log("motor_noise_parity",task=preflight_task,**parity)
        write_json(args.output/"motor-parity.json",parity)
        frozen_before=env.frozen_fingerprint()
        if capture_qualification:
            validate_loaded_capture_runtime(capture_qualification,contract["runtime"],frozen_before)
        initialize_sources()
        for method in args.methods: save_state(method,force=True)
        # Interleave complete K4 trials so both learning arms advance in the
        # first allocation. Pending PPO trials retain sampled old-policy data.
        save_joint(force=True)
        if args.source_iterations:
            for iteration in range(1,args.source_iterations+1):
                while any(s["iterations"]<iteration for s in source_states.values()):
                    method=args.methods[next_method_index]
                    if source_states[method]["iterations"]<iteration: source_round(method)
                    next_method_index=(next_method_index+1)%len(args.methods)
                    save_joint()
                if iteration in args.source_eval_iterations: evaluate_stage(source_iteration=iteration)
        else:
            barriers=args.source_milestones or [args.source_steps]
            for barrier in barriers:
                active_milestone=barrier if args.source_milestones else None
                while any(s["steps"]<barrier for s in source_states.values()):
                    method=args.methods[next_method_index]
                    previous_updates=source_states[method]["learner"].updates
                    if source_states[method]["steps"]<barrier: source_round(method)
                    next_method_index=(next_method_index+1)%len(args.methods)
                    save_joint(force=method!="sac" and source_states[method]["learner"].updates>previous_updates)
                if args.source_milestones: evaluate_stage(barrier)
        for method,state in source_states.items():
            if method!="sac": maybe_ppo_update(method)
            save_state(method,force=True)
            trained[method]=deepcopy(state["learner"].state_dict())
        refresh_summaries()
        if args.source_iterations:
            if any(state["iterations"]!=args.source_iterations
                    or state["completed_trials"]!=args.source_iterations*args.ppo_trials_per_update
                    or (method!="sac" and (state["learner"].updates!=args.source_iterations or state["pending"]))
                    for method,state in source_states.items()):
                raise RuntimeError("Fixed source iterations did not complete their full trial/update schedule")
        elif len({summaries[m]["source_steps"] for m in args.methods})!=1:
            raise RuntimeError("Source interaction budgets do not match")
        if not args.source_milestones and not args.source_iterations: evaluate_frozen_sources(trained)
    except WallBudgetReached as exc:
        status="wall_budget_stop"; log(status,reason=str(exc))
    except CampaignBudgetReached as exc:
        status="campaign_budget_stop"; log(status,reason=str(exc))
    except PPOUpdateRejected as exc:
        status="ppo_update_rejected"; log(status,reason=str(exc))
        raise
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
        source_complete=(len(summaries)==len(args.methods) and
            (all(x["completed_source_iterations"]==args.source_iterations
                 and x["completed_source_trials"]==args.source_iterations*args.ppo_trials_per_update
                 for x in summaries.values()) if args.source_iterations
             else all(x["source_steps"]==args.source_steps for x in summaries.values())))
        summary=dict(status=status,elapsed_seconds=time.monotonic()-started,
                     source=summaries,target=history,contract=contract,
                     source_budget_match=source_complete)
        if args.source_iterations:
            summary["source_action_counts_match"]=len({x["source_steps"] for x in summaries.values()})==1
        if budget is not None: summary["campaign_budget"]=budget.snapshot()
        if args.campaign_root:
            if args.source_milestones or args.source_iterations: summary["stages"]=stage_reports
            summary["evaluation_history"]=history
            summary["target"]=[row for row in history if row.get("phase")=="target"]
            summary["source_validation"]=[row for row in history if row.get("phase")=="source_validation"]
            summary["native"]=[row for row in history if row.get("method")=="native"]
        if corpus_writer is not None: summary["corpus"]=corpus_writer.statistics()
        if collector_time is not None: summary["collector_time_admission"]=collector_time.state_dict()
        if policy_archive is not None:
            summary["behavior_policy_archive"]=dict(bytes_written=policy_archive.bytes_written,
                io_seconds=policy_archive.seconds,acting_versions=len(policy_archive.index))
        if hasattr(env,"counters"):
            value=env.counters
            summary["runtime_counters"]=value() if callable(value) else value
        write_json(args.output/"summary.json",summary)
        try:
            if vector_env is not None: vector_env.close()
            if hasattr(env,"close"): env.close()
        finally:
            if budget is not None: budget.close()
    return 0 if status in ("complete","wall_budget_stop","campaign_budget_stop") else 3


if __name__=="__main__":
    raise SystemExit(main())
