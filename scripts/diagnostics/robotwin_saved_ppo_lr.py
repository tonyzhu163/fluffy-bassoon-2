#!/usr/bin/env python3
"""Source-only PPO learning-rate diagnostic; no robot or VLA is instantiated.

Every candidate forks the same complete pre-update learner/RNG/rollouts. The
batch-three diagnostic never advances the batch-eight matched comparison.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import runpy
import signal
import sys
import time

from robotwin_saved_ppo_update import write_json


RATES=(3e-6,3e-5)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--input-sha256",required=True)
    parser.add_argument("--reference-post",type=Path,required=True)
    parser.add_argument("--reference-post-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=int,default=45)
    args=parser.parse_args()
    if sys.platform!="linux" or not 20<=args.wall_seconds<=45:
        raise ValueError("Allocated Linux diagnostic only, with <=45-second wall allowance")
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status="initializing",source_commit=args.source_commit,
        diagnostic="saved_source_PPO_batch3_learning_rates",main_comparison_advanced=False,
        executed_new_robot_actions=0,input_checkpoint=str(args.input),input_sha256=args.input_sha256,
        original_complete_recovery=str(args.input),reference_post=str(args.reference_post),
        reference_post_sha256=args.reference_post_sha256,candidate_rates=list(RATES),candidates=[],
        provisional_stability_rule=dict(mean_joint_analytic_kl_max=.02,post_clip_fraction_max=.2,
            additional_requirements="finite nonzero gradient, changed weights, one update/five optimizer steps"),
        scientific_scope="source-policy numerical stability only; no target selection or robot-performance claim")
    write_json(args.output/"result.json",report)

    def expired(signum,frame):
        raise TimeoutError("Diagnostic deadline; original recovery and completed candidate files are preserved")
    signal.signal(signal.SIGALRM,expired)
    signal.signal(signal.SIGTERM,expired)
    signal.alarm(args.wall_seconds)
    active_candidate=None
    try:
        if file_sha(args.input)!=args.input_sha256 or file_sha(args.reference_post)!=args.reference_post_sha256:
            raise ValueError("An immutable source/reference checkpoint SHA does not match")
        argv=sys.argv
        sys.argv=["gpu_probe.py","--expected-count","1","--output",str(args.output/"gpu-probe.json"),
                  "--purpose","training","--baige-container"]
        try:
            runpy.run_path(str(Path(__file__).parents[1]/"[train]gpu_probe.py"),run_name="__main__")
        finally:
            sys.argv=argv
        import numpy as np
        import torch
        from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig,RecurrentNoisePPO
        report["initial_float32_matmul_precision"]=torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        report["float32_matmul_precision"]=torch.get_float32_matmul_precision()
        payload=torch.load(args.input,map_location="cpu",weights_only=False)
        reference_payload=torch.load(args.reference_post,map_location="cpu",weights_only=False)
        pending=payload["pending_rollouts"]
        if (payload["method"]!="ppo" or len(pending)!=3 or payload["learner"]["updates"]!=0
                or payload["learner"]["optimizer_steps"]!=0):
            raise ValueError("Require the original unchanged learner and three fresh complete trials")
        for rollout,result in pending:
            ends=rollout["trial_ends"].reshape(-1).tolist()
            if (result["requested_episodes"]!=4 or len(result["episodes"])!=4 or result["budget_truncated"]
                    or not ends[-1] or any(ends[:-1]) or result["task"] not in payload["contract"]["source_tasks"]):
                raise ValueError("Every source trajectory must be a complete eligible K4 trial")
        config=PPOConfig(**payload["learner"]["config"])
        if config.epochs!=5 or config.minibatch_trials<3 or config.reset_each_episode:
            raise ValueError("Preserve the five-epoch single-minibatch persistent PPO objective")
        reference_contract=reference_payload.get("contract",{})
        if (reference_contract.get("input_checkpoint_sha256")!=args.input_sha256
                or reference_payload["learner"]["updates"]!=1
                or reference_payload["learner"]["optimizer_steps"]!=5):
            raise ValueError("Reference post-update checkpoint is not the declared fork of this source")

        def new_learner():
            learner=RecurrentNoisePPO(payload["learner"]["observation_dim"],
                payload["learner"]["noise_shape"],config=config,device="cuda")
            learner.load_state_dict(payload["learner"])
            return learner

        def weight_hash(policy):
            digest=hashlib.sha256()
            for key,value in sorted(policy.state_dict().items()):
                value=value.detach().cpu().contiguous()
                digest.update(key.encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        @torch.no_grad()
        def gaussians(learner):
            means=[]; actions=[]; replay=[]; core_features=[]; log_probs=[]
            learner.policy.eval()
            for rollout,_ in pending:
                captured=[]; captured_features=[]
                def capture_actor(module,inputs,output):
                    captured.append(output.detach())
                    captured_features.append(inputs[0].detach())
                hook=learner.policy.actor.register_forward_hook(
                    capture_actor)
                try:
                    logp,_,values=learner.policy.evaluate_sequence(
                        rollout["observations"].to("cuda"),rollout["raw_actions"].to("cuda"),
                        rollout["resets"].to("cuda"))
                finally:
                    hook.remove()
                mask=rollout["action_mask"].to("cuda")
                means.append(torch.stack(captured)[mask].double().cpu())
                core_features.append(torch.stack(captured_features)[mask].cpu())
                log_probs.append(logp[mask].double().cpu())
                actions.append(rollout["raw_actions"][rollout["action_mask"]].double())
                replay.append(dict(log_prob_max_error=float((logp.cpu()-rollout["log_probs"]).abs().max()),
                                   value_max_error=float((values.cpu()-rollout["values"]).abs().max())))
            return dict(means=torch.cat(means),raw_actions=torch.cat(actions),replay=replay,
                        core_features=torch.cat(core_features),log_probs=torch.cat(log_probs),
                        log_std=learner.policy.log_std.detach().clamp(-5,2).double().cpu())

        original=new_learner()
        old=gaussians(original)
        if any(not all(math.isfinite(v) for v in row.values()) or row["log_prob_max_error"]>2e-4
               or row["value_max_error"]>1e-5 for row in old["replay"]):
            raise RuntimeError("Original stored likelihood/value replay must match before comparing learning rates")
        before_hash=weight_hash(original.policy)
        report.update(behavior_replay=old["replay"],source_episodes=12,
            source_actions=sum(result["primitive_steps"] for _,result in pending),
            chunks=int(old["means"].shape[0]),action_dimensions=int(old["means"].shape[-1]),
            policy_before_sha256=before_hash)

        inputs=torch.cat([r["observations"][r["action_mask"]] for r,_ in pending])
        trial_index=torch.cat([torch.full((int(r["action_mask"].sum()),),i,dtype=torch.int64)
                               for i,(r,_) in enumerate(pending)])
        decision_index=torch.cat([torch.arange(int(r["action_mask"].sum())) for r,_ in pending])
        def magnitude(value):
            return dict(min=float(value.min()),max=float(value.max()),
                        absolute_max=float(value.abs().max()),rms=float(value.double().square().mean().sqrt()))
        obs_dim=payload["learner"]["observation_dim"]
        action_dim=old["means"].shape[-1]
        report["input_magnitudes"]=dict(all_inputs=magnitude(inputs),
            frozen_motor_features=magnitude(inputs[:,:obs_dim]),
            previous_noise=magnitude(inputs[:,obs_dim:obs_dim+action_dim]),
            feedback_and_clocks=magnitude(inputs[:,-4:]),
            feedback_columns=[magnitude(inputs[:,-4+i]) for i in range(4)],
            layout="frozen motor features; previous noise; previous reward, boundary, episode clock, primitive clock")
        if obs_dim==2062:
            report["input_magnitudes"].update(vlm_features=magnitude(inputs[:,:2048]),
                                             normalized_proprioception=magnitude(inputs[:,2048:2062]))

        def divergence(new,label):
            if new["means"].shape!=old["means"].shape:
                raise RuntimeError("Policy Gaussian dimensions changed")
            old_log_std=old["log_std"]; new_log_std=new["log_std"]
            variance_kl=(new_log_std-old_log_std+
                torch.exp(2*old_log_std)/(2*torch.exp(2*new_log_std))-.5).sum()
            mean_shift_kl=((old["means"]-new["means"]).square()/
                           (2*torch.exp(2*new_log_std))).sum(-1)
            analytic=mean_shift_kl+variance_kl
            raw=old["raw_actions"]
            old_logp=(-.5*((raw-old["means"])/old_log_std.exp()).square()
                      -old_log_std-.5*math.log(2*math.pi)).sum(-1)
            new_logp=(-.5*((raw-new["means"])/new_log_std.exp()).square()
                      -new_log_std-.5*math.log(2*math.pi)).sum(-1)
            # Tanh and external noise-scale Jacobians cancel from old/new ratios.
            log_ratio=new_logp-old_logp
            ratio=log_ratio.exp()
            quantities=dict(mean_joint_analytic_kl=float(analytic.mean()),
                max_joint_analytic_kl=float(analytic.max()),
                mean_joint_mean_shift_kl=float(mean_shift_kl.mean()),
                joint_variance_kl=float(variance_kl),
                sampled_mean_old_minus_new_log_probability=float((-log_ratio).mean()),
                sampled_post_approximate_joint_kl=float(((ratio-1)-log_ratio).mean()),
                post_clip_fraction=float(((ratio-1).abs()>config.clip).double().mean()),
                min_post_log_ratio=float(log_ratio.min()),max_post_log_ratio=float(log_ratio.max()))
            finite=all(math.isfinite(v) for v in quantities.values())
            diagnostic_path=args.output/(label+"-decisions.npz")
            arrays=dict(trial_index=trial_index,decision_index=decision_index,
                policy_inputs=inputs,raw_actions=raw,old_mean=old["means"],new_mean=new["means"],
                old_actor_inputs=old["core_features"],new_actor_inputs=new["core_features"],
                old_log_std=old_log_std,new_log_std=new_log_std,
                old_joint_log_probability=old_logp,new_joint_log_probability=new_logp,
                log_ratio=log_ratio,joint_analytic_kl=analytic,joint_mean_shift_kl=mean_shift_kl,
                joint_variance_kl=variance_kl.expand_as(analytic),
                clipped_mask=(ratio-1).abs()>config.clip,
                old_policy_float32_log_probability=old["log_probs"],
                new_policy_float32_log_probability=new["log_probs"])
            temporary=diagnostic_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle,**{key:value.numpy() for key,value in arrays.items()})
            temporary.replace(diagnostic_path)
            return dict(**{k:v if math.isfinite(v) else None for k,v in quantities.items()},
                all_finite=finite,dtype="float64",direction="KL(original || post)",
                aggregation="sum all action dimensions, mean active stored decisions",
                decision_arrays=str(diagnostic_path),decision_arrays_sha256=file_sha(diagnostic_path),
                old_mean=magnitude(old["means"]),new_mean=magnitude(new["means"]),
                old_std=magnitude(old_log_std.exp()),new_std=magnitude(new_log_std.exp()),
                old_actor_inputs=magnitude(old["core_features"]),new_actor_inputs=magnitude(new["core_features"]))

        reference=new_learner()
        reference.load_state_dict(reference_payload["learner"])
        reference_divergence=divergence(gaussians(reference),"reference-lr-3e-04")
        report["reference"]=dict(learning_rate=config.learning_rate,
            policy_after_sha256=weight_hash(reference.policy),post_divergence=reference_divergence,
            existing_update=True,new_optimizer_steps=0)
        write_json(args.output/"result.json",report)

        for rate in RATES:
            if args.wall_seconds-(time.monotonic()-started)<12:
                report["skipped_rates"]=list(RATES[len(report["candidates"]):])
                break
            active_candidate=dict(learning_rate=rate,status="updating",retained_ppo_updates=0)
            report["active_candidate"]=active_candidate
            write_json(args.output/"result.json",report)
            learner=new_learner()  # Restores the SAME original sampler and all RNG state.
            learner.config=replace(config,learning_rate=rate)
            for group in learner.optimizer.param_groups: group["lr"]=rate
            torch.cuda.synchronize()
            update_start=time.monotonic()
            gradient_norms=[]
            original_clip=torch.nn.utils.clip_grad_norm_
            def record_clip(*clip_args,**clip_kwargs):
                norm=original_clip(*clip_args,**clip_kwargs)
                gradient_norms.append(float(norm.detach().cpu()))
                return norm
            torch.nn.utils.clip_grad_norm_=record_clip
            try:
                metrics=learner.update([rollout for rollout,_ in pending])
            finally:
                torch.nn.utils.clip_grad_norm_=original_clip
            torch.cuda.synchronize()
            update_seconds=time.monotonic()-update_start
            after_hash=weight_hash(learner.policy)
            finite=all(math.isfinite(float(v)) for v in metrics.values())
            if (not finite or metrics["gradient_norm"]<=0 or before_hash==after_hash
                    or len(gradient_norms)!=5 or not all(math.isfinite(v) and v>0 for v in gradient_norms)
                    or learner.updates!=1 or learner.optimizer_steps!=5):
                raise RuntimeError("Candidate requires finite nonzero learning and exactly five optimizer steps")
            post_gaussians=gaussians(learner)
            reconstruction=gaussians(learner)
            repeat_errors=dict(mean_max_error=float((post_gaussians["means"]-reconstruction["means"]).abs().max()),
                log_prob_max_error=float((post_gaussians["log_probs"]-reconstruction["log_probs"]).abs().max()),
                method="independent sequence reconstruction from empty history on identical public inputs")
            if (not all(math.isfinite(v) for k,v in repeat_errors.items() if k!="method")
                    or repeat_errors["mean_max_error"]>1e-5 or repeat_errors["log_prob_max_error"]>2e-4):
                raise RuntimeError("Post-update recurrent reconstruction is not reproducible")
            post=divergence(post_gaussians,f"lr-{rate:.0e}")
            stable=(post["all_finite"] and post["mean_joint_analytic_kl"]<=.02
                    and post["post_clip_fraction"]<=.2)
            recovery=deepcopy(payload)
            recovery.update(contract=dict(diagnostic=report["diagnostic"],source_commit=args.source_commit,
                diagnostic_batch_trials=3,learning_rate=rate,original_source_contract=payload["contract"],
                input_checkpoint_sha256=args.input_sha256,main_comparison_advanced=False),
                learner=learner.state_dict(),pending_rollouts=[],consumed_rollouts=pending,
                boundary="complete source-only learning-rate diagnostic update")
            path=args.output/(f"lr-{rate:.0e}-recovery.pt")
            temporary=path.with_suffix(".tmp")
            torch.save(recovery,temporary); temporary.replace(path)
            active_candidate.update(status="complete",retained_ppo_updates=1,optimizer_steps=5,
                update_seconds=update_seconds,metrics=metrics,post_divergence=post,
                per_optimizer_step_unclipped_gradient_norms=gradient_norms,
                post_reconstruction=repeat_errors,
                policy_before_sha256=before_hash,policy_after_sha256=after_hash,
                provisional_stable=bool(stable),recovery=str(path),recovery_sha256=file_sha(path),
                optimizer_state_count=len(learner.optimizer.state_dict()["state"]))
            report["candidates"].append(active_candidate)
            report.pop("active_candidate",None)
            active_candidate=None
            write_json(args.output/"result.json",report)
        report["status"]="complete" if len(report["candidates"])==len(RATES) else "partial_completed_candidates"
    except Exception as error:
        report.update(status="wall_budget_stop" if isinstance(error,TimeoutError) else "failed",
                      error_type=type(error).__name__,reason=str(error),
                      complete_original_recovery_unchanged=True)
        if active_candidate is not None: active_candidate["partial_private_optimization_discarded"]=True
    finally:
        signal.alarm(0)
        report["elapsed_seconds"]=time.monotonic()-started
        report["retained_independent_diagnostic_updates"]=len(report["candidates"])
        write_json(args.output/"result.json",report)
        print(__import__("json").dumps(report,allow_nan=False),flush=True)
    return 0 if report["candidates"] else 2


if __name__=="__main__":
    raise SystemExit(main())
