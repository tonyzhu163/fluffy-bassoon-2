#!/usr/bin/env python3
"""Read-only CPU admission for two original trained RoboTwin source streams.

The caller must externally time-bound this process: the canonical filesystem
can block even a LOCK_NB request. This helper never opens a CampaignBudget,
initializes a motor, restores a learner, allocates CUDA, or launches children.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


RUNTIME_COMMIT="57016ba08eed79043fd67bf8b085bd6d42e6d766"
LINEAGE_COMMIT="db0184089fa88a5f96c153f9b80029bb91b77def"
PARENT_SHA256="1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0"
METHODS=("ppo","ppo_reset")


def require(condition,message):
    if not condition:raise ValueError("Trained pair preflight: "+message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def bound(ref):
    require(isinstance(ref,dict) and set(("path","sha256"))<=set(ref),"missing artifact binding")
    path=Path(ref["path"])
    require(path.is_absolute() and path.is_file(),"bound artifact must be an existing absolute file")
    require(sha256(path)==ref["sha256"],f"artifact hash mismatch: {path}")
    return path


def load_driver(worktree):
    """Import the qualified pinned runtime, without invoking main or CUDA."""
    sys.path.insert(0,str(worktree/"src"))
    spec=importlib.util.spec_from_file_location("trained_pair_qualified_driver",
        worktree/"scripts/diagnostics/robotwin_noise_compare.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    return driver


def load_plan_validator():
    path=Path(__file__).with_name("robotwin_trained_pair.py")
    spec=importlib.util.spec_from_file_location("trained_pair_plan_validator",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def input_due_evaluation(checkpoint):
    method,=checkpoint["contract"]["methods"]
    cursor=checkpoint["states"][method]["completed_source_iterations"]
    completed=set(checkpoint.get("completed_units",[]));due=[]
    for iteration in checkpoint["contract"]["source_iterations"]["evaluation_iterations"]:
        key=f"stage:iteration:{iteration}:complete"
        if iteration>cursor:continue
        if key in completed:
            require(checkpoint.get("stages",{}).get(f"iteration:{iteration}",{}).get("status")=="complete",
                    "completed stage lacks its completed report")
        else:
            require(iteration==cursor,"an earlier original evaluation stage was omitted")
            due.append(iteration)
    return due


def evaluation_unit_plan(checkpoint,parent,iteration):
    """Original570 evaluation keys, case IDs and indivisible result shapes."""
    contract=checkpoint["contract"];method,=contract["methods"]
    limits=contract["task_episode_limits"];native_episodes=parent["workers"][method]["native_episodes"]
    prefix=f"stage:iteration:{iteration}:";units=[]
    def add(unit,trial_id,phase,task,episodes,*,native=False,**selector):
        if episodes:
            units.append(dict(unit=unit,trial_id=trial_id,phase=phase,task=task,episodes=episodes,native=native,
                maximum_actions=episodes*limits[task],history_selector=dict(phase=phase,task=task,
                    stage_source_iteration=iteration,**selector)))
    for task in contract["source_tasks"]:
        add(prefix+f"source_validation:{method}:{task}",prefix+f"source-validation:{method}:{task}",
            "source_validation",task,contract["campaign"]["source_validation_episodes"],method=method)
    for task in contract["eval_tasks"]:
        for phase in ("native_before_target","native_after"):
            for index in range(native_episodes):
                add(prefix+f"{phase}:native:{task}:{index}",prefix+f"{phase}:{task}:{index}",
                    phase,task,1,native=True,method="native",episode=index+1)
        for lesion in (("normal",) if method=="ppo_reset" else ("normal","memory_reset","reward_blind")):
            unit=prefix+f"target:{method}:{lesion}:{task}:0"
            add(unit,unit,"target",task,contract["eval_episodes"],method=method,lesion=lesion,repetition=0)
    return units


def remaining_evaluation_units(checkpoint,parent):
    """Exact worst-case allowance for original units not already committed."""
    completed=set(checkpoint.get("completed_units",[]))
    return [row for iteration in input_due_evaluation(checkpoint)
        for row in evaluation_unit_plan(checkpoint,parent,iteration) if row["unit"] not in completed]


def source_goal_admission(cursor,*,allow_source_goal_boundary=False):
    require(type(cursor) is int and 0<cursor<=100 and (allow_source_goal_boundary or cursor<100),
            "source stream already reached its original goal")
    return 0 if cursor==100 else 10800


def ledger_view(root,*,expected_snapshot):
    """Read an already locked canonical ledger; no ledger constructor/writer."""
    path=root/"snapshot.json";snapshot=json.loads(path.read_text())
    header=snapshot["header"]
    first=json.loads((root/"events/0000000000.json").read_text())
    require(first["kind"]=="header" and first["header"]==header==expected_snapshot["header"],
            "canonical journal/snapshot/original header differs")
    require(snapshot==expected_snapshot,"canonical snapshot differs from the bound predecessor")
    events=sorted((root/"events").glob("*.json"))
    require(events and len(events)==snapshot["journal_events"],"canonical snapshot/head event counts differ")
    head=json.loads(events[-1].read_text())
    require(head["sha256"]==snapshot["journal_sha256"],"canonical snapshot/head hashes differ")
    require(snapshot["pending_actions"]==snapshot["uncertain_actions"]==0
            and not snapshot["unresolved_decisions"],"canonical ledger has unresolved physical work")
    return dict(canonical_root=str(root),snapshot_sha256=sha256(path),
        header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
        total_action_cap=header["total_actions"],source_caps=header["source_caps"],by_method=snapshot["by_method"],
        **{key:snapshot[key] for key in ("journal_events","journal_sha256","charged_actions",
            "actual_actions","decision_count","pending_actions","uncertain_actions")})


def runtime_evidence(args,parent,checkpoint):
    """Bind small immutable runtime inputs before any CUDA probe or loading."""
    contract=checkpoint["contract"];capture=contract["capture_qualification"];runtime=contract["runtime"]
    require(args.capture_qualification.resolve()==Path(capture["path"]).resolve(),
            "capture qualification path differs from trained lineage")
    require(str(args.norm_stats)==contract["norm_stats"]==runtime["norm_stats_path"],
            "normalization path differs from trained lineage")
    require(runtime["episode_limits_source"]["sha256"]==parent["benchmark_horizon_config_sha256"],
            "trained benchmark horizon differs from original parent")
    refs=dict(benchmark_horizons=dict(path=str(args.robotwin_root/"task_config/_eval_step_limit.yml"),
            sha256=parent["benchmark_horizon_config_sha256"]),
        calibration=dict(path=str(Path(parent["calibration"]["root"])/"result.json"),
            sha256=parent["calibration"]["result_sha256"]),
        capture_qualification=dict(path=str(args.capture_qualification),sha256=capture["sha256"]),
        normalization=dict(path=str(args.norm_stats),sha256=runtime["norm_stats_sha256"]))
    for ref in refs.values():bound(ref)
    return refs


def run_preflight(plan_path,plan_sha256,*,allow_source_goal_boundary=False):
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CUDA_VISIBLE_DEVICES must be explicitly empty")
    plan_path=bound(dict(path=str(Path(plan_path).resolve()),sha256=plan_sha256))
    plan=json.loads(plan_path.read_text())
    launcher=load_plan_validator()
    require((launcher.RUNTIME,launcher.LINEAGE,launcher.PARENT_SHA)==(RUNTIME_COMMIT,LINEAGE_COMMIT,PARENT_SHA256),
            "launcher and preflight pins differ")
    launcher.validate_plan(plan,read_evidence=True)
    require(plan["parent_manifest"]["sha256"]==PARENT_SHA256,"original parent SHA is fixed")
    parent_path=bound(plan["parent_manifest"]);parent=json.loads(parent_path.read_text())
    owner_release=bound(plan["owner_release"]);compute_binding=bound(plan["compute_binding"])
    release=json.loads(owner_release.read_text());compute=json.loads(compute_binding.read_text())
    worktree=Path(plan["worktree"]).resolve()
    launcher.pinned_worktree(worktree,RUNTIME_COMMIT)
    driver=load_driver(worktree)
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import load_trial_continuation
    from fluffy_bassoon.meta_rl.robotwin_trained_parallel import (
        validate_lineage_parent,validate_trained_parallel_arguments,validate_trained_parallel_boundary)
    from fluffy_bassoon.meta_rl.collection_admission import assert_resume_allowed,assert_campaign_checkpoint_fence
    require(not driver.torch.cuda.is_initialized(),"CUDA was initialized during CPU preflight")
    rows={};roots={};snapshots={};artifact_refs=[plan["parent_manifest"],plan["owner_release"],plan["compute_binding"],
        compute["authority"],compute["prior_charge_and_physical_snapshot"]]+[
            release["methods"][method]["terminal_receipt"] for method in METHODS]
    for method in METHODS:
        item=plan["methods"][method];command=plan["child_commands"][method]
        require(isinstance(command,list) and len(command)>2 and all(isinstance(v,str) for v in command),
                "child command must be an exact argument vector")
        require(Path(command[1]).resolve()==worktree/"scripts/diagnostics/robotwin_noise_compare.py",
                "child command uses a different driver")
        require(len([v for v in command[2:] if v.startswith("--")])
                ==len(set(v for v in command[2:] if v.startswith("--"))),"duplicate child options")
        args=driver.parse_args(command[2:])
        manifest_path=bound(item["manifest"]);checkpoint_path=bound(item["checkpoint"])
        motor_path=bound(item["frozen_motor"]);output=Path(item["output"])
        require(output.is_absolute() and not output.exists(),"child output must be fresh and absolute")
        require(args.methods==[method] and args.source_commit==RUNTIME_COMMIT
                and args.lineage_source_commit==LINEAGE_COMMIT and args.output.resolve()==output.resolve()
                and args.collector_lanes==6 and args.max_source_trials_per_process==6
                and args.recycle_at_round_boundary,"child execution settings differ")
        resume=(args.resume/"coordinator-recovery.pt" if args.resume.is_dir() else args.resume)
        require(resume.resolve()==checkpoint_path.resolve()
                and args.resume_trial_manifest.resolve()==manifest_path.resolve()
                and args.resume_trial_manifest_sha256==item["manifest"]["sha256"]
                and args.lineage_parent_manifest.resolve()==parent_path.resolve()
                and args.lineage_parent_manifest_sha256==PARENT_SHA256,"child artifact bindings differ")
        root=Path(parent["workers"][method]["ledger_root"]).resolve();roots[method]=root
        require(args.campaign_root.resolve()==root,"child uses a noncanonical ledger root")
        checkpoint=driver.torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        assert_resume_allowed(checkpoint_path,checkpoint=checkpoint)
        continuation=load_trial_continuation(manifest_path,item["manifest"]["sha256"],
            checkpoint_path=checkpoint_path,checkpoint=checkpoint,execution_commit=RUNTIME_COMMIT,lineage_commit=LINEAGE_COMMIT)
        require(continuation.method==method,"manifest method differs")
        validate_trained_parallel_arguments(args,checkpoint)
        validate_lineage_parent(parent_path,PARENT_SHA256,checkpoint=checkpoint,campaign_root=root,
            native_episodes=args.native_episodes)
        audit=validate_trained_parallel_boundary(continuation,checkpoint,manifest_path=manifest_path,campaign_root=root)
        runtime_refs=runtime_evidence(args,parent,checkpoint)
        artifact_refs.extend(runtime_refs.values())
        cursor=audit["completed_source_iterations"]
        required_source=source_goal_admission(cursor,allow_source_goal_boundary=allow_source_goal_boundary)
        motor=json.loads(motor_path.read_text());fingerprint=checkpoint["contract"]["capture_qualification"]["frozen_fingerprint"]
        require(motor.get("unchanged") is True and motor.get("before")==motor.get("after")==fingerprint,
                "frozen motor differs from qualified lineage")
        manifest=json.loads(manifest_path.read_text());artifact_refs.extend(
            [item["manifest"],item["checkpoint"],item["frozen_motor"]]+[manifest[key] for key in
                ("corpus_contract","progress","episodes_file","ledger_snapshot")])
        corpus_root=Path(manifest["corpus_root"]).resolve()
        require(corpus_root==(checkpoint_path.parent/"corpus").resolve()
                and Path(manifest["corpus_contract"]["path"]).resolve()==corpus_root/"contract.json",
                "predecessor corpus identity differs from its checkpoint")
        previous_snapshot=json.loads(bound(manifest["ledger_snapshot"]).read_text());header=previous_snapshot["header"]
        worker=parent["workers"][method]
        require(header["campaign_id"]==worker["campaign_id"]
                and header["total_actions"]==worker["maximum_actions"]
                and header["source_caps"]=={method:worker["source_safety_actions"]}
                and header["contract"]==json.loads(json.dumps(checkpoint["contract"])),
                "bound predecessor header differs from original worker/contract")
        snapshots[method]=previous_snapshot
        due=input_due_evaluation(checkpoint)
        remaining_units=remaining_evaluation_units(checkpoint,parent) if allow_source_goal_boundary else None
        rows[method]=dict(manifest=item["manifest"],checkpoint=item["checkpoint"],frozen_motor=item["frozen_motor"],
            output=str(output),canonical_root=str(root),trained_boundary=audit,input_due_evaluations=due,
            runtime_evidence=runtime_refs,
            maximum_input_due_evaluation_actions=(sum(unit["maximum_actions"] for unit in remaining_units)
                if remaining_units is not None else (31600 if method=="ppo" else 17200) if due else 0),
            source_goal_complete=cursor==100,has_pending_eval=bool(due),remaining_evaluation_units=remaining_units,
            required_source_actions=required_source,completed_units=checkpoint.get("completed_units",[]),
            stage_statuses={key:value["status"] for key,value in checkpoint.get("stages",{}).items()},command=command,
            collector_lanes=6,new_source_trials=0 if cursor==100 else 6,episodes_per_trial=4,
            new_optimizer_steps=0 if cursor==100 else 5,
            original_goal=100,original_evaluation_iterations=[10,25,50,100])
    require(roots["ppo"]!=roots["ppo_reset"],"independent learners require distinct original ledgers")
    require(Path(plan["methods"]["ppo"]["output"]).resolve()!=Path(plan["methods"]["ppo_reset"]["output"]).resolve(),
            "independent learners require distinct fresh outputs")
    with ExitStack() as stack:
        for method in METHODS:
            lock=stack.enter_context((roots[method]/"campaign.lock").open("rb"))
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise RuntimeError("Canonical source owner still holds a campaign ledger") from error
            stack.callback(fcntl.flock,lock.fileno(),fcntl.LOCK_UN)
        before={method:ledger_view(roots[method],expected_snapshot=snapshots[method]) for method in METHODS}
        for method in METHODS:
            fence=rows[method]["trained_boundary"]["campaign_journal_fence"]
            view=before[method]
            require(all(view[key]==fence[key] for key in
                ("journal_events","journal_sha256","charged_actions","decision_count")),
                "canonical head advanced beyond the owner-bound handoff")
            released_head=release["methods"][method]["canonical_head"]
            require(all(view[key]==released_head[key] for key in
                ("journal_events","journal_sha256","charged_actions","decision_count")),
                "canonical head differs from the source owner's release")
            rows[method]["owner_released_head"]=released_head
            source_remaining=view["source_caps"][method]-view["by_method"].get("source/"+method,0)
            total_remaining=view["total_action_cap"]-view["charged_actions"]
            required_source=rows[method]["required_source_actions"]
            required_evaluation=rows[method]["maximum_input_due_evaluation_actions"]
            require(source_remaining>=required_source and total_remaining>=required_source+required_evaluation,
                    "original residual physical allowance cannot fit full N6 and input-due evaluation")
            rows[method]["physical_admission"]=dict(remaining_source_actions=source_remaining,
                remaining_total_actions=total_remaining,maximum_new_source_actions=required_source,
                maximum_input_due_evaluation_actions=required_evaluation)
            # The qualified verifier only needs these two read-only attributes
            # when the caller already holds its canonical lock.
            locked_view=SimpleNamespace(root=roots[method],_require_open=lambda:None)
            assert_campaign_checkpoint_fence(roots[method],dict(campaign_journal_fence=fence),live_budget=locked_view)
        for ref in artifact_refs:bound(ref)
        after={method:ledger_view(roots[method],expected_snapshot=snapshots[method]) for method in METHODS}
        require(before==after,"canonical heads changed during read-only preflight")
    require(sha256(plan_path)==plan_sha256,"plan changed during preflight")
    require(not driver.torch.cuda.is_initialized(),"CPU preflight initialized CUDA")
    return dict(schema="robotwin_trained_pair_preflight_v1",mode="before",status="passed",
        created_at_utc=datetime.now(timezone.utc).isoformat(),runtime_commit=RUNTIME_COMMIT,lineage_commit=LINEAGE_COMMIT,
        plan=dict(path=str(plan_path),sha256=plan_sha256),parent_manifest=plan["parent_manifest"],
        owner_release=plan["owner_release"],compute_binding=plan["compute_binding"],methods=rows,
        ledger_before=before,ledger_after=after,cuda_visible_devices="",cuda_initialized=False,
        campaign_budget_constructed=False,source_mutations=False,
        allow_source_goal_boundary=allow_source_goal_boundary,
        policy="root launcher separately validates release/compute authorization and bounds this process externally")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--mode",choices=("before",),default="before")
    parser.add_argument("--allow-source-goal-boundary",action="store_true",
        help="Future production continuation only; permit completed source100 with remaining original evaluation")
    args=parser.parse_args(argv)
    require(not args.receipt.exists(),"receipt must be fresh")
    try:receipt=run_preflight(args.plan,args.plan_sha256,allow_source_goal_boundary=args.allow_source_goal_boundary)
    except BaseException as error:
        receipt=dict(schema="robotwin_trained_pair_preflight_v1",mode=args.mode,status="rejected",
            plan=dict(path=str(args.plan.resolve()),sha256=args.plan_sha256),
            error_type=type(error).__name__,error=str(error),source_mutations=False)
        args.receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        raise
    args.receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=receipt["status"],receipt=str(args.receipt.resolve()))))
    return 0


if __name__=="__main__":raise SystemExit(main())
