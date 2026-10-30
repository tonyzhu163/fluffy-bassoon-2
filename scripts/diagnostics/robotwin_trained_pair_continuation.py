#!/usr/bin/env python3
"""Future owned continuation of the two original100-round RoboTwin streams.

One fresh admission plan, one initial idle probe, and independent fresh570
children. Every child still loads its motor and performs native parity. No
allocation, source-owner control, fresh learner, or automatic failed retry.
An already complete arm needs an OWNER-ISSUED current whole-boundary manifest
for its final post-evaluation checkpoint/head.570 produces no final manifest;
never reuse the earlier source100 pre-evaluation manifest after its head moved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time

from robotwin_trained_pair import (RUNTIME,LINEAGE,PARENT_SHA,METHODS,ROOT,PYTHON,RESOURCES,
    HEAD_KEYS,atom,sha,read_ref,check_ref,require,validate_plan,build_commands,pinned_worktree,
    ledger_root,runtime_environment)
from robotwin_trained_pair_runtime import (_Guardian,_control_line,_write,session_members,
    TrainedPairRuntime,validate_probe)


MAX_ALLOCATION=46800
MAX_DRIVER=43200
CHILD_WALL=3600
MINIMUM_CHILD_REMAINING=1200


def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def initial_arm_states(initial,admitted):
    states={}
    for method in METHODS:
        row=admitted["methods"][method]
        complete=row["source_goal_complete"] and not row["has_pending_eval"]
        states[method]=dict(input=None if complete else initial["methods"][method],number=0,entry=None,
            complete=complete,parked=False,final_checkpoint=row["checkpoint"],
            completed_units=row["completed_units"],stage_statuses=row["stage_statuses"])
    return states


def arm_result(state):
    return dict(complete=state["complete"],children=state["number"],next_input=state["input"],
        last_audit=state.get("last_audit"),final_checkpoint=state.get("final_checkpoint"),
        completed_units=state.get("completed_units",[]),stage_statuses=state.get("stage_statuses",{}))


def check_active_deadlines(arms):
    now=time.monotonic()
    for state in arms.values():
        if state.get("entry") is not None and now>=state["deadline"]:
            raise TimeoutError("Owned "+state["stage"]+" deadline; no automatic retry")


def validate_authority(value,initial_ref):
    require(value["schema"]=="robotwin_trained_pair_continuation_authority_v1"
        and value["runtime_commit"]==RUNTIME and value["lineage_commit"]==LINEAGE
        and value["parent_manifest_sha256"]==PARENT_SHA and value["initial_admission_plan"]==initial_ref,
        "long continuation authority differs from its fresh initial handoff")
    require(value["source_goal_rounds"]==100 and value["original_total_compute_ceiling_seconds"] is None
        and value["qualification_capacity_reused"] is False and value["automatic_failed_retries"] is False,
        "original goal-completion authority required")
    resources=value["resources"]
    require({key:resources[key] for key in ("gpus","cpus","memory_gib","shared_memory_gib")}
        =={key:RESOURCES[key] for key in ("gpus","cpus","memory_gib","shared_memory_gib")}
        and type(resources["scheduler_cap_seconds"]) is int
        and 2880<=resources["scheduler_cap_seconds"]<=MAX_ALLOCATION,
        "continuation resource/operational ceiling differs")
    require(type(value["maximum_driver_seconds"]) is int
        and 2700<=value["maximum_driver_seconds"]<=min(MAX_DRIVER,resources["scheduler_cap_seconds"]-180)
        and value["child_wall_seconds"]==CHILD_WALL
        and value["initial_subslice_seconds"]==2700 and value["initial_subslice_included"] is True,
        "initial2700s subunit must be INCLUDED in the long operational envelope")
    for name in ("authority","prior_charge_and_physical_snapshot"):
        path=check_ref(value[name]);require(sha(path)==value[name]["sha256"],"continuation authority/accounting changed")


def classify_progress(before,final,summary,method):
    """Only whole updated rounds or genuine closed evaluation progress continue."""
    old=before["states"][method];new=final["states"][method]
    require(final["contract"]==before["contract"]
        and json.loads(json.dumps(final["contract"]))==summary["contract"],"scientific contract changed")
    require(not new.get("pending_rollouts") and not new.get("selected_trial_recovery")
        and new["schedule"]["trials"]==new["completed_source_trials"]==6*new["completed_source_iterations"]
        and new["learner"]["updates"]==new["completed_source_iterations"]
        and new["learner"]["optimizer_steps"]==5*new["completed_source_iterations"]
        and 0<new["completed_source_iterations"]<=100,"partial/selected/over-goal source boundary")
    units=set(final["completed_units"]);old_units=set(before["completed_units"])
    require(old_units<=units and final["history"][:len(before["history"])]==before["history"],
        "completed evaluation state regressed")
    require(final["next_method_index"]==before["next_method_index"],"coordinator method cursor changed")
    for key,stage in before["stages"].items():
        if stage.get("status")=="complete":require(final["stages"].get(key)==stage,"completed stage changed")
    delta=new["completed_source_iterations"]-old["completed_source_iterations"]
    require(delta in (0,1),"child crossed more than one source update")
    if delta:
        require(summary["process_source_trials"]==6,"updated round omitted complete N6 trials")
        return "source_round"
    require(summary["process_source_trials"]==0
        and new["source_steps"]==old["source_steps"]
        and new["task_steps"]==old["task_steps"],"evaluation child advanced source actions")
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import artifact_digest
    for key in ("policy","optimizer","sampler","config","updates","optimizer_steps","primitive_steps","chunks"):
        require(artifact_digest(new["learner"][key])==artifact_digest(old["learner"][key]),
            "evaluation altered trained source learner: "+key)
    require(artifact_digest(new["schedule"])==artifact_digest(old["schedule"])
        and artifact_digest(final["rng"])==artifact_digest(before["rng"]),"evaluation consumed source RNG/task stream")
    evaluation_changes(before,final,method)
    if summary["status"]=="complete":
        require(new["completed_source_iterations"]==100 and all(
            f"stage:iteration:{number}:complete" in units for number in (10,25,50,100)),
            "final evaluation did not complete original100-round protocol")
        return "complete"
    require(summary["status"]=="wall_budget_stop" and units>old_units,
        "zero-source child made no completed evaluation-unit progress")
    return "evaluation_progress"


def evaluation_changes(before,final,method):
    """Match only original completed units to their newly committed results."""
    from robotwin_trained_pair_preflight import evaluation_unit_plan
    cursor=before["states"][method]["completed_source_iterations"]
    parent=dict(workers={method:dict(native_episodes=1 if method=="ppo" else 0)})
    plans={number:evaluation_unit_plan(before,parent,number)
        for number in before["contract"]["source_iterations"]["evaluation_iterations"]}
    units=set(final["completed_units"]);old=set(before["completed_units"])
    allowed={row["unit"]:row for row in plans.get(cursor,[])}
    marker=f"stage:iteration:{cursor}:complete"
    require(units-old<=set(allowed)|{marker},"unknown or out-of-order evaluation unit")
    for number,plan in plans.items():
        if f"stage:iteration:{number}:complete" in units:
            stage=final["stages"].get(f"iteration:{number}",{})
            require(stage.get("status")=="complete" and stage.get("source_rng_restored") is True
                and {row["unit"] for row in plan}<=units,"stage marker lacks completed original units/report")
    histories=final["history"][len(before["history"]):];changes=[]
    new_physical=units-old-{marker}
    require(len(histories)==len(new_physical),"evaluation unit/history count differs")
    for unit in sorted(new_physical):
        row=allowed[unit];selector=row["history_selector"]
        matches=[history for history in histories if all(history.get(key)==value for key,value in selector.items())]
        require(len(matches)==1,"completed evaluation unit has no unique result history")
        history=matches[0]
        if row["native"]:
            require(history.get("chunks",0)>0 and 0<=history["primitive_steps"]<=row["maximum_actions"],
                "native evaluation result is not a complete physical episode")
        else:
            episodes=history.get("episodes",[])
            require(len(episodes)==row["episodes"] and not history.get("budget_truncated",False)
                and all(episode.get("chunks",0)>0 and not episode.get("budget_truncated",False) for episode in episodes)
                and history["primitive_steps"]==sum(episode["primitive_steps"] for episode in episodes)
                and 0<=history["primitive_steps"]<=row["maximum_actions"],"evaluation result lost a whole K4/K8 unit")
        changes.append(dict(row,history=history))
    return changes


def verify_evaluation_raw(output,changes):
    """Whole closed corpus episodes must exactly cover new result units."""
    grouped={}
    for path in (Path(output)/"corpus/episodes").glob("*/reset.json"):
        reset=json.loads(path.read_text());manifest=json.loads((path.parent/"manifest.json").read_text())
        if manifest["completed_chunks"]:
            context=reset["context"];grouped.setdefault(context["trial_id"],[]).append((reset,manifest))
    require(set(grouped)=={row["trial_id"] for row in changes},"new raw corpus units differ from completed evaluation results")
    for row in changes:
        episodes=grouped[row["trial_id"]]
        require(len(episodes)==row["episodes"] and all(reset["reset_info"]["task"]==row["task"]
            and reset["context"]["phase"]==row["phase"]
            and reset["context"]["method"]==row["history_selector"]["method"] for reset,_ in episodes),
            "raw evaluation unit task/method/episode count differs")
        history=row["history"]
        require(sum(manifest["executed_primitive_steps"] for _,manifest in episodes)==history["primitive_steps"],
            "raw/result evaluation action count differs")
        if not row["native"]:
            require(sorted((reset["reset_info"]["requested_seed"],manifest["executed_primitive_steps"])
                for reset,manifest in episodes)==sorted((episode["seed"],episode["primitive_steps"])
                for episode in history["episodes"]),"raw evaluation episode seed/count provenance differs")


def audit_child(request):
    """CPU process, externally bounded; never constructs a physical ledger."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","child audit must hide CUDA")
    import torch
    from robotwin_trained_pair_audit import audit_arm,audit_corpus
    from robotwin_trained_pair_preflight import ledger_view
    from fluffy_bassoon.meta_rl.collection_admission import assert_resume_allowed,assert_campaign_checkpoint_fence
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import load_trial_continuation,artifact_digest
    from fluffy_bassoon.meta_rl.robotwin_trained_parallel import validate_trained_parallel_boundary
    method=request["method"];require(method in METHODS,"unknown source arm")
    arm=request["input"];output=Path(request["output"])
    old_manifest=read_ref(arm["manifest"]);read_ref(arm["frozen_motor"])
    require(old_manifest["checkpoint"]==arm["checkpoint"] and sha(arm["checkpoint"]["path"])==arm["checkpoint"]["sha256"],
        "input checkpoint changed")
    before=torch.load(arm["checkpoint"]["path"],map_location="cpu",weights_only=False)
    summary=json.loads((output/"summary.json").read_text())
    require(summary["status"] in ("wall_budget_stop","complete"),"failed child cannot resume automatically")
    path=output/"coordinator-recovery.pt";final=torch.load(path,map_location="cpu",weights_only=False)
    kind=classify_progress(before,final,summary,method)
    plan=dict(worktree=request["worktree"],methods={method:dict(arm,output=str(output))})
    if kind=="source_round":
        normal=audit_arm(plan,method)  # Qualifiedc353 full corpus/replay/ACK/case/phase audit.
        next_ref=normal["next_manifest"]
    else:
        migration=json.loads((output/"execution-migration.json").read_text())
        require(migration["execution_source_commit"]==RUNTIME and migration["lineage_source_commit"]==LINEAGE
            and migration["manifest_sha256"]==arm["manifest"]["sha256"]
            and migration["checkpoint"]==arm["checkpoint"]
            and Path(migration["manifest_path"]).resolve()==Path(arm["manifest"]["path"]).resolve(),
            "evaluation migration ancestry differs")
        assert_resume_allowed(path,checkpoint=final)
        assert_campaign_checkpoint_fence(ledger_root(method),final)
        require(not list((output/"production-vector").glob("group-*.json")),"evaluation child dispatched a source group")
        old_ledger=read_ref(old_manifest["ledger_snapshot"])
        current=json.loads((ledger_root(method)/"snapshot.json").read_text())
        require(current["header"]==old_ledger["header"] and current["pending_actions"]==current["uncertain_actions"]==0
            and not current["unresolved_decisions"],"evaluation left unresolved physical accounting")
        maximum=dict(source=0,source_validation=10800,target=19200 if method=="ppo" else 6400,
            native_before_target=800 if method=="ppo" else 0,native_after=800 if method=="ppo" else 0)
        for phase in set(current["by_phase"])|set(old_ledger["by_phase"]):
            delta=current["by_phase"].get(phase,0)-old_ledger["by_phase"].get(phase,0)
            require(0<=delta<=maximum.get(phase,0),"evaluation borrowed another original phase")
        corpus=audit_corpus(output/"corpus")
        verify_evaluation_raw(output,evaluation_changes(before,final,method))
        require(corpus["actual_actions"]==current["actual_actions"]-old_ledger["actual_actions"]
            and corpus["source_actions"]==0,"evaluation raw/physical action totals differ")
        next_ref=summary.get("continuation_manifest")
        if kind!="complete":
            require(next_ref is not None,"evaluation progress omitted its570-produced handoff")
            manifest=read_ref(next_ref)
            require(Path(manifest["checkpoint"]["path"]).resolve()==path.resolve(),"evaluation checkpoint escaped output")
            continuation=load_trial_continuation(next_ref["path"],next_ref["sha256"],checkpoint_path=path,
                checkpoint=final,execution_commit=RUNTIME,lineage_commit=LINEAGE)
            validate_trained_parallel_boundary(continuation,final,manifest_path=next_ref["path"],campaign_root=ledger_root(method))
        normal=dict(corpus=corpus,source_actions=0,actual_actions=corpus["actual_actions"])
    frozen=json.loads((output/"frozen-motor.json").read_text());old_motor=read_ref(arm["frozen_motor"])
    require(frozen["unchanged"] is True and frozen["before"]==frozen["after"]==old_motor["before"]==old_motor["after"],
        "frozen motor changed")
    expected=(read_ref(read_ref(next_ref)["ledger_snapshot"]) if next_ref else
              json.loads((ledger_root(method)/"snapshot.json").read_text()))
    with (ledger_root(method)/"campaign.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        view=ledger_view(ledger_root(method),expected_snapshot=expected)
        require(all(view[key]==final["campaign_journal_fence"][key] for key in HEAD_KEYS),
            "canonical head advanced after the owned child's final checkpoint")
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    return dict(status="passed_owned_child",kind=kind,method=method,output=str(output),
        iteration=final["states"][method]["completed_source_iterations"],audit=normal,canonical_view=view,
        final_checkpoint=binding(path),completed_units=final["completed_units"],
        rng_snapshots=dict(coordinator_before=artifact_digest(before["rng"]),
            coordinator_after=artifact_digest(final["rng"]),
            learner_global_before=artifact_digest(before["states"][method]["learner"].get("rng")),
            learner_global_after=artifact_digest(final["states"][method]["learner"].get("rng")),
            interpretation="Coordinator RNG is authoritative; learner rng captures global state and may differ during partial evaluation"),
        stage_statuses={key:value["status"] for key,value in final["stages"].items()},
        next_input=None if kind=="complete" else dict(manifest=next_ref,
            checkpoint=read_ref(next_ref)["checkpoint"],frozen_motor=binding(output/"frozen-motor.json")))


class ContinuationGuardian(_Guardian):
    """One independent guardian owns every fresh child and bounded CPU helper."""
    def launch(self,name,command,cwd,environment,log,*,registered=None):
        # A Python handler that raises between Popen and registry insertion
        # could strand a detached child. Defer only this guardian's handlers;
        # unlike pthread_sigmask, this does not block signals in the exec child.
        handlers={};pending=[]
        def defer(signum,frame):
            if not pending:pending.append((signum,frame))
        try:
            for number in (signal.SIGTERM,signal.SIGINT):
                handlers[number]=signal.getsignal(number)
                signal.signal(number,defer)
            entry=super().launch(name,command,cwd,environment,log)
            if registered is not None:registered(entry)
        finally:
            for number,handler in handlers.items():signal.signal(number,handler)
        if pending:
            number,frame=pending[0];handler=handlers[number]
            if handler!=signal.SIG_IGN:
                if callable(handler):handler(number,frame)
                raise RuntimeError("Continuation guardian terminated during owned spawn")
        return entry

    def cleanup_subset(self,entries):
        selected={id(entry) for entry in entries}
        require(selected<={id(entry) for entry in self.sessions},"cleanup includes an unowned entry")
        # Keep the canonical registry intact even if TERM/INT interrupts any
        # instruction below. A separate view applies the passed cleanup logic
        # only to the selected SIDs; fatal outer cleanup still sees BOTH arms.
        view=object.__new__(_Guardian)
        view.sessions=list(entries);view.deadline=self.deadline;view.output=self.output
        self.cleanup_number+=1  # Reserve even if a fatal signal interrupts the proxy.
        view.failed=False;view.cleanup_number=self.cleanup_number-1
        try:result=view.cleanup(graceful=False)
        except BaseException:
            self.failed=True;raise
        finally:self.cleanup_number=max(self.cleanup_number,view.cleanup_number)
        self.sessions=[entry for entry in self.sessions if id(entry) not in selected]
        return result

    def continuation(self,request):
        initial=read_ref(request["initial_plan"]);launcher=Path(request["launcher"])
        environment=dict(request["env"]);worktree=Path(initial["worktree"])
        preflight=self.output/"initial-preflight.json"
        self.bounded(dict(purpose="cpu_preflight",command=[str(PYTHON),str(launcher/"scripts/diagnostics/robotwin_trained_pair_preflight.py"),
            "--plan",request["initial_plan"]["path"],"--plan-sha256",request["initial_plan"]["sha256"],
            "--receipt",str(preflight),"--mode","before","--allow-source-goal-boundary"],cwd=str(launcher),env=environment,
            log_path=str(self.output/"initial-preflight.log"),timeout_seconds=90))
        admitted=json.loads(preflight.read_text());require(admitted["status"]=="passed" and not admitted["cuda_initialized"]
            and admitted["plan"]["sha256"]==request["initial_plan"]["sha256"]
            and admitted["allow_source_goal_boundary"] is True,
            "full initial CPU admission failed")
        arms=initial_arm_states(initial,admitted)
        for method,state in arms.items():
            if state["complete"]:atom(self.output/(method+"-latest.json"),arm_result(state))
        if all(state["complete"] for state in arms.values()):
            return dict(status="complete",arms={method:arm_result(state) for method,state in arms.items()},
                probe=None,final_gpu=None,automatic_failed_retries=0,no_gpu_needed=True)
        self.probe_path=self.output/"gpu-probe.json"
        self.bounded(dict(purpose="gpu_probe",command=[str(PYTHON),str(worktree/"scripts/[train]gpu_probe.py"),
            "--expected-count","1","--purpose","training","--baige-container","--output",str(self.probe_path)],
            cwd=str(worktree),env=environment,log_path=str(self.output/"gpu-probe.log"),timeout_seconds=90))
        environment.update(LYCO_BAIGE_GPU_MAP=str(self.probe_path),LYCO_BAIGE_ISOLATED_UUID=self.probe_binding["uuid"])
        require(90*(1<<30)<=self.probe["devices"][0]["cuda_total_bytes"]<=100*(1<<30),"reviewed96GB GPU required")
        def start_child(method,state):
            if state["complete"]:return
            check_active_deadlines(arms)
            if self.work_deadline-time.monotonic()<MINIMUM_CHILD_REMAINING:
                state["parked"]=True;return
            plan=deepcopy(initial)
            output=(Path(initial["methods"][method]["output"]) if state["number"]==0 else
                    self.output/method/f"slice-{state['number']:03d}"/"experiment")
            require(not output.exists(),"continuation child output already exists")
            plan["child_wall_seconds"]=(initial["child_wall_seconds"] if state["number"]==0 else
                min(CHILD_WALL,int(self.work_deadline-time.monotonic()-210)))
            plan["methods"][method]=dict(state["input"],output=str(output))
            command=build_commands(plan)[method]
            directory=self.output/method/f"slice-{state['number']:03d}";directory.mkdir(parents=True,exist_ok=True)
            state["request"]=dict(method=method,input=state["input"],output=str(output),worktree=str(worktree))
            atom(directory/"input.json",state["request"])
            env=dict(environment,FB2_ROBOTWIN_RENDER_RECORDS=str(output/"renderer-bindings"))
            deadline=min(self.work_deadline-90,time.monotonic()+plan["child_wall_seconds"]+120)
            prior=state.get("last_audit_ref")
            if prior:require(sha(prior["path"])==prior["sha256"],"prior owned audit changed before respawn")
            launch=dict(status="dispatch_pending",runtime_commit=RUNTIME,method=method,slice=state["number"],
                command=command,input=state["input"],input_receipt=binding(directory/"input.json"),
                initial_admission=binding(preflight),prior_audit=prior,authority=request.get("authority"),
                wall_seconds=plan["child_wall_seconds"],deadline=deadline,identity=None)
            atom(directory/"launch.json",launch)  # Durable exact intent precedes Popen.
            check_active_deadlines(arms)
            def record_child(entry):
                launch.update(status="running",identity=entry["identity"])
                atom(directory/"launch.json",launch)
            entry=self.launch(f"{method}-slice-{state['number']:03d}",command,str(worktree),env,
                directory/"console.log",registered=record_child)
            state.update(entry=entry,stage="child",deadline=deadline,directory=directory)
        for method,state in arms.items():start_child(method,state)
        with (self.output/"gpu-memory.jsonl").open("x",buffering=1) as trace:
            while any(state["entry"] is not None for state in arms.values()):
                self.parent_alive()
                for method,state in arms.items():
                    entry=state["entry"]
                    if entry is None:continue
                    code=entry["process"].poll()
                    if time.monotonic()>=state["deadline"]:raise TimeoutError("Owned "+state["stage"]+" deadline; no automatic retry")
                    if code is None:continue
                    require(code==0,"owned child/audit failed; no automatic retry")
                    cleanup=self.cleanup_subset([entry])
                    atom(state["directory"]/("child-cleanup.json" if state["stage"]=="child" else "audit-cleanup.json"),cleanup)
                    state["entry"]=None
                    check_active_deadlines(arms)
                    if state["stage"]=="child":
                        require(time.monotonic()+90<=self.work_deadline,"no complete CPU audit window remains")
                        state["request"].update(owned_launch=binding(state["directory"]/"launch.json"),
                            owned_cleanup=binding(state["directory"]/"child-cleanup.json"))
                        atom(state["directory"]/"audit-input.json",state["request"])
                        receipt=state["directory"]/"audit.json"
                        command=[str(PYTHON),str(Path(__file__).resolve()),"--audit-child",
                            str(state["directory"]/"audit-input.json"),"--receipt",str(receipt)]
                        check_active_deadlines(arms)
                        def record_audit(entry):
                            atom(state["directory"]/"audit-launch.json",dict(command=command,identity=entry["identity"],
                                runtime_commit=RUNTIME,source_launch=state["request"]["owned_launch"]))
                        audit_entry=self.launch(f"{method}-slice-{state['number']:03d}-audit",command,str(launcher),
                            dict(environment,CUDA_VISIBLE_DEVICES=""),state["directory"]/"audit.log",registered=record_audit)
                        state.update(entry=audit_entry,
                            stage="audit",deadline=min(self.work_deadline,time.monotonic()+90))
                    else:
                        result=json.loads((state["directory"]/"audit.json").read_text())
                        require(result["status"]=="passed_owned_child" and result["method"]==method,"child audit identity differs")
                        state.update(entry=None,input=result["next_input"],number=state["number"]+1,
                            complete=result["kind"]=="complete",last_audit=result,
                            last_audit_ref=binding(state["directory"]/"audit.json"),
                            final_checkpoint=result["final_checkpoint"],completed_units=result["completed_units"],
                            stage_statuses=result["stage_statuses"])
                        atom(self.output/(method+"-latest.json"),arm_result(state))
                        if not state["complete"]:start_child(method,state)
                sample=self.sample_gpu(self.probe_binding["uuid"]);trace.write(json.dumps(sample)+"\n")
                require(sample["used_bytes"]<=min(85*(1<<30),.9*sample["total_bytes"]),"memory headroom exceeded")
                time.sleep(.5)
        final_gpu=self.sample_gpu(self.probe_binding["uuid"]);until=min(self.deadline-1,time.monotonic()+10)
        while final_gpu["used_bytes"]>256*(1<<20) and time.monotonic()<until:
            time.sleep(.2);final_gpu=self.sample_gpu(self.probe_binding["uuid"])
        require(final_gpu["used_bytes"]<=256*(1<<20),"owned GPU contexts remain after all children exited")
        return dict(status="complete" if all(state["complete"] for state in arms.values()) else "allocation_boundary",
            arms={method:arm_result(state) for method,state in arms.items()},probe=self.probe_binding,
            final_gpu=final_gpu,automatic_failed_retries=0,fresh_motor_load_and_native_parity_per_child=True)


def guardian_main():
    line=_control_line(time.monotonic()+10,os.getppid())
    if line is None:return
    guardian=ContinuationGuardian(json.loads(line))
    def stop(*args):raise RuntimeError("Continuation guardian terminated")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        line=_control_line(guardian.deadline,guardian.parent)
        if line is None:return
        result=guardian.continuation(json.loads(line));guardian.cleanup(graceful=False)
        print(json.dumps(dict(ok=True,result=result)),flush=True)
    except BaseException as error:
        try:cleanup=guardian.cleanup()
        except BaseException as cleanup_error:cleanup=dict(clean=False,error=str(cleanup_error))
        failure=dict(ok=False,error=str(error),cleanup=cleanup)
        _write(guardian.output/"continuation-failure.json",failure);print(json.dumps(failure),flush=True)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        guardian.cleanup()


class ContinuationRuntime(TrainedPairRuntime):
    def __init__(self,worktree,output,maximum_seconds):
        require(sys.platform=="linux" and 2700<=maximum_seconds<=MAX_DRIVER,"bounded allocated Linux runtime required")
        self.worktree=Path(worktree).resolve();self.output=Path(output).resolve()
        self.started=time.monotonic();self.deadline=self.started+maximum_seconds
        self.collection_deadline=self.deadline-180;self.failed=False;self.closed=False
        self._errors=(self.output/"guardian-stderr.log").open("x")
        self.guardian=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--guardian"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._errors,text=True,start_new_session=True,close_fds=True)
        self.guardian.stdin.write(json.dumps(dict(parent_pid=os.getpid(),output=str(self.output),
            deadline=self.deadline,cleanup_reserve=180))+"\n");self.guardian.stdin.flush()
        atom(self.output/"runtime-owner.json",dict(parent_pid=os.getpid(),guardian_pid=self.guardian.pid,
            deadline=self.deadline,collection_deadline=self.collection_deadline,environment_saved=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-plan",type=Path,required=True);parser.add_argument("--initial-plan-sha256",required=True)
    parser.add_argument("--authority",type=Path,required=True);parser.add_argument("--authority-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--launcher-commit",required=True)
    args=parser.parse_args(argv);initial_ref=dict(path=str(args.initial_plan.resolve()),sha256=args.initial_plan_sha256)
    initial=read_ref(initial_ref);validate_plan(initial,read_evidence=False)
    authority_ref=dict(path=str(args.authority.resolve()),sha256=args.authority_sha256)
    authority=read_ref(authority_ref);validate_authority(authority,initial_ref)
    require(sys.platform=="linux" and socket.gethostname().startswith("job-"),"exclusive future allocation required")
    launcher=Path(__file__).resolve().parents[2];pinned_worktree(launcher,args.launcher_commit);pinned_worktree(initial["worktree"],RUNTIME)
    require(args.output.is_absolute() and args.output.parent==ROOT/"outputs"
        and args.output.name.startswith("robotwin-trained-continuation-") and not args.output.exists(),"fresh continuation output required")
    require(not Path(initial["output"]).exists(),"initial admission plan is already spent; issue a NEW owner release/plan")
    args.output.mkdir();atom(args.output/"initial-plan.json",initial);atom(args.output/"authority.json",authority)
    result=None
    try:
        with ContinuationRuntime(initial["worktree"],args.output,authority["maximum_driver_seconds"]) as runtime:
            result=runtime._request(dict(initial_plan=initial_ref,authority=authority_ref,launcher=str(launcher),
                env=runtime_environment(initial["worktree"])))
        return 0
    finally:
        atom(args.output/"continuation-result.json",dict(result=result,runtime_commit=RUNTIME,authority=authority_ref,
            no_automatic_failed_retry=True,scheduler_accounting_requires_terminal_owner_receipt=True))


if __name__=="__main__":
    if sys.argv[1:]==["--guardian"]:guardian_main()
    elif len(sys.argv)>2 and sys.argv[1]=="--audit-child":
        parser=argparse.ArgumentParser();parser.add_argument("--audit-child",type=Path,required=True)
        parser.add_argument("--receipt",type=Path,required=True);args=parser.parse_args()
        require(not args.receipt.exists(),"fresh child audit receipt required")
        request=json.loads(args.audit_child.read_text())
        require("owned_launch" in request and "owned_cleanup" in request,"owned process receipts required")
        launch=read_ref(request["owned_launch"]);cleanup=read_ref(request["owned_cleanup"])
        require(launch["status"]=="running" and launch["runtime_commit"]==RUNTIME and cleanup["clean"] is True
            and any(row["identity"]==launch["identity"] and row["returncode"]==0 for row in cleanup["sessions"]),
            "owned source session was not cleanly reaped")
        atom(args.receipt,audit_child(request))
    else:raise SystemExit(main())
