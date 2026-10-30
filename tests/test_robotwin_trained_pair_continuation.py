"""Prepared Linux CPU continuation/ownership tests; never run models on macOS."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest
import torch

from test_robotwin_trial_resume import recovery_runtime
from test_robotwin_trained_parallel import trained_boundary
from test_robotwin_trained_pair_audit import audited_round


diagnostics=Path(__file__).parents[1]/"scripts/diagnostics"
sys.path.insert(0,str(diagnostics))
import robotwin_trained_pair_preflight as preflight
spec=importlib.util.spec_from_file_location("continuation_tests",diagnostics/"robotwin_trained_pair_continuation.py")
continuation=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuation)
pytestmark=pytest.mark.skipif(sys.platform!="linux",reason="Allocated Linux CPU only")


def checkpoint(iteration,units=()):
    if not units:units=[f"stage:iteration:{number}:complete" for number in (10,25,50,100) if number<iteration]
    source=dict(handover_mic=600,pick_diverse_bottles=400,place_a2b_left=400,
        place_object_stand=400,place_phone_stand=400,scan_object=500)
    targets=dict(adjust_bottle=400,move_stapler_pad=400)
    value=dict(contract=dict(methods=["ppo"],noise_shape=(32,32),source_tasks=list(source),eval_tasks=list(targets),
            task_episode_limits=dict(source,**targets),eval_episodes=8,campaign=dict(source_validation_episodes=4),
            source_iterations=dict(evaluation_iterations=[10,25,50,100])),
        rng=dict(source="unchanged"),next_method_index=0,
        states=dict(ppo=dict(completed_source_iterations=iteration,completed_source_trials=6*iteration,
            source_steps=24*iteration,task_steps=dict(task=24*iteration),pending_rollouts=[],
            schedule=dict(trials=6*iteration,rng="unchanged"),learner=dict(
                policy=dict(weight=torch.tensor([1.])),optimizer=dict(moment=torch.tensor([.1])),
                sampler=dict(state=1),rng=dict(captured="source"),config=dict(reset_each_episode=False),updates=iteration,
                optimizer_steps=5*iteration,primitive_steps=24*iteration,chunks=24*iteration))),
        completed_units=list(units),history=[],stages={f"iteration:{number}":dict(status="complete",source_rng_restored=True)
            for number in (10,25,50,100) if f"stage:iteration:{number}:complete" in units})
    for number in (10,25,50,100):
        if f"stage:iteration:{number}:complete" in units:
            value["completed_units"].extend(row["unit"] for row in unit_plan(value,number))
    return value


def unit_plan(value,number):
    return preflight.evaluation_unit_plan(value,dict(workers=dict(ppo=dict(native_episodes=1))),number)


def history_result(row):
    value=dict(row["history_selector"],chunks=1,primitive_steps=1)
    if not row["native"]:
        value.update(episodes=[dict(seed=index,chunks=1,primitive_steps=1,budget_truncated=False)
            for index in range(row["episodes"])],primitive_steps=row["episodes"],budget_truncated=False)
    return value


def summary(value,*,trials,status="wall_budget_stop"):
    return dict(contract=json.loads(json.dumps(value["contract"])),process_source_trials=trials,status=status)


def test_long_authority_includes_initial_subslice_and_preserves_goal(tmp_path):
    path=tmp_path/"owner-evidence.json";path.write_text("{}")
    ref=continuation.binding(path)
    authority=dict(schema="robotwin_trained_pair_continuation_authority_v1",
        runtime_commit=continuation.RUNTIME,lineage_commit=continuation.LINEAGE,
        parent_manifest_sha256=continuation.PARENT_SHA,initial_admission_plan=ref,
        source_goal_rounds=100,original_total_compute_ceiling_seconds=None,
        qualification_capacity_reused=False,automatic_failed_retries=False,
        resources=dict(continuation.RESOURCES,scheduler_cap_seconds=46800),
        maximum_driver_seconds=43200,child_wall_seconds=3600,
        initial_subslice_seconds=2700,initial_subslice_included=True,
        authority=ref,prior_charge_and_physical_snapshot=ref)
    continuation.validate_authority(authority,ref)
    for field,value in (("source_goal_rounds",101),("initial_subslice_included",False),
                        ("qualification_capacity_reused",True),("maximum_driver_seconds",46800)):
        changed=deepcopy(authority);changed[field]=value
        with pytest.raises(ValueError):continuation.validate_authority(changed,ref)
    smaller=deepcopy(authority);smaller["maximum_driver_seconds"]=2700
    smaller["resources"]["scheduler_cap_seconds"]=2700
    with pytest.raises(ValueError):continuation.validate_authority(smaller,ref)
    smaller["resources"]["scheduler_cap_seconds"]=2880
    continuation.validate_authority(smaller,ref)


@pytest.mark.parametrize("start",[52,54])
def test_offset_source_rounds_advance_independently(start):
    before=checkpoint(start);after=checkpoint(start+1)
    assert continuation.classify_progress(before,after,summary(after,trials=6),"ppo")=="source_round"


def test_round100_requires_a_fresh_final_evaluation_child_then_retires():
    units=[f"stage:iteration:{n}:complete" for n in (10,25,50)]
    before=checkpoint(99,units);at100=checkpoint(100,units)
    assert continuation.classify_progress(before,at100,summary(at100,trials=6),"ppo")=="source_round"
    finished=deepcopy(at100);finished["completed_units"].append("stage:iteration:100:complete")
    finished["completed_units"].extend(row["unit"] for row in unit_plan(finished,100))
    finished["history"].extend(history_result(row) for row in unit_plan(finished,100))
    finished["stages"]["iteration:100"]=dict(status="complete",source_rng_restored=True)
    assert continuation.classify_progress(at100,finished,summary(finished,trials=0,status="complete"),"ppo")=="complete"
    too_far=checkpoint(101)
    with pytest.raises(ValueError,match="over-goal"):
        continuation.classify_progress(at100,too_far,summary(too_far,trials=6),"ppo")


def test_eval_only_requires_closed_unit_progress_and_exact_source_identity():
    before=checkpoint(50);after=deepcopy(before)
    unit=next(row for row in unit_plan(before,50) if row["phase"]=="target")
    after["completed_units"].append(unit["unit"])
    after["history"].append(history_result(unit))
    after["states"]["ppo"]["learner"]["rng"]=dict(captured="legitimate evaluation-global snapshot")
    assert continuation.classify_progress(before,after,summary(after,trials=0),"ppo")=="evaluation_progress"
    with pytest.raises(ValueError,match="no completed"):
        continuation.classify_progress(before,before,summary(before,trials=0),"ppo")
    changed=deepcopy(after);changed["states"]["ppo"]["learner"]["policy"]["weight"]+=1
    with pytest.raises(ValueError,match="altered trained source"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")
    changed=deepcopy(after);changed["states"]["ppo"]["pending_rollouts"]=["partial"]
    with pytest.raises(ValueError,match="partial"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")
    changed=deepcopy(after);changed["completed_units"][-1]="stage:iteration:50:target:invented"
    with pytest.raises(ValueError,match="unknown"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")
    changed=deepcopy(after);changed["history"][0]["episodes"].pop()
    with pytest.raises(ValueError,match="K4/K8"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")
    changed=deepcopy(after);changed["next_method_index"]=1
    with pytest.raises(ValueError,match="method cursor"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")


def test_fake_unit_growth_without_closed_raw_evidence_is_rejected(tmp_path):
    before=checkpoint(50);after=deepcopy(before);unit=unit_plan(before,50)[0]
    after["completed_units"].append(unit["unit"]);after["history"].append(history_result(unit))
    changes=continuation.evaluation_changes(before,after,"ppo")
    with pytest.raises(ValueError,match="raw corpus units"):continuation.verify_evaluation_raw(tmp_path,changes)


def test_legal_stage_only_closure_needs_all_units_and_completed_report():
    before=checkpoint(100,[f"stage:iteration:{number}:complete" for number in (10,25,50)])
    before["completed_units"].extend(row["unit"] for row in unit_plan(before,100))
    final=deepcopy(before);final["completed_units"].append("stage:iteration:100:complete")
    final["stages"]["iteration:100"]=dict(status="complete",source_rng_restored=True)
    assert continuation.classify_progress(before,final,summary(final,trials=0,status="complete"),"ppo")=="complete"
    final["stages"]["iteration:100"]["status"]="evaluating"
    with pytest.raises(ValueError,match="stage marker"):
        continuation.classify_progress(before,final,summary(final,trials=0,status="complete"),"ppo")


def test_cleanup_subset_keeps_other_live_arm_untouched(tmp_path,monkeypatch):
    module=sys.modules[continuation._Guardian.__module__]
    live={101:[dict(pid=101)],202:[dict(pid=202)]};signals=[]
    monkeypatch.setattr(module,"session_members",lambda identity:live[identity["pid"]])
    def signal(identity,number,**kwargs):
        assert guardian.sessions==[first,other]  # Fatal cleanup always retains the full registry.
        signals.append(identity["pid"]);live[identity["pid"]]=[]
    monkeypatch.setattr(module,"_signal",signal)
    class Process:
        def poll(self):return 0
    first=dict(name="ppo",identity=dict(pid=101),process=Process())
    other=dict(name="ppo_reset",identity=dict(pid=202),process=Process())
    guardian=object.__new__(continuation.ContinuationGuardian)
    guardian.sessions=[first,other];guardian.deadline=time.monotonic()+30
    guardian.output=tmp_path;guardian.failed=False;guardian.cleanup_number=0
    receipt=guardian.cleanup_subset([first])
    assert receipt["clean"] and signals==[101] and live[202]==[dict(pid=202)]
    assert guardian.sessions==[other] and receipt["sessions"][0]["name"]=="ppo"


def test_interrupted_subset_cleanup_retains_both_arms_for_fatal_cleanup(tmp_path,monkeypatch):
    module=sys.modules[continuation._Guardian.__module__]
    monkeypatch.setattr(module,"session_members",lambda identity:[dict(pid=identity["pid"])])
    def interrupted(*args,**kwargs):raise KeyboardInterrupt("test signal during subset cleanup")
    monkeypatch.setattr(module,"_signal",interrupted)
    first=dict(name="ppo",identity=dict(pid=101));other=dict(name="ppo_reset",identity=dict(pid=202))
    guardian=object.__new__(continuation.ContinuationGuardian)
    guardian.sessions=[first,other];guardian.deadline=time.monotonic()+30
    guardian.output=tmp_path;guardian.failed=False;guardian.cleanup_number=0
    with pytest.raises(KeyboardInterrupt):guardian.cleanup_subset([first])
    assert guardian.sessions==[first,other] and guardian.failed


def test_guardian_independent_respawns_and_single_initial_probe(tmp_path,monkeypatch):
    """No real child/model/GPU: exercise the supervisor's actual scheduling loop."""
    dummy=tmp_path/"dummy.json";dummy.write_text("{}")
    ref=continuation.binding(dummy);first=tmp_path/"first";worktree=tmp_path/"runtime"
    initial=dict(worktree=str(worktree),child_wall_seconds=2160,parent_manifest=ref,
        methods={method:dict(manifest=ref,checkpoint=ref,frozen_motor=ref,output=str(first/method),iteration=start)
            for method,start in (("ppo",99),("ppo_reset",97))})
    plan=tmp_path/"initial.json";plan.write_text(json.dumps(initial))
    output=tmp_path/"owned";output.mkdir();events=[]
    class Process:
        def __init__(self,delay):self.delay=delay
        def poll(self):
            self.delay-=1;return None if self.delay>=0 else 0
    class Guardian(continuation.ContinuationGuardian):
        def parent_alive(self):pass
        def bounded(self,request):
            events.append(request["purpose"])
            if request["purpose"]=="cpu_preflight":
                (self.output/"initial-preflight.json").write_text(json.dumps(dict(status="passed",cuda_initialized=False,
                    plan=continuation.binding(plan),allow_source_goal_boundary=True,methods={method:dict(
                        checkpoint=ref,completed_units=[],stage_statuses={},source_goal_complete=False,has_pending_eval=False)
                        for method in continuation.METHODS})))
            else:
                self.probe=dict(devices=[dict(cuda_total_bytes=96*(1<<30))]);self.probe_binding=dict(uuid="GPU-test")
        def launch(self,name,command,cwd,environment,log):
            directory=Path(log).parent;request=json.loads((directory/"input.json").read_text())
            method=request["method"];number=request["input"]["iteration"]
            if name.endswith("-audit"):
                kind="complete" if number==100 else "source_round";new=number if number==100 else number+1
                item=None if kind=="complete" else dict(manifest=ref,checkpoint=ref,frozen_motor=ref,iteration=new)
                (directory/"audit.json").write_text(json.dumps(dict(status="passed_owned_child",method=method,
                    kind=kind,iteration=new,next_input=item,final_checkpoint=ref,completed_units=[],stage_statuses={})))
            else:events.append((method,number));assert command[command.index("--max-source-trials-per-process")+1]=="6"
            identity=dict(pid=len(events)+10,sid=len(events)+10,pgid=len(events)+10,start_ticks=123,uid=0)
            entry=dict(name=name,identity=identity,process=Process(0 if method=="ppo" else 2));self.sessions.append(entry);return entry
        def cleanup_subset(self,entries):
            for entry in entries:self.sessions.remove(entry)
            return dict(clean=True)
        def sample_gpu(self,uuid):return dict(used_bytes=0,total_bytes=96*(1<<30))
    guardian=Guardian(dict(parent_pid=1,output=str(output),deadline=time.monotonic()+10000,cleanup_reserve=180))
    monkeypatch.setattr(continuation.time,"sleep",lambda seconds:None)
    result=guardian.continuation(dict(initial_plan=continuation.binding(plan),launcher=str(tmp_path),env={}))
    assert result["status"]=="complete"
    assert events.count("cpu_preflight")==events.count("gpu_probe")==1
    assert [event[1] for event in events if isinstance(event,tuple) and event[0]=="ppo"]==[99,100]
    assert [event[1] for event in events if isinstance(event,tuple) and event[0]=="ppo_reset"]==[97,98,99,100]
    assert result["arms"]["ppo"]["children"]==2 and result["arms"]["ppo_reset"]["children"]==4
    launch=json.loads((output/"ppo/slice-001/launch.json").read_text())
    assert launch["identity"]["start_ticks"]==123 and launch["runtime_commit"]==continuation.RUNTIME
    assert launch["prior_audit"]["sha256"]==continuation.sha(output/"ppo/slice-000/audit.json")
    for name in ("child-cleanup.json","audit-cleanup.json","audit-launch.json"):
        assert (output/"ppo/slice-000"/name).is_file()
    assert not (output/"ppo/slice-000/owned-cleanup.json").exists()


def test_initial_goal100_can_retire_one_arm_or_require_final_evaluation():
    initial=dict(methods={method:dict(checkpoint="bound") for method in continuation.METHODS})
    admitted=dict(methods={method:dict(source_goal_complete=True,has_pending_eval=method=="ppo_reset",
        checkpoint="bound",completed_units=[],stage_statuses={}) for method in continuation.METHODS})
    states=continuation.initial_arm_states(initial,admitted)
    assert states["ppo"]["complete"] and states["ppo"]["input"] is None
    assert not states["ppo_reset"]["complete"] and states["ppo_reset"]["input"]==initial["methods"]["ppo_reset"]


def test_fresh_deadline_check_after_other_arm_cleanup(monkeypatch):
    arms=dict(ppo=dict(entry=None),ppo_reset=dict(entry=object(),stage="audit",deadline=10.))
    monkeypatch.setattr(continuation.time,"monotonic",lambda:11.)
    with pytest.raises(TimeoutError,match="audit deadline"):continuation.check_active_deadlines(arms)


def test_initially_complete_pair_exits_before_probe_or_any_model(tmp_path):
    initial=dict(worktree=str(tmp_path),methods={method:dict(checkpoint="bound") for method in continuation.METHODS})
    plan=tmp_path/"plan.json";plan.write_text(json.dumps(initial));output=tmp_path/"output";output.mkdir()
    class Guardian(continuation.ContinuationGuardian):
        def bounded(self,request):
            assert request["purpose"]=="cpu_preflight"
            (self.output/"initial-preflight.json").write_text(json.dumps(dict(status="passed",cuda_initialized=False,
                plan=continuation.binding(plan),allow_source_goal_boundary=True,methods={method:dict(
                    source_goal_complete=True,has_pending_eval=False,checkpoint="bound",completed_units=[],stage_statuses={})
                    for method in continuation.METHODS})))
        def launch(self,*args,**kwargs):raise AssertionError("No model launch for completed goals")
        def sample_gpu(self,*args):raise AssertionError("No GPU query/probe for completed goals")
    guardian=Guardian(dict(parent_pid=1,output=str(output),deadline=time.monotonic()+10000,cleanup_reserve=180))
    result=guardian.continuation(dict(initial_plan=continuation.binding(plan),launcher=str(tmp_path),env={}))
    assert result["no_gpu_needed"] and result["probe"] is None
    assert all(row["complete"] and row["children"]==0 for row in result["arms"].values())


def test_continuation_reuses_real_one_round_c353_audit(audited_round,monkeypatch):
    case=audited_round;monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(continuation,"RUNTIME","b"*40);monkeypatch.setattr(continuation,"LINEAGE","a"*40)
    monkeypatch.setattr(continuation,"ledger_root",case["audit"].ledger_root)
    request=dict(method="ppo",input=case["plan"]["methods"]["ppo"],output=str(case["output"]),worktree=case["plan"]["worktree"])
    result=continuation.audit_child(request)
    assert result["kind"]=="source_round" and result["iteration"]==2
    assert result["audit"]["optimizer_step_increment"]==5 and result["audit"]["source_actions"]==24
    assert result["next_input"]["checkpoint"]==result["audit"]["checkpoint"]
    assert result["final_checkpoint"]==result["audit"]["checkpoint"]
    assert result["stage_statuses"]["iteration:1"]=="complete"
