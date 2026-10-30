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
spec=importlib.util.spec_from_file_location("continuation_tests",diagnostics/"robotwin_trained_pair_continuation.py")
continuation=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuation)
pytestmark=pytest.mark.skipif(sys.platform!="linux",reason="Allocated Linux CPU only")


def checkpoint(iteration,units=()):
    return dict(contract=dict(methods=["ppo"],noise_shape=(32,32)),rng=dict(source="unchanged"),
        states=dict(ppo=dict(completed_source_iterations=iteration,completed_source_trials=6*iteration,
            source_steps=24*iteration,task_steps=dict(task=24*iteration),pending_rollouts=[],
            schedule=dict(trials=6*iteration,rng="unchanged"),learner=dict(
                policy=dict(weight=torch.tensor([1.])),optimizer=dict(moment=torch.tensor([.1])),
                sampler=dict(state=1),config=dict(reset_each_episode=False),updates=iteration,
                optimizer_steps=5*iteration,primitive_steps=24*iteration,chunks=24*iteration))),
        completed_units=list(units),history=[],stages={f"iteration:{number}":dict(status="complete")
            for number in (10,25,50,100) if f"stage:iteration:{number}:complete" in units})


def summary(value,*,trials,status="wall_budget_stop"):
    return dict(contract=json.loads(json.dumps(value["contract"])),process_source_trials=trials,status=status)


@pytest.mark.parametrize("start",[52,54])
def test_offset_source_rounds_advance_independently(start):
    before=checkpoint(start);after=checkpoint(start+1)
    assert continuation.classify_progress(before,after,summary(after,trials=6),"ppo")=="source_round"


def test_round100_requires_a_fresh_final_evaluation_child_then_retires():
    units=[f"stage:iteration:{n}:complete" for n in (10,25,50)]
    before=checkpoint(99,units);at100=checkpoint(100,units)
    assert continuation.classify_progress(before,at100,summary(at100,trials=6),"ppo")=="source_round"
    finished=deepcopy(at100);finished["completed_units"].append("stage:iteration:100:complete")
    finished["stages"]["iteration:100"]=dict(status="complete")
    assert continuation.classify_progress(at100,finished,summary(finished,trials=0,status="complete"),"ppo")=="complete"
    too_far=checkpoint(101)
    with pytest.raises(ValueError,match="over-goal"):
        continuation.classify_progress(at100,too_far,summary(too_far,trials=6),"ppo")


def test_eval_only_requires_closed_unit_progress_and_exact_source_identity():
    before=checkpoint(50);after=deepcopy(before)
    after["completed_units"].append("stage:iteration:50:target:one-complete-K8")
    after["history"].append(dict(complete=True))
    assert continuation.classify_progress(before,after,summary(after,trials=0),"ppo")=="evaluation_progress"
    with pytest.raises(ValueError,match="no completed"):
        continuation.classify_progress(before,before,summary(before,trials=0),"ppo")
    changed=deepcopy(after);changed["states"]["ppo"]["learner"]["policy"]["weight"]+=1
    with pytest.raises(ValueError,match="altered trained source"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")
    changed=deepcopy(after);changed["states"]["ppo"]["pending_rollouts"]=["partial"]
    with pytest.raises(ValueError,match="partial"):
        continuation.classify_progress(before,changed,summary(changed,trials=0),"ppo")


def test_cleanup_subset_keeps_other_live_arm_untouched(tmp_path,monkeypatch):
    module=sys.modules[continuation._Guardian.__module__]
    live={101:[dict(pid=101)],202:[dict(pid=202)]};signals=[]
    monkeypatch.setattr(module,"session_members",lambda identity:live[identity["pid"]])
    def signal(identity,number,**kwargs):signals.append(identity["pid"]);live[identity["pid"]]=[]
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
                    plan=continuation.binding(plan))))
            else:
                self.probe=dict(devices=[dict(cuda_total_bytes=96*(1<<30))]);self.probe_binding=dict(uuid="GPU-test")
        def launch(self,name,command,cwd,environment,log):
            directory=Path(log).parent;request=json.loads((directory/"input.json").read_text())
            method=request["method"];number=request["input"]["iteration"]
            if name.endswith("-audit"):
                kind="complete" if number==100 else "source_round";new=number if number==100 else number+1
                item=None if kind=="complete" else dict(manifest=ref,checkpoint=ref,frozen_motor=ref,iteration=new)
                (directory/"audit.json").write_text(json.dumps(dict(status="passed_owned_child",method=method,
                    kind=kind,iteration=new,next_input=item)))
            else:events.append((method,number));assert command[command.index("--max-source-trials-per-process")+1]=="6"
            entry=dict(name=name,process=Process(0 if method=="ppo" else 2));self.sessions.append(entry);return entry
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


def test_continuation_reuses_real_one_round_c353_audit(audited_round,monkeypatch):
    case=audited_round;monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(continuation,"RUNTIME","b"*40);monkeypatch.setattr(continuation,"LINEAGE","a"*40)
    monkeypatch.setattr(continuation,"ledger_root",case["audit"].ledger_root)
    request=dict(method="ppo",input=case["plan"]["methods"]["ppo"],output=str(case["output"]),worktree=case["plan"]["worktree"])
    result=continuation.audit_child(request)
    assert result["kind"]=="source_round" and result["iteration"]==2
    assert result["audit"]["optimizer_step_increment"]==5 and result["audit"]["source_actions"]==24
    assert result["next_input"]["checkpoint"]==result["audit"]["checkpoint"]
