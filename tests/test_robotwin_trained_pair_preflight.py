"""Prepared Linux CPU checks; this file must not execute models on macOS."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_robotwin_trial_resume import recovery_runtime,common_args,driver
from fluffy_bassoon.meta_rl.robotwin_trial_resume import file_sha256


path=Path(__file__).parents[1]/"scripts/diagnostics/robotwin_trained_pair_preflight.py"
spec=importlib.util.spec_from_file_location("trained_pair_preflight_tests",path)
preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)


def bind(path):return dict(path=str(path.resolve()),sha256=file_sha256(path))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return bind(path)


@pytest.fixture
def prepared_pair(tmp_path,recovery_runtime,monkeypatch):
    import fluffy_bassoon.rl.robotwin_noise as runtime
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    recovery_runtime["fail_reset"]=None
    monkeypatch.setattr(runtime.RobotwinNoiseEnv,"native_horizon",32)
    monkeypatch.setattr(runtime,"load_benchmark_episode_limits",lambda *args:
        {task:1 for task in driver.SOURCE_TASKS+driver.HELDOUT_TASKS})
    monkeypatch.setattr(driver,"validate_capture_qualification",lambda *args,**kwargs:
        dict(frozen_fingerprint="fixed-motor"))
    monkeypatch.setattr(driver,"validate_loaded_capture_runtime",lambda *args,**kwargs:None)
    monkeypatch.setattr(driver.torch.cuda,"is_initialized",lambda:False,raising=False)
    methods={};source_args={};workers={};heads={};worktree=tmp_path/"runtime";worktree.mkdir()
    for method in preflight.METHODS:
        arm=tmp_path/method;arm.mkdir();output=arm/"old";qualification=arm/"qualification.json"
        qualification.write_text("{}")
        args=common_args(arm,method,iterations=100,limit=6)+[
            "--benchmark-episode-limits","--source-eval-iterations","10","25","50","100",
            "--ppo-max-update-kl",".1","--learning-rate","1e-6","--final-observation-only",
            "--capture-qualification",str(qualification)]
        args.remove("--stage-source-validation-only")
        args[args.index("--source-steps")+1]="1100000"
        cap=1226400 if method=="ppo" else 1168800
        args +=["--campaign-max-actions",str(cap)]
        assert driver.main(args+["--output",str(output)])==0
        manifest=json.loads((output/"continuation-manifest.json").read_text())
        manifest["execution_source_commit"]="b"*40
        manifest_ref=write(arm/"new-owner-manifest.json",manifest)
        methods[method]=dict(manifest=manifest_ref,checkpoint=bind(output/"coordinator-recovery.pt"),
            frozen_motor=bind(output/"frozen-motor.json"),output=str(tmp_path/"pair"/method))
        workers[method]=dict(ledger_root=str(arm/"campaign"),native_episodes=0,
            campaign_id="robotwin-source-1m",source_safety_actions=1100000,maximum_actions=cap)
        snapshot=json.loads((arm/"campaign/snapshot.json").read_text())
        heads[method]={key:snapshot[key] for key in ("journal_events","journal_sha256","charged_actions","decision_count")}
        source_args[method]=args
    parent_ref=write(tmp_path/"original-parent.json",dict(format="robotwin_parallel_whole_trial_campaign_v1",
        source_commit="a"*40,source_iterations=100,source_eval_iterations=[10,25,50,100],
        trials_per_iteration=6,train_episodes=4,workers=workers))
    # Only synthetic test evidence substitutes these pins. Production CLI has
    # no override for the fixed original parent/runtime/DB lineage constants.
    monkeypatch.setattr(preflight,"RUNTIME_COMMIT","b"*40)
    monkeypatch.setattr(preflight,"LINEAGE_COMMIT","a"*40)
    monkeypatch.setattr(preflight,"PARENT_SHA256",parent_ref["sha256"])
    commands={}
    for method,args in source_args.items():
        args=list(args);args[args.index("--source-commit")+1]="b"*40
        item=methods[method]
        commands[method]=["/python",str(worktree/"scripts/diagnostics/robotwin_noise_compare.py")]+args+[
            "--output",item["output"],"--resume",item["checkpoint"]["path"],
            "--resume-trial-manifest",item["manifest"]["path"],
            "--resume-trial-manifest-sha256",item["manifest"]["sha256"],"--lineage-source-commit","a"*40,
            "--lineage-parent-manifest",parent_ref["path"],"--lineage-parent-manifest-sha256",parent_ref["sha256"],
            "--collector-lanes","6","--collector-group-seconds","600","--recycle-at-round-boundary"]
    terminal=write(tmp_path/"terminal.json",dict(status="test terminal evidence"))
    released={method:dict(released=True,allocation_id="job-test",terminal_state="Succeeded",
        terminal_receipt=terminal,manifest=methods[method]["manifest"],checkpoint=methods[method]["checkpoint"],
        canonical_head=heads[method]) for method in preflight.METHODS}
    release_ref=write(tmp_path/"owner-release.json",dict(methods=released))
    snapshot_ref=write(tmp_path/"compute-snapshot.json",dict(residual=2700))
    compute_ref=write(tmp_path/"compute.json",dict(authority=snapshot_ref,
        prior_charge_and_physical_snapshot=snapshot_ref))
    plan=dict(schema="robotwin_trained_pair_plan_v1",runtime_commit="b"*40,lineage_commit="a"*40,
        parent_manifest=parent_ref,methods=methods,child_commands=commands,worktree=str(worktree),
        output=str(tmp_path/"pair"),owner_release=release_ref,compute_binding=compute_ref,
        maximum_seconds=2520,child_wall_seconds=2160,maximum_new_source_actions=21600,
        maximum_input_due_evaluation_actions=48800)
    calls=[]
    validator=SimpleNamespace(RUNTIME="b"*40,LINEAGE="a"*40,PARENT_SHA=parent_ref["sha256"],
        validate_plan=lambda value,**kwargs:calls.append((deepcopy(value),kwargs)),
        pinned_worktree=lambda path,commit:None)
    monkeypatch.setattr(preflight,"load_plan_validator",lambda:validator)
    monkeypatch.setattr(preflight,"load_driver",lambda path:driver)
    monkeypatch.setattr(preflight,"runtime_evidence",lambda *args:{})
    plan_path=tmp_path/"plan.json"
    return dict(path=plan_path,plan=plan,calls=calls,tmp=tmp_path)


def execute(pair):
    pair["path"].write_text(json.dumps(pair["plan"]))
    return preflight.run_preflight(pair["path"],file_sha256(pair["path"]))


def mutate_checkpoint(pair,method,change):
    item=pair["plan"]["methods"][method];path=Path(item["checkpoint"]["path"])
    ck=torch.load(path,map_location="cpu",weights_only=False);change(ck);torch.save(ck,path)
    item["checkpoint"]=bind(path)
    manifest_path=Path(item["manifest"]["path"]);manifest=json.loads(manifest_path.read_text())
    manifest["checkpoint"]=item["checkpoint"]
    item["manifest"]=write(manifest_path,manifest)
    command=pair["plan"]["child_commands"][method]
    command[command.index("--resume-trial-manifest-sha256")+1]=item["manifest"]["sha256"]


def test_fixed_production_pins_have_no_cli_overrides():
    assert preflight.RUNTIME_COMMIT=="57016ba08eed79043fd67bf8b085bd6d42e6d766"
    assert preflight.PARENT_SHA256=="1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0"


def test_goal_boundary_admission_is_explicit_and_default_remains_one_round():
    assert preflight.source_goal_admission(53)==10800
    with pytest.raises(ValueError,match="original goal"):preflight.source_goal_admission(100)
    assert preflight.source_goal_admission(100,allow_source_goal_boundary=True)==0
    with pytest.raises(ValueError):preflight.source_goal_admission(101,allow_source_goal_boundary=True)


@pytest.mark.parametrize("method,native,maximum",[("ppo",1,31600),("ppo_reset",0,17200)])
def test_partial_final_evaluation_reserves_only_uncommitted_original_units(method,native,maximum):
    source=dict(handover_mic=600,pick_diverse_bottles=400,place_a2b_left=400,
        place_object_stand=400,place_phone_stand=400,scan_object=500)
    targets=dict(adjust_bottle=400,move_stapler_pad=400)
    ck=dict(contract=dict(methods=[method],source_iterations=dict(evaluation_iterations=[10,25,50,100]),
        task_episode_limits=dict(source,**targets),source_tasks=list(source),eval_tasks=list(targets),
        eval_episodes=8,campaign=dict(source_validation_episodes=4)),
        states={method:dict(completed_source_iterations=100)},
        completed_units=[f"stage:iteration:{number}:complete" for number in (10,25,50)],
        stages={f"iteration:{number}":dict(status="complete") for number in (10,25,50)})
    parent=dict(workers={method:dict(native_episodes=native)})
    remaining=preflight.remaining_evaluation_units(ck,parent)
    assert sum(row["maximum_actions"] for row in remaining)==maximum
    ck["completed_units"] +=[f"stage:iteration:100:source_validation:{method}:handover_mic",
        f"stage:iteration:100:target:{method}:normal:adjust_bottle:0"]
    if native:ck["completed_units"].append("stage:iteration:100:native_before_target:native:adjust_bottle:0")
    remaining=preflight.remaining_evaluation_units(ck,parent)
    assert sum(row["maximum_actions"] for row in remaining)==maximum-2400-3200-400*native
    ck["completed_units"].append("stage:iteration:100:complete");ck["stages"]["iteration:100"]=dict(status="complete")
    assert preflight.remaining_evaluation_units(ck,parent)==[]


def test_complete_pair_passes_without_ledger_mutation_or_cuda(prepared_pair):
    receipt=execute(prepared_pair)
    assert receipt["status"]=="passed" and receipt["ledger_before"]==receipt["ledger_after"]
    assert receipt["cuda_initialized"] is receipt["source_mutations"] is receipt["campaign_budget_constructed"] is False
    assert prepared_pair["calls"][0][1]==dict(read_evidence=True)
    for method in preflight.METHODS:
        row=receipt["methods"][method]
        assert row["trained_boundary"]["completed_source_iterations"]==1
        assert row["trained_boundary"]["optimizer_steps"]==5 and row["input_due_evaluations"]==[]
        assert row["owner_released_head"]["charged_actions"]==24


@pytest.mark.parametrize("case",["fresh","midround","pending","selected"])
def test_non_update_boundary_is_rejected(prepared_pair,case):
    def change(ck):
        s=ck["states"]["ppo"]
        if case=="fresh":s["completed_source_iterations"]=s["learner"]["updates"]=0
        if case=="midround":s["completed_source_trials"]-=1
        if case=="pending":s["pending_rollouts"]=[({}, {})]
        if case=="selected":s["selected_trial_recovery"]={"prefix":True}
    mutate_checkpoint(prepared_pair,"ppo",change)
    with pytest.raises(ValueError):execute(prepared_pair)


def test_motor_and_owner_head_bindings_are_required(prepared_pair):
    plan=prepared_pair["plan"];path=Path(plan["methods"]["ppo"]["frozen_motor"]["path"])
    plan["methods"]["ppo"]["frozen_motor"]=write(path,dict(before="changed",after="changed",unchanged=True))
    with pytest.raises(ValueError,match="frozen motor"):execute(prepared_pair)


def test_released_head_must_equal_canonical_head(prepared_pair):
    plan=prepared_pair["plan"];path=Path(plan["owner_release"]["path"]);release=json.loads(path.read_text())
    release["methods"]["ppo"]["canonical_head"]["charged_actions"]+=1
    plan["owner_release"]=write(path,release)
    with pytest.raises(ValueError,match="owner's release"):execute(prepared_pair)


def test_snapshot_header_cannot_diverge_from_original_journal(prepared_pair):
    parent=json.loads(Path(prepared_pair["plan"]["parent_manifest"]["path"]).read_text())
    snapshot=Path(parent["workers"]["ppo"]["ledger_root"])/"snapshot.json"
    value=json.loads(snapshot.read_text());value["header"]["total_actions"]+=1
    snapshot.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="original header"):execute(prepared_pair)


def test_pair_ownership_check_is_nonblocking(prepared_pair,monkeypatch):
    # Qualified per-arm checks run before the joint-lock check; intercept only
    # this helper's acquisition of the already validated paired canonical roots.
    original=preflight.fcntl.flock;calls=[]
    def busy(fd,flags):
        calls.append(flags)
        if flags & preflight.fcntl.LOCK_NB:raise BlockingIOError("test other owner")
        return original(fd,flags)
    # fcntl is shared by the qualified verifier, so allow its preliminary
    # acquisitions and refuse the first final pair acquisition.
    count=0
    def acquire(fd,flags):
        nonlocal count
        if flags & preflight.fcntl.LOCK_NB:
            count+=1
            if count>2:return busy(fd,flags)
        return original(fd,flags)
    monkeypatch.setattr(preflight.fcntl,"flock",acquire)
    with pytest.raises(RuntimeError,match="source owner"):execute(prepared_pair)
    assert calls and calls[0]&preflight.fcntl.LOCK_NB


def test_cpu_visibility_required_before_any_qualified_import(prepared_pair,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="explicitly empty"):execute(prepared_pair)


def test_due_stage_cursor_and_completed_stage_reports():
    ck=dict(contract=dict(methods=["ppo"],source_iterations=dict(evaluation_iterations=[10,25,50,100])),
        states=dict(ppo=dict(completed_source_iterations=25)),completed_units=["stage:iteration:10:complete"],
        stages={"iteration:10":dict(status="complete")})
    assert preflight.input_due_evaluation(ck)==[25]
    ck["states"]["ppo"]["completed_source_iterations"]=26
    with pytest.raises(ValueError,match="earlier original"):preflight.input_due_evaluation(ck)


def test_small_runtime_evidence_is_hash_bound(tmp_path):
    native=tmp_path/"native";calibration=tmp_path/"calibration"
    horizons=write(native/"task_config/_eval_step_limit.yml",dict(test=400))
    result=write(calibration/"result.json",dict(test="calibrated"))
    capture=write(tmp_path/"qualification.json",dict(test="qualified"))
    norm=write(tmp_path/"norm.json",dict(test="normalization"))
    args=SimpleNamespace(robotwin_root=native,capture_qualification=Path(capture["path"]),norm_stats=Path(norm["path"]))
    parent=dict(benchmark_horizon_config_sha256=horizons["sha256"],
        calibration=dict(root=str(calibration),result_sha256=result["sha256"]))
    checkpoint=dict(contract=dict(capture_qualification=capture,norm_stats=norm["path"],runtime=dict(
        norm_stats_path=norm["path"],norm_stats_sha256=norm["sha256"],
        episode_limits_source=dict(sha256=horizons["sha256"]))))
    assert len(preflight.runtime_evidence(args,parent,checkpoint))==4
    Path(norm["path"]).write_text("changed")
    with pytest.raises(ValueError,match="hash mismatch"):preflight.runtime_evidence(args,parent,checkpoint)
