"""Trained Adam/RNG migration with CPU simulators; run on allocated Linux."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_robotwin_trial_resume import recovery_runtime,common_args,driver
from fluffy_bassoon.meta_rl.robotwin_trial_resume import (
    artifact_digest,file_sha256,load_trial_continuation)
from fluffy_bassoon.meta_rl.robotwin_trained_parallel import (
    validate_trained_parallel_boundary,assert_loaded_trained_identity)


@pytest.fixture
def trained_boundary(tmp_path,recovery_runtime,monkeypatch):
    import fluffy_bassoon.rl.robotwin_noise as runtime
    recovery_runtime["fail_reset"]=None
    monkeypatch.setattr(runtime.RobotwinNoiseEnv,"native_horizon",32)
    monkeypatch.setattr(runtime,"load_benchmark_episode_limits",lambda *args:
        {task:1 for task in driver.SOURCE_TASKS+driver.HELDOUT_TASKS})
    args=common_args(tmp_path,"ppo",iterations=3,limit=6)+[
        "--benchmark-episode-limits","--source-eval-iterations","1","2","3",
        "--source-validation-episodes","4","--ppo-max-update-kl",".1","--learning-rate","1e-6"]
    args.remove("--stage-source-validation-only")
    output=tmp_path/"scalar"/"experiment"
    assert driver.main(args+["--output",str(output)])==0
    summary=json.loads((output/"summary.json").read_text())
    manifest_path=Path(summary["continuation_manifest"]["path"])
    manifest=json.loads(manifest_path.read_text());manifest["execution_source_commit"]="b"*40
    manifest_path=tmp_path/"trained-manifest.json";manifest_path.write_text(json.dumps(manifest))
    checkpoint_path=output/"coordinator-recovery.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    continuation=load_trial_continuation(manifest_path,file_sha256(manifest_path),
        checkpoint_path=checkpoint_path,checkpoint=checkpoint,execution_commit="b"*40,lineage_commit="a"*40)
    parent=dict(format="robotwin_parallel_whole_trial_campaign_v1",source_commit="a"*40,
        source_iterations=3,source_eval_iterations=[1,2,3],trials_per_iteration=6,train_episodes=4,
        workers=dict(ppo=dict(ledger_root=str(tmp_path/"campaign"),campaign_id="robotwin-source-1m",
            source_safety_actions=72,maximum_actions=1_000_000,native_episodes=0)))
    parent_path=tmp_path/"original-parent.json";parent_path.write_text(json.dumps(parent))
    return dict(args=args,output=output,checkpoint=checkpoint,manifest=manifest_path,
        continuation=continuation,root=tmp_path/"campaign",state=recovery_runtime,parent=parent_path)


def test_genuinely_trained_update_boundary_preserves_adam_and_all_streams(trained_boundary):
    b=trained_boundary;checkpoint=b["checkpoint"];saved=checkpoint["states"]["ppo"]
    audit=validate_trained_parallel_boundary(b["continuation"],checkpoint,
        manifest_path=b["manifest"],campaign_root=b["root"])
    assert audit["completed_source_trials"]==6 and audit["optimizer_steps"]==5
    assert saved["learner"]["optimizer"]["state"]
    assert any(value["exp_avg"].abs().sum()>0 for value in saved["learner"]["optimizer"]["state"].values())
    from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig,RecurrentNoisePPO
    learner=RecurrentNoisePPO(2,(1,2),config=PPOConfig(**saved["learner"]["config"]),device="cpu",seed=999)
    learner.load_state_dict(deepcopy(saved["learner"]))
    schedule=driver.BalancedSchedule(driver.SOURCE_TASKS,999);schedule.load_state_dict(saved["schedule"])
    assert_loaded_trained_identity(learner,schedule,saved)
    assert audit["identity_digests"]["optimizer"]==artifact_digest(learner.optimizer.state_dict())
    learner.sampler.random()
    with pytest.raises(ValueError,match="sampler"):assert_loaded_trained_identity(learner,schedule,saved)


@pytest.mark.parametrize("change",["selected","pending","cursor","updates","untrained","lr","adam"])
def test_partial_or_changed_training_state_is_not_parallel_eligible(trained_boundary,change):
    b=trained_boundary;ck=deepcopy(b["checkpoint"]);continuation=b["continuation"]
    saved=ck["states"]["ppo"];learner=saved["learner"]
    if change=="selected":continuation=replace(continuation,trial={"task":"scan_object"})
    if change=="pending":saved["pending_rollouts"]=[({}, {})]
    if change=="cursor":saved["schedule"]["trials"]+=1
    if change=="updates":learner["updates"]+=1
    if change=="untrained":saved["completed_source_trials"]=saved["completed_source_iterations"]=0
    if change=="lr":learner["optimizer"]["param_groups"][0]["lr"]*=2
    if change=="adam":learner["optimizer"]["state"]={}
    with pytest.raises(ValueError):validate_trained_parallel_boundary(continuation,ck,
        manifest_path=b["manifest"],campaign_root=b["root"])


def test_later_physical_ledger_event_rejects_before_cuda(trained_boundary,monkeypatch):
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    b=trained_boundary;contract=b["checkpoint"]["contract"]
    with CampaignBudget(b["root"],campaign_id=contract["campaign"]["id"],
            total_actions=contract["campaign"]["max_actions"],source_caps={"ppo":72},
            contract=contract,run_id="later",source_commit="a"*40) as budget:
        budget.reserve("later:action",1,phase="source",method="ppo",task="scan_object",
            episode=1,chunk=0,policy_version="later")
        budget.resolve("later:action",1)
    def no_cuda():raise AssertionError("CUDA gate must not be reached")
    monkeypatch.setattr(driver.torch.cuda,"is_available",no_cuda)
    with pytest.raises(RuntimeError,match="physical campaign events"):
        driver.main(parallel_args(b,b["output"].parent.parent/"new"/"experiment"))


def parallel_args(b,output):
    return b["args"]+["--source-commit","b"*40,"--output",str(output),"--resume",str(b["output"]),
        "--resume-trial-manifest",str(b["manifest"]),
        "--resume-trial-manifest-sha256",file_sha256(b["manifest"]),"--lineage-source-commit","a"*40,
        "--lineage-parent-manifest",str(b["parent"]),"--lineage-parent-manifest-sha256",file_sha256(b["parent"]),
        "--collector-lanes","6","--collector-group-seconds","1","--recycle-at-round-boundary"]


@pytest.mark.parametrize("change",[
    ["--learning-rate","2e-6"],["--stage-source-validation-only"],
    ["--source-validation-episodes","0"],["--ppo-epochs","4"],["--native-episodes","1"],
])
def test_trained_scientific_argument_changes_fail_before_cuda(trained_boundary,monkeypatch,change):
    b=trained_boundary
    def no_cuda():raise AssertionError("CUDA gate must not be reached")
    monkeypatch.setattr(driver.torch.cuda,"is_available",no_cuda)
    with pytest.raises(ValueError,match="Trained parallel migration"):
        driver.main(parallel_args(b,b["output"].parent.parent/"changed"/"experiment")+change)


def test_cloned_canonical_ledger_root_is_rejected_before_cuda(trained_boundary,tmp_path,monkeypatch):
    import shutil
    b=trained_boundary;clone=tmp_path/"cloned-ledger";shutil.copytree(b["root"],clone)
    def no_cuda():raise AssertionError("CUDA gate must not be reached")
    monkeypatch.setattr(driver.torch.cuda,"is_available",no_cuda)
    with pytest.raises(ValueError,match="canonical worker ledger root"):
        driver.main(parallel_args(b,tmp_path/"cloned-run")+["--campaign-root",str(clone)])


def test_trained_n6_execution_preserves_contract_and_stops_before_due_evaluation(
        tmp_path,trained_boundary,monkeypatch):
    import fluffy_bassoon.rl.robotwin_production_vector as vector_module
    import fluffy_bassoon.rl.robotwin_noise as runtime_module
    b=trained_boundary;instances=[]
    class Vector:
        observation_dim=2;noise_shape=(1,2);native_horizon=1
        def __init__(self,scalar_env,budget,run_id,lanes,corpus_writer,output_dir):
            assert lanes==6;instances.append(self);self.run_id=run_id;self._pool=None
            self.root=Path(output_dir)/"production-vector";self.root.mkdir()
            self.envs=[driver.CampaignEnv(runtime_module.RobotwinNoiseEnv(
                corpus_writer=corpus_writer.for_lane(i)),budget,run_id=run_id+f":lane:{i}") for i in range(6)]
            self.features=np.zeros((6,2),np.float32);self.group=None
        def episode_limit_for_task(self,task):return 1
        def set_lane_context(self,lane,context):self.envs[lane].set_recording_context(context)
        def record_decision_metadata(self,lane,metadata):self.envs[lane].record_decision_metadata(metadata)
        def admit_group(self,tasks,seeds):
            self._pool=object();self.group=dict(group_id=self.run_id+":group",status="admitted",physical_actions=0)
            # Milestone1 must finish before any worker process is created.
            progress=[json.loads(s) for s in (self.root.parent/"progress.jsonl").read_text().splitlines()]
            assert any(x["stage"]=="source_stage_complete" and x["source_iteration"]==1 for x in progress)
        def reset_batch(self,tasks,seeds):return self.reset_lanes(range(6),tasks,seeds)
        def reset_lanes(self,indices,tasks,seeds):
            for lane,task,seed in zip(indices,tasks,seeds):self.features[lane]=self.envs[lane].reset(task,seed)
            return self.features[list(indices)].copy()
        def step_batch(self,noises,active_mask):
            rewards=np.zeros(6);terms=np.zeros(6,bool);counts=np.zeros(6,int)
            for lane in np.flatnonzero(active_mask):
                self.features[lane],rewards[lane],terms[lane],_,info=self.envs[lane].step(noises[lane])
                counts[lane]=info["primitive_steps"];self.group["physical_actions"]+=int(counts[lane])
            return self.features.copy(),rewards,terms,np.zeros(6,bool),dict(
                primitive_steps=counts,success=terms.copy(),budget_truncated=np.zeros(6,bool))
        def finish_group(self):self.group["status"]="physical_complete_uncheckpointed";return dict(self.group)
        def group_recovery_metadata(self):return self.group
        def acknowledge_checkpoint(self,path):
            ck=torch.load(path,map_location="cpu",weights_only=False);s=ck["states"]["ppo"]
            assert s["completed_source_trials"]==12 and len(s["pending_rollouts"])==6
            assert s["learner"]["updates"]==1  # ACK before unchanged second update.
            self.group["status"]="checkpointed"
        def close(self):pass
    monkeypatch.setattr(vector_module,"ProductionRobotwinVectorEnv",Vector)
    header=b["root"]/"events"/"0000000000.json"
    before_header=header.read_bytes()
    before_actions=len(b["state"]["actions"])
    output=tmp_path/"parallel"/"experiment"
    assert driver.main(parallel_args(b,output))==0
    ck=torch.load(output/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    s=ck["states"]["ppo"];summary=json.loads((output/"summary.json").read_text())
    assert ck["contract"]==b["checkpoint"]["contract"] and header.read_bytes()==before_header
    assert "parallel_source_collection" not in ck["contract"]
    assert ck["execution_metadata"]["parallel_source_collection"]["lanes"]==6
    assert s["completed_source_iterations"]==s["learner"]["updates"]==2 and s["learner"]["optimizer_steps"]==10
    assert s["completed_source_trials"]==12 and not s["pending_rollouts"]
    assert summary["process_source_trials"]==6 and summary["status"]=="wall_budget_stop"
    assert summary["continuation_manifest"] and s["source_steps"]==48
    progress=[json.loads(x) for x in (output/"progress.jsonl").read_text().splitlines()]
    assert not any(x["stage"]=="source_stage_evaluation_start" and x["source_iteration"]==2 for x in progress)
    episodes=[json.loads(x) for x in (output/"episodes.jsonl").read_text().splitlines()]
    assert sum(x["phase"]=="source" for x in episodes)==24
    assert len(b["state"]["actions"])-before_actions==len(episodes)>24
    assert instances[0].run_id!="experiment" and instances[0].run_id.startswith("parallel:experiment:")
    manifest=json.loads(Path(summary["continuation_manifest"]["path"]).read_text())
    assert manifest["expected"]["completed_source_iterations"]==2
