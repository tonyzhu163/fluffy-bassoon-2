"""Episode-boundary source recovery; execute only on allocated Linux."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig,RecurrentNoisePPO,capture_rng,restore_rng
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC
from fluffy_bassoon.meta_rl.robotwin_trial_resume import (
    SCHEMA,artifact_digest,file_sha256,load_trial_continuation,combine_episode_rollouts)


spec=importlib.util.spec_from_file_location("resume_campaign_driver",
    Path(__file__).resolve().parents[1]/"scripts/diagnostics/robotwin_noise_compare.py")
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


@pytest.fixture
def recovery_runtime(monkeypatch):
    import fluffy_bassoon.rl.robotwin_noise as runtime_module
    import fluffy_bassoon.meta_rl.vla_noise_ppo as ppo_module
    state=dict(fail_reset=1,actions=[],decisions=[])
    class Runtime:
        observation_dim=2;noise_shape=(1,2);native_horizon=1;episode_limit=1;contract={}
        last_reset_info={"reset_policy_actions":0}
        def __init__(self,*,corpus_writer,**kwargs):
            self.writer=corpus_writer;self.source_resets=0;self.metadata={}
        def set_recording_context(self,context):
            self.context=dict(getattr(self,"context",{}),**context);self.writer.set_context(context)
        def record_decision_metadata(self,metadata):
            self.metadata=metadata;state["decisions"].append((self.context.copy(),deepcopy(metadata)))
        def raw(self):return dict(main_images=np.zeros((1,1,1,3),np.uint8),
            wrist_images=np.zeros((1,2,1,1,3),np.uint8),states=np.zeros((1,2)),task_descriptions=["test"])
        def reset(self,task,seed):
            if self.context["phase"]=="source":
                if state["fail_reset"] is not None and self.source_resets==state["fail_reset"]:
                    state["fail_reset"]=None
                    raise RuntimeError("planner reset failed before another episode began")
                self.source_resets+=1
            self.task=task;self.seed=seed;self.elapsed_steps=0
            self.features=np.array([seed%7,0],np.float32)
            self.last_reset_info=dict(task=task,requested_seed=seed,actual_seed=seed,
                episode_limit=1,reset_policy_actions=0)
            self.writer.begin_episode(self.raw(),self.features,self.last_reset_info)
            return self.features.copy()
        def step(self,noise,*,max_primitive_steps=None):
            self.writer.begin_decision(np.asarray(noise).reshape(1,2),np.zeros((1,2),np.float32),self.metadata)
            following=np.array([self.seed%7,1],np.float32)
            self.writer.complete_native(next_raw=self.raw(),next_features=following,
                raw_rewards=np.array([1.],np.float32),raw_terminations=np.array([True]),
                raw_truncations=np.array([False]),valid_action_mask=np.array([True]),
                reward=1.,terminated=True,truncated=False,executed_steps=1,success=True,duration_seconds=.1)
            self.writer.finalize_learning_flags(reward=1.,terminated=True,truncated=False,
                budget_truncated=False,episode_primitive_steps=1)
            self.elapsed_steps=1;state["actions"].append((self.context["method"],self.task,self.seed))
            return following,1.,True,False,dict(primitive_steps=1,success=True,budget_truncated=False)
        def check_noise_parity(self,noise):return dict(passed=True,max_abs_error=0.)
        def frozen_fingerprint(self):return "fixed-motor"
        def close(self):pass
    def cpu_rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[])
    monkeypatch.setattr(runtime_module,"RobotwinNoiseEnv",Runtime)
    monkeypatch.setattr(driver.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(driver.torch.cuda,"manual_seed_all",lambda seed:None)
    monkeypatch.setattr(ppo_module,"capture_rng",cpu_rng)
    monkeypatch.setattr(driver,"capture_rng",cpu_rng)
    monkeypatch.setattr(driver,"RecurrentNoisePPO",lambda observation_dim,noise_shape,*,config,seed,**kw:
        RecurrentNoisePPO(observation_dim,noise_shape,config=replace(config,width=8,layers=1,heads=2,memory_length=8),
                          device="cpu",seed=seed))
    monkeypatch.setattr(driver,"NoiseSAC",type("TinySAC",(NoiseSAC,),{
        "__init__":lambda self,observation_dim,action_dim,*,config,seed,**kw:NoiseSAC.__init__(
            self,observation_dim,action_dim,config=replace(config,hidden_dims=(8,),batch_size=1,
                warmup_chunks=0,replay_capacity=64),device="cpu",seed=seed)}))
    return state


def common_args(tmp_path,method,*,iterations=1,limit=1):
    return ["--source-commit","a"*40,"--model-path","/model","--norm-stats","/norm",
        "--robotwin-root","/robotwin","--rlinf-root","/rlinf","--methods",method,
        "--source-steps",str(iterations*24),"--source-iterations",str(iterations),"--ppo-trials-per-update","6",
        "--episode-limit","1","--native-episodes","0","--source-validation-episodes","0",
        "--stage-source-validation-only","--record-corpus","--campaign-root",str(tmp_path/"campaign"),
        "--max-source-trials-per-process",str(limit)]


def bind(path):return dict(path=str(path.resolve()),sha256=file_sha256(path))


def selected_manifest(tmp_path,output,method,execution="b"*40):
    checkpoint_path=output/"coordinator-recovery.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    saved=checkpoint["states"][method];learner=saved["learner"]
    corpus=output/"corpus";episodes_path=output/"episodes.jsonl";episodes_path.touch(exist_ok=True)
    progress=[json.loads(line) for line in (output/"progress.jsonl").read_text().splitlines()]
    start=[row for row in progress if row["stage"]=="source_trial_start"][-1]
    trial=dict(task=start["task"],seeds=start["seeds"],trial_id=f"source:{method}:{saved['schedule']['trials']}",
        schedule_trials=saved["schedule"]["trials"],source_iteration=start["source_iteration"],
        trial_in_iteration=start["trial_in_iteration"])
    episode_rows=[json.loads(line) for line in episodes_path.read_text().splitlines()]
    prefix=[]
    for reset_path in (corpus/"episodes").glob("*/reset.json"):
        reset=json.loads(reset_path.read_text())
        if reset["context"].get("trial_id")!=trial["trial_id"]:continue
        index=trial["seeds"].index(reset["reset_info"]["requested_seed"])
        row_index=next(i for i,row in enumerate(episode_rows) if row["seed"]==trial["seeds"][index])
        prefix.append(dict(reset=bind(reset_path),manifest=bind(reset_path.parent/"manifest.json"),episode_row_index=row_index))
    prefix.sort(key=lambda item:item["episode_row_index"])
    ledger_copy=tmp_path/"initial-ledger.json"
    ledger_copy.write_bytes((tmp_path/"campaign/snapshot.json").read_bytes())
    counters={name:learner[name] for name in ("primitive_steps","updates","optimizer_steps","chunks",
        "environment_steps","replay_position","pending_chunks") if name in learner}
    manifest=dict(schema=SCHEMA,kind="selected_trial",execution_source_commit=execution,lineage_source_commit="a"*40,
        method=method,checkpoint=bind(checkpoint_path),corpus_root=str(corpus),corpus_contract=bind(corpus/"contract.json"),
        progress=bind(output/"progress.jsonl"),episodes_file=bind(episodes_path),ledger_snapshot=bind(ledger_copy),
        trial=trial,prefix=prefix,expected=dict(source_steps=saved["source_steps"],
            completed_source_trials=saved["completed_source_trials"],completed_source_iterations=saved["completed_source_iterations"],
            schedule_digest=artifact_digest(saved["schedule"]),learner_counters=counters,
            pending_rollout_digests=[artifact_digest(r) for r,_ in saved["pending_rollouts"]]))
    path=tmp_path/"selected-manifest.json";path.write_text(json.dumps(manifest))
    return path,manifest,checkpoint


@pytest.mark.parametrize("method",["sac","ppo_reset","ppo"])
def test_selected_trial_continuation_preserves_actor_state_clocks_and_original_seeds(tmp_path,recovery_runtime,method):
    state=recovery_runtime
    if method=="ppo":state["fail_reset"]=0
    first=tmp_path/"first";second=tmp_path/"second"
    args=common_args(tmp_path,method)
    with pytest.raises(RuntimeError,match="planner reset"):
        driver.main(args+["--output",str(first)])
    manifest_path,manifest,before=selected_manifest(tmp_path,first,method)
    start_actions=list(state["actions"])
    saved=before["states"][method]["learner"]
    if method=="sac":assert saved["updates"]==20 and saved["primitive_steps"]==1 and len(saved["replay"])==1
    if method=="ppo_reset":
        recovered=load_trial_continuation(manifest_path,file_sha256(manifest_path),checkpoint_path=first/"coordinator-recovery.pt",
            checkpoint=before,execution_commit="b"*40,lineage_commit="a"*40)
        assert recovered.prefix_rollout["log_probs"].shape==recovered.prefix_rollout["resets"].shape==(1,1)
        assert not recovered.prefix_rollout["trial_ends"].any() and recovered.prefix_rollout["attempt_ends"][-1].all()
    resumed=args+["--source-commit","b"*40,"--output",str(second),"--resume",str(first),
        "--resume-trial-manifest",str(manifest_path),"--resume-trial-manifest-sha256",file_sha256(manifest_path),
        "--lineage-source-commit","a"*40]
    assert driver.main(resumed)==0
    after=torch.load(second/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    source=after["states"][method]
    assert source["source_steps"]==4 and source["completed_source_trials"]==1 and source["schedule"]["trials"]==1
    assert [seed for _,_,seed in state["actions"]]==manifest["trial"]["seeds"]
    assert len(start_actions)==(0 if method=="ppo" else 1) and len(state["actions"])==4
    summary=json.loads((second/"summary.json").read_text())
    assert summary["source"][method]["allocation_source_steps"]==(4 if method=="ppo" else 3)
    assert summary["campaign_budget"]["actual_actions"]==4
    assert summary["contract"]["source_commit"]=="a"*40
    assert json.loads((second/"execution-migration.json").read_text())["execution_source_commit"]=="b"*40
    if method=="sac":
        learner=source["learner"]
        assert learner["updates"]==80 and learner["environment_steps"]==4 and len(learner["replay"])==4
        assert {int(v["step"]) for v in learner["actor_optimizer"]["state"].values()}=={80}
    else:
        rollout,result=source["pending_rollouts"][0]
        assert [row["episode"] for row in result["episodes"]]==[1,2,3,4]
        assert rollout["trial_ends"].flatten().tolist()==[False,False,False,True]
        if method=="ppo_reset":
            assert rollout["resets"].flatten().tolist()==[True]*4
            np.testing.assert_allclose(rollout["observations"][:,0,-2].numpy(),np.arange(4)/3)
            metadata=[m for context,m in state["decisions"] if context["phase"]=="source"]
            assert [m["history_index"] for m in metadata]==[0,1,2,3]
            assert [m["episode_index"] for m in metadata]==[0,1,2,3]


def test_manifest_rejects_hash_counter_and_unresolved_or_persistent_prefix(tmp_path,recovery_runtime):
    first=tmp_path/"first"
    with pytest.raises(RuntimeError):driver.main(common_args(tmp_path,"ppo_reset")+["--output",str(first)])
    path,original,checkpoint=selected_manifest(tmp_path,first,"ppo_reset")
    def verify(manifest,*,digest=None):
        path.write_text(json.dumps(manifest))
        return load_trial_continuation(path,file_sha256(path) if digest is None else digest,
            checkpoint_path=first/"coordinator-recovery.pt",checkpoint=checkpoint,
            execution_commit="b"*40,lineage_commit="a"*40)
    with pytest.raises(ValueError,match="manifest hash"):verify(original,digest="0"*64)
    wrong=deepcopy(original);wrong["expected"]["completed_source_trials"]+=1
    with pytest.raises(ValueError,match="counter"):verify(wrong)
    wrong=deepcopy(original);wrong["trial"]["seeds"][0]+=1
    with pytest.raises(ValueError,match="progress"):verify(wrong)
    wrong=deepcopy(original);wrong["prefix"]=[]
    with pytest.raises(ValueError,match="omitted"):verify(wrong)
    wrong=deepcopy(original);wrong["checkpoint"]["sha256"]="0"*64
    with pytest.raises(ValueError,match="hash mismatch"):verify(wrong)
    ledger=json.loads(Path(original["ledger_snapshot"]["path"]).read_text());ledger["uncertain_actions"]=1
    ledger_path=tmp_path/"bad-ledger.json";ledger_path.write_text(json.dumps(ledger))
    wrong=deepcopy(original);wrong["ledger_snapshot"]=bind(ledger_path)
    with pytest.raises(ValueError,match="unresolved"):verify(wrong)
    episode_manifest_path=Path(original["prefix"][0]["manifest"]["path"])
    episode_manifest=json.loads(episode_manifest_path.read_text());chunk_ref=episode_manifest["chunks"][-1]
    chunk_path=episode_manifest_path.parent/chunk_ref["path"];chunk=json.loads(chunk_path.read_text())
    chunk["learning_result"]["terminated"]=False;chunk_path.write_text(json.dumps(chunk))
    chunk_ref["sha256"]=file_sha256(chunk_path);episode_manifest_path.write_text(json.dumps(episode_manifest))
    wrong=deepcopy(original);wrong["prefix"][0]["manifest"]=bind(episode_manifest_path)
    with pytest.raises(ValueError,match="closed episode boundary"):verify(wrong)


def test_nonempty_persistent_prefix_is_refused(tmp_path,recovery_runtime):
    first=tmp_path/"first"
    with pytest.raises(RuntimeError):driver.main(common_args(tmp_path,"ppo")+["--output",str(first)])
    path,manifest,checkpoint=selected_manifest(tmp_path,first,"ppo")
    with pytest.raises(ValueError,match="persistent nonempty"):
        load_trial_continuation(path,file_sha256(path),checkpoint_path=first/"coordinator-recovery.pt",
            checkpoint=checkpoint,execution_commit="b"*40,lineage_commit="a"*40)


def test_reset_prefix_preserves_existing_pending_trials_and_can_recycle_after_full_round(tmp_path,recovery_runtime):
    state=recovery_runtime;state["fail_reset"]=9  # Two full trials, then episode1 of selected trial3.
    first=tmp_path/"first";second=tmp_path/"second"
    args=common_args(tmp_path,"ppo_reset",iterations=2,limit=0)
    with pytest.raises(RuntimeError):driver.main(args+["--output",str(first)])
    path,manifest,before=selected_manifest(tmp_path,first,"ppo_reset")
    prior=[artifact_digest(r) for r,_ in before["states"]["ppo_reset"]["pending_rollouts"]]
    assert len(prior)==2
    # A strict one-trial slice first demonstrates that the original pending
    # trials survive intact; the remaining complete-round guard is independent.
    resumed=args+["--source-commit","b"*40,"--output",str(second),"--resume",str(first),
        "--resume-trial-manifest",str(path),"--resume-trial-manifest-sha256",file_sha256(path),
        "--lineage-source-commit","a"*40,"--max-source-trials-per-process","1"]
    assert driver.main(resumed)==0
    saved=torch.load(second/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    pending=saved["states"]["ppo_reset"]["pending_rollouts"]
    assert [artifact_digest(r) for r,_ in pending[:2]]==prior and len(pending)==3
    summary=json.loads((second/"summary.json").read_text());next_manifest=summary["continuation_manifest"]
    third=tmp_path/"third"
    call=args+["--source-commit","b"*40,"--output",str(third),"--resume",str(second),
        "--resume-trial-manifest",next_manifest["path"],"--resume-trial-manifest-sha256",next_manifest["sha256"],
        "--lineage-source-commit","a"*40,"--max-source-trials-per-process","1","--recycle-at-round-boundary"]
    assert driver.main(call)==0
    summary=json.loads((third/"summary.json").read_text())
    assert summary["process_source_trials"]==3  # Threshold1 waits for the remaining3 complete trials.
    assert summary["source"]["ppo_reset"]["completed_source_trials"]==6
    assert summary["source"]["ppo_reset"]["updates"]==1 and summary["source"]["ppo_reset"]["pending_trials"]==0


def test_operational_process_rollovers_preserve_updates_and_completed_evaluation_units(tmp_path,recovery_runtime):
    state=recovery_runtime;state["fail_reset"]=None
    args=common_args(tmp_path,"ppo",iterations=2,limit=6)
    args +=["--source-eval-iterations","1","2","--source-validation-episodes","4"]
    previous=None;execution="a"*40
    for index in range(3):
        output=tmp_path/f"part{index}"
        call=args+["--output",str(output)]
        if previous is not None:
            manifest=json.loads((previous/"summary.json").read_text())["continuation_manifest"]
            call +=["--resume",str(previous),"--resume-trial-manifest",manifest["path"],
                "--resume-trial-manifest-sha256",manifest["sha256"],"--lineage-source-commit",execution]
        assert driver.main(call)==0
        summary=json.loads((output/"summary.json").read_text())
        assert summary["process_source_trials"]<=6
        if index<2:assert summary["status"]=="wall_budget_stop" and summary["continuation_manifest"]
        previous=output
    assert summary["status"]=="complete" and summary["source"]["ppo"]["updates"]==2
    assert summary["source"]["ppo"]["completed_source_trials"]==12
    assert len(summary["source_validation"])==12
    assert len(state["actions"])==96  #48source+48frozenvalidation; no duplicate completed unit.


def test_real_reset_controller_suffix_keeps_original_episode_and_history_clocks():
    class Env:
        native_horizon=1;episode_limit=7
        def reset(self,task,seed):
            self.seed=seed;self.elapsed_steps=0
            return np.array([seed,0],np.float32)
        def step(self,noise):
            self.elapsed_steps+=1;end=self.elapsed_steps==7
            return np.array([self.seed,self.elapsed_steps],np.float32),float(end),end,False,dict(
                primitive_steps=1,success=end,budget_truncated=False)
    config=PPOConfig(width=8,layers=1,heads=2,memory_length=8,train_episodes=4,
                     episode_limit=7,reset_each_episode=True)
    model=RecurrentNoisePPO(2,(1,2),config=config,device="cpu",seed=7)
    boundary={}
    def episode(row):
        if row["episode"]==1:boundary["rng"]=capture_rng()
    full,result=model.collect_trial(Env(),"scan_object",[1,2,3,4],episode_callback=episode)
    prefix={key:value[:7].clone() for key,value in full.items()}
    restore_rng(boundary["rng"]);metadata=[]
    suffix,remaining=model.collect_trial(Env(),"scan_object",[1,2,3,4],start_episode=1,
        history_offset=7,decision_callback=metadata.append)
    assembled=combine_episode_rollouts(prefix,suffix)
    for key in full:torch.testing.assert_close(full[key],assembled[key],rtol=0,atol=0)
    assert [row["episode"] for row in remaining["episodes"]]==[2,3,4]
    assert [row["history_index"] for row in metadata]==list(range(7,28))
    assert [row["episode_index"] for row in metadata]==[1]*7+[2]*7+[3]*7
    assert assembled["trial_ends"].flatten().tolist()==[False]*27+[True]
    assert model.replay_error(assembled)==dict(log_prob_max_error=0.,value_max_error=0.)
    persistent=RecurrentNoisePPO(2,(1,2),config=replace(config,reset_each_episode=False),device="cpu",seed=7)
    with pytest.raises(ValueError,match="trained reset"):
        persistent.collect_trial(Env(),"scan_object",[1,2,3,4],start_episode=1,history_offset=7)
