"""Prepared end-to-end CPU regression; execute only on allocated Linux."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_robotwin_trial_resume import recovery_runtime
from test_robotwin_trained_parallel import trained_boundary,parallel_args,driver
from fluffy_bassoon.meta_rl.robotwin_trial_resume import file_sha256,artifact_digest


def bind(path):return dict(path=str(path.resolve()),sha256=file_sha256(path))


def write(path,value):path.write_text(json.dumps(value))


def remove_option(args,name,*,value=True):
    args=list(args);index=args.index(name);del args[index:index+(2 if value else 1)];return args


@pytest.fixture
def audited_round(tmp_path,trained_boundary,monkeypatch):
    """Real trained Adam, real corpus/journal, fake native processes only."""
    import fluffy_bassoon.rl.robotwin_production_vector as vector_module
    import fluffy_bassoon.rl.robotwin_noise as runtime_module
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts/diagnostics"))
    audit=importlib.import_module("robotwin_trained_pair_audit")
    preflight=importlib.import_module("robotwin_trained_pair_preflight")
    b=trained_boundary

    # Finish the genuine input-due stage1 in a fresh scalar process, then stop
    # before selecting another source trial. Its completed report/units/history
    # must survive the following N6 migration unchanged.
    ready=tmp_path/"stage-ready"/"experiment"
    args=parallel_args(b,ready)
    for option in ("--collector-lanes","--collector-group-seconds",
                   "--lineage-parent-manifest","--lineage-parent-manifest-sha256"):
        args=remove_option(args,option)
    def source_boundary(self,episodes):
        raise driver.WallBudgetReached("test stop after original due stage, before next source selection")
    with monkeypatch.context() as context:
        context.setattr(driver.BalancedSchedule,"sample",source_boundary)
        assert driver.main(args)==0
    ready_summary=json.loads((ready/"summary.json").read_text())
    assert ready_summary["process_source_trials"]==0
    before=torch.load(ready/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    assert before["states"]["ppo"]["learner"]["updates"]==1
    assert "stage:iteration:1:complete" in before["completed_units"]
    assert before["stages"]["iteration:1"]["status"]=="complete"
    b=dict(b,output=ready,manifest=Path(ready_summary["continuation_manifest"]["path"]),checkpoint=before)
    instances=[]

    class Vector:
        observation_dim=2;noise_shape=(1,2);native_horizon=1
        def __init__(self,scalar_env,budget,run_id,lanes,corpus_writer,output_dir):
            assert lanes==6;instances.append(self);self.run_id=run_id;self._pool=None
            self.output=Path(output_dir);self.root=self.output/"production-vector";self.root.mkdir()
            self.journal=self.root/"group-00000001.json";self.group=None
            self.envs=[driver.CampaignEnv(runtime_module.RobotwinNoiseEnv(
                corpus_writer=corpus_writer.for_lane(lane)),budget,run_id=run_id+f":lane:{lane}")
                for lane in range(6)]
            self.features=np.zeros((6,2),np.float32)
        def persist(self):write(self.journal,self.group)
        def episode_limit_for_task(self,task):return 1
        def set_lane_context(self,lane,metadata):self.envs[lane].set_recording_context(metadata)
        def record_decision_metadata(self,lane,metadata):self.envs[lane].record_decision_metadata(metadata)
        def admit_group(self,tasks,seeds):
            self._pool=object()
            self.group=dict(group_id=self.run_id+":group",status="admitted",journal_path=str(self.journal),
                tasks=list(tasks),episode_seeds=deepcopy(seeds),completed_episodes=[0]*6,
                physical_actions=0,dispatch_count=0)
            write(self.output/"ORDINARY-RESUME-BLOCKED.json",dict(self.group,resume_blocked=True))
            self.persist()
        def reset_batch(self,tasks,seeds):return self.reset_lanes(range(6),tasks,seeds)
        def reset_lanes(self,indices,tasks,seeds):
            indices=list(indices);self.group["dispatch_count"]+=1;self.persist()
            for lane,task,seed in zip(indices,tasks,seeds):self.features[lane]=self.envs[lane].reset(task,seed)
            return self.features[indices].copy()
        def step_batch(self,noises,active_mask):
            rewards=np.zeros(6);terms=np.zeros(6,bool);counts=np.zeros(6,int)
            self.group["dispatch_count"]+=1
            for lane in np.flatnonzero(active_mask):
                self.features[lane],rewards[lane],terms[lane],trunc,info=self.envs[lane].step(noises[lane])
                counts[lane]=info["primitive_steps"]
                self.group["physical_actions"]+=int(counts[lane])
                if terms[lane] or trunc:self.group["completed_episodes"][lane]+=1
            self.persist()
            return self.features.copy(),rewards,terms,np.zeros(6,bool),dict(
                primitive_steps=counts,success=terms.copy(),budget_truncated=np.zeros(6,bool))
        def finish_group(self):
            assert self.group["completed_episodes"]==[4]*6
            self.group["status"]="physical_complete_uncheckpointed";self.persist()
            return deepcopy(self.group)
        def group_recovery_metadata(self):return deepcopy(self.group)
        def acknowledge_checkpoint(self,path):
            checkpoint=torch.load(path,map_location="cpu",weights_only=False);state=checkpoint["states"]["ppo"]
            assert state["completed_source_trials"]==12 and len(state["pending_rollouts"])==6
            assert state["learner"]["updates"]==1 and state["learner"]["optimizer_steps"]==5
            assert checkpoint["parallel_collection_group"]["status"]=="physical_complete_uncheckpointed"
            self.group.update(status="checkpointed",checkpoint_path=str(path),checkpoint_sha256=file_sha256(path))
            self.persist();(self.output/"ORDINARY-RESUME-BLOCKED.json").unlink()
        def close(self):pass

    monkeypatch.setattr(vector_module,"ProductionRobotwinVectorEnv",Vector)
    output=tmp_path/"n6"/"experiment"
    assert driver.main(parallel_args(b,output))==0
    final=torch.load(output/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    old=before["states"]["ppo"]["learner"];new=final["states"]["ppo"]["learner"]
    assert old["updates"]==1 and new["updates"]==2
    assert {int(value["step"]) for value in old["optimizer"]["state"].values()}=={5}
    assert {int(value["step"]) for value in new["optimizer"]["state"].values()}=={10}
    assert any(value["exp_avg"].abs().sum()>0 for value in old["optimizer"]["state"].values())
    assert artifact_digest(old["policy"])!=artifact_digest(new["policy"])
    assert not final["states"]["ppo"]["pending_rollouts"]
    assert final["stages"]["iteration:1"]==before["stages"]["iteration:1"]
    assert "stage:iteration:2:complete" not in final["completed_units"]
    assert final["history"]==before["history"]
    monkeypatch.setattr(audit,"RUNTIME","b"*40)
    monkeypatch.setattr(audit,"LINEAGE","a"*40)
    monkeypatch.setattr(audit,"ledger_root",lambda method:b["root"])
    monkeypatch.setattr(preflight,"load_driver",lambda worktree:driver)
    plan=dict(worktree=str(tmp_path/"qualified-cpu-runtime"),methods=dict(ppo=dict(
        manifest=bind(b["manifest"]),checkpoint=bind(ready/"coordinator-recovery.pt"),
        frozen_motor=bind(ready/"frozen-motor.json"),output=str(output))))
    return dict(audit=audit,plan=plan,output=output,before=before,final=final,vector=instances[0])


def test_genuine_trained_n6_driver_passes_complete_after_audit(audited_round):
    case=audited_round;report=case["audit"].audit_arm(case["plan"],"ppo")
    assert report["completed_iterations"]==2 and report["optimizer_step_increment"]==5
    assert report["source_actions"]==report["actual_actions"]==24
    assert report["corpus"]["completed_chunks"]==24 and report["corpus"]["hashed_raw_artifacts"]>0
    assert report["trained_boundary"]["optimizer_steps"]==10
    assert report["input_due_stages_completed"]==[]
    assert case["vector"].group["checkpoint_sha256"]
    # Exercise the real pickle/JSON shape mismatch that the audit normalizes.
    assert isinstance(case["final"]["contract"]["noise_shape"],tuple)
    summary=json.loads((case["output"]/"summary.json").read_text())
    assert isinstance(summary["contract"]["noise_shape"],list)


@pytest.mark.parametrize("tamper",["summary_contract","missing_ack"])
def test_after_audit_rejects_corrupted_contract_or_missing_group_ack(audited_round,tamper):
    case=audited_round
    if tamper=="summary_contract":
        path=case["output"]/"summary.json";value=json.loads(path.read_text())
        value["contract"]["norm_stats"]="/changed-normalization";write(path,value)
        message="scientific contract"
    else:
        path=case["vector"].journal;value=json.loads(path.read_text())
        value.pop("checkpoint_sha256");write(path,value);message="group ACK"
    with pytest.raises(ValueError,match=message):case["audit"].audit_arm(case["plan"],"ppo")
