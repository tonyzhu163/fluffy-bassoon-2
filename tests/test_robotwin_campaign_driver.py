"""Pilot integration guards; execute on allocated Linux, not the editor host."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget


path=Path(__file__).resolve().parents[1]/"scripts/diagnostics/robotwin_noise_compare.py"
spec=importlib.util.spec_from_file_location("robotwin_campaign_driver",path)
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FakeEnv:
    native_horizon=4
    episode_limit=10
    def __init__(self):
        self.calls=[]; self.elapsed_steps=0
        self.last_reset_info={"reset_policy_actions":0}
    def reset(self,task,seed):
        self.elapsed_steps=0
        return np.zeros(2,np.float32)
    def set_recording_context(self,value):
        self.context=dict(getattr(self,"context",{}),**value)
    def step(self,noise,*,max_primitive_steps=None):
        self.calls.append(max_primitive_steps)
        count=min(self.native_horizon,self.episode_limit-self.elapsed_steps)
        if max_primitive_steps is not None: count=min(count,max_primitive_steps)
        self.elapsed_steps+=count
        natural=self.elapsed_steps==self.episode_limit
        cut=max_primitive_steps is not None and count>=max_primitive_steps and not natural
        return np.zeros(2,np.float32),0.,False,natural or cut,dict(
            primitive_steps=count,budget_truncated=cut)


def ledger(root,total=30):
    return CampaignBudget(root,campaign_id="test",total_actions=total,
        source_caps={"ppo":total},contract={"test":True},run_id="run1",source_commit="a"*40)


def contextual(env,budget):
    proxy=driver.CampaignEnv(env,budget,run_id="run1")
    proxy.set_recording_context(dict(phase="source",method="ppo",trial_id="trial1",policy_version="ppo0"))
    proxy.reset("place_phone_stand",17)
    return proxy


def test_reservation_does_not_turn_each_native_chunk_into_a_time_limit(tmp_path):
    raw=FakeEnv()
    with ledger(tmp_path) as budget:
        env=contextual(raw,budget)
        _,_,_,trunc,info=env.step(np.zeros(8),max_primitive_steps=20)
        assert not trunc and not info["budget_truncated"]
        assert raw.calls==[20]
        env.step(np.zeros(8))
        assert raw.calls[-1] is None
        assert budget.snapshot()["actual_actions"]==8


def test_campaign_boundary_clips_and_bootstraps_instead_of_overexecuting(tmp_path):
    raw=FakeEnv()
    with ledger(tmp_path,total=6) as budget:
        env=contextual(raw,budget)
        env.step(np.zeros(8))
        _,_,end,trunc,info=env.step(np.zeros(8))
        assert not end and trunc and info["budget_truncated"]
        assert raw.calls==[None,2]
        assert budget.snapshot()["actual_actions"]==6
        with pytest.raises(driver.CampaignBudgetReached): env.step(np.zeros(8))
        assert len(raw.calls)==2


def test_native_failure_leaves_conservative_preexecution_intent(tmp_path):
    raw=FakeEnv()
    with ledger(tmp_path) as budget:
        env=contextual(raw,budget)
        raw.step=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("native crash"))
        with pytest.raises(RuntimeError,match="native crash"): env.step(np.zeros(8))
        assert budget.snapshot()["pending_actions"]==4
    with CampaignBudget(tmp_path,campaign_id="test",total_actions=30,
            source_caps={"ppo":30},contract={"test":True},run_id="run2",source_commit="a"*40) as recovered:
        assert recovered.snapshot()["uncertain_actions"]==4


def test_pilot_defaults_are_separate_from_original_recipe():
    required=["--source-commit","a"*40,"--model-path","/model","--norm-stats","/norm",
              "--robotwin-root","/robotwin","--rlinf-root","/rlinf","--output","/new"]
    original=driver.parse_args(required)
    assert (original.source_steps,original.ppo_trials_per_update,original.native_episodes,
            original.source_validation_episodes)==(10_240_000,256,2,0)
    pilot=driver.parse_args(required+["--campaign-root","/campaign","--record-corpus"])
    assert (pilot.source_steps,pilot.ppo_trials_per_update,pilot.native_episodes,
            pilot.source_validation_episodes)==(272_000,8,1,4)
    with pytest.raises(SystemExit): driver.parse_args(required+["--campaign-root","/campaign"])
    with pytest.raises(SystemExit): driver.parse_args(required+["--final-observation-only"])


def test_capture_admission_requires_real_complete_matching_receipt(tmp_path):
    path=tmp_path/"qualification.json"
    base=dict(passed=True,status="passed",timing_comparison_eligible=True,
        same_unchanged_frozen_motor=True,source_commit="b"*40,
        comparisons={str(i):dict(passed=True) for i in range(4)},
        modes={"final_only":dict(final_observation_only=True,native_noise_parity=dict(passed=True),
            runtime_contract=dict(model_path="/model",norm_stats_path="/norm"))})
    path.write_text(json.dumps(base))
    assert driver.validate_capture_qualification(path,model_path="/model",norm_stats="/norm")["source_commit"]=="b"*40
    with pytest.raises(ValueError,match="motor/runtime"):
        driver.validate_capture_qualification(path,model_path="/different",norm_stats="/norm")
    base["comparisons"]["2"]["passed"]=False
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError,match="comparisons"):
        driver.validate_capture_qualification(path,model_path="/model",norm_stats="/norm")


def test_policy_archive_distinguishes_independent_same_counter_policies(tmp_path):
    archive=driver.BehaviorPolicyArchive(tmp_path)
    class Learner:
        def __init__(self,weight):
            self.policy=torch.nn.Linear(2,2); self.updates=0
            with torch.no_grad():
                self.policy.weight.fill_(weight); self.policy.bias.zero_()
    first= Learner(1.)
    second=Learner(2.)
    a=archive.identify(first,"ppo")
    b=archive.identify(second,"ppo")
    assert a["policy_sha256"]!=b["policy_sha256"]
    assert archive.identify(first,"ppo")==a
    assert len(list(tmp_path.glob("*.pt")))==2
    payload=torch.load(a["policy_checkpoint"],map_location="cpu",weights_only=False)
    torch.testing.assert_close(payload["state_dict"]["weight"],first.policy.weight)


class TinyPPOEnv:
    native_horizon=2
    def reset(self,task,seed):
        self.count=0
        return np.array([seed,0],np.float32)
    def step(self,noise):
        self.count+=1
        return np.array([1,self.count],np.float32),float(self.count==2),self.count==2,False,dict(primitive_steps=2)


def test_ppo_metadata_is_fresh_and_matches_every_saved_likelihood():
    model=RecurrentNoisePPO(2,(1,2),device="cpu",seed=7,config=PPOConfig(
        width=8,layers=1,heads=2,memory_length=8,train_episodes=2,episode_limit=4))
    emitted=[]
    rollout,result=model.collect_trial(TinyPPOEnv(),"scan_object",[1,2],decision_callback=emitted.append)
    assert len(emitted)==4 and [row["history_index"] for row in emitted]==list(range(4))
    for index,row in enumerate(emitted):
        np.testing.assert_array_equal(row["pre_squash_actions"],rollout["raw_actions"][index,0].numpy())
        np.testing.assert_array_equal(row["old_log_prob"],rollout["log_probs"][index,0].numpy())
        np.testing.assert_array_equal(row["old_value"],rollout["values"][index,0].numpy())
        np.testing.assert_array_equal(row["policy_input"],rollout["observations"][index,0].numpy())
    assert emitted[2]["previous_reward"]==1 and emitted[2]["episode_boundary"]
    assert result["primitive_steps"]==8 and model.updates==0


def test_loaded_capture_gate_rejects_mutated_weights_and_normalization():
    runtime=dict(norm_stats_sha256="a"*64,rlinf_commit="b"*40,robotwin_commit="c"*40)
    qualification=dict(runtime_contract=runtime,frozen_fingerprint="d"*64)
    driver.validate_loaded_capture_runtime(qualification,runtime,"d"*64)
    with pytest.raises(ValueError,match="norm_stats_sha256"):
        driver.validate_loaded_capture_runtime(qualification,dict(runtime,norm_stats_sha256="e"*64))
    with pytest.raises(ValueError,match="frozen motor"):
        driver.validate_loaded_capture_runtime(qualification,runtime,"e"*64)


def test_completed_validation_and_target_units_do_not_reexecute_on_resume(tmp_path,monkeypatch):
    """Exercise driver resume with physical calls counted outside its cursor."""
    import fluffy_bassoon.rl.robotwin_noise as runtime_module
    executions=[]
    interrupt_after_first_validation=[True]

    class Runtime(FakeEnv):
        observation_dim=2
        noise_shape=(4,2)
        contract={}
        def __init__(self,**kwargs):
            super().__init__()
            self.validation_resets=0
        def reset(self,task,seed):
            if self.context["phase"]=="source_validation":
                if self.validation_resets and interrupt_after_first_validation[0]:
                    raise driver.WallBudgetReached("Injected allocation boundary after a completed validation unit")
                self.validation_resets+=1
            return super().reset(task,seed)
        def step(self,*args,**kwargs):
            result=super().step(*args,**kwargs)
            executions.append(result[-1]["primitive_steps"])
            return result
        def record_decision_metadata(self,data): pass
        def check_noise_parity(self,noise): return dict(passed=True,max_abs_error=0.)
        def frozen_fingerprint(self): return "motor-fixed"
        def close(self): pass

    class Collector:
        def __init__(self,*args,**kwargs):
            self.policy=torch.nn.Linear(2,2)
            self.updates=self.optimizer_steps=self.primitive_steps=self.chunks=0
        def state_dict(self):
            return dict(updates=self.updates,optimizer_steps=self.optimizer_steps,
                        primitive_steps=self.primitive_steps,chunks=self.chunks,
                        weights=self.policy.state_dict())
        def load_state_dict(self,state):
            self.policy.load_state_dict(state["weights"])
            for key in ("updates","optimizer_steps","primitive_steps","chunks"):
                setattr(self,key,state[key])
        def collect_trial(self,env,task,seeds,*,primitive_budget=None,episode_callback=None,
                          decision_callback=None,**kwargs):
            env.reset(task,seeds[0])
            if decision_callback: decision_callback(dict(history_index=0))
            _,reward,end,trunc,info=env.step(np.zeros(8),max_primitive_steps=primitive_budget)
            row=dict(task=task,episode=1,seed=seeds[0],success=False,return_=reward,
                     primitive_steps=info["primitive_steps"],chunks=1)
            if episode_callback: episode_callback(row)
            return {"fake":torch.zeros(1)},dict(task=task,episodes=[row],chunks=1,
                primitive_steps=info["primitive_steps"])
        def replay_error(self,rollout): return dict(log_prob_max_error=0.,value_max_error=0.)
        def update(self,rollouts):
            self.updates+=1; self.optimizer_steps+=1
            return dict(updates=self.updates)

    monkeypatch.setattr(runtime_module,"RobotwinNoiseEnv",Runtime)
    monkeypatch.setattr(driver,"RecurrentNoisePPO",Collector)
    monkeypatch.setattr(driver.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(driver.torch.cuda,"manual_seed_all",lambda seed:None)
    monkeypatch.setattr(driver,"capture_rng",lambda:{"fixed":True})
    monkeypatch.setattr(driver,"restore_rng",lambda state:None)
    common=["--source-commit","a"*40,"--model-path","/model","--norm-stats","/norm",
        "--robotwin-root","/robotwin","--rlinf-root","/rlinf","--methods","ppo",
        "--source-steps","1","--ppo-trials-per-update","1","--native-episodes","1",
        "--source-validation-episodes","1","--eval-tasks","adjust_bottle",
        "--eval-episodes","1","--campaign-root",str(tmp_path/"campaign"),"--record-corpus"]
    first=tmp_path/"run1"
    assert driver.main(common+["--output",str(first)])==0
    count=len(executions)
    assert count==2  # One source and one completed frozen validation unit.
    first_saved=torch.load(first/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    assert len(first_saved["completed_units"])==len(first_saved["history"])==1
    interrupt_after_first_validation[0]=False
    second=tmp_path/"run2"
    assert driver.main(common+["--output",str(second),"--resume",str(first)])==0
    # Six validation trials, three target lesions, two 3-chunk native episodes,
    # and the single source call, each executed exactly once across allocations.
    assert len(executions)==16
    saved=torch.load(second/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    assert len(saved["completed_units"])==len(saved["history"])==11
    third=tmp_path/"run3"
    assert driver.main(common+["--output",str(third),"--resume",str(second)])==0
    assert len(executions)==16
