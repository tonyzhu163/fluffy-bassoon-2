"""Pilot integration guards; execute on allocated Linux, not the editor host."""
import importlib.util
import json
import random
from copy import deepcopy
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
    assert original.ppo_max_update_kl is None
    assert driver.parse_args(required+["--ppo-max-update-kl",".1"]).ppo_max_update_kl==.1
    for value in ("0","-1","nan","inf"):
        with pytest.raises(SystemExit): driver.parse_args(required+["--ppo-max-update-kl",value])
    pilot=driver.parse_args(required+["--campaign-root","/campaign","--record-corpus"])
    assert (pilot.source_steps,pilot.ppo_trials_per_update,pilot.native_episodes,
            pilot.source_validation_episodes)==(272_000,8,1,4)
    with pytest.raises(SystemExit): driver.parse_args(required+["--campaign-root","/campaign"])
    with pytest.raises(SystemExit): driver.parse_args(required+["--final-observation-only"])


class GuardLearner:
    """A partial optimizer step mutates weights, moments, counters and RNG."""
    def __init__(self,metrics,fail=False):
        self.weight=torch.tensor([1.])
        self.updates=self.optimizer_steps=0
        self.optimizer=dict(step=0,exp_avg=torch.tensor([0.]))
        self.sampler=np.random.default_rng(41)
        self.metrics=metrics; self.fail=fail
    def state_dict(self):
        return dict(weight=self.weight,updates=self.updates,optimizer_steps=self.optimizer_steps,
                    optimizer=self.optimizer,sampler=self.sampler.bit_generator.state)
    def load_state_dict(self,state):
        self.weight=state["weight"].clone()
        self.updates=state["updates"]; self.optimizer_steps=state["optimizer_steps"]
        self.optimizer=deepcopy(state["optimizer"])
        self.sampler.bit_generator.state=state["sampler"]
    def update(self,rollouts):
        self.weight.add_(2.)
        self.updates+=1; self.optimizer_steps+=5
        self.optimizer["step"]+=5; self.optimizer["exp_avg"].add_(7.)
        random.random(); np.random.random(); torch.rand(1); self.sampler.random()
        if self.fail: raise ArithmeticError("exception after a partial optimizer step")
        return dict(self.metrics)


@pytest.mark.parametrize("metrics,fail",[
    ({"approximate_kl":.11,"gradient_norm":2.},False),
    ({"approximate_kl":float("nan"),"gradient_norm":2.},False),
    ({"approximate_kl":.01,"gradient_norm":float("inf")},False),
    ({"gradient_norm":2.},False),
    ({"approximate_kl":.01,"gradient_norm":2.},True),
])
def test_rejected_update_restores_complete_state_and_rng_and_retains_forensics(metrics,fail):
    random.seed(17); np.random.seed(18); torch.manual_seed(19)
    learner=GuardLearner(metrics,fail)
    pending=[{"observations":torch.tensor([11.])}]
    before=deepcopy(learner.state_dict())
    process_rng=driver.capture_rng()
    expected=(random.random(),np.random.random(),torch.rand(1))
    driver.restore_rng(process_rng)
    with pytest.raises(driver.PPOUpdateRejected) as rejected:
        driver.guarded_ppo_update(learner,pending,max_kl=.1)
    torch.testing.assert_close(learner.weight,before["weight"])
    torch.testing.assert_close(learner.optimizer["exp_avg"],before["optimizer"]["exp_avg"])
    assert learner.updates==learner.optimizer_steps==learner.optimizer["step"]==0
    assert learner.sampler.bit_generator.state==before["sampler"]
    assert random.random()==expected[0] and np.random.random()==expected[1]
    torch.testing.assert_close(torch.rand(1),expected[2],rtol=0,atol=0)
    assert len(pending)==1 and pending[0]["observations"].item()==11.
    error=rejected.value
    torch.testing.assert_close(error.rejected_state["weight"],torch.tensor([3.]))
    assert error.rejected_state["updates"]==1 and error.rejected_state["optimizer_steps"]==5
    assert error.rejected_state["optimizer"]["step"]==5
    assert error.rejected_rng is not None
    assert error.cause_type==("ArithmeticError" if fail else "ValueError")


def test_guard_accepts_a_finite_update_at_the_declared_limit():
    learner=GuardLearner({"approximate_kl":.1,"gradient_norm":2.})
    result=driver.guarded_ppo_update(learner,[{}],max_kl=.1)
    assert result["approximate_kl"]==.1
    assert learner.weight.item()==3. and learner.updates==1 and learner.optimizer_steps==5
    assert learner.optimizer["step"]==5


def test_disabled_guard_preserves_original_update_without_snapshot_overhead():
    class Original:
        def state_dict(self): raise AssertionError("Disabled guard must not snapshot")
        def update(self,rollouts): return {"approximate_kl":99.}
    assert driver.guarded_ppo_update(Original(),[{}])=={"approximate_kl":99.}


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


def test_stage_configuration_requires_exact_ordered_campaign_limits():
    required=["--source-commit","a"*40,"--model-path","/model","--norm-stats","/norm",
        "--robotwin-root","/robotwin","--rlinf-root","/rlinf","--output","/new",
        "--source-steps","128000","--source-milestones","32000","64000","128000",
        "--eval-tasks","adjust_bottle","move_stapler_pad"]
    with pytest.raises(SystemExit): driver.parse_args(required)
    args=driver.parse_args(required+["--campaign-root","/campaign","--record-corpus"])
    assert args.source_milestones==[32000,64000,128000]
    assert args.ppo_trials_per_update==8 and args.source_validation_episodes==4
    with pytest.raises(SystemExit):
        driver.parse_args(required+["--campaign-root","/campaign","--record-corpus",
            "--source-milestones","64000","32000","128000"])
    with pytest.raises(SystemExit):
        driver.parse_args(required+["--campaign-root","/campaign","--record-corpus",
            "--source-milestones","32000","64000"])


def test_exact_stage_barriers_preserve_pending_policy_rng_and_resumed_native_noise(tmp_path,monkeypatch):
    import fluffy_bassoon.rl.robotwin_noise as runtime_module
    stream=[0]
    stop_after_first_native=[True]
    source_tokens=[]
    native_noise={}
    current_phase=[None]
    source_sac_metadata=[]

    class Runtime(FakeEnv):
        observation_dim=2
        noise_shape=(4,2)
        contract={}
        def __init__(self,**kwargs):
            super().__init__(); self.native_resets=0
        def set_recording_context(self,context):
            super().set_recording_context(context)
            current_phase[0]=self.context["phase"]
        def reset(self,task,seed):
            self.task=task
            if self.context["phase"]=="native_before_target":
                if self.native_resets and stop_after_first_native[0]:
                    raise driver.WallBudgetReached("Injected interruption after first completed native unit")
                self.native_resets+=1
            return super().reset(task,seed)
        def step(self,noise,**kwargs):
            if self.context["method"]=="native":
                key=(self.context["stage_source_actions"],self.context["phase"],self.task,self.elapsed_steps)
                assert key not in native_noise  # Completed units never execute twice.
                native_noise[key]=np.asarray(noise).copy()
                stream[0]+=1  # Evaluation is allowed to consume its own global stream.
            return super().step(noise,**kwargs)
        def record_decision_metadata(self,data):
            if self.context["phase"]=="source" and self.context["method"]=="sac":
                source_sac_metadata.append(deepcopy(data))
        def check_noise_parity(self,noise): return dict(passed=True,max_abs_error=0.)
        def frozen_fingerprint(self): return "motor-fixed"
        def close(self): pass

    class Collector:
        def __init__(self,*args,**kwargs):
            stream[0]+=10  # Cloning a controller must not perturb source sampling.
            self.policy=torch.nn.Linear(2,2)
            with torch.no_grad(): self.policy.weight.fill_(1.); self.policy.bias.zero_()
            self.updates=self.optimizer_steps=self.primitive_steps=self.chunks=0
        def state_dict(self):
            return dict(updates=self.updates,optimizer_steps=self.optimizer_steps,
                primitive_steps=self.primitive_steps,chunks=self.chunks,weights=self.policy.state_dict())
        def load_state_dict(self,state):
            self.policy.load_state_dict(state["weights"])
            for key in ("updates","optimizer_steps","primitive_steps","chunks"): setattr(self,key,state[key])
        def collect_trial(self,env,task,seeds,*,primitive_budget=None,episode_callback=None,
                          decision_callback=None,**kwargs):
            stream[0]+=1
            if env.context["phase"]=="source": source_tokens.append(stream[0])
            env.reset(task,seeds[0])
            if decision_callback: decision_callback(dict(history_index=0))
            _,reward,end,trunc,info=env.step(np.zeros(8),max_primitive_steps=primitive_budget)
            row=dict(task=task,episode=1,seed=seeds[0],success=False,return_=reward,
                primitive_steps=info["primitive_steps"],chunks=1)
            if episode_callback: episode_callback(row)
            return {"fake":torch.tensor([stream[0]])},dict(task=task,episodes=[row],chunks=1,
                primitive_steps=info["primitive_steps"],budget_truncated=bool(info["budget_truncated"]))
        def replay_error(self,rollout): return dict(log_prob_max_error=0.,value_max_error=0.)
        def update(self,rollouts):
            self.updates+=1; self.optimizer_steps+=1
            with torch.no_grad(): self.policy.weight.add_(1.)
            return dict(updates=self.updates)

    class SACCollector:
        """Small stateful dispatch fixture; real SAC math has independent tests."""
        fixture_warmup=1
        fixture_batch_size=1
        def __init__(self,observation_dim,action_dim,*,config,seed=0,action_limit=1.5,**kwargs):
            assert (config.warmup_chunks,config.batch_size,config.utd)==(500,256,20)
            stream[0]+=10
            self.actor=torch.nn.Linear(2,2)
            with torch.no_grad(): self.actor.weight.fill_(1.); self.actor.bias.zero_()
            self.config=config; self.action_dim=action_dim; self.action_limit=action_limit
            self.environment_steps=self.primitive_steps=self.updates=0
            self.replay=[]; self.target_mode=None; self.target_start=None
            self.rng=np.random.default_rng(seed)
            self.torch_rng=torch.Generator().manual_seed(seed)
        @property
        def optimizer_steps(self): return self.updates*3
        @property
        def warming_up(self):
            return self.target_mode is None and self.environment_steps<self.fixture_warmup
        def _sample(self,warmup):
            stream[0]+=1
            if current_phase[0]=="source": source_tokens.append(stream[0])
            if warmup: return self.rng.uniform(-self.action_limit,self.action_limit,self.action_dim).astype(np.float32)
            return np.tanh(self.rng.normal(size=self.action_dim)).astype(np.float32)*self.action_limit
        def sample_warmup(self,observation): return self._sample(True)
        def act(self,observation,deterministic=False): return self._sample(False)
        def observe(self,observation,action,reward,following,end,trunc,*,primitive_steps):
            self.environment_steps+=1; self.primitive_steps+=primitive_steps
            self.replay.append(dict(reward=reward,primitive_steps=primitive_steps))
        def update(self,gradient_steps):
            assert gradient_steps==20  # Driver keeps the actual scientific recipe.
            ready=(bool(self.replay) if self.target_mode else
                   self.environment_steps>self.fixture_warmup and len(self.replay)>=self.fixture_batch_size)
            if not ready: return {}
            self.updates+=1  # One fake dispatch event, not simulated SAC math.
            with torch.no_grad(): self.actor.weight.add_(.5)
            return dict(updates=self.updates)
        def state_dict(self):
            return dict(actor=self.actor.state_dict(),environment_steps=self.environment_steps,
                primitive_steps=self.primitive_steps,updates=self.updates,replay=deepcopy(self.replay),
                target_mode=self.target_mode,target_start=deepcopy(self.target_start),
                rng=deepcopy(self.rng.bit_generator.state),torch_rng=self.torch_rng.get_state())
        def load_state_dict(self,state):
            self.actor.load_state_dict(state["actor"])
            for name in ("environment_steps","primitive_steps","updates","target_mode"):
                setattr(self,name,state[name])
            self.replay=deepcopy(state["replay"]); self.target_start=deepcopy(state["target_start"])
            self.rng.bit_generator.state=deepcopy(state["rng"])
            self.torch_rng.set_state(state["torch_rng"])
        def contract(self): return dict(fixture=True,fixture_warmup=self.fixture_warmup)
        def begin_target_adaptation(self,*,replay_mode):
            self.target_mode=replay_mode
            self.target_start=dict(updates=self.updates,environment_steps=self.environment_steps)
            self.replay=[]
            return dict(replay_mode=replay_mode)
        def target_counters(self):
            return dict(updates=self.updates-self.target_start["updates"],
                        environment_steps=self.environment_steps-self.target_start["environment_steps"])

    monkeypatch.setattr(runtime_module,"RobotwinNoiseEnv",Runtime)
    monkeypatch.setattr(driver,"RecurrentNoisePPO",Collector)
    monkeypatch.setattr(driver,"NoiseSAC",SACCollector)
    monkeypatch.setattr(driver.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(driver.torch.cuda,"manual_seed_all",lambda seed:None)
    monkeypatch.setattr(driver,"capture_rng",lambda:{"stream":stream[0]})
    monkeypatch.setattr(driver,"restore_rng",lambda state:stream.__setitem__(0,state["stream"]))
    common=["--source-commit","a"*40,"--model-path","/model","--norm-stats","/norm",
        "--robotwin-root","/robotwin","--rlinf-root","/rlinf","--methods","ppo","sac","ppo_reset",
        "--source-steps","3","--source-milestones","1","2","3","--ppo-trials-per-update","8",
        "--native-episodes","1","--eval-tasks","adjust_bottle","move_stapler_pad","--record-corpus"]
    first=tmp_path/"interrupted1"
    campaign=tmp_path/"interrupted-campaign"
    assert driver.main(common+["--campaign-root",str(campaign),"--output",str(first)])==0
    saved=torch.load(first/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    assert saved["rng"]=={"stream":33}
    assert len(saved["completed_units"])==19  # 18 validation units + one native episode.
    assert all(len(s["pending_rollouts"])==1 and s["learner"]["updates"]==0
               for method,s in saved["states"].items() if method!="sac")
    assert saved["states"]["sac"]["learner"]["updates"]==0
    assert len(saved["states"]["sac"]["learner"]["replay"])==1
    stop_after_first_native[0]=False
    second=tmp_path/"interrupted2"
    assert driver.main(common+["--campaign-root",str(campaign),"--output",str(second),"--resume",str(first)])==0
    completed=torch.load(second/"coordinator-recovery.pt",map_location="cpu",weights_only=False)
    resumed_summary=json.loads((second/"summary.json").read_text())
    for stage in (1,2,3):
        report=completed["stages"][str(stage)]
        assert report["status"]=="complete" and set(report["source_steps"].values())=={stage}
        expected_updates=1 if stage==3 else 0
        expected_pending=0 if stage==3 else stage
        assert report["source_updates"]==dict(ppo=expected_updates,sac=stage-1,ppo_reset=expected_updates)
        assert report["pending_trials"]==dict(ppo=expected_pending,sac=0,ppo_reset=expected_pending)
        for method,snapshot in report["snapshots"].items():
            point=torch.load(snapshot["path"],map_location="cpu",weights_only=False)
            if method=="sac":
                assert len(point["pending_rollouts"])==0
                assert point["learner"]["updates"]==stage-1
                assert point["learner"]["environment_steps"]==len(point["learner"]["replay"])==stage
                torch.testing.assert_close(point["learner"]["actor"]["weight"],
                                           torch.full((2,2),1.+.5*(stage-1)))
                continue
            assert len(point["pending_rollouts"])==expected_pending
            torch.testing.assert_close(point["learner"]["weights"]["weight"],
                                       torch.full((2,2),float(1+expected_updates)))
    resumed_tokens=list(source_tokens)
    resumed_noise={key:value.copy() for key,value in native_noise.items()}
    assert resumed_tokens==list(range(31,40))
    assert [row["source_warmup"] for row in source_sac_metadata]==[True,False,False]
    assert [row["optimizer_updates"] for row in source_sac_metadata]==[0,0,1]
    progress=[json.loads(line) for output in (first,second)
              for line in (output/"progress.jsonl").read_text().splitlines()]
    cuts=[(row["stage_source_actions"],row["external_stage_budget_cut"])
          for row in progress if row["stage"]=="source_trial" and row["method"]=="sac"]
    assert cuts==[(1,True),(2,True),(3,False)]
    source_sac_rows=[json.loads(line) for output in (first,second)
                     for line in (output/"episodes.jsonl").read_text().splitlines()
                     if json.loads(line)["phase"]=="source" and json.loads(line)["method"]=="sac"]
    assert [row["stage_source_actions"] for row in source_sac_rows]==[1,2,3]
    online=[row for row in resumed_summary["target"] if row["method"]=="sac_online"]
    frozen=[row for row in resumed_summary["target"] if row["method"]=="sac_frozen"]
    assert len(online)==len(frozen)==6
    assert all(row["target_update_costs"]["updates"]>0 for row in online)
    assert all(row["target_update_costs"] is None for row in frozen)
    stream[0]=0; source_tokens.clear(); native_noise.clear(); source_sac_metadata.clear()
    baseline=tmp_path/"uninterrupted"
    assert driver.main(common+["--campaign-root",str(tmp_path/"uninterrupted-campaign"),
                              "--output",str(baseline)])==0
    assert source_tokens==resumed_tokens and set(native_noise)==set(resumed_noise)
    for key in native_noise: np.testing.assert_array_equal(native_noise[key],resumed_noise[key])
    baseline_summary=json.loads((baseline/"summary.json").read_text())
    assert baseline_summary["campaign_budget"]["actual_actions"]==resumed_summary["campaign_budget"]["actual_actions"]
    assert len(baseline_summary["source_validation"])==54
    assert len(baseline_summary["target"])==36
