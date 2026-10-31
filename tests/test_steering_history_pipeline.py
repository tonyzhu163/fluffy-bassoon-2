"""Allocated Linux wiring only: real raw writer/runner, tiny policies, no robot claim."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import socket

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import steering_history as sh
from fluffy_bassoon.meta_rl.steering_history_prefix import build_prefix
from fluffy_bassoon.meta_rl.steering_history_readout import (FORMAT, FrozenReference,
    admit_readout_budget, matrix_schedule, run_readout,stationary_reference_factory)
from fluffy_bassoon.meta_rl.steering_condition_collection import BoundedTeacherRecovery
from fluffy_bassoon.meta_rl.robotwin_steering_screen import run_complete_episode, paired_reset_fields, PHASE_CAPS
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, load_dataset
from fluffy_bassoon.meta_rl.vla_history import sample_batch
from fluffy_bassoon.meta_rl.vla_history_train import supervised_update
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC
from fluffy_bassoon.meta_rl.robotwin_history_corpus import parameter_digest
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter
from test_steering_history import source, profile, ref, json_ref, raw_observation


class TinyCanonicalControl:
    """Test world; actual existing runner/corpus path with one valid action per episode."""
    native_horizon = 32
    def __init__(self, root):
        self.env = SimpleNamespace(corpus_writer=RobotwinCorpusWriter(root, {"source_commit": "f"*40}))
        self.active, self.elapsed_steps, self.success = False, 0, True
        self.previous_agent_command = np.zeros(14, np.float32)
        self.coefficients = np.zeros(4, np.float32)
        self.calls, self.pending = {}, {}

    def episode_limit_for_task(self, task): return 600
    def set_recording_context(self, metadata): self.env.corpus_writer.set_context(metadata)
    def set_coefficients(self, arm, coefficients):
        self.arm = arm; self.coefficients = np.clip(np.asarray(coefficients, np.float32), -1., 1.)
    def record_decision_metadata(self, metadata): self.pending = deepcopy(metadata)
    def student_observation(self, native): return np.r_[native, self.previous_agent_command].astype(np.float32)
    def reset(self, task, seed, *, motor_seed, paired_instruction=None):
        key = (task, seed)
        sampled = f"instruction-{self.calls.get(key, 0) % 2}"
        self.calls[key] = self.calls.get(key, 0)+1
        instruction = [sampled] if paired_instruction is None else list(paired_instruction)
        native = np.full(3, (seed % 100003)/100003 + int(instruction[0][-1]), np.float32)
        raw = raw_observation(seed % 200); raw["task_descriptions"] = instruction
        identity = hashlib.sha256(str(seed).encode()+instruction[0].encode()).hexdigest()
        self.last_reset_info = dict(task=task, requested_seed=seed, actual_seed=seed,
            observation_sha256=identity, features_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
            instruction=instruction, reset_policy_actions=0)
        self.previous_agent_command.fill(0.); self.elapsed_steps=0; self.active=True
        self.raw, self.features = raw, native
        episode = self.env.corpus_writer.begin_episode(raw, native, self.last_reset_info)
        self.last_reset_info["corpus_episode_id"] = episode
        return native.copy()
    def step(self, coefficients):
        self.set_coefficients(self.arm, coefficients)
        writer = self.env.corpus_writer
        metadata = dict(self.pending, student_previous_agent_command=self.previous_agent_command.copy())
        commands = np.tile(np.r_[self.coefficients, np.zeros(10, np.float32)], (32, 1))
        writer.begin_decision(np.zeros((32, 32), np.float32), commands, metadata)
        mask = np.zeros((1, 32), bool); mask[0, 0] = True
        writer.complete_native(next_raw=self.raw, next_features=self.features,
            raw_rewards=np.array([float(self.success)]), raw_terminations=np.array([True]),
            raw_truncations=np.array([False]), valid_action_mask=mask, reward=float(self.success),
            terminated=True, truncated=False, executed_steps=1, success=self.success, duration_seconds=.001)
        path = writer.finalize_learning_flags(reward=float(self.success), terminated=True, truncated=False,
            budget_truncated=False, episode_primitive_steps=1)
        self.previous_agent_command = commands[0].copy()
        self.active=False; self.elapsed_steps=1
        return self.features.copy(), float(self.success), True, False, dict(primitive_steps=1,
            success=self.success, corpus_chunk=path, budget_truncated=False)
    def snapshot(self): return dict(arm=getattr(self, "arm", "kv"), motor_rng=None,
                                   previous_agent_command=self.previous_agent_command.copy())


class Guard:
    maximum_episode_seconds = 1.
    def __init__(self): self.episodes=0
    def __call__(self): pass
    def admit_episode(self): self.episodes+=1


class ActualFrozenValidation:
    """Existing source SAC policy restored read-only, with the actual warmup rule."""
    def __init__(self,state,seed,root):
        c=state["contract"]
        self.learner=NoiseSAC(c["observation_dim"],4,action_limit=1.,device="cpu",seed=c["seed"],config=c["config"])
        self.learner.load_state_dict(state)
        self.learner.rng=np.random.default_rng(seed);self.learner.torch_rng.manual_seed(seed)
        self.path=root;self.meta=None
    def act(self,features):
        l=self.learner
        warmup=l.updates==0
        action=l.sample_warmup(features) if warmup else l.act(features,deterministic=False)
        self.meta=dict(source_warmup=warmup,optimizer_updates=l.updates,steering_coefficients=action.copy())
        if warmup:self.meta.update(policy_version="bounded_uniform_warmup",policy_checkpoint=None,policy_sha256=None)
        else:
            actor={k:v.detach().cpu().clone() for k,v in l.actor.state_dict().items()};sha=parameter_digest(actor)
            torch.save(dict(kind="sac_actor",parameter_sha256=sha,state_dict=actor),self.path)
            self.meta.update(policy_version=f"sac:update:{l.updates}:{sha}",policy_checkpoint=str(self.path),
                policy_sha256=sha,policy_checkpoint_sha256=sh.digest(self.path))
        return action
    def decision_metadata(self):return deepcopy(self.meta)
    def observe(self,*args):pass


def _budget(root, p, run_id):
    return CampaignBudget(root, campaign_id=p["shared_reservation"]["pilot_id"], total_actions=300000,
        source_caps={row["lineage_id"]: 10000 for row in p["lineage_slots"]},
        phase_caps=PHASE_CAPS,
        contract=dict(scope_sha256=p["scope_sha256"]), run_id=run_id, source_commit="f"*40)


@pytest.mark.parametrize("objectives", [["ad"], ["dpt"], ["ad", "dpt"]])
def test_shared_cli_and_driver_budget_admission_counts_all_readout_methods(tmp_path, monkeypatch, objectives):
    p = profile()
    schedule = matrix_schedule(p, objectives)
    with _budget(tmp_path/"admission-ledger", p, "admission-fixture") as budget:
        remaining = budget.remaining
        methods = []
        def observed_remaining(*, phase, method):
            methods.append(method)
            return remaining(phase=phase, method=method)
        monkeypatch.setattr(budget, "remaining", observed_remaining)
        admit_readout_budget(budget, schedule)
        assert methods == [objectives[0]+"-full"]
        # Another actual readout policy's cost reduces the same phase slice.
        spent = PHASE_CAPS["student_evaluation"] - schedule["maximum_actions"] + 1
        budget.import_receipt("native-readout-cost", spent, "a"*64,
            phase="student_evaluation", method="native")
        with pytest.raises(ValueError, match="complete student grid"):
            admit_readout_budget(budget, schedule)
        assert methods[-1] == objectives[0]+"-full"


def test_shared_readout_admission_rejects_unresolved_work(tmp_path):
    p = profile()
    with _budget(tmp_path/"pending-ledger", p, "pending-fixture") as budget:
        budget.reserve("pending-query", 1, phase="student_evaluation", method="ad-full",
            task=p["source_task"], episode="pending-query", chunk=0, policy_version="test-only")
        with pytest.raises(ValueError, match="Unresolved prior physical work"):
            admit_readout_budget(budget, matrix_schedule(p, ["ad", "dpt"]))


def test_actual_runner_executes_shared_prefix_grid_without_query_feedback_or_text_drift(tmp_path):
    torch.set_num_threads(1)
    p = profile()
    models = {}
    for objective in ("ad", "dpt"):
        for mode in ("full", "static"):
            torch.manual_seed(7)
            policy = HistoryNoisePolicy(HistoryConfig(17,4,width=8,layers=1,heads=2,context_length=16))
            models[objective+"-"+mode] = (policy, dict(contract=dict(mode=mode,
                train_metadata=dict(noise_contract=p["interface"]))))
    plan = dict(format=FORMAT,profile=p,objectives=["ad","dpt"],schedule=matrix_schedule(p,["ad","dpt"]),
        models={key:{} for key in models},assemblies={"ad":{},"dpt":{}},teachers={c:{} for c in p["conditions"]},
        source_actual_seeds=[],source_initial_observation_sha256=[])
    control, guard = TinyCanonicalControl(tmp_path/"corpus"), Guard()
    sink = BoundedTeacherRecovery(tmp_path/"recovery", guard=guard, control=control)
    def factory(condition, seed):
        return FrozenReference(dict(native_dim=3, device="cpu", mode="fixed_coefficients", coefficients=np.zeros(4,np.float32),
            identity="a"*64,condition=condition,lineage="fixture-reference",updates=0,reference={}),seed=seed)
    with _budget(tmp_path/"ledger",p,"readout-fixture") as budget:
        result = run_readout(plan, control=control, budget=budget, students=models, teacher_factory=factory,
            stationary_factory=factory,
            select_condition=lambda _:None, output=tmp_path/"matrix",run_id="readout-fixture",source_commit="f"*40,
            guard=guard,checkpoint=sink)
        assert result["status"] == "complete_four_block_mechanism_readout"
        assert len(result["completed_queries"]) == 88 and guard.episodes == 144
        assert budget.snapshot()["actual_actions"] == 144
        assert all(not r["query_feedback_reused"] for r in result["completed_queries"])
        assert sum("query_nll" in r for r in result["completed_queries"]) == 80


def test_builder_uses_real_sink_validation_input_and_published_state_identity(source, tmp_path):
    p, old = source
    p["teacher_gate"] = dict(paired_cases=2, minimum_success_rate=1., minimum_success_gain=.5)
    control, guard = TinyCanonicalControl(tmp_path/"validation-corpus"), Guard()
    publications = {}
    with _budget(tmp_path/"validation-ledger",p,"prefix-fixture") as budget:
        for index, lineage in enumerate(old["lineages"]):
            sink = BoundedTeacherRecovery(tmp_path/f"lineage-{index}",guard=guard,control=control)
            initial_state = torch.load(lineage["initial_recovery"]["path"],weights_only=False)
            final_state = torch.load(lineage["final_recovery"]["path"],weights_only=False)
            def inner(state):
                return dict(format="source_coefficient_teacher_v1",learner=state,active_episode=None,pending=None,
                    pending_update_cycles=0,updating=False,validation_done=[],validation_receipts={},
                    lineage=dict(method=lineage["lineage_id"],partition=lineage["partition"],learner_seed=lineage["seed"]),
                    profile=dict(task=p["source_task"],motor_sha256=p["interface"]["motor_sha256"],
                        mapping_sha256=p["interface"]["mapping_sha256"],interface_sha256=p["interface"]["control_interface_sha256"],scope_sha256=p["scope_sha256"]))
            first = sink("lineage_initial_full_state",inner(initial_state),immutable=True)
            receipts, baseline_records = {}, []
            for milestone, state in ((0,initial_state),(3,final_state)):
                payload = inner(state)
                validated_ref = sink("source_validation_input",payload,immutable=True)
                records = []
                for case in range(2):
                    spec = dict(id=f"validation-{index}-{milestone}-{case}",task=p["source_task"],case=case,
                        reset_seed=3300000+index*10000+case*997,motor_seed=71+case,arm="kv",method=lineage["lineage_id"],
                        condition_id=lineage["condition_id"],horizon=600,milestone=milestone,coefficients=[0.]*4)
                    if milestone: spec.update(paired_reset_fields(baseline_records[case]))
                    control.success = bool(milestone)
                    frozen=ActualFrozenValidation(state,spec["reset_seed"]+3000017,tmp_path/f"validated-actor-{index}-{milestone}-{case}.pt")
                    record = run_complete_episode(control,spec,budget,run_id="prefix-fixture",phase="source_validation",controller=frozen)
                    records.append({k:v for k,v in record.items() if k!="control_recovery"})
                if not milestone: baseline_records=records
                path=tmp_path/f"validation-{index}-{milestone}.json"
                measured=json_ref(path,dict(complete=True,phase="source_validation",source_checkpoint=validated_ref,
                    episode_ids=[r["spec"]["id"] for r in records],results=records,
                    behavior="bounded uniform iff completed optimizer updates0; otherwise frozen sampled actual actor"))
                receipts[str(milestone)]=dict(measured,input_checkpoint=validated_ref)
                # Actual sequence: validate input -> record receipt -> NEW wrapper,
                # with precisely the same learner actor/update state.
                payload["validation_done"].append(milestone);payload["validation_receipts"]=deepcopy(receipts)
                last=sink("published_prefix_full_state",payload,immutable=True)
                assert last["sha256"]!=validated_ref["sha256"]
            manifests=[e["raw_trace"] for e in lineage["episodes"]]
            original_refs=[];actual=[]
            for reference in manifests:
                path=sh.reference(reference);manifest=json.loads(path.read_text());reset=json.loads((path.parent/manifest["reset"]).read_text())
                original_refs.append(dict(root=lineage["corpus_root"],path=str(path.relative_to(lineage["corpus_root"])),sha256=reference["sha256"]))
                actual.append(reset["reset_info"])
            publications[lineage["lineage_id"]]=dict(lineage=dict(method=lineage["lineage_id"],partition=lineage["partition"],learner_seed=lineage["seed"]),
                condition_id=lineage["condition_id"],mapping_sha256=p["interface"]["mapping_sha256"],interface_sha256=p["interface"]["control_interface_sha256"],
                complete_episode_manifests=original_refs,actual_resets=actual,initial_full_state=first,final_full_state=last,
                actual_optimizer_updates=lineage["final_updates"],actual_primitive_steps=3,source_validation_receipts=receipts)
    prefix = build_prefix(p,publications,control_qualification=old["control_qualification"],output=tmp_path/"prefix.json",expected_hostname=socket.gethostname())
    assert len(prefix["lineages"])==6 and all(len(x["episodes"])==3 for x in prefix["lineages"])
    selected=next(x for x in prefix["lineages"] if x["lineage_id"]==p["teacher_lineage_ids"]["left"])
    q=sh.read_json(prefix["learning_qualifications"]["left"])
    assert q["teacher_recovery"]!=selected["final_recovery"]
    assert q["baseline_recovery"]!=selected["initial_recovery"]
    cross_dir=tmp_path/"original-cross"
    cross_dir.mkdir()
    cross_plan=json_ref(cross_dir/"plan.json",dict(original=True))
    original_cross=dict(json_ref(cross_dir/"result.json",dict(status="complete")),
        complete=True,phase="condition_cross",criterion_pass=False,recovery_ref={"original":"retained"})
    rebound=build_prefix(p,publications,control_qualification=old["control_qualification"],
        learning_qualifications=prefix["learning_qualifications"],teacher_qualifications=prefix["teacher_qualifications"],
        condition_cross=original_cross,expected_hostname=socket.gethostname())
    assert rebound["condition_cross"]==dict(result=original_cross,plan=cross_plan)
    assert "condition_cross" not in json.loads((tmp_path/"prefix.json").read_text())
    blind=stationary_reference_factory(dict(profile=p,blind_selection=prefix["blind_selection"]),
        {"dpt":dict(prefix=prefix)},device="cpu")
    left,right=blind("left",71),blind("right",71)
    assert left.definition["identity"]==right.definition["identity"]
    np.testing.assert_array_equal(left.act(np.zeros(3,np.float32)),right.act(np.zeros(3,np.float32)))
    sh.admit_prefix(p,prefix,objective="ad");sh.admit_prefix(p,prefix,objective="dpt")
    result=sh.export_prefix(p,prefix,tmp_path/"actual-export",objective="dpt",expected_hostname=socket.gethostname())
    trials,_=load_dataset(result["datasets"]["dpt"]["train"]["path"])
    assert len(trials)==4 and trials[0]["observations"].shape[1]==17


def test_execution_microbatches_preserve_global_valid_position_loss_and_one_step(source):
    p,prefix=source
    lineage=prefix["lineages"][0]
    data=sh.canonical_trial(lineage,p,require_learning=True)
    data.pop("primitive_steps");data.pop("reset_policy_actions");data["target_noise"]=data["executed_noise"].copy()
    torch.manual_seed(4)
    first=HistoryNoisePolicy(HistoryConfig(17,4,width=8,layers=1,heads=2,context_length=8))
    second=deepcopy(first)
    batch=sample_batch([data],first,np.random.default_rng(8),batch_size=5,max_episodes=3)
    a=supervised_update(first,torch.optim.SGD(first.parameters(),lr=.01),batch,mode="full")
    b=supervised_update(second,torch.optim.SGD(second.parameters(),lr=.01),batch,mode="full",execution_batch_size=2)
    assert a["valid_positions"]==b["valid_positions"] and a["logical_batch_size"]==b["logical_batch_size"]==5
    assert abs(a["nll"]-b["nll"])<1e-5
    for x,y in zip(first.parameters(),second.parameters()):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)
