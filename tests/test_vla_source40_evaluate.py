"""Portable support/query semantics using an event-logged simulator and policy."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_evaluate import (CostLedger, METHODS, SUPPORT_BUDGETS,
    TargetCase, evaluate_case, load_candidates)
from fluffy_bassoon.meta_rl.vla_source40_corpus import PI05_RECEIPT
from fluffy_bassoon.meta_rl.vla_source40_training import NOISE_CONTRACT
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, file_sha256


class Simulator:
    def __init__(self, *, fail_on_step=None):
        self.metadata = dict(suite="libero_90", task_id=57, initial_states=50, horizon=20,
            settle_steps=3, instruction="place the object", fresh_environment_on_reset=True, pid=100)
        self.events, self.episodes, self.steps, self.fail_on_step = [], 0, 0, fail_on_step
    def reset(self, state, seed):
        self.episodes += 1
        self.in_episode = 0
        self.events.append(("reset", self.episodes, state, seed))
        return dict(observation=dict(step=0), initial_observation_sha256="initial-public-observation",
                    initial_state_vector_sha256="nominal-fixed-case-state")
    def step(self, actions):
        self.steps += 1
        self.events.append(("execute", self.episodes, self.in_episode))
        if self.steps == self.fail_on_step:
            raise RuntimeError("injected RPC failure")
        self.in_episode += 1
        done = self.in_episode == 2
        return dict(observation=dict(step=self.in_episode), primitive_steps=10,
            executed_actions=np.asarray(actions), rewards=[float(done)] * 10,
            success=done, done=done)


class Policy:
    config = HistoryConfig(4104, 32, width=32, layers=1, heads=4, context_length=64)
    def __init__(self):
        self.weight = np.array([1.], np.float32)
    def state_dict(self):
        return dict(weight=self.weight)


class Motor:
    horizon = 10
    metadata = dict(receipt_sha256=PI05_RECEIPT)
    def __init__(self):
        self.weight = 7
        self.noises = []
        self.feature_encoder = lambda observation, instruction: np.full(4104, observation["step"], np.float32)
    def frozen_signature(self):
        return self.weight
    def infer(self, observation, instruction, noise):
        self.noises.append(np.asarray(noise).copy())
        return np.zeros((10, 7), np.float32)


class Controller:
    def __init__(self, spec, sim):
        self.policy, self.mode, self.sim = spec["policy"], spec["mode"], sim
        self.feedback, self.pending, self.episode_start = [], False, True
        self.acts = []
    def set_episode_seed(self, seed):
        self.seed = seed
    def act(self, feature):
        assert not self.pending
        if self.mode == "static" or self.mode == "episode" and self.episode_start:
            self.feedback = []
        self.acts.append((self.sim.episodes, len(self.feedback), self.seed))
        self.sim.events.append(("act", self.sim.episodes, len(self.feedback)))
        self.pending = True
        return np.full(32, 9., np.float32)  # Must feed back the clipped executed coordinate.
    def observe(self, *, executed_noise, reward, done):
        assert self.pending and self.sim.events[-1][0] == "execute"
        np.testing.assert_array_equal(executed_noise, np.ones(32, np.float32))
        self.sim.events.append(("observe", self.sim.episodes, reward, done))
        self.feedback.append((reward, done))
        self.pending, self.episode_start = False, done
    def snapshot(self):
        return dict(feedback=deepcopy(self.feedback), pending=self.pending, episode_start=self.episode_start)
    def restore(self, saved):
        self.feedback = deepcopy(saved["feedback"])
        self.pending, self.episode_start = saved["pending"], saved["episode_start"]


def case():
    return TargetCase("libero90", "libero90:57/state-0/seed-12", "libero_90", 57, 0, 12)


def specs():
    full, static = Policy(), Policy()
    return dict(ad_full=dict(policy=full, mode="full", sha256="full-hash"),
        ad_episode_reset=dict(policy=full, mode="episode", sha256="full-hash"),
        ad_static=dict(policy=static, mode="static", sha256="static-hash"))


def run(tmp_path, sim=None, motor=None, candidates=None, **kwargs):
    sim, motor, candidates = sim or Simulator(), motor or Motor(), candidates or specs()
    controllers = []
    def factory(spec):
        controller = Controller(spec, sim)
        controllers.append(controller)
        return controller
    result = evaluate_case(sim, motor, candidates, case=case(), output=tmp_path,
        provenance=dict(source_primitive_steps_total=11644), max_primitive_steps=10000,
        max_settling_steps=1000, methods=("ad_full",), controller_factory=factory,
        **kwargs)
    return result, sim, motor, controllers


def test_one_actual_support_plus_separate_query_and_role_costs(tmp_path):
    result, sim, motor, controllers = run(tmp_path, support_budgets=(1,))
    assert sim.episodes == 2 and len(result["episodes"]) == 2
    assert [row["role"] for row in result["episodes"]] == ["support", "query"]
    row = result["trials"][0]
    assert row["support_primitive_steps"] == row["query_primitive_steps"] == 20
    assert row["support_settling_steps"] == row["query_settling_steps"] == 3
    assert row["query_episode_id"] not in row["support_episode_ids"]
    assert controllers[0].acts[2][1] == 2  # Query sees actual completed support, including its terminal reward.
    assert result["source_provenance"]["source_primitive_steps_total"] == 11644
    assert result["weights_unchanged"] and result["budget_matched_primary_eligible"]


def test_all_shot_budgets_have_fresh_context_and_independent_query_rng(tmp_path):
    result, sim, motor, controllers = run(tmp_path, support_budgets=SUPPORT_BUDGETS)
    assert sim.episodes == sum(value + 1 for value in SUPPORT_BUDGETS) == 20
    assert len(controllers) == len(SUPPORT_BUDGETS)
    assert all(controller.acts[0][1] == 0 for controller in controllers)
    query_seeds = [row["episode_random_seed"] for row in result["episodes"] if row["role"] == "query"]
    assert len(set(query_seeds)) == 1
    assert [row["support_budget"] for row in result["trials"]] == list(SUPPORT_BUDGETS)
    for row in result["trials"]:
        assert len(row["support_episode_ids"]) == row["support_budget"]


def test_full_episode_reset_and_trained_static_context_controls(tmp_path):
    sim, motor, candidates, controllers = Simulator(), Motor(), specs(), []
    def factory(spec):
        value = Controller(spec, sim)
        controllers.append(value)
        return value
    result = evaluate_case(sim, motor, candidates, case=case(), output=tmp_path,
        provenance={}, max_primitive_steps=1000, max_settling_steps=100,
        support_budgets=(1,), methods=tuple(candidates), controller_factory=factory)
    assert controllers[0].acts[2][1] == 2  # Persistent full history.
    assert controllers[1].acts[2][1] == 0  # Same full-trained policy, reset at query episode.
    assert all(row[1] == 0 for row in controllers[2].acts)  # Separately trained static policy.
    assert len(result["trials"]) == 3


def test_current_query_reward_is_never_visible_before_execution(tmp_path):
    _, sim, _, _ = run(tmp_path, support_budgets=(0,))
    names = [event[0] for event in sim.events]
    assert names == ["reset", "act", "execute", "observe", "act", "execute", "observe"]
    assert sim.events[1][2] == 0 and sim.events[4][2] == 1


def test_native_and_tied_gaussian_are_distinct_nonlearning_baselines(tmp_path):
    sim, motor = Simulator(), Motor()
    result = evaluate_case(sim, motor, {}, case=case(), output=tmp_path, provenance={},
        max_primitive_steps=1000, max_settling_steps=100, support_budgets=(0,), methods=("native", "tied_gaussian"))
    assert not np.array_equal(motor.noises[0][0], motor.noises[0][1])
    np.testing.assert_array_equal(motor.noises[2][0], motor.noises[2][1])
    assert {row["method"] for row in result["trials"]} == {"native", "tied_gaussian"}


def test_weight_mutation_is_rejected(tmp_path):
    motor = Motor()
    original = motor.infer
    def mutate(*args):
        motor.weight += 1
        return original(*args)
    motor.infer = mutate
    with pytest.raises(RuntimeError, match="parameters changed"):
        run(tmp_path, motor=motor, support_budgets=(0,))


def test_interrupted_rpc_keeps_conservative_cost_and_episode_boundary_recovery(tmp_path):
    sim = Simulator(fail_on_step=3)  # Support completes; first query execution has unknown cost.
    with pytest.raises(RuntimeError, match="injected RPC"):
        run(tmp_path, sim=sim, support_budgets=(1,))
    costs = CostLedger(tmp_path / "physical-journal", max_primitive_steps=10000, max_settling_steps=1000).costs()
    assert costs["support"]["primitive_steps"] == 20
    assert costs["query"]["uncertain_primitive_steps_ceiling"] == 10
    import pickle
    with (tmp_path / "evaluation-recovery.pkl").open("rb") as stream:
        recovery = pickle.load(stream)
    assert recovery["next_episode"] == 1 and recovery["controller"]["pending"] is False
    assert len(recovery["controller"]["feedback"]) == 2
    sim.fail_on_step = None
    sim.metadata["pid"] = 999  # A replacement server does not change scientific identity.
    result, _, _, controllers = run(tmp_path, sim=sim, support_budgets=(1,), resume=True)
    assert controllers[0].acts[0][1] == 2
    assert len(result["episodes"]) == 2
    assert not result["budget_matched_primary_eligible"]
    assert result["interrupted_physical_costs"]["query"]["uncertain_primitive_steps_ceiling"] == 10


def test_physical_ceiling_is_checked_before_rpc(tmp_path):
    ledger = CostLedger(tmp_path, max_primitive_steps=9, max_settling_steps=3)
    with pytest.raises(RuntimeError, match="ceiling"):
        ledger.intent(kind="step", role="query", episode_id="query", maximum_steps=10)
    assert not ledger.rows


def test_no_semantic_goal_filter_and_explicit_benchmark_identity():
    metadata = Simulator().metadata
    for task in range(90):
        TargetCase("libero90", str(task), "libero_90", task, 0, 0).validate(dict(metadata, task_id=task))
    with pytest.raises(ValueError, match="benchmark"):
        TargetCase("plus", "case", "libero_90", 57, 0, 0).validate(metadata)


def test_candidate_loader_uses_assembly_costs_and_rejects_old_noise_contract(tmp_path, monkeypatch):
    import fluffy_bassoon.meta_rl.vla_source40_evaluate as module
    feature = dict(kind="prefix-v2")
    assembly = dict(assembly_sha256="assembly-hash", feature_contract=feature, small_pilot=True,
        source_task_keys=["libero_spatial:00"], collection_costs_count_once={
            "train": dict(actual_primitive_steps=5796, actual_settling_steps=610, source_updates=2220),
            "validation": dict(actual_primitive_steps=5848, actual_settling_steps=690, source_updates=2400)},
        datasets={partition: {objective: dict(sha256=f"{partition}-{objective}") for objective in ("ad", "dpt")}
                  for partition in ("train", "validation")})
    metadata = dict(motor_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT, feature_contract=feature,
        global_task_split="source40", source_task_keys=assembly["source_task_keys"], small_pilot=True)
    models, rows = {}, []
    for objective in ("ad", "dpt"):
        for mode in ("full", "static"):
            name = f"{objective}-{mode}"
            contract = dict(objective=objective, mode=mode, seed=0, train_metadata=deepcopy(metadata),
                train_sha256=f"train-{objective}", validation_sha256=f"validation-{objective}",
                config=asdict(Policy.config), batch_size=2, learning_rate=.001, max_context=32)
            models[name] = (Policy(), dict(contract=contract, update=25, selection="fixed requested final update",
                assembly_sha256="assembly-hash", full_recovery_sha256=name + "-recovery"))
            rows.append(dict(seed=0, objective=objective, mode=mode, final_path=name, final_sha256=name,
                full_recovery_sha256=name + "-recovery", updates=25, elapsed_seconds=1.))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(completed=True, contract=dict(assembly_sha256="assembly-hash"), controllers=rows)))
    motor = SimpleNamespace(metadata=dict(receipt_sha256=PI05_RECEIPT), feature_encoder=SimpleNamespace(contract=feature))
    monkeypatch.setattr(module, "load_assembly", lambda *a, **k: assembly)
    monkeypatch.setattr(module, "load_policy", lambda path, **kwargs: models[path])
    candidates, provenance = load_candidates(index, file_sha256(index), Path("assembly.json"), motor, seed=0, device="cpu", allow_small_pilot=True)
    assert len(candidates) == 6 and provenance["source_primitive_steps_total"] == 11644
    assert candidates["ad_full"]["policy"] is candidates["ad_episode_reset"]["policy"]
    models["ad-full"][1]["contract"]["train_metadata"]["noise_contract"] = "old RLinf noise"
    with pytest.raises(ValueError, match="contract differs"):
        load_candidates(index, file_sha256(index), Path("assembly.json"), motor, seed=0, device="cpu", allow_small_pilot=True)
