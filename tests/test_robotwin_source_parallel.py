"""Allocated Linux CPU qualification of prospective source concurrency.

All episodes/qualification tickets here are synthetic; they admit no data,
frontend, teacher or GPU job. Actual runtime qualification remains separate.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import robotwin_source_parallel as parallel
from fluffy_bassoon.meta_rl import robotwin_steering_source as serial
from fluffy_bassoon.meta_rl.steering_condition_collection import lineage_specs, SOURCE_METHOD_KEYS
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_source_broker import SourceBudgetBroker, SourceBudgetClient
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution
from test_robotwin_steering_source import tiny_factory, a1_fixture
from test_steering_condition_collection import profile
from test_steering_actuator_condition import make_pair

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Allocated Linux CPU qualification only")


def _wave_fixture(connection, args, allocation, profile_data, interface, qualification, lineage_data, header, source):
    method = lineage_data["method"]
    client = SourceBudgetClient(connection, worker=method, header=header)
    spec = dict(id=f"{method}:test-only", task=profile_data["task"], method=method, horizon=400, phase="source")
    client.call("admit", specs=[spec], phase="source")
    allocation["test_only_barrier"].wait(timeout=10)
    identity = f"{args.run_id}:{method}:{spec['id']}:0"
    grant = client.reserve(identity, 1, phase="source", method=method,
        task=spec["task"], episode=spec["id"], chunk=0, policy_version="test-only")
    assert grant.execute
    client.resolve(identity, 1)
    root = Path(args.output)/"synthetic-shards"/method
    root.mkdir(parents=True)
    serial._atomic_json(root/"state.json", dict(lineage=lineage_data, pid=__import__("os").getpid()))
    client.call("complete", episode=spec["id"], receipt=dict(complete=True, chunks=1, primitive_steps=1,
        episode_manifest_ref=dict(path=str(root/"state.json"), sha256=serial.sha(root/"state.json"))))
    connection.send(dict(operation="finished", result=dict(status="synthetic_fixture_complete", lineage=lineage_data)))
    connection.close()


def test_all_six_process_lineages_execute_in_qualified_two_worker_waves(monkeypatch, tmp_path):
    p = profile()
    args = SimpleNamespace(output=tmp_path/"job", run_id="fixture")
    args.output.mkdir()
    allocation = dict(max_concurrency=2, worker_method_limits=parallel.method_limits(p),
        test_only_barrier=mp.get_context("spawn").Barrier(2))
    monkeypatch.setattr(parallel, "_worker", _wave_fixture)
    with CampaignBudget(tmp_path/"ledger", campaign_id="synthetic", total_actions=300000,
            source_caps=p.source_caps(), phase_caps=serial.PHASE_CAPS,
            contract=dict(scope_sha256=p.scope_sha256), run_id="fixture", source_commit="a"*40) as budget:
        result = parallel.supervise(args, allocation, p, {}, {}, budget, "a"*40)
        assert set(result["results"]) == set(SOURCE_METHOD_KEYS)
        assert not result["not_started"] and budget.snapshot()["actual_actions"] == 6
        assert not budget.snapshot()["unresolved_decisions"]
    rows = [json.loads((args.output/"synthetic-shards"/m/"state.json").read_text()) for m in SOURCE_METHOD_KEYS]
    assert len({r["pid"] for r in rows}) == 6
    assert [r["lineage"] for r in rows] == [asdict(l) for l in lineage_specs(p.learner_seed)]


def test_worker_reuses_exact_episode_boundary_learning_and_fixed_validation(monkeypatch, tmp_path):
    tiny_factory(monkeypatch)
    p, lineage = profile(), lineage_specs()[0]
    output = tmp_path/"worker"
    output.mkdir()
    _, pair = make_pair()
    distribution = CalibrationDistribution(pair)
    control = SimpleNamespace(active=False, post_command_transform=distribution,
        condition_contract=distribution.contract(), interface_sha256=p.interface_sha256,
        episode_limit_for_task=lambda _: 400, set_coefficients=lambda *_: None,
        mapping_sha256=lambda: p.mapping_sha256, snapshot=lambda: dict(synthetic=True))
    class Guard:
        allocation = dict(max_full_recovery_bytes=1 << 30)
        maximum_episode_seconds = .01
        def __call__(self):
            pass
        def admit_episode(self, *, episodes=1):
            assert episodes in (1, 2)
    modes, history = [], []
    def runner(current, spec, ledger, *, controller, run_id, phase, **kwargs):
        obs = np.zeros(2062, np.float32)
        before = controller.learner.updates
        refs = []
        for chunk in range(2):
            action = controller.act(obs)
            acting = controller.decision_metadata()
            assert controller.learner.updates == before  # No optimization inside episode.
            identity = f"{run_id}:{spec['id']}:{chunk}"
            grant = ledger.reserve(identity, 1, phase=phase, method=spec["method"], task=spec["task"],
                episode=spec["id"], chunk=chunk, policy_version=acting["policy_version"])
            assert grant.execute
            ledger.resolve(identity, 1)
            path = output/(spec["id"]+f":chunk:{chunk}.json")
            serial._atomic_json(path, dict(synthetic=True, campaign_decision_id=identity))
            ref = dict(root=str(output), path=path.name, sha256=serial.sha(path))
            refs.append(ref)
            controller.observe(obs, action, float(chunk), obs, chunk == 1, False, 1, ref)
        history.append((phase, spec["id"], before))
        if phase == "source_validation":
            modes.append((spec["milestone"], controller.decision_metadata()["source_warmup"]))
            if spec["milestone"]:
                assert spec["paired_instruction"] == ["Actual fixed fixture instruction"]
        return dict(spec=deepcopy(spec), complete=True, success=True, primitive_steps=2, chunks=2,
            wall_seconds=.001, corpus_refs=refs, episode_manifest_ref=refs[-1],
            reset_info=dict(actual_seed=spec["reset_seed"], requested_seed=spec["reset_seed"],
                observation_sha256="a"*64, features_sha256="b"*64, reset_policy_actions=0,
                instruction=["Actual fixed fixture instruction"]),
            reset_identity=dict(actual_seed=spec["reset_seed"], observation_sha256="a"*64, features_sha256="b"*64))
    monkeypatch.setattr(serial, "run_complete_episode", runner)
    with CampaignBudget(tmp_path/"ledger", campaign_id="synthetic", total_actions=300000,
            source_caps=p.source_caps(), phase_caps=serial.PHASE_CAPS,
            contract=dict(scope_sha256=p.scope_sha256), run_id="fixture", source_commit="a"*40) as budget:
        owner = SourceBudgetBroker(budget, tmp_path/"ownership", method_limits=parallel.method_limits(p),
            episode_bound=400, is_live=lambda method, pid: method == lineage.method and pid == 10)
        owner.register(lineage.method, 10)
        class DirectClient:
            header = budget.header
            def call(self, operation, **arguments):
                return owner.dispatch(lineage.method, operation, arguments)
            def remaining(self, **arguments):
                return self.call("remaining", **arguments)
            def reserve(self, identity, requested, **arguments):
                return self.call("reserve", decision_id=identity, requested_actions=requested, **arguments)
            def resolve(self, identity, actual):
                return self.call("resolve", decision_id=identity, actual_actions=actual)
            def snapshot(self, **arguments):
                return self.call("snapshot", **arguments)
        state = parallel.run_lineage(control=control, budget=DirectClient(), profile=p, lineage=lineage,
            distribution=distribution, output=output, guard=Guard(),
            archive=serial.existing_actor_archive(output/"actors"), run_id=f"fixture:{lineage.method}", device="cpu")
        assert state["status"] == "initial_prefix_complete" and state["complete_episodes"] == 24
        assert state["actual_optimizer_updates"] == 64  # (48-32) chunks strictly after warmup x UTD4.
        assert state["actual_primitive_steps"] == 48
        assert set(state["publications"]) == {"12", "24"}
        assert len(modes) == 6 and modes[:4] == [(0, True)]*2+[(12, True)]*2
        assert modes[-2:] == [(24, False)]*2
        final = serial.BoundedTeacherRecovery.load(state["full_recovery"])
        learner = final["collector"]["learner"]
        assert final["collector"]["pending"] is None and final["collector"]["pending_update_cycles"] == 0
        assert final["collector"]["validation_done"] == [0, 12, 24]
        assert learner["updates"] == 64
        assert all(key in final for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng", "control"))
        assert not budget.snapshot()["unresolved_decisions"]


def test_current_ineligible_frontend_cannot_be_admitted_by_concurrency():
    p, selection, qualification, a1, rule = a1_fixture()
    for record in a1["records"].values():
        if record["spec"]["stage"] == "confirmation":
            record["success"] = False
    a1["restricted_summary"] = serial.restricted_summary(a1["plan"], a1["records"])
    with pytest.raises(ValueError):
        serial.validate_selection(selection, qualification, a1, p, dict(motor_sha256=p.motor_sha256),
            interface_sha=p.interface_sha256, selection_rule=rule)


def test_actual_capacity_receipt_cannot_be_replaced_by_cpu_fixtures():
    p = profile()
    with pytest.raises(ValueError, match="Actual exact-code"):
        parallel.validate_capacity(dict(status="passed", synthetic=True),
            dict(max_concurrency=2, worker_cpu_threads=1, allocated_cpu_count=2), p, None,
            source="a"*40, hostname="synthetic", gpu_uuid="GPU-synthetic", frozen_fingerprint="b"*64)


def test_only_original_six_lineage_method_phase_quotas():
    p = profile()
    assert set(parallel.method_limits(p)) == set(SOURCE_METHOD_KEYS)
    assert set(tuple(sorted(v.items())) for v in parallel.method_limits(p).values()) == {
        (("source", 9600), ("source_validation", 2400))}
    assert all(v == dict(source=19200, source_validation=4000) for v in parallel.method_limits(p, 48).values())
    with pytest.raises(ValueError):
        parallel.method_limits(p, 36)


def test_new_parent_job_uses_under8h_authorization_without_rewriting_historical_costs(tmp_path):
    p, phase_id = profile(), "original-scientific-phase"
    phase = tmp_path/"phase"
    phase.mkdir()
    with CampaignBudget(phase/"ledger", campaign_id=phase_id, total_actions=300000,
            source_caps=p.source_caps(), phase_caps=serial.PHASE_CAPS,
            contract=dict(scope_sha256=p.scope_sha256, phase_id=phase_id, motor_sha256=p.motor_sha256),
            run_id="synthetic", source_commit="a"*40) as budget:
        ledger = dict(budget.header, root=str(budget.root))
    scope = dict(schema="frozen_pi05_endtoend_scope_v1", phase_id=phase_id,
        phase_limits=dict(cumulative_gpu_seconds=28800, all_in_simulator_actions=300000,
                          outputs_gib=64, fresh_cpu_qualification_worker_seconds=1800))
    allocation = dict(schema=parallel.ALLOCATION, max_concurrency=2, worker_cpu_threads=1, allocated_cpu_count=2,
        scope_sha256=p.scope_sha256, profile_sha256="f"*64, source_commit="a"*40, hostname="synthetic",
        gpu_count=1, job_id="synthetic", episode_target=24,
        worker_devices=dict.fromkeys(SOURCE_METHOD_KEYS, "cuda:0"), worker_method_limits=parallel.method_limits(p),
        output=str(phase/"job"), phase_output_root=str(phase), source_corpus_root=str(phase/"source-index"),
        authorization_basis="human_each_job_under_8h", reserved_gpu_seconds=1000,
        prior_phase_gpu_seconds=100000, other_reserved_phase_gpu_seconds=100000,
        driver_deadline_unix=2000, allocation_started_unix=1000, maximum_episode_seconds=30,
        maximum_episode_update_seconds=10, cleanup_reserve_seconds=60, action_ledger=ledger,
        max_full_recovery_bytes=1000000, max_chunk_output_bytes=100000,
        recovery_output_reservation_bytes=62000000, other_future_output_reservation_bytes=0)
    kwargs = dict(source="a"*40, hostname="synthetic", output=phase/"job", profile_sha="f"*64)
    parallel.validate_parallel_admission(scope, allocation, p, **kwargs)
    assert allocation["prior_phase_gpu_seconds"] == 100000  # No forecast/historical refund.
    for field, value in (("max_concurrency", 1), ("reserved_gpu_seconds", 28800),
                         ("worker_devices", dict.fromkeys(SOURCE_METHOD_KEYS, "cuda:1")),
                         ("authorization_basis", "fresh_worker_allowances")):
        with pytest.raises(ValueError):
            parallel.validate_parallel_admission(scope, dict(allocation, **{field: value}), p, **kwargs)
