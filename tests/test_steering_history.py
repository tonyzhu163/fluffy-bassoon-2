"""Synthetic admission/causality tests; run only on allocated Linux, never macOS."""
from copy import deepcopy
import json
from pathlib import Path
import socket

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import steering_history as sh
from fluffy_bassoon.meta_rl.robotwin_history_corpus import parameter_digest
from fluffy_bassoon.meta_rl.vla_history import (HistoryConfig, HistoryNoisePolicy,
    load_dataset, sample_batch, trial_tokens)
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC, SACConfig
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter


def ref(path):
    return dict(path=str(path.resolve()), sha256=sh.digest(path))


def json_ref(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return ref(path)


def profile():
    return dict(format=sh.PROFILE, scope_sha256="d" * 64, source_task="place_phone_stand", wrong_task="place_object_stand",
        interface=dict(arm="kv", coefficient_dim=4, native_horizon=32, denoise_steps=5,
            motor_sha256="a" * 64, mapping_sha256="b" * 64, control_interface_sha256="c" * 64, native_observation_dim=3, observation_dim=17, coefficient_limit=1.,
            label_coordinates="actual requested post-bound 4D coefficients before hidden actuator transform",
            public_command_history="previous last-valid pre-condition 14D request; zero every physical reset",
            feature_contract=dict(public_only=True, kind="fixed motor public features"),
            mapping_contract=dict(basis_sha256="c" * 64, scale=.1, temporal="constant coefficients; fixed basis")),
        conditions=["left", "right"], max_support_episodes=8, query_k=[0, 1, 2, 4], prefix_episodes=3,
        teacher_lineage_ids=dict(left="left-train-0", right="right-train-0"),
        lineage_slots=[dict(lineage_id=f"{c}-{p}-{o if p == 'train' else 0}", condition_id=c,
            partition=p, seed=i * 3 + o) for i, c in enumerate(("left", "right"))
            for o, p in enumerate(("train", "train", "validation"))],
        teacher_gate=dict(paired_cases=8, minimum_success_rate=.5, minimum_success_gain=.25),
        episode_action_limit=600, wrong_task_action_limit=600, reset_policy_action_bound=0, locked_query_case_ids=["q0", "q1", "q2", "q3"],
        fit=dict(seed=0, updates=100, width=8, layers=1, heads=2, context_length=256,
            batch_size=4, eval_every=25, validation_batches=4, learning_rate=.0003, max_seconds_per_fit=180),
        shared_reservation=dict(owner="existing LIBERO root / finish_assembly", pilot_id="synthetic-pilot",
            pilot_gpu_seconds=28800, pilot_actions=300000, pilot_storage_bytes=64 * 1024**3,
            fit_gpu_seconds=720, evaluation_bucket="student_evaluation", evaluation_actions=84000))


def raw_observation(value):
    return dict(main_images=np.full((1, 2, 2, 3), value, np.uint8),
        wrist_images=np.full((1, 2, 2, 2, 3), value, np.uint8), states=np.zeros((1, 14), np.float32),
        task_descriptions=["Put the phone on its stand."])


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    p = profile()
    qualified = json_ref(tmp_path / "control.json", dict(interface_sha256=p["interface"]["control_interface_sha256"],
        actual_motor=True, zero_edit_parity=True, frozen_motor=True, real_episode_complete=True,
        suffix_gradient_into_kv=True))
    prefix = dict(format=sh.PREFIX, completed=True, profile_sha256=sh.content_hash(p),
        source_task=p["source_task"], interface=p["interface"], source_only=True,
        control_qualification=qualified, lineages=[], learning_qualifications={}, teacher_qualifications={})
    for condition_index, condition in enumerate(p["conditions"]):
        for ordinal, partition in enumerate(("train", "train", "validation")):
            lineage_id = f"{condition}-{partition}-{ordinal if partition == 'train' else 0}"
            root = tmp_path / lineage_id
            root.mkdir()
            learner_seed = condition_index * 3 + ordinal
            learner = NoiseSAC(3, 4, device="cpu", action_limit=1., seed=learner_seed,
                config=SACConfig(hidden_dims=(8,), batch_size=1, warmup_chunks=0, utd=1, replay_capacity=10))
            initial = root / "initial.pt"
            learner.save(initial)
            writer = RobotwinCorpusWriter(root / "corpus", {"source_commit": "d" * 40})
            episodes = []
            for episode_index in range(3):
                actor = {key: value.detach().cpu().clone() for key, value in learner.actor.state_dict().items()}
                actor_hash = parameter_digest(actor)
                actor_path = writer.root / f"acting-{episode_index}.pt"
                torch.save(dict(kind="sac_actor", parameter_sha256=actor_hash, state_dict=actor), actor_path)
                writer.set_context(dict(phase="source", method="sac", trial_id=lineage_id,
                    policy_version=f"sac:update:{learner.updates}:{actor_hash}", policy_sha256=actor_hash,
                    policy_checkpoint=str(actor_path), policy_checkpoint_sha256=sh.digest(actor_path)))
                # TRAIN/VAL reset identities are disjoint; conditions share public initial features.
                actual_seed = ordinal * 100 + episode_index
                episode_id = writer.begin_episode(raw_observation(ordinal * 3 + episode_index),
                    np.ones(3, np.float32) * ordinal,
                    dict(task=p["source_task"], requested_seed=actual_seed, actual_seed=actual_seed, reset_policy_actions=0))
                obs_hash = writer.previous_frame["content_sha256"]
                choice = learner.act(np.ones(3, np.float32) * ordinal)
                writer.begin_decision(np.full((32, 32), .8, np.float32), np.zeros((1, 14), np.float32),
                    dict(student_previous_agent_command=np.zeros(14, np.float32), steering_coefficients=choice, steering_mapping_sha256=p["interface"]["mapping_sha256"],
                        steering_interface_sha256=p["interface"]["control_interface_sha256"], steering_lineage_id=lineage_id,
                        steering_condition_id=condition, steering_partition=partition, optimizer_updates=learner.updates,
                        hidden_actuator_command=np.full(14, 999., np.float32)))
                writer.complete_native(next_raw=raw_observation(8), next_features=np.ones(3, np.float32),
                    raw_rewards=np.array([1.]), raw_terminations=np.array([True]), raw_truncations=np.array([False]),
                    valid_action_mask=np.ones((1, 1), bool), reward=1., terminated=True, truncated=False,
                    executed_steps=1, success=True, duration_seconds=.01)
                writer.finalize_learning_flags(reward=1., terminated=True, truncated=False,
                    budget_truncated=False, episode_primitive_steps=1)
                episodes.append(dict(case_id=f"scene-{ordinal}-{episode_index}", actual_seed=actual_seed,
                    initial_observation_sha256=obs_hash,
                    raw_trace=ref(writer.root / "episodes" / episode_id / "manifest.json")))
                learner.observe(np.ones(3, np.float32) * ordinal, choice, 1., np.ones(3, np.float32), True)
                learner.update()
            final = root / "final.pt"
            learner.save(final)
            prefix["lineages"].append(dict(lineage_id=lineage_id, condition_id=condition, partition=partition,
                seed=learner_seed, source_only=True, completed_episodes=3, initial_updates=0, final_updates=learner.updates,
                initial_recovery=ref(initial), final_recovery=ref(final), episodes=episodes,
                corpus_root=str(writer.root.resolve())))
        selected = next(x for x in prefix["lineages"] if x["lineage_id"] == p["teacher_lineage_ids"][condition])
        qualification = dict(condition_id=condition, interface_sha256=p["interface"]["control_interface_sha256"],
            teacher_lineage_id=selected["lineage_id"],
            teacher_kind="noise_sac", teacher_recovery=selected["final_recovery"],
            actor_parameter_sha256=parameter_digest(torch.load(selected["final_recovery"]["path"], weights_only=False)["actor"]),
            baseline_recovery=selected["initial_recovery"],
            evidence=json_ref(tmp_path / f"physical-evidence-{condition}.json", {"synthetic": True}),
            paired_cases=[dict(case_id=f"qualify-{condition}-{i}", baseline_success=False,
                               teacher_success=i < 6) for i in range(8)])
        prefix["learning_qualifications"][condition] = json_ref(tmp_path / f"learn-{condition}.json", qualification)
        prefix["teacher_qualifications"][condition] = json_ref(tmp_path / f"teacher-{condition}.json", qualification)
    return p, prefix


def test_exact_four_block_grid_charges_support_once():
    plan = sh.evaluation_plan(profile())
    assert len(plan["cards"]) == 80
    assert sum(row["episodes"] for row in plan["supports"]) == 56
    assert plan["maximum_complete_episodes"] == 136
    assert plan["maximum_actions"] == 81600
    assert len({x["id"] for x in plan["supports"]}) == len(plan["supports"])
    intact = [x for x in plan["cards"] if x["policy"] == "ad-full" and x["case_id"] == "q0" and x["lesion"] == "intact"]
    assert [x["k"] for x in intact] == [0, 1, 2, 4]
    assert len({x["reset_seed"] for x in intact}) == len({x["action_seed"] for x in intact}) == 1


@pytest.mark.parametrize("objective", ["ad", "dpt"])
def test_actual_corpus_exports_requested_coefficients_not_native_noise_or_audit_commands(source, tmp_path, objective):
    p, prefix = source
    result = sh.export_prefix(p, prefix, tmp_path / objective, objective=objective, expected_hostname=socket.gethostname())
    train = result["datasets"][objective]["train"]
    trials, metadata = load_dataset(train["path"], expected_sha256=train["sha256"])
    assert len(trials) == 4 and metadata["noise_dim"] == 4 and result["source_actions_count_once"] == 18
    assert all(np.abs(t["executed_noise"]).max() <= 1 for t in trials)
    assert all(t["observations"].shape == (3, 17) for t in trials)
    assert all(set(t) == set(trials[0]) for t in trials)
    if objective == "ad":
        assert all(np.array_equal(t["executed_noise"], t["target_noise"]) for t in trials)
    else:
        assert any(not np.array_equal(t["executed_noise"], t["target_noise"]) for t in trials)
    before = trial_tokens(trials[0])
    changed = deepcopy(trials[0]); changed["target_noise"][:] = 999
    np.testing.assert_array_equal(before, trial_tokens(changed))
    commands = sh.fit_commands(result, output=tmp_path / "fits", source_commit="e" * 40, expected_hostname=socket.gethostname())
    assert len(commands) == 2
    assert all(c[c.index("--checkpoint-selection") + 1] == "final" for c in commands)
    assert all(c[c.index("--updates") + 1] == "100" and c[c.index("--max-episodes-per-window") + 1] == "9" for c in commands)
    assert commands[0][commands[0].index("--train-sha256") + 1] == commands[1][commands[1].index("--train-sha256") + 1]


@pytest.mark.parametrize("mutation", ["case", "seed", "observation", "mapping", "partial", "query"])
def test_admission_rejects_split_leakage_wrong_mapping_and_partial_prefix(source, mutation):
    p, prefix = source
    train = prefix["lineages"][0]
    validation = next(x for x in prefix["lineages"] if x["partition"] == "validation")
    if mutation == "case": validation["episodes"][0]["case_id"] = train["episodes"][0]["case_id"]
    if mutation == "seed": validation["episodes"][0]["actual_seed"] = train["episodes"][0]["actual_seed"]
    if mutation == "observation": validation["episodes"][0]["initial_observation_sha256"] = train["episodes"][0]["initial_observation_sha256"]
    if mutation == "mapping": prefix["interface"] = dict(p["interface"], mapping_sha256="f" * 64)
    if mutation == "partial": prefix["completed"] = False
    if mutation == "query": train["episodes"][0]["case_id"] = "q0"
    with pytest.raises(ValueError): sh.admit_prefix(p, prefix, objective="ad")


def test_dpt_oracle_is_independent_of_uninformative_behavior_learning(source, tmp_path):
    p, prefix = source
    for condition in p["conditions"]:
        failed = sh.read_json(prefix["learning_qualifications"][condition])
        for row in failed["paired_cases"]: row["teacher_success"] = False
        prefix["learning_qualifications"][condition] = json_ref(tmp_path / f"failed-{condition}.json", failed)
        provenance = json_ref(tmp_path / f"oracle-proof-{condition}.json", dict(kind="synthetic same-mapping search proof",
            mapping_sha256=p["interface"]["mapping_sha256"], coefficient_dimension=4,
            physical_execution_verified=True, executed_coefficients=[.2, -.2, .1, -.1]))
        oracle = json_ref(tmp_path / f"oracle-{condition}.json", dict(condition_id=condition,
            interface_sha256=p["interface"]["control_interface_sha256"], coefficients=[.2, -.2, .1, -.1], provenance=provenance))
        qualified = sh.read_json(prefix["teacher_qualifications"][condition])
        qualified.update(teacher_kind="constant_4d_oracle", oracle=oracle)
        qualified.pop("teacher_recovery")
        prefix["teacher_qualifications"][condition] = json_ref(tmp_path / f"oracle-qualified-{condition}.json", qualified)
    with pytest.raises(ValueError, match="AD gate failed"): sh.admit_prefix(p, prefix, objective="ad")
    result = sh.export_prefix(p, prefix, tmp_path / "oracle-export", objective="dpt", expected_hostname=socket.gethostname())
    trials, _ = load_dataset(result["datasets"]["dpt"]["train"]["path"])
    np.testing.assert_array_equal(trials[0]["target_noise"], np.tile(np.array([.2, -.2, .1, -.1], np.float32), (3, 1)))


def controller(*, lesion="intact", mode="full"):
    torch.manual_seed(4)
    policy = HistoryNoisePolicy(HistoryConfig(17, 4, width=8, layers=1, heads=2, context_length=16))
    interface = profile()["interface"]
    manifest = dict(contract=dict(mode=mode, train_metadata=dict(noise_contract=interface)))
    return sh.FrozenCoefficientController(policy, manifest, interface=interface, seed=7, case_id="q0", lesion=lesion), manifest


def support():
    return [[dict(observation=np.ones(17, np.float32) * i,
        executed_coefficients=np.ones(4, np.float32) * i / 10, reward=float(i), done=True)] for i in range(4)]


def test_memory_snapshots_pair_query_rng_without_query_feedback_contaminating_support():
    c, _ = controller()
    snapshots = [c.capture_state()]
    for episode in support():
        for row in episode:
            c.act(row["observation"]); c.observe(row["executed_coefficients"], row["reward"], row["done"])
        snapshots.append(c.capture_state())
    c.restore_state(snapshots[2], query_action_seed=99)
    first = c.act(np.ones(17, np.float32)); c.observe(first, 999., True)
    c.restore_state(snapshots[2], query_action_seed=99)
    np.testing.assert_array_equal(first, c.act(np.ones(17, np.float32)))
    assert all(not x.requires_grad for x in c.controller.policy.parameters())


def test_complete_reset_clears_memory_and_all_previous_feedback():
    c, _ = controller(lesion="reset")
    c.prime_support(support())
    assert c.controller.state is None and not c.controller.previous_noise.any()
    assert c.controller.previous_reward == 0 and not c.controller.previous_done and c.controller.episode_start


def test_reward_shuffle_preserves_every_observation_action_boundary_and_action_rng():
    c, _ = controller(lesion="shuffle")
    episodes = support(); original = deepcopy(episodes)
    before_rng = c.controller.generator.get_state().clone()
    seen = []
    observe = c.observe
    def trace(action, reward, done):
        seen.append((action.copy(), reward, done)); observe(action, reward, done)
    c.observe = trace
    permutation = c.prime_support(episodes)
    assert sorted(permutation) == list(range(4)) and permutation != list(range(4))
    assert [x[1] for x in seen] == [float(i) for i in permutation]
    for i, (action, _, done) in enumerate(seen):
        np.testing.assert_array_equal(action, original[i][0]["executed_coefficients"])
        assert done
    assert torch.equal(before_rng, c.controller.generator.get_state())


def test_query_nll_is_per_k_and_target_labels_never_feed_back():
    c, manifest = controller()
    result = sh.query_nll(c.controller.policy, manifest, interface=c.interface, support=support()[:2],
        query=support()[2], query_targets=[np.zeros(4, np.float32)], seed=7, case_id="q0")
    assert result["k"] == 2 and result["query_decisions"] == 1
    assert np.isfinite(result["mean_query_nll"])


def test_profile_fails_closed_when_shared_evaluation_slice_is_too_small():
    p = profile(); p["shared_reservation"]["evaluation_actions"] = 80000
    with pytest.raises(ValueError, match="evaluation slice"): sh.validate_profile(p)


def test_episode_windows_preserve_boundaries_limit_context_and_pair_full_static_positions():
    learner = HistoryNoisePolicy(HistoryConfig(3, 4, width=8, layers=1, heads=2, context_length=32))
    done = np.tile([False, True], 12)
    choice = np.arange(96, dtype=np.float32).reshape(24, 4) / 100
    trial = dict(observations=np.repeat(np.arange(12, dtype=np.float32), 2)[:, None] * np.ones((24, 3), np.float32),
        executed_noise=choice, target_noise=choice.copy(), rewards=np.arange(24, dtype=np.float32),
        dones=done, episode_starts=np.r_[True, done[:-1]], teacher_steps=np.repeat(np.arange(12, dtype=np.int64), 2))
    full = sample_batch([trial], learner, np.random.default_rng(72), batch_size=16, max_episodes=9)
    static = sample_batch([trial], learner, np.random.default_rng(72), batch_size=16, max_episodes=9, mode="static")
    torch.testing.assert_close(full[0][:, :, :3], static[0][:, :, :3], rtol=0, atol=0)
    for i in (1, 2, 3): torch.testing.assert_close(full[i], static[i], rtol=0, atol=0)
    lengths = full[2].sum(0)
    assert (lengths <= 18).all() and (lengths % 2 == 0).all()
    assert full[3][0].all() and not full[0][0, :, 3:-1].any()


def test_wrapped_collector_recovery_binds_mapping_and_complete_update_boundary(source, tmp_path):
    p, prefix = source
    lineage = prefix["lineages"][0]
    state = torch.load(lineage["final_recovery"]["path"], weights_only=False)
    wrapper = dict(format="source_coefficient_teacher_v1", learner=state, active_episode=None,
        pending=None, pending_update_cycles=0, updating=False,
        lineage=dict(method=lineage["lineage_id"], learner_seed=lineage["seed"], partition=lineage["partition"]),
        profile=dict(task=p["source_task"], mapping_sha256=p["interface"]["mapping_sha256"],
            interface_sha256=p["interface"]["control_interface_sha256"], motor_sha256=p["interface"]["motor_sha256"],
            scope_sha256=p["scope_sha256"]))
    path = tmp_path / "wrapped.pt"; torch.save(wrapper, path)
    loaded = sh._source_state(ref(path), lineage=lineage, profile=p)
    assert loaded["updates"] == state["updates"]
    wrapper["pending_update_cycles"] = 1; torch.save(wrapper, path)
    with pytest.raises(ValueError, match="completed update boundary"):
        sh._source_state(ref(path), lineage=lineage, profile=p)


def test_last_valid_public_request_excludes_hidden_commands_and_unexecuted_tail():
    class Reader:
        def arrays(self, ref, names):
            return {"valid_action_mask": np.array([[1, 0, 1, 0]], bool)}
    requested = np.arange(56, dtype=np.float32).reshape(1, 4, 14)
    decision = {"issued_motor_joint_targets": np.full((4, 14), 999., np.float32)}
    chunk = dict(native_result=dict(arrays={}, executed_primitive_steps=2))
    metadata = dict(frozen_pi05_residual=dict(public_agent_requested_actions=requested,
        audit_only_post_command_transform={"hidden": True}))
    actual = sh.public_requested_command(Reader(), chunk, metadata=metadata, decision=decision)
    np.testing.assert_array_equal(actual, requested[0, 2])
    metadata["frozen_pi05_residual"].pop("public_agent_requested_actions")
    with pytest.raises(ValueError, match="Hidden actuator"):
        sh.public_requested_command(Reader(), chunk, metadata=metadata, decision=decision)


def test_live_adapter_delegates_2076_projection_to_the_single_driver_owner():
    history, _ = controller()
    class Control:
        calls = 0
        def student_observation(self, native):
            self.calls += 1
            return np.r_[native, np.full(14, .25, np.float32)]
    control = Control()
    adapter = sh.DriverCoefficientStudent(history, control=control, condition_id="audit-only")
    native = np.ones(3, np.float32)
    choice = adapter.act(native)
    assert control.calls == 1 and choice.shape == (4,)
    np.testing.assert_array_equal(adapter.last_augmented_observation, np.r_[native, np.full(14, .25, np.float32)])
    assert adapter.decision_metadata()["steering_partition"] == "evaluation"
    adapter.observe(native, choice, 0., native, True, False, 1, None)
    assert history.controller.episode_start


@pytest.mark.parametrize("complete", [False, True])
def test_final_only_selection_never_promotes_a_partial_and_retains_complete_resume(source, tmp_path, monkeypatch, complete):
    from fluffy_bassoon.meta_rl import vla_history_train as trainer
    p, prefix = source
    p["fit"].update(updates=2, eval_every=1, max_seconds_per_fit=180 if complete else 1e-12)
    prefix["profile_sha256"] = sh.content_hash(p)
    assembly = sh.export_prefix(p, prefix, tmp_path / "data", objective="ad", expected_hostname=socket.gethostname())
    commands = sh.fit_commands(assembly, output=tmp_path / "fits", source_commit="e" * 40,
                               expected_hostname=socket.gethostname())
    # CPU test of the existing learner; no model execution is permitted on macOS.
    command = commands[0]
    command[command.index("--device") + 1] = "cpu"
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda args, **kw: "e" * 40 if "rev-parse" in args else "")
    trainer.main(command[3:])
    output = tmp_path / "fits/ad-full"
    saved = torch.load(output / "latest.pt", weights_only=False)
    assert {"policy", "optimizer", "rng", "contract", "update"} <= saved.keys()
    assert saved["contract"]["schedule"]["checkpoint_selection"] == "final"
    assert (output / "candidate.pt").exists() == complete
    if complete:
        policy, manifest = sh.load_student(ref(output / "candidate.pt"), assembly, mode="full", source_commit="e" * 40)
        assert manifest["update"] == 2 and all(not x.requires_grad for x in policy.parameters())
