"""Lossless corpus and interrupted-intent checks; run on allocated Linux."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter, _atomic_json
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv, _capture_sparse_feedback


def public_observation(value):
    return dict(main_images=np.full((1, 4, 5, 3), value, np.uint8),
                wrist_images=np.full((1, 2, 4, 5, 3), value + 1, np.uint8),
                states=np.full((1, 14), value / 10, np.float32),
                task_descriptions=["Put the phone on its stand."])


def writer_at(path):
    writer = RobotwinCorpusWriter(path, {"source_commit": "a" * 40,
                                       "motor_sha256": "b" * 64})
    writer.set_context(dict(phase="source", method="ppo", trial_id="trial-0001",
                            policy_version=0, checkpoint_sha256="c" * 64))
    return writer


def start_episode(writer, value=0):
    return writer.begin_episode(public_observation(value), np.full(3, value, np.float32),
                                dict(task="place_phone_stand", requested_seed=7, actual_seed=8,
                                     reset_policy_actions=0,
                                     reset_internal_physics_steps="not separately instrumented"))


def finish_native(writer, *, value=2, executed=3, terminal=False):
    writer.complete_native(next_raw=public_observation(value),
        next_features=np.full(3, value, np.float32),
        raw_rewards=np.array([float(terminal)]),
        raw_terminations=np.array([terminal]), raw_truncations=np.array([False]),
        valid_action_mask=np.array([[1] * executed + [0] * (4 - executed)], np.bool_),
        reward=float(terminal), terminated=terminal, truncated=False,
        executed_steps=executed, success=terminal, duration_seconds=.02,
        adapter_rewards=np.array([0., float(terminal)]),
        native_info={"executed_action_count": executed})


def finalize(writer, *, steps, terminated=False, truncated=False, budget=False):
    return writer.finalize_learning_flags(reward=float(terminated), terminated=terminated,
        truncated=truncated, budget_truncated=budget, episode_primitive_steps=steps)


def test_complete_public_transition_and_ppo_metadata_are_lossless_and_linked(tmp_path):
    writer = writer_at(tmp_path)
    episode = start_episode(writer)
    raw_action = np.arange(8, dtype=np.float32).reshape(4, 2)
    actions = np.arange(56, dtype=np.float64).reshape(4, 14)
    path = writer.begin_decision(raw_action, actions,
        dict(pre_squash_actions=raw_action + .25, old_log_prob=-1.5, old_value=.2,
             policy_input=np.arange(10, dtype=np.float32), recurrent_reset_mask=True,
             history_index=0, nested={"boundary": 1, "previous_reward": 0.}))
    intent = json.loads((tmp_path / path).read_text())
    assert intent["status"] == "intent" and "native_result" not in intent
    assert intent["observation"]["language"] == ["Put the phone on its stand."]
    with np.load(tmp_path / intent["decision_arrays"]["path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sampled_noise"], raw_action)
        np.testing.assert_array_equal(data["issued_motor_joint_targets"], actions)
        np.testing.assert_array_equal(data["metadata/pre_squash_actions"], raw_action + .25)
        assert all(not data[name].dtype.hasobject for name in data.files)
    finish_native(writer)
    finalize(writer, steps=3, truncated=True, budget=True)
    first = json.loads((tmp_path / path).read_text())
    assert first["status"] == "complete"
    assert not first["native_result"]["truncated"]
    assert first["learning_result"]["truncated"] and first["learning_result"]["budget_truncated"]
    assert first["native_result"]["executed_primitive_steps"] == 3
    manifest = json.loads((tmp_path / "episodes" / episode / "manifest.json").read_text())
    assert manifest["completed_chunks"] == 1 and manifest["executed_primitive_steps"] == 3
    assert manifest["chunks"][0]["path"] == "chunks/000001.json"
    assert manifest["pending_decision"] is None
    writer.set_context(dict(policy_version=1, checkpoint_sha256="d" * 64))
    second_path = writer.begin_decision(raw_action, actions, {"history_index": 1})
    second = json.loads((tmp_path / second_path).read_text())
    assert second["episode_id"] == episode
    assert second["previous_chunk_index"] == 1
    assert second["context"]["policy_version"] == 1
    assert second["observation"] == first["next_observation"]
    assert writer.statistics()["frame_count"] == 2
    with np.load(tmp_path / second["observation"]["path"], allow_pickle=False) as frame:
        np.testing.assert_array_equal(frame["states"], public_observation(2)["states"])
        np.testing.assert_array_equal(frame["main_images"], public_observation(2)["main_images"])


def test_crash_recovery_reports_intent_without_inventing_a_transition(tmp_path):
    writer = writer_at(tmp_path)
    first_episode = start_episode(writer)
    path = writer.begin_decision(np.zeros((4, 2)), np.zeros((4, 14)), {})
    # No simulator outcome exists. Reopening must not count issued actions or
    # fabricate next observations as if the action had completed.
    resumed = writer_at(tmp_path)
    stats = resumed.statistics()
    assert stats["completed_chunks"] == stats["executed_primitive_steps"] == 0
    assert stats["unresolved_decisions"] == [path]
    assert json.loads((tmp_path / path).read_text())["status"] == "intent"
    assert start_episode(resumed) != first_episode
    with pytest.raises(ValueError, match="contract changed"):
        RobotwinCorpusWriter(tmp_path, {"source_commit": "other"})


def test_native_result_is_not_complete_until_outer_boundary_is_recorded(tmp_path):
    writer = writer_at(tmp_path)
    start_episode(writer)
    path = writer.begin_decision(np.zeros((4, 2)), np.zeros((4, 14)), {})
    finish_native(writer, executed=0, terminal=True)
    reopened = writer_at(tmp_path)
    assert reopened.statistics()["native_completed_chunks"] == 1
    assert reopened.statistics()["unresolved_decisions"] == [path]
    finalize(writer, steps=0, terminated=True)
    final = json.loads((tmp_path / path).read_text())
    assert final["learning_result"]["terminated"] and not final["learning_result"]["truncated"]
    assert writer_at(tmp_path).statistics()["completed_chunks"] == 1


def test_exact_mask_mismatch_cannot_create_a_completed_record(tmp_path):
    writer = writer_at(tmp_path)
    start_episode(writer)
    path = writer.begin_decision(np.zeros((4, 2)), np.zeros((4, 14)), {})
    with pytest.raises(ValueError, match="valid-action mask"):
        writer.complete_native(next_raw=public_observation(1), next_features=np.zeros(3),
            raw_rewards=np.zeros(1), raw_terminations=np.zeros(1), raw_truncations=np.zeros(1),
            valid_action_mask=np.ones((1, 4)), reward=0., terminated=False, truncated=False,
            executed_steps=2, success=False, duration_seconds=.1)
    assert json.loads((tmp_path / path).read_text())["status"] == "intent"
    assert writer.statistics()["completed_chunks"] == 0


def test_orphan_frame_is_verified_and_recovered_without_overwrite(tmp_path):
    writer = writer_at(tmp_path)
    start_episode(writer)
    ref = writer.previous_frame
    blob = tmp_path / ref["path"]
    original = blob.read_bytes()
    blob.with_suffix(".json").unlink()  # Simulate crash between blob and sidecar.
    resumed = writer_at(tmp_path)
    start_episode(resumed)
    assert blob.read_bytes() == original
    blob.with_suffix(".json").unlink()
    with blob.open("wb") as stream:
        np.savez_compressed(stream, states=np.zeros((1, 14)))
    with pytest.raises(ValueError, match="Orphan corpus frame"):
        start_episode(writer_at(tmp_path))


def test_context_and_numeric_validation_fail_before_simulator_intent(tmp_path):
    writer = RobotwinCorpusWriter(tmp_path, {})
    with pytest.raises(ValueError, match="Missing corpus context"):
        start_episode(writer)
    writer.set_context(dict(phase="source", method="sac", trial_id="one", policy_version=0))
    start_episode(writer)
    with pytest.raises(ValueError, match="finite"):
        writer.begin_decision(np.full((4, 2), np.nan), np.zeros((4, 14)), {})
    assert not list((tmp_path / "episodes").glob("*/chunks/*.json"))
    with pytest.raises(ValueError, match="non-object"):
        writer.begin_decision(np.zeros((4, 2)), np.zeros((4, 14)), {"bad": np.array([object()])})
    writer.register_runtime({"native_horizon": 32})
    with pytest.raises(ValueError, match="runtime contract changed"):
        writer.register_runtime({"native_horizon": 50})


def test_native_feedback_tap_preserves_return_identity_and_restores_on_error():
    result = (np.array([1.]), np.array([True]), np.array([False]), {"executed_action_count": 3}, [])
    task = SimpleNamespace(gen_sparse_reward_data=lambda *args, **kwargs: result)
    original = task.gen_sparse_reward_data
    with _capture_sparse_feedback(task) as feedback:
        assert task.gen_sparse_reward_data(np.zeros((4, 14)), collect_obs=True) is result
    assert task.gen_sparse_reward_data is original
    np.testing.assert_array_equal(feedback["raw_rewards"], result[0])
    result[0][:] = 0
    assert feedback["raw_rewards"].tolist() == [1.]
    with pytest.raises(RuntimeError, match="native failed"):
        with _capture_sparse_feedback(task):
            raise RuntimeError("native failed")
    assert task.gen_sparse_reward_data is original


class CorpusBackend:
    observation_dim = 3
    noise_shape = (4, 2)

    def __init__(self, writer):
        self.writer = writer
        self._decision_metadata = {}

    def reset(self, task, seed):
        info = dict(task=task, requested_seed=seed, actual_seed=seed)
        self.writer.begin_episode(public_observation(0), np.zeros(3), info)
        return np.zeros(3), info

    def step(self, noise, action_count):
        self.writer.begin_decision(noise, np.zeros((action_count, 14)), self._decision_metadata)
        self._decision_metadata = {}
        self.writer.complete_native(next_raw=public_observation(1), next_features=np.ones(3),
            raw_rewards=np.array([0.]), raw_terminations=np.array([False]),
            raw_truncations=np.array([False]), valid_action_mask=np.ones((1, action_count)),
            reward=0., terminated=False, truncated=False, executed_steps=action_count,
            success=False, duration_seconds=.01)
        return np.ones(3), 0., False, False, dict(primitive_steps=action_count)

    def close(self):
        pass


def test_environment_records_final_budget_truncation_without_changing_native_flags(tmp_path):
    writer = writer_at(tmp_path)
    env = RobotwinNoiseEnv(backend=CorpusBackend(writer), corpus_writer=writer)
    env.reset("place_phone_stand", 5)
    env.record_decision_metadata({"old_value": .25, "history_index": 0})
    _, _, end, trunc, info = env.step(np.zeros(8), max_primitive_steps=2)
    assert not end and trunc and info["budget_truncated"]
    record = json.loads((tmp_path / info["corpus_chunk"]).read_text())
    assert record["native_result"]["executed_primitive_steps"] == 2
    assert not record["native_result"]["truncated"]
    assert record["learning_result"]["truncated"]
    assert record["decision_metadata"]["old_value"] == .25


def test_lane_handles_interleave_without_crossed_context_intents_frames_or_chronology(tmp_path):
    root = writer_at(tmp_path)
    zero,one = root.for_lane(0),root.for_lane(1)
    assert zero is root.for_lane(0) is one.for_lane(0)
    assert zero is not one and zero.root == one.root == root.root
    zero.set_context(dict(trial_id='lane-zero',policy_version=7))
    one.set_context(dict(trial_id='lane-one',policy_version=9))
    episodes = [start_episode(zero),start_episode(one)]
    assert episodes[0] != episodes[1] and root.episode_id is None
    paths = [zero.begin_decision(np.zeros((4,2)),np.zeros((4,14)),dict(old_log_prob=-.7)),
             one.begin_decision(np.ones((4,2)),np.ones((4,14)),dict(old_log_prob=-.9))]
    assert paths[0] != paths[1] and zero.pending is not one.pending and root.pending is None
    stats = root.statistics()
    assert stats['frame_count']==1 and stats['completed_chunks']==0
    assert set(stats['unresolved_decisions'])==set(paths)
    for lane,path in enumerate(paths):
        record=json.loads((tmp_path/path).read_text())
        assert record['context']['trial_id']==('lane-zero'if lane==0 else'lane-one')
        assert record['context']['policy_version']==(7 if lane==0 else 9)
    finish_native(one,executed=2,terminal=True)
    finish_native(zero,executed=4)
    assert root.statistics()['native_completed_chunks']==2
    assert zero.statistics()['active_episode_id']==episodes[0]
    assert one.statistics()['active_episode_id']==episodes[1]
    finalize(one,steps=2,terminated=True)
    finalize(zero,steps=4,truncated=True,budget=True)
    stats=root.statistics()
    assert stats['completed_chunks']==2 and stats['native_completed_chunks']==0
    assert stats['executed_primitive_steps']==6 and stats['unresolved_decisions']==[]
    assert stats['frame_count']==2 and stats['episode_count']==2
    assert len(list((tmp_path/'frames').glob('*.npz')))==2
    persisted=json.loads((tmp_path/'statistics.json').read_text())
    assert persisted['completed_chunks']==2 and persisted['executed_primitive_steps']==6
    next_episodes=[start_episode(zero,value=2),start_episode(one,value=2)]
    for current,previous in zip(next_episodes,episodes):
        reset=json.loads((tmp_path/'episodes'/current/'reset.json').read_text())
        assert reset['previous_episode_id']==previous
    # Serial evaluation on the root never inherits a lane's episode chain.
    serial=start_episode(root,value=2)
    assert json.loads((tmp_path/'episodes'/serial/'reset.json').read_text())['previous_episode_id'] is None
    assert root.context['trial_id']=='trial-0001'


def test_reopening_lane_corpus_reports_all_unresolved_without_adopting_or_replaying(tmp_path):
    root=writer_at(tmp_path);zero,one=root.for_lane(0),root.for_lane(1)
    start_episode(zero);start_episode(one)
    missing=zero.begin_decision(np.zeros((4,2)),np.zeros((4,14)),{})
    partial=one.begin_decision(np.ones((4,2)),np.ones((4,14)),{})
    finish_native(one,executed=2)
    reopened=writer_at(tmp_path)
    inspection=(tmp_path/'recovery-inspection.json').read_bytes()
    child=reopened.for_lane(0)
    assert child.pending is None and child.episode_id is None
    with pytest.raises(RuntimeError,match='Native result'):
        child.finalize_learning_flags(reward=0.,terminated=False,truncated=True,
            budget_truncated=False,episode_primitive_steps=2)
    with pytest.raises(RuntimeError,match='No recorded simulator intent'):
        finish_native(child)
    assert (tmp_path/'recovery-inspection.json').read_bytes()==inspection
    stats=reopened.statistics()
    assert set(stats['unresolved_decisions'])=={missing,partial}
    assert stats['completed_chunks']==0 and stats['native_completed_chunks']==1
    assert json.loads((tmp_path/missing).read_text())['status']=='intent'
    assert json.loads((tmp_path/partial).read_text())['status']=='native_complete'
    old_episodes={json.loads((tmp_path/path).read_text())['episode_id']for path in(missing,partial)}
    assert start_episode(child)not in old_episodes


def test_explicit_root_statistics_reconcile_independent_on_disk_records(tmp_path):
    root=writer_at(tmp_path);child=root.for_lane(0)
    other=writer_at(tmp_path)
    start_episode(other)
    other.begin_decision(np.zeros((4,2)),np.zeros((4,14)),{})
    finish_native(other,executed=3);finalize(other,steps=3,truncated=True)
    assert root.statistics()['completed_chunks']==1
    assert child.statistics()['executed_primitive_steps']==3
    assert child._stats is root._stats
    root.register_runtime({'native_horizon':32})
    child.register_runtime({'native_horizon':32})
    with pytest.raises(ValueError,match='runtime contract changed'):
        child.register_runtime({'native_horizon':50})
    with pytest.raises(ValueError,match='nonnegative integer'):
        root.for_lane(True)
    with pytest.raises(ValueError,match='nonnegative integer'):
        root.for_lane(-1)


def test_payload_fsync_precedes_publication_and_directory_fsync_follows_it(tmp_path,monkeypatch):
    import os
    import stat
    writer=writer_at(tmp_path/'corpus')
    events=[]
    real_fsync,real_link,real_replace=os.fsync,os.link,os.replace
    def fsync(descriptor):
        is_directory=stat.S_ISDIR(os.fstat(descriptor).st_mode)
        path=Path(os.readlink(f'/proc/self/fd/{descriptor}'))
        events.append(('directory'if is_directory else'payload',path))
        return real_fsync(descriptor)
    def link(source,destination):
        events.append(('publish_link',Path(destination)))
        return real_link(source,destination)
    def replace(source,destination):
        events.append(('publish_replace',Path(destination)))
        return real_replace(source,destination)
    monkeypatch.setattr(os,'fsync',fsync)
    monkeypatch.setattr(os,'link',link)
    monkeypatch.setattr(os,'replace',replace)
    def checked(call,path,publication):
        events.clear();call()
        published=events.index((publication,path))
        payload=[index for index,(kind,_)in enumerate(events)if kind=='payload']
        directories=[index for index,item in enumerate(events)if item==('directory',path.parent)]
        assert payload and max(payload)<published
        assert directories and max(directories)>published
    path=tmp_path/'intent.json'
    checked(lambda:_atomic_json(path,{'version':1}),path,'publish_link')
    checked(lambda:_atomic_json(path,{'version':2},replace=True),path,'publish_replace')
    blob=writer.root/'decision.npz'
    checked(lambda:writer._save_arrays(Path('decision.npz'),{'values':np.arange(4)}),blob,'publish_link')
    with np.load(blob,allow_pickle=False)as saved:
        np.testing.assert_array_equal(saved['values'],np.arange(4))


def test_new_intent_directory_hierarchy_is_synced_and_directory_errors_propagate(tmp_path,monkeypatch):
    import os
    import stat
    real_fsync=os.fsync
    directories=[]
    def observed_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directories.append(Path(os.readlink(f'/proc/self/fd/{descriptor}')))
        return real_fsync(descriptor)
    monkeypatch.setattr(os,'fsync',observed_fsync)
    path=tmp_path/'episode'/'chunks'/'intent.json'
    _atomic_json(path,{'intent':True})
    assert tmp_path in directories and path.parent.parent in directories and path.parent in directories
    writer=writer_at(tmp_path/'corpus')
    def failed_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError('directory durability failed')
        return real_fsync(descriptor)
    monkeypatch.setattr(os,'fsync',failed_directory_sync)
    with pytest.raises(OSError,match='directory durability failed'):
        _atomic_json(tmp_path/'failed-intent.json',{'intent':True})
    assert(tmp_path/'failed-intent.json').exists()  # Visible is not yet acknowledged durable.
    with pytest.raises(OSError,match='directory durability failed'):
        writer._save_arrays(Path('unacknowledged.npz'),{'values':np.arange(4)})
    assert(writer.root/'unacknowledged.npz').exists()
