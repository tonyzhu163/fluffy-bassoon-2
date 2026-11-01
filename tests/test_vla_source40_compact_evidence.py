"""Synthetic producer/legacy invariants; source preparation, NOT qualification.

The integration fixture explicitly replaces the simulator, motor and learner.
No fixture is proof of a real motor, actor, source corpus, allocation or grant.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_compact_evidence as evidence
from fluffy_bassoon.meta_rl import vla_source40_corpus as corpus
from fluffy_bassoon.meta_rl import vla_libero
from fluffy_bassoon.meta_rl.vla_source40_schedule import SOURCE_SUITES, content_hash
from test_vla_source40_lineage_adapter import feature


def contract():
    return dict(format='openpi_pi05_libero40_persistent_source_v1', source_only=True,
        source_data_from_targets=False, task=dict(suite='libero_10', task_id=0, task_key='libero_10:00'),
        history_id='libero_10:00/train/synthetic', motor=dict(receipt_sha256=evidence.PI05_RECEIPT,
            predicted_horizon=10, action_dim=32), feature_contract=feature(),
        compact_history_evidence=deepcopy(evidence.PROTOCOL))


class Learner:
    """Counters and toy serialized state only; no SAC/model implementation."""
    def __init__(self, *args):
        self.updates, self.replay, self.saved = 0, [], []
        self.agent = SimpleNamespace(_rng=np.array([1, 2], np.uint32), _actor=dict(
            params=dict(weight=np.zeros(1, np.float32)), batch_stats=dict(mean=np.zeros(1, np.float32)),
            step=0, opt_state=dict(moment=np.zeros(1, np.float32))))

    def action(self, observation):
        self.agent._rng = self.agent._rng+np.uint32(1)
        return np.full(32, 2., np.float32)  # Original collector clips learned requests.

    def update(self, batch, sampler):
        assert batch == 256
        sampler.integers(0, len(self.replay), batch)
        self.updates += 1
        self.agent._actor['step'] = self.updates
        return dict(updates=float(self.updates))

    def save(self, path, sampler, contract, **progress):
        self.saved.append(str(path))
        Path(path).write_bytes(pickle.dumps(dict(synthetic_only=True, updates=self.updates, progress=progress)))


class Features:
    contract = feature()
    calls, elapsed_seconds = 0, 0.

    def __call__(self, *args):
        self.calls += 1
        return np.zeros(4104, np.float32)

    def qualify(self, *args):
        return dict(synthetic_only=True)


class Motor:
    def __init__(self, args=None):
        import jax
        self.policy = SimpleNamespace(_rng=jax.random.PRNGKey(5))
        self.feature_encoder, self.calls, self.horizon = Features(), 0, 10
        self.metadata = dict(receipt_sha256=evidence.PI05_RECEIPT, weights_frozen=True,
            openpi_commit=vla_libero.OPENPI_COMMIT, checkpoint='gs://openpi-assets/checkpoints/pi05_libero',
            flow_steps=10, predicted_horizon=10, action_dim=32)

    def infer(self, observation, instruction, noise):
        import jax
        self.policy._rng = jax.random.split(self.policy._rng)[0]
        self.calls += 1
        return np.tile(np.asarray(noise)[0, :7], (10, 1)).astype(np.float32)


class Simulator:
    metadata = dict(suite='libero_10', task_id=0, bddl_sha256='synthetic-bddl',
        bddl_file='synthetic.bddl', instruction='public synthetic instruction',
        initial_states=50, horizon=400, fresh_environment_on_reset=True, settle_steps=10)

    def __init__(self, *args):
        self.resets, self.steps = [], []
        self.observation = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8),
                                state=np.zeros(8, np.float32))

    def reset(self, state, seed):
        self.resets.append((state, seed))
        return dict(observation=self.observation, initial_state_vector_sha256=content_hash(['state', state]),
            initial_observation_sha256=content_hash(['observation', state, seed]))

    def step(self, actions):
        self.steps.append(np.asarray(actions).copy())
        success = len(self.steps) % 2 == 0
        return dict(observation=self.observation, done=True, success=success, primitive_steps=len(actions),
            executed_actions=np.asarray(actions).copy(), rewards=[float(success)]*len(actions))

    def close(self):
        pass


def test_full_actor_identity_is_mapping_order_independent_and_not_params_only():
    learner = Learner()
    first = evidence.actor_identity(learner)
    learner.agent._actor = dict(reversed(list(learner.agent._actor.items())))
    assert evidence.actor_identity(learner) == first
    learner.agent._actor['batch_stats']['mean'][0] = 1.
    changed = evidence.actor_identity(learner)
    assert changed['actor_parameter_sha256'] == first['actor_parameter_sha256']
    assert changed['actor_state_sha256'] != first['actor_state_sha256']


@pytest.mark.parametrize('field', ['batch_stats', 'opt_state'])
def test_pre_action_guard_detects_full_actor_mutation_with_same_params(tmp_path, field):
    learner, sampler, motor = Learner(), np.random.default_rng(3), Motor()
    learner.updates = 20
    writer = evidence.CompactHistoryEvidence(tmp_path, contract())
    writer.begin(learner, sampler, motor, case=dict(case_id='case'), episode_in_case=0, ordinal=0)
    wrapped = corpus.OnlineSourceSimulator(Simulator(), learner, sampler, vla_libero.sac_observation)
    wrapped.reset(0, 1)
    learner.agent._actor[field][next(iter(learner.agent._actor[field]))][0] = 7.
    with pytest.raises(ValueError, match='Full acting state'):
        writer.choose(wrapped)(wrapped.sim.observation)
    assert len(learner.replay) == 0 and learner.updates == 20


@pytest.fixture
def integration(tmp_path, monkeypatch):
    """REAL collector control flow, explicitly FAKE runtime/model/simulator."""
    import fluffy_bassoon.gpu_execution as gpu
    selected_case = dict(case_id='libero_10:00/train/case-0', split='train', initial_state=0,
        scene_seed=123, episodes=2, learner_reset=False, simulator_reset='same fixed state/seed at every episode')
    tasks = [dict(task_key=f'{suite}:{i:02d}', suite=suite, task_id=i,
        bddl_sha256='synthetic-bddl', semantic_task='synthetic', partitions=dict(train=[selected_case]))
        for suite in SOURCE_SUITES for i in range(10)]
    plan = dict(tasks=tasks, seed=0, schedule_sha256='synthetic', collection_mode='complete_cases',
        target_chunks_per_task_partition=2, publish_every_cases=1, deployment='synthetic no science',
        partition_budgets=dict(train=dict(maximum_primitive_steps=20)),
        source_recipe=dict(warmup_chunks=500, batch_size=256, utd=20, gamma_per_primitive=.999))
    registry = {}
    def learner_factory(*args): registry['learner'] = Learner(); return registry['learner']
    def motor_factory(*args): registry['motor'] = Motor(); return registry['motor']
    def sim_factory(*args): registry['sim'] = Simulator(); return registry['sim']
    actual_save = Learner.save
    def save(self, path, sampler, *args, **kwargs):
        registry['sampler'] = sampler
        actual_save(self, path, sampler, *args, **kwargs)
    monkeypatch.setattr(Learner, 'save', save)
    monkeypatch.setattr(corpus, 'load_schedule', lambda *a: deepcopy(plan))
    monkeypatch.setattr(corpus.sys, 'platform', 'linux')
    monkeypatch.setattr(gpu, 'validate_execution', lambda *a: None)
    monkeypatch.setattr(corpus.subprocess, 'check_output', lambda args, **kwargs:
        'synthetic-source\n' if args[1:] == ['rev-parse', 'HEAD'] else '')
    monkeypatch.setattr(vla_libero, 'OfficialSAC', learner_factory)
    monkeypatch.setattr(vla_libero, 'FrozenOpenPI', motor_factory)
    monkeypatch.setattr(vla_libero, 'SimulatorClient', sim_factory)
    actual_sha = vla_libero.sha256
    monkeypatch.setattr(vla_libero, 'sha256', lambda path:
        evidence.PI05_RECEIPT if Path(path).name == 'READY.json' else actual_sha(path))
    def run(name, enabled=False, extra=(), continuation=None):
        output = tmp_path/name
        argv = ['--schedule', str(tmp_path/'schedule.json'), '--output', str(output),
            '--checkpoint', str(tmp_path/'weights'), '--openpi-root', str(tmp_path/'openpi'),
            '--dsrl-root', str(tmp_path/'dsrl'), '--task-key', 'libero_10:00', '--partition', 'train',
            '--source-commit', 'synthetic-source', '--max-primitive-steps', '20']
        if enabled: argv.append('--compact-history-evidence')
        code = corpus.main(argv+list(extra), continuation=continuation)
        index = json.loads((output/'index.json').read_text())
        traces = []
        for receipt in index['episode_receipts']:
            with gzip.open(output/receipt['trace'], 'rb') as stream: traces.append(pickle.load(stream))
        return dict(root=output, code=code, index=index, traces=traces, runtime=dict(registry))
    return run


def test_opt_in_does_not_change_noise_data_actions_costs_or_checkpoint_cadence(integration):
    legacy, opted = integration('legacy'), integration('opted', enabled=True)
    assert legacy['code'] == opted['code'] == 3  # No learned source; deliberately NOT qualification.
    for value in (legacy, opted):
        assert [r['summary']['success'] for r in value['traces']] == [False, True]
    assert not (legacy['root']/'compact-history').exists()
    assert 'compact_history_evidence' not in json.loads((legacy['root']/'contract.json').read_text())
    assert 'compact_history_evidence' not in legacy['index']
    for left, right in zip(legacy['traces'], opted['traces'], strict=True):
        assert len(left['rows']) == len(right['rows']) == 1
        for field in ('executed_noise', 'proposed_noise', 'native_initial_noise', 'public_features', 'executed_actions'):
            np.testing.assert_array_equal(left['rows'][0][field], right['rows'][0][field])
        for field in ('reward', 'raw_rewards', 'done', 'terminated', 'truncated', 'primitive_steps',
                      'source_updates_before', 'source_updates_after', 'source_updates_after_episode'):
            assert left['rows'][0][field] == right['rows'][0][field]
        assert 'source_behavior_identity' not in left['rows'][0]
        assert right['rows'][0]['source_behavior_identity']['actor_not_used'] is True
    for field in ('actual_primitive_steps', 'actual_settling_steps', 'total_action_budget_charge', 'source_updates', 'chunks'):
        assert legacy['index'][field] == opted['index'][field]
    assert len(legacy['runtime']['learner'].saved) == len(opted['runtime']['learner'].saved)
    assert [Path(p).name for p in legacy['runtime']['learner'].saved] == [Path(p).name for p in opted['runtime']['learner'].saved]
    assert legacy['runtime']['motor'].calls == opted['runtime']['motor'].calls
    assert legacy['runtime']['motor'].feature_encoder.calls == opted['runtime']['motor'].feature_encoder.calls
    chronology = json.loads(Path(opted['index']['compact_history_evidence']['path']).read_text())
    assert chronology['completed'] is True and len(chronology['episodes']) == 2
    assert all(r['before_recovery'] is None and r['after_recovery'] is None for r in chronology['episodes'])


def test_original_raw_pending_and_closed_trace_bind_same_compact_evidence(integration):
    value = integration('original-binding', enabled=True)
    for trace, receipt in zip(value['traces'], value['index']['episode_receipts'], strict=True):
        record = json.loads(Path(receipt['compact_episode_ref']['path']).read_text())
        assert trace['compact_history_format'] == evidence.FORMAT
        assert trace['behavior_identity'] == record['behavior_identity'] == trace['rows'][0]['source_behavior_identity']
        assert trace['compact_episode_clocks'] == record['clocks']
        assert trace['reset_ref'] == record['reset_ref']
        assert trace['rng_pre_sgd_ref'] == record['rng_pre_sgd']
        assert record['raw_trace']['sha256'] == receipt['trace_sha256']
    journals = sorted((value['root']/'raw/decision-journal').glob('decision-*.pkl.gz'))
    with gzip.open(journals[0], 'rb') as stream: pending = pickle.load(stream)['pending']
    assert pending['source_behavior_identity'] == value['traces'][0]['behavior_identity']


def test_same_contract_resume_uses_original_cursor_and_rejects_missing_history(integration):
    value = integration('resume-source', enabled=True)
    contract_value = json.loads((value['root']/'contract.json').read_text())
    writer = evidence.CompactHistoryEvidence(value['root'], contract_value)
    runtime = value['runtime']; before = len(runtime['sim'].resets)
    writer.resume(value['index']['episode_receipts'], runtime['learner'], runtime['sampler'], runtime['motor'])
    assert writer.cursor == 2 and len(runtime['sim'].resets) == before
    broken = deepcopy(value['index']['episode_receipts']); broken[-1].pop('compact_episode_ref')
    with pytest.raises(ValueError, match='retrofit'):
        writer.resume(broken, runtime['learner'], runtime['sampler'], runtime['motor'])


def test_resume_rejects_actual_rng_reset_before_any_episode(integration):
    value = integration('rng-reset-source', enabled=True)
    writer = evidence.CompactHistoryEvidence(value['root'], json.loads((value['root']/'contract.json').read_text()))
    runtime = value['runtime']; before = len(runtime['sim'].steps)
    runtime['sampler'].integers(0, 100)
    with pytest.raises(ValueError, match='Restored compact'):
        writer.resume(value['index']['episode_receipts'], runtime['learner'], runtime['sampler'], runtime['motor'])
    assert len(runtime['sim'].steps) == before


def test_interrupted_compact_attempt_is_retained_without_advancing_cursor(tmp_path):
    learner, sampler, motor = Learner(), np.random.default_rng(3), Motor()
    writer = evidence.CompactHistoryEvidence(tmp_path, contract())
    writer.begin(learner, sampler, motor, case=dict(case_id='case'), episode_in_case=0, ordinal=0)
    original = writer.active['rng_before']
    trace = writer.trace_path(0); trace.parent.mkdir(exist_ok=True)
    trace.write_bytes(b'original interrupted raw attempt')
    resumed = evidence.CompactHistoryEvidence(tmp_path, contract())
    resumed.begin(learner, sampler, motor, case=dict(case_id='case'), episode_in_case=0, ordinal=0)
    assert resumed.cursor == 0 and resumed.active['rng_before']['path'] != original['path']
    assert resumed.trace_path(0) != trace and trace.read_bytes() == b'original interrupted raw attempt'
    evidence.checked(original)
    assert learner.saved == []  # No additional full recovery cadence.


@pytest.mark.parametrize('mutation', [
    lambda x: x['task'].update(suite='libero_90', task_id=44, task_key='libero_90:44'),
    lambda x: x.update(data_role='target-onlineDEV'),
    lambda x: x.update(source_data_from_targets=True),
    lambda x: x['motor'].update(receipt_sha256='RLinf130'),
])
def test_producer_never_wraps_dev_or_an_incompatible_source_contract(tmp_path, mutation):
    value = contract(); mutation(value)
    with pytest.raises(ValueError, match='SOURCE40'):
        evidence.CompactHistoryEvidence(tmp_path, value)
    assert not (tmp_path/'compact-history').exists()


def test_opt_in_cannot_migrate_a_legacy_source_prefix(integration):
    with pytest.raises(ValueError, match='retrofit a legacy'):
        integration('migration', enabled=True, continuation=object())


def test_legacy_contract_cannot_enable_compact_evidence_on_resume(integration):
    legacy = integration('legacy-resume')
    original = (legacy['root']/'contract.json').read_bytes()
    with pytest.raises(ValueError, match='Resumed source contract changed'):
        integration('legacy-resume', enabled=True, extra=['--resume'])
    assert (legacy['root']/'contract.json').read_bytes() == original
    assert not (legacy['root']/'compact-history').exists()


def test_v1_params_only_evidence_does_not_gain_v2_identity(tmp_path):
    value = contract(); value['compact_history_evidence'] = dict(format='source40_compact_history_evidence_v1')
    with pytest.raises(ValueError): evidence.CompactHistoryEvidence(tmp_path, value)


@pytest.mark.parametrize('wrong_counter', [False, True])
def test_original_warmup_crossing_records_real_pre_and_post_burst_states(tmp_path, wrong_counter):
    """Toy preloaded replay is only an SGD-boundary fixture, never source data."""
    learner, sampler, motor, sim = Learner(), np.random.default_rng(3), Motor(), Simulator()
    learner.replay = [dict(synthetic_prefix_only=True)]*500
    writer = evidence.CompactHistoryEvidence(tmp_path, contract())
    case = dict(case_id='boundary-fixture', initial_state=0, scene_seed=123)
    writer.begin(learner, sampler, motor, case=case, episode_in_case=0, ordinal=0)
    def reset(state, seed, result):
        path = tmp_path/'reset.json'
        vla_libero.write_json(path, dict(returned=True, case_id=case['case_id'],
            initial_state=state, scene_seed=seed, initial_state_vector_sha256=result['initial_state_vector_sha256'],
            initial_observation_sha256=result['initial_observation_sha256']))
        writer.reset(path, result)
    wrapped = corpus.OnlineSourceSimulator(sim, learner, sampler, vla_libero.sac_observation, after_reset=reset)
    rows, summary = vla_libero.run_episode(wrapped, motor, writer.choose(wrapped),
        initial_state=0, scene_seed=123, chunk=10, bounded=False, record_features=True)
    assert wrapped.provenance[0]['source_updates_before'] == wrapped.provenance[0]['source_updates_after'] == 0
    writer.before_sgd(learner, sampler, motor, rows, summary)
    metrics = wrapped.finish_episode()
    assert learner.updates == 20 and len(learner.replay) == 501
    if wrong_counter:
        learner.updates += 1
        with pytest.raises(ValueError, match='Actual post-episode'):
            writer.after_sgd(learner, sampler, motor, metrics)
        return
    payload = writer.after_sgd(learner, sampler, motor, metrics)
    assert payload['behavior_identity']['actor_not_used'] is True
    assert payload['behavior_identity']['actor_state_sha256'] is None
    assert payload['actor_state_after_sha256'] == evidence.actor_identity(learner)['actor_state_sha256']
    assert payload['compact_episode_clocks']['before_updates'] == 0
    assert payload['compact_episode_clocks']['after_updates'] == 20
    assert payload['rng_pre_sgd_ref']['sha256'] != payload['rng_after_ref']['sha256']
    assert learner.saved == []
