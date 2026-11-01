"""Synthetic boundary regressions; SOURCE ONLY until qualified on allocated Linux.

No fixture represents a qualified motor, actor, source run or resource grant.
Teacher inference is stubbed explicitly where the assembly wiring is tested.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_lineage_adapter as adapter
from fluffy_bassoon.meta_rl.vla_history import FIELDS, load_dataset, save_dataset
from fluffy_bassoon.meta_rl.vla_source40_corpus import paired_arrays
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path.resolve()), sha256=adapter.sha(path))


def feature():
    return dict(kind='openpi_final_prefix_visual_mean_language_mean_public_state_v2',
        checkpoint_receipt_sha256=adapter.PI05_RECEIPT, model_config='pi05_libero',
        openpi_commit=adapter.OPENPI_COMMIT, visual_dim=2048, instruction_dim=2048, public_state_dim=8,
        pooling_dtype='float32', kv_cache_retained=False,
        pooling='separate NumPy float32 mask-weighted means after native hidden materialization',
        hidden_materialization='standalone module_jit output transferred to host before reduction',
        observation_dim=4104, trainable_feature_parameters=0,
        normalization='raw frozen hidden values; student fits normalization on training tasks only')


def declaration():
    slots = []
    for partition, tasks in (('train', sorted(adapter.ALL40)), ('validation', sorted(adapter.FIXED8))):
        for task in tasks:
            slots.append(dict(lineage_id=f'{task}/{partition}/seed-{len(slots)}', task_key=task,
                partition=partition, learner_seed=len(slots), condition_id=None,
                label_teacher_lineage_id=f'{task}/train/seed-{sorted(adapter.ALL40).index(task)}',
                cases=[dict(case_id=f'{task}/{partition}/case-0', split=partition,
                    initial_state=0 if partition == 'train' else 40,
                    scene_seed=100+len(slots), episodes=15, learner_reset=False)]))
    return dict(format=adapter.DECLARATION, source_only=True, source_data_from_targets=False,
        source_training_task_keys=sorted(adapter.ALL40), source_validation_task_keys=sorted(adapter.FIXED8),
        motor=deepcopy(adapter.MOTOR), feature_contract=feature(), noise_contract=adapter.NOISE,
        teacher_gate=dict(paired_cases=2, minimum_success_rate=.5), ad_improvement_gate=None,
        locked_query_cases=[dict(task_key='libero_90:57', initial_state=49, scene_seed=1)], lineages=slots)


def test_all40_scope_accepts_multiple_independent_lineages_of_one_task():
    value = declaration()
    extra = deepcopy(value['lineages'][0])
    extra.update(lineage_id=extra['lineage_id']+'-independent', learner_seed=1000)
    extra['cases'][0]['scene_seed'] = 3000
    value['lineages'].append(extra)
    assert adapter.validate_declaration(value) is value


@pytest.mark.parametrize('mutation', [
    lambda x: x['source_training_task_keys'].pop(),
    lambda x: x.update(source_validation_task_keys=x['source_validation_task_keys'][:1]),
    lambda x: x['motor'].update(parameter_sha256='0'*64),
    lambda x: x['motor'].update(model_config='RLinf-Pi05-LIBERO-130-fullshot-SFT'),
    lambda x: x['motor'].update(native_horizon=32, noise_dim=4),
    lambda x: x['feature_contract'].update(observation_dim=58),
    lambda x: x.update(data_role='target-onlineDEV'),
    lambda x: x.update(source_data_from_targets=True),
    lambda x: x['lineages'][0].update(learner_seed=x['lineages'][1]['learner_seed']),
    lambda x: x['lineages'][0].update(label_teacher_lineage_id=x['lineages'][-1]['lineage_id']),
    lambda x: x['lineages'][0]['cases'][0].update(initial_state=40),
    lambda x: x.update(locked_query_cases=[dict(task_key=x['lineages'][0]['task_key'],
        initial_state=0, scene_seed=x['lineages'][0]['cases'][0]['scene_seed'])]),
])
def test_declaration_refuses_scope_motor_split_role_and_teacher_leaks(mutation):
    value = declaration(); mutation(value)
    with pytest.raises(ValueError):
        adapter.validate_declaration(value)


def synthetic_source(root, partition='train'):
    """15 real-schema synthetic closed episodes cross the actual >500 warmup.

    Shared tiny images/zero features reduce fixture bytes, not production dims.
    The first13 episodes are Gaussian; the next two act at800/1600 updates.
    """
    root.mkdir(parents=True)
    task = 'libero_10:00'
    case = dict(case_id=f'{task}/{partition}/case-0', split=partition,
        initial_state=0 if partition == 'train' else 40, scene_seed=123,
        episodes=15, learner_reset=False)
    contract = dict(format='openpi_pi05_libero40_persistent_source_v1', source_only=True,
        source_data_from_targets=False, task=dict(task_key=task, suite='libero_10', task_id=0),
        partition=partition, history_id=f'{task}/{partition}/synthetic', cases=[case], learner_seed=17,
        motor=dict(receipt_sha256=adapter.PI05_RECEIPT, weights_frozen=True, openpi_commit=adapter.OPENPI_COMMIT,
            checkpoint='gs://openpi-assets/checkpoints/pi05_libero', flow_steps=10, predicted_horizon=10, action_dim=32),
        feature_contract=feature(), noise_contract=adapter.NOISE,
        source_recipe=dict(warmup_chunks=500, batch_size=256, utd=20, gamma_per_primitive=.999),
        compact_history_evidence=deepcopy(adapter.PROTOCOL))
    write(root/'contract.json', contract)
    observation = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8),
                       state=np.zeros(8, np.float32))
    sac = dict(pixels=np.zeros((64, 64, 3, 1), np.uint8), state=np.zeros((8, 1), np.float32))
    noise, features = np.zeros(32, np.float32), np.zeros(4104, np.float32)
    all_rows, receipts, updates, replay = [], [], 0, 0
    for ordinal in range(15):
        before = updates
        after = before + (800 if replay+40 > 500 else 0)
        rows = []
        for chunk in range(40):
            row = dict(motor_observation=observation, motor_next_observation=observation,
                sac_observation=sac, sac_next_observation=sac, instruction='synthetic public instruction',
                source_task_key=task, public_features=features, proposed_noise=noise, executed_noise=noise,
                native_initial_noise=np.tile(noise, (10, 1)), executed_actions=np.zeros((10, 7), np.float32),
                reward=-1., raw_rewards=[0.]*10, done=chunk == 39, success=False,
                terminated=False, truncated=chunk == 39, primitive_steps=10,
                primitive_discount=.999**10, requested_chunk_steps=10,
                source_partition=partition, case_id=case['case_id'], source_episode_in_case=ordinal,
                source_updates_before=before, source_updates_after=before,
                source_updates_after_episode=after, source_replay_before=replay,
                source_replay_after=replay+1, source_stage='warmup_gaussian' if before == 0 else 'sac_online',
                source_behavior_identity=None, source_choice_rng_before_sha256='c'*64)
            replay += 1; rows.append(row)
        summary = dict(initial_state=case['initial_state'], scene_seed=123, suite='libero_10', task_id=0,
            initial_state_vector_sha256=content_hash([partition, 'state']),
            initial_observation_sha256=content_hash([partition, 'obs']), success=False,
            chunks=40, primitive_steps=400)
        path = root/'raw'/f'episode-{ordinal:06d}.pkl.gz'; path.parent.mkdir(exist_ok=True)
        with gzip.open(path, 'wb') as stream:
            pickle.dump(dict(rows=rows, summary=summary, case=case,
                source_contract_sha256=content_hash(contract), compact_history_format=adapter.COMPACT_FORMAT), stream)
        receipts.append(dict(summary, case_id=case['case_id'], episode_in_case=ordinal,
            trace=str(path.relative_to(root)), trace_sha256=adapter.sha(path), source_updates=after))
        all_rows.extend(rows); updates = after
    checkpoint = root/'source.pkl'; checkpoint.write_bytes(b'synthetic not a model')
    datasets = {}
    trials = paired_arrays(all_rows, np.full((len(all_rows), 32), .25, np.float32))
    for objective, trial in trials.items():
        path = root/f'{objective}.npz'
        manifest = save_dataset(path, [trial], objective=objective, metadata=dict(
            source_contract=contract, task_key=task, global_task_split='source40', local_scene_split=partition,
            feature_contract=feature(), motor_sha256=adapter.PI05_RECEIPT, noise_contract=adapter.NOISE,
            trial_ids=[contract['history_id']], source_case_ids=[case['case_id']],
            source_generation_episodes=15, source_generation_chunks=600,
            source_actor_checkpoint_sha256=adapter.sha(checkpoint)))
        datasets[objective] = dict(path=path.name, sha256=manifest['sha256'])
    publication = dict(status='paired_ad_dpt', through_case=0, case_count=1,
        episodes=15, transitions=600, datasets=datasets, source_checkpoint=checkpoint.name,
        source_checkpoint_sha256=adapter.sha(checkpoint))
    index = dict(format='openpi_pi05_libero40_source_history_index_v1', completed=True,
        collection_qualified=True, source_only=True, task_key=task, partition=partition,
        source_updates=updates, chunks=600, collection_mode='complete_cases',
        case_coverage=dict(all_requested_cases_complete=True), source_contract_sha256=content_hash(contract),
        feature_contract=feature(), motor_receipt_sha256=adapter.PI05_RECEIPT, noise_contract=adapter.NOISE,
        episode_receipts=receipts, publications=[publication], actual_primitive_steps=6000,
        actual_settling_steps=150)
    chronology = write(root/'synthetic-chronology.json', dict(format=adapter.COMPACT_FORMAT, completed=True))
    index['compact_history_evidence'] = chronology
    index_ref = write(root/'index.json', index)
    motor_ref = write(root/'motor-proof.json', dict(sha256=adapter.MOTOR_PARAMETER_SHA256))
    binding = dict(lineage_id=contract['history_id'], index=index_ref, motor_before=motor_ref,
                   motor_after=motor_ref, chronology=chronology, teacher_qualification=None)
    slot = dict(lineage_id=contract['history_id'], task_key=task, condition_id=None, partition=partition,
        learner_seed=17, cases=[case], label_teacher_lineage_id=contract['history_id'])
    return adapter._load_source(slot, binding, dict(feature_contract=feature())), binding


def mutate_raw(source, ordinal, mutate):
    receipt = source['index']['episode_receipts'][ordinal]
    path = source['root']/receipt['trace']
    with gzip.open(path, 'rb') as stream: payload = pickle.load(stream)
    mutate(payload)
    with gzip.open(path, 'wb') as stream: pickle.dump(payload, stream)
    receipt['trace_sha256'] = adapter.sha(path)


def test_raw_projection_preserves_actual_history_and_separate_sgd_clocks(tmp_path):
    source, _ = synthetic_source(tmp_path/'source')
    rows, episodes = adapter._rows(source)
    assert len(rows) == 600 and len(episodes) == 15
    assert episodes[12]['behavior_updates_before'] == 0
    assert episodes[12]['behavior_updates_after_episode'] == 800
    assert rows[520]['source_updates_before'] == 800
    projected = adapter._project(source, rows, lambda obs: np.full((len(obs['state']), 32), .75, np.float32))
    for field in set(FIELDS)-{'target_noise'}:
        np.testing.assert_array_equal(projected['ad'][field], projected['dpt'][field])
    np.testing.assert_array_equal(projected['ad']['target_noise'], projected['ad']['executed_noise'])
    assert np.all(projected['dpt']['target_noise'] == .75)
    ad_only = adapter._project(source, rows, None)
    assert set(ad_only) == {'ad'}  # AD neither loads nor depends on a DPT teacher.
    for field in FIELDS:
        np.testing.assert_array_equal(ad_only['ad'][field], source['datasets']['ad'][field])


@pytest.mark.parametrize('mutation', [
    lambda x: x['rows'][0].update(source_updates_after=x['rows'][0]['source_updates_after_episode']),
    lambda x: x['rows'][0].update(source_updates_after_episode=900),
    lambda x: x['rows'][0].update(source_replay_before=0),
    lambda x: x['rows'][0].update(data_role='target-onlineDEV', source_assembly_admission=False),
    lambda x: x['rows'][0].update(done=True),
    lambda x: x['rows'][0].update(native_initial_noise=np.ones((10, 32), np.float32)),
    lambda x: x['rows'][0].update(motor_next_observation=dict(image=np.ones((2, 2, 3), np.uint8),
        wrist=np.zeros((2, 2, 3), np.uint8), state=np.zeros(8, np.float32))),
])
def test_raw_refuses_clock_conflation_replay_resets_dev_and_broken_transitions(tmp_path, mutation):
    source, _ = synthetic_source(tmp_path/'source')
    mutate_raw(source, 13, mutation)
    with pytest.raises(ValueError): adapter._rows(source)


def test_original_headroom_role_cannot_be_wrapped_as_source(tmp_path):
    index = write(tmp_path/'index.json', dict(source_only=True, data_role='target-onlineDEV'))
    write(tmp_path/'contract.json', dict(source_only=True))
    with pytest.raises(ValueError, match='repurposed-role'):
        adapter._load_source({}, dict(index=index), {})


def test_qualification_rejects_val_actor_wrong_checkpoint_and_low_competence(tmp_path):
    source, binding = synthetic_source(tmp_path/'source')
    slot = source['slot']; spec = declaration()
    evidence = write(tmp_path/'evidence.json', dict(synthetic_only=True))
    qualification = dict(source_only=True, source_data_from_targets=False,
        lineage_id=slot['lineage_id'], task_key=slot['task_key'], condition_id=None,
        checkpoint_sha256=source['selected']['source_checkpoint_sha256'], actor_parameter_sha256='a'*64,
        actor_state_sha256='b'*64,
        evidence=evidence, sampling_mode='deterministic_mean', paired_cases=[dict(
            case_id=f'fresh-{i}', task_key=slot['task_key'], initial_state=1+i, scene_seed=900+i,
            baseline_success=False, teacher_success=True) for i in range(2)])
    binding['teacher_qualification'] = write(tmp_path/'qualification.json', qualification)
    assert adapter._qualification(source, binding, spec) == qualification
    ceiling = deepcopy(qualification)
    for row in ceiling['paired_cases']: row['baseline_success'] = True
    binding['teacher_qualification'] = write(tmp_path/'qualification.json', ceiling)
    assert adapter._qualification(source, binding, spec) == ceiling  # Competent ceiling teacher is valid DPT.
    for key, bad in (('lineage_id', 'validation-actor'), ('checkpoint_sha256', 'b'*64),
                     ('sampling_mode', 'sampled_gaussian')):
        changed = dict(qualification, **{key:bad})
        binding['teacher_qualification'] = write(tmp_path/'qualification.json', changed)
        with pytest.raises(ValueError): adapter._qualification(source, binding, spec)
    qualification['paired_cases'][0]['teacher_success'] = False
    qualification['paired_cases'][1]['teacher_success'] = False
    binding['teacher_qualification'] = write(tmp_path/'qualification.json', qualification)
    with pytest.raises(ValueError, match='competence'):
        adapter._qualification(source, binding, spec)


@pytest.mark.parametrize('teacher_drift', [False, True])
def test_all40_assembly_keys_trials_by_lineage_and_uses_train_teacher_for_both_splits(tmp_path, monkeypatch, teacher_drift):
    """Assembly wiring ONLY: original raw admission/actor inference explicitly mocked."""
    spec = declaration()
    extra = deepcopy(spec['lineages'][0]); extra.update(lineage_id='independent-second-seed', learner_seed=1000)
    spec['lineages'].append(extra)
    ref = write(tmp_path/'original.json', {})
    spec_ref = write(tmp_path/'declaration.json', spec)
    bindings = [dict(lineage_id=s['lineage_id'], index=ref, motor_before=ref,
        motor_after=ref, chronology=ref, teacher_qualification=ref) for s in spec['lineages']]
    trial = dict(observations=np.zeros((3, 4104), np.float32), executed_noise=np.zeros((3, 32), np.float32),
        target_noise=np.zeros((3, 32), np.float32), rewards=np.array([-1., -1., 0.], np.float32),
        dones=np.array([True, False, True]), episode_starts=np.array([True, True, False]),
        teacher_steps=np.array([0, 20, 20], np.int64))
    loaded, labeled = [], []
    public = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8),
                  state=np.zeros(8, np.float32))
    recorded = dict(motor_observation=public, sac_observation=dict(
        pixels=np.zeros((64, 64, 3, 1), np.uint8), state=np.zeros((8, 1), np.float32)))
    def source(slot, binding, declaration):
        return dict(slot=slot, history_id=slot['lineage_id'], case_ids={slot['partition']+slot['task_key']},
            states={(slot['task_key'], slot['cases'][0]['initial_state'])},
            state_hashes={slot['partition']+slot['task_key']}, root=tmp_path,
            contract_ref=ref, dataset_manifest_refs=[], costs=dict(actual_primitive_steps=1),
            datasets=dict(dpt=dict(target_noise=np.zeros((1, 32), np.float32))),
            selected=dict(source_checkpoint='original.json', source_checkpoint_sha256=ref['sha256'], datasets={}))
    def teacher(source, qualification, dsrl_root):
        key = source['slot']['lineage_id']; loaded.append(key)
        def evaluate(obs): return np.full((len(obs['state']), 32), .001 if teacher_drift else 0., np.float32)
        evaluate.lineage_id = key
        return evaluate
    def project(source, rows, teacher):
        labeled.append((source['slot']['lineage_id'], teacher.lineage_id))
        return dict(ad=deepcopy(trial), dpt=dict(deepcopy(trial), target_noise=np.ones((3, 32), np.float32)))
    monkeypatch.setattr(adapter, '_linux', lambda *a: None)
    monkeypatch.setattr(adapter, '_load_source', source)
    monkeypatch.setattr(adapter, '_qualification', lambda *a: dict(evidence=ref))
    monkeypatch.setattr(adapter, '_teacher', teacher)
    monkeypatch.setattr(adapter, '_rows', lambda *a, **k: ([recorded], []))
    monkeypatch.setattr(adapter, '_chronology', lambda *a, **k: [])
    monkeypatch.setattr(adapter, '_project', project)
    if teacher_drift:
        with pytest.raises(ValueError, match='teacher-label replay'):
            adapter.assemble(spec_ref, bindings, tmp_path/'assembly', dsrl_root=None,
                expected_hostname='synthetic-no-allocation', source_commit='synthetic', objective='dpt')
        assert not (tmp_path/'assembly').exists()
        return
    result = adapter.assemble(spec_ref, bindings, tmp_path/'assembly', dsrl_root=None,
        expected_hostname='synthetic-no-allocation', source_commit='synthetic', objective='dpt')
    assert len(loaded) == 40  # One exact TRAIN teacher per task, not each behavior seed or VAL actor.
    assert dict(labeled) == {s['lineage_id']:s['label_teacher_lineage_id'] for s in spec['lineages']}
    for partition, count in (('train', 41), ('validation', 8)):
        artifact = result['datasets'][partition]['dpt']
        trials, manifest = load_dataset(artifact['path'], expected_sha256=artifact['sha256'])
        assert len(trials) == count and len(set(manifest['metadata']['trial_ids'])) == count
        assert manifest['metadata']['feature_contract'] == feature()
        assert 'TRAIN observations only' in manifest['metadata']['normalization']
        assert all(set(t) == set(FIELDS) for t in trials)


def test_full_recovery_refuses_missing_rng_and_inference_only_weights(tmp_path):
    from flax import serialization
    contract = dict(source_only=True, source_data_from_targets=False)
    train_state = dict(params=dict(weight=np.zeros(1, np.float32)), opt_state=[], step=0)
    state = dict(actor=train_state, critic=train_state, temp=train_state, target_critic_params={})
    payload = dict(contract=contract, boundary='complete physical episode',
        agent=serialization.msgpack_serialize(state), agent_rng=np.zeros(2, np.uint32),
        replay=[], updates=0, sampler={}, numpy_rng=(), python_rng=(),
        progress=dict(motor_rng=dict(format='jax_motor_key_v1', key_data=[0, 1])))
    def save(value):
        path = tmp_path/'recovery.pkl'
        with path.open('wb') as stream: pickle.dump(value, stream)
        return dict(path=str(path), sha256=adapter.sha(path))
    identity = adapter._recovery(save(payload), contract)
    assert identity['updates'] == 0 and identity['replay_rows'] == 0
    missing = deepcopy(payload); missing.pop('python_rng')
    with pytest.raises(ValueError, match='full source recovery'):
        adapter._recovery(save(missing), contract)
    weights = deepcopy(payload)
    weights['agent'] = serialization.msgpack_serialize(dict(actor=dict(params=state['actor']['params'])))
    with pytest.raises(ValueError, match='optimizer'):
        adapter._recovery(save(weights), contract)


@pytest.mark.parametrize('corruption', ['actor', 'replay', 'rng', 'order'])
def test_chronology_refuses_real_identity_resets_even_with_advancing_clocks(tmp_path, monkeypatch, corruption):
    ref = write(tmp_path/'original.json', {})
    rngs = [write(tmp_path/f'rng-{i}.json', dict(synthetic_rng=i)) for i in range(4)]
    episodes, records = [], []
    for i in range(3):
        behavior = dict(kind='gaussian_warmup' if i == 0 else 'sac', actor_not_used=i == 0,
            actor_parameter_sha256=None if i == 0 else str(i)*64,
            actor_state_sha256=None if i == 0 else str(i)*64, actor_state_scope=adapter.ACTOR_SCOPE,
            source_contract_sha256=content_hash({}),
            sampler_configuration=deepcopy(adapter.WARMUP) if i == 0 else None)
        raw = write(tmp_path/f'raw-{i}.json', dict(synthetic_raw=i))
        reset_values = dict(returned=True, case_id='synthetic-case', initial_state=0, scene_seed=123,
            initial_state_vector_sha256='a'*64, initial_observation_sha256='b'*64)
        reset = write(tmp_path/f'reset-{i}.json', reset_values)
        clocks = dict(before_updates=i*20, pre_sgd_updates=i*20, after_updates=(i+1)*20,
            replay_before=i, replay_after=i+1, chunks=1, primitive_steps=10)
        episodes.append(dict(raw_trace=raw, behavior_updates_before=i*20,
            behavior_updates_after_episode=(i+1)*20, chunks=1, behavior_identity=deepcopy(behavior),
            actor_after_parameter_sha256=str(i+1)*64, actor_state_after_sha256=str(i+1)*64,
            rng_before_ref=rngs[i], rng_pre_sgd_ref=rngs[i], rng_after_ref=rngs[i+1],
            reset_ref=reset, compact_episode_clocks=clocks, primitive_steps=10,
            **{k:v for k,v in reset_values.items() if k != 'returned'}))
        records.append(dict(raw_trace=raw, behavior_identity=behavior,
            actor_after_parameter_sha256=str(i+1)*64, actor_state_after_sha256=str(i+1)*64,
            rng_before=rngs[i], rng_pre_sgd=rngs[i], rng_after=rngs[i+1], reset_ref=reset, clocks=clocks,
            before_recovery=None, after_recovery=None))
    final = dict(updates=60, replay_rows=3, actor_parameter_sha256='3'*64, actor_state_sha256='3'*64,
        train_state_sha256='final-optimizer', replay_sha256='actual-recorded-replay', rng_sha256=rngs[3]['sha256'])
    source = dict(slot=dict(lineage_id='synthetic'), contract={}, root=tmp_path,
        expected_replay_sha256='actual-recorded-replay',
        selected=dict(source_checkpoint='original.json', source_checkpoint_sha256=ref['sha256']))
    # Only checkpoint/RNG numerical decoding is stubbed, not original-ref,
    # warmup/actor semantics, compact continuity or final replay comparison.
    monkeypatch.setattr(adapter, '_recovery', lambda *a: deepcopy(final))
    monkeypatch.setattr(adapter, '_compact_rng', lambda r: adapter.reference(r) and r['sha256'])
    chronology = dict(source_only=True, source_data_from_targets=False, lineage_id='synthetic', episodes=records,
        format=adapter.COMPACT_FORMAT, completed=True, source_contract_sha256=content_hash({}))
    binding = dict(chronology=write(tmp_path/'chronology.json', chronology))
    assert len(adapter._chronology(source, binding, episodes)) == 3
    if corruption == 'order':
        records[1]['raw_trace'] = episodes[0]['raw_trace']
    elif corruption == 'actor':
        records[1]['behavior_identity']['actor_parameter_sha256'] = 'a'*64
        episodes[1]['behavior_identity']['actor_parameter_sha256'] = 'a'*64
    elif corruption == 'replay':
        final['replay_sha256'] = 'query-contaminated-replay'
    else:
        reset = write(tmp_path/'reset-rng.json', dict(synthetic_rng='reset'))
        records[1]['rng_before'] = reset; episodes[1]['rng_before_ref'] = reset
    binding['chronology'] = write(tmp_path/'chronology.json', chronology)
    with pytest.raises(ValueError): adapter._chronology(source, binding, episodes)


def test_missing_original_behavior_identity_cannot_be_retrofitted(tmp_path):
    source, _ = synthetic_source(tmp_path/'source')
    _, episodes = adapter._rows(source)
    records = [dict(raw_trace=e['raw_trace'], behavior_identity=None,
        actor_after_parameter_sha256=None, actor_state_after_sha256=None,
        rng_before=None, rng_pre_sgd=None, rng_after=None, reset_ref=None) for e in episodes]
    chronology = write(tmp_path/'chronology.json', dict(source_only=True, source_data_from_targets=False,
        lineage_id=source['slot']['lineage_id'], episodes=records, format=adapter.COMPACT_FORMAT,
        completed=True, source_contract_sha256=content_hash(source['contract'])))
    with pytest.raises(ValueError, match='cannot be retrofitted'):
        adapter._chronology(source, dict(chronology=chronology), episodes)


def test_ad_improvement_is_separate_from_dpt_competence_and_cannot_filter_all40(tmp_path):
    spec = declaration()
    chosen = spec['lineages'][0]
    spec['ad_improvement_gate'] = dict(lineage_ids=[chosen['lineage_id']], paired_cases=1, minimum_success_gain=.5)
    evidence = write(tmp_path/'original-evidence.json', dict(synthetic_only=True))
    value = dict(source_only=True, source_data_from_targets=False, evidence=evidence,
        lineage_ids=[chosen['lineage_id']], paired_cases=[dict(lineage_id=chosen['lineage_id'],
            task_key=chosen['task_key'], case_id='fresh-assessment', initial_state=1, scene_seed=900,
            baseline_success=False, teacher_success=True)])
    ref = write(tmp_path/'improvement.json', value)
    result = adapter._ad_improvement(ref, spec)
    assert result['measured_success_gain'] == 1 and len(spec['source_training_task_keys']) == 40
    assert len(spec['lineages']) == 48  # Assessment scope does not select or drop data.
    value['paired_cases'][0]['baseline_success'] = True
    ref = write(tmp_path/'improvement.json', value)
    with pytest.raises(ValueError, match='improvement threshold'):
        adapter._ad_improvement(ref, spec)
    spec['ad_improvement_gate']['minimum_success_gain'] = 0
    assert adapter._ad_improvement(ref, spec)['measured_success_gain'] == 0
    with pytest.raises(ValueError, match='separate original'):
        adapter._ad_improvement(None, spec)
