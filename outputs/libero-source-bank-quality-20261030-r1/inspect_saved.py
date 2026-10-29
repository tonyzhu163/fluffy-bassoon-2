"""Read-only saved-array/trace inspection; no model imports or computation."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
SCALES = {
    'source1': (ROOT / 'libero40-subset-scales-20261030-r2/experiment/scales.json',
        'a52a2317f5b353d8e6afa0bc81131896426987197b49256d80b8faef354139dc'),
    'source2': (ROOT / 'libero40-two-goal-scales-20261030-r1/experiment/scales.json',
        'f1577bcbb5351743871ed2b3b0bcff627ea952f1a1aa0b724e147b7085e5d263')}
FIELDS = ('observations', 'executed_noise', 'target_noise', 'rewards', 'dones', 'episode_starts', 'teacher_steps')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while value := stream.read(8 * 1024 * 1024):
            digest.update(value)
    return digest.hexdigest()


def summary(values):
    return dict(episodes=len(values), successes=sum(row['success'] for row in values),
        failures=sum(not row['success'] for row in values), chunks=sum(row['chunks'] for row in values),
        primitive_steps=sum(row['primitive_steps'] for row in values))


def labels(array):
    value = np.asarray(array, np.float64)
    return dict(rows=len(value), coordinates=value.size, finite=bool(np.isfinite(value).all()),
        mean=float(value.mean()), std=float(value.std()), minimum=float(value.min()), maximum=float(value.max()),
        row_l2_mean=float(np.linalg.norm(value, axis=1).mean()),
        coordinates_outside_unit_box=int((np.abs(value) > 1).sum()),
        rows_outside_unit_box=int((np.abs(value) > 1).any(axis=1).sum()),
        mean_per_coordinate_across_row_std=float(value.std(axis=0).mean()))


def inspect_source(path):
    path = Path(path)
    index = json.loads(path.read_text())
    contract = json.loads((path.parent / 'contract.json').read_text())
    paired = max((row for row in index['publications'] if row['status'] == 'paired_ad_dpt'), key=lambda row: row['through_case'])
    arrays, datasets = {}, {}
    for objective in ('ad', 'dpt'):
        item = paired['datasets'][objective]
        data_path = path.parent / item['path']
        manifest = json.loads(data_path.with_suffix('.json').read_text())
        with np.load(data_path, allow_pickle=False) as archive:
            arrays[objective] = {name: archive[name].copy() for name in archive.files}
        datasets[objective] = dict(path=str(data_path), sha256=sha(data_path), expected_sha256=item['sha256'],
            manifest_actor_checkpoint_sha256=manifest['metadata']['source_actor_checkpoint_sha256'],
            trials=manifest['trials'], transitions=manifest['transitions'],
            finite=all(bool(np.isfinite(arrays[objective][name]).all()) for name in FIELDS),
            labels=labels(arrays[objective]['target_noise']), compressed_bytes=data_path.stat().st_size)
    ad, dpt = arrays['ad'], arrays['dpt']
    errors = Counter()
    matched_histories = {field: bool(np.array_equal(ad[field], dpt[field])) for field in FIELDS if field != 'target_noise'}
    episodes, cursor, raw_bytes, checked = [], 0, 0, 0
    raw_observation_keys = set()
    for ordinal, receipt in enumerate(index['episode_receipts']):
        trace = path.parent / receipt['trace']
        if sha(trace) != receipt['trace_sha256']:
            errors['raw_trace_sha'] += 1
        with gzip.open(trace, 'rb') as stream:
            payload = pickle.load(stream)
        rows = payload['rows']
        raw_bytes += trace.stat().st_size
        clocks = ad['teacher_steps'][cursor:cursor + len(rows)]
        if len(rows) != receipt['chunks'] or not np.all(clocks == clocks[0]):
            errors['fixed_episode_clock_or_length'] += 1
        episode = dict(ordinal=ordinal, case_id=receipt['case_id'], episode_in_case=receipt['episode_in_case'],
            initial_state=receipt['initial_state'], success=bool(receipt['success']), chunks=len(rows),
            primitive_steps=receipt['primitive_steps'], clock_before=int(clocks[0]),
            updates_after=receipt['source_updates'], actual_behavior='post_update' if clocks[0] > 0 else 'pre_update')
        episodes.append(episode)
        for row in rows:
            raw_observation_keys.add(tuple(sorted(row['motor_observation'])))
            for field, raw in (('observations', row['public_features']), ('executed_noise', row['executed_noise']),
                    ('target_noise', row['executed_noise']), ('rewards', row['reward']),
                    ('dones', row['done']), ('teacher_steps', row['source_updates_before'])):
                if not np.array_equal(ad[field][cursor], np.asarray(raw)):
                    errors['raw_to_ad_' + field] += 1
            noise = np.asarray(row['native_initial_noise'])
            if noise.shape != (10, 32) or not np.array_equal(noise, np.broadcast_to(row['executed_noise'], (10, 32))):
                errors['tied_native_noise'] += 1
            count = int(row['primitive_steps'])
            if np.asarray(row['executed_actions']).shape != (count, 7) or len(row['raw_rewards']) != count:
                errors['action_reward_count'] += 1
            if row['source_stage'] != ('sac_online' if row['source_updates_before'] > 0 else 'warmup_gaussian'):
                errors['stage_clock'] += 1
            cursor += 1
            checked += 1
    expected_starts = np.r_[True, ad['dones'][:-1]]
    if not np.array_equal(ad['episode_starts'], expected_starts) or not ad['dones'][-1]:
        errors['episode_boundary'] += 1
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode['case_id']].append(episode)
    adjacent_recoveries = [(prior, current) for values in groups.values() for prior, current in zip(values, values[1:])
        if not prior['success'] and current['success']]
    failures = [row for row in episodes if not row['success']]
    versions, counts = np.unique(ad['teacher_steps'], return_counts=True)
    checkpoint = path.parent / paired['source_checkpoint']
    recovery = path.parent / 'source-recovery.pkl'
    return dict(index_path=str(path), index_sha256=sha(path), task=index['task_key'], partition=index['partition'],
        source_contract_sha256=index['source_contract_sha256'], learner_seed=contract['learner_seed'],
        overall=summary(episodes), pre_update=summary([row for row in episodes if row['clock_before'] == 0]),
        post_update=summary([row for row in episodes if row['clock_before'] > 0]),
        initial_state_ids=sorted({row['initial_state'] for row in index['episode_receipts']}),
        nominal_state_hashes=sorted({row['initial_state_vector_sha256'] for row in index['episode_receipts']}),
        complete_seeded_cases=len(groups), episodes_per_case=sorted({len(values) for values in groups.values()}),
        first_postepisode_optimizer_burst=next(row for row in episodes if row['updates_after'] > 0),
        first_actual_postupdate_behavior=next(row for row in episodes if row['clock_before'] > 0),
        final_source_optimizer_updates=index['source_updates'],
        actual_behavior_versions=[dict(updates=int(version), chunks=int(count)) for version, count in zip(versions, counts)],
        by_prior_same_case_support_episodes={str(position): dict(
            **summary([row for row in episodes if row['episode_in_case'] == position]),
            postupdate_chunks=sum(row['chunks'] for row in episodes if row['episode_in_case'] == position and row['clock_before'] > 0))
            for position in range(4)},
        cases_containing_both_pre_and_postupdate_behavior=[case for case, values in groups.items()
            if {row['actual_behavior'] for row in values} == {'pre_update', 'post_update'}],
        failure_episodes=failures,
        adjacent_failed_then_success_same_case=[dict(case_id=current['case_id'],
            from_episode=prior['episode_in_case'], to_episode=current['episode_in_case'],
            before_clock=prior['clock_before'], after_clock=current['clock_before']) for prior, current in adjacent_recoveries],
        datasets=datasets, shared_ad_dpt_history_fields=matched_histories,
        ad_target_equals_actual_executed_noise=bool(np.array_equal(ad['target_noise'], ad['executed_noise'])),
        dpt_is_final_actor_mean_label_from_publish_code=True,
        raw_checked_chunks=checked, raw_trace_bytes=raw_bytes, raw_public_observation_keys=[list(keys) for keys in sorted(raw_observation_keys)],
        raw_and_array_issues=dict(errors),
        source_actor=dict(path=str(checkpoint), sha256=sha(checkpoint), expected_sha256=paired['source_checkpoint_sha256'],
            bytes=checkpoint.stat().st_size),
        source_full_recovery=dict(path=str(recovery), sha256=sha(recovery), expected_sha256=index['latest_recovery_sha256'],
            bytes=recovery.stat().st_size),
        physical_costs=dict(primitive_steps=index['actual_primitive_steps'], settling_steps=index['actual_settling_steps'],
            charged_actions=index.get('total_action_budget_charge', index['actual_primitive_steps'] + index['actual_settling_steps'])))


def assembled_arrays(assembly_path, assembly):
    results = {}
    for partition in ('train', 'validation'):
        for objective in ('ad', 'dpt'):
            item = assembly['datasets'][partition][objective]
            path = assembly_path.parent / item['path']
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            cursor, matches = 0, []
            for source in assembly['selected_sources'][partition]:
                original = Path(source['index_path']).parent / source['selected_publication']['datasets'][objective]['path']
                with np.load(original, allow_pickle=False) as archive:
                    count = len(archive['observations'])
                    matches.append(dict(task=source['task'], rows=count,
                        all_fields_exact=all(np.array_equal(arrays[field][cursor:cursor + count], archive[field]) for field in FIELDS)))
                cursor += count
            results[partition + '_' + objective] = dict(path=str(path), sha256=sha(path), expected_sha256=item['sha256'],
                lengths=arrays['lengths'].tolist(), rows=cursor, original_history_matches=matches,
                source_goal_trials_retained=len(matches) == len(arrays['lengths']))
    return results


def sampled_windows(scales, assembly):
    # Deserialize saved tensor/checkpoint metadata on CPU only. No policy/model
    # class is constructed; no forward pass, loss or optimizer is evaluated.
    import torch
    matrix_path = Path(next(row['index_path'] for row in scales['scales'] if row['updates'] == 500))
    matrix = json.loads(matrix_path.read_text())
    full = next(row for row in matrix['controllers'] if row['objective'] == 'ad' and row['mode'] == 'full')
    saved = torch.load(full['full_recovery_path'], map_location='cpu', weights_only=False)
    contract = saved['contract']
    root = Path(scales['assembly_path']).parent
    with np.load(root / assembly['datasets']['train']['ad']['path'], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    trials, offset = [], 0
    for source, length in zip(assembly['selected_sources']['train'], arrays['lengths']):
        index = json.loads(Path(source['index_path']).read_text())
        row_episodes, row_cases, roles = [], [], []
        for episode, receipt in enumerate(index['episode_receipts']):
            row_episodes += [episode] * receipt['chunks']
            row_cases += [receipt['case_id']] * receipt['chunks']
            roles += [receipt['episode_in_case']] * receipt['chunks']
        episode_starts = np.flatnonzero(arrays['episode_starts'][offset:offset + length])
        episode_ids = np.asarray(row_episodes)
        trials.append(dict(length=int(length), clocks=arrays['teacher_steps'][offset:offset + length],
            episode_start=episode_starts[episode_ids],
            previous_episode_start=episode_starts[np.maximum(episode_ids - 1, 0)],
            cases=np.asarray(row_cases), roles=np.asarray(roles)))
        offset += int(length)
    rng, counts = np.random.default_rng(contract['seed']), Counter()
    per_goal = defaultdict(Counter)
    for _ in range(saved['update']):
        for trial_index in rng.integers(len(trials), size=contract['batch_size']):
            trial = trials[int(trial_index)]
            end = int(rng.integers(1, trial['length'] + 1))
            length = int(rng.integers(1, min(contract['max_context'], end) + 1))
            start = end - length
            post = trial['clocks'][start:end] > 0
            counts['windows'] += 1
            counts['all_loss_label_positions'] += length
            counts['preupdate_loss_label_positions'] += int((~post).sum())
            counts['postupdate_loss_label_positions'] += int(post.sum())
            counts['windows_crossing_physical_cases'] += int(len(set(trial['cases'][start:end])) > 1)
            per_goal[int(trial_index)]['windows'] += 1
            per_goal[int(trial_index)]['postupdate_loss_label_positions'] += int(post.sum())
            if post.any():
                latest = start <= trial['episode_start'][start:end][post]
                previous = start <= trial['previous_episode_start'][start:end][post]
                counts['windows_with_postupdate_labels'] += 1
                counts['postupdate_positions_with_latest_terminal_feedback'] += int(latest.sum())
                counts['postupdate_positions_with_full_previous_episode'] += int(previous.sum())
                counts['windows_all_postupdate_positions_have_latest_terminal_feedback'] += int(latest.all())
                counts['windows_no_postupdate_position_has_latest_terminal_feedback'] += int(not latest.any())
                counts['postupdate_positions_with_entire_prior_learning_run'] += int(post.sum()) * int(start == 0)
                one_shot = post & (trial['roles'][start:end] == 1)
                counts['postupdate_positions_in_exact_one_support_query_episode'] += int(one_shot.sum())
                counts['those_positions_with_full_actual_support_episode'] += int((start <= trial['previous_episode_start'][start:end][one_shot]).sum())
    checks = []
    for controller in matrix['controllers']:
        checkpoint = torch.load(controller['full_recovery_path'], map_location='cpu', weights_only=False)
        checks.append(dict(objective=controller['objective'], mode=controller['mode'], update=checkpoint['update'],
            full_recovery_sha256=sha(controller['full_recovery_path']), expected_sha256=controller['full_recovery_sha256'],
            final_sampler_state_matches_reconstructed_windows=checkpoint['rng']['sampler'] == rng.bit_generator.state))
    return dict(segmentation='One whole learner-run trial per goal, not individual K4 cases; finite random windows start with fresh recurrent state',
        saved_ad_full_contract=dict(seed=contract['seed'], batch_size=contract['batch_size'], max_context=contract['max_context'],
            recurrent_memory_length=contract['config']['context_length'], updates=saved['update']),
        counts=dict(counts), per_trial={str(key): dict(value) for key, value in per_goal.items()}, full_recovery_checks=checks,
        observable_context='First window token still contains the actual immediately preceding noise/reward/done; feedback before that token is absent',
        limitation='Teacher replay/optimizer persist across cases. Full responsible replay-gradient history is not an input; coverage statistics describe observable feedback, not exact reconstruction of SAC learning')


sources, banks = {}, {}
for name, (path, expected) in SCALES.items():
    if sha(path) != expected:
        raise ValueError('Pinned scale file differs: ' + name)
    scales = json.loads(path.read_text())
    assembly_path = Path(scales['assembly_path'])
    if sha(assembly_path) != scales['assembly_file_sha256']:
        raise ValueError('Pinned assembly file differs: ' + name)
    assembly = json.loads(assembly_path.read_text())
    banks[name] = dict(scales_path=str(path), scales_sha256=expected, assembly_path=str(assembly_path),
        assembly_file_sha256=scales['assembly_file_sha256'], assembly_sha256=assembly['assembly_sha256'],
        source_task_keys=assembly['source_task_keys'], datasets=assembly['datasets'],
        source_costs_count_once=assembly['collection_costs_count_once'], sources={})
    for partition, rows in assembly['selected_sources'].items():
        banks[name]['sources'][partition] = []
        for row in rows:
            index_path = row['index_path']
            if sha(index_path) != row['index_sha256']:
                raise ValueError('Pinned source index differs')
            if index_path not in sources:
                sources[index_path] = inspect_source(index_path)
            banks[name]['sources'][partition].append(index_path)
    banks[name]['assembled_array_verification'] = assembled_arrays(assembly_path, assembly)
    banks[name]['actual_training_window_analysis'] = sampled_windows(scales, assembly)
qualification_path = ROOT / 'libero40-pi05-source-smoke-20261029-s00-train-r2/experiment/index.json'
qualification = json.loads(qualification_path.read_text())
unselected = dict(index_path=str(qualification_path), index_sha256=sha(qualification_path),
    selected_in_either_training_bank=str(qualification_path) in sources,
    episodes=len(qualification['episode_receipts']), successes=sum(row['success'] for row in qualification['episode_receipts']),
    chunks=qualification['chunks'], source_updates=qualification['source_updates'],
    primitive_steps=qualification['actual_primitive_steps'], settling_steps=qualification['actual_settling_steps'],
    charged_actions=qualification['actual_primitive_steps'] + qualification['actual_settling_steps'])
print(json.dumps(dict(format='libero_saved_source_bank_quality_v1', date='2026-10-30',
    inspection='Read-only NumPy/JSON/gzip/pickle inspection and CPU torch.load of saved checkpoint metadata; no model construction, forward/loss/optimizer computation, simulator, tests or GPU',
    banks=banks, sources=sources, original_unselected_spatial_training_qualification=unselected), indent=2, allow_nan=False))
