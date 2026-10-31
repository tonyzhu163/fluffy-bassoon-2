"""Read canonical source indices/episode receipts only; no raw arrays or models.

Run on the file host with --ledger and --outputs-root. JSON is printed to
stdout; the source ledger, indices and active artifacts are never modified.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
STAGES = ('warmup', 'learned', 'mixed', 'missing')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(row):
    stages = row.get('source_stages')
    if not isinstance(stages, dict) or not stages:
        return 'missing'
    active = {key for key, value in stages.items() if value}
    if active == {'warmup_gaussian'}:
        return 'warmup'
    if active == {'sac_online'}:
        return 'learned'
    return 'mixed'


def bounds(values):
    values = [value for value in values if type(value) is int]
    return dict(min=min(values), max=max(values)) if values else None


def empty_stage():
    return dict(episodes=0, successes=0, failures=0, success_missing=0,
        chunks=0, primitive_steps=0, settling_steps=0, before_optimizer_clock=[], after_optimizer_clock=[])


def analyze(ledger_path, outputs_root):
    ledger_path, outputs_root = Path(ledger_path), Path(outputs_root)
    ledger = json.loads(ledger_path.read_text())
    tasks, manifest, recovery_cases = [], [], []
    aggregate = dict(episodes=0, complete_k4_cases=0, actual_state_ids_summed_across_goals=0,
        nominal_state_vectors_summed_across_goals=0, measured_post_settle_states_summed_across_goals=0,
        successes=0, failures=0, charged_total_actions=0, committed_episode_primitives=0,
        committed_episode_settling=0, query_chunks=0, final_optimizer_updates_sum=0)
    missing = Counter()
    for suite in SUITES:
        for number in range(10):
            key = f'{suite}:{number:02d}'
            reservation = ledger['reservations'][key + '/train']
            path = outputs_root / reservation['run_id'] / 'experiment/index.json'
            index = json.loads(path.read_text())
            if (index.get('task_key') != key or index.get('partition') != 'train'
                    or index.get('completed') is not True):
                raise ValueError('Canonical TRAIN index is missing, mismatched or incomplete: ' + key)
            rows = index.get('episode_receipts', [])
            stages = {name: empty_stage() for name in STAGES}
            cases = defaultdict(list)
            episode_stages, before_clocks = [], []
            previous_after = 0
            fields_missing = Counter()
            for ordinal, row in enumerate(rows):
                stage = classify(row)
                episode_stages.append(stage)
                group = stages[stage]
                group['episodes'] += 1
                outcome = row.get('success')
                group['successes' if outcome is True else 'failures' if outcome is False else 'success_missing'] += 1
                for field in ('case_id', 'episode_in_case', 'initial_state', 'scene_seed',
                              'initial_state_vector_sha256', 'initial_simulator_state_sha256',
                              'initial_observation_sha256', 'source_stages', 'source_updates',
                              'chunks', 'primitive_steps', 'success'):
                    if field not in row or row[field] is None:
                        fields_missing[field] += 1
                group['chunks'] += row.get('chunks', 0)
                group['primitive_steps'] += row.get('primitive_steps', 0)
                group['settling_steps'] += 10  # Qualified fixed legacy reset contract.
                source_stages = row.get('source_stages', {})
                if sum(source_stages.values()) != row.get('chunks'):
                    fields_missing['stage_chunk_total_mismatch'] += 1
                after = row.get('source_updates')
                if (type(after) is not int or type(previous_after) is not int or after < previous_after):
                    fields_missing['missing_or_nonmonotone_optimizer_clock'] += 1
                    before = None
                else:
                    before = previous_after
                before_clocks.append(before)
                group['before_optimizer_clock'].append(before)
                group['after_optimizer_clock'].append(after)
                previous_after = after
                cases[row.get('case_id', f'missing-{ordinal}')].append(dict(row, ordinal=ordinal,
                    behavior_stage=stage, inferred_source_updates_before=before))
            patterns, case_stages = Counter(), Counter()
            case_counts = Counter()
            for case_id, episodes in cases.items():
                episodes = sorted(episodes, key=lambda row: row.get('episode_in_case', -1))
                complete = (len(episodes) == 4 and [row.get('episode_in_case') for row in episodes] == list(range(4)))
                if not complete:
                    fields_missing['incomplete_or_non_k4_case'] += 1
                    continue
                case_counts['complete_k4_cases'] += 1
                labels = {row['behavior_stage'] for row in episodes}
                case_stage = ('warmup_only' if labels == {'warmup'} else
                    'learned_only' if labels == {'learned'} else
                    'warmup_learned_transition' if labels == {'warmup', 'learned'} else 'unknown_or_mixed')
                case_stages[case_stage] += 1
                pattern = ''.join('S' if row.get('success') is True else 'F' if row.get('success') is False else '?' for row in episodes)
                patterns[pattern] += 1
                nominal = {row.get('initial_state_vector_sha256') for row in episodes}
                physical = {row.get('initial_simulator_state_sha256') for row in episodes}
                if len(nominal) != 1 or None in nominal:
                    fields_missing['case_nominal_state_mismatch_or_missing'] += 1
                if None in physical:
                    fields_missing['case_post_settle_state_unavailable'] += 1
                elif len(physical) != 1:
                    fields_missing['case_post_settle_state_mismatch'] += 1
                pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)
                         if episodes[i].get('success') is False and episodes[j].get('success') is True]
                if pairs:
                    case_counts['failure_then_later_success_cases'] += 1
                    case_counts['failure_then_later_success_pairs'] += len(pairs)
                    case_counts['recovery_cases_' + case_stage] += 1
                    if episodes[0].get('success') is False:
                        case_counts['first_episode_failure_then_later_success_cases'] += 1
                    if any(episodes[i]['behavior_stage'] == episodes[j]['behavior_stage'] == 'learned' for i, j in pairs):
                        case_counts['cases_with_learned_failure_then_learned_success'] += 1
                    if any(episodes[i]['behavior_stage'] == 'warmup' and episodes[j]['behavior_stage'] == 'learned' for i, j in pairs):
                        case_counts['cases_with_warmup_failure_then_learned_success'] += 1
                    recovery_cases.append(dict(task_key=key, case_id=case_id, pattern=pattern,
                        stages=[row['behavior_stage'] for row in episodes], initial_state=episodes[0].get('initial_state'),
                        initial_state_vector_sha256=episodes[0].get('initial_state_vector_sha256'),
                        optimizer_before=[row['inferred_source_updates_before'] for row in episodes],
                        optimizer_after=[row.get('source_updates') for row in episodes], ordered_failure_success_pairs=pairs))
            for group in stages.values():
                group['before_optimizer_clock_range'] = bounds(group.pop('before_optimizer_clock'))
                group['after_optimizer_clock_range'] = bounds(group.pop('after_optimizer_clock'))
            actual_ids = sorted({row['initial_state'] for row in rows if type(row.get('initial_state')) is int})
            nominal_hashes = {row['initial_state_vector_sha256'] for row in rows if row.get('initial_state_vector_sha256')}
            actual_hashes = {row['initial_simulator_state_sha256'] for row in rows if row.get('initial_simulator_state_sha256')}
            goals = dict(task_key=key, run_id=reservation['run_id'], episodes=len(rows),
                cases=len(cases), complete_k4_cases=case_counts['complete_k4_cases'],
                actual_state_ids=actual_ids, distinct_state_ids=len(actual_ids),
                distinct_nominal_vectors=len(nominal_hashes), measured_distinct_post_settle_states=len(actual_hashes) if actual_hashes else None,
                successes=sum(row.get('success') is True for row in rows), failures=sum(row.get('success') is False for row in rows),
                chunks=index.get('chunks'), charged_total_actions=index.get('total_action_budget_charge'),
                committed_episode_primitives=sum(row.get('primitive_steps', 0) for row in rows),
                committed_episode_settling=len(rows) * 10, stage=stages,
                case_outcome_patterns=dict(patterns), case_stage_counts=dict(case_stages),
                failure_success_counts=dict(case_counts), final_source_optimizer_updates=index.get('source_updates'),
                source_clock_before_range=bounds(before_clocks),
                source_clock_after_range=bounds([row.get('source_updates') for row in rows]),
                missing_fields=dict(fields_missing))
            tasks.append(goals)
            missing.update(fields_missing)
            manifest.append(dict(task_key=key, canonical_slot=key + '/train', run_id=reservation['run_id'],
                index_path=str(path), index_sha256=sha(path), reservation=reservation))
            for field, value in dict(episodes=len(rows), complete_k4_cases=case_counts['complete_k4_cases'],
                actual_state_ids_summed_across_goals=len(actual_ids), nominal_state_vectors_summed_across_goals=len(nominal_hashes),
                measured_post_settle_states_summed_across_goals=len(actual_hashes), successes=goals['successes'], failures=goals['failures'],
                charged_total_actions=goals['charged_total_actions'], committed_episode_primitives=goals['committed_episode_primitives'],
                committed_episode_settling=goals['committed_episode_settling'], query_chunks=index['chunks'],
                final_optimizer_updates_sum=index['source_updates']).items():
                aggregate[field] += value
    stage_totals, case_total, patterns = {}, Counter(), Counter()
    for stage in STAGES:
        stage_totals[stage] = {field: sum(task['stage'][stage][field] for task in tasks)
            for field in ('episodes', 'successes', 'failures', 'success_missing', 'chunks', 'primitive_steps', 'settling_steps')}
    for task in tasks:
        case_total.update(task['failure_success_counts'])
        patterns.update(task['case_outcome_patterns'])
    aggregate['episodes_with_measured_post_settle_state_hash'] = aggregate['episodes'] - missing['initial_simulator_state_sha256']
    if aggregate['episodes_with_measured_post_settle_state_hash'] == 0:
        aggregate['measured_post_settle_states_summed_across_goals'] = None
    return dict(format='libero40_train_metadata_statistics_v1', source_ledger_path=str(ledger_path),
        source_ledger_sha256=sha(ledger_path), source_streams=40, aggregate=aggregate,
        stages=stage_totals, case_counts=dict(case_total), case_outcome_patterns=dict(patterns), missing_fields=dict(missing),
        per_goal=tasks, failure_then_later_success_cases=recovery_cases, source_index_manifest=manifest,
        definitions=dict(stage='Each episode source_stages counter: warmup_gaussian or sac_online; mixed/missing reported separately',
            clocks='source_updates in each receipt is AFTER that episode. BEFORE is previous completed episode source_updates; initial fresh learner0',
            nominal_states='initial_state_vector_sha256 is the requested nominal simulator vector, not a measured post-settle state',
            post_settle_states='Only explicit initial_simulator_state_sha256 counts as measured post-settle physical state; missing is unknown, not zero',
            recovery='An actually recorded failed episode followed by a later success within the same complete K4; optimizer and exploration may change between episodes',
            costs='Committed episode costs separated from total action charge;10 settling actions per episode from the qualified legacy contract',
            limits='Metadata-only, all40 train streams, no raw arrays/NN/tests/VLA compute, no semantic exclusions. Outcome counts do not establish expert quality, memory use or target transfer'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--outputs-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.ledger, args.outputs_root), indent=2, allow_nan=False))
