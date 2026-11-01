"""Frozen ALL40 profile checks, prepared for allocated Linux CPU execution.

Only the three published JSON fixtures and pure metadata/panel helpers are used.
No original source checkpoint, JAX state, model or simulator is opened here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Allocated Linux CPU only')
ROOT = Path(__file__).resolve().parents[1] / 'outputs/libero40-teacher-assessment-20261101-r1'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
ACTIVE = ('raw_native_full_H_Gaussian', 'exact_source_tied_Gaussian_warmup',
          'preselected_final_TRAIN_SAC_mean', 'preselected_final_TRAIN_SAC_sample')
OMITTED = ('original_initialized_SAC_mean_if_retained', 'original_initialized_SAC_sample_if_retained')
FULL_ORDER = (ACTIVE[0], ACTIVE[1], OMITTED[0], ACTIVE[2], OMITTED[1], ACTIVE[3])


@pytest.fixture(scope='module')
def assessment():
    # Deferred so the Linux-only test module does not import numerical runtime
    # dependencies merely during accidental discovery on a different host.
    from fluffy_bassoon.meta_rl import vla_source40_teacher_assessment
    return vla_source40_teacher_assessment


@pytest.fixture(scope='module')
def frozen(assessment):
    documents = {}
    for name, expected in (('ACTOR-TABLE.json', assessment.TABLE_SHA),
                           ('METADATA-INVENTORY.json', assessment.INVENTORY_SHA),
                           ('ASSET-BINDING.json', assessment.ASSET_SHA)):
        payload = (ROOT / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        documents[name] = json.loads(payload)
    return documents


@pytest.fixture
def assets(frozen):
    return (deepcopy(frozen['ACTOR-TABLE.json']),
            deepcopy(frozen['METADATA-INVENTORY.json']),
            deepcopy(frozen['ASSET-BINDING.json']))


def test_frozen_all40_interleaved_ordinals_and_current_canonical_train_leaves(assessment, assets):
    table, metadata, binding = assets
    assert assessment.FORMAT == 'libero40_existing_train_teacher_assessment_v1'
    assert assessment.SUITES == SUITES
    assert assessment.ROLES == ACTIVE and assessment.OMITTED == OMITTED
    before = deepcopy((table, metadata))
    provenance = assessment.validate_assets(table, metadata)
    expected = [f'{SUITES[i % 4]}:{i // 4:02d}' for i in range(40)]
    assert len(table['actors']) == len(metadata['all40']) == 40
    assert [row['source_ordinal'] for row in table['actors']] == list(range(40))
    assert [row['task_key'] for row in table['actors']] == expected
    assert [row['task_key'] for row in metadata['all40']] == expected
    assert len({row['terminal_train_root'] for row in table['actors']}) == 40
    assert len({row['canonical_terminal_train']['fixed_contract']['history_id']
                for row in metadata['all40']}) == 40
    assert binding['new_assessment_outcomes_seen'] is False
    assert binding['assessment_only_no_source_training_admission'] is True
    assert provenance['scope'] == 'Bound metadata, not a fresh raw-journal replay'
    assert (table, metadata) == before


def test_active_panel_exact320_cells_seeds_roles_and_natural_episode_action_ceiling(assessment, assets):
    table, metadata, binding = assets
    assessment.validate_assets(table, metadata)
    planned = assessment.panel(table)
    assert len(planned) == binding['planned_episodes'] == 320
    assert binding['mandatory_roles'] == list(ACTIVE)
    assert binding['physical_initial_state_ids'] == [40, 41] and binding['scene_seed'] == 123
    expected = [(i, state, role) for i in range(40) for state in (40, 41) for role in ACTIVE]
    assert [(s['source_ordinal'], s['state'], s['role']) for s in planned] == expected
    assert len({(s['task_key'], s['state'], s['role']) for s in planned}) == 320
    for spec in planned:
        ordinal = spec['source_ordinal']
        assert spec['task_key'] == f'{SUITES[ordinal % 4]}:{ordinal // 4:02d}'
        assert spec['scene_seed'] == 123 and spec['status'] == 'PLANNED'
        assert spec['seed'] == 20261101 + 1000 * ordinal + spec['state']
    assert assessment.SCHEDULE_ACTIONS == binding['maximum_new_actions_for_bound_schedule'] == 320 * (400 + 10) == 131200
    assert assessment.MAX_ACTIONS == binding['original_grant_action_ceiling_unchanged'] == 480 * (400 + 10) == 196800
    assert binding['optional_initial_capacity_not_repurposable'] is True
    assert binding['execution_source'] is None and binding['GPU_allocation'] is None


def test_missing_originals_stay160_explicit_omissions_without_fallback_or_new_draws(assessment, assets):
    table, metadata, binding = assets
    assert table['initial_actor_candidates_proven'] == binding['initialized_roles_available'] == 0
    assert all(row['retained_original_initial_actor'] is None for row in table['actors'])
    assert all(row['canonical_terminal_train']['pre_SGD_candidates'] == [] for row in metadata['all40'])
    full = assessment.panel(table, include_omitted=True)
    planned = assessment.panel(table)
    assert len(full) == 480
    assert [(s['source_ordinal'], s['state'], s['role']) for s in full] == [
        (i, state, role) for i in range(40) for state in (40, 41) for role in FULL_ORDER]
    omitted = [s for s in full if s['role'] in OMITTED]
    assert len(omitted) == binding['initialized_roles_omitted'] == 160
    assert all(s['status'] == 'OMITTED_UNPROVEN_INITIAL_IN_BOUNDED_INDEX_SCOPE' for s in omitted)
    assert [s for s in full if s['status'] == 'PLANNED'] == planned
    assert {s['role'] for s in omitted} == set(OMITTED)
    # Omitted placeholders use the same already-declared case seed, without
    # creating another physical draw or relabeling native/final as initial.
    case_seeds = {(s['source_ordinal'], s['state']): s['seed'] for s in planned}
    assert all(s['seed'] == case_seeds[s['source_ordinal'], s['state']] for s in omitted)
    assert not any(s['role'] in OMITTED for s in planned)


def test_prior_train_costs_count_only_current_cumulative_leaves_never_ancestors(assessment, assets):
    table, metadata, _ = assets
    provenance = assessment.validate_assets(table, metadata)
    expected = sum(row['canonical_terminal_train']['actions'] for row in metadata['all40'])
    ancestor_actions = sum(ancestor['actions'] for row in metadata['all40'] for ancestor in row['ancestry'])
    assert expected == provenance['metadata_cumulative_TRAIN_actions'] == 351297
    assert ancestor_actions > 0
    assert provenance['metadata_cumulative_TRAIN_actions'] != expected + ancestor_actions
    assert provenance['prior_costs_charged_again'] is False
    assert provenance['canonical_ledger_ref'] == metadata['canonical_ledger_ref']
    assert len(provenance['source_cost_groups']) == 40
    for group, row in zip(provenance['source_cost_groups'], metadata['all40']):
        leaf = row['canonical_terminal_train']
        assert group == dict(history_id=leaf['fixed_contract']['history_id'], current_root=leaf['root'],
                             cumulative_actions=leaf['actions'], ancestors_not_added=row['ancestry'])


@pytest.mark.parametrize('fault', [
    'missing_actor', 'missing_original', 'reordered_actor', 'reordered_original',
    'duplicate_ordinal', 'duplicate_task', 'duplicate_root', 'duplicate_history',
    'fabricated_initial_count', 'fabricated_initial_actor', 'fabricated_initial_candidate',
    'wrong_motor', 'wrong_noise', 'wrong_public_input', 'wrong_horizon', 'wrong_chunk',
    'phone_warmup', 'wrong_publication_sha', 'missing_publication_checkpoint',
    'wrong_publication_chunks', 'wrong_publication_episodes', 'incomplete_leaf',
    'wrong_canonical_root', 'noncanonical_plan_leaf', 'wrong_index', 'wrong_contract',
    'BDDL_mismatch', 'unresolved_competing_leaf', 'post_outcome_inventory', 'wrong_inventory_binding',
])
def test_rejects_changed_or_fabricated_frozen_profile_bindings(assessment, assets, fault):
    table, metadata, _ = assets
    row = table['actors'][0]
    original = metadata['all40'][0]
    leaf = original['canonical_terminal_train']
    fixed = leaf['fixed_contract']
    if fault == 'missing_actor':
        table['actors'].pop()
    elif fault == 'missing_original':
        metadata['all40'].pop()
    elif fault == 'reordered_actor':
        table['actors'][0], table['actors'][1] = table['actors'][1], table['actors'][0]
    elif fault == 'reordered_original':
        metadata['all40'][0], metadata['all40'][1] = metadata['all40'][1], metadata['all40'][0]
    elif fault == 'duplicate_ordinal':
        table['actors'][1]['source_ordinal'] = 0
    elif fault == 'duplicate_task':
        table['actors'][1]['task_key'] = row['task_key']
    elif fault == 'duplicate_root':
        table['actors'][1]['terminal_train_root'] = row['terminal_train_root']
        metadata['all40'][1]['canonical_terminal_train']['root'] = leaf['root']
    elif fault == 'duplicate_history':
        metadata['all40'][1]['canonical_terminal_train']['fixed_contract']['history_id'] = fixed['history_id']
    elif fault == 'fabricated_initial_count':
        table['initial_actor_candidates_proven'] = 1
    elif fault == 'fabricated_initial_actor':
        row['retained_original_initial_actor'] = deepcopy(row['checkpoint'])
    elif fault == 'fabricated_initial_candidate':
        leaf['pre_SGD_candidates'] = [deepcopy(row['checkpoint'])]
    elif fault == 'wrong_motor':
        fixed['motor']['receipt_sha256'] = '0' * 64
    elif fault == 'wrong_noise':
        fixed['noise_contract'] = 'bounded Gaussian or rescaled noise is a different archived actor interface'
    elif fault == 'wrong_public_input':
        fixed['feature_contract']['observation_dim'] = 58
    elif fault == 'wrong_horizon':
        fixed['horizon'] = 520
    elif fault == 'wrong_chunk':
        fixed['chunk'] = 32
    elif fault == 'phone_warmup':
        fixed['source_recipe']['warmup_chunks'] = 32
    elif fault == 'wrong_publication_sha':
        row['checkpoint']['published_sha256'] = '0' * 64
    elif fault == 'missing_publication_checkpoint':
        leaf['final_publication_checkpoint']['exists'] = False
    elif fault == 'wrong_publication_chunks':
        leaf['final_publication']['transitions'] -= 1
    elif fault == 'wrong_publication_episodes':
        leaf['final_publication']['episodes'] += 1
    elif fault == 'incomplete_leaf':
        leaf['terminal_flags']['completed'] = False
    elif fault == 'wrong_canonical_root':
        row['terminal_train_root'] = row['ancestor_roots'][0] if row['ancestor_roots'] else '/wrong/ancestor'
    elif fault == 'noncanonical_plan_leaf':
        original['canonical_equals_production_plan'] = False
    elif fault == 'wrong_index':
        row['index']['sha256'] = '0' * 64
    elif fault == 'wrong_contract':
        row['contract']['sha256'] = '0' * 64
    elif fault == 'BDDL_mismatch':
        row['BDDL']['matches'] = False
    elif fault == 'unresolved_competing_leaf':
        metadata['unresolved_competing_leaves'] = [{'root': '/unresolved/leaf'}]
    elif fault == 'post_outcome_inventory':
        metadata['new_assessment_outcomes_seen'] = True
    elif fault == 'wrong_inventory_binding':
        table['inventory_sha256'] = '0' * 64
    else:
        raise AssertionError('Unknown profile fault')
    with pytest.raises(ValueError):
        assessment.validate_assets(table, metadata)
