"""One explicit post-SGD execution migration for the diagnosed b665 guard.

The scientific contracts, selected task, raw data and cumulative worker clock
stay in their original locations. This is not a general code-source alias.
"""
import json
from pathlib import Path
import re

LEGACY_SOURCE = 'b6655659d8bff2c6b5d14eaf05d63f4647176d4f'
CHECKPOINT_SHA = '4413afdfb69baff3761dba6a630092f217ee9d1b20832991e2a3eb4e873c1150'
CURSOR_SHA = '633e88838b7fda8ccdf3a1f401dbdbbc71b12f771505310705c829116c3d95c3'
CONTRACT_SHA = '96d90d35de2ed50dc4a103f7ddec8781031c2cede74402825f43a72d0b6114e9'


def read_bound(ref):
    from .vla_libero import sha256
    path = Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or sha256(path) != ref['sha256']:
        raise ValueError('Execution migration requires exact immutable input bytes')
    return path


def validate_migration(root, runtime_source, path, expected_sha, *, stage, screen_source):
    """Validate the unchanged old prefix before begin, or its owned successor."""
    from .vla_libero import sha256
    from .vla_source40_schedule import content_hash
    root = Path(root).resolve()
    path = read_bound(dict(path=str(path), sha256=expected_sha))
    value = json.loads(path.read_text())
    if (value.get('format') != 'pi05_headroom_exact_restore_execution_migration_v1'
            or value.get('admitted') is not True or stage not in ('begin', 'run')
            or value.get('legacy_execution_source') != LEGACY_SOURCE
            or value.get('scientific_trainer_source') != LEGACY_SOURCE
            or value.get('execution_source') != runtime_source or runtime_source == LEGACY_SOURCE
            or not re.fullmatch('[0-9a-f]{40}', runtime_source)
            or Path(value['phase_root']).resolve() != root
            or value.get('allowed_change') != 'Exact restore identity; MsgPack mapping order only'
            or value.get('scientific_contract_unchanged') is not True):
        raise ValueError('Only the diagnosed b665 execution-only migration is admitted')
    refs = {key: read_bound(value[key]) for key in (
        'checkpoint', 'cursor', 'learning_contract', 'diagnosis', 'prior_worker_clock',
        'screen_plan', 'selection', 'screen_admission')}
    if (value['checkpoint']['sha256'] != CHECKPOINT_SHA or value['cursor']['sha256'] != CURSOR_SHA
            or value['learning_contract']['sha256'] != CONTRACT_SHA):
        raise ValueError('Migration cannot replace the closed301700 recovery/cursor/contract')
    prior = json.loads(refs['prior_worker_clock'].read_text())
    current = json.loads((root / 'worker-time.json').read_text())
    cursor = json.loads(refs['cursor'].read_text())
    diagnosis = json.loads(refs['diagnosis'].read_text())
    science = json.loads(refs['learning_contract'].read_text())
    admission = json.loads(refs['screen_admission'].read_text())
    selection = json.loads(refs['selection'].read_text())
    original_paths = dict(screen_plan=root/'experiment/plan.json', selection=root/'experiment/frozen-selection.json',
        screen_admission=root/'screen-to-learning-source.json',
        learning_contract=root/'experiment/case-learn-44/contract.json')
    if any(sha256(original_paths[key]) != value[key]['sha256'] for key in original_paths):
        raise ValueError('Original phase science/selection/admission bytes changed')
    if (science['source_commit'] != LEGACY_SOURCE or science['selected_task'] != 44
            or science['frozen_selection_sha256'] != value['selection']['sha256']
            or admission['learning_source'] != LEGACY_SOURCE or admission['screen_source'] != screen_source
            or selection['task_id'] != 44 or selection['optimizer_updates'] != 0
            or not selection['selected_before_any_sgd']
            or selection['plan_sha256'] != value['screen_plan']['sha256']
            or prior['worker_seconds'] != 57600 or current['worker_seconds'] != 57600
            or prior['contract']['source_commit'] != screen_source
            or current['contract'] != prior['contract']
            or content_hash(current['contract']) != value['original_worker_contract_sha256']
            or current['attempts'][:len(prior['attempts'])] != prior['attempts']
            or any(row['status'] != 'finished' for row in prior['attempts'])
            or sum(row['charged_seconds'] for row in prior['attempts']) != 28217
            or cursor['model']['sha256'] != CHECKPOINT_SHA or cursor['model']['updates'] != 301700
            or cursor['model']['replay_rows'] != 15579 or cursor['training']['episodes'] != 1720
            or diagnosis.get('classification') != 'identical_saved_state_raw_representation_differs'
            or diagnosis.get('qualified_learner_source') != LEGACY_SOURCE
            or diagnosis.get('legacy_saved_signature_matches') is not True
            or diagnosis.get('canonical_full_state_exact') is not True
            or diagnosis.get('exact_difference_count') != 0
            or diagnosis.get('all_inputs_unchanged') is not True
            or set(diagnosis['RNG']) != {'agent_exact', 'sampler_exact', 'numpy_global_exact',
                'python_global_exact', 'motor_key_roundtrip_exact'}
            or not all(diagnosis['RNG'].values()) or diagnosis['replay'].get('exact') is not True
            or diagnosis.get('legacy_cursor_signature') != cursor['model']['signature']
            or diagnosis.get('saved_agent_bytes_SHA') != cursor['model']['signature']
            or diagnosis.get('canonical_expected_SHA') != diagnosis.get('canonical_actual_SHA')
            or diagnosis.get('new_SGD') != 0 or diagnosis.get('new_robot_actions') != 0
            or diagnosis['checkpoint']['sha256'] != CHECKPOINT_SHA or diagnosis['cursor']['sha256'] != CURSOR_SHA
            or diagnosis['original_guard_predicates'] != dict(training_equal=True, updates_equal=True,
                replay_rows_equal=True, raw_restored_signature_equal=False)):
        raise ValueError('Exact original scientific, clock and diagnosed state ancestry required')
    successors = current['attempts'][len(prior['attempts']):]
    if stage == 'run' and not successors:
        raise ValueError('Execution migration run requires its existing clock reservation')
    for index, row in enumerate(successors):
        metadata = json.loads((Path(row['path'])/'phase-stage.json').read_text())
        if (metadata.get('execution_source') != runtime_source or metadata.get('scientific_source') != LEGACY_SOURCE
                or metadata.get('execution_migration_sha256') != expected_sha or metadata.get('stage') != 'learn'
                or row['status'] != ('started' if stage == 'run' and index == len(successors)-1 else 'finished')):
            raise ValueError('Only owned same-migration cumulative attempts may continue')
    if not successors and sha256(root/'experiment/case-learn-44/state.json') != CURSOR_SHA:
        raise ValueError('First repaired execution must start from the original301700 cursor')
    return LEGACY_SOURCE
