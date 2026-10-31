"""Derived-label admission checks; run only on allocated Linux.

The writer has separate actor-inference/provenance tests. Here its verified
loader is a metadata test double so the assembly boundary can be attacked
without constructing any SAC, Transformer or VLA model.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_topup_assembly as topup
from fluffy_bassoon.meta_rl import vla_source40_training as training
from fluffy_bassoon.meta_rl.vla_history import FIELDS, file_sha256, load_dataset, save_dataset
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
from test_vla_source40_topup_assembly import captured


def write(path, value):
    Path(path).write_text(json.dumps(value))


def array_sha(value):
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
        + np.ascontiguousarray(value).tobytes()).hexdigest()


def binding_fixture(tmp_path, monkeypatch):
    _, _, admission, admission_path, descriptor = captured(tmp_path, monkeypatch)
    selected = admission['source_indices']
    train = {row['slot'].split('/')[0]: row for row in selected['train']}
    rows = []
    for number, validation in enumerate(selected['validation']):
        task = validation['slot'].split('/')[0]
        train_index = topup.read(train[task]['index_path'])
        publication = train_index['publications'][0]
        val_index = topup.read(validation['index_path'])
        original = val_index['publications'][0]['datasets']['dpt']
        original_path = Path(validation['index_path']).parent / original['path']
        old, manifest = load_dataset(original_path)
        teacher = dict(checkpoint_path=str(Path(train[task]['index_path']).parent / publication['source_checkpoint']),
            checkpoint_sha256=publication['source_checkpoint_sha256'], pure_actor_sha256=f'{number + 1:064x}',
            train_dpt_sha256=publication['datasets']['dpt']['sha256'], source_updates=train_index['source_updates'])
        derived = copy.deepcopy(old[0])
        derived['target_noise'].fill(-.125 - number / 100)
        path = tmp_path / 'derived' / task.replace(':', '-') / 'dpt.npz'
        metadata = copy.deepcopy(manifest['metadata'])
        metadata.update(metric_name=training.DPT_SAME_TEACHER_METRIC, label_teacher=teacher)
        saved = save_dataset(path, [derived], objective='dpt', metadata=metadata)
        rows.append(dict(task_key=task, train_index_path=train[task]['index_path'],
            train_index_sha256=train[task]['index_sha256'], validation_index_path=validation['index_path'],
            validation_index_sha256=validation['index_sha256'], label_teacher=teacher,
            original_validation_dpt=dict(path=str(original_path), sha256=original['sha256']),
            derived_validation_dpt=dict(path=str(path), sha256=saved['sha256'],
                sidecar_path=str(path.with_suffix('.json')), sidecar_sha256=file_sha256(path.with_suffix('.json'))),
            non_target_field_sha256={key: array_sha(old[0][key]) for key in FIELDS if key != 'target_noise'},
            actor_params_unchanged=True, train_label_replay=dict(passed=True, max_absolute_delta=0., tolerance=1e-5)))
    value = dict(format='source40_dpt_same_train_teacher_labels_v1', completed=True,
        metric_name=training.DPT_SAME_TEACHER_METRIC, admission_path=str(admission_path),
        admission_sha256=file_sha256(admission_path), source_action_charge=descriptor['cumulative_source_action_charge'],
        rows=rows)
    path = tmp_path / 'binding.json'
    write(path, value)

    def verified_metadata(path, expected_sha256, *, admission_path, admission_sha256):
        assert file_sha256(path) == expected_sha256
        result = topup.read(path)
        assert result['admission_path'] == str(admission_path)
        assert result['admission_sha256'] == admission_sha256
        return result

    monkeypatch.setitem(sys.modules, 'fluffy_bassoon.meta_rl.vla_source40_dpt_same_teacher',
        SimpleNamespace(load_binding=verified_metadata,
            non_target_hashes=lambda trial: {key: array_sha(trial[key]) for key in FIELDS if key != 'target_noise'}))
    return admission_path, path, value


def assemble(admission, binding, output):
    return topup.assemble_topup(admission, file_sha256(admission), output,
        dpt_validation_binding_path=binding, dpt_validation_binding_sha256=file_sha256(binding))


def test_only_eight_val_dpt_label_arrays_change_and_full_cost_is_counted_once(tmp_path, monkeypatch):
    admission, binding, value = binding_fixture(tmp_path, monkeypatch)
    legacy_root, derived_root = tmp_path / 'legacy', tmp_path / 'same-teacher'
    original_files = {str(path): file_sha256(path) for path in (tmp_path / 'snapshot').rglob('*') if path.is_file()}
    legacy = topup.assemble_topup(admission, file_sha256(admission), legacy_root)
    derived = assemble(admission, binding, derived_root)
    old, new = (topup.read(root / 'assembly.json') for root in (legacy_root, derived_root))
    assert new['selected_sources'] == old['selected_sources']
    assert new['collection_costs_count_once'] == old['collection_costs_count_once']
    assert derived['cumulative_source_action_charge'] == legacy['cumulative_source_action_charge'] == value['source_action_charge']
    assert derived['additional_simulator_actions'] == 0
    assert 'dpt_validation_labels' not in legacy and 'dpt_validation_labels' not in old
    assert legacy['dpt_validation_metric'] == training.DPT_LEGACY_METRIC
    assert derived['dpt_validation_metric'] == training.DPT_SAME_TEACHER_METRIC
    assert len(new['source_task_keys']) == 40 and len(new['validation_task_keys']) == 8
    roles = new['dpt_validation_labels']['label_roles']
    assert len(roles) == 8 and all(row['train'] == row['validation'] for row in roles)
    for split in ('train', 'validation'):
        for objective in ('ad', 'dpt'):
            before, before_metadata = load_dataset(legacy_root / f'{split}-{objective}.npz')
            after, after_metadata = load_dataset(derived_root / f'{split}-{objective}.npz')
            changed_role = split == 'validation' and objective == 'dpt'
            for first, second in zip(before, after, strict=True):
                for key in FIELDS:
                    if changed_role and key == 'target_noise':
                        assert not np.array_equal(first[key], second[key])
                    else:
                        assert training._same_array(first[key], second[key])
            if changed_role:
                assert after_metadata['metadata']['metric_name'] == training.DPT_SAME_TEACHER_METRIC
                for key, item in before_metadata['metadata'].items():
                    assert after_metadata['metadata'][key] == item
            else:
                assert before_metadata['metadata'] == after_metadata['metadata']
    assert original_files == {str(path): file_sha256(path) for path in (tmp_path / 'snapshot').rglob('*') if path.is_file()}
    assert training.load_assembly(derived_root / 'assembly.json') == new
    manifest = derived_root / 'topup-inputs.json'
    command = topup.milestone_command(manifest, file_sha256(manifest), output=tmp_path / 'fit',
        source_commit='d' * 40, hostname='job-test')
    assert command[command.index('--scales') + 1:command.index('--seed')] == ['25', '100', '500']
    assert command[command.index('--assembly') + 1] == str(derived_root / 'assembly.json')


@pytest.mark.parametrize('change', [
    lambda value: value['rows'].pop(),
    lambda value: value['rows'].reverse(),
    lambda value: value.update(source_action_charge=value['source_action_charge'] - 1),
    lambda value: value['rows'][0]['label_teacher'].update(checkpoint_sha256='0' * 64),
    lambda value: value['rows'][0]['label_teacher'].update(train_dpt_sha256='0' * 64),
    lambda value: value['rows'][0].update(validation_index_path=value['rows'][1]['validation_index_path']),
])
def test_missing_reordered_rebound_or_discounted_validation_labels_rejected(tmp_path, monkeypatch, change):
    admission, binding, value = binding_fixture(tmp_path, monkeypatch)
    change(value)
    write(binding, value)
    with pytest.raises(ValueError, match='Derived DPT'):
        assemble(admission, binding, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('change', ['observation', 'teacher_steps', 'source_contract'])
def test_derived_labels_cannot_modify_behavior_or_rename_the_behavior_teacher(tmp_path, monkeypatch, change):
    admission, binding, value = binding_fixture(tmp_path, monkeypatch)
    row = value['rows'][0]
    old_path = Path(row['derived_validation_dpt']['path'])
    trials, manifest = load_dataset(old_path)
    if change == 'observation':
        trials[0]['observations'][0, 0] += .1
    elif change == 'teacher_steps':
        trials[0]['teacher_steps'] += 1
    else:
        manifest['metadata']['source_contract']['history_id'] += '/incorrect-label-role'
    path = old_path.parent / 'modified.npz'
    saved = save_dataset(path, trials, objective='dpt', metadata=manifest['metadata'])
    row['derived_validation_dpt'] = dict(path=str(path), sha256=saved['sha256'],
        sidecar_path=str(path.with_suffix('.json')), sidecar_sha256=file_sha256(path.with_suffix('.json')))
    write(binding, value)
    with pytest.raises(ValueError, match='behavior arrays'):
        assemble(admission, binding, tmp_path / 'invalid')


def test_reloading_assembly_rechecks_actual_labels_not_only_the_metric_name(tmp_path, monkeypatch):
    admission, binding, _ = binding_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'same-teacher'
    assemble(admission, binding, output)
    path = output / 'validation-dpt.npz'
    trials, manifest = load_dataset(path)
    trials[0]['target_noise'] += .01
    different = output / 'different-labels.npz'
    saved = save_dataset(different, trials, objective='dpt', metadata=manifest['metadata'])
    assembly = topup.read(output / 'assembly.json')
    assembly['datasets']['validation']['dpt'].update(path=different.name, sha256=saved['sha256'])
    assembly['assembly_sha256'] = content_hash({key: item for key, item in assembly.items() if key != 'assembly_sha256'})
    write(output / 'assembly.json', assembly)
    with pytest.raises(ValueError, match='explicitly bound derived labels'):
        training.load_assembly(output / 'assembly.json')


def test_legacy_sidecar_cannot_claim_same_teacher_metric_without_explicit_binding(tmp_path, monkeypatch):
    admission, _, _ = binding_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'legacy'
    topup.assemble_topup(admission, file_sha256(admission), output)
    sidecar = output / 'validation-dpt.json'
    value = topup.read(sidecar)
    value['metadata']['metric_name'] = training.DPT_SAME_TEACHER_METRIC
    write(sidecar, value)
    with pytest.raises(ValueError, match='explicit derived validation label admission'):
        training.load_assembly(output / 'assembly.json')


def test_sidecar_and_training_manifest_cannot_drop_the_binding(tmp_path, monkeypatch):
    admission, binding, value = binding_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'same-teacher'
    assemble(admission, binding, output)
    path = output / 'topup-inputs.json'
    manifest = topup.read(path)
    manifest.pop('dpt_validation_labels')
    manifest['content_sha256'] = content_hash({key: item for key, item in manifest.items() if key != 'content_sha256'})
    write(path, manifest)
    with pytest.raises(ValueError, match='explicit DPT validation label binding'):
        topup.milestone_command(path, file_sha256(path), output=tmp_path / 'fit', source_commit='d' * 40, hostname='job-test')
    sidecar = Path(value['rows'][0]['derived_validation_dpt']['sidecar_path'])
    sidecar.write_text(sidecar.read_text() + '\n')  # Same JSON values, different immutable bytes.
    with pytest.raises(ValueError, match='sidecar requires'):
        assemble(admission, binding, tmp_path / 'changed-sidecar')


def test_opt_in_cannot_omit_binding_hash_or_use_pilot_admission(tmp_path, monkeypatch):
    admission, binding, _ = binding_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='explicit complete-prefix'):
        topup.assemble_topup(admission, file_sha256(admission), tmp_path / 'no-hash',
            dpt_validation_binding_path=binding)
    with pytest.raises(ValueError, match='explicit complete-prefix'):
        training._admit_dpt_validation_binding(binding, file_sha256(binding), None, {})
