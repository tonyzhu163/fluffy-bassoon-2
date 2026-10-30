"""Frozen-manifest delegation and the separately funded interim fitting clock."""
import json
from pathlib import Path
import sys
import types

import pytest

from fluffy_bassoon.meta_rl import vla_source40_topup_interim_fit as fit
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish


def manifest(tmp_path):
    path = tmp_path / 'cpu-built-inputs.json'
    path.write_text('{"immutable": true}\n')
    return path, fit.digest(path)


def baseline_command(output, *, resume=False):
    value = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.vla_source40_subset_training', '--all40-source',
        '--train-index', '/snapshot/train-0/index.json', '/snapshot/train-39/index.json',
        '--validation-index', '/snapshot/validation-0/index.json',
        '--assembly', '/cpu-built/assembly.json', '--assembly-file-sha256', 'b' * 64,
        '--output', str(output), '--source-commit', 'a' * 40, '--expected-hostname', 'allocated',
        '--scales', '25', '100', '500', '--seed', '0',
        '--max-controller-seconds', '600', '--max-total-seconds', '2700']
    if resume:
        value.append('--resume')
    return value


def install_admission(monkeypatch, callback):
    module = types.ModuleType('fluffy_bassoon.meta_rl.vla_source40_topup_assembly')
    module.milestone_command = callback
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_interim_delegates_original_admission_and_preserves_frozen_inputs(tmp_path, monkeypatch):
    path, sha = manifest(tmp_path)
    output = tmp_path / 'experiment'
    calls = []
    original = baseline_command(output)
    def admission(given_path, given_sha, **kwargs):
        calls.append((given_path, given_sha, kwargs))
        return list(original)
    install_admission(monkeypatch, admission)
    command = fit.fitting_command(path, sha, output=output, source_commit='a' * 40, hostname='allocated')
    assert calls == [(path, sha, dict(output=output, source_commit='a' * 40, hostname='allocated', resume=False))]
    assert original[original.index('--scales') + 1:original.index('--seed')] == ['25', '100', '500']
    assert command[command.index('--scales') + 1:command.index('--seed')] == ['25', '100']
    assert command[command.index('--max-controller-seconds') + 1] == '300'
    assert command[command.index('--max-total-seconds') + 1] == '1500'
    assert command[:command.index('--scales')] == original[:original.index('--scales')]
    assert command[command.index('--seed') + 1] == '0'
    for name, value in fit.OPTIMIZATION.items():
        assert command[command.index('--' + name.replace('_', '-')) + 1] == str(value)
    assert '--resume' not in command and '--all40-source' in command


def test_interim_never_bypasses_existing_admission_or_changed_defaults(tmp_path, monkeypatch):
    path, sha = manifest(tmp_path)
    def rejected(*args, **kwargs):
        raise ValueError('Source admission rejected')
    install_admission(monkeypatch, rejected)
    with pytest.raises(ValueError, match='admission rejected'):
        fit.fitting_command(path, sha, output=tmp_path, source_commit='a' * 40, hostname='allocated')
    command = baseline_command(tmp_path, resume=True)
    command[command.index('--scales') + 1] = '10'
    install_admission(monkeypatch, lambda *args, **kwargs: command)
    with pytest.raises(ValueError, match='default milestones changed'):
        fit.fitting_command(path, sha, output=tmp_path, source_commit='a' * 40, hostname='allocated', resume=True)


def test_retry_keeps_original_1800_seconds_and_charges_setup_and_failures(tmp_path):
    path, sha = manifest(tmp_path)
    root = tmp_path / 'fresh-worker'
    first = fit.begin_profile(root, path, sha, 'a' * 40, 'fresh-worker', started=1000.)
    assert first['slice_seconds'] == 1740
    closed = finish(root, first['attempt'], 1, now=1100.)
    assert closed['charged_seconds'] == 160  #100sactual setup/failure +60allowance.
    second = fit.begin_profile(root, path, sha, 'a' * 40, 'fresh-worker', resume=True, started=1200.)
    assert second['slice_seconds'] == 1580
    saved = json.loads((root / 'worker-time.json').read_text())
    assert saved['worker_seconds'] == 1800
    assert saved['attempts'][0]['charged_seconds'] + saved['attempts'][1]['reserved_seconds'] == 1800
    assert saved['contract']['scales'] == [25, 100]
    assert saved['contract']['max_total_seconds'] == 1500
    assert json.loads((root / 'profile.json').read_text()) == saved['contract']
    with pytest.raises(RuntimeError, match='Insufficient original worker time'):
        fit.begin_profile(root, path, sha, 'a' * 40, 'fresh-worker', resume=True, started=1201.)


def test_changed_data_code_or_500_phase_cannot_reuse_interim_worker(tmp_path):
    path, sha = manifest(tmp_path)
    root = tmp_path / 'worker'
    first = fit.begin_profile(root, path, sha, 'a' * 40, 'worker', started=1000.)
    finish(root, first['attempt'], 1, now=1001.)
    with pytest.raises(ValueError, match='fresh immutable'):
        fit.begin_profile(root, path, sha, 'a' * 40, 'worker')
    with pytest.raises(ValueError, match='exact interim profile'):
        fit.begin_profile(root, path, sha, 'b' * 40, 'worker', resume=True)
    path.write_text('{"immutable": "changed dataset"}\n')
    with pytest.raises(ValueError, match='checksum differs'):
        fit.begin_profile(root, path, sha, 'a' * 40, 'worker', resume=True)
    with pytest.raises(ValueError, match='exact interim profile'):
        fit.begin_profile(root, path, fit.digest(path), 'a' * 40, 'worker', resume=True)
    with pytest.raises(ValueError, match='without the original'):
        fit.begin_profile(tmp_path / 'missing', path, fit.digest(path), 'a' * 40, 'worker', resume=True)


def completed_fixture(tmp_path):
    scales = []
    for update in fit.SCALES:
        rows = []
        folder = tmp_path / f'update-{update}'
        folder.mkdir()
        for objective in ('ad', 'dpt'):
            for mode in ('full', 'static'):
                row = dict(objective=objective, mode=mode, seed=0, updates=update)
                for stem in ('final', 'full_recovery'):
                    path = folder / f'{objective}-{mode}-{stem}.pt'
                    path.write_bytes(f'{objective}:{mode}:{update}:{stem}'.encode())
                    row[stem + '_path'], row[stem + '_sha256'] = str(path), fit.digest(path)
                rows.append(row)
        index = folder / 'index.json'
        index.write_text(json.dumps(dict(completed=True, controllers=rows)))
        scales.append(dict(updates=update, index_path=str(index), index_sha256=fit.digest(index)))
    value = dict(completed=True, qualification_subset=False, unique_controller_optimizer_updates=400,
        scales=scales, assembly_sha256='c' * 64, source_collection_costs_count_once=dict(actual_actions=457851))
    (tmp_path / 'scales.json').write_text(json.dumps(value))
    return value


def test_completion_requires_all_four_exact_models_both_scales_and_full_recovery_bytes(tmp_path):
    value = completed_fixture(tmp_path)
    result = fit.verify_completed(tmp_path)
    assert len(result['controllers']) == 8 and result['milestones'] == [25, 100]
    assert result['completed'] is True and result['full_phase_complete'] is False
    assert result['additional_simulator_actions'] == 0
    assert result['source_collection_costs_count_once'] == value['source_collection_costs_count_once']
    Path(result['controllers'][0]['full_recovery_path']).write_bytes(b'changed full recovery')
    with pytest.raises(ValueError, match='recovery bytes changed'):
        fit.verify_completed(tmp_path)


def test_completion_rejects_partial_point_or_silently_extended_milestones(tmp_path):
    value = completed_fixture(tmp_path)
    value['scales'].append(dict(updates=500))
    (tmp_path / 'scales.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='incomplete'):
        fit.verify_completed(tmp_path)
