"""Milestone chronology and immutable full recoveries for subset qualification."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, file_sha256, load_policy
from fluffy_bassoon.meta_rl.vla_history_train import RESUME_FORMAT, _rng_state
from fluffy_bassoon.meta_rl.vla_source40_subset_training import (
    admit_assembly, archive_scale, immutable_copy, parser, training_command, validate_scales)


def arguments(tmp_path):
    return parser().parse_args(['--qualification-subset', '--train-index', '/source/train/index.json',
        '--validation-index', '/source/val/index.json', '--output', str(tmp_path),
        '--source-commit', 'a' * 40, '--expected-hostname', 'allocated-host'])


def assembly():
    return dict(datasets={partition: {objective: dict(path=partition + '-' + objective + '.npz', sha256='f' * 64)
        for objective in ('ad', 'dpt')} for partition in ('train', 'validation')})


def test_subset_is_explicit_and_scale_axis_is_update_count(tmp_path):
    args = arguments(tmp_path)
    assert args.qualification_subset and args.scales == [25, 100, 500]
    assert validate_scales(args.scales) == [25, 100, 500]
    for values in ([], [25, 25, 500], [100, 25], [501], [True]):
        with pytest.raises(ValueError, match='optimizer milestones'):
            validate_scales(values)
    with pytest.raises(SystemExit):
        parser().parse_args(['--train-index', '/train', '--validation-index', '/val',
            '--output', str(tmp_path), '--source-commit', 'a' * 40, '--expected-hostname', 'allocated-host'])


def test_milestones_continue_one_optimizer_identity_instead_of_retraining(tmp_path):
    args = arguments(tmp_path)
    data = assembly()
    commands = [training_command(args, data, objective='ad', mode='full', updates=scale, resume=i > 0)
                for i, scale in enumerate(args.scales)]
    outputs = [command[command.index('--output-dir') + 1] for command in commands]
    assert len(set(outputs)) == 1
    assert [int(command[command.index('--updates') + 1]) for command in commands] == [25, 100, 500]
    assert '--resume' not in commands[0]
    assert all(command[command.index('--resume') + 1] == str(Path(outputs[0]) / 'latest.pt') for command in commands[1:])
    assert sum([25, 100 - 25, 500 - 100]) == 500
    for command in commands:
        assert command[command.index('--max-elapsed-seconds') + 1] == '600.0'
        assert command[command.index('--seed') + 1] == '0'


def test_full_static_and_ad_dpt_receive_matching_optimization_parameters(tmp_path):
    args = arguments(tmp_path)
    fields = ('--updates', '--batch-size', '--learning-rate', '--width', '--layers', '--heads',
              '--context-length', '--max-window', '--eval-every', '--validation-batches')
    commands = [training_command(args, assembly(), objective=objective, mode=mode, updates=500, resume=True)
                for objective in ('ad', 'dpt') for mode in ('full', 'static')]
    assert len({tuple(command[command.index(field) + 1] for field in fields) for command in commands}) == 1
    assert {command[command.index('--mode') + 1] for command in commands} == {'full', 'static'}


def test_optional_data_scale_uses_hash_bound_assembly_directory(tmp_path):
    args = arguments(tmp_path / 'students')
    args.assembly = tmp_path / 'case16/assembly.json'
    command = training_command(args, assembly(), objective='ad', mode='full', updates=500, resume=False)
    assert command[command.index('--train-data') + 1] == str(tmp_path / 'case16/train-ad.npz')
    assert command[command.index('--validation-data') + 1] == str(tmp_path / 'case16/validation-ad.npz')


def test_all40_route_requires_a_strict_campaign_assembly_and_cannot_admit_pilot_data(tmp_path, monkeypatch):
    args = parser().parse_args(['--all40-source', '--train-index', '/train', '--validation-index', '/validation',
        '--output', str(tmp_path / 'students'), '--source-commit', 'a' * 40, '--expected-hostname', 'allocated-host'])
    with pytest.raises(ValueError, match='independently assembled'):
        admit_assembly(args)
    args.assembly = tmp_path / 'assembly.json'
    args.assembly.write_text('{}')
    args.assembly_file_sha256 = file_sha256(args.assembly)
    calls = []
    def loader(path, *, allow_small_pilot):
        calls.append(allow_small_pilot)
        return dict(small_pilot=True)
    monkeypatch.setattr('fluffy_bassoon.meta_rl.vla_source40_subset_training.load_assembly', loader)
    with pytest.raises(ValueError, match='scope differs'):
        admit_assembly(args)
    assert calls == [False]


def test_immutable_copy_rejects_changed_milestone_bytes(tmp_path):
    source, destination = tmp_path / 'live.pt', tmp_path / 'scale/recovery.pt'
    source.write_bytes(b'complete saved optimizer and RNG bytes')
    assert immutable_copy(source, destination) == file_sha256(source)
    source.write_bytes(b'later update')
    with pytest.raises(ValueError, match='immutable milestone differs'):
        immutable_copy(source, destination)
    assert destination.read_bytes() == b'complete saved optimizer and RNG bytes'


@pytest.mark.parametrize('small_pilot', [True, False])
def test_policy_export_comes_from_exact_full_milestone_recovery(tmp_path, small_pilot):
    config = HistoryConfig(2, 2, width=8, layers=1, heads=2, context_length=4)
    policy = HistoryNoisePolicy(config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    sum(value.square().sum() for value in policy.parameters()).backward()
    optimizer.step()
    contract = dict(config=asdict(config), objective='ad', mode='full', seed=0)
    saved = dict(format=RESUME_FORMAT, update=25, policy=policy.state_dict(), optimizer=optimizer.state_dict(),
        rng=_rng_state(np.random.default_rng(0)), contract=contract, best_validation_nll=1., elapsed_seconds=3.)
    source = tmp_path / 'latest.pt'
    torch.save(saved, source)
    exposure = dict(training_task_keys=['libero_spatial:00'], small_pilot=small_pilot)
    row = archive_scale(source, tmp_path / 'scale25', updates=25, assembly_sha256='a' * 64, exposure=exposure)
    recovered = torch.load(row['full_recovery_path'], map_location='cpu', weights_only=False)
    assert recovered['update'] == 25 and recovered['optimizer']['state']
    assert set(recovered['rng']) == {'sampler', 'python', 'numpy', 'torch', 'cuda'}
    assert row['full_recovery_sha256'] == file_sha256(source)
    candidate, manifest = load_policy(row['final_path'], expected_sha256=row['final_sha256'])
    assert manifest['selection'] == 'fixed requested final update'
    assert manifest['qualification_subset'] is small_pilot and manifest['source_exposure'] == exposure
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in policy.state_dict().items())
    assert archive_scale(source, tmp_path / 'scale25', updates=25,
        assembly_sha256='a' * 64, exposure=exposure) == row
    with pytest.raises(ValueError, match='exact update'):
        archive_scale(source, tmp_path / 'wrong-scale', updates=100, assembly_sha256='a' * 64, exposure=exposure)
