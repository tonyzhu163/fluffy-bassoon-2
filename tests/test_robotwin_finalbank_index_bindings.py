"""Deferred metadata-only final100 index guards; NOT part of SAC65 gate.

Prepared but unexecuted. Run only in a separately admitted Linux CPU check.
No source100 runner, model, labels or fitting is invoked by these fixtures.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'outputs/robotwin-finalbank-offline-plan-20261031-r1'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

project = load('deferred_project', PLAN / 'project_corpus.py')
runner = load('deferred_runner', PLAN / 'run_offline.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path.resolve()), sha256=sha(path))

def indexes_fixture(tmp_path):
    """Metadata-only100 guard fixture; never passed to trainer/fit/main."""
    corpus = tmp_path / 'projection'; corpus.mkdir()
    source = tmp_path / 'source'; source.write_bytes(b'fixture-source100-reference-only')
    source_ref = dict(path=str(source), sha256=sha(source))
    contract = dict(source_commit=project.LINEAGE,
                    source_iterations=dict(count=100, trials_per_iteration=6, episodes_per_trial=4))
    write(corpus / 'contract.json', contract); write(corpus / 'runtime-contract.json', dict(norm='fixed'))
    tip_ref = write(tmp_path / 'tip.json', dict(method='sac', kind='whole_trial_boundary',
        expected=dict(completed_source_iterations=100, completed_source_trials=600, source_steps=2400)))
    ancestry_ref = write(tmp_path / 'ancestry.json', {})
    projection = dict(tip_manifest=tip_ref, source_checkpoint=source_ref, ancestry=ancestry_ref,
                      roots=[], files={})
    bindings = dict(projection_root=str(corpus), exports_root=str(tmp_path / 'exports'),
                    source100_stage=source_ref, teacher100_source=source_ref)
    for partition, count, objectives in [('train', 600, ['ad', 'dpt']),
                                         ('validation', 6, ['behavior_cloning', 'dpt'])]:
        output = Path(bindings['exports_root']) / partition; output.mkdir(parents=True)
        trials = [dict(number=n + 1, task='task' + str(n % 6), decisions=4,
            cases=[dict(episode_index=e, actual_seed=n * 4 + e + (3000000000 if partition == 'validation' else 0))
                   for e in range(4)]) for n in range(count)]
        datasets = {}
        for objective in objectives:
            path = output / (objective + '.npz'); path.write_bytes(objective.encode())
            datasets[objective] = dict(path=str(path), sha256=sha(path), objective=objective)
        metadata = dict(format='robotwin_native_history_dataset_v1', completed=True, partition=partition,
            source_method='sac', teacher_method='sac', source_recovery_sha256=source_ref['sha256'],
            teacher_recovery_sha256=source_ref['sha256'], policy_input_dim=3090,
            validation_iteration=100 if partition == 'validation' else None,
            datasets=datasets, counts=dict(complete_K4_trials=count, episodes=count * 4,
                committed_source_actions=2400, selected_actions=count * 4, decisions=count * 4),
            source_trials=trials, case_ids=[partition + str(n) for n in range(count)],
            additional_simulator_actions=0, additional_vla_queries=0, source_contract=contract,
            interface=dict(native=3090), input_sha256={str(source): source_ref['sha256'],
                str(corpus / 'contract.json'): sha(corpus / 'contract.json'),
                str(corpus / 'runtime-contract.json'): sha(corpus / 'runtime-contract.json')})
        bindings[partition + '_index'] = write(output / 'index.json', metadata)
    return bindings, projection


@pytest.mark.parametrize('kind', ['valid', 'old_source', 'old_teacher', 'source65', 'scalar_schema',
                                 'wrong_objectives', 'wrong_path', 'foreign_projection', 'overlap'])
def test_final_index_guard_binds_actual_schema_source_teacher_and_cases(tmp_path, kind):
    b, projection = indexes_fixture(tmp_path)
    path = Path(b['train_index']['path']); m = json.loads(path.read_text())
    if kind == 'old_source':
        m['source_recovery_sha256'] = '5' * 64
    elif kind == 'old_teacher':
        m['teacher_recovery_sha256'] = '6' * 64
    elif kind == 'source65':
        m['counts']['complete_K4_trials'] = 390
    elif kind == 'scalar_schema':
        m['source_contract']['source_iterations'] = 100
    elif kind == 'wrong_objectives':
        m['datasets'].pop('ad')
    elif kind == 'wrong_path':
        b['train_index']['path'] = str(tmp_path / 'old-bank-index.json')
        write(Path(b['train_index']['path']), m)
    elif kind == 'foreign_projection':
        m['input_sha256'][str(Path(b['projection_root']) / 'contract.json')] = '0' * 64
    elif kind == 'overlap':
        val = Path(b['validation_index']['path']); v = json.loads(val.read_text())
        v['source_trials'][0]['cases'][0]['actual_seed'] = 0; write(val, v)
    write(path, m)
    if kind == 'valid':
        runner.verify_final_indexes(b, projection)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            runner.verify_final_indexes(b, projection)
