"""Owned tiny CPU fixtures for the prepared source-bank compatibility gate.

Run only under the admitted gate; no simulator, production data mutation,
optimizer step or source100 execution. Current65 metadata fixtures exercise
source/index linkage; optional final100 guards live in a separate deferred file.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import signal
import subprocess
import sys
import time

import pytest
import torch
import numpy as np

from test_robotwin_history_corpus import fixture
from fluffy_bassoon.meta_rl.robotwin_history_corpus import (
    CorpusReader, export_corpus, read_source, source_recovery)

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'outputs/robotwin-finalbank-offline-plan-20261031-r1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


project = load('fixture_project_corpus', PLAN / 'project_corpus.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path.resolve()), sha256=sha(path))


def projection_fixture(tmp_path):
    """Real qualified miniature records, one K4 split across two roots."""
    original, arguments = fixture(tmp_path, 'sac')
    export_corpus(original, tmp_path / 'unsplit-baseline', **arguments)
    suffix = tmp_path / 'suffix/corpus'; suffix.mkdir(parents=True)
    for name in ('contract.json', 'runtime-contract.json'):
        shutil.copyfile(original / name, suffix / name)
    for name in ('frames', 'policies'):
        shutil.copytree(original / name, suffix / name)
    (suffix / 'episodes').mkdir()
    for episode in ('trial1-ep2', 'trial1-ep3', 'trial2-ep0', 'trial2-ep1', 'trial2-ep2', 'trial2-ep3'):
        shutil.move(str(original / 'episodes' / episode), suffix / 'episodes' / episode)
    roots = [original, suffix]
    checkpoint = dict(path=str(arguments['source_path']), sha256=arguments['source_sha256'])
    old_contract = dict(path=str(original / 'contract.json'), sha256=sha(original / 'contract.json'))
    old_manifest = dict(corpus_root=str(original), corpus_contract=old_contract,
        kind='selected_trial', method='sac', checkpoint=checkpoint,
        expected=dict(source_steps=64, completed_source_trials=0, completed_source_iterations=0))
    old_ref = write(tmp_path / 'old-manifest.json', old_manifest)
    producer = json.loads((suffix / 'contract.json').read_text())
    producer.update(execution_source_commit='e' * 40, resume_trial_manifest_sha256=old_ref['sha256'])
    new_contract = write(suffix / 'contract.json', producer)
    migration = dict(method='sac', lineage_source_commit='a' * 40, execution_source_commit='e' * 40,
                     manifest_path=old_ref['path'], manifest_sha256=old_ref['sha256'])
    migration_ref = write(suffix.parent / 'execution-migration.json', migration)
    root_info = [dict(root=str(r), contract_sha256=sha(r / 'contract.json'),
                      runtime_sha256=sha(r / 'runtime-contract.json')) for r in roots]
    tip = dict(kind='whole_trial_boundary', method='sac', lineage_source_commit='a' * 40,
        corpus_root=str(suffix), corpus_contract=new_contract, checkpoint=checkpoint,
        expected=dict(completed_source_iterations=1, completed_source_trials=2, source_steps=256,
                      pending_rollout_digests=[]))
    tip_ref = write(tmp_path / 'tip.json', tip)
    entries = []
    for root, info, manifest, manifest_ref, contract_ref, execution in (
            (original, root_info[0], old_manifest, old_ref, old_contract, None),
            (suffix, root_info[1], tip, tip_ref, new_contract, dict(ref=migration_ref, value=migration))):
        entries.append(dict(corpus_root=str(root), method='sac', manifest=manifest_ref,
            kind=manifest['kind'], expected=manifest['expected'], checkpoint=checkpoint,
            canonical_corpus_contract=contract_ref, bound_corpus_contract=contract_ref,
            corpus_contract=json.loads((root / 'contract.json').read_text()), execution_migration=execution))
    ancestry = dict(lineage_commit='a' * 40, root_count=2, ordered_corpus_roots=[str(r) for r in roots],
        tip_manifest=tip_ref, ordered_ancestry=entries)
    ancestry_ref = write(tmp_path / 'ancestry.json', ancestry)
    output = tmp_path / 'projection'; output.mkdir()
    files = {}
    for root in roots:
        for folder in ('episodes', 'frames', 'policies'):
            for path in (root / folder).rglob('*'):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(root)); destination = output / relative
                if relative not in files:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(path, destination)
                    files[relative] = dict(sha256=sha(path), origins=[])
                assert sha(destination) == sha(path)
                files[relative]['origins'].append(str(path))
    for name in ('contract.json', 'runtime-contract.json'):
        shutil.copyfile(original / name, output / name)
    receipt = dict(status='byte_projection_only_not_export_admission', purpose='reference',
        source_rounds=1, method='sac', lineage_commit='a' * 40, roots=root_info, files=files,
        ancestry=ancestry_ref, tip_manifest=tip_ref, source_checkpoint=tip['checkpoint'])
    write(output / 'projection.json', receipt)
    return output, roots, arguments, receipt


def test_original_single_root_still_uses_original_containment(tmp_path):
    root, args = fixture(tmp_path, 'sac')
    source = source_recovery(args['source_path'], args['source_sha256'], 'sac')
    reader, trials, _ = read_source(root, source)
    assert len(trials) == 2 and not reader.policy_aliases
    with pytest.raises(ValueError, match='escapes'):
        reader.path(tmp_path / 'not-in-corpus')


def test_split_prefix_absolute_archive_and_actual_masks_export_unchanged(tmp_path):
    output, roots, args, _ = projection_fixture(tmp_path)
    result = export_corpus(output, tmp_path / 'paired', **args)
    assert result['counts']['complete_K4_trials'] == 2
    assert result['counts']['selected_actions'] == 256
    assert len(list((roots[0] / 'episodes').iterdir())) == 2
    assert len(list((roots[1] / 'episodes').iterdir())) == 6
    assert all(sha(path) == digest for path, digest in result['input_sha256'].items())
    baseline = json.loads((tmp_path / 'unsplit-baseline/index.json').read_text())
    for objective in ('ad', 'dpt'):
        with np.load(baseline['datasets'][objective]['path'], allow_pickle=False) as before, \
                np.load(result['datasets'][objective]['path'], allow_pickle=False) as after:
            assert set(before.files) == set(after.files)
            for name in before.files:
                np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize('change', ['none', 'actor', 'source_steps', 'task_steps', 'updates', 'trials'])
def test_distinct_stage_serialization_requires_actual_closed_actor_and_counter_match(tmp_path, change):
    output, _, args, _ = projection_fixture(tmp_path)
    payload = torch.load(args['source_path'], weights_only=False, map_location='cpu')
    payload['source_iteration'] = 1
    if change == 'actor':
        key = next(iter(payload['learner']['actor']))
        payload['learner']['actor'][key] = payload['learner']['actor'][key] + 0.01
    elif change == 'source_steps':
        payload['source_steps'] += 1
    elif change == 'task_steps':
        payload['task_steps'][next(iter(payload['task_steps']))] += 1
    elif change == 'updates':
        payload['learner']['updates'] += 1
    elif change == 'trials':
        payload['schedule']['trials'] += 1
    stage = tmp_path / 'stage-recovery.pt'; torch.save(payload, stage)
    source = source_recovery(stage, sha(stage), 'sac')
    if change != 'none':
        with pytest.raises(ValueError, match='differs|does not bind'):
            CorpusReader(output, source)
    else:
        reader = CorpusReader(output, source)
        assert reader.policy_aliases


def test_import_firewall_and_optimizer_guard_are_cpu_only(tmp_path):
    module = load('fixture_cpu_import_firewall', REPO / 'outputs/robotwin-sac65-cpu-compat-20261031-r1/phase_worker.py')
    guard = module.CPUImportGuard()
    for name in ('jax', 'openpi', 'sapien', 'rlinf', 'fluffy_bassoon.rl.robotwin_noise'):
        with pytest.raises(ImportError, match='forbids'):
            guard.find_spec(name)
        assert name not in sys.modules
    assert not module.forbidden('fluffy_bassoon.meta_rl.vla_noise_sac')
    assert not torch.cuda.is_initialized()
    value = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.Adam([value])
    with pytest.raises(RuntimeError, match='Optimizer steps'):
        optimizer.step()
    assert value.item() == 1.0


@pytest.mark.parametrize('kind', ['unlisted', 'foreign_origin', 'foreign_origin_symlink',
                                 'context_sha', 'changed_origin', 'changed_destination',
                                 'destination_symlink', 'stale_tip', 'foreign_method', 'incomplete'])
def test_projection_alias_failures_do_not_open_general_paths(tmp_path, kind):
    output, roots, args, receipt = projection_fixture(tmp_path)
    source = source_recovery(args['source_path'], args['source_sha256'], 'sac')
    record = json.loads((roots[0] / 'episodes/trial1-ep0/chunks/000001.json').read_text())
    context = record['context']
    if kind in ('foreign_origin', 'foreign_origin_symlink'):
        key = next(k for k in receipt['files'] if k.startswith('policies/'))
        foreign = tmp_path / 'foreign' / key
        if kind == 'foreign_origin_symlink':
            foreign.parent.mkdir(parents=True)
            foreign.symlink_to(receipt['files'][key]['origins'][0])
        receipt['files'][key]['origins'] = [str(foreign)]
    elif kind == 'destination_symlink':
        policy = next((output / 'policies').glob('*.pt'))
        policy.unlink(); policy.symlink_to(tmp_path / 'outside.pt')
    elif kind == 'stale_tip':
        Path(receipt['tip_manifest']['path']).write_text('{}')
    elif kind == 'foreign_method':
        receipt['method'] = 'ppo'
    elif kind == 'incomplete':
        (output / 'ASSEMBLY-INCOMPLETE').write_text('fixture')
    write(output / 'projection.json', receipt)
    if kind in ('foreign_origin', 'foreign_origin_symlink', 'destination_symlink',
                'stale_tip', 'foreign_method', 'incomplete'):
        with pytest.raises((ValueError, FileNotFoundError)):
            CorpusReader(output, source)
        return
    reader = CorpusReader(output, source)
    with pytest.raises(ValueError, match='escapes'):
        reader.path(context['policy_checkpoint'])  # Generic RGB/native path allowance stays closed.
    if kind == 'unlisted':
        context['policy_checkpoint'] = str(tmp_path / 'outside.pt')
    elif kind == 'context_sha':
        context['policy_checkpoint_sha256'] = '0' * 64
    elif kind == 'changed_destination':
        path = output / 'policies' / Path(context['policy_checkpoint']).name
        path.unlink(); path.write_bytes(b'changed projected archive; original preserved')
    else:
        path = Path(context['policy_checkpoint'])
        path.unlink(); path.write_bytes(b'changed original archive, projected hardlink retained')
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.acting_weights(context, 'sac')


@pytest.mark.parametrize('kind', ['unlisted_file_symlink', 'unlisted_parent_symlink',
                                 'admitted_parent_symlink', 'unrelated_identical_archive'])
def test_alias_context_requires_exact_lexical_origin_without_symlink_components(tmp_path, kind):
    output, roots, args, _ = projection_fixture(tmp_path)
    source = source_recovery(args['source_path'], args['source_sha256'], 'sac')
    reader = CorpusReader(output, source)
    context = json.loads((roots[0] / 'episodes/trial1-ep0/chunks/000001.json').read_text())['context']
    original = Path(context['policy_checkpoint'])
    if kind == 'unlisted_file_symlink':
        alias = tmp_path / 'unlisted.pt'; alias.symlink_to(original)
        context['policy_checkpoint'] = str(alias)
    elif kind == 'unlisted_parent_symlink':
        alias = tmp_path / 'unlisted-directory'; alias.symlink_to(original.parent)
        context['policy_checkpoint'] = str(alias / original.name)
    elif kind == 'admitted_parent_symlink':
        moved = tmp_path / 'moved-policies'; original.parent.rename(moved)
        original.parent.symlink_to(moved)
    else:
        alias = tmp_path / 'foreign/policies' / original.name; alias.parent.mkdir(parents=True)
        shutil.copyfile(original, alias)
        assert sha(alias) == context['policy_checkpoint_sha256']
        context['policy_checkpoint'] = str(alias)
    with pytest.raises(ValueError, match='Unlisted|Redirected'):
        reader.acting_weights(context, 'sac')


@pytest.mark.parametrize('change', ['reorder', 'drop', 'duplicate', 'count', 'lineage', 'contract',
                                   'old_manifest_root', 'stale_old_manifest'])
def test_rehashed_ancestry_still_needs_actual_manifest_root_counter_and_lineage_bindings(tmp_path, change):
    output, _, args, receipt = projection_fixture(tmp_path)
    source = source_recovery(args['source_path'], args['source_sha256'], 'sac')
    ancestry_path = Path(receipt['ancestry']['path']); ancestry = json.loads(ancestry_path.read_text())
    if change in ('reorder', 'drop', 'duplicate'):
        indices = {'reorder':[1,0], 'drop':[1], 'duplicate':[0,0,1]}[change]
        ancestry['ordered_ancestry'] = [ancestry['ordered_ancestry'][i] for i in indices]
        ancestry['ordered_corpus_roots'] = [ancestry['ordered_corpus_roots'][i] for i in indices]
        ancestry['root_count'] = len(indices)
        receipt['roots'] = [receipt['roots'][i] for i in indices]
    elif change in ('count', 'lineage'):
        tip_path = Path(receipt['tip_manifest']['path']); tip = json.loads(tip_path.read_text())
        if change == 'count':
            tip['expected']['source_steps'] += 1
        else:
            tip['lineage_source_commit'] = receipt['lineage_commit'] = ancestry['lineage_commit'] = 'b' * 40
        ref = write(tip_path, tip)
        receipt['tip_manifest'] = ancestry['tip_manifest'] = ref
        ancestry['ordered_ancestry'][-1]['manifest'] = ref
        ancestry['ordered_ancestry'][-1]['expected'] = tip['expected']
    elif change == 'contract':
        entry = ancestry['ordered_ancestry'][-1]
        contract_path = Path(entry['canonical_corpus_contract']['path'])
        producer = json.loads(contract_path.read_text()); producer['model_path'] = '/wrong/motor'
        contract_ref = write(contract_path, producer)
        entry['canonical_corpus_contract'] = entry['bound_corpus_contract'] = contract_ref
        entry['corpus_contract'] = producer
        receipt['roots'][-1]['contract_sha256'] = contract_ref['sha256']
        tip_path = Path(receipt['tip_manifest']['path']); tip = json.loads(tip_path.read_text())
        tip['corpus_contract'] = contract_ref
        tip_ref = write(tip_path, tip)
        receipt['tip_manifest'] = ancestry['tip_manifest'] = entry['manifest'] = tip_ref
    elif change == 'stale_old_manifest':
        Path(ancestry['ordered_ancestry'][0]['manifest']['path']).write_text('{}')
    else:
        entry = ancestry['ordered_ancestry'][0]
        path = Path(entry['manifest']['path']); manifest = json.loads(path.read_text())
        manifest['corpus_root'] = str(tmp_path / 'foreign')
        entry['manifest'] = write(path, manifest)
    receipt['ancestry'] = write(ancestry_path, ancestry)
    write(output / 'projection.json', receipt)
    with pytest.raises(ValueError, match='Projection|projection|ancestor'):
        CorpusReader(output, source)


def test_shortened_ancestry_must_still_end_at_the_advertised_full_tip_before_aliases(tmp_path):
    output, roots, args, receipt = projection_fixture(tmp_path)
    source = source_recovery(args['source_path'], args['source_sha256'], 'sac')
    ancestry_path = Path(receipt['ancestry']['path']); ancestry = json.loads(ancestry_path.read_text())
    ancestry['ordered_ancestry'] = ancestry['ordered_ancestry'][:1]
    ancestry['ordered_corpus_roots'] = ancestry['ordered_corpus_roots'][:1]
    ancestry['root_count'] = 1; receipt['roots'] = receipt['roots'][:1]
    # Retain only first-root origins so the old alias loop would not fail for
    # an unrelated reason; the separately advertised full tip is untouched.
    first = str(roots[0]) + '/'
    receipt['files'] = {key:dict(item, origins=[x for x in item['origins'] if x.startswith(first)])
                        for key,item in receipt['files'].items() if any(x.startswith(first) for x in item['origins'])}
    receipt['ancestry'] = write(ancestry_path, ancestry)
    write(output / 'projection.json', receipt)
    with pytest.raises(ValueError, match='terminal ancestor'):
        CorpusReader(output, source)


def assembly_fixture(tmp_path):
    roots = [tmp_path / name / 'corpus' for name in ('old', 'new')]
    base = dict(source_commit=project.LINEAGE, schema='robotwin_public_decisions_v1')
    for root in roots:
        for name in ('episodes', 'frames', 'policies'):
            (root / name).mkdir(parents=True)
        write(root / 'runtime-contract.json', dict(norm='fixed', actual_render_pci=str(root)))
    old_contract = write(roots[0] / 'contract.json', base)
    checkpoint = tmp_path / 'checkpoint'; checkpoint.write_bytes(b'owned tiny fixture, not tensors')
    old_manifest = write(tmp_path / 'old-manifest.json', dict(corpus_root=str(roots[0]),
        corpus_contract=old_contract, kind='selected_trial'))
    new_contract = write(roots[1] / 'contract.json', dict(base, execution_source_commit='e' * 40,
        resume_trial_manifest_sha256=old_manifest['sha256']))
    write(roots[1].parent / 'execution-migration.json', dict(lineage_source_commit=project.LINEAGE,
        execution_source_commit='e' * 40, manifest_path=old_manifest['path'], manifest_sha256=old_manifest['sha256']))
    tip_ref = write(tmp_path / 'manifest.json', dict(corpus_root=str(roots[1]), corpus_contract=new_contract,
        method='sac', kind='whole_trial_boundary', checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)),
        expected=dict(completed_source_iterations=65, completed_source_trials=390,
                      source_steps=651638, pending_rollout_digests=[])))
    ancestry_ref = write(tmp_path / 'ancestry.json', dict(tip_manifest=tip_ref, root_count=2,
        ordered_corpus_roots=[str(r) for r in roots]))
    admission = dict(closed_reference_authorized=True, assembly_cpu_gate_passed=False,
        closed_source100_authorized=False, expected_source_rounds=65,
        lineage_commit=project.LINEAGE, method='sac', ancestry=ancestry_ref)
    ref = write(tmp_path / 'admission.json', admission)
    return roots, ref


def invoke_projection(tmp_path, ref, monkeypatch, purpose='reference'):
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['project', '--purpose', purpose,
        '--admission', ref['path'], '--admission-sha256', ref['sha256'],
        '--chain-reader', os.environ['FB2_CPU_COMPAT_READER'], '--output', str(output),
        '--expected-hostname', socket.gethostname()])
    project.main()
    return output


@pytest.mark.parametrize('kind', ['same', 'different', 'top_symlink', 'nested_symlink',
                                 'metadata_symlink', 'runtime', 'partial', 'finalbank'])
def test_tiny_projection_collisions_runtime_symlinks_and_reference_role(tmp_path, monkeypatch, kind):
    roots, ref = assembly_fixture(tmp_path)
    for root in roots:
        (root / 'frames/shared').write_bytes(b'same fixture bytes')
    if kind == 'different':
        (roots[1] / 'frames/shared').write_bytes(b'different')
    elif kind == 'top_symlink':
        shutil.rmtree(roots[1] / 'frames'); (roots[1] / 'frames').symlink_to(roots[0] / 'frames')
    elif kind == 'nested_symlink':
        (roots[1] / 'frames/shared').unlink(); (roots[1] / 'frames/shared').symlink_to(roots[0] / 'frames/shared')
    elif kind == 'metadata_symlink':
        p = roots[1] / 'runtime-contract.json'; p.unlink(); p.symlink_to(roots[0] / p.name)
    elif kind == 'runtime':
        write(roots[1] / 'runtime-contract.json', dict(norm='different'))
    elif kind == 'partial':
        (roots[1] / 'frames/uncommitted.partial').write_bytes(b'partial')
    if kind == 'same':
        output = invoke_projection(tmp_path, ref, monkeypatch)
        assert (output / 'frames/shared').read_bytes() == b'same fixture bytes'
        receipt = json.loads((output / 'projection.json').read_text())
        assert len(receipt['files']['frames/shared']['origins']) == 2
        assert receipt['purpose'] == 'reference' and receipt['source_rounds'] == 65
    else:
        with pytest.raises((ValueError, OSError)):
            invoke_projection(tmp_path, ref, monkeypatch, 'finalbank' if kind == 'finalbank' else 'reference')
        assert not (tmp_path / 'output/projection.json').exists()


def test_no_copy_fallback_and_incomplete_projection_retained(tmp_path, monkeypatch):
    roots, ref = assembly_fixture(tmp_path)
    (roots[0] / 'frames/data').write_bytes(b'fixture')
    def cross_device(*args, **kwargs):
        raise OSError(18, 'fixture cross-device link')
    monkeypatch.setattr(project.os, 'link', cross_device)
    with pytest.raises(OSError):
        invoke_projection(tmp_path, ref, monkeypatch)
    assert (tmp_path / 'output/ASSEMBLY-INCOMPLETE').exists()
    assert not (tmp_path / 'output/projection.json').exists()


@pytest.mark.parametrize('kind', ['valid', 'scalar_schema', 'old_source', 'old_teacher',
                                 'wrong_count', 'wrong_projection', 'wrong_phase'])
def test_closed65_index_descriptor_requires_current_source_teacher_schema_and_projection(tmp_path, kind):
    worker = load('fixture_source65_descriptor', REPO / 'outputs/robotwin-sac65-cpu-compat-20261031-r1/phase_worker.py')
    source = tmp_path / 'source65'; source.write_bytes(b'owned metadata fixture, not tensors')
    source_ref = dict(path=str(source), sha256=sha(source))
    projection = tmp_path / 'projection.json'; projection.write_text('{}')
    ctx = dict(tip=dict(checkpoint=source_ref))
    index = dict(partition='train', source_method='sac', teacher_method='sac',
        datasets={'ad':{}, 'dpt':{}}, source_recovery_sha256=source_ref['sha256'],
        teacher_recovery_sha256=source_ref['sha256'],
        counts=dict(complete_K4_trials=390, episodes=1560, committed_source_actions=651638, selected_actions=651638),
        source_contract=dict(source_iterations=dict(count=100, trials_per_iteration=6, episodes_per_trial=4)),
        input_sha256={str(source):source_ref['sha256'], str(projection):sha(projection)})
    if kind == 'scalar_schema':
        index['source_contract']['source_iterations'] = 100
    elif kind == 'old_source':
        index['source_recovery_sha256'] = '5' * 64
    elif kind == 'old_teacher':
        index['teacher_recovery_sha256'] = '5' * 64
    elif kind == 'wrong_count':
        index['counts']['complete_K4_trials'] = 30
    elif kind == 'wrong_projection':
        index['input_sha256'][str(projection)] = '0' * 64
    elif kind == 'wrong_phase':
        index['partition'] = 'validation'
    if kind == 'valid':
        worker.verify_source65_index(index, ctx, projection)
    else:
        with pytest.raises(ValueError):
            worker.verify_source65_index(index, ctx, projection)


def bootstrap_ticket(path, code='fixture-pin', hostname='fixture-host', output='/fixture-output'):
    return write(path, dict(schema='robotwin_cpu_gate_actual_bootstrap_v1',
        run_id='robotwin-sac65-cpu-compat-20261031-r1', code_commit=code,
        hostname=hostname, output=output, resource_receipt=dict(path='/fixture-resource.json', sha256='a'*64)))


def test_bootstrap_wait_uses_same_deadline_and_accepts_only_atomic_final_ticket(tmp_path):
    job = load('fixture_bootstrap_job', REPO / 'outputs/robotwin-sac65-cpu-compat-20261031-r1/gate_job.py')
    ticket = tmp_path / 'actual.json'; now = [100.]
    def clock(): return now[0]
    def sleep(seconds):
        now[0] += seconds
        if now[0] == 101:
            bootstrap_ticket(tmp_path / 'upload.partial')
        elif now[0] == 102:
            (tmp_path / 'upload.partial').rename(ticket)
    ref, receipt = job.wait_bootstrap(ticket, code_commit='fixture-pin', output=Path('/fixture-output'),
        hostname='fixture-host', deadline=110., clock=clock, sleep=sleep)
    assert ref['sha256'] == 'a'*64 and receipt['wait_seconds'] == 2
    assert receipt['remaining_work_seconds'] == 8  # Not a fresh1700s after wait.


def test_bootstrap_missing_ticket_cannot_extend_the_shared_deadline(tmp_path):
    job = load('fixture_bootstrap_timeout', REPO / 'outputs/robotwin-sac65-cpu-compat-20261031-r1/gate_job.py')
    now = [1698.]
    with pytest.raises(TimeoutError):
        job.wait_bootstrap(tmp_path/'absent', code_commit='fixture-pin', output=Path('/fixture-output'),
            hostname='fixture-host', deadline=1700., clock=lambda:now[0],
            sleep=lambda seconds:now.__setitem__(0,now[0]+seconds))
    assert now[0] == 1700.


@pytest.mark.parametrize('field', ['code_commit', 'hostname', 'output', 'schema'])
def test_bootstrap_rejects_stale_scope_without_accepting_a_resource_template(tmp_path, field):
    job = load('fixture_bootstrap_scope', REPO / 'outputs/robotwin-sac65-cpu-compat-20261031-r1/gate_job.py')
    path=tmp_path/'ticket.json'; bootstrap_ticket(path)
    value=json.loads(path.read_text()); value[field]='wrong'; write(path,value)
    with pytest.raises(ValueError,match='Stale or mismatched'):
        job.wait_bootstrap(path,code_commit='fixture-pin',output=Path('/fixture-output'),
                           hostname='fixture-host',deadline=time.monotonic()+5)


@pytest.mark.parametrize('mode', ['timeout', 'TERM', 'INT', 'KILL'])
def test_owned_cpu_guardian_reaps_worker_and_changed_pgid_descendant_only(tmp_path, mode):
    adapter_path = REPO / 'outputs/robotwin-sac65-cpu-compat-20261031-r1/owned_cpu_runtime.py'
    adapter = load('fixture_cpu_owned_runtime', adapter_path)
    adapter.become_subreaper()  # Only this owned pytest worker's descendants.
    helper_dir = os.environ['FB2_CPU_OWNED_HELPERS']
    worker_info, grand_info = tmp_path/'worker.json', tmp_path/'grand.json'
    grand = tmp_path/'grand.py'
    grand.write_text("import os,signal,time,json\nfrom pathlib import Path\n"
        "os.setpgid(0,0)\n"
        +f"Path({str(grand_info)!r}).write_text(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),pgid=os.getpgrp(),"
        "blocked=[int(x) for x in signal.pthread_sigmask(signal.SIG_BLOCK,[])],"
        "term_default=signal.getsignal(signal.SIGTERM)==signal.SIG_DFL,"
        "int_normal=signal.getsignal(signal.SIGINT) in (signal.SIG_DFL,signal.default_int_handler))))\n"
        "time.sleep(120)\n")
    worker = tmp_path/'worker.py'
    worker.write_text("import os,signal,subprocess,sys,time,json\nfrom pathlib import Path\n"
        +f"child=subprocess.Popen([sys.executable,'-I','-B','-S',{str(grand)!r}])\n"
        +f"Path({str(worker_info)!r}).write_text(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),child=child.pid)))\n"
        "time.sleep(120)\n")
    driver = tmp_path/'driver.py'; owned_output=tmp_path/'owned'
    driver.write_text("import importlib.util,sys,signal,time,os\nfrom pathlib import Path\n"
        +f"spec=importlib.util.spec_from_file_location('owned',{str(adapter_path)!r})\n"
        "m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)\n"
        "def stop(signum,frame):raise RuntimeError('fixture parent signal')\n"
        "signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)\n"
        "runtime=None\ntry:\n"
        +f" runtime=m.CPUPhaseRuntime({helper_dir!r},Path({str(owned_output)!r}),deadline=time.monotonic()+40)\n"
        +f" runtime.run('fixture',[sys.executable,'-I','-B','-S',{str(worker)!r}],{str(tmp_path)!r},os.environ,Path({str(tmp_path/'worker.log')!r}),{1 if mode=='timeout' else 15})\n"
        "except BaseException:pass\nfinally:\n"
        " signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
        " if runtime is not None:runtime.close()\n")
    unrelated = subprocess.Popen([sys.executable,'-I','-B','-S','-c','import time;time.sleep(120)'],start_new_session=True)
    process = subprocess.Popen([sys.executable,'-I','-B','-S',str(driver)],start_new_session=True,
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    identities = []
    try:
        until=time.monotonic()+8
        while not (worker_info.exists() and grand_info.exists()) and time.monotonic()<until:time.sleep(.02)
        assert worker_info.exists() and grand_info.exists()
        w,g=json.loads(worker_info.read_text()),json.loads(grand_info.read_text())
        identities=[w['pid'],g['pid']]
        assert w['sid']==g['sid'] and w['sid']!=g['pgid']
        assert int(signal.SIGTERM) not in g['blocked'] and int(signal.SIGINT) not in g['blocked']
        assert g['term_default'] and g['int_normal']
        if mode!='timeout':process.send_signal(getattr(signal,'SIG'+mode))
        process.wait(timeout=12)
        guardian=json.loads((owned_output/'owned-runtime.json').read_text())['guardian_pid']
        until=time.monotonic()+10
        while time.monotonic()<until:
            try:os.waitpid(guardian,os.WNOHANG)
            except ChildProcessError:pass
            if not any(Path('/proc',str(pid)).exists() for pid in [*identities,guardian]):break
            time.sleep(.02)
        assert not any(Path('/proc',str(pid)).exists() for pid in [*identities,guardian])
        assert unrelated.poll() is None
        receipt=json.loads((owned_output/'owned-cleanup-latest.json').read_text())
        assert receipt['clean'] and receipt['adopted']['clean']
        assert any(row['signal']==int(signal.SIGTERM) for row in receipt['signal_attempts'])
        receipts=[json.loads(p.read_text()) for p in owned_output.glob('owned-cleanup-[0-9]*.json')]
        assert any(row['pid']==g['pid'] for r in receipts for row in r['adopted']['reaped'])
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)
        unrelated.terminate();unrelated.wait(timeout=5)
