"""Allocated-Linux regressions for the two retained ancestry encodings.

Only tiny synthetic corpora are opened. Production ancestry JSON is read as
bounded metadata; no production payload, simulator, optimizer or job is used.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_robotwin_offline_compat_gate import projection_fixture,write,sha
from test_robotwin_ad_first import ad_indexes,runner
from fluffy_bassoon.meta_rl import robotwin_history_corpus as corpus


def projected(tmp_path,form='direct',genesis='omitted'):
    output,roots,args,receipt=projection_fixture(tmp_path)
    path=Path(receipt['ancestry']['path']);ancestry=json.loads(path.read_text())
    if form=='direct':ancestry['ordered_ancestry'][1]['execution_migration']=ancestry['ordered_ancestry'][1]['execution_migration']['ref']
    else:
        ref=ancestry['ordered_ancestry'][1]['execution_migration']['ref']
        ref['bytes']=Path(ref['path']).stat().st_size  # Retained legacy reference shape.
    if genesis=='omitted':ancestry['ordered_ancestry'][0].pop('execution_migration')
    publish(output,receipt,path,ancestry)
    return output,roots,args,receipt,path,ancestry


def publish(output,receipt,path,ancestry):
    receipt['ancestry']=write(path,ancestry)
    write(output/'projection.json',receipt)


def migration_ref(entry):
    value=entry['execution_migration']
    return value['ref'] if 'ref' in value else value


def replace_migration(entry,value,path=None):
    migration=entry['execution_migration'];ref=migration_ref(entry)
    updated=write(Path(path or ref['path']),value)
    if 'ref' in migration:
        migration['ref']=updated;migration['value']=value
    else:entry['execution_migration']=updated


def test_exact_retained_final62_metadata_explains_the_runtime_schema_mismatch():
    path=Path(__file__).resolve().parents[1]/'outputs/robotwin-finalbank-offline-plan-20261031-r1/SAC100-FINAL-REFERENCE-20261031.json'
    assert sha(path)=='0af5082349ec1a032eff6b752e3b91fd473b8c922a394c496b4f75bceff68c7e'
    a=json.loads(path.read_text());entries=a['ordered_ancestry']
    assert len(entries)==62 and 'execution_migration' not in entries[0]
    assert not entries[0]['corpus_contract'].get('resume_trial_manifest_sha256')
    assert all(set(e['execution_migration'])=={'path','sha256'} for e in entries[1:])
    assert all(e['corpus_contract']['resume_trial_manifest_sha256'] for e in entries[1:])


@pytest.mark.parametrize('form',['direct','wrapped'])
@pytest.mark.parametrize('genesis',['omitted','null'])
def test_both_encodings_export_identical_ad_arrays_and_keep_paid_projection_readonly(tmp_path,monkeypatch,form,genesis):
    output,_,args,receipt,ancestry_path,ancestry=projected(tmp_path,form,genesis)
    before={p:sha(p) for p in (output/'projection.json',ancestry_path,Path(migration_ref(ancestry['ordered_ancestry'][1])['path']))}
    def forbidden(*args,**kwargs):raise AssertionError('AD must not load or label with a teacher')
    monkeypatch.setattr(corpus,'load_teacher',forbidden);monkeypatch.setattr(corpus,'teacher_labels',forbidden)
    source_only={k:v for k,v in args.items() if not k.startswith('teacher_')}
    result=corpus.export_corpus(output,tmp_path/'ad-compatible',objectives=('ad',),**source_only)
    baseline=json.loads((tmp_path/'unsplit-baseline/index.json').read_text())
    with np.load(baseline['datasets']['ad']['path'],allow_pickle=False) as a, np.load(result['datasets']['ad']['path'],allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    assert result['counts']['complete_K4_trials']==2 and result['counts']['episodes']==8
    assert result['counts']['selected_actions']==256 and result['teacher_recovery_sha256'] is None
    assert all(sha(p)==h for p,h in before.items())
    ref=migration_ref(ancestry['ordered_ancestry'][1]);assert result['input_sha256'][ref['path']]==ref['sha256']


@pytest.mark.parametrize('form',['direct','wrapped'])
@pytest.mark.parametrize('change',['missing','null','SHA','foreign_path','predecessor_SHA','predecessor_path','executor','lineage','method'])
def test_resumed_root_still_requires_actual_bound_migration_proofs(tmp_path,form,change):
    output,_,args,receipt,path,ancestry=projected(tmp_path,form)
    entry=ancestry['ordered_ancestry'][1]
    if change=='missing':entry.pop('execution_migration')
    elif change=='null':entry['execution_migration']=None
    elif change=='SHA':migration_ref(entry)['sha256']='0'*64
    else:
        ref=migration_ref(entry);value=json.loads(Path(ref['path']).read_text())
        if change=='predecessor_SHA':value['manifest_sha256']='f'*64
        elif change=='predecessor_path':value['manifest_path']=str(tmp_path/'foreign-manifest.json')
        elif change=='executor':value['execution_source_commit']='f'*40
        elif change=='lineage':value['lineage_source_commit']='f'*40
        elif change=='method':value['method']='ppo'
        replace_migration(entry,value,tmp_path/'foreign-migration.json' if change=='foreign_path' else None)
    publish(output,receipt,path,ancestry)
    source=corpus.source_recovery(args['source_path'],args['source_sha256'],'sac')
    with pytest.raises(ValueError,match='Projection|projection'):
        corpus.CorpusReader(output,source)


def test_legacy_embedded_value_still_must_match_the_actual_hash_bound_file(tmp_path):
    output,_,args,receipt,path,ancestry=projected(tmp_path,'wrapped')
    ancestry['ordered_ancestry'][1]['execution_migration']['value']['method']='ppo'
    publish(output,receipt,path,ancestry)
    with pytest.raises(ValueError,match='migration binding'):
        corpus.CorpusReader(output,corpus.source_recovery(args['source_path'],args['source_sha256'],'sac'))


@pytest.mark.parametrize('form',['direct','wrapped'])
def test_original_prefix_cannot_claim_a_predecessor_even_when_its_ref_is_valid(tmp_path,form):
    output,_,args,receipt,path,ancestry=projected(tmp_path,form)
    ancestry['ordered_ancestry'][0]['execution_migration']=copy.deepcopy(ancestry['ordered_ancestry'][1]['execution_migration'])
    publish(output,receipt,path,ancestry)
    with pytest.raises(ValueError,match='original prefix'):
        corpus.CorpusReader(output,corpus.source_recovery(args['source_path'],args['source_sha256'],'sac'))


@pytest.mark.parametrize('malformed',['string','empty','wrapper_without_value','ref_without_path','mixed'])
def test_unsupported_migration_reference_shapes_do_not_fall_back_to_unbound_paths(tmp_path,malformed):
    output,_,args,receipt,path,ancestry=projected(tmp_path)
    entry=ancestry['ordered_ancestry'][1];ref=copy.deepcopy(entry['execution_migration'])
    bad={'string':ref['path'],'empty':{},'wrapper_without_value':dict(ref=ref),
        'ref_without_path':dict(sha256=ref['sha256']),
        'mixed':dict(ref=ref,value={},**ref)}
    entry['execution_migration']=bad[malformed];publish(output,receipt,path,ancestry)
    with pytest.raises(ValueError,match='migration reference'):
        corpus.CorpusReader(output,corpus.source_recovery(args['source_path'],args['source_sha256'],'sac'))


def index_ancestry(tmp_path,form,genesis):
    indices=tmp_path/'indices';indices.mkdir()
    b,projection=ad_indexes(indices)
    original=write(tmp_path/'original-manifest.json',{})
    contract=write(tmp_path/'producer-contract.json',{})
    value=dict(method='sac',manifest_path=original['path'],manifest_sha256=original['sha256'])
    ref=write(tmp_path/'execution-migration.json',value)
    first=dict(manifest=original,bound_corpus_contract=contract,canonical_corpus_contract=contract,corpus_contract={})
    if genesis=='null':first['execution_migration']=None
    resumed=dict(manifest=projection['tip_manifest'],bound_corpus_contract=contract,canonical_corpus_contract=contract,
        corpus_contract=dict(resume_trial_manifest_sha256=original['sha256']),
        execution_migration=ref if form=='direct' else dict(ref=ref,value=value))
    ancestry=dict(ordered_ancestry=[first,resumed])
    projection['ancestry']=write(tmp_path/'index-ancestry.json',ancestry)
    for partition in ('train','validation'):
        path=Path(b[partition+'_index']['path']);m=json.loads(path.read_text());m['input_sha256'][ref['path']]=ref['sha256']
        b[partition+'_index']=write(path,m)
    return b,projection,ancestry,ref


@pytest.mark.parametrize('form',['direct','wrapped'])
@pytest.mark.parametrize('genesis',['omitted','null'])
def test_post_export_index_guard_admits_only_the_declared_normalized_migration_ref(tmp_path,form,genesis):
    b,projection,_,_=index_ancestry(tmp_path,form,genesis)
    runner.verify_final_indexes(b,projection,('ad',))


@pytest.mark.parametrize('change',['missing','null'])
def test_post_export_index_guard_cannot_silently_omit_a_resumed_migration(tmp_path,change):
    b,projection,ancestry,_=index_ancestry(tmp_path,'direct','omitted')
    if change=='missing':ancestry['ordered_ancestry'][1].pop('execution_migration')
    else:ancestry['ordered_ancestry'][1]['execution_migration']=None
    projection['ancestry']=write(Path(projection['ancestry']['path']),ancestry)
    with pytest.raises(ValueError,match='lost its predecessor'):runner.verify_final_indexes(b,projection,('ad',))


def test_post_export_index_guard_keeps_foreign_unlisted_migration_inputs_rejected(tmp_path):
    b,projection,_,ref=index_ancestry(tmp_path,'direct','omitted')
    path=Path(b['train_index']['path']);m=json.loads(path.read_text())
    foreign=write(tmp_path/'unlisted-migration.json',json.loads(Path(ref['path']).read_text()))
    m['input_sha256'][foreign['path']]=foreign['sha256'];b['train_index']=write(path,m)
    with pytest.raises(ValueError,match='unbound input paths'):runner.verify_final_indexes(b,projection,('ad',))
