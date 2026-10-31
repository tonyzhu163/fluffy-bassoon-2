"""Objective-aware AD/BC regressions; allocated Linux CPU, no simulator/fit."""
import json
from pathlib import Path
import shlex

import numpy as np
import pytest

from test_robotwin_history_corpus import fixture,write_json
from test_robotwin_finalbank_index_bindings import indexes_fixture,runner,write
from fluffy_bassoon.meta_rl import robotwin_history_corpus as corpus


@pytest.mark.parametrize('method,partition',[
    ('ppo','train'),('ppo_reset','train'),('sac','train'),('sac','validation')])
def test_ad_and_bc_arrays_equal_paired_export_without_teacher_loading_or_inference(tmp_path,monkeypatch,method,partition):
    root,args=fixture(tmp_path,method,partition=partition)
    objective='ad' if partition=='train' else 'behavior_cloning'
    paired=corpus.export_corpus(root,tmp_path/'paired',objectives=(objective,'dpt'),**args)
    unused_teacher=Path(args['teacher_path']);unused_teacher.unlink()
    def forbidden(*args,**kwargs):raise AssertionError('AD/BC must not load or infer an unused teacher')
    monkeypatch.setattr(corpus,'load_teacher',forbidden)
    monkeypatch.setattr(corpus,'teacher_labels',forbidden)
    # Stale optional teacher arguments cannot create an unverified teacher claim.
    result=corpus.export_corpus(root,tmp_path/'executed-only',objectives=(objective,),**args)
    with np.load(paired['datasets'][objective]['path'],allow_pickle=False) as before, \
            np.load(result['datasets'][objective]['path'],allow_pickle=False) as after:
        assert before.files==after.files
        for name in before.files:
            assert before[name].dtype==after[name].dtype
            np.testing.assert_array_equal(before[name],after[name])
        np.testing.assert_array_equal(after['target_noise'],after['executed_noise'])
    assert result['counts']==paired['counts'] and result['source_trials']==paired['source_trials']
    assert result['interface']==paired['interface'] and result['source_contract']==paired['source_contract']
    assert result['teacher_method'] is result['teacher_recovery_sha256'] is None
    assert str(unused_teacher.resolve()) not in result['input_sha256']


def test_ad_only_cli_api_needs_source_but_no_teacher_arguments(tmp_path,monkeypatch):
    root,args=fixture(tmp_path,'sac')
    args={k:v for k,v in args.items() if not k.startswith('teacher_')}
    monkeypatch.setattr(corpus,'load_teacher',lambda *a,**k:pytest.fail('unused teacher'))
    result=corpus.export_corpus(root,tmp_path/'ad',objectives=('ad',),**args)
    assert set(result['datasets'])=={'ad'} and result['counts']['selected_actions']==256
    with pytest.raises(ValueError,match='DPT requires an explicitly pinned'):
        corpus.export_corpus(root,tmp_path/'dpt-missing',objectives=('dpt',),**args)


@pytest.mark.parametrize('change',['fixed_history','acting_archive','frame','native_duration'])
def test_ad_only_keeps_actual_behavior_provenance_frame_and_mask_guards(tmp_path,change):
    root,args=fixture(tmp_path,'sac',fixed=change=='fixed_history')
    args={k:v for k,v in args.items() if not k.startswith('teacher_')}
    if change=='acting_archive':next((root/'policies').glob('*.pt')).unlink()
    elif change=='frame':next((root/'frames').glob('*.npz')).write_bytes(b'corrupt')
    elif change=='native_duration':
        p=root/'episodes/trial1-ep0/chunks/000001.json';value=json.loads(p.read_text())
        value['native_result']['executed_primitive_steps']=30;write_json(p,value)
        manifest=p.parent.parent/'manifest.json';value=json.loads(manifest.read_text())
        value['chunks'][0]['sha256']=corpus.file_sha256(p);write_json(manifest,value)
    with pytest.raises((ValueError,FileNotFoundError)):
        corpus.export_corpus(root,tmp_path/'invalid',objectives=('ad',),**args)
    assert not (tmp_path/'invalid').exists()


def ad_indexes(tmp_path):
    b,projection=indexes_fixture(tmp_path)
    b.pop('teacher100_source');b['objectives']=['ad']
    for partition in ('train','validation'):
        key=partition+'_index';path=Path(b[key]['path']);m=json.loads(path.read_text())
        objective='ad' if partition=='train' else 'behavior_cloning'
        m['datasets']={objective:m['datasets'][objective]}
        m['teacher_method']=m['teacher_recovery_sha256']=None
        b[key]=write(path,m)
    return b,projection


@pytest.mark.parametrize('change',['valid','source65','old_source','foreign_projection','overlap'])
def test_final100_ad_only_indexes_keep_coverage_input_binding_and_disjoint_cases(tmp_path,change):
    b,projection=ad_indexes(tmp_path);path=Path(b['train_index']['path']);m=json.loads(path.read_text())
    if change=='source65':m['counts']['complete_K4_trials']=390
    elif change=='old_source':m['source_recovery_sha256']='f'*64
    elif change=='foreign_projection':m['input_sha256'][str(Path(b['projection_root'])/'contract.json')]='f'*64
    elif change=='overlap':
        val=Path(b['validation_index']['path']);v=json.loads(val.read_text())
        v['source_trials'][0]['cases'][0]['actual_seed']=0;write(val,v)
    write(path,m)
    if change=='valid':runner.verify_final_indexes(b,projection,('ad',))
    else:
        with pytest.raises(ValueError):runner.verify_final_indexes(b,projection,('ad',))


def bindings_file(tmp_path,objectives,*,execute=False):
    b=json.loads((Path(__file__).resolve().parents[1]/
        'outputs/robotwin-finalbank-offline-plan-20261031-r1/bindings.template.json').read_text())
    b['objectives']=list(objectives)
    if 'dpt' not in objectives:b.pop('teacher100_source')
    if execute:
        b['allocated_hostname']='job-cpu-fake-master-0'
        b['admission']={key:True for key in b['admission']}
        b['admission']['teacher_source_qualification_reviewed']=False
    path=tmp_path/'bindings.json';path.write_text(json.dumps(b));return path,b


def invoke(monkeypatch,path,phase,objectives,*,execute=False):
    argv=['run_offline','--bindings',str(path),'--bindings-sha256',runner.digest(path),'--phase',phase,
          '--objectives',*objectives]
    if execute:argv.append('--execute')
    monkeypatch.setattr(runner.sys,'argv',argv)
    return runner.main()


def test_default_paired_request_still_prints_four_fixed500_fits(tmp_path,monkeypatch,capsys):
    path,b=bindings_file(tmp_path,('ad','dpt'))
    monkeypatch.setattr(runner.sys,'argv',['run_offline','--bindings',str(path),
        '--bindings-sha256',runner.digest(path),'--phase','fit'])
    runner.main();commands=[shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert [(c[c.index('--objective')+1],c[c.index('--mode')+1]) for c in commands]==[
        ('ad','full'),('ad','static'),('dpt','full'),('dpt','static')]
    assert all(c[c.index('--updates')+1]=='500' and c[c.index('--milestones')+1:c.index('--milestones')+4]==['25','100','500'] for c in commands)


def test_ad_only_prints_ad_bc_exports_without_teacher_and_only_two_existing_fits(tmp_path,monkeypatch,capsys):
    path,b=bindings_file(tmp_path,('ad',))
    invoke(monkeypatch,path,'export',('ad',));commands=[shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands)==2 and [c[c.index('--objectives')+1] for c in commands]==['ad','behavior_cloning']
    assert all('--teacher-recovery' not in c and '--teacher-sha256' not in c and '--teacher-method' not in c for c in commands)
    invoke(monkeypatch,path,'fit',('ad',));commands=[shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands)==2 and all(c[c.index('--objective')+1]=='ad' for c in commands)
    assert [c[c.index('--mode')+1] for c in commands]==['full','static']


@pytest.mark.parametrize('phase,objectives',[('export',('dpt',)),('fit',('dpt',)),
    ('export',('ad','dpt')),('fit',('ad','dpt'))])
def test_unreviewed_dpt_is_rejected_before_git_models_or_subprocesses(tmp_path,monkeypatch,phase,objectives):
    path,b=bindings_file(tmp_path,objectives,execute=True)
    monkeypatch.setattr(runner.socket,'gethostname',lambda:b['allocated_hostname'])
    def forbidden(*a,**k):pytest.fail('Unreviewed DPT must fail before any Git/model/subprocess call')
    monkeypatch.setattr(runner.subprocess,'check_output',forbidden);monkeypatch.setattr(runner.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='teacher_source_qualification_reviewed'):
        invoke(monkeypatch,path,phase,objectives,execute=True)


@pytest.mark.parametrize('missing',['owner_compute_reservation_admitted','disjoint_source_cases_verified'])
def test_ad_only_still_requires_actual_funding_and_source_split_admissions(tmp_path,monkeypatch,missing):
    path,b=bindings_file(tmp_path,('ad',),execute=True);b['admission'][missing]=False
    path.write_text(json.dumps(b));monkeypatch.setattr(runner.socket,'gethostname',lambda:b['allocated_hostname'])
    with pytest.raises(ValueError,match=missing):invoke(monkeypatch,path,'fit',('ad',),execute=True)


def test_ad_only_passes_unused_teacher_gate_without_a_teacher_receipt(tmp_path,monkeypatch):
    path,b=bindings_file(tmp_path,('ad',),execute=True)
    receipt=write(tmp_path/'fake-owner-receipt.json',dict(scope='CPU metadata fake only'))
    b['admission_receipts']={key:receipt for key in runner.required_admissions('fit',('ad',))}
    path.write_text(json.dumps(b));monkeypatch.setattr(runner.socket,'gethostname',lambda:b['allocated_hostname'])
    class ReachedGit(Exception):pass
    def stop(*a,**k):raise ReachedGit
    monkeypatch.setattr(runner.subprocess,'check_output',stop)
    with pytest.raises(ReachedGit):invoke(monkeypatch,path,'fit',('ad',),execute=True)


def test_ad_two_fit_subset_cannot_use_full_four_fit_time_allowance(tmp_path,monkeypatch):
    b,projection=ad_indexes(tmp_path);projection.update(status='byte_projection_only_not_export_admission',
        purpose='finalbank',source_rounds=100,method='sac')
    b['projection_receipt']=write(Path(b['projection_root'])/'projection.json',projection)
    b.update(python='/not-launched/python',code_root=str(tmp_path),code_commit='a'*40,
        allocated_hostname='job-cpu-fake-master-0',fits_root=str(tmp_path/'fits'),admitted_parent_seconds=3300)
    b['admission']={k:True for k in runner.required_admissions('fit',('ad',))}
    b['admission']['teacher_source_qualification_reviewed']=False
    receipt=write(tmp_path/'fake-owner-receipt.json',dict(scope='CPU metadata fake only'))
    b['admission_receipts']={k:receipt for k in runner.required_admissions('fit',('ad',))}
    path=tmp_path/'bindings.json';path.write_text(json.dumps(b))
    monkeypatch.setattr(runner.socket,'gethostname',lambda:b['allocated_hostname'])
    monkeypatch.setattr(runner.subprocess,'check_output',lambda argv,**kwargs:b['code_commit'] if 'rev-parse' in argv else '')
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Over-cap subset must launch no fit'))
    with pytest.raises(ValueError,match='Remaining parent time exceeds admitted cap'):
        invoke(monkeypatch,path,'fit',('ad',),execute=True)
