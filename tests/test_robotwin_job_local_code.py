"""Allocated-Linux regressions for exact Git code materialization only."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/robotwin-sac65-cpu-compat-20261031-r4'
spec=importlib.util.spec_from_file_location('fixture_local_code',ROOT/'stage_local_code.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def git(cwd,*arguments):
    result=subprocess.run(['git',*arguments],cwd=cwd,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    return result.stdout.strip()


def linked_source(tmp_path):
    repo=tmp_path/'original-repo';repo.mkdir();git(repo,'init','-b','main')
    (repo/'bootstrap.py').write_text("marker='reviewed'\n")
    git(repo,'add','bootstrap.py')
    git(repo,'-c','user.name=CPU fixture','-c','user.email=cpu-fixture@example.invalid',
        'commit','-m','tiny reviewed code')
    pin=git(repo,'rev-parse','HEAD');source=tmp_path/'original-linked-worktree'
    git(repo,'worktree','add','--detach',str(source),pin)
    return source,pin


def test_linked_worktree_materializes_exact_full_clean_local_code_and_preserves_source(tmp_path,monkeypatch):
    source,pin=linked_source(tmp_path);output=tmp_path/'receipts';output.mkdir()
    base=Path(tempfile.mkdtemp(prefix='r4-fixture-',dir='/tmp'))
    try:
        with monkeypatch.context() as context:
            context.setenv('GIT_DIR',str(source/'.git'))
            context.setenv('GIT_WORK_TREE',str(source))
            context.setenv('GIT_INDEX_FILE',str(tmp_path/'foreign-index'))
            receipt=module.stage(source,pin,output,base=base,bootstrap=('bootstrap.py',))
        effective=Path(receipt['effective_code_root'])
        assert receipt['original_head']==receipt['effective_head']==pin
        assert receipt['full_clean_git_status'] and Path(receipt['status_path']).read_bytes()==b''
        assert receipt['source_device']!=receipt['local_device']
        assert (effective/'bootstrap.py').read_bytes()==(source/'bootstrap.py').read_bytes()
        assert not (tmp_path/'foreign-index').exists()
        assert git(source,'rev-parse','HEAD')==pin and git(source,'status','--porcelain')==''
    finally:shutil.rmtree(base)


def test_dirty_deployed_bootstrap_is_rejected_without_overwriting_original_changes(tmp_path):
    source,pin=linked_source(tmp_path);output=tmp_path/'receipts';output.mkdir()
    (source/'bootstrap.py').write_text("marker='uncommitted original change'\n")
    base=Path(tempfile.mkdtemp(prefix='r4-fixture-',dir='/tmp'))
    try:
        with pytest.raises(ValueError,match='Deployed bootstrap differs'):
            module.stage(source,pin,output,base=base,bootstrap=('bootstrap.py',))
        assert (source/'bootstrap.py').read_text()=="marker='uncommitted original change'\n"
        assert not (output/'local-code.json').exists()
    finally:shutil.rmtree(base)
