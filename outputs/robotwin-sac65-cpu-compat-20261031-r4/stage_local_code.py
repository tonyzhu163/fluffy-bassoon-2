"""Materialize exact reviewed Git code locally; source/data remain read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

RUN='robotwin-sac65-cpu-compat-20261031-r4'
BOOTSTRAP=tuple('outputs/'+RUN+'/'+name for name in
    ('job.sh','gate_job.py','stage_local_code.py','owned_cpu_runtime.py','visibility_probe.py'))


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_environment(environment):
    selectors=('GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_INDEX_FILE',
               'GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES')
    # GIT_CONFIG_COUNT/KEY_*/VALUE_* and GIT_CONFIG_PARAMETERS can also
    # redirect core.worktree. Remove inherited config override channels.
    return {key:value for key,value in environment.items()
            if key not in selectors and not key.startswith('GIT_CONFIG')}


def local_mount(path):
    candidates=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);parts=left.split()
        point=Path(parts[4].replace('\\040',' ').replace('\\134','\\'))
        if point==path or point in path.parents:candidates.append((len(point.parts),str(point),right.split()[0]))
    require(candidates,'Cannot identify local code filesystem')
    _,point,kind=max(candidates)
    require(kind in ('tmpfs','overlay','ext4','xfs','btrfs'),'Code staging must use a local filesystem')
    return dict(mountpoint=point,filesystem=kind)


def stage(source,pin,output,*,base=Path('/tmp'),bootstrap=BOOTSTRAP):
    source,base,output=Path(source).resolve(),Path(base).resolve(),Path(output).resolve()
    require(source.is_dir() and base.is_dir(),'Missing original code or local temp directory')
    filesystem=local_mount(base);free=shutil.disk_usage(base).free
    require(free>=512<<20,'Need512MiB free for20MiB code plus Git/index headroom')
    require(base.stat().st_dev!=source.stat().st_dev,'Local code must leave the original shared filesystem')
    scratch=Path(tempfile.mkdtemp(prefix=RUN+'-',dir=base));local=scratch/'repo'
    probe=scratch/'write-probe'
    with probe.open('xb') as stream:stream.write(b'local-code');stream.flush();os.fsync(stream.fileno())
    probe.unlink()
    logs=output/'local-code-logs';logs.mkdir()
    # Repository selectors inherited from a caller must not redirect checkout
    # or its index back into the protected original worktree.
    environment=git_environment(os.environ)
    def git(name,arguments,cwd=None):
        path=logs/(name+'.log')
        with path.open('xb') as stream,(logs/(name+'.stderr')).open('xb') as error:
            result=subprocess.run(['git',*arguments],cwd=cwd,env=environment,stdout=stream,stderr=error,check=False)
        require(result.returncode==0,'Owned local-code Git step failed: '+name)
        return path.read_bytes()
    original_head=git('original-head',['rev-parse','HEAD'],source).decode().strip()
    require(original_head==pin,'Original deployed Git HEAD differs from reviewed pin')
    # --shared borrows existing Git objects read-only; it never resets or writes
    # the source worktree and does not copy data/checkpoints or model caches.
    git('clone',['clone','--shared','--no-checkout',str(source),str(local)])
    git('checkout',['checkout','--detach',pin],local)
    effective_head=git('effective-head',['rev-parse','HEAD'],local).decode().strip()
    require(effective_head==pin,'Job-local code HEAD differs')
    status=git('full-status',['--no-optional-locks','-c','core.fsmonitor=false','status','--porcelain'],local)
    require(status==b'','Exact full clean job-local checkout required')
    bootstrap_hashes={}
    for relative in bootstrap:
        original,effective=source/relative,local/relative
        require(original.is_file() and effective.is_file() and not original.is_symlink()
                and not effective.is_symlink(),'Redirected/missing bootstrap code')
        require(sha(original)==sha(effective),'Deployed bootstrap differs from reviewed Git blobs: '+relative)
        bootstrap_hashes[relative]=sha(effective)
    receipt=dict(schema='robotwin_exact_job_local_code_v1',completed=True,
        original_code_root=str(source),effective_code_root=str(local),local_scratch_root=str(scratch),
        code_commit=pin,original_head=original_head,effective_head=effective_head,
        full_clean_git_status=True,status_path=str(logs/'full-status.log'),status_sha256=sha(logs/'full-status.log'),
        local_mount=filesystem,free_bytes_before=free,local_device=base.stat().st_dev,
        source_device=source.stat().st_dev,source_objects='read-only Git alternates; original worktree unchanged',
        bootstrap_sha256=bootstrap_hashes)
    path=output/'local-code.json';temporary=path.with_name(path.name+'.partial')
    with temporary.open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);temporary.unlink()
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-code',type=Path,required=True);p.add_argument('--code-commit',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    stage(a.source_code,a.code_commit,a.output)
