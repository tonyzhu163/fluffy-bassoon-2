"""Download fixed public Pro bytes; stdlib only, no credentials or deserialization."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import urllib.parse
import urllib.request

REVISION = 'c86fc3b8293185a6f373677018ff3e37f8391602'
SOURCE = 'eafdb809426b13153aa1e4c42d6601844217dfec'
REPO = 'zhouxueyang/LIBERO-Pro'
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
META = PROJECT/'deps/libero-pro-source-20261029-r1'
TARGET = PROJECT/'inputs/libero-pro-full-c86fc3b8-20261031-r1'
MAX_SECONDS, WORKERS, RETRIES, REQUEST_SECONDS = 600, 4, 3, 15
SUITES = ('libero_spatial','libero_object','libero_goal','libero_10')
TYPES = ('object','swap','lan','task')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def safe_path(name):
    if (not isinstance(name,str) or not name or name.startswith('/') or '\\' in name
            or any(ord(c)<32 for c in name) or any(p in ('','.','..') for p in name.split('/'))
            or name.split('/')[0] in ('READY.json','receipt.json','.download-partials','.provenance')):
        raise ValueError('Unsafe or reserved public dataset path')
    return str(PurePosixPath(name))


def main():
    started = time.monotonic()
    deadline = started+MAX_SECONDS
    tree_raw, info_raw = (META/'hf-tree.json').read_bytes(), (META/'hf-info.json').read_bytes()
    tree, info = json.loads(tree_raw), json.loads(info_raw)
    files = sorted((r for r in tree if r['type']=='file'),key=lambda r:r['path'])
    if (info['id']!=REPO or info['sha']!=REVISION or info['private'] or info['gated']
            or len(files)!=676 or sum(r['size'] for r in files)!=10269485
            or {r['path'] for r in files}!={r['rfilename'] for r in info['siblings']}):
        raise ValueError('Pinned public676-file metadata differs')
    names=set()
    for r in files:
        p=safe_path(r['path'])
        if p in names or type(r['size']) is not int or r['size']<0 or not re.fullmatch('[0-9a-f]{40}',r['oid']):
            raise ValueError('Duplicate path or invalid expected Git blob metadata')
        names.add(p)
        if r.get('lfs'):
            lfs=r['lfs']
            pointer=(f'version https://git-lfs.github.com/spec/v1\noid sha256:{lfs["oid"]}\nsize {lfs["size"]}\n').encode()
            if (lfs['size']!=r['size'] or len(pointer)!=lfs['pointerSize'] or git_blob(pointer)!=r['oid']
                    or not re.fullmatch('[0-9a-f]{64}',lfs['oid'])):
                raise ValueError('LFS content/pointer Git identity differs')
    if any('/'.join(PurePosixPath(name).parts[:i]) in names for name in names for i in range(1,len(PurePosixPath(name).parts))):
        raise ValueError('Dataset file path collides with a directory prefix')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=META/'repo',text=True).strip()
    if source!=SOURCE:
        raise ValueError('Official Pro source pin changed')
    if TARGET.exists():
        raise FileExistsError('Fresh immutable data identity required; preserve existing partials')
    capacity=os.statvfs(TARGET.parent)
    free=capacity.f_bavail*capacity.f_frsize
    if free<64*1024**2:
        raise RuntimeError('Insufficient free space for the bounded10MB public stage')
    TARGET.mkdir()
    probe=TARGET/'.write-probe'
    with probe.open('xb') as f:
        f.write(bytes(4096));f.flush();os.fsync(f.fileno())
    probe.unlink()
    partials, provenance = TARGET/'.download-partials', TARGET/'.provenance'
    partials.mkdir();provenance.mkdir()
    (provenance/'hf-tree.json').write_bytes(tree_raw)
    (provenance/'hf-info.json').write_bytes(info_raw)
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    config=dict(format='libero_pro_public_stage_download_v1',repository=REPO,revision=REVISION,
        source_commit=SOURCE,target=str(TARGET),hf_token=False,credentials_read=False,
        source_pretraining_data=False,public_target_data_only=True,
        expected_files=676,expected_bytes=10269485,maximum_elapsed_seconds=MAX_SECONDS,
        concurrency=WORKERS,maximum_attempts_per_file=RETRIES,request_timeout_seconds=REQUEST_SECONDS,
        free_bytes_before=free,write_probe_fsynced=True,
        metadata_sha256=dict(tree=sha(tree_raw),info=sha(info_raw)))
    write_new(TARGET/'download-config.json',config)
    print(json.dumps(dict(started=True,files=676,bytes=10269485,free_bytes=free,token=False)),flush=True)

    def download(ordinal,row):
        target=TARGET/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'+urllib.parse.quote(row['path'],safe='/')
        errors=[]
        for attempt in range(1,RETRIES+1):
            if time.monotonic()+REQUEST_SECONDS>=deadline:
                errors.append(dict(attempt=attempt,error='bounded_campaign_deadline'));break
            part=partials/f'{ordinal:04d}-attempt-{attempt}.part'
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'public-libero-pro-stage/1.0'})
                with part.open('xb') as f:
                    with urllib.request.urlopen(req,timeout=REQUEST_SECONDS) as response:
                        if urllib.parse.urlsplit(response.url).scheme!='https':
                            raise ValueError('Public download redirected away from HTTPS')
                        count=0
                        while True:
                            block=response.read(min(65536,row['size']+1-count))
                            if not block:break
                            f.write(block);count+=len(block)
                            if count>row['size']:
                                raise ValueError('Response exceeds pinned file size')
                    f.flush();os.fsync(f.fileno())
                data=part.read_bytes()
                if len(data)!=row['size']:
                    raise ValueError('Downloaded file size differs')
                digest, blob=sha(data),git_blob(data)
                if (digest!=row['lfs']['oid']) if row.get('lfs') else (blob!=row['oid']):
                    raise ValueError('Downloaded Git blob/LFS content identity differs')
                # Exclusive hard-link publication cannot overwrite another path.
                os.link(part,target);part.unlink()
                return dict(path=row['path'],size=len(data),sha256=digest,expected_git_oid=row['oid'],
                    verification='LFS SHA256 + exact reconstructed pointer Git blob' if row.get('lfs') else 'Git blob SHA1 + size',
                    downloaded_git_blob_oid=blob,lfs=row.get('lfs'),attempts=attempt,prior_errors=errors)
            except Exception as error:
                errors.append(dict(attempt=attempt,error_type=type(error).__name__,error=str(error)[:240],partial=str(part.relative_to(TARGET))))
                if attempt<RETRIES and time.monotonic()+REQUEST_SECONDS+attempt<deadline:
                    time.sleep(attempt)
        return dict(path=row['path'],failed=True,expected_size=row['size'],expected_git_oid=row['oid'],errors=errors)

    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(download,i,row) for i,row in enumerate(files)]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            if len(rows)%64==0:
                print(json.dumps(dict(completed=len(rows),failed=sum(r.get('failed',False) for r in rows))),flush=True)
    rows.sort(key=lambda r:r['path'])
    failed=[r for r in rows if r.get('failed')]
    receipt=dict(config,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,verified_files=len(rows)-len(failed),failed_files=len(failed),files=rows,
        partials_preserved=True,deserialization=False,models_tests_simulators_allocations=0)
    write_new(TARGET/'receipt.json',receipt)
    if failed:
        print(json.dumps(dict(status='partial_failure',failed_files=len(failed),receipt=str(TARGET/'receipt.json'))),flush=True)
        return 3
    pairs=[]
    for suite in SUITES:
        for kind in TYPES:
            family=f'{suite}_{kind}'
            bddls={p.stem:p for p in (TARGET/'bddl_files'/family).glob('*.bddl')}
            inits={p.name.removesuffix('.pruned_init'):p for p in (TARGET/'init_files'/family).glob('*.pruned_init')}
            if len(bddls)!=10 or set(bddls)!=set(inits):
                raise ValueError('Core family does not have its10 matched BDDL/init file pairs')
            pairs.extend(dict(family=family,bddl=str(bddls[k].relative_to(TARGET)),initialization=str(inits[k].relative_to(TARGET))) for k in sorted(bddls))
    if len(pairs)!=160:
        raise ValueError('Core160 public paired assets are incomplete')
    env_source=META/'repo/libero_ood/ood_environment.yaml'
    envfamilies=[suite+'_env' for suite in SUITES]
    if any((TARGET/k/family).exists() for k in ('bddl_files','init_files') for family in envfamilies):
        raise ValueError('Unexpected HF environment-family directory; recheck declared partial coverage')
    ready=dict(format='libero_pro_full_public_hf_stage_v1',status='verified_public_hf_files',
        dataset_repo=REPO,revision=REVISION,source_commit=SOURCE,public_files=676,public_bytes=10269485,
        receipt_sha256=sha((TARGET/'receipt.json').read_bytes()),metadata_sha256=config['metadata_sha256'],
        core_registered_family_keys=[s+'_'+k for s in SUITES for k in (*TYPES,'env')],
        core_available_pair_count=160,core_available_family_count=16,core_pairs=pairs,
        missing_environment_family_keys=envfamilies,missing_environment_declared_case_count=40,
        environment_declaration=dict(path=str(env_source),sha256=sha(env_source.read_bytes())),
        future_work='Official environment generation/staging is still required; no generation performed here',
        fully_runnable_fullsuite=False,full_suite_score_ready=False,source_pretraining_data=False,
        public_target_data_only=True,hf_token=False,raw_pickle_loading=False)
    write_new(TARGET/'READY.json',ready)
    print(json.dumps(dict(status=ready['status'],verified_files=676,verified_bytes=10269485,core_pairs=160,
        missing_environment_cases=40,fully_runnable_fullsuite=False,READY_sha256=sha((TARGET/'READY.json').read_bytes()),
        receipt_sha256=ready['receipt_sha256'],elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
