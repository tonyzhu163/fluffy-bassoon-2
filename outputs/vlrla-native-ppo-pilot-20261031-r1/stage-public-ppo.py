"""Download one pinned public RLinf checkpoint; no model imports or credentials."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = 'RLinf/RLinf-Pi05-RoboTwin-PPO-adjust_bottle'
REVISION = '1223d8b9c387eb22c51350900d842d0f3b86cf14'
ROOT = Path('/mowenjie/fluffy-bassoon-2/inputs/pi05-robotwin-ppo-adjust-bottle-1223d8b9-20261031-r1')
LIMIT_SECONDS = 1800
MAX_NEW_BYTES = 16 * 1024 ** 3
STARTED = time.monotonic()
LOCK = threading.Lock()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def event(value):
    with LOCK, (ROOT / 'download-events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(elapsed_seconds=time.monotonic() - STARTED, **value)) + '\n')


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'rlinf-public-checkpoint-staging/1.0', **(headers or {})}), timeout=60)


def file_hash(path, *, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(row):
    relative = PurePosixPath(row['rfilename'])
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe public repository path')
    final = ROOT.joinpath(*relative.parts)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + '.download-partial')
    expected = int(row['size'])
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/' + urllib.parse.quote(str(relative), safe='/')
    for attempt in range(3):
        try:
            if time.monotonic() - STARTED >= LIMIT_SECONDS:
                raise TimeoutError('Staging walltime ceiling')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise ValueError('Partial exceeds pinned size')
            if offset < expected:
                with request(url, {'Range': f'bytes={offset}-'} if offset else {}) as response:
                    if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('Public server did not honor exact partial resume')
                    with partial.open('ab' if partial.exists() else 'xb') as stream:
                        while block := response.read(8 * 1024 * 1024):
                            if time.monotonic() - STARTED >= LIMIT_SECONDS or stream.tell() + len(block) > expected:
                                raise TimeoutError('Pinned file size/walltime ceiling')
                            stream.write(block)
            if partial.stat().st_size != expected:
                raise ValueError('Downloaded byte count differs')
            digest = file_hash(partial)
            if row.get('lfs'):
                if digest != row['lfs']['sha256']:
                    raise ValueError('Published LFS SHA256 differs')
                blob_verified = None
            else:
                blob_verified = file_hash(partial, git_blob=True) == row['blobId']
                if not blob_verified:
                    raise ValueError('Published Git blob OID differs')
            if final.exists():
                raise ValueError('Fresh checkpoint destination already exists')
            partial.rename(final)
            result = dict(path=str(relative), size=expected, sha256=digest, published_blob_oid=row['blobId'],
                published_lfs_sha256=row.get('lfs', {}).get('sha256'), git_blob_oid_verified=blob_verified)
            event(dict(kind='file_complete', **result))
            print(json.dumps(dict(kind='file_complete', path=str(relative), size=expected)), flush=True)
            return result
        except Exception as error:
            # Do not emit signed redirect URLs or auth-bearing exception text.
            event(dict(kind='attempt_failed', path=str(relative), attempt=attempt + 1,
                error_type=type(error).__name__, http_status=getattr(error, 'code', None),
                retained_partial_bytes=partial.stat().st_size if partial.exists() else 0))
            if attempt == 2 or time.monotonic() - STARTED >= LIMIT_SECONDS:
                raise RuntimeError(f'Public checkpoint staging incomplete: {relative}; partial bytes retained') from None
            time.sleep(2 * (attempt + 1))


ROOT.parent.mkdir(parents=True, exist_ok=True)
free = os.statvfs(ROOT.parent).f_bavail * os.statvfs(ROOT.parent).f_frsize
if free < 500 * 1024 ** 3 + 128 * 1024 ** 3 + 64 * 1024 ** 3:
    raise RuntimeError('Require free floor plus concurrent benchmark and this phase storage reserves')
ROOT.mkdir(exist_ok=False)
probe = ROOT / '.write-probe'
with probe.open('xb') as stream:
    stream.write(b'public-checkpoint-staging-write-probe\n')
probe.unlink()
with request(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true') as response:
    metadata = json.load(response)
if metadata['sha'] != REVISION or metadata.get('private') or metadata.get('gated'):
    raise ValueError('Require the exact public ungated checkpoint revision')
rows = metadata['siblings']
if len(rows) != 17 or sum(row['size'] for row in rows) > MAX_NEW_BYTES:
    raise ValueError('Pinned public checkpoint inventory/storage ceiling differs')
write_json(ROOT / 'hf-metadata.json', metadata)
write_json(ROOT / 'STAGING.json', dict(repo=REPO, revision=REVISION, token=False,
    requested_files=len(rows), requested_bytes=sum(row['size'] for row in rows),
    maximum_new_bytes=MAX_NEW_BYTES, maximum_seconds=LIMIT_SECONDS, max_workers=3,
    initial_free_bytes=free, write_probe_passed=True,
    scope='Public download and byte verification only; no model/tensor deserialization or source-training admission'))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    verified = list(pool.map(download, rows))
norm = next(row for row in verified if row['path'] == 'physical-intelligence/robotwin/norm_stats.json')
receipt = dict(format='rlinf_public_checkpoint_staging_v1', completed=True, repo=REPO, revision=REVISION,
    token=False, files=verified, file_count=len(verified), total_bytes=sum(row['size'] for row in verified),
    robotwin_normalization=norm, elapsed_seconds=time.monotonic()-STARTED, model_constructed=False,
    source_pretraining_admission=False, initial_sft_checkpoint_untouched=True)
write_json(ROOT / 'READY.json', receipt)
print(json.dumps(dict(kind='complete', root=str(ROOT), ready_sha256=file_hash(ROOT / 'READY.json'),
    total_bytes=receipt['total_bytes'], normalization_sha256=norm['sha256'])), flush=True)
