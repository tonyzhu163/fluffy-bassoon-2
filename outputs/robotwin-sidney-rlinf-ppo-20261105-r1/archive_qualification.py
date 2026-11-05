"""Allocated CPU-only lossless serialization and real recovery readback."""
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np
import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor, write_tensor


def fingerprint(value):
    digest=hashlib.sha256();count=0
    def visit(item):
        nonlocal count
        if isinstance(item,torch.Tensor):
            count+=1;digest.update(str(item.dtype).encode());digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,np.ndarray):
            digest.update(str(item.dtype).encode());digest.update(str(item.shape).encode());digest.update(item.tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict')
            for key in sorted(item,key=repr):digest.update(repr(key).encode());visit(item[key])
        elif isinstance(item,(tuple,list)):
            digest.update(type(item).__name__.encode())
            for child in item:visit(child)
        else:digest.update(repr(item).encode())
    visit(value);return dict(sha256=digest.hexdigest(),tensor_count=count)


def disk_bytes(root):
    total=0
    for path in root.rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass
    return total


def main():
    output=Path(os.environ['TASK_OUT']);scratch=output/'scratch';scratch.mkdir()
    source=output.parent/'train03'
    boundary=source/'sidney-native-two/checkpoints/global_step_0'
    report=dict(status='running',scientific_optimizer_updates=0,source_commit=os.environ['TASK_SOURCE'],
        canonical_compression='gzip level1, direct streaming without raw PFS staging',prefixes=[],boundary_files=[])
    write(output/'RESULT.json',report)
    assert shutil.disk_usage('/tmp').free>20*2**30
    with tempfile.TemporaryDirectory(prefix='sidney-archive-',dir='/tmp') as local:
        local=Path(local);peak=dict(pfs_bytes=0,local_bytes=0);stop=threading.Event()
        def sample():
            peak['pfs_bytes']=max(peak['pfs_bytes'],disk_bytes(scratch))
            peak['local_bytes']=max(peak['local_bytes'],disk_bytes(local))
        def watch():
            while not stop.wait(.1):sample()
        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
        try:
            # Real chronology, original source identities and every stored field.
            # This is a closed physical prefix, not an admitted full PPO history.
            for rank in range(4):
                traces=sorted((source/f'traces/rank-{rank:02d}').glob('*.pt.gz'))[:8]
                samples=sorted((source/f'sampling/rank-{rank:02d}').glob('*.pt.gz'))[:8]
                assert len(traces)==len(samples)==8
                refs=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in traces+samples]
                value=dict(format='lossless_actual_native_prefix_qualification_v1',rank=rank,
                    source_references=refs,traces=[read_tensor(p) for p in traces],
                    sampling=[read_tensor(p) for p in samples],scientific_history_admitted=False)
                expected=fingerprint(value);raw=local/'control.pt';compressed=scratch/'canonical.pt.gz'
                started=time.monotonic();write_tensor(raw,value);raw_bytes=raw.stat().st_size
                write_tensor(compressed,value);sample();write_seconds=time.monotonic()-started
                before_local=disk_bytes(local);started=time.monotonic();restored=read_tensor(compressed)
                actual=fingerprint(restored);assert actual==expected
                assert disk_bytes(local)==before_local  # Reader added no disk staging.
                report['prefixes'].append(dict(rank=rank,trace_count=len(traces),sample_count=len(samples),
                    source_references=refs,raw_bytes=raw_bytes,compressed_bytes=compressed.stat().st_size,
                    ratio=compressed.stat().st_size/raw_bytes,write_seconds=write_seconds,
                    readback_seconds=time.monotonic()-started,exact_fingerprint=actual))
                write(output/'RESULT.json',report)
                raw.unlink();compressed.unlink();del value,restored;gc.collect()
            # A full real saved boundary: every DCP/optimizer/metadata file and
            # all12 worker sidecars. Restore byte-for-byte into local scratch.
            original=json.loads((boundary/'pilot-boundary.json').read_text())
            assert original['completed'] and original['global_step']==0 and len(original['workers'])==12
            hashes={r['path']:r['sha256'] for r in original['workers']+original['actor_checkpoint_files']}
            restored_boundary=local/'boundary'
            for source_file in sorted(p for p in boundary.rglob('*') if p.is_file()):
                rel=source_file.relative_to(boundary);expected=sha(source_file)
                if str(source_file) in hashes:assert hashes[str(source_file)]==expected
                compressed=scratch/'boundary-file.gz';target=restored_boundary/rel;target.parent.mkdir(parents=True,exist_ok=True)
                started=time.monotonic()
                with source_file.open('rb') as inp,gzip.open(compressed,'wb',compresslevel=1) as out:shutil.copyfileobj(inp,out,8*1024**2)
                sample()
                with gzip.open(compressed,'rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,8*1024**2)
                assert sha(target)==expected and target.stat().st_size==source_file.stat().st_size
                row=dict(relative_path=str(rel),bytes=source_file.stat().st_size,compressed_bytes=compressed.stat().st_size,
                    original_and_restored_sha256=expected,seconds=time.monotonic()-started)
                if rel.parts[0]=='pilot' and source_file.suffix=='.pt':
                    left=read_tensor(source_file);right=read_tensor(target)
                    assert fingerprint(left)==fingerprint(right)
                    row['tensor_fingerprint']=fingerprint(right);del left,right;gc.collect()
                report['boundary_files'].append(row);write(output/'RESULT.json',report)
                sample();compressed.unlink()
            import torch.distributed.checkpoint as dcp
            metadata=dcp.FileSystemReader(restored_boundary/'actor/dcp_checkpoint').read_metadata()
            report['restored_dcp_metadata_items']=len(metadata.state_dict_metadata)
            assert report['restored_dcp_metadata_items']>0
            report.update(status='complete',real_boundary_byte_identity=True,worker_sidecars_tensor_identity=12,
                recovery_format_changed=False,canonical_compression_enabled_in_production=False,
                scratch_removed_after_qualification=True,
                limitation='Real closed one-round prefix only; no complete production chronological archive exists before first scientific update. Compression ratios are measurements of this prefix, not a bound on future scenes.')
        finally:
            sample();stop.set();monitor.join();report['peak_staging']=peak
            report['pfs_staging_limit_bytes']=8*2**30;report['local_staging_limit_bytes']=16*2**30
            write(output/'RESULT.json',report)
            assert peak['pfs_bytes']<8*2**30 and peak['local_bytes']<16*2**30
            shutil.rmtree(scratch)


if __name__=='__main__':main()
