"""Keep native scratch DCP and sampler/RNG sidecars off the shared hot volume."""
import json
import os
from pathlib import Path
import shutil
import time

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write


def verify(root, manifest):
    files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                   and p.name != 'MANIFEST.json')
    assert files == sorted(manifest['files'])
    for relative, receipt in manifest['files'].items():
        path = root/relative
        assert path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256']


def preserve(checkpoint, local):
    started = time.monotonic()
    cold, output = Path(os.environ['TASK_COLD']),Path(os.environ['TASK_OUT'])
    for rank in range(4):
        shutil.copyfile(local/f'fixture-rank-{rank}.pt.gz',checkpoint/f'fixture-rank-{rank}.pt.gz')
    manifest = dict(files={str(p.relative_to(checkpoint)):dict(bytes=p.stat().st_size,sha256=sha(p))
                          for p in sorted(checkpoint.rglob('*')) if p.is_file()},
                    source_commit=os.environ['TASK_SOURCE'],optimizer_steps=4)
    assert sum(r['bytes'] for r in manifest['files'].values()) <= 32*2**30
    write(checkpoint/'MANIFEST.json',manifest)
    assert not cold.exists()
    shutil.copytree(checkpoint,cold)
    verify(cold,manifest)
    roundtrip = local/'roundtrip'
    shutil.copytree(cold,roundtrip)
    verify(roundtrip,manifest)
    assert sha(cold/'MANIFEST.json') == sha(roundtrip/'MANIFEST.json') == sha(checkpoint/'MANIFEST.json')
    shutil.rmtree(roundtrip)  # Exact retained copies remain local and cold.
    write(output/'CHECKPOINT-PRESERVED.json',dict(status='complete',cold=str(cold),
        manifest_sha256=sha(cold/'MANIFEST.json'),data_files=len(manifest['files']),
        bytes=sum(r['bytes'] for r in manifest['files'].values()),cold_full_readback=True,
        local_full_stageback=True,seconds=time.monotonic()-started))


def stage(target):
    cold, output = Path(os.environ['TASK_COLD']),Path(os.environ['TASK_OUT'])
    receipt = json.loads((output/'CHECKPOINT-PRESERVED.json').read_text())
    assert receipt['status'] == 'complete' and receipt['cold'] == str(cold)
    assert sha(cold/'MANIFEST.json') == receipt['manifest_sha256']
    manifest = json.loads((cold/'MANIFEST.json').read_text())
    assert not target.exists()
    shutil.copytree(cold,target)
    verify(target,manifest)
    assert sha(target/'MANIFEST.json') == receipt['manifest_sha256']
