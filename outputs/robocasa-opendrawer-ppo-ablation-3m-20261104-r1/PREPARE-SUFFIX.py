"""Restore the step-zero suffix critic and verify it shares the Stage 1 actor."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import torch
import zstandard

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
RUN = ROOT / 'robocasa-opendrawer-ppo-ablation-3m-20261104-r1'
ORIGINAL = ROOT / 'robocasa-flow-sde-10m-per-task-20261103-r1/OpenDrawer/q1-r1/OpenDrawer/checkpoints/global_step_0'
PREFIX = RUN / 'input-prefix-r1'
TARGET = RUN / 'input-suffix-r1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert shutil.disk_usage(RUN).free >= 230 * (1 << 30)
    TARGET.mkdir(exist_ok=True)
    assert not any(TARGET.iterdir()), "Only resume preparation from an empty directory"
    cold = json.loads((ORIGINAL / 'COLD-STORAGE.json').read_text())
    boundary = json.loads((ORIGINAL / 'pilot-boundary.json').read_text())
    assert boundary['completed'] and boundary['global_step'] == 0
    assert cold['full_BOS_readback'] and cold['tar_compare']
    original_archive = Path(cold['archive'])
    archive = RUN / 'staged-OpenDrawer-step0.tar.zst'
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with partial.open('xb') as out:
            while out.tell() < cold['archive_bytes']:
                start = out.tell()
                for retry in range(4):
                    try:
                        with original_archive.open('rb', buffering=0) as inp:
                            inp.seek(out.tell())
                            while out.tell() < cold['archive_bytes']:
                                chunk = inp.read(1 << 20)
                                if not chunk:
                                    raise IOError('Unexpected end of BOS archive')
                                out.write(chunk)
                        break
                    except OSError:
                        if retry == 3:
                            raise
                        print('Retrying BOS read at offset', out.tell(), flush=True)
                        time.sleep(3)
                assert out.tell() > start
        assert sha(partial) == cold['sha256']
        partial.rename(archive)
    assert sha(archive) == cold['sha256']
    checkpoint = TARGET / 'global_step_0'
    files = {Path(row['path']).relative_to(ORIGINAL): row['sha256'] for row in boundary['actor_checkpoint_files']}
    with archive.open('rb') as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as data:
        with tarfile.open(fileobj=data, mode='r|') as tar:
            for member in tar:
                path = Path(member.name)
                assert not path.is_absolute() and '..' not in path.parts and path.parts[0]=='global_step_0'
                rel = path.relative_to('global_step_0')
                if rel not in files:
                    continue
                assert member.isfile()
                dest = checkpoint / rel
                dest.parent.mkdir(exist_ok=True, parents=True)
                with dest.open('xb') as output:
                    shutil.copyfileobj(tar.extractfile(member), output, 16 << 20)
    assert len(files) == 4 and all(sha(checkpoint / rel) == digest for rel, digest in files.items())
    same = PREFIX / 'global_step_0'
    prefix_states = json.loads((same / 'pilot-boundary.json').read_text())
    shutil.copytree(same / 'pilot', checkpoint / 'pilot')
    workers = []
    for item in prefix_states['workers']:
        path = checkpoint / 'pilot' / Path(item['path']).name
        assert sha(path) == item['sha256']
        workers.append(dict(path=str(path), sha256=item['sha256']))
    name = 'actor/model_state_dict/full_weights.pt'
    a,b = (torch.load(loc/name, mmap=True, map_location='cpu', weights_only=True) for loc in (same,checkpoint))
    assert set(a) == set(b)
    heads, nonhead = {}, 0
    for k in a:
        if k.startswith('value_head.'):
            heads[k] = [list(a[k].shape), list(b[k].shape)]
        else:
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
            assert torch.equal(a[k].reshape(-1).view(torch.uint8), b[k].reshape(-1).view(torch.uint8)), k
            nonhead += 1
    assert nonhead and any(shapes[0] != shapes[1] for shapes in heads.values())
    boundary['workers'] = workers
    boundary['actor_checkpoint_files'] = [dict(path=str(checkpoint/p),sha256=h,bytes=(checkpoint/p).stat().st_size) for p,h in files.items()]
    boundary['assembly'] = dict(native_suffix_step0=str(ORIGINAL), rng_and_cursors=str(same),
        nonhead_tensors_equal=nonhead, head_shapes_prefix_suffix=heads)
    marker = checkpoint/'pilot-boundary.json'
    marker.write_text(json.dumps(boundary,indent=2)+'\n')
    reference = json.loads((PREFIX/'BEFORE.json').read_text())
    reference.update(checkpoint=str(checkpoint), checkpoint_boundary_sha256=sha(marker))
    out=TARGET/'BEFORE.json'
    out.write_text(json.dumps(reference,indent=2)+'\n')
    (TARGET/'PREPARED.json').write_text(json.dumps(dict(completed=True, checkpoint=str(checkpoint),
        archive_sha256=cold['sha256'], nonhead_tensors_bitwise_equal=nonhead, head_shapes_prefix_suffix=heads,
        copied_worker_sidecars=len(workers), before_reference=str(out), before_reference_sha256=sha(out),
        controls=0, SGD=0),indent=2)+'\n')


if __name__ == '__main__':
    main()
