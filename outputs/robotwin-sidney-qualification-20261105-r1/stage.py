"""Download immutable public Sidney weights; performs no model computation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

REPO = 'SidneyXie/pi05_robotwin'
REVISION = 'e49e2ab6c11f07511573b67261bd129e88d0a416'
DEST = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-qualification-20261105-r1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert not DEST.exists(), 'Use a fresh model cache destination'
    free = shutil.disk_usage(DEST.parent).free
    assert free > 64 * 1024**3
    with tempfile.NamedTemporaryFile(dir=DEST.parent, prefix='.sidney-write-probe-') as probe:
        probe.write(b'write probe\n'); probe.flush()
    DEST.mkdir()
    manifest = json.loads((Path(__file__).parent / 'HF-MANIFEST.json').read_text())
    assert manifest['sha'] == REVISION
    files = []
    for row in manifest['siblings']:
        name = row['rfilename']
        assert '/' not in name and name not in ('.', '..')
        expected = row.get('lfs', {}).get('sha256')
        target = DEST / name
        h = hashlib.sha256()
        print(json.dumps(dict(event='download', file=name, bytes=row['size'])), flush=True)
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=120) as inp, target.with_suffix(target.suffix + '.part').open('xb') as out:
            while block := inp.read(16 * 1024**2):
                out.write(block); h.update(block)
        partial = target.with_suffix(target.suffix + '.part')
        assert partial.stat().st_size == row['size'], name
        actual = h.hexdigest()
        if expected:
            assert actual == expected, name
        else:
            # Git blob identity for the small files is supplied by the pinned API.
            blob = hashlib.sha1(b'blob ' + str(row['size']).encode() + b'\0' + partial.read_bytes()).hexdigest()
            assert blob == row['blobId'], name
        partial.rename(target)
        files.append(dict(path=name, bytes=row['size'], sha256=actual))
        print(json.dumps(dict(event='verified', **files[-1])), flush=True)
    receipt = dict(status='verified', repo=REPO, revision=REVISION,
                   model_path=str(DEST), files=files, gpu_hours=0,
                   available_before_bytes=free, write_probe=True,
                   quota='not exposed by PFS mount', model_loaded=False,
                   ended_unix=time.time())
    (args.output / 'MOTOR-READY.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
