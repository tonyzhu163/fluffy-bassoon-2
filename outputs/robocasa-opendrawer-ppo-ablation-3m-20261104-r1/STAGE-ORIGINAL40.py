"""Stage only the original 3.072M policy for the common OpenDrawer evaluation."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import zstandard

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
RUN = ROOT / 'robocasa-opendrawer-ppo-ablation-3m-20261104-r1'
SOURCE = ROOT / 'robocasa-flow-sde-10m-per-task-20261103-r1/OpenDrawer/production-r1/OpenDrawer/checkpoints/global_step_40'
EXPECTED = 'a0d5c35ef3b609785ea903405fa07437a510edc9d3c9b20d2fbf74e18c457f63'
ARCHIVE = 'd15e4279b14d8954308eb4b19be9a52f106592958396917afd0123e818a9c6b6'


def main():
    cold = json.loads((SOURCE / 'COLD-STORAGE.json').read_text())
    boundary = json.loads((SOURCE / 'pilot-boundary.json').read_text())
    assert cold['full_BOS_readback'] and cold['tar_compare'] and cold['sha256'] == ARCHIVE
    assert boundary['completed'] and boundary['global_step'] == 40
    original = SOURCE / 'actor/model_state_dict/full_weights.pt'
    assert next(row['sha256'] for row in boundary['actor_checkpoint_files'] if row['path'] == str(original)) == EXPECTED
    target = RUN / 'eval-input-original40'
    target.mkdir(exist_ok=False)
    assert shutil.disk_usage(target).free > 230 << 30
    archive = target / 'archive.partial'
    h = hashlib.sha256()
    with archive.open('xb') as out:
        for attempt in range(5):
            try:
                with Path(cold['archive']).open('rb', buffering=0) as inp:
                    inp.seek(out.tell())
                    while out.tell() < cold['archive_bytes']:
                        chunk = inp.read(1 << 20)
                        if not chunk:
                            raise IOError('Short read')
                        out.write(chunk)
                        h.update(chunk)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(3)
    assert h.hexdigest() == ARCHIVE and archive.stat().st_size == cold['archive_bytes']
    member = 'global_step_40/actor/model_state_dict/full_weights.pt'
    with archive.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            weight = next(m for m in tar if m.name == member and m.isfile())
            with tar.extractfile(weight) as inp, (target/'full_weights.partial').open('xb') as out:
                shutil.copyfileobj(inp, out, 4 << 20)
    with (target/'full_weights.partial').open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == EXPECTED
    path = target/'full_weights.pt'
    (target/'full_weights.partial').rename(path)
    archive.unlink()
    (target/'READY.json').write_text(json.dumps(dict(completed=True, step=40, path=str(path),
        sha256=EXPECTED, original_boundary=str(SOURCE/'pilot-boundary.json'), archive_sha256=ARCHIVE),indent=2)+'\n')


if __name__ == '__main__':
    main()
