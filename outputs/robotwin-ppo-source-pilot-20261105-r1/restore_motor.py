"""Restore a verified cold Motus cache, without importing a model or simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ARCHIVE = Path('/mnt/bos/adgfyl/strong-plant-archives/storage-audit-baige-20261104-r1/legacy-models/fluffy-bassoon-2__cache__robotwin.tar.zst')
EXPECTED = '11a3e40fbb4103e8011a64dd20fe1f2ffef20b1969f50e1cc8558272bef3097b'
PREFIX = 'fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1'
DEST = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert not DEST.exists(), 'Use a fresh cache destination'
    assert ARCHIVE.stat().st_size == 12653848348
    free = shutil.disk_usage(DEST.parent).free
    assert free >= 96 * 1024**3, 'Reserve restoration, pilot outputs and concurrent headroom'
    with tempfile.NamedTemporaryFile(dir=DEST.parent, prefix='.robotwin-write-probe-') as probe:
        probe.write(b'write probe\n'); probe.flush()
    DEST.mkdir()
    # BOS FUSE returned a read error to zstd after a successful Python hash
    # pass. Stream and hash onto PFS before letting tar/zstd read it locally.
    local_archive = DEST / 'verified-cache.tar.zst'
    print('Copying and verifying archived cache on PFS', flush=True)
    h = hashlib.sha256()
    with ARCHIVE.open('rb') as inp, local_archive.open('xb') as out:
        for block in iter(lambda: inp.read(16 * 1024**2), b''):
            out.write(block); h.update(block)
    assert local_archive.stat().st_size == ARCHIVE.stat().st_size
    assert h.hexdigest() == EXPECTED, 'Archive copy SHA256 differs'
    print('Verified archive copy; extracting common motor', flush=True)
    subprocess.run(['tar', '--keep-old-files', '-I', 'zstd', '-xf', str(local_archive),
                    '--strip-components=3', '-C', str(DEST), PREFIX], check=True)
    model = DEST / 'pi05-motus-effc6e1a-r1'
    norm = model / 'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert (model / 'model.safetensors').is_file()
    assert digest(norm) == '1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb'
    files = [{'path': str(p.relative_to(model)), 'bytes': p.stat().st_size,
              'sha256': digest(p)} for p in sorted(model.rglob('*')) if p.is_file()]
    record = dict(status='restored', ended_unix=time.time(), archive=str(ARCHIVE),
                  archive_sha256=EXPECTED, archive_bytes=ARCHIVE.stat().st_size,
                  verified_pfs_archive=str(local_archive),
                  model_path=str(model), norm_stats_path=str(norm), files=files,
                  available_before_bytes=free, quota='not exposed by the PFS mount',
                  write_probe=True, gpu_hours=0, model_loaded=False)
    (args.output / 'MOTOR-READY.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
