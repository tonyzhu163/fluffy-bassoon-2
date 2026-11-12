"""Preserve a terminal teacher attempt in private BOS before releasing duplicate hot data."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-six-source-decision-20261112-r1')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / ROOT.name


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--terminal-job', required=True, help='Exact owned job confirmed terminal in the scheduler')
    parser.add_argument('--release-hot', action='store_true')
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.terminal_job.startswith('job-')
    source = args.source.resolve()
    relative = source.relative_to(ROOT)
    assert len(relative.parts) == 3 and relative.parts[0] == 'teachers'
    assert not args.source.is_symlink() and (source / 'EXIT.json').is_file()
    assert not (source / 'COLD-ARCHIVE.json').exists()
    files = sorted(p for p in source.rglob('*') if p.is_file())
    assert files and not any(p.is_symlink() for p in source.rglob('*'))
    work = Path(tempfile.mkdtemp(prefix='robocasa-teacher-archive-', dir='/tmp'))
    assert source.stat().st_dev != work.stat().st_dev
    size = sum(p.stat().st_size for p in files)
    assert shutil.disk_usage(work).free > size * 2
    began = time.time()
    inventory = [dict(relative=str(p.relative_to(source)), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    local = work / 'attempt.tar'
    with tarfile.open(local, 'w') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
    digest = sha(local)
    cold = COLD / relative.with_suffix('.tar')
    cold.parent.mkdir(parents=True, exist_ok=True)
    with local.open('rb') as incoming, cold.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    assert cold.stat().st_size == local.stat().st_size and sha(cold) == digest
    expected = {row['relative']: row for row in inventory}
    restored = set()
    with cold.open('rb') as incoming, tarfile.open(fileobj=incoming, mode='r|') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            assert member.name in expected and member.name not in restored
            row = expected[member.name]
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
            assert member.size == row['bytes']
            assert sha(source / member.name) == row['sha256']
            restored.add(member.name)
    assert restored == set(expected)
    receipt = dict(verified=True, terminal_job=args.terminal_job, source=str(source), cold=str(cold),
        archive_sha256=digest, archive_bytes=local.stat().st_size, payload_bytes=size, files=inventory,
        full_archive_SHA_readback=True, all_extracted_member_SHA_verified=True,
        all_original_source_SHA_reverified=True, checked_unix=time.time(), elapsed_seconds=time.time()-began,
        hot_released=False, restore_instruction='Extract the verified tar into a fresh directory, compare every inventory SHA, then explicitly restore original paths before any raw reader.')
    manifest = ROOT / 'archives' / relative / 'MANIFEST.json'
    manifest.parent.mkdir(parents=True, exist_ok=False)
    manifest.write_text(json.dumps(receipt, indent=2)+'\n')
    if args.release_hot:
        metadata = source.with_name(source.name + '.verified-metadata')
        quarantine = source.with_name(source.name + '.verified-duplicate')
        assert not metadata.exists() and not quarantine.exists()
        metadata.mkdir()
        for path in source.iterdir():
            if path.is_file() and (path.suffix == '.json' or path.name == 'console.log'):
                shutil.copyfile(path, metadata / path.name)
                assert sha(path) == sha(metadata / path.name)
        (metadata / 'COLD-ARCHIVE.json').write_text(json.dumps(dict(
            verified=True, manifest=str(manifest), cold=str(cold), archive_sha256=digest,
            raw_stageback_required=True), indent=2)+'\n')
        source.rename(quarantine)
        try:
            metadata.rename(source)
        except BaseException:
            quarantine.rename(source)
            raise
        # Only verified duplicate owned data are released. The private tar and
        # full inventory retain the complete checkpoint and recovery state.
        shutil.rmtree(quarantine)
        receipt['hot_released'] = True
        manifest.write_text(json.dumps(receipt, indent=2)+'\n')
    local.unlink()
    work.rmdir()
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
