"""Archive only audited closed waves; keep metadata hot and stage raw bytes back explicitly."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-kv-comparison-20261112-r1')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / ROOT.name
MARKER = 'PACKED-ARCHIVE.json'
METADATA = ('RESULT.json', 'AUDIT.json', 'FITTED-DEPLOYMENT.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def hot_bytes():
    """Count this run's hot readouts and packing spool; never follow cold links."""
    total = 0
    for root in (ROOT / 'readouts', ROOT / 'archives'):
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
            for name in files:
                path = Path(directory) / name
                try:
                    if not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    pass  # Another owned archive may just have atomically closed.
    return total


def archive_closed_bank(bank, audit_result):
    bank = Path(bank)
    if bank.is_symlink():
        return
    relative = bank.relative_to(ROOT)
    if relative.parts[0] not in ('readouts', 'qualification-fitted-pfs03',
            'qualification-fitted-oodfull-pfs04', 'qualification-fitted-reset-pfs04'):
        raise ValueError('Archive only owned closed rollout evidence')
    if (bank / MARKER).exists():
        marker = json.loads((bank / MARKER).read_text())
        assert marker['verified'] and sha(bank / 'RESULT.json') == marker['result_sha256']
        return
    result = json.loads((bank / 'RESULT.json').read_text())
    assert result['optimizer_updates'] == 0 and result['native_parameters_frozen']
    if not (bank / 'AUDIT.json').exists():
        audit_result(bank / 'RESULT.json', bank / 'AUDIT.json')
    audit = json.loads((bank / 'AUDIT.json').read_text())
    assert audit['passed'] and audit['result_sha256'] == sha(bank / 'RESULT.json')
    work = ROOT / 'archives' / relative / 'packed'
    manifest = work / 'RESULT.json'
    if not manifest.exists():
        command = [sys.executable, '-B', str(Path(__file__).with_name('PACK-CLOSED.py')),
            '--bank', str(bank), '--output', str(work), '--cold', str(COLD / 'packed-waves' / relative.with_suffix('.tar'))]
        child = subprocess.run(command, capture_output=True, text=True)
        if child.returncode:
            raise RuntimeError('Packed archive failed; all hot evidence retained: ' + child.stderr[-2000:])
    saved = json.loads(manifest.read_text())
    assert saved['passed'] and saved['source'] == str(bank)
    assert sha(Path(saved['cold'])) == saved['archive_sha256']
    expected = {row['relative'] for row in saved['files']}
    assert {str(path.relative_to(bank)) for path in bank.rglob('*') if path.is_file()} == expected
    for row in saved['files']:
        assert sha(bank / row['relative']) == row['sha256']
    # No raw reader is admitted after the closure/audit marker. The reducer
    # reads these unchanged hot metadata files; raw inspection requires restore.
    metadata = bank.with_name(bank.name + '.packed-metadata')
    quarantine = bank.with_name(bank.name + '.verified-hot-copy')
    assert not metadata.exists() and not quarantine.exists()
    metadata.mkdir()
    for name in METADATA:
        if (bank / name).exists():
            shutil.copyfile(bank / name, metadata / name)
            assert sha(metadata / name) == sha(bank / name)
    marker = dict(verified=True, format='audited_packed_wave_v1', source=str(bank),
        manifest=str(manifest), archive_sha256=saved['archive_sha256'], cold=saved['cold'],
        result_sha256=sha(bank / 'RESULT.json'), raw_stageback_required=True,
        restore_command=f'{sys.executable} {Path(__file__)} --restore {manifest} --work <fresh-PFS-directory>',
        raw_reader_admission='None after audit; explicitly restore before inspecting raw references',
        original_metadata_paths_and_bytes_preserved=True)
    write(metadata / MARKER, marker)
    bank.rename(quarantine)
    try:
        metadata.rename(bank)
        assert sha(bank / 'RESULT.json') == marker['result_sha256']
    except BaseException:
        if bank.exists():
            shutil.rmtree(bank)
        quarantine.rename(bank)
        raise
    shutil.rmtree(quarantine)
    # These are verified redundant spool copies. The immutable cold object,
    # complete member manifest and hot metadata remain authoritative.
    (work / 'wave.tar').unlink(missing_ok=True)
    if (work / 'restored').exists():
        shutil.rmtree(work / 'restored')
    write(work.parent / 'RECEIPT.json', dict(completed=True, packed=True,
        payload_bytes=saved['payload_bytes'], files=len(saved['files']), manifest=str(manifest),
        archive_sha256=saved['archive_sha256'], cold=saved['cold'],
        raw_stageback_required=True, original_metadata_paths_and_bytes_preserved=True,
        all_raw_bytes_preserved_and_stageback_SHA_verified=True, finished_unix=time.time()))


def restore(manifest, work):
    saved = json.loads(manifest.read_text())
    bank = Path(saved['source'])
    assert bank.is_relative_to(ROOT) and work.is_relative_to(ROOT)
    marker = json.loads((bank / MARKER).read_text())
    assert marker['verified'] and marker['raw_stageback_required']
    cold = Path(saved['cold'])
    assert cold.is_relative_to(COLD) and sha(cold) == saved['archive_sha256'] == marker['archive_sha256']
    work.mkdir(parents=True, exist_ok=False)
    with cold.open('rb') as incoming, tarfile.open(fileobj=incoming, mode='r|') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            path = work / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('xb') as output:
                shutil.copyfileobj(source, output, 8 << 20)
    assert {str(path.relative_to(work)) for path in work.rglob('*') if path.is_file()} == {row['relative'] for row in saved['files']}
    for row in saved['files']:
        assert (work / row['relative']).stat().st_size == row['bytes']
        assert sha(work / row['relative']) == row['sha256']
    assert sha(work / 'RESULT.json') == marker['result_sha256']
    marker.update(raw_stageback_required=False, restored_unix=time.time(),
        raw_reader_admission='Explicitly restored; retain this hot copy for the requesting reader')
    write(work / MARKER, marker)
    previous = bank.with_name(bank.name + '.metadata-before-stageback')
    assert not previous.exists()
    bank.rename(previous)
    try:
        work.rename(bank)
    except BaseException:
        previous.rename(bank)
        raise
    shutil.rmtree(previous)
    return dict(restored=True, source=str(bank), files=len(saved['files']),
        all_original_raw_paths_and_bytes_restored=True, manifest=str(manifest))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(restore(args.restore, args.work)))
