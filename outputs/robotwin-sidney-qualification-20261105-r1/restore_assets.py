"""Restore official object files using the existing non-overwriting installer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

REV = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
SHA = '6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27'
SIZE = 3737778549
CACHE = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-objects-20261105-r1')
ROBOTWIN = Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin')
TASKS = ('handover_mic', 'pick_diverse_bottles', 'place_a2b_left',
         'place_object_stand', 'place_phone_stand', 'scan_object',
         'adjust_bottle', 'move_stapler_pad', 'place_a2b_right',
         'place_dual_shoes', 'place_empty_cup', 'stamp_seal')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    free = shutil.disk_usage(CACHE.parent).free
    assert free > 32*1024**3
    with tempfile.NamedTemporaryFile(dir=CACHE.parent, prefix='.sidney-assets-write-') as probe:
        probe.write(b'write probe\n'); probe.flush()
    CACHE.mkdir(exist_ok=True)
    archive = CACHE/'objects.zip'
    url = f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/objects.zip'
    if not archive.exists():
        print('Downloading pinned official objects archive', flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, archive.with_suffix('.part').open('xb') as target:
            while block := source.read(16*1024**2):
                target.write(block)
        archive.with_suffix('.part').rename(archive)
    assert archive.stat().st_size == SIZE
    with archive.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == SHA
    objects = ROBOTWIN/'assets/objects'
    manifest = json.loads((objects/'.fb2-archive-manifest.json').read_text())
    assert manifest['archive_sha256'] == SHA and manifest['hf_revision'] == REV
    assert not (objects/'.fb2-install.lock').exists()
    families = sorted({Path(row['path']).parts[0] for row in manifest['files']})
    backups = args.output/'link-backups'
    backups.mkdir()
    link_changes = []
    for name in families:
        current = objects/name
        if not current.is_symlink() or current.exists():
            continue
        old_target = os.readlink(current)
        backup_target = Path(old_target) if Path(old_target).is_absolute() else current.parent/old_target
        (backups/name).symlink_to(backup_target)
        destination = CACHE/f'restored-families-{args.output.name}'/name
        destination.mkdir(parents=True, exist_ok=False)
        row = dict(path=str(current), original_target=old_target,
                   backup=str(backups/name), replacement_target=str(destination), status='planned')
        link_changes.append(row)
        journal = args.output/'LINK-REPAIRS.json'
        journal.write_text(json.dumps(link_changes, indent=2)+'\n')
        assert current.is_symlink() and not current.exists() and os.readlink(current) == old_target
        staged = current.with_name(f'.{name}.sidney-new-link')
        staged.symlink_to(destination)
        os.replace(staged, current)
        row['status'] = 'repaired'
        journal.write_text(json.dumps(link_changes, indent=2)+'\n')
        print(json.dumps(dict(event='retargeted_dangling_family', family=name)), flush=True)
    project = Path(__file__).resolve().parents[2]
    installer = module(project/'scripts/diagnostics/robotwin_install_objects.py', 'sidney_object_installer')
    receipt = installer.install(archive, ROBOTWIN)
    validator = module(project/'src/fluffy_bassoon/rl/robotwin_assets.py', 'sidney_static_assets')
    receipt.update(validation=validator.validate_robotwin_assets(ROBOTWIN, TASKS),
                   repaired_dangling_family_links=link_changes,
                   free_before_bytes=free, write_probe=True, gpu_hours=0,
                   quota='not exposed by PFS mount', model_imported=False)
    (args.output/'ASSETS-READY.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
