"""Restore official object files using the existing non-overwriting installer."""
import argparse
import hashlib
import importlib.util
import json
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
    assert not CACHE.exists()
    free = shutil.disk_usage(CACHE.parent).free
    assert free > 32*1024**3
    with tempfile.NamedTemporaryFile(dir=CACHE.parent, prefix='.sidney-assets-write-') as probe:
        probe.write(b'write probe\n'); probe.flush()
    CACHE.mkdir()
    archive = CACHE/'objects.zip'
    h = hashlib.sha256()
    url = f'https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REV}/objects.zip'
    print('Downloading pinned official objects archive', flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, archive.with_suffix('.part').open('xb') as target:
        while block := source.read(16*1024**2):
            target.write(block); h.update(block)
    partial = archive.with_suffix('.part')
    assert partial.stat().st_size == SIZE and h.hexdigest() == SHA
    partial.rename(archive)
    project = Path(__file__).resolve().parents[2]
    installer = module(project/'scripts/diagnostics/robotwin_install_objects.py', 'sidney_object_installer')
    receipt = installer.install(archive, ROBOTWIN)
    validator = module(project/'src/fluffy_bassoon/rl/robotwin_assets.py', 'sidney_static_assets')
    receipt.update(validation=validator.validate_robotwin_assets(ROBOTWIN, TASKS),
                   free_before_bytes=free, write_probe=True, gpu_hours=0,
                   quota='not exposed by PFS mount', model_imported=False)
    (args.output/'ASSETS-READY.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
