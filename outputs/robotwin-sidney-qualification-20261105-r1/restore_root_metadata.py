"""Restore the pinned root object metadata, preserving the broken link."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-objects-20261105-r1')
    archive = cache/'objects.zip'
    with archive.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == '6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27'
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read('objects/same.json')
    expected = '3bcaa1618ff9d0b50478dd22183f77ca3557d9d5406442b17e39337e68cb5edd'
    assert len(data) == 5865 and hashlib.sha256(data).hexdigest() == expected
    assert isinstance(json.loads(data), dict)
    target = cache/'root-metadata-3bcaa161/same.json'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        assert target.read_bytes() == data
    else:
        with target.open('xb') as stream:
            stream.write(data)
    original = Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin/assets/objects/same.json')
    receipt = dict(file=str(original), restored_target=str(target), bytes=len(data),
                   sha256=expected, gpu_hours=0, model_imported=False)
    if original.exists():
        assert original.read_bytes() == data
        receipt['preserved_existing'] = True
    else:
        if original.is_symlink():
            old_target = os.readlink(original)
            (args.output/'original-same-json-link').symlink_to(old_target)
            receipt['original_target'] = old_target
        staged = original.with_name('.same.json.sidney-new-link')
        staged.symlink_to(target)
        os.replace(staged, original)
        receipt['restored'] = True
    assert original.read_bytes() == data
    (args.output/'ROOT-METADATA-READY.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
