"""Copy and check exactly the official lerobot/libero-assets dataset at ASSET_REV."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_cache import verify

ASSET_REV = '0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, help='Parent directory for the verified snapshot')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / ASSET_REV
    with Path(__file__).with_name('libero-assets.tsv').open() as fp:
        rows = list(csv.DictReader(fp, delimiter='\t'))
    with TemporaryDirectory(dir=args.destination) as staging:
        working = target if target.exists() else Path(staging) / 'assets'
        for item in rows:
            dest = working / item['path']
            if working != target:
                dest.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(args.source / item['path'], dest)
            meta = dict(oid=item['git_oid'], size=int(item['size']))
            if item['lfs_sha256']:
                meta.update(lfs=True, sha256=item['lfs_sha256'])
            verify(dest, meta)
        files = {f.relative_to(working).as_posix() for f in working.rglob('*') if f.is_file()}
        if files != {r['path'] for r in rows}:
            raise ValueError(f'{working}: unexpected asset files')
        if working != target:
            working.rename(target)
    print(json.dumps(dict(repository='lerobot/libero-assets', revision=ASSET_REV,
                          files=len(rows), bytes=sum(int(r['size']) for r in rows),
                          snapshot=str(target.resolve()))))


if __name__ == '__main__':
    main()
