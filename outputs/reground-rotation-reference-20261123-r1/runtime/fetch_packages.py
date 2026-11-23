"""Fetch the exact Linux distributions; verify the publisher digest before use."""
import argparse
import csv
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit


def fetch(record, destination, mirror):
    url = record['url']
    parts = urlsplit(url)
    path = destination / unquote(Path(parts.path).name)
    if not path.exists():
        sources = [url]
        if mirror and parts.hostname == 'files.pythonhosted.org':
            sources.insert(0, mirror.rstrip('/') + parts.path)
        partial = path.with_suffix(path.suffix + '.partial')
        for source in sources:
            result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2',
                                     '--connect-timeout', '12', '--speed-time', '60', '--speed-limit', '1024',
                                     '--output', str(partial), source])
            if result.returncode == 0:
                break
        else:
            raise RuntimeError(f'Cannot retrieve {path.name}')
        verify(partial, record['sha256'])
        partial.replace(path)
    else:
        verify(path, record['sha256'])
    print(path.name, flush=True)


def verify(path, digest):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != digest:
        raise ValueError(f'{path}: expected {digest}, read {actual}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('wheels', type=Path)
    p.add_argument('--pypi-mirror', help='Optional mirror base ending immediately before /packages')
    p.add_argument('--pypi-only', action='store_true', help='Stage PyPI while recovering PyTorch CDN downloads')
    args = p.parse_args()
    args.wheels.mkdir(parents=True, exist_ok=True)
    with Path(__file__).with_name('packages.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if args.pypi_only:
        rows = [row for row in rows if urlsplit(row['url']).hostname == 'files.pythonhosted.org']
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row: fetch(row, args.wheels, args.pypi_mirror), rows))
