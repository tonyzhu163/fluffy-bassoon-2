"""Recover compact code/results from the nine already completed BOS archives."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT = Path('/mowenjie')
BASE = ROOT / 'fluffy-bassoon-2/outputs/legacy-experiment-retention-audit-20261104-r2'
OLD = ROOT / 'fluffy-bassoon-2/outputs/experiment-bundles-cold-archive-20261104-r1'
TEXT = {'.md', '.py', '.sh', '.json', '.jsonl', '.csv', '.txt', '.log', '.yaml', '.yml', '.toml', '.ini', '.cfg'}


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    spec = importlib.util.spec_from_file_location('retention_audit', BASE / 'audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    plan = json.loads((BASE / 'PLAN.json').read_text())
    family = {r['source']: r['family'] for r in plan['sources']}
    cold = json.loads((OLD / 'SUMMARY.json').read_text())['completed']
    assert len(cold) == 9
    directory = BASE / 'cold-evidence'
    directory.mkdir(exist_ok=False)
    lists = BASE / 'private-cold-selection'
    lists.mkdir(exist_ok=False)
    selections = []
    for receipt in cold:
        assert all(receipt[k] for k in ('source_reclaimed', 'source_compare', 'bos_readback_sha256_verified'))
        source = Path(receipt['source'])
        assert not source.exists()
        key = str(source.relative_to(ROOT)).replace('/', '__')
        rows = json.loads((OLD / 'receipts' / (key + '.inventory.json')).read_text())
        selected = []
        for row in rows:
            relative = Path(row[0]).relative_to(source.relative_to(ROOT))
            if not stat.S_ISREG(row[1]) or row[2] > 16 << 20:
                continue
            if any(p in audit.SECRET for p in relative.parts) or audit.bulk_reason(relative, row[2]):
                continue
            if relative.suffix.lower() not in TEXT:
                continue
            if any(p.lower() in audit.RAW_CONTEXT for p in relative.parts[:-1]):
                continue
            is_code = relative.suffix.lower() in ('.py', '.sh') and not any(p in {'vendor', 'site-packages', '.venv', 'env', 'third_party'} for p in relative.parts)
            if len(relative.parts) <= 2 or audit.representative(relative, row[2]) or is_code:
                selected.append(row)
        target = directory / family[str(source)] / source.name
        target.mkdir(parents=True, exist_ok=False)
        names = lists / (key + '.names')
        names.write_bytes(b''.join(os.fsencode(row[0]) + b'\0' for row in selected))
        selections.append({'source': str(source), 'archive': receipt['archive'], 'archive_sha256': receipt['sha256'],
                           'destination': str(target), 'selection': str(names), 'records': selected,
                           'selected_bytes': sum(row[2] for row in selected)})
    save(BASE / 'COLD-RECOVERY-PLAN.json', [{k: v for k, v in r.items() if k != 'records'} for r in selections])

    def extract(selection):
        target = Path(selection['destination'])
        log = lists / (target.name + '.log')
        with log.open('xb') as stream:
            subprocess.run(['tar', '--ignore-zeros', '--no-recursion', '--no-same-owner', '--no-same-permissions',
                            '-I', 'zstd', '-xf', selection['archive'], '--strip-components=3', '-C', str(target),
                            '--null', '--verbatim-files-from', '-T', selection['selection']],
                           stdout=stream, stderr=stream, check=True)
        hashes = {}
        source = Path(selection['source']).relative_to(ROOT)
        for row in selection['records']:
            relative = Path(row[0]).relative_to(source)
            p = target / relative
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == row[2]
            hashes[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {'source': selection['source'], 'archive': selection['archive'],
                'archive_sha256': selection['archive_sha256'], 'destination': str(target),
                'records_recovered': len(hashes), 'recovered_bytes': selection['selected_bytes'],
                'file_sha256': hashes, 'full_archive_compression_integrity_passed': True}

    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(extract, selection): selection['source'] for selection in selections}
        for future in as_completed(futures):
            try:
                result = future.result()
                completed.append(result)
                print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), Path(result['source']).name,
                      result['records_recovered'], 'compact records recovered', flush=True)
            except Exception as error:
                errors.append({'source': futures[future], 'error': repr(error)})
            save(BASE / 'COLD-RECOVERY.json', {'finished': False, 'completed': completed, 'errors': errors})
    save(BASE / 'COLD-RECOVERY.json', {'finished': True, 'completed': completed, 'errors': errors,
                                     'recovered_bytes': sum(r['recovered_bytes'] for r in completed),
                                     'full_BOS_recovery_archives_retained': 9, 'GPU_jobs_launched': 0})


if __name__ == '__main__':
    main()
