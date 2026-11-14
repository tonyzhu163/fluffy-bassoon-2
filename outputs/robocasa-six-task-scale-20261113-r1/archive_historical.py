"""Retire the closed original comparison from PFS after verified cold archival.

Reuses the previously qualified exact-tar/source comparison and BOS readback.
No active teacher, initializer, environment or corrected corpus is in scope.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


NAMES = (
    'robocasa-dit-diagnostic-20261113-r1',
    'robocasa-kv-comparison-20261112-r1',
    'robocasa-three-task-kv-velocity-meta-20261105-r1',
    'robocasa-three-task-meta-baseline-20261104-r2',
    'robocasa-flow-sde-10m-per-task-20261103-r1',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[2]
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    spec = importlib.util.spec_from_file_location('historical_archive',
        code / 'outputs/storage-audit-baige-20261104-r1/archive.py')
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    arch.RUN = 'robocasa-original-headline-retention-20261114-r1'
    arch.AUDIT = arch.ROOT / 'outputs' / arch.RUN
    arch.BOS = Path('/mnt/bos/adgfyl/strong-plant-archives') / arch.RUN
    arch.STAGE = Path('/tmp') / arch.RUN
    assert shutil.which('zstd') and shutil.which('tar')
    for path in (arch.AUDIT, arch.BOS, arch.STAGE):
        path.mkdir(exist_ok=False)
        probe = path / '.write-probe'
        probe.write_bytes(b'archive'); assert probe.read_bytes() == b'archive'; probe.unlink()
    before = shutil.disk_usage(arch.ROOT)._asdict()
    plan = dict(source_commit=args.pin, selected=list(NAMES),
        user_instruction='Keep the old headline; archive the rest and delete if wrong',
        dependency_review='All five closed; active source6 pin6c0ef27a uses none. No new labels/fits reuse original130 data. Other development monitors refer to a different project.',
        invalidity='Earlier critic-mismatch data excluded from new corpora; historical numerical observations are not declared fabricated or corrupt.',
        policy='Preserve complete archive, compact audit and restore pointer; reclaim hot copy only after exact tar comparison and full BOS SHA256 readback.',
        started_unix=time.time())
    arch.save(arch.AUDIT / 'PLAN.json', plan)
    completed, failures = [], []
    for name in NAMES:
        try:
            completed.append(arch.archive('outputs/' + name))
        except Exception as error:
            failures.append(dict(source=name, error=repr(error)))
            print(f'PRESERVED {name}: {error!r}', flush=True)
        result = dict(plan, completed=completed, failures=failures,
            finished=False, filesystem_before=before,
            filesystem_now=shutil.disk_usage(arch.ROOT)._asdict(),
            reclaimed_allocated_bytes=sum(r['source_allocated_bytes'] for r in completed))
        arch.save(arch.AUDIT / 'SUMMARY.json', result)
    result.update(finished=True, ended_unix=time.time())
    arch.save(arch.AUDIT / 'SUMMARY.json', result)
    arch.save(arch.BOS / 'SUMMARY.json', result)
    print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
