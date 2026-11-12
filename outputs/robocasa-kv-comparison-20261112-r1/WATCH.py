"""Run the original independent audit and verified bank archival on Linux CPUs."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs') / HERE.name


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    assert sys.platform == 'linux'
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    packed = module('packed', HERE / 'PACKED-WAVES.py')
    old = HERE.parent
    audit = module('audit', old / 'robocasa-eight-hour-babysit-20261104-r1/AUDIT-ROLLOUT.py')
    reducer = module('reduce', old / 'robocasa-three-task-kv-velocity-meta-20261105-r1/REDUCE.py')
    packed.COLD.mkdir(parents=True, exist_ok=True)
    probe = packed.COLD / ('write-probe-' + str(time.time_ns()))
    probe.write_text('verified private archive access\n')
    assert probe.read_text() == 'verified private archive access\n'
    # Keep the tiny successful write/read probe as deployment evidence.
    packed.write(ROOT / 'monitor/WATCH-RECEIPT.json', dict(
        pid=os.getpid(), hostname=socket.gethostname(), code=str(HERE),
        started_unix=time.time(), archive_write_readback=str(probe),
        packed_archive_enabled=True, maximum_CPU_workers=4))
    pending, completed, failures = {}, set(), {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            for bank, future in list(pending.items()):
                if future.done():
                    try:
                        future.result()
                        completed.add(str(bank))
                    except Exception as error:
                        failures[str(bank)] = repr(error)
                    del pending[bank]
            for result in sorted(ROOT.glob('readouts/*/*/*/*/*/bank-*/RESULT.json')):
                bank = result.parent
                # Archival briefly renames raw data to bank-NNN.verified-hot-copy.
                # Only canonical bank names are independent rollout units.
                if re.fullmatch(r'bank-\d{3}', bank.name) is None:
                    continue
                if str(bank) not in completed and str(bank) not in failures and bank not in pending:
                    if (bank / packed.MARKER).exists():
                        completed.add(str(bank))
                    else:
                        pending[bank] = pool.submit(packed.archive_closed_bank, bank, audit.audit_result)
            summaries = []
            for group in sorted(ROOT.glob('readouts/*/*/*/RESULT.json')):
                if any(bank.is_relative_to(group.parent) for bank in pending):
                    continue
                summary = reducer.reduce_group(group.parent)
                if summary:
                    summaries.append(str(group.parent / 'SUMMARY.json'))
            packed.write(ROOT / 'monitor/HOT-STORAGE.json', dict(
                checked_unix=time.time(), packed_archive_enabled=not failures,
                current_hot_bytes=packed.hot_bytes(), free_PFS_bytes=shutil.disk_usage(ROOT).free,
                completed_archives=len(completed), pending_archives=len(pending),
                failures=failures, completed_summaries=summaries))
            if failures:
                raise RuntimeError(json.dumps(failures))
            if len(summaries) == 6 and not pending:
                packed.write(ROOT / 'monitor/COMPLETE.json', dict(
                    completed=True, checked_unix=time.time(), summaries=summaries,
                    audited_archived_banks=len(completed)))
                return
            time.sleep(15)


if __name__ == '__main__':
    main()
