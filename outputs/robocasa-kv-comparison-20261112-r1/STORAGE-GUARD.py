"""Capacity and archive health for the explicitly migrated local-scratch readout."""
import json
import os
from pathlib import Path
import shutil
import time


def snapshot():
    root = Path(os.environ['KV_LOCAL_ROOT'])
    assert root.is_relative_to('/tmp/robocasa-kv-comparison-20261112-r1')
    heartbeat = json.loads((root / 'monitor/HOT-STORAGE.json').read_text())
    free = shutil.disk_usage(root).free
    return dict(admitted=bool(free >= 96 * 2**30
        and heartbeat['packed_archive_enabled']
        and time.time() - heartbeat['checked_unix'] < 120
        and heartbeat['current_hot_bytes'] < 12 * 2**30),
        storage='allocated-container-local-scratch', free_bytes=free,
        minimum_free_GiB=96, hot_bytes=heartbeat['current_hot_bytes'],
        archive_errors=heartbeat.get('errors', []),
        shared_PFS_raw_growth_bytes=0)
