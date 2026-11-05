"""Forward PFS capacity: protected floor plus unmaterialized reservations once."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

GIB = 2 ** 30
ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1'


def allocated_bytes(paths):
    seen = set()
    rows = []
    for root in paths:
        amount = 0
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [n for n in subdirs if not (Path(directory) / n).is_symlink()]
            for name in files:
                path = Path(directory) / name
                try:
                    value = path.lstat()
                except FileNotFoundError:
                    continue
                identity = (value.st_dev, value.st_ino)
                if stat.S_ISREG(value.st_mode) and identity not in seen:
                    seen.add(identity)
                    amount += value.st_blocks * 512
        rows.append(amount)
    return rows


def snapshot(policy_path=None):
    path = Path(policy_path or os.environ['STEERING_STORAGE_ACCOUNTING'])
    policy = json.loads(path.read_text())
    assert policy['format'] == 'remaining_shared_PFS_reservations_v1'
    assert policy['protected_floor_GiB'] == 220
    assert policy['other_owner_contingency_GiB'] == 26
    assert policy['readout_hot_cap_GiB'] == 40
    plan_path = Path(policy['robotwin_plan'])
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == policy['robotwin_plan_sha256']
    plan = json.loads(plan_path.read_text())
    active = [r for r in plan['storage_reservations'] if not r.get('closed')]
    expected = {str(ROOT / 'outputs/robotwin-sidney-matched-baselines-20261106-r1' / n)
                for n in ('ppo02', 'grpo01')}
    assert {r['path'] for r in active} == expected
    assert all(r['maximum_gib'] == 135 for r in active)
    used = allocated_bytes([Path(r['path']) for r in active])
    reservations = []
    for row, amount in zip(active, used):
        cap = row['maximum_gib'] * GIB
        assert amount <= cap, 'RoboTwin storage envelope exceeded'
        reservations.append(dict(path=row['path'], cap_bytes=cap,
                                 allocated_bytes=amount, remaining_bytes=cap - amount))
    # Include readouts AND packing/staging spool. These bytes already reduce
    # filesystem free capacity and therefore must not be reserved a second time.
    hot = sum(allocated_bytes([RUN / 'readouts', RUN / 'archives']))
    remaining_hot = max(0, 40 * GIB - hot)
    required = (220 + 26) * GIB + remaining_hot + sum(r['remaining_bytes'] for r in reservations)
    volume = os.statvfs(ROOT)
    free = volume.f_bavail * volume.f_frsize
    return dict(checked_unix=time.time(), free_bytes=free, required_free_bytes=required,
                margin_bytes=free - required, admitted=free >= required,
                protected_floor_bytes=220 * GIB, other_owner_contingency_bytes=26 * GIB,
                readout_hot_allocated_bytes=hot, readout_hot_remaining_bytes=remaining_hot,
                robotwin=reservations, policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require_capacity():
    value = snapshot()
    if not value['admitted']:
        raise ValueError('Insufficient forward PFS capacity: ' + json.dumps(value))
    return value
