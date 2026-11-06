"""Read-only aggregation of this scratch capacity/recovery benchmark."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path



def comparable(rows):
    result = deepcopy(rows)
    for row in result:
        row['actor'].pop('optimizer_sha256')
    return result


def summarize(root, recovery_root=None):
    files = {}

    def record(relative, *, base=None):
        path = (root if base is None else base) / relative
        if not path.exists():
            return None
        payload = path.read_bytes()
        files[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    probe = record('gpu-probe.json')
    profile = record('profile/RESULT.json')
    preserved = record('CHECKPOINT-PRESERVED.json')
    restore = record('restore/RESULT.json', base=recovery_root)
    actual = record('restore/RESTORE-ACTUAL.json', base=recovery_root)
    expected = record('profile/RECOVERY-EXPECTED.json')
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=None if profile is None else profile['source_commit'],
        recovery_source_commit=None if restore is None else restore['source_commit'],
        scientific_training=False, global_batch_size=2048,
        device_probe_passed=probe is not None and probe['status'] == 'passed' and len(probe['devices']) == 4,
        profile_status=None if profile is None else profile['status'],
        recovery_status=None if restore is None else restore['status'], candidates=[])
    for micro in (64, 128, 256, 512):
        rows = [record(f'profile/micro-{micro}/rank-{rank}.json') for rank in range(4)]
        if not any(row is not None for row in rows):
            continue
        complete = all(row is not None and row.get('status') == 'complete' for row in rows)
        item = dict(micro=micro, completed_ranks=sum(row is not None and row.get('status') == 'complete' for row in rows),
            updates_per_rank=[0 if row is None else len(row['updates']) for row in rows],
            errors=[None if row is None else row.get('error') for row in rows], complete=complete)
        if complete:
            assert all(row['rank'] == rank and row['micro'] == micro and row['global_batch_size'] == 2048
                       and row['accumulation'] == 512 // micro and len(row['updates']) == 4
                       for rank, row in enumerate(rows))
            item.update(memory_safe=all(row['memory_safe'] for row in rows),
                frozen_unchanged=all(row['frozen_unchanged'] for row in rows),
                same_partition_replay_exact=all(row['same_partition_ratio_one'] for row in rows),
                steady_update_seconds=max(row['steady_median_seconds'] for row in rows),
                reference_seconds=max(row['reference_seconds'] for row in rows),
                minimum_free_gib=min(update['conservative_minimum_free_bytes'] for row in rows for update in row['updates']) / 2**30,
                peak_reserved_gib=max(update['peak_reserved_bytes'] for row in rows for update in row['updates']) / 2**30,
                peak_allocated_gib=max(update['peak_allocated_bytes'] for row in rows for update in row['updates']) / 2**30)
            # One selected-partition reference before a five-epoch minibatch.
            item['reference_amortized_update_seconds'] = item['steady_update_seconds'] + item['reference_seconds'] / 5
            if micro == 64:
                item['initial_native_chunk_likelihood_passed'] = all(row['initial_native_likelihood_replay']['passed'] for row in rows)
                item['initial_max_abs_chunk_log_ratio'] = max(row['initial_native_likelihood_replay']['max_abs_chunk_log_ratio'] for row in rows)
        report['candidates'].append(item)
    recovery = (preserved is not None and preserved['status'] == 'complete'
        and preserved['cold_full_readback'] and preserved['local_full_stageback']
        and restore is not None and restore['status'] == 'complete'
        and restore.get('full_recovery_exact') and restore.get('next_update_exact')
        and actual is not None and expected is not None
        and comparable(actual['before']) == comparable(expected['before'])
        and comparable(actual['next_update']) == comparable(expected['next_update']))
    report['independent_recovery_comparison_exact'] = bool(recovery)
    eligible = [row for row in report['candidates'] if row['complete'] and row['memory_safe']
                and row['frozen_unchanged'] and row['same_partition_replay_exact']]
    initial = next((row for row in report['candidates'] if row['micro'] == 64), {})
    qualified = report['device_probe_passed'] and initial.get('initial_native_chunk_likelihood_passed') and recovery
    report['largest_qualified_micro'] = max((row['micro'] for row in eligible), default=None) if qualified else None
    report['fastest_qualified_micro'] = min(eligible, key=lambda row: row['reference_amortized_update_seconds'])['micro'] if qualified and eligible else None
    report['scope'] = 'Real 64 distinct observations/rank repeated to512 for systems measurements; no live simulators or scientific learning/GAE qualification. Capacity is for measured candidates and actual partition, with max(8GiB,8%) reserve.'
    report['timing_scope'] = 'Three steady scratch updates after one warmup. Reference measured separately, amortized over five epochs; exact-repeat audit and checkpoint I/O excluded from steady update timing.'
    report['files_sha256'] = files
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--recovery-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.recovery_root), indent=2, allow_nan=False))
