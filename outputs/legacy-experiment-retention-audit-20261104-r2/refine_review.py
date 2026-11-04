"""Finalize reviewed payload classes and protect cited historical endpoints."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/legacy-experiment-retention-audit-20261104-r2')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    audit_path = BASE / 'AUDIT.json'
    original = BASE / 'AUDIT-INITIAL.json'
    assert not original.exists() and not (BASE / 'REVIEW.json').exists()
    original.write_bytes(audit_path.read_bytes())
    audit = read(audit_path)
    assert audit['complete'] and all(not r.get('errors') and not r.get('protected') for r in audit['sources'])
    spec = importlib.util.spec_from_file_location('refined_classifier', BASE / 'audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protected = []
    compressed = Counter()
    for entry in audit['sources']:
        if entry['status'] != 'audited_hot_source':
            continue
        path = Path(entry['inventory'])
        inventory = read(path)
        name = Path(entry['source']).name
        cited_robotwin = any(s in name for s in ('robotwin-trained-continuation', 'robotwin-wholetrial', 'robotwin-staged'))
        latest = defaultdict(set)
        if cited_robotwin:
            for row in inventory['rows']:
                parts = Path(row[0]).parts
                for index, part in enumerate(parts):
                    match = re.fullmatch(r'slice-(\d+)', part)
                    if match:
                        latest['/'.join(parts[:index])].add(int(match.group(1)))
        selected_slices = {key: set(sorted(values)[-2:]) for key, values in latest.items()}
        for row in inventory['rows']:
            if not stat.S_ISREG(row[1]):
                continue
            relative = Path(row[0])
            parts = relative.parts
            reason = module.bulk_reason(relative, row[2])
            if row[7] == 'keep' and reason == 'retired compressed raw RPC or trajectory snapshot':
                row[7:9] = ['bulk', reason]
                compressed[entry['family']] += row[4] * 512
            # Retain compact context/RNG records from actual evaluation banks.
            evaluation = any(p.lower() in {'eval', 'evals', 'evaluation', 'evaluations', 'validation', 'test', 'dev'} for p in parts[:-1])
            keep_evaluation = (evaluation and row[2] <= 2 << 20 and relative.suffix.lower() in {'.npz', '.npy', '.pt', '.pkl'}
                               and any(p in relative.stem.lower() for p in ('rng', 'context', 'initial_state', 'seed'))
                               and not set(parts).intersection({'corpus', 'replay', 'buffer'}))
            # The published FULL/RESET negative result needs its terminal model
            # state, not every intermediate slice or cached corpus policy.
            keep_endpoint = False
            if cited_robotwin and relative.suffix.lower() == '.pt' and relative.name.endswith('recovery.pt') and 'corpus' not in parts and 'optimizer-calibration' not in parts:
                slice_info = next(((i, re.fullmatch(r'slice-(\d+)', p)) for i, p in enumerate(parts) if re.fullmatch(r'slice-(\d+)', p)), None)
                if slice_info:
                    index, match = slice_info
                    keep_endpoint = int(match.group(1)) in selected_slices['/'.join(parts[:index])] and 'stages' not in parts
                else:
                    keep_endpoint = 'stages' not in parts
            if row[7] == 'bulk' and (keep_evaluation or keep_endpoint):
                explanation = 'compact evaluation context/RNG record' if keep_evaluation else 'terminal cited FULL/RESET recovery; latest two slices cover an incomplete last slice'
                row[7:9] = ['keep', explanation]
                protected.append({'source': entry['source'], 'relative': row[0], 'bytes': row[2], 'reason': explanation})
        save(path, inventory)
        bulk = [r for r in inventory['rows'] if r[7] == 'bulk']
        keep = [r for r in inventory['rows'] if r[7] == 'keep' and stat.S_ISREG(r[1])]
        types = Counter()
        for row in bulk:
            types[Path(row[0]).suffix.lower()] += row[4] * 512
        entry.update(bulk_files=len(bulk), bulk_allocated_bytes=sum(r[4] for r in bulk) * 512,
                     bulk_logical_bytes=sum(r[2] for r in bulk), bulk_by_extension=dict(types),
                     retained_files=len(keep), retained_allocated_bytes=sum(r[4] for r in keep) * 512,
                     retained_large_files=[{'relative': r[0], 'bytes': r[2]} for r in keep if r[2] > 16 << 20])
    audit.update(bulk_files=sum(r.get('bulk_files', 0) for r in audit['sources']),
                 bulk_allocated_bytes=sum(r.get('bulk_allocated_bytes', 0) for r in audit['sources']),
                 retained_allocated_bytes=sum(r.get('retained_allocated_bytes', 0) for r in audit['sources']),
                 reviewed_shanghai=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    save(audit_path, audit)
    review = {'bulk_prune_review_complete': True, 'audit_sha256': sha(audit_path),
              'initial_audit_sha256': sha(original), 'plan_sha256': sha(BASE / 'PLAN.json'),
              'classifier_sha256': sha(BASE / 'audit.py'),
              'basis': 'Human requested audit followed by deletion of bulky one-time artifacts. Existing Findings retain DSRL controls and negative LIBERO/RoboTwin outcomes; configurations, logs, compact evaluation records, unknown code and cited terminal recoveries stay.',
              'reported_robomimic_recovery_zip_files_preserved': True,
              'compressed_raw_payload_allocation_bytes': dict(compressed),
              'protected_cited_endpoints_and_evaluation_records': protected,
              'new_whole_bundle_archives': 0, 'existing_BOS_full_archives_retained': 9,
              'no_active_folder_refresh': True, 'GPU_jobs_launched': 0}
    save(BASE / 'REVIEW.json', review)
    print(json.dumps({'bulk_GiB': audit['bulk_allocated_bytes'] / (1 << 30), 'bulk_files': audit['bulk_files'],
                      'retained_GiB': audit['retained_allocated_bytes'] / (1 << 30), 'protected_compact_records': len(protected)}, indent=2))


if __name__ == '__main__':
    main()
