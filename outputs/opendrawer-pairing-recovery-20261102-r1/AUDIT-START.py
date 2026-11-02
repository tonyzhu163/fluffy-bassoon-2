"""Read-only early admission audit; run on allocated Linux CPU, not an outcome summary."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import runpy

import torch

ROOT = pathlib.Path('/mowenjie/fluffy-bassoon-2')
OUT = ROOT / 'outputs/opendrawer-pairing-recovery-20261102-r1'
CODE = ROOT / 'worktrees/opendrawer-paired-eval-20261102-r1'
CORE = '68f2c5aa33218709453a7821f6c6223ac3803035'
DRIVER = '63a173362461d206ad65f39a1134342c3537f195'
pairing = runpy.run_path(str(CODE / 'src/fluffy_bassoon/meta_rl/robocasa_pairing.py'))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def ref_read(ref):
    assert sha(ref['path']) == ref['sha256']
    return json.loads(pathlib.Path(ref['path']).read_text())


phases = (
    ('opendrawer-ad-first-id-ood-20261102-r2', 'ad-paired-r2', '29ea1d309ca1b896aab1f25abc656fbcfc04ebbd37d17392abdff3c865cffec7'),
    ('opendrawer-dpt-full-id-ood-20261102-r2', 'dpt-full-paired-r2', 'd4a6609f2e905c9a48261487512ee5b741d0d8bd54b9bdbbca087b3df44f7c28'),
    ('opendrawer-dpt-static-id-ood-20261102-r1', 'dpt-static-paired-r1', '7a7c77c4b65e7e0de6d0f2b4c5032170a28af1fca8525af3aaf01deebc23f9eb'),
)
report = dict(checked_at=datetime.now(timezone.utc).isoformat(), phases=[], outcome_summary=False)
all_uuids = []
for name, attempt, binding_sha in phases:
    root = ROOT / 'outputs' / name
    assert sha(root / 'binding.json') == binding_sha
    binding = json.loads((root / 'binding.json').read_text())
    assert binding['runtime_core_commit'] == CORE and binding['evaluation_driver_commit'] == DRIVER
    ref_read(binding['assembly'])
    for item in binding['selected'].values():
        assert sha(item['candidate']['path']) == item['candidate']['sha256']
        assert ref_read(item['status'])['converged']
    manifests = []
    for file in sorted((root / 'attempts' / attempt).glob('node-*/manifest.json')):
        manifest = json.loads(file.read_text())
        assert manifest['source'] == DRIVER and manifest['runtime_core'] == CORE
        assert manifest['binding']['sha256'] == binding_sha
        probe = json.loads((file.parent / 'gpu-probe.json').read_text())
        assert probe['status'] == 'passed' and len(probe['devices']) == 2
        uuids = [d['uuid'] for d in probe['devices']]
        assert len(set(uuids)) == 2
        all_uuids.extend(uuids)
        manifests.append(dict(node=manifest['node'], hostname=manifest['hostname'], gpu_uuids=uuids,
                              manifest_sha256=sha(file), probe_sha256=sha(file.parent / 'gpu-probe.json')))
    paths = sorted((root / 'episodes').rglob('result.json'))
    counts = Counter()
    for file in paths:
        row = json.loads(file.read_text())
        counts[(row['cohort'], row['method'], row['shot'], row['kind'])] += 1
    verified = []
    # Verify the first complete case available for every requested endpoint.
    for cohort in ('ID', 'OOD'):
        for method in binding['methods']:
            endpoints = [('K0', 'query')]
            if method.endswith('_full'):
                endpoints += [('K1', 'support'), ('K1', 'query')]
            for shot, kind in endpoints:
                choices = [p for p in paths if p.parts[-6] == cohort and p.parts[-4:-1] == (method, shot, kind)]
                if not choices:
                    continue
                file = choices[0]
                row = json.loads(file.read_text())
                candidate = (binding['initial'] if method == 'ppo_initial' else binding['teacher'] if method == 'ppo_final'
                             else binding['selected'][method]['candidate'])
                assert row['complete'] and row['candidate'] == candidate
                assert row['training_source'] == CORE and row['evaluation_source'] == DRIVER
                assert sha(row['records']['path']) == row['records']['sha256']
                truth = ref_read(row['truth'])
                assert truth['case'] == row['native_case']
                assert int(truth['simulator_final_success']) == row['success']
                assert row['horizon'] == (750 if cohort == 'ID' else 900)
                initial = row['initial_pairing']
                assert sha(initial['snapshot']['path']) == initial['snapshot']['sha256']
                canonical = ref_read(initial['canonical'])
                canonical_path = pathlib.Path(initial['canonical']['path']).parent / 'initial.pt'
                assert sha(canonical_path) == canonical['snapshot_sha256']
                comparison = pairing['compare_native_pair'](
                    torch.load(canonical_path, map_location='cpu', weights_only=False),
                    torch.load(initial['snapshot']['path'], map_location='cpu', weights_only=False))
                assert comparison == initial['report'] and comparison['matched']
                records = torch.load(row['records']['path'], map_location='cpu', weights_only=False)
                assert len(records) == row['horizon'] // 50
                assert float(records[-1]['reward']) == float(row['success']) and records[-1]['done']
                verified.append(dict(result=str(file), result_sha256=sha(file), canonical=initial['canonical'],
                                     comparison=comparison, simulator_truth_sha256=row['truth']['sha256'],
                                     records_terminal_feedback_verified=True))
    errors = []
    for log in (root / 'attempts' / attempt).rglob('*.log'):
        lines = log.read_text(errors='replace').splitlines()
        matches = [line for line in lines if 'Traceback (most recent call last)' in line or 'AssertionError' in line or 'ValueError:' in line]
        if matches:
            errors.append(dict(path=str(log), errors=matches))
    phase = dict(run=name, manifests=manifests, completed_episodes=len(paths),
                 completed_by_endpoint={'/'.join(k): v for k, v in sorted(counts.items())},
                 first_verified_native_endpoints=verified, worker_errors=errors)
    report['phases'].append(phase)
assert len(all_uuids) == len(set(all_uuids)), 'Overlapping allocation UUIDs'
report['unique_allocated_gpu_uuids'] = len(all_uuids)

train = ROOT / 'outputs/opendrawer-ppo-ad-dpt-20261102-r1'
recovery = json.loads((train / 'FULL-RECOVERY-20261102-1446.json').read_text())
assert recovery['all_four_converged'] and recovery['source'] == CORE
for item in recovery['recoveries']:
    directory = train / 'fits' / item['method']
    assert sha(directory / 'latest.pt') == item['latest_sha256']
    assert sha(directory / 'candidate.pt') == item['candidate_sha256']
legacy_journals = []
for path in sorted((train / 'accounting').rglob('*.json')):
    value = json.loads(path.read_text())
    calls = value.get('calls', [])
    legacy_journals.append(dict(path=str(path), sha256=sha(path), calls=len(calls),
        unresolved=sum(not call['resolved'] for call in calls),
        conservative_controls=sum(call.get('executed_or_conservative', call['requested']) for call in calls)))
report['training_closeout'] = dict(job='job-8kb761dxfvtm', scheduler_status='stopped',
    scheduler_created='2026-11-02T11:48:26+08:00', scheduler_running='2026-11-02T11:48:41+08:00',
    scheduler_terminal='2026-11-02T14:48:58+08:00', allocated_gpus=2,
    conservative_creation_to_terminal_gpu_seconds=21664,
    complete_legacy_episodes=len(list((train / 'evaluation').rglob('result.json'))),
    legacy_admissible_for_repaired_comparison=False, journals=legacy_journals,
    recoveries_preserved=True, selected_and_latest_hashes_unchanged=True)
(OUT / 'ADMISSION-20261102-1500.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(unique_GPU_UUIDs=len(all_uuids),
    phases=[dict(run=p['run'], nodes=len(p['manifests']), completed=p['completed_episodes'],
                 verified_endpoints=len(p['first_verified_native_endpoints']), errors=p['worker_errors']) for p in report['phases']],
    training_closeout=report['training_closeout']), indent=2))
