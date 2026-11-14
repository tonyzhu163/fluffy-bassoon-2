"""Audit the saved SFT / fixed PPO30 panels, retaining missing endpoints.

CPU-only on Linux. The archive metadata directory is populated by a selected
member extraction with every member checked against its verified inventory.
No model selection, additional episodes, or training occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

TASKS = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug',
         'PickPlaceCounterToStove', 'TurnOffStove', 'TurnOnSinkFaucet')
SFT_SHA = '661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
NORM_SHA = '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Source:
    def __init__(self, root):
        self.root, self.refs = root, []
        inv = root / 'VERIFIED-INVENTORY.json'
        self.inventory = read(inv) if inv.exists() else None
        self.files = {r['path']: r for r in self.inventory['files']} if self.inventory else None
        if self.inventory:
            assert sha(self.inventory['receipt']) == self.inventory['receipt_sha256']
            original = read(self.inventory['receipt'])
            assert original['files'] == self.inventory['files'] and original['verified']

    def get(self, name):
        p = self.root / name
        digest = sha(p)
        if self.files is not None:
            assert name in self.inventory['metadata_verified']
            assert digest == self.files[name]['sha256']
        self.refs.append(dict(path=str(p), sha256=digest))
        return read(p)

    def recorded_payload(self, name, digest):
        if self.files is not None:
            assert self.files[name]['sha256'] == digest
        else:
            assert sha(self.root / name) == digest


def panel(source, task, phase, step, eval_seed):
    rows, canonical = {}, []
    image_diagnostics = []
    for rank in range(8):
        prefix = f'environment/rank-{rank:02d}/{phase}-i{step:03d}-round00'
        data = source.get(prefix + '-episodes.json')
        initial = source.get(prefix + '-initial.json')
        bank = source.get(f'canonical-eval/rank-{rank:02d}/complete.json')
        assert data['completed'] and data['rank'] == rank and data['iteration'] == step
        assert data['phase'] == phase and len(data['rows']) == 8
        assert data['initial'] == initial['initial']
        report = initial['report']
        assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
        assert not report['observations_modified']
        source.recorded_payload(prefix + '-initial.pt', data['initial']['sha256'])
        source.recorded_payload(f'canonical-eval/rank-{rank:02d}/initial.pt', bank['snapshot_sha256'])
        expected_seeds = list(range(eval_seed + rank * 8, eval_seed + (rank + 1) * 8))
        assert bank['case'] == dict(task=task, seeds=expected_seeds, split='target', horizon=600)
        assert [row['seed'] for row in data['rows']] == expected_seeds
        canonical.append(bank)
        image_diagnostics.append(dict(rank=rank, images_exact=report['images_exact'], images=report['images']))
        for row in data['rows']:
            assert row['episode_id'] == f"{task}/{row['seed']}"
            assert row['episode_id'] not in rows
            counts = row['native_counts']
            assert counts['controls'] - counts['reset_controls'] == 600
            assert isinstance(row['success_once'], bool)
            rows[row['episode_id']] = row['success_once']
    assert len(rows) == 64
    return rows, canonical, image_diagnostics


def paired(before, after, seed):
    import numpy as np
    assert before.keys() == after.keys()
    differences = np.array([int(after[k]) - int(before[k]) for k in sorted(before)])
    wins, losses = int((differences == 1).sum()), int((differences == -1).sum())
    counts = np.array([losses, 64 - wins - losses, wins])
    draws = np.random.default_rng(seed).multinomial(64, counts / 64, size=10000)
    interval = np.quantile((draws[:, 2] - draws[:, 0]) * 100 / 64, [.025, .975])
    return dict(cases=64, SFT_successes=sum(before.values()), PPO30_successes=sum(after.values()),
        SFT_percent=100 * sum(before.values()) / 64, PPO30_percent=100 * sum(after.values()) / 64,
        delta_percentage_points=100 * (wins - losses) / 64, paired_wins=wins, paired_losses=losses,
        paired_ties=64 - wins - losses, paired_bootstrap95_percentage_points=interval.tolist(),
        bootstrap_seed=seed, bootstrap_draws=10000, intervals_simultaneous=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--faucet', type=Path, required=True)
    p.add_argument('--label-inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert sys.platform == 'linux'
    teacher_refs = {}
    for path in args.label_inputs:
        teacher_refs.update({task: row['teacher'] for task, row in read(path)['tasks'].items()})
    results = {}
    for ordinal, task in enumerate(TASKS):
        faucet = task == 'TurnOnSinkFaucet'
        after = Source(args.faucet / 'attempt-02' if faucet else args.metadata / task)
        before = Source(args.faucet / 'attempt-01') if faucet else (
            Source(args.metadata / (task + '-sft')) if task == 'PickPlaceCounterToStove' else after)
        contract = after.get('flow-contract.json')
        manifest = after.get('manifest.json' if faucet else 'MANIFEST.json')
        assert manifest['model_sha256'] == SFT_SHA and manifest['norm_sha256'] == NORM_SHA
        assert contract['task'] == task and contract['horizon'] == 600
        assert contract['iterations'] == (40 if faucet else 30)
        assert contract['eval_cases'] == 64 and contract['eval_interval'] == 10
        assert contract['value_after_vlm'] and contract['denoising_steps'] == 10
        boundary_name = f'{task}/checkpoints/global_step_30/pilot-boundary.json'
        boundary = after.get(boundary_name)
        assert boundary['completed'] and boundary['global_step'] == 30
        assert sha(after.root / boundary_name) == teacher_refs[task]['boundary_sha256']
        if faucet:
            export_path = Path(teacher_refs[task]['path']).parent / 'EXPORT.json'
            export = read(export_path)
            assert export['passed'] and export['CPU_roundtrip_exact'] and export['all_finite']
            assert export['label_policy_step'] == 30 and not export['includes_step40']
            assert export['source_boundary_sha256'] == teacher_refs[task]['boundary_sha256']
            assert export['sha256'] == teacher_refs[task]['sha256'] == sha(teacher_refs[task]['path'])
            after.refs.append(dict(path=str(export_path), sha256=sha(export_path)))
            reference_path = args.faucet / 'RESUME-REFERENCE-01.json'
            reuse = after.get('before-evaluation.json')['reused_reference']
            assert reuse['sha256'] == sha(reference_path)
            reference = read(reference_path)
            assert reference['original_before_sha256'] == sha(before.root / 'before-evaluation.json')
            for ref in reference['episode_refs']:
                assert sha(ref['path']) == ref['sha256']
            before.refs.append(dict(path=str(reference_path), sha256=sha(reference_path)))
        else:
            weight = next(row for row in boundary['actor_checkpoint_files'] if row['path'].endswith('/full_weights.pt'))
            assert weight['sha256'] == teacher_refs[task]['sha256']
        original = before.get('before-evaluation.json')
        assert original['completed']
        b, bcanon, bimages = panel(before, task, 'before', 0, contract['eval_seed'])
        assert sum(b.values()) / 64 == original['metrics']['success_once']
        result = dict(task=task, PPO_teacher_step=30, teacher=teacher_refs[task],
            source_contract=contract, SFT_successes=sum(b.values()), SFT_cases=64,
            SFT_percent=sum(b.values()) * 100 / 64, fixed_endpoint_selected_before_outcomes=True,
            physical_and_public_initial_states_exact=True)
        endpoint_name = 'evaluations/step-030.json' if faucet else 'after-evaluation.json'
        if not (after.root / endpoint_name).exists():
            assert task == 'OpenDrawer'
            exit_record = after.get('operator-checkpoint-exit.json')
            assert exit_record['completed'] and exit_record['global_step'] == 30
            result.update(status='PPO30 evaluation missing', reason=exit_record,
                paired_result=None, cases=[dict(case=k, SFT=b[k], PPO30=None) for k in sorted(b)])
        else:
            endpoint = after.get(endpoint_name)
            assert endpoint['completed'] and endpoint['global_step'] == 30
            phase = 'eval_step_030' if faucet else 'after'
            a, acanon, aimages = panel(after, task, phase, 30, contract['eval_seed'])
            assert acanon == bcanon, 'Both policies must restore the same archived canonical states'
            assert sum(a.values()) / 64 == endpoint['metrics']['success_once']
            result.update(status='saved paired panel verified', paired_result=paired(b, a, 20261114 + ordinal),
                image_diagnostics=dict(SFT=bimages, PPO30=aimages),
                cases=[dict(case=k, SFT=b[k], PPO30=a[k]) for k in sorted(b)])
        result['evidence'] = before.refs + ([] if before is after else after.refs)
        result['archive_receipts'] = [dict(path=s.inventory['receipt'], sha256=s.inventory['receipt_sha256'])
                                     for s in ([after] if before is after else [before, after]) if s.inventory]
        results[task] = result
        print(json.dumps(dict(task=task, status=result['status'], paired=result['paired_result'])), flush=True)
    output = dict(audit_completed=True, all_six_PPO30_panels_complete=False, checked_unix=time.time(),
        audit_script_sha256=sha(__file__), GPU_hours=0, results=results,
        limitations=['64 existing canonical cases per task, one training seed; descriptive marginal paired intervals.',
            'OpenDrawer training/checkpoint completed, but its final evaluation was skipped at orderly checkpoint exit.',
            'Faucet uses its saved step30 evaluation, never the step40 after panel.',
            'No pooled six-task result while one endpoint is missing.',
            'These source-teacher panels are distinct from the fresh transformer ID/OOD comparison.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
