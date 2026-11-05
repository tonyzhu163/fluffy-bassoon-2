"""Three prospectively fixed paired endpoint tests; reuse the qualified native B8 worker."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
EVAL = ROOT / 'sources/opendrawer-ablation-3m-eval-r3'
LABELS = ('SFT', 'teacher')
SEEDS = dict(PickPlaceCounterToCabinet=73220000, CoffeeSetupMug=73250000, TurnOnSinkFaucet=73280000)


def order(bank):
    return list(LABELS if bank % 2 == 0 else reversed(LABELS)) + (['repeat'] if bank == 0 else [])


def configure(task):
    sys.path.insert(0, str(EVAL / 'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1'))
    import matched_endpoint as native
    native.CASE_BASE, native.RNG_BASE = SEEDS[task], SEEDS[task] + 10000000
    native.LABELS, native.order = LABELS, order
    return native


def prepare(task, manifest):
    import os
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_contract, flow_seeds
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    native = configure(task)
    assert not manifest.exists()
    budget = native.read(RUN / 'SOURCE5-BUDGET.json')
    assert budget['approved'] and budget['independent_endpoint_gate_authorized']
    assert set(budget['source_gate']['tasks']) == set(SEEDS)
    previous = native.read(ROOT / 'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/EVAL-MANIFEST.json')
    initial = copy.deepcopy(previous['policies']['SFT'])
    assert initial['step'] == 0 and initial['prefix'] and initial['ignore_last']
    if task == 'CoffeeSetupMug':
        source = ROOT / 'outputs/robocasa-flow-sde-10m-per-task-20261103-r1/CoffeeSetupMug/production-r1'
        step, prefix, ignore_last = 130, False, False
    else:
        binding = RUN / 'source5' / task / 'ENDPOINT-SOURCE.json'
        if binding.exists():
            bound = native.read(binding)
            assert bound['task'] == task and bound['step'] == 40 and bound['fixed_endpoint'] and not bound['DEV_selection']
            source = Path(bound['path'])
        else:
            source = RUN / 'source5' / task / 'attempt-01'
        step, prefix, ignore_last = 40, True, True
    source_cfg_path = source / 'resolved-config.json'
    source_cfg = native.read(source_cfg_path)
    assert flow_contract(source_cfg)['task'] == task
    assert native.read(source / 'progress.json')['completed_iterations'] == step
    assert native.read(source / 'after-evaluation.json')['completed']
    folder = source / task / 'checkpoints' / f'global_step_{step}'
    marker = folder / 'pilot-boundary.json'
    boundary = native.read(marker)
    assert boundary['completed'] and boundary['global_step'] == step
    weight = folder / 'actor/model_state_dict/full_weights.pt'
    expected = next(row['sha256'] for row in boundary['actor_checkpoint_files'] if row['path'] == str(weight))
    teacher = dict(step=step, path=str(weight), sha256=expected, marker=str(marker),
        marker_sha256=native.sha(marker), prefix=prefix, ignore_last=ignore_last)
    # The already-qualified common eval factory starts with the prefix critic.
    # Coffee's suffix-critic teacher is loaded in its own native architecture by
    # MatchedRollout; neither critic is used to select inference actions.
    cfg = copy.deepcopy(source_cfg)
    cfg['actor']['model']['openpi']['value_after_vlm'] = True
    cfg['actor']['model']['openpi']['ignore_last'] = True
    cfg['rollout']['model'].setdefault('openpi', {})['value_after_vlm'] = True
    cfg['rollout']['model']['openpi']['ignore_last'] = True
    assert tuple(cfg['env'][mode]['seed'] for mode in ('train', 'eval')) == flow_seeds(task)
    assert flow_contract(cfg)['task'] == task
    eval_config = manifest.parent / 'native-eval-config.json'
    assert not eval_config.exists()
    native.write(eval_config, cfg)
    policies = dict(SFT=initial, teacher=teacher)
    for row in policies.values():
        assert native.sha(Path(row['path'])) == row['sha256']
        assert native.sha(Path(row['marker'])) == row['marker_sha256']
    fixture = Path(previous['fixture'])
    assert native.sha(fixture) == previous['fixture_sha256']
    native.write(manifest, dict(task=task, config=str(eval_config), config_sha256=native.sha(eval_config),
        source_config=str(source_cfg_path), source_config_sha256=native.sha(source_cfg_path),
        fixture=str(fixture), fixture_sha256=previous['fixture_sha256'], policies=policies,
        sequential_order=[order(bank) for bank in range(64)],
        case_base=SEEDS[task], rng_base=SEEDS[task]+10000000, cases=512, bank_size=8,
        policy_controls=619200, primary='teacher minus original SFT first success',
        family_tasks=list(SEEDS), family_gate='wins>losses and Holm-adjusted exact McNemar p<.05 across all three',
        fixed_endpoint=True, DEV_selection=False, SGD=0, maximum_allocation_GPU_hours=2.5,
        native_worker_commit='81caf87060c9d966e0256e94da2ae705366ec33f'))


def combine(task, manifest, output):
    import torch
    from endpoint import paired
    native = configure(task)
    m = native.read(manifest)
    assert m['task'] == task and m['case_base'] == SEEDS[task]
    done = native.read(output / 'node-00/RESULT.json')
    assert done['completed'] and done['manifest_sha256'] == native.sha(manifest)
    assert done['banks'] == list(range(64)) and done['policy_controls'] == 619200
    outcomes = {label: {} for label in LABELS}
    endings = {label: {} for label in LABELS}
    repeat = None
    for bank in range(64):
        base = output / 'node-00/banks' / f'bank-{bank:03d}'
        panels = {label: native.read(base / f'{label}.json') for label in order(bank)}
        assert len({row['canonical'] for row in panels.values()}) == 1
        for label, panel in panels.items():
            assert panel['seeds'] == list(range(SEEDS[task]+bank*8, SEEDS[task]+bank*8+8))
            assert panel['common_sampler'] and panel['sampler_seed'] == SEEDS[task]+10000000+bank
            assert native.sha(Path(panel['episodes_path'])) == panel['episodes_sha256']
            expected = m['policies']['SFT' if label == 'repeat' else label]
            assert panel['load']['strict_load'] and panel['load']['sha256'] == expected['sha256']
            if label != 'repeat':
                for row in panel['episodes']:
                    seed = row['seed']
                    assert seed not in outcomes[label]
                    assert row['native_counts']['controls']-row['native_counts']['reset_controls'] == 600
                    outcomes[label][seed] = bool(row['success_once'])
                    endings[label][seed] = bool(row['success_at_end'])
        if bank == 0:
            original, repeated = panels['SFT'], panels['repeat']
            differences = []
            for chunk in range(1, 13):
                actions = []
                for panel in (original, repeated):
                    path = Path(panel['episodes_path'].removesuffix('-episodes.json') + f'-chunk{chunk:02d}.pt.gz')
                    with gzip.open(path, 'rb') as stream:
                        actions.append(torch.as_tensor(torch.load(stream, map_location='cpu', weights_only=False)['decoded_actions']))
                delta = (actions[0]-actions[1]).abs().float()
                differences.append(dict(max_abs=float(delta.max()), mean_abs=float(delta.mean())))
            repeat = dict(cases=8, decoded_actions_by_chunk=differences,
                matching_action_hashes=sum(x == y for x, y in zip(original['sampler_actions'], repeated['sampler_actions'], strict=True)),
                outcomes_equal=sum(x['success_once'] == y['success_once'] for x, y in zip(original['episodes'], repeated['episodes'], strict=True)))
    assert all(set(rows) == set(range(SEEDS[task], SEEDS[task]+512)) for rows in outcomes.values())
    native.write(output / 'RESULT.json', dict(completed=True, task=task, cases=512, policy_controls=619200,
        charged_controls=done['charged_controls'], first_successes={k: sum(v.values()) for k, v in outcomes.items()},
        end_successes={k: sum(v.values()) for k, v in endings.items()},
        primary=paired(outcomes['SFT'], outcomes['teacher'], SEEDS[task]+20000000),
        descriptive_end=paired(endings['SFT'], endings['teacher'], SEEDS[task]+20000001),
        repeat_SFT=repeat, manifest_sha256=native.sha(manifest), readiness='pending three-test Holm family', SGD=0))


def family():
    native = configure('CoffeeSetupMug')
    results = {}
    for task in SEEDS:
        folder = RUN / 'source5-tests' / task / 'attempt-01'
        binding = folder.parent / 'ENDPOINT-RESULT-SOURCE.json'
        if binding.exists():
            bound = native.read(binding)
            assert bound['task'] == task and bound['cases'] == 512 and bound['scientific_comparison_unchanged']
            folder = Path(bound['path'])
            assert folder.parent == binding.parent and bound['manifest_sha256'] == native.sha(folder / 'MANIFEST.json')
        result = native.read(folder / 'RESULT.json')
        manifest = folder / 'MANIFEST.json'
        assert result['completed'] and result['task'] == task and result['cases'] == 512
        assert result['manifest_sha256'] == native.sha(manifest)
        audit_path = folder / 'INDEPENDENT-AUDIT.json'
        audit = native.read(audit_path)
        assert audit['verified'] and not audit['qualification_only'] and audit['cases_per_policy'] == 512
        assert audit['manifest_sha256'] == result['manifest_sha256'] and audit['combined_sha256'] == native.sha(folder / 'RESULT.json')
        results[task] = dict(result=str(folder / 'RESULT.json'), sha256=native.sha(folder / 'RESULT.json'),
            independent_audit=str(audit_path), independent_audit_sha256=native.sha(audit_path), **result['primary'])
    adjusted = 0.
    for index, task in enumerate(sorted(results, key=lambda task: results[task]['exact_McNemar_p'])):
        row = results[task]
        adjusted = max(adjusted, min(1., (3-index)*row['exact_McNemar_p']))
        row.update(holm_adjusted_p=adjusted, improvement=row['wins'] > row['losses'] and adjusted < .05)
    native.write(RUN / 'SOURCE5-READINESS.json', dict(completed=True, family_size=3, tasks=results,
        all_three_improved=all(row['improvement'] for row in results.values()),
        retained_independent_sources=['OpenDrawer130', 'CloseToasterOvenDoor130'],
        old_stopped_campaign_remains_stopped=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'combine', 'family'))
    parser.add_argument('--task', choices=tuple(SEEDS))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode == 'family':
        family()
    elif args.mode == 'prepare':
        prepare(args.task, args.manifest)
    elif args.mode == 'combine':
        configure(args.task)
        combine(args.task, args.manifest, args.output)
    else:
        native = configure(args.task)
        assert native.read(args.manifest)['case_base'] == SEEDS[args.task]
        native.execute(args.manifest, 0, 1, args.output, args.deadline)
