"""Fresh matched PPO/GRPO baseline using immutable, already-qualified workers."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
EVAL = ROOT / 'sources/opendrawer-ablation-3m-eval-r3'
LABELS = ('SFT', 'PPO', 'GRPO')
CASE_BASE, RNG_BASE = 65311000, 75311000


def order(bank):
    result = list(tuple(itertools.permutations(LABELS))[bank % 6])
    return result + (['repeat'] if bank == 0 else [])


def configure():
    sys.path.insert(0, str(EVAL / 'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1'))
    import matched_endpoint as native
    native.CASE_BASE, native.RNG_BASE, native.LABELS, native.order = CASE_BASE, RNG_BASE, LABELS, order
    return native


def prepare(manifest):
    import os
    import torch
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preparation belongs to allocated Linux CPUs with no selected accelerator')
    torch.set_num_threads(2)
    native = configure()
    previous = native.read(ROOT / 'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/EVAL-MANIFEST.json')
    ready = native.read(ROOT / 'outputs/robocasa-ppo40-matched-dev64-20261104-r1/READY-FOUR.json')
    grpo = ready['models']['grpo']['source']
    assert file_sha256(grpo['path']) == grpo['sha256']
    initial = previous['policies']['SFT']
    assert file_sha256(initial['path']) == initial['sha256']
    raw = torch.load(grpo['path'], mmap=True, map_location='cpu', weights_only=True)
    reference = torch.load(initial['path'], mmap=True, map_location='cpu', weights_only=True)
    heads = {key for key in reference if key.startswith('value_head.')}
    actor = set(reference) - heads
    assert len(actor) == 667 and len(heads) == 8 and set(raw) == actor
    output = manifest.parent / 'inputs'
    output.mkdir(parents=True, exist_ok=False)
    converted = output / 'grpo40-prefix-eval.pt'
    torch.save({key: reference[key] if key in heads else raw[key] for key in reference}, converted)
    checked = torch.load(converted, mmap=True, map_location='cpu', weights_only=True)
    assert checked.keys() == reference.keys()
    assert all(checked[key].dtype == raw[key].dtype and checked[key].shape == raw[key].shape
        and torch.equal(checked[key].contiguous().reshape(-1).view(torch.uint8),
                        raw[key].contiguous().reshape(-1).view(torch.uint8)) for key in actor)
    policies = dict(SFT=initial, PPO=previous['policies']['stage1'], GRPO=dict(step=40,
        path=str(converted), sha256=file_sha256(converted), marker=grpo['marker'],
        marker_sha256=grpo['marker_sha256'], prefix=True, ignore_last=True,
        original_actor_source=grpo, unchanged_actor_tensors=667, unused_inference_head_tensors=8))
    for row in policies.values():
        assert file_sha256(row['path']) == row['sha256']
        assert file_sha256(row['marker']) == row['marker_sha256']
    native.write(manifest, dict(task='OpenDrawer', config=previous['config'],
        config_sha256=previous['config_sha256'], fixture=previous['fixture'], fixture_sha256=previous['fixture_sha256'],
        policies=policies, sequential_order=[order(bank) for bank in range(64)],
        case_base=CASE_BASE, rng_base=RNG_BASE, cases=512, bank_size=8, policy_controls=926400,
        primary='GRPO minus PPO first-success; paired two-sided exact McNemar alpha .05',
        maximum_all_attempt_GPU_seconds=27000))


def combine(manifest, folder):
    native = configure()
    from endpoint import paired
    outcomes = {label: {} for label in LABELS}
    endings = {label: {} for label in LABELS}
    total = charged = 0
    repeats = []
    for index in range(2):
        node = folder / f'node-{index:02d}'
        done = native.read(node / 'RESULT.json')
        assert done['completed'] and done['manifest_sha256'] == native.sha(manifest)
        assert done['banks'] == list(range(index, 64, 2))
        total += done['policy_controls']; charged += done['charged_controls']
        for bank in done['banks']:
            base = node / 'banks' / f'bank-{bank:03d}'
            panels = {label: native.read(base / f'{label}.json') for label in order(bank)}
            assert len({row['canonical'] for row in panels.values()}) == 1
            for label, panel in panels.items():
                assert panel['seeds'] == list(range(CASE_BASE + bank * 8, CASE_BASE + bank * 8 + 8))
                assert panel['common_sampler'] and panel['sampler_seed'] == RNG_BASE + bank
                assert native.sha(Path(panel['episodes_path'])) == panel['episodes_sha256']
                if label == 'repeat':
                    repeats.append(panel); continue
                for row in panel['episodes']:
                    seed = row['seed']; assert seed not in outcomes[label]
                    outcomes[label][seed] = bool(row['success_once'])
                    endings[label][seed] = bool(row['success_at_end'])
    assert total == 926400 and all(set(rows) == set(range(CASE_BASE, CASE_BASE + 512)) for rows in outcomes.values())
    result = dict(completed=True, cases=512, policy_controls=total, charged_controls=charged,
        first_successes={key: sum(rows.values()) for key, rows in outcomes.items()},
        end_successes={key: sum(rows.values()) for key, rows in endings.items()},
        primary_GRPO_minus_PPO=paired(outcomes['PPO'], outcomes['GRPO'], 85311000),
        descriptive_versus_SFT={key: paired(outcomes['SFT'], outcomes[key], 85311001+i) for i, key in enumerate(('PPO','GRPO'))},
        descriptive_end_GRPO_minus_PPO=paired(endings['PPO'], endings['GRPO'], 85311003),
        repeat_SFT=repeats, manifest_sha256=native.sha(manifest), SGD=0)
    native.write(folder / 'RESULT.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'combine'))
    parser.add_argument('--manifest', type=Path, default=RUN / 'BASELINE-MANIFEST.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.manifest)
    elif args.mode == 'combine':
        combine(args.manifest, args.output)
    else:
        configure().execute(args.manifest, args.index, 2, args.output, args.deadline)


if __name__ == '__main__':
    main()
