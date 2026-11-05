"""Fresh paired four-epoch PPO40 versus one-epoch GRPO40, without training."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-ppo4-grpo40-test512-20261105-r1'
EVAL = ROOT / 'sources/opendrawer-ablation-3m-eval-r3'
LABELS = ('SFT', 'PPO4', 'GRPO')
CASE_BASE, RNG_BASE = 85411000, 95411000


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
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not manifest.exists()
    torch.set_num_threads(2)
    native = configure()
    ablation = native.read(ROOT / 'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/EVAL-MANIFEST.json')
    baseline = native.read(ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1/BASELINE-MANIFEST.json')
    policies = dict(SFT=ablation['policies']['SFT'], PPO4=ablation['policies']['E'], GRPO=baseline['policies']['GRPO'])
    for row in policies.values():
        assert file_sha256(row['path']) == row['sha256']
        assert file_sha256(row['marker']) == row['marker_sha256']
        assert row['prefix'] and row['ignore_last']
        boundary = native.read(Path(row['marker']))
        assert boundary['completed'] and boundary['global_step'] == row['step']
    # Reverify the already-staged inference adapter, without changing any actor bytes.
    grpo = policies['GRPO']['original_actor_source']
    assert file_sha256(grpo['path']) == grpo['sha256']
    raw = torch.load(grpo['path'], mmap=True, map_location='cpu', weights_only=True)
    adapted = torch.load(policies['GRPO']['path'], mmap=True, map_location='cpu', weights_only=True)
    heads = {key for key in adapted if key.startswith('value_head.')}
    assert len(raw) == 667 and len(heads) == 8 and set(adapted)-heads == set(raw)
    assert all(adapted[k].dtype == raw[k].dtype and adapted[k].shape == raw[k].shape
               and torch.equal(adapted[k].contiguous().reshape(-1).view(torch.uint8),
                               raw[k].contiguous().reshape(-1).view(torch.uint8)) for k in raw)
    for key in ('config', 'fixture'):
        assert file_sha256(ablation[key]) == ablation[key+'_sha256']
    native.write(manifest, dict(task='OpenDrawer', config=ablation['config'],
        config_sha256=ablation['config_sha256'], fixture=ablation['fixture'], fixture_sha256=ablation['fixture_sha256'],
        policies=policies, sequential_order=[order(bank) for bank in range(64)],
        case_base=CASE_BASE, rng_base=RNG_BASE, cases=512, bank_size=8, policy_controls=926400,
        primary='PPO4 minus GRPO first-success; paired two-sided exact McNemar alpha .05',
        TRAIN_controls_per_policy=3072000, rollout_iterations=40,
        optimizer_steps=dict(PPO4=960, GRPO=240), update_epochs=dict(PPO4=4, GRPO=1),
        unchanged_GRPO_actor_tensors=667, unused_GRPO_inference_head_tensors=8,
        maximum_all_attempt_GPU_seconds=27000))


def combine(manifest, folder):
    import gzip
    import torch
    native = configure()
    from endpoint import paired
    outcomes = {label: {} for label in LABELS}
    endings = {label: {} for label in LABELS}
    total = charged = 0
    repeat = None
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
                    continue
                for row in panel['episodes']:
                    seed = row['seed']; assert seed not in outcomes[label]
                    outcomes[label][seed] = bool(row['success_once'])
                    endings[label][seed] = bool(row['success_at_end'])
            if bank == 0:
                original, repeated = panels['SFT'], panels['repeat']
                differences = []
                for chunk in range(1,13):
                    actions = []
                    for panel in (original,repeated):
                        path = Path(panel['episodes_path'].removesuffix('-episodes.json') + f'-chunk{chunk:02d}.pt.gz')
                        with gzip.open(path,'rb') as stream:
                            actions.append(torch.as_tensor(torch.load(stream,map_location='cpu',weights_only=False)['decoded_actions']))
                    delta = (actions[0]-actions[1]).abs().float()
                    differences.append(dict(max_abs=float(delta.max()),mean_abs=float(delta.mean())))
                repeat = dict(cases=8,
                    matching_action_hashes=sum(x==y for x,y in zip(original['sampler_actions'],repeated['sampler_actions'],strict=True)),
                    decoded_actions_by_chunk=differences,
                    outcomes_equal=sum(x['success_once']==y['success_once'] for x,y in zip(original['episodes'],repeated['episodes'],strict=True)))
    assert total == 926400 and all(set(rows) == set(range(CASE_BASE, CASE_BASE + 512)) for rows in outcomes.values())
    result = dict(completed=True, cases=512, policy_controls=total, charged_controls=charged,
        first_successes={key: sum(rows.values()) for key, rows in outcomes.items()},
        end_successes={key: sum(rows.values()) for key, rows in endings.items()},
        primary_PPO4_minus_GRPO=paired(outcomes['GRPO'], outcomes['PPO4'], 105411000),
        descriptive_versus_SFT={key: paired(outcomes['SFT'], outcomes[key], 105411001+i) for i, key in enumerate(('PPO4','GRPO'))},
        descriptive_end_PPO4_minus_GRPO=paired(endings['GRPO'], endings['PPO4'], 105411003),
        repeat_SFT=repeat, manifest_sha256=native.sha(manifest), SGD=0)
    native.write(folder / 'RESULT.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'combine'))
    parser.add_argument('--manifest', type=Path, default=RUN / 'MANIFEST.json')
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
