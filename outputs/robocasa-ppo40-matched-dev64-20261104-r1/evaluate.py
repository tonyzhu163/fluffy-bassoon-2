"""Matched DEV64 using the qualified native checkpoint evaluation workers."""
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
from native_eval import SelectionEnvironment, SelectionRollout, configuration
from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa-flow-sde-endpoint-test512-20261104-r1'))
from endpoint import paired, read, sha, write

POLICIES = ('sft', 'old', 'new')
STEP = {'sft': 0, 'old': 40, 'new': 40}


def decoded(path):
    with gzip.open(path, 'rb') as stream:
        return torch.as_tensor(torch.load(stream, map_location='cpu', weights_only=False)['decoded_actions'])


def run(ready, output, deadline):
    assert torch.cuda.device_count() == 1
    spec = read(ready)
    cfg = configuration(dict(task='OpenDrawer', config=spec['source_config'],
        config_sha256=spec['source_config_sha256']), output, deadline, maximum_actions=148000)
    write(output / 'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == 1
    rollout = SelectionRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = SelectionEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    loaded = None

    def load(policy, phase):
        nonlocal loaded
        if loaded == policy:
            return
        model = spec['models'][policy]
        record, = rollout.select_checkpoint(model['path'], model['sha256'], STEP[policy], phase, 1).wait()
        assert record['strict_load'] and record['model_sha256'] == model['sha256']
        loaded = policy

    load('sft', 'preflight')
    parity, = rollout.initial_likelihood_parity(spec['fixture'], spec['fixture_sha256']).wait()
    assert parity['passed']
    write(output / 'SFT-PARITY.json', parity)
    orders = list(itertools.permutations(POLICIES))
    panels = []
    for bank in range(8):
        start = 610100 + 8 * bank
        root = output / 'banks' / f'{bank:02d}'
        ref = spec['resets'][bank]
        assert sha(ref['path']) == ref['sha256']
        original = torch.load(ref['path'], map_location='cpu', weights_only=False)
        reference_rng, sft_actions, sft_episodes = None, None, None
        labels = list(orders[bank % len(orders)]) + (['repeat'] if bank == 0 else [])
        for label in labels:
            assert time.time() < deadline - 120
            assert os.statvfs(output).f_bavail * os.statvfs(output).f_frsize > 200 << 30
            policy = 'sft' if label == 'repeat' else label
            phase = f'dev-{label}-b{bank:02d}'
            load(policy, phase)
            seed, step = 42610000 + bank, STEP[policy]
            rollout.select_sampling_scope(phase, seed).wait()
            env.select_bank(phase, step, start, str(root)).wait()
            begin = time.monotonic()
            runner.evaluate()
            prefix = root / 'environment/rank-00' / f'{phase}-i{step:03d}-round00'
            snap = torch.load(str(prefix) + '-initial.pt', map_location='cpu', weights_only=False)
            reset = compare_native_pair(original, snap)
            episode_path = Path(str(prefix) + '-episodes.json')
            records = read(episode_path)
            episodes = records['rows']
            assert records['completed'] and [e['seed'] for e in episodes] == list(range(start, start + 8))
            assert all(e['native_counts']['controls'] - e['native_counts']['reset_controls'] == 600 for e in episodes)
            sampler_paths = sorted((output / 'sampler').glob(phase + '-*.json'))
            sampling = [read(p) for p in sampler_paths]
            assert len(sampling) == 12 and all(s['actions_shape'] == [8, 50, 12] for s in sampling)
            rng = [(s['before_rng_sha256'], s['after_rng_sha256']) for s in sampling]
            if reference_rng is None:
                reference_rng = rng
            assert rng == reference_rng
            chunks = sorted(prefix.parent.glob(prefix.name + '-chunk*.pt.gz'))
            assert len(chunks) == 12
            summary = dict(bank=bank, policy=label, source_step=step, seeds=list(range(start, start + 8)),
                sampler_seed=seed, rng_hashes=rng, source_reset_match=reset, episode_path=str(episode_path),
                episode_sha256=sha(episode_path), seconds=time.monotonic()-begin,
                outcomes={e['seed']:[bool(e['success_once']), bool(e['success_at_end'])] for e in episodes},
                decoded_action_files=[str(p) for p in chunks], sampler_files=[str(p) for p in sampler_paths])
            if bank == 0 and label == 'sft':
                sft_actions, sft_episodes = chunks, summary['outcomes']
            if label == 'repeat':
                summary['repeat_max_abs_action_difference_by_call'] = [
                    float((decoded(a) - decoded(b)).abs().max()) for a,b in zip(chunks, sft_actions, strict=True)]
                summary['first_success_disagreements'] = sum(v[0] != sft_episodes[k][0] for k,v in summary['outcomes'].items())
                summary['end_success_disagreements'] = sum(v[1] != sft_episodes[k][1] for k,v in summary['outcomes'].items())
            write(output / 'panels' / (phase+'.json'), summary)
            panels.append(summary)
            print(phase, sum(v[0] for v in summary['outcomes'].values()), flush=True)
    env.release_environments().wait()
    runner.metric_logger.finish()
    calls = read(output.parent / 'accounting' / 'rank-00' / (output.name+'.json'))['calls']
    assert calls and all(r['resolved'] for r in calls)
    executed = sum(r['executed_or_conservative'] for r in calls)
    assert executed >= 120000 and executed <= 148000
    assert len(panels) == 25
    stats = {}
    for idx, metric in enumerate(('first_success','end_success')):
        by = {label:{k:v[idx] for p in panels if p['policy']==label for k,v in p['outcomes'].items()} for label in POLICIES}
        assert all(set(d) == set(range(610100,610164)) for d in by.values())
        stats[metric] = {f'{a}_to_{b}':paired(by[a],by[b],616441) for a,b in itertools.combinations(POLICIES,2)}
    repeat = next(p for p in panels if p['policy']=='repeat')
    write(output/'RESULT.json',dict(completed=True, stats=stats, SGD=0, valid_controls=120000,
          all_journaled_controls=executed, repeat=repeat, ready_sha256=sha(ready),
          panels={p.name:sha(p) for p in sorted((output/'panels').glob('*.json'))}))


if __name__ == '__main__':
    run(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
