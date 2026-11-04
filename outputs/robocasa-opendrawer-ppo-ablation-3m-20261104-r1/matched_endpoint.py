"""Prebind six policies; evaluate complete B8 case banks and reduce paired outcomes."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1'
CONTROL = ROOT / 'outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01'
HISTORICAL = ROOT / 'outputs/robocasa-flow-sde-10m-per-task-20261103-r1'
LABELS = ('SFT', 'stage1', 'original', 'E', 'C', 'D')
CASE_BASE, RNG_BASE = 31941000, 41941000


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def order(bank):
    # Balanced Williams design, each six consecutive banks form a complete square.
    labels = [LABELS[(position + bank) % 6] for position in (0, 1, 5, 2, 4, 3)]
    return labels + (['repeat'] if bank == 0 else [])


def prepare(output):
    assert not output.exists()
    original_ready = read(RUN / 'eval-input-original40/READY.json')
    assert original_ready['completed']
    folders = dict(SFT=RUN/'input-prefix-r1/global_step_0', stage1=CONTROL/'OpenDrawer/checkpoints/global_step_40',
                   original=Path(original_ready['original_boundary']).parent)
    for label in ('E', 'C', 'D'):
        attempt = 'attempt-02' if label == 'E' else 'attempt-01'
        training = RUN / label / attempt
        assert read(training/'progress.json')['completed_iterations'] == 40
        assert read(training/'after-evaluation.json')['completed']
        folders[label] = training/'OpenDrawer/checkpoints/global_step_40'
    policies = {}
    for label in LABELS:
        folder = folders[label]
        step = 0 if label == 'SFT' else 40
        marker = folder/'pilot-boundary.json'
        boundary = read(marker)
        assert boundary['completed'] and boundary['global_step'] == step
        weight = folder/'actor/model_state_dict/full_weights.pt'
        bound_weight = CONTROL/'OpenDrawer/checkpoints/global_step_0/actor/model_state_dict/full_weights.pt' if label == 'SFT' else weight
        expected = next(r['sha256'] for r in boundary['actor_checkpoint_files'] if r['path'] == str(bound_weight))
        if label == 'original':
            weight = Path(original_ready['path'])
            assert expected == original_ready['sha256']
        assert sha(weight) == expected
        policies[label] = dict(step=step, path=str(weight), sha256=expected, marker=str(marker),
            marker_sha256=sha(marker), prefix=label not in ('C','original'), ignore_last=label not in ('D','original'))
    cfg = CONTROL/'resolved-config.json'
    fixture = HISTORICAL/'offline-initial-ratio-inputs-r2/OpenDrawer.pt'
    assert sha(fixture) == 'e1b7d41082c7c9206a4c1416adb278d25c0a02d29fb967e0a8db0a32f4894d22'
    write(output, dict(task='OpenDrawer', config=str(cfg), config_sha256=sha(cfg), fixture=str(fixture),
        fixture_sha256=sha(fixture), policies=policies, sequential_order=[order(b) for b in range(64)],
        case_base=CASE_BASE, rng_base=RNG_BASE, cases=512, bank_size=8, policy_controls=1848000))


def execute(manifest, index, workers, output, deadline):
    import numpy as np
    import ray
    import torch
    from matched_eval_worker import MatchedRollout
    from native_eval import configuration, SelectionEnvironment
    from omegaconf import OmegaConf
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    assert torch.cuda.device_count() == 1 and 0 <= index < workers
    m = read(manifest)
    assert m['sequential_order'] == [order(b) for b in range(64)]
    init = ray.init

    def local_ray(**kwargs):
        if kwargs.get('address') == 'auto' and not ray.is_initialized():
            kwargs.update(address='local', _temp_dir=f'/tmp/matched-eval-{index}-{os.getpid()}',
                          num_cpus=16, object_store_memory=2 << 30, include_dashboard=False)
        return init(**kwargs)

    ray.init = local_ray
    banks = list(range(index, 64, workers))
    expected = sum(len(order(b)) for b in banks) * 4800
    cfg = configuration(m, output, deadline, maximum_actions=expected + 64 * 8 * 32)
    write(output/'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == 1
    rollout = MatchedRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = SelectionEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    rollout.select_policy(m['policies']['SFT'], 'initial-parity', 1).wait()
    parity = rollout.initial_likelihood_parity(m['fixture'], m['fixture_sha256']).wait()
    assert len(parity) == 1 and parity[0]['passed']
    write(output/'initial-parity.json', parity)
    panels = 0
    for bank in banks:
        base = output/'banks'/f'bank-{bank:03d}'
        case_seed, rng_seed = CASE_BASE + 8*bank, RNG_BASE + bank
        seeds = list(range(case_seed, case_seed+8))
        reference = None
        for label in order(bank):
            if time.time() >= deadline-200:
                raise TimeoutError('Insufficient time for next full bank')
            fs = os.statvfs(output)
            assert fs.f_bavail*fs.f_frsize > 200 << 30
            row = m['policies']['SFT' if label == 'repeat' else label]
            phase = f'test-{label}-b{bank:03d}'
            receipt, = rollout.select_policy(row, phase, rng_seed).wait()
            assert receipt['strict_load'] and receipt['sha256'] == row['sha256']
            env.select_bank(phase, row['step'], case_seed, str(base)).wait()
            metrics = runner.evaluate()
            key = base/'environment/rank-00'/f'{phase}-i{row["step"]:03d}-round00'
            episodes_path = Path(str(key)+'-episodes.json')
            episodes = read(episodes_path)
            assert episodes['completed'] and [r['seed'] for r in episodes['rows']] == seeds
            assert all(r['native_counts']['controls']-r['native_counts']['reset_controls'] == 600 for r in episodes['rows'])
            initial = read(Path(str(key)+'-initial.json'))
            report = initial['report']
            assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
            samples = [read(output/'sampler'/f'{phase}-{k:02d}.json') for k in range(1,13)]
            assert all(s['batch_size']==8 and s['actions_shape']==[8,50,12] for s in samples)
            hashes = [(s['before_rng_sha256'],s['after_rng_sha256']) for s in samples]
            if reference is None:
                reference = hashes
                write(base/'rng-reference.json', hashes)
            assert hashes == reference
            panel = dict(bank=bank, label=label, phase=phase, seeds=seeds, sampler_seed=rng_seed, load=receipt,
                sampler_actions=[s['actions_sha256'] for s in samples], common_sampler=True,
                canonical=initial['canonical'], images_exact=report['images_exact'], episodes=episodes['rows'],
                episodes_path=str(episodes_path), episodes_sha256=sha(episodes_path),
                metrics={k:np.asarray(v).tolist() for k,v in metrics.items()})
            write(base/f'{label}.json',panel)
            panels += 1
            write(output/'progress.json',dict(panels=panels, bank=bank, label=label, controls=panels*4800))
    env.release_environments().wait()
    runner.metric_logger.finish()
    calls = read(output.parent/'accounting/rank-00'/f'{output.name}.json')['calls']
    assert all(r['resolved'] for r in calls)
    charged = sum(r['executed_or_conservative'] for r in calls)
    assert charged >= expected
    write(output/'RESULT.json',dict(completed=True, banks=banks, panels=panels, policy_controls=expected,
        charged_controls=charged, manifest_sha256=sha(manifest), SGD=0))


def combine(manifest, folder, workers):
    import torch
    from endpoint import paired
    outcomes = {label:{} for label in LABELS}
    endings = {label:{} for label in LABELS}
    total = charged = 0
    for index in range(workers):
        node = folder/f'node-{index:02d}'
        done = read(node/'RESULT.json')
        assert done['completed'] and done['manifest_sha256'] == sha(manifest)
        assert done['banks'] == list(range(index,64,workers))
        total += done['policy_controls']
        charged += done['charged_controls']
        for bank in done['banks']:
            base = node/'banks'/f'bank-{bank:03d}'
            panels = {label:read(base/f'{label}.json') for label in order(bank)}
            assert len({p['canonical'] for p in panels.values()}) == 1
            for label,p in panels.items():
                assert p['seeds'] == list(range(CASE_BASE+8*bank, CASE_BASE+8*bank+8))
                assert p['common_sampler'] and p['sampler_seed'] == RNG_BASE+bank
                assert sha(Path(p['episodes_path'])) == p['episodes_sha256']
                if label != 'repeat':
                    for r in p['episodes']:
                        assert r['seed'] not in outcomes[label]
                        outcomes[label][r['seed']] = bool(r['success_once'])
                        endings[label][r['seed']] = bool(r['success_at_end'])
            if bank == 0:
                a,b = panels['SFT'],panels['repeat']
                differences = []
                for k in range(1,13):
                    actions=[]
                    for panel in (a,b):
                        path=Path(panel['episodes_path'].removesuffix('-episodes.json')+f'-chunk{k:02d}.pt.gz')
                        with gzip.open(path,'rb') as stream:
                            actions.append(torch.as_tensor(torch.load(stream,map_location='cpu',weights_only=False)['decoded_actions']))
                    delta=(actions[0]-actions[1]).abs().float()
                    differences.append(dict(max_abs=float(delta.max()),mean_abs=float(delta.mean())))
                repeat = dict(cases=8, matching_action_hashes=sum(x==y for x,y in zip(a['sampler_actions'],b['sampler_actions'],strict=True)),
                    decoded_actions_by_chunk=differences,
                    outcomes_equal=sum(x['success_once']==y['success_once'] for x,y in zip(a['episodes'],b['episodes'],strict=True)))
    assert total == 1848000 and all(len(o)==512 for o in outcomes.values())
    contrasts = {label:paired(outcomes['stage1'],outcomes[label],51941000+i) for i,label in enumerate(('E','C','D'))}
    adjusted=0.
    for i,label in enumerate(sorted(contrasts,key=lambda x:contrasts[x]['exact_McNemar_p'])):
        row=contrasts[label]
        adjusted=max(adjusted,min(1.,(3-i)*row['exact_McNemar_p']))
        row.update(holm_adjusted_p=adjusted, significant=adjusted<=.05, improvement=adjusted<=.05 and row['wins']>row['losses'])
    write(folder/'RESULT.json',dict(completed=True,cases=512,policy_controls=total,charged_controls=charged,
        first_successes={k:sum(v.values()) for k,v in outcomes.items()},end_successes={k:sum(v.values()) for k,v in endings.items()},
        primary=contrasts,original_minus_stage1=paired(outcomes['stage1'],outcomes['original'],51941010),
        versus_SFT={k:paired(outcomes['SFT'],v,51941020+i) for i,(k,v) in enumerate(outcomes.items()) if k!='SFT'},
        repeat_SFT=repeat,manifest_sha256=sha(manifest)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run','combine'))
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--index',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.mode=='prepare':
        prepare(args.manifest)
    elif args.mode=='run':
        execute(args.manifest,args.index,args.workers,args.output,args.deadline)
    else:
        combine(args.manifest,args.output,args.workers)
