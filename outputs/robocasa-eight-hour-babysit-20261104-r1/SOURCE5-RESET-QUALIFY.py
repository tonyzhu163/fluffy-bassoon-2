"""Qualify fresh evaluation processes against all64 original source DEV starts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
GROUP = RUN / 'reset-qualification/counter-r1'
TASK = 'PickPlaceCounterToCabinet'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prepare():
    assert sys.platform == 'linux' and not GROUP.exists()
    source = RUN / 'source5' / TASK / 'attempt-01'
    cfg = read(source / 'resolved-config.json')
    assert cfg['env']['eval']['task_filter'] == [TASK]
    assert not cfg['native_pilot'].get('fresh_eval_processes', False)
    cfg['native_pilot']['fresh_eval_processes'] = True
    write(GROUP / 'CONFIG.json', cfg)
    references = []
    for rank in range(8):
        original = source / 'canonical-eval' / f'rank-{rank:02d}'
        destination = GROUP / 'banks' / f'bank-{rank:03d}' / 'canonical-eval/rank-00'
        assert sha(original / 'initial.pt') == read(original / 'complete.json')['snapshot_sha256']
        shutil.copytree(original, destination)
        references.append(dict(rank=rank, source=str(original), copied_snapshot_sha256=sha(destination / 'initial.pt')))
    previous = read(RUN / 'source5-tests/CoffeeSetupMug/attempt-01/MANIFEST.json')
    write(GROUP / 'MANIFEST.json', dict(task=TASK, config=str(GROUP / 'CONFIG.json'),
        config_sha256=sha(GROUP / 'CONFIG.json'), fixture=previous['fixture'],
        fixture_sha256=previous['fixture_sha256'], policies={'SFT': previous['policies']['SFT']},
        source=str(source), source_config_sha256=sha(source / 'resolved-config.json'),
        references=references, operations=list(range(8)) + [6], cases=64,
        case_base=cfg['env']['eval']['seed'],
        repeated_cases=8, policy_controls=43200, maximum_controls=54000,
        original_pairing_check_retained=True, optimizer_updates=0,
        purpose='Reproduce original source DEV starts with fresh processes; no source-readiness test'))
    print(json.dumps(dict(prepared=True, manifest_sha256=sha(GROUP / 'MANIFEST.json'), GPU_hours=0)))


def capture_factory(original, capture_root, *args, **kwargs):
    """Observe the exact XML string used by the unchanged native fingerprint."""
    factory = original(*args, **kwargs)

    def create():
        import os
        from fluffy_bassoon.meta_rl import rlinf_robocasa_flow_ppo_runtime as runtime
        fingerprint = runtime.native_fingerprint

        def observed_fingerprint(env):
            # Native constructors may leave sim=None until reset. This hook
            # runs at the existing fingerprint call after native reset, and
            # records exactly its one original get_xml return value.
            model_class = type(env.sim.model)
            get_xml = model_class.get_xml
            if not getattr(get_xml, '_source5_xml_capture', False):
                def recorded_xml(model):
                    value = get_xml(model)
                    digest = hashlib.sha256(value.encode()).hexdigest()
                    root = Path(capture_root)
                    root.mkdir(parents=True, exist_ok=True)
                    destination = root / (digest + '.xml')
                    if not destination.exists():
                        temp = root / f'{digest}.{os.getpid()}.tmp'
                        temp.write_text(value)
                        temp.replace(destination)
                    return value
                recorded_xml._source5_xml_capture = True
                model_class.get_xml = recorded_xml
            return fingerprint(env)

        runtime.native_fingerprint = observed_fingerprint
        return factory()
    return create


def execute(output, deadline):
    import numpy as np
    import ray
    import torch
    from matched_eval_worker import MatchedRollout
    from native_eval import configuration, SelectionEnvironment
    from omegaconf import OmegaConf
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    assert torch.cuda.device_count() == 1
    m = read(GROUP / 'MANIFEST.json')
    m.setdefault('case_base', read(Path(m['config']))['env']['eval']['seed'])
    init = ray.init

    def local_ray(**kwargs):
        if kwargs.get('address') == 'auto' and not ray.is_initialized():
            kwargs.update(address='local', _temp_dir=f'/tmp/source5-reset-{__import__("os").getpid()}',
                          num_cpus=16, object_store_memory=2 << 30, include_dashboard=False)
        return init(**kwargs)

    ray.init = local_ray
    cfg = configuration(m, output, deadline, maximum_actions=m['maximum_controls'])
    cfg.native_pilot.accounting_root = str(output / 'accounting')
    cfg.selection.output = str(GROUP)
    assert cfg.native_pilot.fresh_eval_processes
    write(output / 'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = MatchedRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    class RecordingResetEnvironment(SelectionEnvironment):
        def init_worker(self):
            from fluffy_bassoon.meta_rl import rlinf_robocasa_flow_ppo_runtime as runtime
            original = runtime.fresh_factory
            def observed(*args, **kwargs):
                return capture_factory(original, str(GROUP / 'captured-xml'), *args, **kwargs)
            runtime.fresh_factory = observed
            return super().init_worker()

    env = RecordingResetEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    rollout.select_policy(m['policies']['SFT'], 'initial-parity', 1).wait()
    parity = rollout.initial_likelihood_parity(m['fixture'], m['fixture_sha256']).wait()
    assert len(parity) == 1 and parity[0]['passed']
    write(output / 'initial-parity.json', parity)
    rows = []
    for ordinal, bank in enumerate(m['operations']):
        assert time.time() < deadline - 180
        base = GROUP / 'banks' / f'bank-{bank:03d}'
        phase = f'reset-{ordinal:02d}-r{bank:02d}'
        load, = rollout.select_policy(m['policies']['SFT'], phase, 87600000+m['case_base']+bank).wait()
        assert load['strict_load']
        env.select_bank(phase, 0, m['case_base']+8*bank, str(base)).wait()
        metrics = runner.evaluate()
        path = base / 'environment/rank-00' / f'{phase}-i000-round00-episodes.json'
        data = read(path)
        assert data['completed'] and [row['seed'] for row in data['rows']] == list(range(m['case_base']+8*bank, m['case_base']+8*bank+8))
        initial = read(path.with_name(path.name.replace('-episodes.json', '-initial.json')))
        assert initial['report']['matched'] and initial['report']['physical_state_exact'] and initial['report']['public_state_exact']
        assert all(row['native_counts']['controls']-row['native_counts']['reset_controls'] == 600 for row in data['rows'])
        rows.append(dict(bank=bank, phase=phase, report=initial['report'], episodes=str(path),
            episodes_sha256=sha(path), metrics={key: np.asarray(value).tolist() for key, value in metrics.items()}))
        write(output / 'PROGRESS.json', dict(completed=False, operations=len(rows), rows=rows, SGD=0))
    env.release_environments().wait()
    runner.metric_logger.finish()
    calls = read(output / 'accounting/rank-00' / f'{output.name}.json')['calls']
    assert all(row['resolved'] for row in calls)
    charged = sum(row['executed_or_conservative'] for row in calls)
    assert m['policy_controls'] <= charged <= m['maximum_controls']
    write(output / 'RESULT.json', dict(passed=True, completed=True, original_cases=64,
        repeated_cases=8, rows=rows, policy_controls=43200, charged_controls=charged,
        fresh_eval_processes=True, original_pairing_check_retained=True,
        source_before_reference=f'{TASK} attempt01 original canonical64',
        manifest_sha256=sha(GROUP / 'MANIFEST.json'), SGD=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--task', choices=('PickPlaceCounterToCabinet', 'TurnOnSinkFaucet'), default=TASK)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--group')
    args = parser.parse_args()
    TASK = args.task
    prefix = 'counter' if TASK == 'PickPlaceCounterToCabinet' else 'faucet'
    group = args.group or prefix + '-r1'
    import re
    assert re.fullmatch(prefix + r'-r[1-9][0-9]*', group)
    GROUP = RUN / 'reset-qualification' / group
    prepare() if args.mode == 'prepare' else execute(args.output, args.deadline)
