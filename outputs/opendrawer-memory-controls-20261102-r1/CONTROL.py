"""Matched episode-reset / raw-window64 evaluation; frozen AD and DPT."""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
CODE = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('TASK_MEMORY_OUTPUT', ROOT / 'outputs/opendrawer-memory-controls-20261102-r1'))
CORE = ROOT / 'worktrees/opendrawer-ad-dpt-20261102-r1'
os.environ.setdefault('TASK_CURVE_METHOD', 'ad_full')
os.environ.setdefault('TASK_CURVE_OUTPUT', str(OUT))
OLD = runpy.run_path(str(CODE / 'outputs/opendrawer-attempts50-20261102-r1/ATTEMPTS.py'))
BASE = OLD['BASE']
sha, ref, read_ref, write = (OLD[name] for name in ('sha', 'ref', 'read_ref', 'write'))
PAIRING, replay, signature = (OLD[name] for name in ('PAIRING', 'replay', 'memory_signature'))
native_case, motor_seeds, horizon = (OLD[name] for name in ('native_case', 'motor_seeds', 'horizon'))
TASKS = OLD['TASKS']
WORLD = 8


def bind_historical_horizons():
    # This retrospective comparison must keep its original 750/900 protocol.
    # Bind only this process; current prospective training defaults stay intact.
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS as registered
    for cohort in ('ID', 'OOD'):
        task = native_case(cohort, 0)['task']
        registered[task] = (horizon(cohort), *registered[task][1:])
    return {native_case(c, 0)['task']: horizon(c) for c in ('ID', 'OOD')}


def recovery_key(family, arm, cohort, index, attempt):
    return f'{family}/{arm}/{cohort}/{index:03d}/{attempt}'


def source_episode(family, binding, cohort, index, attempt):
    original = ROOT / 'outputs' / f'opendrawer-{family}-attempts50-20261102-r1'
    prefix = binding['practice_prefixes'][f'{cohort}-{index:03d}']
    return prefix[str(attempt)] if str(attempt) in prefix else ref(
        original / 'episodes' / cohort / f'case-{index:03d}' / f'K{attempt}' / 'support/result.json')


def reusable(arm, cohort, attempt):
    return attempt == 1 if arm == 'reset' else attempt <= (4 if cohort == 'ID' else 3)


def bind():
    import torch
    assert sys.platform == 'linux' and not torch.cuda.is_available()
    horizons = bind_historical_horizons()
    # Check both cohorts before allocating GPUs; no native episode is started.
    import tempfile
    from fluffy_bassoon.meta_rl.robocasa_residual import RoboCasaResidualEnvironment
    with tempfile.TemporaryDirectory() as tmp:
        for c in ('ID', 'OOD'):
            RoboCasaResidualEnvironment(None, [native_case(c, 0)], horizon=horizon(c),
                output=Path(tmp)/c, monitoring_root=Path(tmp)/f'{c}-truth', journal=None,
                writer_factory=None)
    models = {}
    for family in ('ad', 'dpt'):
        original = ROOT / 'outputs' / f'opendrawer-{family}-attempts50-20261102-r1'
        b = json.loads((original / 'binding.json').read_text())
        result = json.loads((original / 'RESULT.json').read_text())
        assert result['completed'] and result['binding'] == ref(original / 'binding.json')
        episodes = {}
        for c, i in TASKS:
            episodes[f'{c}-{i:03d}'] = {}
            for k in range(1, 6):
                reference = source_episode(family, b, c, i, k)
                r = read_ref(reference)
                assert r['candidate'] == b['selected']['candidate']
                assert r['native_case'] == native_case(c, i) and r['motor_seeds'] == motor_seeds(c, i, k)
                assert sha(r['records']['path']) == r['records']['sha256']
                assert int(read_ref(r['truth'])['simulator_final_success']) == r['success']
                episodes[f'{c}-{i:03d}'][str(k)] = reference
        models[family] = dict(original_binding=ref(original / 'binding.json'),
            original_result=ref(original / 'RESULT.json'), selected=b['selected'], assembly=b['assembly'],
            pairing=b['pairing'], episodes=episodes)
    assert models['ad']['pairing'] == models['dpt']['pairing']
    assert models['ad']['assembly'] == models['dpt']['assembly']
    recovered = {}; recovered_binding = None; recovered_controls = 0
    if os.environ.get('TASK_MEMORY_REUSE'):
        previous = Path(os.environ['TASK_MEMORY_REUSE'])
        assert previous.resolve() != OUT.resolve()
        recovered_binding = ref(previous/'binding.json')
        prior = read_ref(recovered_binding)
        assert prior['models'] == models and prior['cases'] == [list(x) for x in TASKS]
        for path in sorted(previous.glob('episodes/*/*/*/*/*/result.json')):
            r = json.loads(path.read_text()); f, a, c, i, k = (r[x] for x in ('family','arm','cohort','index','attempt'))
            assert r['complete'] and r['source'] == prior['source'] and not reusable(a, c, k)
            assert r['candidate'] == models[f]['selected']['candidate']
            old = read_ref(models[f]['episodes'][f'{c}-{i:03d}'][str(k)])
            assert r['original'] == models[f]['episodes'][f'{c}-{i:03d}'][str(k)]
            assert r['native_case'] == old['native_case'] and r['motor_seeds'] == old['motor_seeds']
            assert sha(r['records']['path']) == r['records']['sha256']
            assert int(read_ref(r['truth'])['simulator_final_success']) == r['success']
            assert r['initial_pairing']['report']['matched']
            recovered[recovery_key(f, a, c, i, k)] = ref(path)
            recovered_controls += horizon(c)
    assert BASE['clean'](CODE)
    write(OUT / 'binding.json', dict(source=BASE['head'](CODE), models=models, cases=TASKS,
        world=WORLD, arms=['reset', 'window64'], fresh_episodes=1100-len(recovered),
        fresh_policy_controls=915000-recovered_controls, recovered=recovered,
        recovered_binding=recovered_binding, historical_horizons=horizons,
        reused_endpoints=900, evaluation_endpoints=2000, optimizer=None,
        original_runtime_core=OLD['CORE_PIN'], training_window=64,
        memory_window='64 raw input tokens, including the current query; recompute after eviction'))
    print(json.dumps(dict(binding=ref(OUT / 'binding.json'), fresh_episodes=1100-len(recovered),
        recovered=len(recovered), historical_horizons=horizons)), flush=True)


def worker(rank, binding, deadline):
    import multiprocessing as mp
    import imageio.v2 as imageio
    import torch
    from hydra import compose, initialize_config_dir
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
    from fluffy_bassoon.meta_rl.robocasa_residual import FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, tensor_hash
    from fluffy_bassoon.meta_rl.vla_history import load_policy
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from fluffy_bassoon.rl.action_residual import ResidualBound
    assert bind_historical_horizons() == binding['historical_horizons']
    assert torch.cuda.device_count() == 1
    verify_selected_cuda(torch); torch.set_num_threads(4); mp.set_start_method('spawn', force=True)
    assert BASE['head'](CODE) == binding['source'] and BASE['clean'](CODE)
    assembly = read_ref(binding['models']['ad']['assembly']); init = assembly['contract']['initializer']
    profile = json.loads((ROOT / 'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2/profile.json').read_text())
    assert BASE['head'](os.environ['RLINF_PATH']) == profile['rlinf_execution_commit']
    with initialize_config_dir(config_dir=str(CORE / 'configs/native_ppo'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    cfg.rollout.model.model_path = init['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path = init['norm_path']
    assert sha(init['norm_path']) == init['normalization_sha256']
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats = json.loads(Path(init['norm_path']).read_text())
    motor = FrozenRoboCasaMotor(model, stats.get('norm_stats', stats), seed=991)
    bound = ResidualBound(**assembly['contract']['bound'])
    journal = ActionJournal(OUT / 'actions', rank=rank, world=WORLD, attempt=OUT.name, maximum_actions=None)
    completed = 0
    for family, mb in binding['models'].items():
        policy, _ = load_policy(mb['selected']['candidate']['path'],
            expected_sha256=mb['selected']['candidate']['sha256'], device='cuda')
        assert policy.config.context_length == 128
        weights = tensor_hash(policy.state_dict())
        for c, i in TASKS[rank::WORLD]:
            for arm in binding['arms']:
                actor = HistoryResidualActor(policy, reset_each_episode=arm == 'reset',
                    max_context=64 if arm == 'window64' else None)
                for k in range(1, 6):
                    original = read_ref(mb['episodes'][f'{c}-{i:03d}'][str(k)])
                    if reusable(arm, c, k):
                        if arm == 'window64':
                            if k > 1:
                                assert signature(actor) == original['memory_entry']['signature']
                            replay(actor, torch.load(original['records']['path'], map_location='cpu', weights_only=False))
                        continue
                    previous = binding['recovered'].get(recovery_key(family, arm, c, i, k))
                    if previous:
                        saved = read_ref(previous)
                        if arm == 'window64':
                            assert signature(actor) == saved['memory_entry']['signature']
                            replay(actor, torch.load(saved['records']['path'], map_location='cpu', weights_only=False))
                        continue
                    assert time.time() + 300 < deadline, 'Deadline: preserve completed episodes, no retry'
                    directory = OUT / 'episodes' / family / arm / c / f'case-{i:03d}' / f'p{k}'
                    directory.mkdir(parents=True, exist_ok=False)
                    # Capture the state the rollout will actually consume.
                    actor.begin_episode()
                    length = 0 if actor.controller.state is None else int(actor.controller.state['length'][0])
                    expected = 0 if arm == 'reset' else min(64, (k - 1) * horizon(c) // 50)
                    assert length == expected
                    entry = dict(signature=signature(actor), cache_length=length,
                        previous_reward=actor.controller.previous_reward, previous_done=actor.controller.previous_done,
                        raw_tokens=len(actor.controller._tokens))
                    if arm == 'reset':
                        assert not actor.controller.previous_done and actor.controller.previous_reward == 0
                        assert not actor.controller.previous_noise.any()
                    elif k == (5 if c == 'ID' else 4):
                        assert signature(actor) == original['memory_entry']['signature']
                    class Paired(RoboCasaResidualEnvironment):
                        def reset(self):
                            obs = super().reset()
                            physical = self.native.env.get_env_attr('_native_pairing')[0]
                            snapshot = PAIRING['public_snapshot'](obs, native_case(c, i), physical)
                            atomic_save(directory / 'initial.pt', snapshot)
                            marker, report = PAIRING['canonical_case'](Path(mb['pairing']['directory']) / c / f'case-{i:03d}',
                                snapshot, save=atomic_save,
                                load=lambda p: torch.load(p, map_location='cpu', weights_only=False),
                                source=dict(driver=binding['source'], method=family, arm=arm, attempt=k))
                            self.initial_pairing = dict(canonical=ref(marker), report=report,
                                snapshot=ref(directory / 'initial.pt'))
                            return obs
                    env = Paired(PAIRING['pairing_native_factory'](cfg.env.eval), [native_case(c, i)],
                        horizon=horizon(c), output=directory / 'videos', monitoring_root=directory / 'monitor',
                        journal=journal, writer_factory=imageio.get_writer,
                        episode_id_prefix=f'{family}-{arm}-{c}-{i}-p{k}-', rank=rank)
                    try:
                        rows, episodes = collect_pending(actor, bound, motor, env, episodes=1, generator=None,
                            deterministic=True, planned_motor_seeds=motor_seeds(c, i, k))
                    finally:
                        env.close()
                    assert len(rows) == horizon(c) // 50 and len(episodes) == 1
                    truth_path = directory / 'monitor' / (episodes[0]['id'] + '.json')
                    truth = json.loads(truth_path.read_text()); success = int(truth['simulator_final_success'])
                    rows[-1]['reward'] = float(success); actor.install_terminal_reward(success)
                    atomic_save(directory / 'records.pt', rows)
                    write(directory / 'result.json', dict(complete=True, family=family, arm=arm, cohort=c,
                        index=i, attempt=k, success=success, original=mb['episodes'][f'{c}-{i:03d}'][str(k)],
                        candidate=mb['selected']['candidate'], native_case=native_case(c, i),
                        motor_seeds=motor_seeds(c, i, k), horizon=horizon(c), source=binding['source'],
                        records=ref(directory / 'records.pt'), truth=ref(truth_path),
                        initial_pairing=env.initial_pairing, memory_entry=entry,
                        final_cache_length=int(actor.controller.state['length'][0])))
                    completed += 1
                    write(OUT / 'progress' / f'rank-{rank:02d}.json', dict(rank=rank, new_episodes=completed,
                        family=family, arm=arm, cohort=c, index=i, attempt=k, updated_unix=time.time()))
        assert tensor_hash(policy.state_dict()) == weights
        del policy
    write(OUT / 'workers' / f'rank-{rank:02d}.json', dict(complete=True, rank=rank,
        new_episodes=completed, binding=ref(OUT / 'binding.json'), weights_unchanged=True,
        charged_actions=journal.charged()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bind', action='store_true')
    p.add_argument('--worker', type=int); p.add_argument('--deadline', type=float)
    args = p.parse_args()
    if args.bind:
        bind(); return
    binding = json.loads((OUT / 'binding.json').read_text())
    assert BASE['head'](CODE) == binding['source'] and BASE['clean'](CODE)
    if args.worker is not None:
        worker(args.worker, binding, args.deadline); return
    import torch
    node = int(os.environ['RANK'])
    nodes = int(os.environ['WORLD_SIZE'])
    local_gpus = int(os.environ.get('TASK_MEMORY_GPUS_PER_NODE', '2'))
    assert nodes * local_gpus == WORLD and torch.cuda.device_count() == local_gpus
    attempt = OUT / 'job' / f'node-{node:02d}'; attempt.mkdir(parents=True, exist_ok=False)
    assert os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize >= 96 << 30
    probe = attempt / 'gpu-probe.json'
    subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]gpu_probe.py'), '--expected-count', str(local_gpus),
        '--purpose', 'evaluation', '--baige-container', '--output', str(probe)], check=True)
    devices = json.loads(probe.read_text())['devices']; children = []; streams = []
    write(attempt / 'manifest.json', dict(node=node, source=binding['source'], binding=ref(OUT / 'binding.json'),
        devices=devices, deadline=args.deadline))
    for local, device in enumerate(devices):
        rank = node * local_gpus + local
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device['visibility_token']),
            LYCO_BAIGE_GPU_MAP=str(probe), LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
            MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
        stream = (attempt / f'worker-{rank:02d}.log').open('x'); streams.append(stream)
        children.append(subprocess.Popen([sys.executable, '-B', __file__, '--worker', str(rank),
            '--deadline', str(args.deadline)], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
    while not all(child.poll() is not None for child in children):
        if any(child.poll() not in (None, 0) for child in children):
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            break
        time.sleep(5)
    codes = [child.wait() for child in children]
    for stream in streams:
        stream.close()
    write(attempt / 'workers-terminal.json', dict(exit_codes=codes, finished_unix=time.time()))
    if any(codes):
        raise RuntimeError('Control failed; preserve all records and diagnose before retry')
    if node == 0:
        BASE['wait_for_nodes'](OUT / 'job', args.deadline, nodes=nodes)


if __name__ == '__main__':
    main()
