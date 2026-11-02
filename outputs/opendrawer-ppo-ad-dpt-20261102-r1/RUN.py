"""Convergence fits and paired native evaluation inside a fresh Baige allocation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT/'outputs/opendrawer-ppo-ad-dpt-20261102-r1'
CODE = Path(__file__).resolve().parents[2]
METHODS = ('ppo_initial', 'ppo_final', 'ad_full', 'ad_static', 'dpt_full', 'dpt_static')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()


def checked(ref):
    assert sha(ref['path']) == ref['sha256'], ref['path']
    return Path(ref['path'])


def fit(objective, mode, source, deadline):
    assembly = json.loads((RUN/'data/assembly.json').read_text())
    datasets = assembly['datasets']
    out = RUN/'fits'/f'{objective}_{mode}'
    status = out/'training-status.json'
    if status.exists() and json.loads(status.read_text())['converged']:
        return
    train, validation = datasets[f'{objective}_train'], datasets[f'{objective}_validation']
    args = [sys.executable, '-B', '-m', 'fluffy_bassoon.meta_rl.vla_history_train',
        '--train-data', str(checked(train)), '--train-sha256', train['sha256'],
        '--validation-data', str(checked(validation)), '--validation-sha256', validation['sha256'],
        '--output-dir', str(out), '--mode', mode, '--width', '128', '--layers', '2', '--heads', '4',
        '--context-length', '128', '--backbone', 'decoder', '--max-window', '64',
        '--batch-size', '8', '--learning-rate', '0.0003', '--eval-every', '100',
        '--validation-batches', '16', '--seed', '0', '--threads', '4', '--device', 'cuda:0',
        '--checkpoint-selection', 'validation-nll', '--until-converged', '--minimum-updates', '1000',
        '--validation-patience', '10', '--validation-relative-delta', '0.001',
        '--max-elapsed-seconds', '12000', '--source-commit', source,
        '--expected-hostname', socket.gethostname()]
    if (out/'latest.pt').exists():
        args += ['--resume', str(out/'latest.pt')]
    if time.time()+300 >= deadline:
        raise TimeoutError('Preserve fits at an operational boundary; resume the same source and contract')
    (RUN/'fit-logs').mkdir(exist_ok=True)
    with (RUN/'fit-logs'/f'{objective}_{mode}.log').open('a') as stream:
        subprocess.run(args, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=max(1, deadline-time.time()-180))
    if not status.exists() or not json.loads(status.read_text())['converged']:
        raise TimeoutError('Fit checkpointed before convergence; no simulator evaluation')


def load_ppo(ref, device):
    from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
    saved = torch.load(checked(ref), map_location='cpu', weights_only=False)
    policy = ResidualActorCritic(ResidualPPOConfig(**saved['config']['ppo'])).to(device)
    policy.load_state_dict(saved['policy'], strict=True)
    return policy.eval().requires_grad_(False)


def noise_plan(case_index, support=False):
    return np.random.default_rng(881210000+case_index+10000*support).integers(0, 2**63-1, 15).tolist()


def case(case_index):
    return dict(bank='heldout_ad_dpt', task='OpenDrawer', split='target',
        native_seed=880110000+case_index, reset_ordinal=0, slot=case_index,
        counted=True, layout_source='native_target_constructor_seed_first_reset')


def evaluate(rank, source, deadline):
    import imageio.v2 as imageio
    from hydra import compose, initialize_config_dir
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
    from fluffy_bassoon.meta_rl.robocasa_residual import (
        FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending, native_environment_factory,
    )
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    from fluffy_bassoon.meta_rl.vla_history import load_policy
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from fluffy_bassoon.rl.action_residual import ResidualBound

    verify_selected_cuda(torch)
    assembly = json.loads((RUN/'data/assembly.json').read_text())
    contract = assembly['contract']
    initializer = contract['initializer']
    profile_path = ROOT/'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2/profile.json'
    profile = json.loads(profile_path.read_text())
    assert subprocess.check_output(['git', '-C', os.environ['RLINF_PATH'], 'rev-parse', 'HEAD'], text=True).strip() == profile['rlinf_execution_commit']
    with initialize_config_dir(config_dir=str(CODE/'configs/native_ppo'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    cfg.rollout.model.model_path = initializer['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path = initializer['norm_path']
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats = json.loads(Path(initializer['norm_path']).read_text())
    motor = FrozenRoboCasaMotor(model, stats.get('norm_stats', stats), seed=991)
    bound = ResidualBound(**contract['bound'])
    policies = dict(ppo_initial=load_ppo(assembly['initial'], 'cuda'), ppo_final=load_ppo(assembly['teacher'], 'cuda'))
    refs = {}
    for objective in ('ad', 'dpt'):
        for mode in ('full', 'static'):
            name = f'{objective}_{mode}'
            path = RUN/'fits'/name/'candidate.pt'
            refs[name] = dict(path=str(path), sha256=sha(path))
            policy, manifest = load_policy(path, expected_sha256=refs[name]['sha256'], device='cuda')
            assert manifest['contract']['source_commit'] == source
            policies[name] = HistoryResidualActor(policy, mode=mode)
    journal = ActionJournal(RUN/'accounting', rank=rank, world=2, attempt=os.environ['TASK_ATTEMPT'])
    ordinal = int(os.environ['TASK_ALLOCATION_ORDINAL'])
    assert rank in (0, 1)
    worker_out = RUN/'evaluation'/f'rank-{rank:02d}'
    worker_out.mkdir(parents=True, exist_ok=True)

    def episode(name, index, actor, *, support=False, phase='K0'):
        directory = worker_out/phase/name/f'case-{index:03d}'/('support' if support else 'query')
        result = directory/'result.json'
        if result.exists():
            payload = json.loads(result.read_text())
            records = torch.load(directory/'records.pt', map_location='cpu', weights_only=False)
            return payload, records
        assert time.time()+300 < deadline, 'Operational deadline; completed cases remain immutable'
        # A previous partial episode cannot be silently recollected.
        directory.mkdir(parents=True, exist_ok=False)
        env = RoboCasaResidualEnvironment(native_environment_factory(cfg.env.eval), [case(index)], horizon=750,
            output=directory/'videos', monitoring_root=directory/'monitor', journal=journal,
            writer_factory=imageio.get_writer, episode_id_prefix=f'{phase}-{name}-{index:03d}-', rank=rank)
        try:
            rows, episodes = collect_pending(actor, bound, motor, env, episodes=1, generator=None,
                deterministic=True, verify_zero=name == 'ppo_initial' and index == rank,
                planned_motor_seeds=noise_plan(index, support))
        finally:
            env.close()
        assert len(rows) == 15 and len(episodes) == 1
        truth_path = directory/'monitor'/(episodes[0]['id']+'.json')
        truth = json.loads(truth_path.read_text())
        reward = int(truth['simulator_final_success'])
        rows[-1]['reward'] = float(reward)
        assert journal.charged() <= 1000000//2
        initial_features = np.asarray(rows[0]['features'], np.float32)
        initial_base = np.asarray(rows[0]['base_chunk'], np.float32)
        payload = dict(method=name, case_index=index, native_case=case(index), phase=phase, support=support,
            success=reward, complete=True, primitive_policy_steps=750, motor_seeds=noise_plan(index, support),
            initial_features_sha256=hashlib.sha256(initial_features.tobytes()).hexdigest(),
            initial_base_sha256=hashlib.sha256(initial_base.tobytes()).hexdigest(),
            records_path=str(directory/'records.pt'), truth=dict(path=str(truth_path), sha256=sha(truth_path)),
            source_commit=source, candidate=refs.get(name, assembly['initial'] if name == 'ppo_initial' else assembly['teacher']))
        atomic_save(directory/'records.pt', rows)
        payload['records_sha256'] = sha(directory/'records.pt')
        write(result, payload)
        print(json.dumps(dict(phase=phase, method=name, case=index, success=reward,
                              actions_charged=journal.charged())), flush=True)
        return payload, rows

    for index in range(rank, 50, 2):
        initial_identity = None
        for name in METHODS:
            row, _ = episode(name, index, policies[name])
            identity = row['initial_features_sha256'], row['initial_base_sha256']
            if initial_identity is None:
                initial_identity = identity
            assert identity == initial_identity, 'Paired initial reset/features/motor draw changed'
    for index in range(rank, 8, 2):
        for objective in ('ad', 'dpt'):
            name = objective+'_full'
            original = policies[name]
            actor = HistoryResidualActor(original.policy, mode='full', reset_each_episode=False)
            support, rows = episode(name, index, actor, support=True, phase='K1')
            # On resume, rebuild memory from real executed support feedback.
            actor.controller.reset_trial()
            for row in rows:
                actor.controller.act(row['features'], deterministic=True)
                actor.controller.observe(row['control'], row['reward'], row['done'])
            query, _ = episode(name, index, actor, phase='K1')
            baseline = json.loads((worker_out/'K0'/name/f'case-{index:03d}'/'query/result.json').read_text())
            assert query['initial_features_sha256'] == baseline['initial_features_sha256']
            assert query['initial_base_sha256'] == baseline['initial_base_sha256']
    write(worker_out/'complete.json', dict(completed=True, source_commit=source,
        primary_cases=25, methods=list(METHODS), support_query_cases=4, all_in_actions=journal.charged(),
        zero_residual_qualification='first retained initializer episode; all15 chunks replayed before execution'))


def summarize(source):
    rows = []
    for rank in (0, 1):
        assert json.loads((RUN/'evaluation'/f'rank-{rank:02d}'/'complete.json').read_text())['completed']
        for path in (RUN/'evaluation'/f'rank-{rank:02d}'/'K0').glob('*/case-*/query/result.json'):
            rows.append(json.loads(path.read_text()))
    grid = {method: {r['case_index']: r['success'] for r in rows if r['method'] == method} for method in METHODS}
    assert all(set(cases) == set(range(50)) for cases in grid.values())
    comparisons = {}
    for method in METHODS[2:]:
        for baseline in METHODS[:2]:
            wins = sum(grid[method][i] > grid[baseline][i] for i in range(50))
            losses = sum(grid[method][i] < grid[baseline][i] for i in range(50))
            n = wins+losses
            p = min(1., 2*sum(math.comb(n, j) for j in range(min(wins, losses)+1))/2**n) if n else 1.
            comparisons[f'{method}-vs-{baseline}'] = dict(wins=wins, losses=losses, ties=50-n, paired_exact_p=p)
    fits = {name: dict(status=json.loads((RUN/'fits'/name/'training-status.json').read_text()),
        candidate=dict(path=str(RUN/'fits'/name/'candidate.pt'), sha256=sha(RUN/'fits'/name/'candidate.pt')))
        for name in METHODS[2:]}
    mechanism = {}
    for name in ('ad_full', 'dpt_full'):
        support, query, reset = [], [], []
        for index in range(8):
            directory = RUN/'evaluation'/f'rank-{index%2:02d}'/'K1'/name/f'case-{index:03d}'
            support.append(json.loads((directory/'support/result.json').read_text())['success'])
            query.append(json.loads((directory/'query/result.json').read_text())['success'])
            reset.append(grid[name][index])
        mechanism[name] = dict(cases=8, support_successes=sum(support), query_successes=sum(query),
            matched_reset_query_successes=sum(reset), reset_results_reused_from_identical_K0=True,
            support_counted_in_deployment_cost=True)
    result = dict(completed=True, source_commit=source, primary_cases=50,
        successes={name: sum(values.values()) for name, values in grid.items()}, paired=comparisons,
        fits=fits, support_query=mechanism, single_seed=True,
        reward='simulator-reward development pilot; not Flora method result')
    write(RUN/'RESULT.json', result)
    lines = ['# OpenDrawer PPO-history AD/DPT result', '', 'All policies: same50 fresh cases, frozen motor/bounds, deterministic means.', '',
        '| Policy | Successes /50 |', '| --- | --- |']
    lines += [f'| {name} | {result["successes"][name]} /50 |' for name in METHODS]
    lines += ['', 'All four fits reached their declared validation plateau; best validation checkpoints were selected before opening evaluation.',
        'Paired wins/losses, exact tests, fit endpoints and the separate eight-case support/query check are in RESULT.json.',
        'This single-task, single-seed simulator-reward pilot does not establish a Flora result or cross-task in-context learning.', '']
    (RUN/'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.source_commit
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
    torch.set_num_threads(4)
    if args.worker is not None:
        assert torch.cuda.device_count() == 1
        objective = ('ad', 'dpt')[args.worker]
        for mode in ('full', 'static'):
            fit(objective, mode, args.source_commit, args.deadline)
        while not all((RUN/'fits'/name/'training-status.json').exists()
                      and json.loads((RUN/'fits'/name/'training-status.json').read_text())['converged'] for name in METHODS[2:]):
            if time.time()+300 >= args.deadline:
                raise TimeoutError('Preserve all fits; peer has not converged yet')
            time.sleep(5)
        evaluate(args.worker, args.source_commit, args.deadline)
        return
    assert torch.cuda.device_count() == 2
    attempt = Path(os.environ['TASK_ATTEMPT_OUT'])
    attempt.mkdir(parents=True, exist_ok=False)
    stat = os.statvfs(RUN)
    assert stat.f_bavail*stat.f_frsize >= 32<<30
    probe_file = attempt/'write-probe'
    probe_file.write_text('write probe\n')
    probe_file.unlink()
    probe = attempt/'gpu-probe.json'
    subprocess.run([sys.executable, '-B', str(CODE/'scripts/[train]gpu_probe.py'), '--expected-count', '2',
        '--purpose', 'training', '--baige-container', '--baige-exclusive-training', '--output', str(probe)], check=True)
    devices = json.loads(probe.read_text())['devices']
    write(attempt/'manifest.json', dict(source_commit=args.source_commit, hostname=socket.gethostname(),
        allocation_devices=devices, source_assembly=dict(path=str(RUN/'data/assembly.json'), sha256=sha(RUN/'data/assembly.json')),
        job_deadline=args.deadline, convergence_criterion='minimum1000; patience10; every100; relative_delta0.001',
        task='OpenDrawer', cases=50, constructor_seed_first=880110000, new_ppo_updates=0,
        available_bytes=stat.f_bavail*stat.f_frsize, quota='not exposed; capacity/write probe verified'))
    children, streams = [], []
    for rank, device in enumerate(devices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device['visibility_token']),
            LYCO_BAIGE_GPU_MAP=str(probe), LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
            MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']), TASK_ALLOCATION_ORDINAL=str(device['allocation_ordinal']))
        stream = (attempt/f'worker-{rank:02d}.log').open('x')
        streams.append(stream)
        children.append(subprocess.Popen([sys.executable, '-B', __file__, '--worker', str(rank),
            '--source-commit', args.source_commit, '--deadline', str(args.deadline)], env=env,
            stdout=stream, stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    for stream in streams:
        stream.close()
    write(attempt/'workers-terminal.json', dict(exit_codes=codes, finished_unix=time.time()))
    if any(codes):
        raise RuntimeError('Worker stopped; diagnose saved logs/checkpoints before any new allocation')
    summarize(args.source_commit)


if __name__ == '__main__':
    main()
