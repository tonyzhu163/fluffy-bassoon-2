"""50-case native pi0.5 qualification of two TurnOffStove replacements at horizon 600.

Adapted from ../robocasa-atomic-baseline-sweep-20261103-r1/EVAL.py: same native
lane collector, zero residual, cameras and checks; two tasks, 50 cases each,
the PPO training horizon (600) for both, and WORKERS allocated GPUs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# PPO horizon for both; TurnOnElectricKettle's native registry horizon is 450.
TASKS = [('TurnOnSinkFaucet', 600), ('TurnOnElectricKettle', 600)]
NATIVE_HORIZONS = {'TurnOnSinkFaucet': 600, 'TurnOnElectricKettle': 450}
CASES, WORKERS, LANES = 50, 2, 4


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def cases():
    # Fresh bank, disjoint from the atomic sweep (710100..721119), portfolio
    # banks (1x0100, 4x0100, 5x0100, 6x0100), PPO TRAIN (2.1M+) and the
    # checkpoint-selection DEV/TEST/qualification banks (8M+).
    return [dict(bank='buttons_replacement_qualification', task=task, split='target',
                 native_seed=730100 + 1000 * index + slot, reset_ordinal=0,
                 slot=slot, task_index=index, counted=True,
                 layout_source='native_target_constructor_seed_first_reset')
            for slot in range(CASES) for index, (task, _) in enumerate(TASKS)]


def motor_seeds(bank):
    horizons = dict(TASKS)
    # Sweep motor seeds reach 171,119,xxx; start above them.
    return [172000000 + 100000 * c['task_index'] + 1000 * c['slot'] + chunk
            for c in bank for chunk in range(horizons[c['task']] // 50)]


def register_evaluation_tasks():
    # Process-local: adds TurnOnElectricKettle at 600. PPO_TASKS, OOD_TASKS and
    # production training contracts stay intact.
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS as registered
    for task, horizon in TASKS:
        entry = (horizon, 'atomic_seen', 'atomic', None)
        if task in registered:
            assert registered[task][:3] == entry[:3]
        else:
            registered[task] = entry


def interval(successes, count=CASES):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0., center-half), min(1., center+half)]


def worker(rank, profile, output):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
    from fluffy_bassoon.meta_rl.robocasa_residual import (
        FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending_lanes,
        native_environment_factory,
    )
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from fluffy_bassoon.rl.action_residual import ResidualBound
    assert torch.cuda.device_count() == 1
    verify_selected_cuda(torch)
    register_evaluation_tasks()
    code = Path(profile['code'])
    assert subprocess.check_output(['git', '-C', os.environ['RLINF_PATH'], 'rev-parse', 'HEAD'], text=True).strip() == profile['rlinf_execution_commit']
    torch.manual_seed(42)
    with initialize_config_dir(config_dir=str(code/'configs/native_ppo'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    initializer = profile['initializer']
    cfg.rollout.model.model_path = initializer['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path = initializer['norm_path']
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats = json.loads(Path(initializer['norm_path']).read_text())
    motor = FrozenRoboCasaMotor(model, stats.get('norm_stats', stats))
    policy = ResidualActorCritic(ResidualPPOConfig(feature_dim=2088, residual_dim=6)).to('cuda').eval().requires_grad_(False)
    bound = ResidualBound(12, tuple(range(6)), (1.,)*6)
    bank = cases()[rank::WORKERS]
    seeds = motor_seeds(bank)
    write(output/'CASE-PLAN.json', dict(rank=rank, cases=bank, motor_seeds=seeds,
                                       lanes=LANES, deterministic_zero_residual=True))
    journals = [ActionJournal(profile['output']+'/actions', rank=rank, world=WORKERS,
        attempt=f'buttons-qual-r1-lane{lane}', maximum_actions=None) for lane in range(LANES)]
    environments = [RoboCasaResidualEnvironment(native_environment_factory(cfg.env.eval), bank,
        horizon=None, output=output/'videos', monitoring_root=output/'monitoring',
        journal=journal, writer_factory=imageio.get_writer,
        episode_id_prefix=f'buttons-rank{rank}-', rank=rank) for journal in journals]
    began = time.time()

    def completed(slot, records, episode):
        assert all(np.count_nonzero(r['control']) == 0 and
                   np.array_equal(r['base_chunk'], r['executed_chunk']) for r in records)
        atomic_save(output/'episodes'/f'{slot:03d}.pt', dict(case=bank[slot], records=records, episode=episode))
        write(output/'progress'/f'{slot:03d}.json', dict(completed=True, slot=slot,
            task=bank[slot]['task'], native_seed=bank[slot]['native_seed'],
            policy_controls=episode['horizon'], chunks=len(records), elapsed_seconds=time.time()-began))
        print(json.dumps(dict(rank=rank, episode_complete=slot, task=bank[slot]['task'],
                              horizon=episode['horizon'], elapsed_seconds=time.time()-began)), flush=True)

    try:
        records, episodes = collect_pending_lanes(policy, bound, motor, environments,
            planned_motor_seeds=seeds, deterministic=True, verify_zero=True, episode_callback=completed)
        assert len(episodes) == len(bank) and len(records) == len(seeds)
        write(output/'COMPLETED.json', dict(completed=True, rank=rank, episodes=len(episodes),
            policy_controls=sum(e['horizon'] for e in episodes), chunks=len(records),
            elapsed_seconds=time.time()-began, optimizer_updates=0,
            maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    finally:
        for env in environments:
            env.close()


def summarize(output, profile):
    import numpy as np
    import torch
    rows = []
    for rank in range(WORKERS):
        directory = output/f'rank-{rank}'
        bank = cases()[rank::WORKERS]
        assert json.loads((directory/'COMPLETED.json').read_text())['episodes'] == len(bank)
        for slot, case in enumerate(bank):
            payload = torch.load(directory/'episodes'/f'{slot:03d}.pt', map_location='cpu', weights_only=False)
            episode, records = payload['episode'], payload['records']
            truth = json.loads((directory/'monitoring'/(episode['id']+'.json')).read_text())
            assert payload['case'] == truth['case'] == case
            assert episode['horizon'] == dict(TASKS)[case['task']]
            assert episode['camera_order'] == ['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right']
            assert episode['frame_clock']['frame_count'] == episode['horizon'] + 1
            assert len(records)*50 == episode['horizon'] and records[-1]['done']
            assert all(np.count_nonzero(r['control']) == 0 and np.array_equal(r['base_chunk'],r['executed_chunk']) for r in records)
            for video in episode['videos']:
                h = hashlib.sha256()
                with Path(video['path']).open('rb') as stream:
                    for block in iter(lambda: stream.read(8<<20), b''):h.update(block)
                assert h.hexdigest() == video['sha256']
            rows.append(dict(case=case, episode_id=episode['id'], horizon=episode['horizon'],
                first_success=truth['simulator_success_once'], first_success_step=truth['first_success_step'],
                first_success_within_native=bool(truth['simulator_success_once']) and
                    truth['first_success_step'] is not None and
                    truth['first_success_step'] <= NATIVE_HORIZONS[case['task']],
                end_success=truth['simulator_final_success'], video_refs=episode['videos'],
                monitoring=str(directory/'monitoring'/(episode['id']+'.json'))))
    assert len(rows) == len({(r['case']['task'],r['case']['native_seed']) for r in rows}) == CASES*len(TASKS)
    summary = []
    for task, horizon in TASKS:
        panel = [r for r in rows if r['case']['task'] == task]
        assert len(panel) == CASES
        first, end = (sum(r[k] for r in panel) for k in ('first_success','end_success'))
        native = sum(r['first_success_within_native'] for r in panel)
        category = lambda n:'below_5pct' if n < .05*CASES else 'above_70pct' if n > .70*CASES else '5_to_70pct'
        summary.append(dict(task=task,horizon=horizon,episodes=CASES,
            native_horizon=NATIVE_HORIZONS[task],first_successes_within_native_horizon=native,
            first_successes=first,first_success_rate=first/20,first_success_wilson95=interval(first),
            end_successes=end,end_success_rate=end/20,end_success_wilson95=interval(end),
            first_success_category=category(first),end_success_category=category(end)))
    charged = []
    for rank in range(WORKERS):
        calls = [call for path in (output/'actions'/f'rank-{rank:02d}').glob('*.json')
                 for call in json.loads(path.read_text())['calls']]
        charged.append(dict(rank=rank, pending=sum(not c['resolved'] for c in calls),
            physical_controls=sum(c.get('executed_or_conservative',c['requested']) for c in calls)))
    assert sum(r['horizon'] for r in rows) == CASES*sum(h for _,h in TASKS)
    assert sum(c['pending'] for c in charged) == 0
    write(output/'CASE-RESULTS.json',rows)
    write(output/'RESULT.json',dict(completed=True,source_commit=profile['source_commit'],
        tasks=summary,episodes=len(rows),policy_controls=sum(r['horizon'] for r in rows),
        physical_accounting_by_rank=charged,optimizer_updates=0,
        selection_rule='TurnOnSinkFaucet if first success at 600 is 5-70%, else TurnOnElectricKettle if 5-70%, else neither',
        task_selection='human_pending'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    profile=json.loads(Path(args.profile).read_text())
    output=Path(profile['output'])
    if args.worker is not None:
        worker(args.worker,profile,output/f'rank-{args.worker}')
        return
    probe=json.loads(Path(profile['gpu_probe']).read_text())
    assert probe['status']=='passed' and len(probe['devices'])==WORKERS
    processes=[]
    def halt(signum, frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,halt)
    signal.signal(signal.SIGINT,halt)
    try:
        for rank,device in enumerate(probe['devices']):
            directory=output/f'rank-{rank}'
            directory.mkdir()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_GPU_MAP=profile['gpu_probe'],LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
                MUJOCO_EGL_DEVICE_ID='0',ROBOCASA_NATIVE_OUTPUT=str(directory))
            with (directory/'console.log').open('x') as stream:
                processes.append(subprocess.Popen([sys.executable,'-B',__file__,'--profile',args.profile,
                    '--worker',str(rank)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Native worker failed; retain complete episodes, no automatic retry')
            if time.time()>=profile['deadline_unix']:raise TimeoutError('Finite qualification deadline')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        summarize(output,profile)
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL)
                p.wait()


if __name__=='__main__':main()
