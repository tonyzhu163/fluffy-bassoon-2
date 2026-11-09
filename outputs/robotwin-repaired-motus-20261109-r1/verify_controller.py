"""Fresh paired controller and K/horizon verification; allocated Linux only."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, cpu_tree, json_metrics, sha, write
from fluffy_bassoon.rl.robotwin_batch import FastMeasuredSimulator, RoboTwinBatchRuntime, accounted_controller_source
from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor
from fluffy_bassoon.rl.robotwin_noise import _observation_hash, model_config
from fluffy_bassoon.rl.robotwin_production_process import ProductionSimulatorPool
from qualify_batch_runtime import write_trace
from qualify_motor import MOTOR, Memory

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'robotwin-restoration-20261106-r1'
PROFILES = [
    dict(name='legacy_k5_h192', waypoint_controller=False, num_steps=5, horizon=192),
    dict(name='waypoint_k5_h192', waypoint_controller=True, num_steps=5, horizon=192),
    dict(name='waypoint_k10_h384', waypoint_controller=True, num_steps=10, horizon=384),
]


class AuditedSimulator(FastMeasuredSimulator):
    """Observe planner/arm calls without changing their arguments or results."""
    def step(self, actions):
        robot = self.env.venv.envs[0].task.robot
        counts = {side: dict(plans=0, failures=0, empty_plans=0, arm_commands=0,
                            path_points=[], nonstationary=False) for side in ('left', 'right')}
        original_arm = robot.set_arm_joints
        originals = []

        def arm(position, velocity, side):
            counts[side]['arm_commands'] += 1
            return original_arm(position, velocity, side)

        robot.set_arm_joints = arm
        for side, start in (('left', 0), ('right', 7)):
            current = np.asarray(getattr(robot, f'get_{side}_arm_jointState')())[:-1]
            counts[side]['nonstationary'] = bool(np.max(np.abs(actions[:, start:start+len(current)] - current)) > 1e-5)
            planner = getattr(robot, f'{side}_mplib_planner')
            original = planner.TOPP
            originals.append((planner, original))

            def plan(path, *args, _side=side, _original=original, **kwargs):
                row = counts[_side]
                row['plans'] += 1
                row['path_points'].append(len(path))
                try:
                    result = _original(path, *args, **kwargs)
                except Exception:
                    row['failures'] += 1
                    raise
                row['empty_plans'] += int(len(result[1]) == 0)
                return result

            planner.TOPP = plan
        try:
            obs, reward, term, trunc, info = super().step(actions)
        finally:
            robot.set_arm_joints = original_arm
            for planner, original in originals:
                planner.TOPP = original
        delivered = info['execution']['delivered_targets']
        for row in counts.values():
            row['full_chunk_freeze'] = bool(delivered and row['nonstationary'] and row['arm_commands'] == 0)
        info['controller_calls'] = counts
        return obs, reward, term, trunc, info


def deadline():
    assert time.time() < float(os.environ['TASK_DEADLINE']) - 90, 'Worker deadline reserve'


def run_profile(profile, cases, root, local, journal, initial):
    deadline()
    name, horizon = profile['name'], profile['horizon']
    from rlinf.models import get_model
    model = get_model(model_config(os.environ['REPO_PATH'], MOTOR,
        MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json',
        action_horizon=32, num_steps=profile['num_steps'])).cuda()
    motor = FrozenRoboTwinMotor(model)
    options = accounted_controller_source(os.environ['ROBOTWIN_PATH'], local/name/'controller',
        waypoint_controller=profile['waypoint_controller'])
    pool = ProductionSimulatorPool(2, rlinf_root=os.environ['REPO_PATH'],
        robotwin_root=os.environ['ROBOTWIN_PATH'], output_dir=local/name/'simulators',
        task_episode_limits={'place_phone_stand': horizon}, final_observation_only=False,
        task_config_name='demo_clean', _simulator_factory=AuditedSimulator, worker_options=options)
    runtime = RoboTwinBatchRuntime(motor, pool, seed=20261109)
    entry = dict(profile, controller_sha256=options['controller_sha256'], episodes=[], resets=[],
                 traces=[], started_unix=time.time(), full_chunk_arm_freezes=0,
                 planner_failures=0, empty_plans=0)
    try:
        for pair in range(len(cases)//2):
            deadline()
            selected = cases[pair*2:pair*2+2]
            receipts = runtime.reset('place_phone_stand', selected,
                case_ids=[f'paired-controller/case-{seed}' for seed in selected])
            entry['resets'].extend(receipts)
            for lane, case in enumerate(selected):
                signature = _observation_hash(runtime._raw[lane])
                if case not in initial:
                    initial[case] = signature
                assert initial[case] == signature, ('Paired initial observation differs', name, case)
            episodes = [dict(case=seed, success=False, complete=False, decisions=0,
                             delivered_targets=0, physics_steps=0) for seed in selected]
            entry['episodes'].extend(episodes)
            decision = 0
            while any(runtime._active):
                deadline()
                base, _, metadata = runtime.infer(fixed_native_batch=True)
                label = f'{name}-pair-{pair:02d}-decision-{decision:02d}'
                write_trace(root/(label+'-intent.pt.gz'), cpu_tree(dict(
                    observations=runtime._pending['observations'], commands=base, native_batch=metadata)))
                requested = len(metadata['active_lanes'])*32
                cursor = journal.begin(requested, phase=name)
                transitions = runtime.step(base)
                write_trace(root/(label+'-transitions.pt.gz'), transitions)
                journal.finish(cursor, requested)
                entry['traces'].append(dict(intent=label+'-intent.pt.gz', transitions=label+'-transitions.pt.gz'))
                for transition in transitions:
                    episode = episodes[transition['native_batch']['row']]
                    episode['decisions'] += 1
                    episode['success'] |= bool(transition['audit']['success'])
                    episode['complete'] = bool(transition['terminated'] or transition['truncated'])
                    for key in ('delivered_targets', 'physics_steps'):
                        episode[key] += transition['execution'][key]
                    for counts in transition['audit']['controller_calls'].values():
                        entry['full_chunk_arm_freezes'] += int(counts['full_chunk_freeze'])
                        entry['planner_failures'] += counts['failures']
                        entry['empty_plans'] += counts['empty_plans']
                        if profile['waypoint_controller']:
                            assert all(n == 2 for n in counts['path_points'])
                decision += 1
                write(root/f'{name}-progress.json', entry)
            assert all(ep['complete'] for ep in episodes)
        motor.check_frozen()
    finally:
        runtime.close()
        assert not any(p.is_alive() for p in pool.processes)
        del runtime, pool, motor, model
        gc.collect()
        torch.cuda.empty_cache()
    entry.update(ended_unix=time.time(), owned_workers_closed=True, frozen_model_unchanged=True,
                 successes=sum(ep['success'] for ep in entry['episodes']), total_episodes=len(cases))
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, choices=(16, 64), default=16)
    args = parser.parse_args()
    assert os.uname().sysname == 'Linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    cases_path = PREVIOUS/'MOTUS-SOURCE-READOUT-CASES.json'
    cases = json.loads(cases_path.read_bytes())['dev'][:args.max_cases]
    assert len(cases) == len(set(cases)) == args.max_cases
    if args.max_cases == 16:
        assert cases == json.loads((PREVIOUS/'MOTUS-DEV-CASES.json').read_bytes())['seeds']
    assert args.output_dir.is_dir() and not (args.output_dir/'RESULT.json').exists()
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    reference = json.loads((PREVIOUS/'MOTOR01-CONTRACT.json').read_bytes())
    assert sha(MOTOR/'model.safetensors') == reference['model_sha256']
    assert sha(norm) == reference['norm_stats_sha256']
    report = dict(status='running', source_commit=os.environ['TASK_SOURCE'], task='place_phone_stand',
                  cases=cases, cases_sha256=sha(cases_path), model_sha256=reference['model_sha256'],
                  norm_stats_sha256=reference['norm_stats_sha256'], native_batch=2, optimizer_updates=0,
                  motor_noise_seed=20261109, profiles={}, started_unix=time.time())
    journal = ActionJournal(args.output_dir/'action-accounting', rank=0, world=1,
                            attempt=f'controller-dev{args.max_cases}-01', maximum_actions=100000)
    initial = {}
    try:
        with Memory(torch) as memory:
            for profile in PROFILES:
                report['profiles'][profile['name']] = run_profile(profile, cases, args.output_dir,
                    args.local_dir, journal, initial)
                write(args.output_dir/'RESULT.json', report)
        report['memory'] = memory.result()
        report['gate_passed'] = (report['profiles']['waypoint_k10_h384']['successes'] >= len(cases)//4
            and all(report['profiles'][p['name']]['full_chunk_arm_freezes'] == 0
                    for p in PROFILES if p['waypoint_controller']))
        report.update(status='completed', independent_readback_pending=True,
                      charged_requested_targets=journal.charged())
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['ended_unix'] = time.time()
        write(args.output_dir/'RESULT.json', report)
    print(json.dumps(json_metrics(report), indent=2))


if __name__ == '__main__':
    main()
