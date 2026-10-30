"""Allocated reset-only qualification of the immutable PPO repair path.

Four real resets, zero motor calls/optimizer steps. Settling is charged into
its original cumulative PPO execution journal; source RNG/cursor is untouched.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time

import numpy as np

from .vla_source40_multitask_ppo import OwnedSimulator, TaskCycleSimulator, source_tasks
from .vla_source40_ppo import JournaledSimulator, physical_charge, write_json
from .vla_source40_schedule import load_schedule


def compare_resets(results, *, camera_max_lsb=0):
    if len(results) != 4:
        raise ValueError('Repair qualification requires four actual resets')
    first = results[0]
    if camera_max_lsb:
        from .vla_reset_identity import comparison
        metrics = [comparison(first, result, camera_max_lsb=camera_max_lsb) for result in results]
        if not all(row['allowed'] for row in metrics):
            raise RuntimeError('Explicit source40 physical/proprio/camera precision qualification failed')
        return dict(exact_actual_simulator_states=True, exact_nominal_states=True,
            exact_proprioception=True, camera_max_uint8_lsb=1, reset_comparisons=metrics,
            raw_observations_unchanged=True, source40_renderer_tolerance_explicit=True)
    for result in results:
        if (not result.get('initial_state_vector_sha256')
                or result['initial_state_vector_sha256'] != first['initial_state_vector_sha256']
                or not result.get('initial_simulator_state_sha256')
                or result['initial_simulator_state_sha256'] != first['initial_simulator_state_sha256']
                or result.get('initial_observation_sha256') != first['initial_observation_sha256']
                or not np.array_equal(result['initial_simulator_state'], first['initial_simulator_state'])
                or any(not np.array_equal(result['observation'][key], first['observation'][key])
                    for key in ('image', 'wrist', 'state'))):
            raise RuntimeError('Fresh-process retry qualification changed the physical state or exact public observation')
    return dict(exact_public_observations=True, exact_actual_simulator_states=True,
        nominal_state_sha256=first['initial_state_vector_sha256'],
        actual_state_sha256=first['initial_simulator_state_sha256'],
        initial_observation_sha256=first['initial_observation_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('migration', 'schedule', 'output', 'experiment', 'simulator-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--simulator-pythonpath', required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--camera-reset-max-lsb', type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Actual reset qualification requires the named allocated Linux host')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    receipt = json.loads(args.migration.read_text())
    original_sha = __import__('hashlib').sha256((args.experiment / 'learner/latest.pt').read_bytes()).hexdigest()
    selected = receipt['next_case']
    tasks = [row for row in source_tasks(load_schedule(args.schedule)) if row['task_key'] == selected['task_key']]
    if len(tasks) != 1:
        raise ValueError('Migration next case is absent from the original schedule')
    args.output.mkdir(parents=True, exist_ok=False)
    counter = 0
    def factory(task):
        nonlocal counter
        counter += 1
        return OwnedSimulator(task, python=args.simulator_python, pythonpath=args.simulator_pythonpath,
            output=args.output / f'server-{counter:02d}', port=args.port)
    cycle = TaskCycleSimulator(tasks, 4, factory, receipts=args.output / 'isolated-probe-resets')
    sim = JournaledSimulator(cycle, args.experiment / 'execution-journal', 172000)
    before = physical_charge(sim.journal)['total_action_charge']
    results = []
    try:
        cycle.begin_collection()
        for ordinal in range(4):
            result = sim.reset(selected['initial_state'], selected['scene_seed'])
            results.append(result)
            with gzip.open(args.output / f'reset-{ordinal:02d}.pkl.gz', 'xb', compresslevel=1) as stream:
                pickle.dump(result, stream, protocol=5)
        outcome = compare_resets(results, camera_max_lsb=args.camera_reset_max_lsb)
        after = physical_charge(sim.journal)['total_action_charge']
        if after - before != 40:
            raise RuntimeError('Reset-only probe cost differs from four10-action settling resets')
        if __import__('hashlib').sha256((args.experiment / 'learner/latest.pt').read_bytes()).hexdigest() != original_sha:
            raise RuntimeError('Reset qualification changed the source recovery/cursor/RNG')
        write_json(args.output / 'index.json', dict(completed=True, **outcome, next_case=selected,
            settling_actions=40, motor_calls=0, optimizer_updates=0, checkpoint_sha256=original_sha,
            source_checkpoint_unchanged=True, isolated_probe_task_cursor=cycle.state_dict()))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
