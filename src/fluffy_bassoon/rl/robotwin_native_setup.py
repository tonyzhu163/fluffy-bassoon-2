"""Shared native RoboTwin GPU binding and opt-in simulator lifecycle setup.

No learner, model factory or distributed worker imports are needed to prepare
an environment-only qualification.
"""
import os
from pathlib import Path

from fluffy_bassoon.meta_rl.rlinf_native_ppo import write


def worker_setup(worker, role):
    import torch
    from fluffy_bassoon.baige_gpu import mapping, cuda_pci, pci
    count = torch.cuda.device_count()
    assert count in (1, 4)
    torch.cuda.set_device(0 if count == 1 else worker._rank)
    actual = cuda_pci(torch.cuda.current_device())
    rows = [row for row in mapping() if pci(row['pci_bus_id']) == actual]
    assert len(rows) == 1
    os.environ['LYCO_BAIGE_ISOLATED_UUID'] = rows[0]['uuid']
    write(Path(worker.cfg.native_pilot.output)/'worker-gpus'/f'{role}-{worker._rank:02d}.json',
        dict(role=role, rank=worker._rank, cuda_pci=actual, scheduler_uuid=rows[0]['uuid'],
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_device_count=count))


def configure_environment(worker):
    from fluffy_bassoon.rl.robotwin_noise import (
        _install_allocated_renderer, _install_reset_retry_limit,
        _install_vector_executor_recycling)
    defer = worker.cfg.native_pilot.get('defer_scene_setup', False)
    recycle = worker.cfg.native_pilot.get('recycle_scene_executor', False)
    if type(defer) is not bool or type(recycle) is not bool:
        raise ValueError('Simulator setup profiles require explicit booleans')
    worker_setup(worker, 'env')
    root = Path(worker.cfg.native_pilot.output)
    _install_allocated_renderer(root/'renderers'/f'rank-{worker._rank:02d}')
    _install_reset_retry_limit(defer_scene=defer)
    if recycle:
        _install_vector_executor_recycling()
    from robotwin.envs.vector_env import SubEnv
    SubEnv._fb2_trim_after_scene_close = False
    write(root/'setup-profiles'/f'rank-{worker._rank:02d}.json', dict(
        defer_scene_setup=defer, recycle_scene_executor=recycle,
        shared_native_training_and_evaluation=True))
    return root
