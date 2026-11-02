"""Native-only timing panel; one shared frozen motor and up to eight lanes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from fluffy_bassoon.baige_gpu import verify_selected_cuda
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import write_new
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
from fluffy_bassoon.meta_rl.robocasa_residual import (
    FrozenRoboCasaMotor, RoboCasaResidualEnvironment, native_environment_factory,
)


def combine(rows):
    return {key: torch.cat([row[key] for row in rows], 0) if torch.is_tensor(rows[0][key])
            else np.concatenate([row[key] for row in rows], 0) if isinstance(rows[0][key], np.ndarray)
            else sum((list(row[key]) for row in rows), []) for key in rows[0]}


def seed(case, chunk):
    value = f"pi05-first-success-v1/{case['task']}/{case['native_seed']}/{chunk}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big')


def run(profile):
    assert torch.cuda.device_count() == 1
    uuid = verify_selected_cuda(torch)
    out = Path(profile['output']); out.mkdir(exist_ok=False)
    start = time.time()
    with initialize_config_dir(config_dir=str(Path(profile['code'])/'configs/native_ppo'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    initializer = profile['initializer']
    cfg.rollout.model.model_path = initializer['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path = initializer['norm_path']
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    stats = json.loads(Path(initializer['norm_path']).read_text())
    motor = FrozenRoboCasaMotor(model, stats.get('norm_stats', stats))
    rank = profile['rank']; cases = profile['cases']
    planned = [dict(case=case, seeds=[seed(case, chunk) for chunk in range(TASKS[case['task']][0]//50)])
               for case in cases]
    write_new(out/'plan.json', planned)
    rows = []; active = []
    try:
        for task in dict.fromkeys(case['task'] for case in cases):
            group = [case for case in cases if case['task'] == task]
            assert 1 <= len(group) <= 8
            horizon = TASKS[task][0]
            for case in group:
                key = f"timing-r{rank}-{task}-s{case['slot']:02d}-"
                journal = ActionJournal(profile['accounting_root'], rank=rank, world=8,
                    attempt=key, maximum_actions=8*sum(TASKS[c['task']][0]+128 for c in cases))
                active.append(RoboCasaResidualEnvironment(native_environment_factory(cfg.env.eval), [case],
                    horizon=horizon, output=out/key, monitoring_root=out/'monitoring', journal=journal,
                    writer_factory=imageio.get_writer, episode_id_prefix=key, rank=rank))
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                observations = list(pool.map(lambda env: env.reset(), active))
                for chunk in range(horizon//50):
                    if time.time()+profile['closure_seconds'] >= profile['deadline_unix']:
                        raise TimeoutError('Finite native timing clock; preserve incomplete cases without retry')
                    chunks, _, _ = motor.infer_batch(combine(observations),
                        noise_seeds=[seed(case, chunk) for case in group])
                    results = list(pool.map(lambda pair: pair[0].step_chunk(pair[1]), zip(active, chunks)))
                    observations = [row[0] for row in results]
                    assert all(row[1] == (chunk == horizon//50-1) for row in results)
                    temporary = out/'progress.tmp'
                    temporary.write_text(json.dumps(dict(task=task, chunk=chunk+1,
                        completed_episodes=len(rows), controls_current_group=(chunk+1)*50*len(group))))
                    temporary.replace(out/'progress.json')
                for env, (_, _, episode) in zip(active, results):
                    monitor = out/'monitoring'/(episode['id']+'.json')
                    row = json.loads(monitor.read_text())
                    assert row['first_success_step'] == (row['success_steps'][0] if row['success_steps'] else None)
                    row['horizon'] = horizon
                    rows.append(row)
            for env in active: env.close()
            active = []
            write_new(out/(task+'-timing.json'), [row for row in rows if row['case']['task'] == task])
        assert len(rows) == len(cases)
        write_new(out/'result.json', dict(completed=True, rank=rank, gpu_uuid=uuid,
            source_commit=profile['source_commit'], initializer=initializer,
            rows=rows, policy_controls=sum(TASKS[case['task']][0] for case in cases),
            optimizer_updates=0, elapsed_seconds=time.time()-start))
    finally:
        for env in active: env.close()


if __name__ == '__main__':
    import sys
    run(json.loads(Path(sys.argv[1]).read_text()))
