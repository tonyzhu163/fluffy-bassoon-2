"""Cold CPU audit of the completed eight-rank optimizer boundary, not a GPU restore."""
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
CHECKPOINT = BASE/'motusoptimized03/checkpoints/global_step_1'
MANIFEST_SHA = 'dc81e593f08fd630dd4206761229f9f1793338e7536c310b941f2cfa9d97ad2c'


def audit():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    root=Path(__file__).resolve().parent
    plan=json.loads((root/'MOTUSRESUME01-CONTRACT.json').read_bytes())
    for name,digest in plan['source_files'].items(): assert sha(root.parents[1]/name)==digest
    assert sha(root/'motus_resume_entry.py')==plan['entry_sha256']
    assert sha(root/'resume_motus01.sh')==plan['launcher_sha256']
    import motus_resume_entry  # Import/serialization boundary only; no workers/models constructed.
    from hydra import compose, initialize_config_dir
    from motus_configuration import overrides, validate
    with initialize_config_dir(version_base='1.1', config_dir=str(root.parents[1]/'configs/native_ppo')):
        cfg=compose(config_name='motus_phone_two', overrides=overrides('ppo',optimized_setup=True)+[
            'native_pilot.stop_after_iteration=2', 'runner.resume_dir=/tmp/rtr1/global_step_1'])
    assert validate(cfg)['world']==8
    assert cfg.runner.resume_dir=='/tmp/rtr1/global_step_1'
    output = BASE/'motusoptimized03-boundary01'; output.mkdir()
    manifest = json.loads((CHECKPOINT/'MANIFEST.json').read_bytes())
    assert sha(CHECKPOINT/'MANIFEST.json') == MANIFEST_SHA
    assert manifest['global_step'] == 1 and len(manifest['files']) == 58
    assert manifest['source_commit'] == 'fa5d3c8f381ed1a5d032bf569c16c1f882bf405e'
    for name, expected in manifest['files'].items():
        path = CHECKPOINT/name
        assert path.resolve().is_relative_to(CHECKPOINT.resolve())
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    rows = []
    for rank in range(8):
        pilot = CHECKPOINT/'pilot'
        signature = json.loads((pilot/f'motus-actor-signature-{rank:02d}.json').read_bytes())
        assert signature['optimizer_steps'] == 15
        for role in ('actor', 'rollout', 'env'):
            state = torch.load(pilot/f'{role}-{rank:02d}.pt', map_location='cpu', weights_only=False)
            assert state['role'] == role and state['rank'] == rank
            assert {'python', 'numpy', 'torch', 'cuda'} <= state['rng'].keys()
            assert len(state['rng']['cuda']) == 8
            if role == 'actor':
                assert state['optimizer_steps'] == 15 and state['optimizer_boundary']
            if role == 'env':
                assert {row['mode'] for row in state['samplers']} == {'train', 'eval'}
                schedule = json.loads((pilot/f'motus-case-schedule-env-{rank:02d}.json').read_bytes())
                assert schedule['next_reset_ordinal'] == 4
            rows.append(dict(rank=rank, role=role, tensor_loaded=True,
                             rng_present=True, global_step=state['global_step'], version=state['version']))
            del state; gc.collect()
    root = BASE/'motusoptimized03/state'
    references, memories = [], []
    for rank in range(8):
        reference = json.loads((root/f'learner-reference/iteration-0001-rank-{rank:02d}.json').read_bytes())
        assert reference['passed'] and reference['algorithm'] == 'ppo'
        references.append(reference)
        memory = json.loads((root/f'actor-memory/iteration-0001-rank-{rank:02d}.json').read_bytes())
        assert (memory['micro_batch'], memory['accumulation'], memory['global_batch']) == (256, 1, 2048)
        assert memory['includes_recompute_and_five_native_epochs']
        assert memory['free_bytes_at_end'] >= max(8*2**30, .08*memory['total_bytes'])
        memories.append(memory)
    escrow = []
    for rank in range(8):
        paths = sorted((BASE/f'motusoptimized03/action-accounting/rank-{rank:02d}').glob('*.json'))
        assert paths
        charged = 0
        for path in paths:
            journal = json.loads(path.read_bytes())
            assert journal['rank_action_ceiling'] == 62500
            charged += sum(row.get('executed_or_conservative', row['requested']) for row in journal['calls'])
        # Four fresh rounds plus DEV and constructor/reset reserve, per rank.
        assert charged + 27000 <= 62500, (rank, charged)
        escrow.append(dict(rank=rank, charged=charged, proposed_additional_ceiling=27000,
                           journals={str(p):sha(p) for p in paths}))
    dev = [json.loads((root/f'dev/iteration-{step:04d}.json').read_bytes()) for step in (0, 1)]
    assert all(row['completed'] and row['episodes'] == 16 for row in dev)
    assert not torch.cuda.is_initialized()
    report = dict(status='passed', source_commit=os.environ['TASK_SOURCE'], gpu_seconds=0,
        checkpoint=str(CHECKPOINT), manifest_sha256=MANIFEST_SHA,
        verified_files=len(manifest['files']), verified_bytes=sum(row['bytes'] for row in manifest['files'].values()),
        workers=rows, optimizer_steps_each_rank=[15]*8, learner_references=references,
        actor_memories=memories, retained_action_escrow=escrow, dev=dev,
        all_worker_tensors_cold_loaded=True, independent_gpu_operational_restore_still_required=True,
        seconds=time.monotonic()-started)
    write(output/'RESULT.json', report)


if __name__ == '__main__':
    audit()
