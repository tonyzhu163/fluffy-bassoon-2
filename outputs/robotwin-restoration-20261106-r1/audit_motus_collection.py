"""Cold-read a terminated pilot's complete pre-SGD inputs on allocated Linux.

This admits a learner-replay input, not an updated checkpoint, a simulator
resume, a learning-history corpus or a completed source experiment.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from motus_learner_reference import audit


def available_memory():
    root = Path('/sys/fs/cgroup')
    if (root/'memory.max').exists():
        maximum = (root/'memory.max').read_text().strip()
        if maximum != 'max':
            return int(maximum) - int((root/'memory.current').read_text())
    root = root/'memory'
    if (root/'memory.limit_in_bytes').exists():
        return int((root/'memory.limit_in_bytes').read_text()) - int((root/'memory.usage_in_bytes').read_text())
    rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(rows['MemAvailable'].split()[0])*1024


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    source, output = map(Path, sys.argv[1:])
    assert source.name == 'motusppo04' and not output.exists()
    output.mkdir()
    started = time.monotonic()
    state = source/'state'
    report = dict(status='checking', source=str(source), ranks=[], source_corpus_admitted=False,
                  exact_simulator_or_all_worker_resume=False, optimizer_updates_verified=False)
    try:
        cfg = json.loads((state/'resolved-native-config.json').read_bytes())
        assert cfg['algorithm']['adv_type'] == 'gae'
        assert cfg['algorithm']['update_epoch'] == 5 and cfg['actor']['global_batch_size'] == 2048
        assert cfg['actor']['micro_batch_size'] == 256
        assert cfg['env']['train']['total_num_envs'] == 256
        assert cfg['env']['train']['rollout_epoch'] == 4
        evidence = {}
        for folder, count in [('native-values', 4), ('group-resets', 16),
                              ('learner-reference', 4), ('first-replay', 4)]:
            paths = sorted((state/folder).glob('*.json'))
            assert len(paths) == count, (folder, len(paths))
            values = [json.loads(path.read_bytes()) for path in paths]
            assert all(row['passed'] for row in values), folder
            evidence[folder] = {path.name: sha(path) for path in paths}
        report['qualification_receipt_sha256'] = evidence
        paths = sorted((state/'training-rollouts').glob('*.json'))
        assert len(paths) == 4
        for rank, receipt in enumerate(paths):
            available = available_memory()
            assert available >= 24*2**30, ('Need24GiB available for sequential full readback', available)
            info = json.loads(receipt.read_bytes())
            path = Path(info['path'])
            assert path.parent == state/'training-rollouts' and path.name.endswith(f'rank-{rank:02d}.pt.gz')
            assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256']
            value = read_tensor(path)
            assert value['format'] == 'rlinf_native_ppo_chronological_rollout_v1'
            assert value['rank'] == rank and value['optimizer_steps_before'] == 0
            assert value['collection_iteration'] == 1 and value['producing_policy_iteration'] == 0
            assert value['rollout_epoch'] == 4 and not value['evaluation_data']
            assert value['dropped_fields'] == []
            rng = value['rng_before_sgd']
            assert set(rng) == {'python', 'numpy', 'torch', 'cuda'}
            assert rng['torch'].dtype == torch.uint8 and len(rng['cuda']) == 4
            assert all(s.dtype == torch.uint8 and s.numel() for s in rng['cuda'])
            batch = value['rollout_batch']
            assert batch['rewards'].shape[:2] == (6, 256)
            assert batch['dones'].shape[:2] == (7, 256)
            assert batch['prev_values'].shape[:2] == (7, 256)
            assert batch['prev_logprobs'].shape == (6, 256, 32, 14)
            chains = batch['forward_inputs']['chains']
            assert chains.shape == (6, 256, 6, 32, 32)
            assert torch.isfinite(chains).all()
            for key in ('prev_logprobs', 'prev_values', 'advantages', 'returns', 'rewards'):
                assert torch.isfinite(batch[key]).all(), key
            reference = audit(batch, 'ppo')
            assert reference['passed'], reference
            row = dict(rank=rank, receipt_sha256=sha(receipt), path=str(path),
                       sha256=info['sha256'], bytes=info['bytes'],
                       available_memory_before_bytes=available, native_decisions=6*256,
                       independent_learner_reference=reference, full_pre_sgd_rng=True,
                       dropped_fields=[], full_gzip_and_tensor_readback=True)
            report['ranks'].append(row)
            write(output/f'rank-{rank:02d}.json', row)
            write(output/'RESULT.json', report)
            print(json.dumps(row), flush=True)
            del value, batch, chains, rng
            gc.collect()
        assert sum(row['native_decisions'] for row in report['ranks']) == 6144
        report.update(status='learner_replay_inputs_complete', native_decisions=6144,
                      seconds=time.monotonic()-started, cuda_initialized=torch.cuda.is_initialized(),
                      required_next='Restore checkpoint0 actor/Adam/scheduler and each saved pre-SGD RNG; run the unchanged native learner with global2048/five epochs; qualify a complete new resumable checkpoint.')
        assert not report['cuda_initialized']
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      seconds=time.monotonic()-started)
        raise
    finally:
        write(output/'RESULT.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
