"""Recover the baseline's64-case endpoint using qualified native B8 panels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from native_eval import SelectionEnvironment, SelectionRollout, configuration
from fluffy_bassoon.meta_rl.rlinf_native_ppo import restore_rng, sha, snapshot_rng, tensor_hash, write
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import unwrapped


class EndpointEnvironment(SelectionEnvironment):
    def restore_rank_bank(self, path, expected_sha, phase, rank, bank_output):
        assert sha(path) == expected_sha
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['rank'] == rank
        rows = [row for row in state['samplers'] if row['mode'] == 'eval']
        assert len(rows) == len(self.eval_env_list) == 1 and rows[0]['ordinal'] == 0
        row = rows[0]
        env = unwrapped(self.eval_env_list[0])
        env.seed = row['seed']
        env._generator.bit_generator.state = deepcopy(row['generator'])
        env.task_ids = row['task_ids'].copy()
        env._ordered_task_cursor = row['ordered_task_cursor']
        env._refresh_task_context()
        self._prefetched_train_bootstrap = None
        result = self.select_bank(phase, 3, 630100 + 8 * rank, bank_output)
        assert state['paired_eval_rng'] is not None
        restore_rng(state['paired_eval_rng'])
        return dict(result, original_rank=rank, original_sidecar_sha256=expected_sha,
            sampler_generator_restored=True, paired_environment_rng_restored=True)


class EndpointRollout(SelectionRollout):
    def restore_rank_rng(self, path, expected_sha, phase, rank):
        assert sha(path) == expected_sha
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['rank'] == rank and state['paired_eval_rng'] is not None
        self._selection_phase, self._selection_calls = phase, 0
        restore_rng(state['paired_eval_rng'])
        expected = tensor_hash({key: state['paired_eval_rng'][key] for key in ['torch', 'cuda']})
        actual = tensor_hash({key: snapshot_rng()[key] for key in ['torch', 'cuda']})
        assert actual == expected
        return dict(original_rank=rank, paired_rng_sha256=actual, sidecar_sha256=expected_sha)


def run(root, output, deadline):
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    assert torch.cuda.device_count() == 1
    recovery = json.loads((root / 'comparison-input-r2/RESTORED.json').read_text())
    assert recovery['completed'] and recovery['canonical_ranks'] == 8
    admission = json.loads((root / 'BASELINE-ENDPOINT-ADMISSION.json').read_text())
    assert admission['completed'] and admission['step'] == 3
    cfg = configuration(admission, output, deadline, maximum_actions=50000)
    # Every original panel remains B8; only their execution becomes serial.
    write(output / 'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == 1
    rollout = EndpointRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = EndpointEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    loaded = rollout.select_checkpoint(admission['model'], admission['model_sha256'], 3, 'load', 0).wait()
    rows = []
    for rank in range(8):
        tick = time.monotonic()
        bank = output / 'banks' / f'original-rank-{rank:02d}'
        target = bank / 'canonical-eval/rank-00'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / f'comparison-input-r2/canonical-eval/rank-{rank:02d}', target)
        marker = json.loads((target / 'complete.json').read_text())
        assert marker['snapshot_sha256'] == sha(target / 'initial.pt')
        phase = f'after-rank{rank:02d}'
        reference = admission['sidecars'][rank]
        environment = env.restore_rank_bank(reference['environment'], reference['environment_sha256'],
            phase, rank, str(bank)).wait()
        sampler = rollout.restore_rank_rng(reference['rollout'], reference['rollout_sha256'], phase, rank).wait()
        metrics = runner.evaluate()
        episodes = bank / f'environment/rank-00/{phase}-i003-round00-episodes.json'
        data = json.loads(episodes.read_text())
        assert data['completed'] and len(data['rows']) == 8
        assert [row['seed'] for row in data['rows']] == list(range(630100 + 8 * rank, 630108 + 8 * rank))
        assert all(row['native_counts']['controls'] == 600 for row in data['rows'])
        samples = sorted((output / 'sampler').glob(phase + '-*.json'))
        assert len(samples) == 12
        expected_rng = sampler[0]['paired_rng_sha256']
        assert json.loads(samples[0].read_text())['before_rng_sha256'] == expected_rng
        rows.append(dict(original_rank=rank, cases=8, successes=sum(row['success_once'] for row in data['rows']),
            episodes=str(episodes), episodes_sha256=sha(episodes), seconds=time.monotonic() - tick,
            environment_restore=environment, sampler_restore=sampler,
            canonical_snapshot_sha256=marker['snapshot_sha256'], sampler_calls=12,
            metrics={key: np.asarray(value).tolist() for key, value in metrics.items()}))
        write(output / 'PROGRESS.json', dict(completed=False, rows=rows, SGD=0))
        print(json.dumps(rows[-1]), flush=True)
    released = env.release_environments().wait()
    runner.metric_logger.finish()
    calls = [call for path in (output.parent / 'accounting').glob('rank-*/*.json')
        for call in json.loads(path.read_text())['calls']]
    assert calls and all(call['resolved'] for call in calls)
    charge = sum(call['executed_or_conservative'] for call in calls)
    assert charge == 38400
    write(output / 'RESULT.json', dict(completed=True, step=3, rows=rows, cases=64,
        successes=sum(row['successes'] for row in rows), policy_controls=38400, all_controls=charge,
        SGD=0, strict_load=loaded, released=released, model_sha256=admission['model_sha256'],
        original_rank_rng_restored=True, original_env_samplers_restored=True,
        fixed_eight_lane_batches=True, bitwise_gpu_kernels_required=False,
        execution_migration='Original8 rank-local B8 panels serialized on1 GPU; no training repeated.',
        script_sha256=sha(__file__), admission_sha256=sha(root / 'BASELINE-ENDPOINT-ADMISSION.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.root, args.output, args.deadline)


if __name__ == '__main__':
    main()
