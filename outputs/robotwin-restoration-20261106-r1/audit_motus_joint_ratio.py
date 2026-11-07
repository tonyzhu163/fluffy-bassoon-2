"""Read-only eight-rank replay of actual policy0 chunk likelihood references."""
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import ray
import torch

from motus_teacher_entry import MotusActor, ROOT, validate, validate_cfg, Cluster, HybridComponentPlacement
from motus_learner_reference import audit
from recovery_signature import digest
from fluffy_bassoon.meta_rl.rlinf_native_ppo import restore_rng, sha, snapshot_rng, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from rlinf.utils.nested_dict_process import process_nested_dict_for_train

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
SOURCE = BASE/'motusoptimized03'


def describe(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    assert len(x) and np.isfinite(x).all()
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
        minimum=float(x.min()), maximum=float(x.max()),
        quantiles=dict(zip(['p01','p05','p50','p95','p99'],
                          map(float, np.quantile(x, [.01,.05,.5,.95,.99])))))


class RatioActor(MotusActor):
    def inspect(self):
        plan = json.loads((ROOT/'MOTUSJOINTAUDIT01-CONTRACT.json').read_bytes())
        checkpoint = Path(os.environ['TASK_LOCAL'])/'global_step_0'
        self.load_checkpoint(str(checkpoint/'actor'))
        self.pilot_restore(str(checkpoint/'pilot'))
        assert self.optimizer_steps == 0
        receipt = json.loads((SOURCE/f'state/training-rollouts/iteration-0001-policy-0000-rank-{self._rank:02d}.json').read_bytes())
        path = Path(receipt['path'])
        assert sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes']
        saved = read_tensor(path)
        assert saved['rank'] == self._rank and saved['optimizer_steps_before'] == 0
        assert saved['producing_policy_iteration'] == 0 and audit(saved['rollout_batch'], 'ppo')['passed']
        self.rollout_batch = saved['rollout_batch']; rng = saved['rng_before_sgd']
        del saved
        restore_rng(rng)
        if self.is_weight_offloaded: self.load_param_and_grad(self.device)
        self.model.train()  # Same patched frozen-image eval behavior as native training.
        frozen_before = self._pilot_frozen_signature()
        policy_before = digest({n:p for n,p in self.model.named_parameters() if p.requires_grad})
        size = int(np.prod(self.rollout_batch['prev_logprobs'].shape[:2]))
        generator = torch.Generator().manual_seed(self.cfg.actor.seed+self._rank)
        shuffle = torch.randperm(size, generator=generator)
        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(self.rollout_batch, shuffle)
        old = self.rollout_batch['prev_logprobs'].clone().cpu()
        assert old.shape == (768,32,14)
        mask = self.rollout_batch['loss_mask'].bool().reshape(768,-1).any(-1).cpu()
        advantages = self.rollout_batch['advantages'].reshape(768).float().cpu()
        report = dict(rank=self._rank, optimizer_updates=0, simulator_actions=0,
            source_collection_sha256=receipt['sha256'], valid_decisions=int(mask.sum()),
            native_shuffle_seed=self.cfg.actor.seed+self._rank, comparisons={})
        fixtures = dict(original_rollout_logprobs=old, loss_mask=mask, advantages=advantages)
        for micro in (256,32):
            self.cfg.actor.micro_batch_size = micro
            self.rollout_batch['prev_logprobs'] = old.clone()
            restore_rng(rng)
            torch.cuda.reset_peak_memory_stats(self.device)
            started = time.monotonic()
            metrics = self.recompute_prev_logprobs(256)
            replay = self.rollout_batch['prev_logprobs'].cpu()
            # Match the actual loss's separate float32 sums over H and action_dim.
            joint = replay.float().sum((1,2))-old.float().sum((1,2))
            exact_difference = (replay.double()-old.double()).sum((1,2))
            ratio = joint.double().exp()
            assert torch.isfinite(ratio).all()
            clipped = ratio.clamp(.8,1.2)
            would_clip_loss = (-advantages.double()*ratio < -advantages.double()*clipped)
            name = f'actor_batch_{micro}'
            report['comparisons'][name] = dict(
                coordinate_mae=float((replay-old).abs()[mask].mean()), native_metrics=metrics,
                joint_log_ratio=describe(joint[mask].numpy()),
                ratio=describe(ratio[mask].numpy()),
                ratio_outside_08_12_fraction=float(((ratio<.8)|(ratio>1.2))[mask].float().mean()),
                clipped_surrogate_fraction_without_sgd=float(would_clip_loss[mask].float().mean()),
                separate_float32_sum_vs_float64_difference_max_abs=float((joint.double()-exact_difference).abs()[mask].max()),
                seconds=time.monotonic()-started, peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            fixtures[name] = replay
        self.cfg.actor.micro_batch_size = 256
        assert self.optimizer_steps == 0 and self._pilot_frozen_signature() == frozen_before
        assert digest({n:p for n,p in self.model.named_parameters() if p.requires_grad}) == policy_before
        assert all(p.grad is None for p in self.model.parameters())
        output = Path(self.cfg.native_pilot.output)
        torch.save(fixtures, output/f'joint-ratios-rank-{self._rank:02d}.pt')
        report.update(status='passed_read_only_likelihood_measurement', parameters_unchanged=True,
            fixture_sha256=sha(output/f'joint-ratios-rank-{self._rank:02d}.pt'),
            learning_gain_claim=False, root_cause_established=False)
        write(output/f'rank-{self._rank:02d}.json', report)
        self.rollout_batch = None; gc.collect()
        return report


def main():
    plan = json.loads((ROOT/'MOTUSJOINTAUDIT01-CONTRACT.json').read_bytes())
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    torch.multiprocessing.set_start_method('spawn', force=True)
    output = Path(os.environ['TASK_OUT'])
    cfg = OmegaConf.create(json.loads((SOURCE/'state/resolved-native-config.json').read_bytes()))
    cfg.native_pilot.output = str(output)
    cfg.native_pilot.deadline_unix = int(os.environ['TASK_DEADLINE'])
    for prefix,name in [('case','MOTUS-TRAIN-CASES.json'),('dev','MOTUS-DEV-CASES.json')]:
        cfg.native_pilot[f'{prefix}_manifest_path'] = str(ROOT/name)
    cfg.runner.logger.log_path = str(Path(os.environ['TASK_LOCAL'])/'logs')
    cfg.runner.logger.experiment_name = 'motus-joint-audit01'
    cfg.runner.per_worker_log_path = str(output/'workers')
    assert not cfg.runner.resume_dir and int(cfg.native_pilot.world_size) == 8
    validate(cfg); cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = None
    try:
        actor = RatioActor.create_group(cfg).launch(cluster, name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy('actor'))
        actor.init_worker().wait()
        rows = actor.inspect().wait()
        assert len(rows) == 8 and all(r['status']=='passed_read_only_likelihood_measurement' for r in rows)
        write(output/'RESULT.json', dict(status='completed_read_only_joint_likelihood_measurement',
            source_commit=os.environ['TASK_SOURCE'], optimizer_updates=0, simulator_actions=0,
            actors=rows, root_cause_established=False, independent_cpu_readback_required=True))
    finally:
        failure = sys.exc_info()[0] is not None
        errors = []
        if actor is not None and ray.is_initialized():
            try: actor._close()
            except Exception as error: errors.append(str(error))
        ray.shutdown()
        if errors:
            write(output/'cleanup-errors.json', dict(errors=errors, primary_failure_preserved=failure))
            if not failure: raise RuntimeError('Owned actor cleanup failed')


if __name__ == '__main__': main()
