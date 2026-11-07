"""Measure restored FSDP optimizer state before/after a native DCP save.

No SGD and no simulator. The candidate normalization is diagnostic only:
the production checkpoint guard stays unchanged until this measurement passes.
"""
import copy
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf
import ray
import torch

from motus_teacher_entry import MotusActor, ROOT, validate, validate_cfg, Cluster, HybridComponentPlacement
from optimizer_evidence import evidence
from optimizer_semantics import semantic_digest
from recovery_signature import actor_signature, digest
from fluffy_bassoon.meta_rl.rlinf_native_ppo import restore_rng, snapshot_rng, write


def candidate_digest(layout):
    """Reuse the strict v1 rule after one explicitly described representation fix."""
    candidate = copy.deepcopy(layout)
    normalized = []
    for name, row in candidate['parameters'].items():
        if (row['registered_model_parameter'] and row['requires_grad']
                and row['parameter_shape'] == [0] and row['parameter_numel'] == 0
                and not row['local_nonempty_parameter'] and row['state_present']
                and row['state'] == {}):
            row['state_present'] = False
            normalized.append(name)
    return semantic_digest(candidate), normalized


class CheckpointActor(MotusActor):
    def inspect(self):
        output = Path(self.cfg.native_pilot.output)
        local = Path(os.environ['TASK_LOCAL'])
        checkpoint = local/'global_step_1'
        self.load_checkpoint(str(checkpoint/'actor'))
        self.pilot_restore(str(checkpoint/'pilot'))
        assert self.optimizer_steps == 15
        before_rng = snapshot_rng()
        before = evidence(self)
        before_signature = actor_signature(self, before_rng, optimizer_layout=before)
        write(output/f'before-rank-{self._rank:02d}.json', before)
        target = local/'diagnostic-save'
        self.save_checkpoint(str(target/'actor'), 1)
        after = evidence(self)
        write(output/f'after-save-rank-{self._rank:02d}.json', after)
        after_digest, normalized = candidate_digest(after)
        # Owned Adam moments and every group option still use the original
        # exact named-byte comparison; no state content is dropped there.
        assert after_digest == before_signature['optimizer_semantic_sha256']
        assert digest({n:p for n,p in self.model.named_parameters() if p.requires_grad}) == before_signature['trainable_parameters_sha256']
        old_error = None
        try:
            semantic_digest(after)
        except AssertionError as error:
            old_error = str(error)
        assert old_error is not None and normalized, 'Expected rejected empty mapping was not reproduced'
        assert all(not p.grad.numel() for p in self.model.parameters()
                   if p.numel() == 0 and p.grad is not None)
        self.load_checkpoint(str(target/'actor'))
        restored = evidence(self)
        write(output/f'after-restore-rank-{self._rank:02d}.json', restored)
        restored_digest, restored_empty = candidate_digest(restored)
        assert restored_digest == before_signature['optimizer_semantic_sha256']
        assert self.optimizer_steps == 15
        assert digest({n:p for n,p in self.model.named_parameters() if p.requires_grad}) == before_signature['trainable_parameters_sha256']
        restore_rng(before_rng)
        assert digest(snapshot_rng()) == before_signature['rng_sha256']
        result = dict(status='passed_measured_empty_unowned_mapping_roundtrip', rank=self._rank,
            optimizer_updates=0, optimizer_steps=15, simulator_actions=0,
            original_guard_failure=old_error, normalized_names=normalized,
            after_restore_empty_names=restored_empty, owned_optimizer_state_exact=True,
            policy_exact=True, rng_restored_exactly=True, candidate_semantic_sha256=after_digest,
            production_rule_changed=False, learning_gain_claim=False)
        write(output/f'rank-{self._rank:02d}.json', result)
        return result


def main():
    plan = json.loads((ROOT/'MOTUSCHECKPOINTAUDIT01-CONTRACT.json').read_bytes())
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    torch.multiprocessing.set_start_method('spawn', force=True)
    output = Path(os.environ['TASK_OUT'])
    source = Path(plan['original_resolved_config'])
    cfg = OmegaConf.create(json.loads(source.read_bytes()))
    cfg.native_pilot.output = str(output)
    cfg.native_pilot.deadline_unix = int(os.environ['TASK_DEADLINE'])
    for prefix,name in [('case','MOTUS-TRAIN-CASES.json'),('dev','MOTUS-DEV-CASES.json')]:
        cfg.native_pilot[f'{prefix}_manifest_path'] = str(ROOT/name)
    cfg.runner.logger.log_path = str(Path(os.environ['TASK_LOCAL'])/'logs')
    cfg.runner.logger.experiment_name = 'motus-checkpoint-audit01'
    cfg.runner.per_worker_log_path = str(output/'workers')
    assert not cfg.runner.resume_dir and int(cfg.native_pilot.world_size) == 8
    validate(cfg); cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = None
    try:
        actor = CheckpointActor.create_group(cfg).launch(cluster, name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy('actor'))
        actor.init_worker().wait()
        rows = actor.inspect().wait()
        assert len(rows) == 8 and all(r['status']=='passed_measured_empty_unowned_mapping_roundtrip' for r in rows)
        write(output/'RESULT.json', dict(status='passed_measured_checkpoint_representation_roundtrip',
            source_commit=os.environ['TASK_SOURCE'], optimizer_updates=0, simulator_actions=0,
            actors=rows, production_rule_changed=False, independent_cpu_readback_required=True))
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
