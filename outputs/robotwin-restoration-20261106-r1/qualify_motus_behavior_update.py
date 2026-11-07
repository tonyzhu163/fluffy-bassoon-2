"""Actual retained-policy1 B32 update and native checkpoint roundtrip.

This qualifies the learner path only. It is not a fresh online learning run.
"""
import json
import os
from pathlib import Path
import shutil
import sys

from omegaconf import OmegaConf
import ray
import torch

from motus_teacher_entry import MotusActor, ROOT, validate, validate_cfg, Cluster, HybridComponentPlacement
from motus_learner_reference import audit
from audit_motus_checkpoint import candidate_digest
from optimizer_evidence import evidence
from optimizer_semantics import semantic_digest
from recovery_signature import digest
from fluffy_bassoon.meta_rl.rlinf_native_ppo import json_metrics, restore_rng, snapshot_rng, sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import PilotActor
from profile_checkpoint import verify as verify_archive

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


class BehaviorUpdateActor(MotusActor):
    def inspect(self):
        output = Path(self.cfg.native_pilot.output)
        checkpoint = Path(os.environ['TASK_LOCAL'])/'global_step_1'
        self.load_checkpoint(str(checkpoint/'actor'))
        self.pilot_restore(str(checkpoint/'pilot'))
        assert self.optimizer_steps == 15
        self.set_global_step(1)
        receipt = json.loads((BASE/f'motusresume01/state/training-rollouts/iteration-0002-policy-0001-rank-{self._rank:02d}.json').read_bytes())
        path = Path(receipt['path'])
        assert path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256']
        saved = read_tensor(path)
        assert saved['rank'] == self._rank and saved['optimizer_steps_before'] == 15
        assert saved['producing_policy_iteration'] == 1 and audit(saved['rollout_batch'],'ppo')['passed']
        self.rollout_batch = saved['rollout_batch']
        restore_rng(saved['rng_before_sgd']); del saved
        frozen = self._pilot_frozen_signature()
        metrics = self.run_training()
        assert self.optimizer_steps == 30 and self._pilot_frozen_signature() == frozen
        rng = snapshot_rng()
        policy = digest({n:p for n,p in self.model.named_parameters() if p.requires_grad})
        baseline = evidence(self)
        write(output/f'after-update-rank-{self._rank:02d}.json',baseline)
        baseline_digest, normalized = candidate_digest(baseline)
        old_error = None
        try: semantic_digest(baseline)
        except AssertionError as error: old_error = str(error)
        target = Path(os.environ['TASK_LOCAL'])/'diagnostic-save'
        self.save_checkpoint(str(target/'actor'),2)
        after = evidence(self)
        write(output/f'after-save-rank-{self._rank:02d}.json',after)
        assert candidate_digest(after)[0] == baseline_digest
        self.load_checkpoint(str(target/'actor'))
        restored = evidence(self)
        write(output/f'after-restore-rank-{self._rank:02d}.json',restored)
        assert candidate_digest(restored)[0] == baseline_digest
        assert digest({n:p for n,p in self.model.named_parameters() if p.requires_grad}) == policy
        assert self._pilot_frozen_signature() == frozen and self.optimizer_steps == 30
        restore_rng(rng); assert digest(snapshot_rng()) == digest(rng)
        PilotActor.pilot_capture(self,str(target/'pilot'),retain_rollout=False)
        write(target/f'pilot/motus-optimizer-layout-{self._rank:02d}.json',restored)
        write(target/f'pilot/motus-actor-signature-{self._rank:02d}.json',dict(
            optimizer_steps=30,trainable_parameters_sha256=policy,
            optimizer_sha256=restored['optimizer_sha256'],optimizer_semantic_version=1,
            optimizer_semantic_sha256=baseline_digest,
            lr_scheduler_sha256=digest(self.lr_scheduler.state_dict()),
            grad_scaler_sha256=digest(self.grad_scaler.state_dict()),rng_sha256=digest(rng)))
        row = dict(status='passed_actual_B32_update_checkpoint_roundtrip',rank=self._rank,
            native_adam_updates_per_rank=15, optimizer_steps=30, simulator_actions=0,
            micro_batch=32, accumulation=8, global_batch=2048, epochs=5,
            original_behavior_match_required=True, original_guard_failure=old_error,
            normalized_empty_unowned_names=normalized, native_metrics=json_metrics(metrics),
            source_collection_sha256=receipt['sha256'], owned_optimizer_state_exact=True,
            policy_exact_after_restore=True, frozen_parameters_exact=True,
            production_rule_changed=False, learning_gain_claim=False)
        write(output/f'rank-{self._rank:02d}.json',row)
        return row


def main():
    plan = json.loads((ROOT/'MOTUSBEHAVIORUPDATE01-CONTRACT.json').read_bytes())
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    torch.multiprocessing.set_start_method('spawn',force=True)
    cfg = OmegaConf.create(json.loads(Path(plan['original_resolved_config']).read_bytes()))
    output = Path(os.environ['TASK_OUT'])
    cfg.native_pilot.output = str(output)
    cfg.native_pilot.deadline_unix = int(os.environ['TASK_DEADLINE'])
    cfg.native_pilot.require_behavior_likelihood_match = True
    cfg.native_pilot.common_dev_noise_seed = 31000
    cfg.actor.micro_batch_size = 32
    for prefix,name in [('case','MOTUS-TRAIN-CASES.json'),('dev','MOTUS-DEV-CASES.json')]:
        cfg.native_pilot[f'{prefix}_manifest_path'] = str(ROOT/name)
    cfg.runner.logger.log_path = str(Path(os.environ['TASK_LOCAL'])/'logs')
    cfg.runner.logger.experiment_name = 'motus-behavior-update01'
    cfg.runner.per_worker_log_path = str(output/'workers')
    validate(cfg);cfg=validate_cfg(cfg)
    cluster=Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement=HybridComponentPlacement(cfg,cluster)
    actor=None
    try:
        actor=BehaviorUpdateActor.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy('actor'))
        actor.init_worker().wait();rows=actor.inspect().wait()
        assert len(rows)==8 and all(r['status']=='passed_actual_B32_update_checkpoint_roundtrip' for r in rows)
        target=Path(os.environ['TASK_LOCAL'])/'diagnostic-save'
        manifest=dict(global_step=2,scope='learner_only_retained_batch_qualification',
            source_commit=os.environ['TASK_SOURCE'],files={str(p.relative_to(target)):
                dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.rglob('*')) if p.is_file()})
        write(target/'MANIFEST.json',manifest)
        cold=Path(os.environ['TASK_COLD'])/'checkpoints/learner_step_2'
        assert not cold.exists();cold.parent.mkdir();shutil.copytree(target,cold)
        verify_archive(cold,manifest)
        write(output/'RESULT.json',dict(status='passed_actual_B32_update_checkpoint_roundtrip',
            source_commit=os.environ['TASK_SOURCE'],optimizer_updates=15,simulator_actions=0,
            actors=rows,production_rule_changed=False,fresh_online_collection=False,
            independent_cpu_readback_required=True,learning_gain_claim=False,
            retained_learner_checkpoint=str(cold),checkpoint_manifest_sha256=sha(cold/'MANIFEST.json'),
            checkpoint_scope='actor/Adam/scheduler/scaler/RNG; source collection retained; no simulator constructed'))
    finally:
        failure=sys.exc_info()[0] is not None;errors=[]
        if actor is not None and ray.is_initialized():
            try:actor._close()
            except Exception as error:errors.append(str(error))
        ray.shutdown()
        if errors:
            write(output/'cleanup-errors.json',dict(errors=errors,primary_failure_preserved=failure))
            if not failure:raise RuntimeError('Owned actor cleanup failed')


if __name__=='__main__':main()
