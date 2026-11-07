"""Complete native PPO from a qualified, unshuffled pre-SGD collection.

No simulator runs here. The retained scientific state stops after15 updates;
an unsaved update16 is only the independent recovery control.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import ray
import torch

from motus_teacher_entry import MotusActor, ROOT, validate, validate_cfg, Cluster, HybridComponentPlacement
from motus_learner_reference import audit
from optimizer_evidence import evidence
from recovery_signature import actor_signature, digest
from gradient_audit import audit as audit_gradients
from profile_checkpoint import verify
from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, json_metrics, restore_rng, sha, snapshot_rng, write
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import save_tensor
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import checkpoint_weight_export
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
SOURCE = BASE/'motusppo04'


def contract():
    value = json.loads((ROOT/'MOTUSREPLAY01-CONTRACT.json').read_bytes())
    assert value['status'] in ('prepared', 'admitted')
    assert value['closed_conservative_gpu_seconds'] + value['reserved_gpu_seconds'] <= 180000
    path = BASE/'collection-audit02/RESULT.json'
    assert sha(path) == value['collection_audit_sha256']
    audit_result = json.loads(path.read_bytes())
    assert audit_result['status'] == 'learner_replay_inputs_complete'
    assert audit_result['native_decisions'] == 6144 and len(audit_result['ranks']) == 4
    return value, audit_result


def configuration(output):
    """Keep the full original actor, learner and rollout numerical configuration."""
    original = json.loads((SOURCE/'state/resolved-native-config.json').read_bytes())
    cfg = OmegaConf.create(original)
    cfg.native_pilot.output = str(output)
    cfg.native_pilot.deadline_unix = int(os.environ['TASK_DEADLINE'])
    cfg.native_pilot.accounting_root = str(BASE/'motusreplay01/action-accounting')
    cfg.native_pilot.attempt = 'learner-replay01'
    for prefix, filename in [('case', 'MOTUS-TRAIN-CASES.json'), ('dev', 'MOTUS-DEV-CASES.json')]:
        cfg.native_pilot[f'{prefix}_manifest_path'] = str(ROOT/filename)
    cfg.runner.logger.log_path = str(Path(os.environ['TASK_LOCAL'])/'logs')
    cfg.runner.logger.experiment_name = 'motus-replay-'+output.name
    cfg.runner.per_worker_log_path = str(output/'workers')
    validate(cfg)
    return cfg, original


def comparable(rows):
    result = sorted(cpu_tree(rows), key=lambda r: r['rank'])
    for row in result:
        # Native DCP omits only qualified empty, unowned Adam shards.
        row['actor'].pop('optimizer_sha256')
    return result


class ReplayActor(MotusActor):
    def optimizer_step(self):
        before = self.optimizer_steps
        norm, rates = super().optimizer_step()
        value = float(norm)
        assert np.isfinite(value) and all(np.isfinite(float(r)) for r in rates)
        assert self.optimizer_steps == before+1
        write(Path(self.cfg.native_pilot.output)/'updates'/
              f'rank-{self._rank:02d}-step-{self.optimizer_steps:04d}.json',
              dict(rank=self._rank, optimizer_steps=self.optimizer_steps, grad_norm=value,
                   learning_rates=[float(r) for r in rates], unsaved_recovery_control=before == 15,
                   at_unix=time.time()))
        return norm, rates

    def signature(self):
        return dict(rank=self._rank, actor=actor_signature(self, snapshot_rng(), optimizer_layout=evidence(self)),
                    frozen=self._pilot_frozen_signature())

    def load_boundary(self, path, expected_steps):
        path = Path(path)
        self.load_checkpoint(str(path/'actor'))
        self.pilot_restore(str(path/'pilot'))
        assert self.optimizer_steps == expected_steps
        return self.signature()

    def load_collection(self):
        _, certified = contract()
        row = certified['ranks'][self._rank]
        path = Path(row['path'])
        assert path.parent == SOURCE/'state/training-rollouts'
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        saved = read_tensor(path)
        assert saved['rank'] == self._rank and saved['optimizer_steps_before'] == self.optimizer_steps == 0
        assert saved['collection_iteration'] == 1 and saved['producing_policy_iteration'] == 0
        assert not saved['evaluation_data'] and saved['dropped_fields'] == []
        self.rollout_batch = saved['rollout_batch']
        reference = audit(self.rollout_batch, 'ppo')
        assert reference['passed']
        assert len(saved['rng_before_sgd']['cuda']) == torch.cuda.device_count()
        write(Path(self.cfg.native_pilot.output)/f'collection-rank-{self._rank:02d}.json',
              dict(source=str(path), sha256=row['sha256'], independent_reference=reference,
                   saved_rng_sha256=digest(saved['rng_before_sgd']), optimizer_steps_before=0))
        # Last: loading, hashing and initialization consume no saved draws.
        restore_rng(saved['rng_before_sgd'])
        return dict(rank=self._rank, loaded=True, decisions=1536)

    def replay(self):
        assert self.optimizer_steps == 0
        result = self.run_training()  # Exact pinned post-shuffle loop and5 epochs.
        assert self.optimizer_steps == 15
        assert all(np.isfinite(np.asarray(value)).all() for value in result.values())
        return dict(rank=self._rank, optimizer_steps=15, metrics=json_metrics(result))

    def capture_boundary(self, path):
        path = Path(path)
        assert self.optimizer_steps == 15
        # Same RoboCasa export schedule helper: full DCP always, plain inference
        # weights only at this declared evaluation boundary.
        with checkpoint_weight_export(self._cfg.fsdp_config, step=1, final_step=1, export_steps=[1]):
            self.save_checkpoint(str(path/'actor'), 1)
        self.pilot_capture(str(path/'pilot'), retain_rollout=False)
        self._control_batch = cpu_tree(split_dict_to_chunk(self.rollout_batch, 3)[0])
        assert self._control_batch['prev_logprobs'].shape[0] == 512
        fixture = dict(rank=self._rank, optimizer_steps=15, batch=self._control_batch,
                       purpose='unsaved next-update numerical recovery control; no new policy lineage')
        saved = save_tensor(path/'control'/f'rank-{self._rank:02d}.pt.gz', fixture)
        self.rollout_batch = None
        gc.collect()
        return dict(rank=self._rank, control=saved, control_semantic_sha256=digest(fixture))

    def load_control(self, path, expected):
        rng = snapshot_rng()
        try:
            fixture = read_tensor(Path(path)/'control'/f'rank-{self._rank:02d}.pt.gz')
            assert fixture['rank'] == self._rank and fixture['optimizer_steps'] == self.optimizer_steps == 15
            assert digest(fixture) == next(r for r in expected if r['rank'] == self._rank)['control_semantic_sha256']
            self._control_batch = fixture['batch']
        finally:
            restore_rng(rng)
        return self.signature()

    def control_next_update(self):
        assert self.optimizer_steps == 15 and self.gradient_accumulation == 2
        if self.is_weight_offloaded: self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded: self.load_optimizer(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        metrics = {}
        for index, batch in enumerate(split_dict_to_chunk(self._control_batch, 2)):
            last = index == 1
            # Weights have changed since policy0, so ratio1 is not expected here.
            EmbodiedFSDPActor.train_micro_batch(self, batch, metrics, is_last=last)
            audit_gradients(self, is_last=last, ordinal=100+index)
            assert all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                       if p.requires_grad and p.grad is not None)
        norm, _ = self.optimizer_step()
        assert np.isfinite(float(norm)) and self.optimizer_steps == 16
        self.optimizer.zero_grad()
        return self.signature()

    def calibrate_export(self, path):
        """Actual plain-load bytes plus same-B4 frozen inference likelihoods."""
        assert self.optimizer_steps == 15
        rng = snapshot_rng()
        prior_training = self.model.training
        before = self.signature()
        report = dict(rank=self._rank, qualified=False, optimizer_steps=15)
        try:
            self.model.eval()
            batch = split_dict_to_chunk(self._control_batch, 128)[0]
            assert batch['prev_logprobs'].shape[0] == 4
            inputs = put_tensor_device(batch['forward_inputs'], self.device)
            with torch.no_grad(), self.amp_context:
                reference = self.model(forward_inputs=inputs, compute_logprobs=True,
                                       compute_values=True, compute_entropy=False, use_cache=False)
            if self._rank == 0:
                paths = list((Path(path)/'actor').rglob('full_weights.pt'))
                assert len(paths) == 1
                weights = torch.load(paths[0], map_location='cpu', weights_only=True)
                assert weights and all(type(v) is torch.Tensor and v.device.type == 'cpu' for v in weights.values())
                import rlinf.models.embodiment.openpi as factory
                plain = factory.get_model(self.cfg.actor.model).to(self.device)
                loaded = plain.load_state_dict(weights, strict=True)
                assert not loaded.missing_keys and not loaded.unexpected_keys
                actual = plain.state_dict()
                assert actual.keys() == weights.keys()
                for key, value in weights.items():
                    assert actual[key].dtype == value.dtype and actual[key].shape == value.shape, key
                    assert digest(actual[key]) == digest(value), key
                plain.eval().requires_grad_(False)
                with torch.no_grad(), self.amp_context:
                    result = plain(forward_inputs=inputs, compute_logprobs=True,
                                   compute_values=True, compute_entropy=False, use_cache=False)
                errors = {key: float((result[key].float()-reference[key].float()).abs().max())
                          for key in ('logprobs', 'values')}
                assert all(np.isfinite(v) for v in errors.values())
                assert errors['logprobs'] <= 1e-3 and errors['values'] <= 1e-4, errors
                report.update(qualified=True, full_weights=str(paths[0]), sha256=sha(paths[0]),
                              strict_plain_tensor_load=True, all_tensor_bytes_dtypes_shapes_exact=True,
                              same_batch=4, max_abs_error=errors, new_simulator_controls=0)
                del actual, plain, weights
                gc.collect()
            del reference, inputs
        finally:
            self.model.train(prior_training)
            restore_rng(rng)
        assert self.signature() == before, 'Inference export changed actor/Adam/scheduler/scaler/RNG'
        report['actor_state_unchanged'] = True
        write(Path(self.cfg.native_pilot.output)/f'export-rank-{self._rank:02d}.json', report)
        return report


def stage(cold, target, expected_sha):
    assert not target.exists() and sha(cold/'MANIFEST.json') == expected_sha
    manifest = json.loads((cold/'MANIFEST.json').read_bytes())
    shutil.copytree(cold, target)
    verify(target, manifest)
    assert sha(target/'MANIFEST.json') == expected_sha


def preserve(checkpoint, cold, expected):
    assert not cold.exists()
    files = {str(p.relative_to(checkpoint)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(checkpoint.rglob('*')) if p.is_file()}
    assert sum(r['bytes'] for r in files.values()) <= 48*2**30
    manifest = dict(source_commit=os.environ['TASK_SOURCE'], files=files,
                    native_optimizer_steps=15, logical_policy_iteration=1,
                    boundary='complete native learner; collector continuation requires explicit migration',
                    expected=expected)
    write(checkpoint/'MANIFEST.json', manifest)
    shutil.copytree(checkpoint, cold)
    verify(cold, manifest)
    back = Path(os.environ['TASK_LOCAL'])/'stageback'
    stage(cold, back, sha(checkpoint/'MANIFEST.json'))
    shutil.rmtree(back)
    return dict(completed=True, cold=str(cold), manifest_sha256=sha(cold/'MANIFEST.json'),
                data_bytes=sum(r['bytes'] for r in files.values()), full_cold_readback=True,
                full_local_stageback=True, optimizer_steps=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'restore'], required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    terms, _ = contract()
    assert terms['status'] == 'admitted'
    signals, local = Path(os.environ['TASK_OUT']), Path(os.environ['TASK_LOCAL'])
    output = signals/args.phase
    output.mkdir()  # Native validate_cfg also creates output children.
    cfg, original = configuration(output)
    cfg = validate_cfg(cfg)
    validate(cfg)
    for key in ('actor', 'algorithm', 'rollout'):
        assert OmegaConf.to_container(cfg[key], resolve=True) == original[key], key
    write(output/'resolved-native-config.json', OmegaConf.to_container(cfg, resolve=True))
    report = dict(status='starting', phase=args.phase, source_commit=os.environ['TASK_SOURCE'],
                  new_simulator_controls=0, scientific_source_admission=False)
    write(output/'RESULT.json', report)
    if args.phase == 'train':
        checkpoint = local/'initial0'
        stage(SOURCE/'checkpoints/global_step_0', checkpoint, terms['checkpoint0_manifest_sha256'])
    else:
        receipt = json.loads((signals/'CHECKPOINT.json').read_bytes())
        assert receipt['completed'] and receipt['optimizer_steps'] == 15
        checkpoint = local/'restored1'
        stage(Path(receipt['cold']), checkpoint, receipt['manifest_sha256'])
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    actor = ReplayActor.create_group(cfg).launch(cluster, name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy('actor'))
    started = time.monotonic()
    try:
        actor.init_worker().wait()
        restored = actor.load_boundary(str(checkpoint), 0 if args.phase == 'train' else 15).wait()
        if args.phase == 'train':
            actor.load_collection().wait()
            report['training'] = actor.replay().wait()
            write(output/'RESULT.json', report)
            boundary = local/'global_step_1'
            controls = actor.capture_boundary(str(boundary)).wait()
            before = actor.signature().wait()
            expected = dict(before=before, controls=controls)
            write(boundary/'EXPECTED.json', expected)
            write(boundary/'logical-sampler.json', dict(policy_iteration=1, optimizer_steps=15,
                source_case_next_reset_ordinal=4, actor_shuffle_rule='native cfg.actor.seed+rank; private generator per iteration',
                source_collection=str(SOURCE), exact_all_worker_resume=False,
                next_collector='explicit fresh bootstrap/RNG migration required before continuing online'))
            receipt = preserve(boundary, BASE/'motusreplay01/checkpoints/global_step_1', expected)
            write(signals/'CHECKPOINT.json', receipt)
            # Preserve the completed scientific boundary before any extra check.
            after = actor.control_next_update().wait()
            control = dict(after_control=after, checkpoint_manifest_sha256=receipt['manifest_sha256'])
            write(signals/'EXPECTED_CONTROL.json', control)
            write(signals/'CONTROL-RECEIPT.json', dict(sha256=sha(signals/'EXPECTED_CONTROL.json')))
            report.update(status='complete', retained_optimizer_steps=15, unsaved_recovery_control_step=16,
                          checkpoint=receipt, independent_process_restore_still_required=True)
        else:
            expected = json.loads((checkpoint/'EXPECTED.json').read_bytes())
            control_path = signals/'EXPECTED_CONTROL.json'
            assert sha(control_path) == json.loads((signals/'CONTROL-RECEIPT.json').read_bytes())['sha256']
            control = json.loads(control_path.read_bytes())
            assert control['checkpoint_manifest_sha256'] == sha(checkpoint/'MANIFEST.json')
            assert comparable(restored) == comparable(expected['before'])
            loaded = actor.load_control(str(checkpoint), expected['controls']).wait()
            assert comparable(loaded) == comparable(expected['before'])
            exports = actor.calibrate_export(str(checkpoint)).wait()
            assert all(r['actor_state_unchanged'] for r in exports)
            assert next(r for r in exports if r['rank'] == 0)['qualified']
            after = actor.control_next_update().wait()
            assert comparable(after) == comparable(control['after_control'])
            report.update(status='complete', retained_optimizer_steps=15, full_owned_restore_exact=True,
                          next_update_exact=True, exports=exports, scientific_optimizer_updates_in_phase=0,
                          unsaved_recovery_control_step=16)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json', report)
        actor._close()
        ray.shutdown()


if __name__ == '__main__':
    main()
