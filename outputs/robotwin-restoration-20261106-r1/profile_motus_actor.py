"""Native Motus FSDP actor capacity on real inputs, synthetic learning targets.

No scientific policy or initialization is produced. Global2048 and native loss
parameters stay fixed. Native Flow-SDE generates the fixture, not ODE case_noise.
"""
import gc
import gzip
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import statistics
import time
from types import MethodType

import hydra
import numpy as np
import ray
import torch
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, snapshot_rng, write
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import PilotActor
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import keep_frozen_image_encoder_in_eval, flow_logprob_parity
from fluffy_bassoon.rl.robotwin_motor import stack_public_observations
from gradient_audit import audit
from optimizer_evidence import evidence
from recovery_signature import actor_signature, digest
from qualify_batch_runtime import write_trace
from qualify_motor import BASE, READOUT, Memory, sha


def take(value, index):
    if torch.is_tensor(value): return value.index_select(0, torch.as_tensor(index)).contiguous()
    if isinstance(value, np.ndarray): return value[index].copy()
    if isinstance(value, dict): return {k:take(v,index) for k,v in value.items()}
    if isinstance(value, list): return [value[int(i)] for i in index]
    raise TypeError(f'Unexpected batched fixture field: {type(value).__name__}')


class ActorMemory(Memory):
    def __init__(self, device):
        self.device = device
        super().__init__(torch)

    def poll(self):
        # CUDA current-device state is thread-local when several GPUs are visible.
        torch.cuda.set_device(self.device)
        super().poll()


class MotusBenchmarkActor(PilotActor):
    def model_provider_func(self):
        from fluffy_bassoon.baige_gpu import mapping, cuda_pci, pci
        count = torch.cuda.device_count()
        assert count in (1,4)
        torch.cuda.set_device(0 if count == 1 else self._rank)
        actual = cuda_pci(torch.cuda.current_device())
        rows = [r for r in mapping() if pci(r['pci_bus_id']) == actual]
        assert len(rows) == 1
        os.environ['LYCO_BAIGE_ISOLATED_UUID'] = rows[0]['uuid']
        self._profile_root = Path(self.cfg.native_pilot.output)
        self._local = Path(os.environ['TASK_LOCAL'])
        write(self._profile_root/f'gpu-rank-{self._rank}.json', dict(pci=actual,
            uuid=rows[0]['uuid'],visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
        model = super().model_provider_func()
        assert model.action_horizon == model.action_chunk == 32 and model.action_env_dim == 14
        assert model.rl_cfg.train_expert_only and model.rl_cfg.add_value_head
        assert model.rl_cfg.noise_method == 'flow_sde' and model.rl_cfg.noise_level == .3
        assert not any(p.requires_grad for p in model.img.parameters())
        original_train = model.train

        def train(module, mode=True):
            result = original_train(mode)
            keep_frozen_image_encoder_in_eval(module)
            return result

        model.train = MethodType(train, model)
        model.eval()
        if os.environ['TASK_PROFILE_PHASE'] == 'profile':
            manifest = json.loads((BASE/'outputs/robotwin-restoration-20261106-r1/motor03/INPUTS.json').read_text())
            paths = manifest['paths'][self._rank*64:(self._rank+1)*64]
            assert len(paths) == len(set(paths)) == 64
            observations, files = [], []
            for relative in paths:
                path = READOUT/relative
                with np.load(path, allow_pickle=False) as data:
                    images, state, prompt = data['images'].copy(), data['state'].copy(), str(data['instruction'].item())
                observations.append(dict(main_images=images[:1],wrist_images=images[None,1:],
                    states=state[None].astype(np.float32),task_descriptions=[prompt]))
                files.append(dict(path=str(path),sha256=sha(path)))
            model.cuda()
            random.seed(50000+self._rank)
            rng = torch.Generator(device=model.device).manual_seed(60000+self._rank)
            with torch.no_grad():
                actions, sampled = model.predict_action_batch(stack_public_observations(observations),
                    mode='train',compute_values=True,rng=rng)
            self._fixture = cpu_tree(dict(forward_inputs=sampled['forward_inputs'],
                native_logprobs=sampled['prev_logprobs'],native_mean_values=sampled['prev_values'],
                actions=actions,source_files=files,sampler=dict(kind='fixed real fixture repeated to512 rows',
                    next_data_cursor=0,constructor_seed=1234,python_seed=50000+self._rank,
                    gaussian_seed=60000+self._rank,gaussian_generator_state_after=rng.get_state())))
            model.cpu(); del sampled, actions, observations
            write_trace(self._local/f'fixture-rank-{self._rank}.pt.gz', self._fixture)
            gc.collect(); torch.cuda.empty_cache()
        else:
            with gzip.open(self._local/'restored'/f'fixture-rank-{self._rank}.pt.gz','rb') as stream:
                self._fixture = torch.load(stream,map_location='cpu',weights_only=False)
        self._fixture_signature = digest(self._fixture)
        write(self._profile_root/f'fixture-rank-{self._rank}.json', dict(sha256=self._fixture_signature,
            source_files=self._fixture['source_files'],native_batch=64,
            native_value_protocol='rollout mean across denoising calls; benchmark returns use selected-step actor values only'))
        self._audit_ordinal = 0
        return model

    def signature(self):
        state = snapshot_rng()
        return dict(rank=self._rank,fixture_sha256=self._fixture_signature,
            actor=actor_signature(self,state,optimizer_layout=evidence(self)),
            frozen=self._pilot_frozen_signature())

    def checkpoint(self, path):
        self.save_checkpoint(str(Path(path)/'actor'), self.optimizer_steps)
        return self.pilot_capture(str(Path(path)/'pilot'),retain_rollout=False)

    def restore_profile(self, path):
        self.load_checkpoint(str(Path(path)/'actor'))
        self.pilot_restore(str(Path(path)/'pilot'))
        assert self.optimizer_steps == 4
        return self.signature()

    def _reference(self, inputs, micro):
        parts = []
        with torch.no_grad(), self.amp_context:
            for start in range(0,512,micro):
                part = put_tensor_device(take(inputs,range(start,start+micro)),self.device)
                output = self.model(forward_inputs=part,compute_logprobs=True,compute_values=True,
                                    compute_entropy=False,use_cache=False)
                parts.append((output['logprobs'].float().cpu(),output['values'].float().cpu().reshape(-1,1)))
                del output, part
        return torch.cat([p[0] for p in parts]),torch.cat([p[1] for p in parts])

    def benchmark(self, micro, updates=4, *, control=False):
        assert self._world_size == 4 and micro in (64,128,256,512) and 512 % micro == 0
        self.cfg.actor.micro_batch_size = micro
        self.gradient_accumulation = 512//micro
        name = f'micro-{micro}' if not control else 'next-update-control'
        path = self._profile_root/name/f'rank-{self._rank}.json'
        report = dict(rank=self._rank,micro=micro,accumulation=self.gradient_accumulation,
            global_batch_size=2048,scientific_training=False,updates=[],
            source_rows=64,local_rows=512,repeated_real_rows_for_systems_measurement=True,
            value_target='synthetic returns around selected-step actor values; not scientific GAE')
        write(path,report)
        try:
            if self.is_weight_offloaded: self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded: self.load_optimizer(self.device)
            self.model.train()
            inputs = take(self._fixture['forward_inputs'],torch.arange(512)%64)
            torch.distributed.barrier(); torch.cuda.synchronize(); started = time.monotonic()
            logprobs, values = self._reference(inputs,micro)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.monotonic()-started,device=self.device)
            torch.distributed.all_reduce(elapsed,op=torch.distributed.ReduceOp.MAX)
            report['reference_seconds'] = float(elapsed)
            if micro == 64 and self.optimizer_steps == 0:
                parity = flow_logprob_parity(logprobs,self._fixture['native_logprobs'].repeat(8,1,1),
                                            action_dim=14,tolerance=1e-3)
                report['initial_native_likelihood_replay'] = parity
                write(path,report); assert parity['passed'], 'Native rollout/actor chunk likelihood drift'
            repeat_logprobs, repeat_values = self._reference(inputs,micro)
            assert torch.equal(logprobs,repeat_logprobs) and torch.equal(values,repeat_values)
            report['same_partition_ratio_one'] = True
            advantages = torch.linspace(-1,1,512)[:,None]
            batch = dict(forward_inputs=inputs,prev_logprobs=logprobs,prev_values=values,
                advantages=advantages,returns=values+advantages,loss_mask=torch.ones((512,1),dtype=torch.bool))
            for update in range(updates):
                assert time.time() < float(os.environ['TASK_DEADLINE'])-180
                torch.distributed.barrier(); torch.cuda.synchronize()
                with ActorMemory(self.device) as memory:
                    started = time.monotonic(); self.optimizer.zero_grad(); metrics = {}
                    for start in range(0,512,micro):
                        part = take(batch,range(start,start+micro))
                        is_last = start+micro == 512
                        EmbodiedFSDPActor.train_micro_batch(self,part,metrics,is_last=is_last)
                        if update == 0 and not control:
                            audit(self,is_last=is_last,ordinal=self._audit_ordinal)
                            self._audit_ordinal += 1
                        if is_last:
                            assert all(p.grad is None for p in self.model.parameters() if not p.requires_grad)
                        del part
                    self.torch_platform.empty_cache()
                    grad_norm,_ = self.optimizer_step(); self.optimizer.zero_grad()
                    torch.cuda.synchronize()
                    elapsed = torch.tensor(time.monotonic()-started,device=self.device)
                    torch.distributed.all_reduce(elapsed,op=torch.distributed.ReduceOp.MAX)
                row = dict(update=update,warmup=update==0,seconds=float(elapsed),
                    grad_norm=float(grad_norm),**memory.result())
                assert np.isfinite(row['grad_norm'])
                reserve = max(8*2**30,int(.08*row['total_bytes']))
                row['memory_safe'] = row['conservative_minimum_free_bytes'] >= reserve
                report['updates'].append(row); write(path,report)
            assert self._pilot_frozen_signature() == self._pilot_frozen_before
            report.update(status='complete',frozen_unchanged=True,optimizer_steps=self.optimizer_steps,
                memory_safe=all(v['memory_safe'] for v in report['updates']))
            if updates > 1:
                report['steady_median_seconds'] = statistics.median(v['seconds'] for v in report['updates'][1:])
            write(path,report); return report
        except BaseException as error:
            report.update(status='failed',error_type=type(error).__name__,error=str(error))
            write(path,report); raise


def comparable(rows):
    result = deepcopy(rows)
    for row in result:
        # Keep the raw hash in evidence; compare all owned Adam tensors/groups
        # with the measured native zero-length-shard representation contract.
        row['actor'].pop('optimizer_sha256')
    return result


@hydra.main(version_base='1.1',config_path=str(Path(os.environ['TASK_CODE'])/'configs/native_ppo'),config_name='motus_phone_two')
def main(cfg):
    from profile_checkpoint import preserve, stage
    cfg.actor.enable_offload = False
    cfg = validate_cfg(cfg)
    root, local = Path(cfg.native_pilot.output),Path(os.environ['TASK_LOCAL'])
    phase = os.environ['TASK_PROFILE_PHASE']
    if phase == 'restore': stage(local/'restored')
    cluster = Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg,cluster)
    actor = MotusBenchmarkActor.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy('actor'))
    report = dict(status='starting',phase=phase,scientific_training=False,
        source_commit=os.environ['TASK_SOURCE'],candidates=[])
    write(root/'RESULT.json',report)
    try:
        actor.init_worker().wait()
        if phase == 'profile':
            for micro in (64,128,256,512):
                rows = actor.benchmark(micro).wait()
                report['candidates'].append(dict(micro=micro,ranks=rows));write(root/'RESULT.json',report)
                if micro == 64:
                    checkpoint = local/'checkpoint'
                    before = actor.signature().wait()
                    actor.checkpoint(str(checkpoint)).wait()
                    after = actor.signature().wait()
                    assert comparable(before) == comparable(after)
                    preserve(checkpoint,local)
                    # A reference next update makes the saved boundary reviewable.
                    actor.benchmark(64,updates=1,control=True).wait()
                    expected = actor.signature().wait()
                    write(root/'RECOVERY-EXPECTED.json',dict(before=before,next_update=expected))
                if not all(v['memory_safe'] for v in rows): break
            report.update(status='complete',scientific_optimizer_updates=0)
        else:
            actual = actor.restore_profile(str(local/'restored')).wait()
            expected = json.loads((root.parent/'profile/RECOVERY-EXPECTED.json').read_text())
            write(root/'RESTORE-ACTUAL.json',dict(before=actual))
            assert comparable(actual) == comparable(expected['before'])
            actor.benchmark(64,updates=1,control=True).wait()
            after = actor.signature().wait()
            write(root/'RESTORE-ACTUAL.json',dict(before=actual,next_update=after))
            assert comparable(after) == comparable(expected['next_update'])
            report.update(status='complete',full_recovery_exact=True,next_update_exact=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        write(root/'RESULT.json',report)
        actor._close();ray.shutdown()


if __name__ == '__main__': main()
