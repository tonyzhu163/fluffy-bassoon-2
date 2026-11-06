"""Restore the qualified four-rank endpoint and qualify a plain inference copy."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

CODE = Path(os.environ['TASK_CODE'])
PAIRED = CODE/'outputs/robotwin-sidney-matched-baselines-20261106-r1'
sys.path.insert(0,str(PAIRED))

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from paired_entry import PairedActor
from paired_recipe import validate_pair
from recovery_signature import actor_signature
from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng,restore_rng,write


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def raw_hash(tensor):
    value = tensor.detach().cpu().reshape(-1).contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class ExportActor(PairedActor):
    def export_inference(self, output):
        output = Path(output); assert self.optimizer_steps == 40
        rng = snapshot_rng(); before = actor_signature(self,rng)
        # Native FSDP gathers its logical full weights. All ranks participate;
        # cpu_offload/full_state_dict returns the saved full state on rank0.
        state = self.get_model_state_dict(cpu_offload=True,full_state_dict=True)
        result = dict(rank=self._rank,optimizer_steps=40,extra_optimizer_updates=0,new_simulator_controls=0)
        try:
            if self._rank == 0:
                assert state and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in state.values())
                assert all(type(v) is torch.Tensor for v in state.values()), 'Logical full tensors required'
                packed = {k:v.detach().clone().contiguous() for k,v in state.items()}
                parameter_bytes = sum(v.numel()*v.element_size() for v in packed.values())
                assert parameter_bytes < 12*2**30
                target = output/'policy.pt'; temporary = output/'policy.pt.writing'
                assert not target.exists() and not temporary.exists()
                torch.save(packed,temporary); temporary.replace(target)
                disk = torch.load(target,map_location='cpu',weights_only=True)
                assert disk.keys() == packed.keys()
                for k in packed:
                    assert packed[k].dtype == disk[k].dtype and packed[k].shape == disk[k].shape
                    assert raw_hash(packed[k]) == raw_hash(disk[k]), k
                del packed; gc.collect()
                import rlinf.models.embodiment.openpi as factory
                plain = factory.get_model(self.cfg.actor.model).to(self.device)
                loaded = plain.load_state_dict(disk,strict=True)
                assert not loaded.missing_keys and not loaded.unexpected_keys
                actual = plain.state_dict(); assert actual.keys() == disk.keys()
                tensor_rows = []
                for k,v in disk.items():
                    assert v.dtype == actual[k].dtype and v.shape == actual[k].shape
                    expected = raw_hash(v); assert expected == raw_hash(actual[k]), k
                    tensor_rows.append(dict(name=k,shape=list(v.shape),dtype=str(v.dtype),sha256=expected))
                plain.eval().requires_grad_(False)
                versions = {n:p._version for n,p in plain.named_parameters()}
                source = Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-initial-performance-20261106-r1/eval01/case-00/sde5_fast/chunk-00-observation.npz')
                with np.load(source,allow_pickle=False) as obs:
                    images = obs['images'].copy(); state_np = obs['state'].copy()
                    raw = dict(main_images=torch.from_numpy(images[0:1]),
                        wrist_images=torch.from_numpy(images[None,1:]),
                        states=torch.from_numpy(state_np[None]),task_descriptions=[str(obs['instruction'].item())])
                random.seed(1000); np.random.seed(1000); torch.manual_seed(1000); torch.cuda.manual_seed_all(1000)
                with torch.inference_mode():
                    noise = torch.randn((1,50,32),device=self.device,dtype=torch.float32)
                    actions,detail = plain.predict_action_batch(raw,mode='train',compute_values=False,noise=noise)
                assert actions.shape == (1,50,14) and torch.isfinite(actions).all()
                assert torch.isfinite(detail['prev_logprobs']).all()
                assert versions == {n:p._version for n,p in plain.named_parameters()}
                np.savez(output/'calibration.npz',actions=actions.float().cpu().numpy(),
                    denoise_indices=detail['forward_inputs']['denoise_inds'].cpu().numpy(),
                    logprobs=detail['prev_logprobs'].float().cpu().numpy())
                write(output/'TENSORS.json',tensor_rows)
                result.update(status='qualified',path=str(target),bytes=target.stat().st_size,
                    sha256=sha(target),logical_tensor_count=len(tensor_rows),logical_parameter_bytes=parameter_bytes,
                    strict_plain_model_load=True,all_plain_model_tensor_bytes_dtypes_shapes_exact=True,
                    calibration_observation=str(source),calibration_observation_sha256=sha(source),
                    calibration_noise_seed=1000,calibration_sampler='K5 Flow-SDE0.3 batch1',
                    calibration_file_sha256=sha(output/'calibration.npz'),frozen_inference=True)
                write(output/'EXPORT.json',result)
                del actual,plain,disk; gc.collect();torch.cuda.empty_cache()
        finally:
            restore_rng(rng)
        after = actor_signature(self,snapshot_rng())
        assert self.optimizer_steps == 40 and before == after, 'Export changed restored model/optimizer/RNG'
        result['restored_actor_signature_unchanged'] = True
        write(output/f'export-rank-{self._rank:02d}.json',result)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=['ppo','grpo'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert sys.platform=='linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count()==4
    original=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1')/('ppo02' if args.algorithm=='ppo' else 'grpo01')
    checkpoint=original/f'sidney-paired-{args.algorithm}/checkpoints/global_step_2'
    assert json.loads((original/'final-restore/VERIFIED.json').read_text())['status']=='complete'
    cfg=OmegaConf.create(json.loads((original/'resolved-native-config.json').read_text()))
    cfg.native_pilot.output=str(args.output)
    cfg.native_pilot.case_manifest_path=str(PAIRED/'CASES.json')
    cfg.runner.logger.log_path=str(args.output)
    cfg.runner.per_worker_log_path=str(args.output/'worker_logs')
    cfg=validate_cfg(cfg);validate_pair(cfg)
    args.output.mkdir();write(args.output/'resolved-native-config.json',OmegaConf.to_container(cfg,resolve=True))
    cluster=Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement=HybridComponentPlacement(cfg,cluster)
    actor=ExportActor.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy('actor'))
    started=time.time()
    try:
        actor.init_worker().wait();actor.load_checkpoint(str(checkpoint/'actor')).wait()
        restored=actor.pilot_restore(str(checkpoint/'pilot')).wait()
        assert len(restored)==4 and all(r['restored'] for r in restored)
        ranks=actor.export_inference(str(args.output)).wait()
        assert len(ranks)==4 and all(r['restored_actor_signature_unchanged'] for r in ranks)
        rank0=next(r for r in ranks if r['rank']==0);assert rank0['status']=='qualified'
        write(args.output/'RESULT.json',dict(status='complete',algorithm=args.algorithm,
            source_checkpoint=str(checkpoint),optimizer_steps_per_rank=[40]*4,restored=restored,
            ranks=ranks,extra_optimizer_updates=0,new_simulator_controls=0,seconds=time.time()-started))
    finally:
        actor._close();ray.shutdown()


if __name__=='__main__':main()
