"""Four-GPU native FSDP/PPO throughput on scratch copies of saved source inputs.

Synthetic advantages/returns make this a systems benchmark, not a scientific
PPO run. Scratch optimizer updates are never published or used as initialization.
"""
import gzip
import json
import os
from pathlib import Path
import time

import hydra
import ray
import torch
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.utils.nested_dict_process import put_tensor_device
from train_entry import SidneyActor
from fluffy_bassoon.rl.sidney import CONFIG_NAME
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write


class BenchmarkActor(SidneyActor):
    def benchmark(self):
        root=Path(self.cfg.native_pilot.output)
        micro=int(self.cfg.actor.micro_batch_size);local=2048//self._world_size
        assert self._world_size==4 and local%micro==0
        report=dict(rank=self._rank,micro=micro,local_global_batch=local,
            accumulation=local//micro,scientific_training=False,updates=[])
        path=root/f'rank-{self._rank:02d}.json';write(path,report)
        try:
            if self.is_weight_offloaded:self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:self.load_optimizer(self.device)
            self.model.train()
            source=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-rlinf-ppo-20261105-r1/train03')
            initial=torch.load(source/'initial-states'/f'source-rank-{self._rank:02d}-0000.pt',map_location='cpu',weights_only=False)
            with gzip.open(source/'sampling'/f'rank-{self._rank:02d}/00000.pt.gz','rb') as f:
                sampled=torch.load(f,map_location='cpu',weights_only=False)
            from rlinf.models.embodiment.openpi.env_io import repack_env_obs
            raw=initial['initial_observation']
            # Keep real raw camera/state replay and CPU/GPU preprocessing cost.
            # Access the wrapped module only for its input transform helper.
            model=self.model
            while not hasattr(model,'env_obs_to_observation'):model=model.module
            prepared=model.env_obs_to_observation(raw)
            repacked=repack_env_obs(CONFIG_NAME,raw,state_indices=None)
            base={k:torch.as_tensor(v).cpu() for k,v in repacked.items() if k!='prompt'}
            base.update(chains=sampled['forward_sampling_fields']['chains'],
                denoise_inds=sampled['forward_sampling_fields']['denoise_inds'],
                tokenized_prompt=prepared.tokenized_prompt.cpu(),
                tokenized_prompt_mask=prepared.tokenized_prompt_mask.cpu())
            index=torch.arange(local)%64
            inputs={k:v.index_select(0,index).contiguous() for k,v in base.items()}
            del base,prepared,repacked,raw,initial,sampled
            parts=[]
            with torch.no_grad(),self.amp_context:
                for start in range(0,local,micro):
                    part=put_tensor_device({k:v[start:start+micro] for k,v in inputs.items()},self.device)
                    out=self.model(forward_inputs=part,compute_logprobs=True,compute_values=True,
                        compute_entropy=False,use_cache=False)
                    parts.append((out['logprobs'].float().cpu(),out['values'].float().cpu().reshape(-1,1)))
                    del out,part
            old_logprobs=torch.cat([p[0] for p in parts]);old_values=torch.cat([p[1] for p in parts])
            advantages=torch.linspace(-1.,1.,local)[:,None]
            batch=dict(forward_inputs=inputs,prev_logprobs=old_logprobs,prev_values=old_values,
                advantages=advantages,returns=old_values+advantages,
                loss_mask=torch.ones((local,1),dtype=torch.bool))
            # First step warms kernels and allocates real Adam state. Three
            # subsequent global2048 updates measure native PPO/FSDP throughput.
            for update in range(4):
                torch.distributed.barrier();torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                self.optimizer.zero_grad();metrics={}
                for start in range(0,local,micro):
                    part={k:({sub:v[start:start+micro] for sub,v in value.items()}
                        if isinstance(value,dict) else value[start:start+micro]) for k,value in batch.items()}
                    EmbodiedFSDPActor.train_micro_batch(self,part,metrics,is_last=start+micro==local)
                    del part
                grad_norm,_=self.optimizer_step();self.optimizer.zero_grad()
                torch.cuda.synchronize();elapsed=time.monotonic()-started
                maximum=torch.tensor(elapsed,device=self.device)
                torch.distributed.all_reduce(maximum,op=torch.distributed.ReduceOp.MAX)
                row=dict(update=update,warmup=update==0,seconds=float(maximum),
                    global_examples_per_second=2048/float(maximum),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    total_device_bytes=torch.cuda.get_device_properties(self.device).total_memory,
                    grad_norm=float(grad_norm))
                assert torch.isfinite(torch.tensor(row['grad_norm']))
                report['updates'].append(row);write(path,report)
            report['status']='complete';write(path,report);return report
        except BaseException as error:
            report.update(status='failed',error_type=type(error).__name__,error=str(error),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write(path,report);raise


@hydra.main(version_base='1.1',config_path=str(Path(os.environ['TASK_CODE'])/'configs/native_ppo'),config_name='sidney_phone_two')
def main(cfg):
    cfg=validate_cfg(cfg)
    cluster=Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement=HybridComponentPlacement(cfg,cluster)
    actor=BenchmarkActor.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy('actor'))
    try:
        actor.init_worker().wait()
        results=actor.benchmark().wait()
        write(Path(cfg.native_pilot.output)/'RESULT.json',dict(status='complete',ranks=results,
            scientific_training=False,source_commit=os.environ['TASK_SOURCE']))
    finally:
        actor._close();ray.shutdown()


if __name__=='__main__':main()
