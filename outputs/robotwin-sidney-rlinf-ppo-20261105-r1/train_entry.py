"""Sidney model boundary + corrected scenes around the unchanged RLinf learner."""
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType

import torch
from omegaconf import OmegaConf

from fluffy_bassoon.rl.sidney import CONFIG_NAME,REVISION,install_rlinf_transforms,deterministic_reference
from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng,restore_rng,write
from recipe import validate_sidney
from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime


def worker_setup(worker,role):
    from fluffy_bassoon.baige_gpu import mapping,cuda_pci,pci
    install_rlinf_transforms();deterministic_reference()
    count=torch.cuda.device_count()
    assert count in (1,int(worker.cfg.native_pilot.world_size))
    torch.cuda.set_device(0 if count==1 else worker._rank)
    actual=cuda_pci(torch.cuda.current_device())
    rows=[r for r in mapping() if pci(r['pci_bus_id'])==actual]
    assert len(rows)==1
    os.environ['LYCO_BAIGE_ISOLATED_UUID']=rows[0]['uuid']
    write(Path(worker.cfg.native_pilot.output)/'worker-gpus'/f'{role}-{worker._rank:02d}.json',
        dict(role=role,rank=worker._rank,cuda_pci=actual,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
             scheduler_uuid=rows[0]['uuid'],cuda_device_count=count))


class SidneyActor(runtime.PilotActor):
    def init_worker(self):
        worker_setup(self,'actor');return super().init_worker()

    def train_micro_batch(self,micro_batch,metrics,*,is_last):
        if not getattr(self,'_sidney_first_replay_checked',False):
            from rlinf.utils.nested_dict_process import put_tensor_device
            batch=put_tensor_device(micro_batch,self.device);rng=snapshot_rng()
            try:
                with torch.no_grad(),self.amp_context:
                    replay=self.model(forward_inputs=batch['forward_inputs'],compute_logprobs=True,
                        compute_entropy=False,compute_values=True,use_cache=False)
                a,b=batch['prev_logprobs'].float(),replay['logprobs'].float()
                assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                old_values=batch['prev_values'].float().reshape(-1)
                new_values=replay['values'].float().reshape(-1)
                assert old_values.shape==new_values.shape
                assert torch.isfinite(old_values).all() and torch.isfinite(new_values).all()
                errors=dict(logprobs=float((b-a).abs().max()),
                    native_mean_vs_selected_value_max_abs=float((new_values-old_values).abs().max()))
                # Native RLinf rollout values average all K suffix calls;
                # actor values use the selected denoising step. They are
                # intentionally different estimands. The bridge verifies each.
                passed=errors['logprobs']<=1e-3
                write(Path(self.cfg.native_pilot.output)/'first-replay'/f'rank-{self._rank:02d}.json',
                    dict(passed=passed,max_abs_error=errors,optimizer_steps=self.optimizer_steps))
                assert passed,'Rollout/actor replay differs before first SGD'
            finally:restore_rng(rng)
            self._sidney_first_replay_checked=True
        return super().train_micro_batch(micro_batch,metrics,is_last=is_last)


class SidneyRollout(runtime.PilotRollout):
    def init_worker(self):
        worker_setup(self,'rollout');return super().init_worker()


class SidneyEnvironment(runtime.PilotEnvironment):
    def init_worker(self):
        from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer,_install_reset_retry_limit
        worker_setup(self,'env')
        root=Path(self.cfg.native_pilot.output)
        _install_allocated_renderer(root/'renderers'/f'rank-{self._rank:02d}')
        _install_reset_retry_limit()
        result=super().init_worker();self._binding_ordinal=0
        for wrapped in self.env_list:
            env=runtime.base_env(wrapped);original=env.reset;owner=self
            def reset(physical,env_idx=None,env_seeds=None,_original=original):
                result=_original(env_idx=env_idx,env_seeds=env_seeds)
                indices=range(physical.num_envs) if env_idx is None else env_idx
                rows=[]
                for index in indices:
                    sub=physical.venv.envs[index];info=sub.task.get_info()
                    assert sub.episode_info_list==[info]
                    assert sub.task.get_instruction()==result[0]['task_descriptions'][index]
                    rows.append(dict(env_index=int(index),requested_seed=int(sub.env_seed),
                        actual_seed=int(sub.task.ep_num),actual_scene_info=info,
                        instruction=sub.task.get_instruction()))
                write(root/'scene-bindings'/f'rank-{owner._rank:02d}-{owner._binding_ordinal:05d}.json',rows)
                if owner._binding_ordinal==0:
                    from PIL import Image
                    raw=result[0];quality={}
                    for key in ('main_images','wrist_images'):
                        image=torch.as_tensor(raw[key]).cpu()
                        assert image.dtype==torch.uint8 and image.shape[-1]==3
                        flat=image.reshape(-1,*image.shape[-3:])[:,::16,::16].float().flatten(1)
                        spread=flat.std(dim=1)
                        assert bool((spread>1).all()),f'Blank camera observations: {key}'
                        quality[key]=dict(shape=list(image.shape),minimum_image_std=float(spread.min()))
                    assert bool(torch.isfinite(torch.as_tensor(raw['states'])).all())
                    views=[torch.as_tensor(raw['main_images'][0]),*torch.as_tensor(raw['wrist_images'][0])]
                    montage=Image.new('RGB',(sum(v.shape[1] for v in views),max(v.shape[0] for v in views)))
                    offset=0
                    for view in views:
                        montage.paste(Image.fromarray(view.cpu().numpy()),(offset,0));offset+=view.shape[1]
                    target=root/'camera-checks';target.mkdir(exist_ok=True)
                    montage.save(target/f'rank-{owner._rank:02d}.jpg',quality=85)
                    write(target/f'rank-{owner._rank:02d}.json',dict(passed=True,quality=quality,
                        scene_instruction_bindings=len(rows),weights_or_rng_changed=False))
                owner._binding_ordinal+=1
                return result
            env.reset=MethodType(reset,env)
        return result


class SidneyRunner(runtime.PilotRunner):
    def run(self):
        from rlinf.runners.embodied_runner import EmbodiedRunner
        from fluffy_bassoon.meta_rl.rlinf_native_ppo_data import collection_contract
        recipe=validate_sidney(self.cfg)
        self._pilot_write('resolved-native-config.json',OmegaConf.to_container(self.cfg,resolve=True))
        self._pilot_write('resolved-pilot-contract.json',recipe)
        self._pilot_write('collection-contract.json',collection_contract())
        resume=self.cfg.runner.get('resume_dir')
        if resume:
            rows=[]
            for group in (self.actor,self.rollout,self.env):
                rows.extend(group.pilot_restore(str(Path(resume)/'pilot')).wait())
            self._pilot_write('operational-restore.json',dict(global_step=self.global_step,workers=rows))
        else:
            self.update_rollout_weights()
            self._capture_boundary()
        self.env.pilot_phase('training').wait()
        # Keep the native loop plus PilotRunner's iteration-boundary capture;
        # omit its historical128-case SFT/before/after panels.
        EmbodiedRunner.run(self)
        assert self.global_step==2,'Two complete PPO rollout iterations required'
        self._pilot_write('RESULT.json',dict(status='complete',rollout_iterations=2,
            baseline_reference='published Sidney SFT',new_sft_evaluation_episodes=0,
            source_task='place_phone_stand',paper_numeric_reproduction=False))


def main():
    runtime.validate_recipe=validate_sidney
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    actor_module.EmbodiedFSDPActor=SidneyActor
    path=Path(os.environ['EMBODIED_PATH'])/'train_embodied_agent.py'
    spec=importlib.util.spec_from_file_location('sidney_native_entry',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    entry.EmbodiedRunner=SidneyRunner;entry.EnvWorker=SidneyEnvironment
    entry.MultiStepRolloutWorker=SidneyRollout
    entry.main()


if __name__=='__main__':main()
