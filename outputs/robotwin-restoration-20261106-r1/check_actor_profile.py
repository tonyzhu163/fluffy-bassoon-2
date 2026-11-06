"""Compose/check this run on allocated Linux CPU; never instantiate a model."""
import json
import os
from pathlib import Path
import shutil

from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf


def main():
    code,root = Path(os.environ['TASK_CODE']),Path(os.environ['TASK_OUT'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    with initialize_config_dir(config_dir=str(code/'configs/native_ppo'),version_base='1.1'):
        # Native validate_cfg also inspects the real accelerator rank inventory.
        # It runs unchanged in profile_motus_actor.main on the four-GPU job.
        cfg = compose(config_name='motus_phone_two',overrides=['actor.micro_batch_size=64'])
    expected = {
        'actor.global_batch_size':2048,'actor.micro_batch_size':64,
        'actor.model.num_action_chunks':32,'actor.model.openpi.action_horizon':32,
        'actor.model.openpi.action_chunk':32,'actor.model.openpi.config_name':'pi05_aloha_robotwin',
        'actor.model.openpi.noise_method':'flow_sde','actor.model.openpi.noise_level':.3,
        'actor.model.openpi.num_steps':5,'actor.model.openpi.train_expert_only':True,
        'actor.model.openpi_data.adapt_to_pi':True,'actor.model.openpi_data.extra_delta_transform':True,
        'actor.model.openpi.detach_critic_input':True,'algorithm.update_epoch':5,
        'algorithm.gamma':.99,'algorithm.gae_lambda':.95,'algorithm.clip_ratio_high':.2,
        'algorithm.value_clip':.2,'algorithm.huber_delta':10,'actor.optim.lr':5e-6,
        'actor.optim.value_lr':1e-4,'actor.optim.adam_beta1':.9,'actor.optim.adam_beta2':.95,
        'actor.optim.adam_eps':1e-8,'actor.optim.weight_decay':.01,'actor.optim.clip_grad':1.,
        'env.train.max_episode_steps':192,'env.eval.max_episode_steps':192,'native_pilot.bounded_io':True,
    }
    for key,value in expected.items():
        assert OmegaConf.select(cfg,key) == value,(key,OmegaConf.select(cfg,key),value)
    assert Path(cfg.actor.model.openpi_data.norm_stats_path).is_file()
    from profile_motus_actor import take
    import numpy as np
    import torch
    values = dict(tensor=torch.arange(12).reshape(4,3),array=np.arange(8).reshape(4,2),
                  prompt=['a','b','c','d'])
    selected = take(values,[3,0,2,3])
    assert torch.equal(selected['tensor'],values['tensor'][[3,0,2,3]])
    assert np.array_equal(selected['array'],values['array'][[3,0,2,3]])
    assert selected['prompt'] == ['d','a','c','d']
    record = dict(status='passed',source_commit=os.environ['TASK_SOURCE'],resolved_fields=expected,
        model_computation=False,slicing_tensor_numpy_prompt_exact=True,
        native_device_validation_deferred_to_actual_four_gpu_job=True,
        bos_mount_on_development={str(p):os.path.ismount(p) for p in (Path('/mnt/bos'),Path('/mnt/bos/adgfyl'))},
        hot_free_gib=shutil.disk_usage(root).free/2**30)
    (root/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__': main()
