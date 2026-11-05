"""Six-task-arm pilot migration of the existing qualified RLinf worker wrappers.

Only task and common-motor metadata differ from the historical two-update
pilot. The PPO learner, native losses, sampler and recovery stay unchanged.
"""
import importlib.util
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng, restore_rng, validate_recipe, write
from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime


def validate_source(cfg):
    # Validate the actual migration first, then apply the old invariant checker
    # to a metadata copy with its historical task name. The executed cfg is
    # never rewritten to the held-out adjust_bottle task.
    for mode in ('train', 'eval'):
        assert cfg.env[mode].task_config.task_name == 'place_phone_stand'
    expected = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1')
    assert Path(cfg.actor.model.model_path) == expected
    assert Path(cfg.actor.model.openpi_data.norm_stats_path) == expected / 'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert cfg.actor.model.openpi_data.adapt_to_pi is True
    assert cfg.actor.model.openpi_data.extra_delta_transform is True
    assert cfg.native_pilot.initializer_revision == 'effc6e1a99bd4d0edf9447e8ba313def67649cfe'
    copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for mode in ('train', 'eval'):
        copy.env[mode].task_config.task_name = 'adjust_bottle'
    result = validate_recipe(copy)
    return dict(result, source_task='place_phone_stand', initializer='common Motus pi0.5',
                initialization_reconstruction=True, original_initializer_reproduction=False)


class SourceActor(runtime.PilotActor):
    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        if not getattr(self, '_source_first_replay_checked', False):
            from rlinf.utils.nested_dict_process import put_tensor_device
            batch = put_tensor_device(micro_batch, self.device)
            rng = snapshot_rng()
            try:
                with torch.no_grad(), self.amp_context:
                    replay = self.model(forward_inputs=batch['forward_inputs'],
                        compute_logprobs=True, compute_entropy=False,
                        compute_values=True, use_cache=False)
                errors = {}
                for output_key, old_key in (('logprobs', 'prev_logprobs'), ('values', 'prev_values')):
                    old, new = batch[old_key].float(), replay[output_key].float()
                    assert old.shape == new.shape, (old_key, old.shape, new.shape)
                    assert torch.isfinite(old).all() and torch.isfinite(new).all()
                    errors[output_key] = float((new-old).abs().max().item())
                passed = errors['logprobs'] <= 1e-3 and errors['values'] <= 1e-4
                write(Path(self.cfg.native_pilot.output) / 'first-replay' / f'rank-{self._rank:02d}.json',
                      dict(passed=passed, max_abs_error=errors, optimizer_steps=self.optimizer_steps,
                           logprob_tolerance=1e-3, value_tolerance=1e-4))
                assert passed, 'Actual rollout/actor replay differs before the first optimizer step'
            finally:
                restore_rng(rng)
            self._source_first_replay_checked = True
        return super().train_micro_batch(micro_batch, metrics, is_last=is_last)


def main():
    runtime.validate_recipe = validate_source
    import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
    actor_module.EmbodiedFSDPActor = SourceActor
    path = Path(os.environ['EMBODIED_PATH']) / 'train_embodied_agent.py'
    spec = importlib.util.spec_from_file_location('robotwin_source_official_entry', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.EmbodiedRunner = runtime.PilotRunner
    entry.EnvWorker = runtime.PilotEnvironment
    entry.MultiStepRolloutWorker = runtime.PilotRollout
    entry.main()


if __name__ == '__main__':
    main()
