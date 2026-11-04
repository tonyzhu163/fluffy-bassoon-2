"""Load the six matched policies into their native eval-only architecture."""
from pathlib import Path

import torch
from omegaconf import OmegaConf
from native_eval import SelectionRollout
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
from rlinf.models import get_model
from rlinf.models.embodiment.openpi.checkpoint import _normalize_state_dict, _convert_official_openpi_keys
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict


class MatchedRollout(SelectionRollout):
    def init_worker(self):
        super().init_worker()
        assert self.only_eval and self.model_cfg.openpi.value_after_vlm
        self.architectures = {(True, True): self.hf_model}
        self.checked_states = {}
        self.loaded = {}

    def select_policy(self, row, phase, rng_seed):
        architecture = (row['prefix'], row['ignore_last'])
        if architecture not in self.architectures:
            cfg = OmegaConf.merge(self.model_cfg, self.cfg.rollout.model)
            cfg.openpi.value_after_vlm, cfg.openpi.ignore_last = architecture
            self.architectures[architecture] = get_model(cfg)
        self.hf_model = self.architectures[architecture]
        digest, path = row['sha256'], Path(row['path'])
        if digest not in self.checked_states:
            assert sha(path) == digest
            raw = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            self.checked_states[digest] = _convert_official_openpi_keys(_normalize_state_dict(as_state_dict(raw)), path)
        if self.loaded.get(architecture) != digest:
            self.hf_model.load_state_dict(self.checked_states[digest], strict=True)
            self.hf_model.eval().requires_grad_(False)
            self.hf_model.set_global_step(row['step'])
            self.loaded[architecture] = digest
        self.select_sampling_scope(phase, rng_seed)
        assert (self.hf_model.rl_cfg.value_after_vlm, self.hf_model.rl_cfg.ignore_last) == architecture
        return dict(strict_load=True, sha256=digest, step=row['step'], prefix=row['prefix'],
                    ignore_last=row['ignore_last'], phase=phase, rng_seed=rng_seed)
