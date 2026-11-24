"""Checkpoint-native preprocessing, flow sampling and action unnormalization."""

import hashlib

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import processor_smolvla  # register checkpoint processor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep

WEIGHTS_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"


class Motor:
    def __init__(self, root):
        path = root / "policy"
        with (path / "model.safetensors").open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHTS_SHA256
        cfg = PreTrainedConfig.from_pretrained(str(path))
        cfg.device = "cpu"
        cfg.vlm_model_name = str(root / "vlm")
        # Initialize from config, then load ALL checkpoint tensors with strict=True.
        cfg.load_vlm_weights = False
        self.policy = SmolVLAPolicy.from_pretrained(path, config=cfg, strict=True)
        self.policy.float().eval().requires_grad_(False)
        assert all(not m.training for m in self.policy.modules())
        assert sum(p.numel() for p in self.policy.parameters() if p.requires_grad) == 0
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": "cpu"},
                                    "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name}},
        )
        self.env_pre = LiberoProcessorStep()

    @torch.inference_mode()
    def propose(self, images, record, noise):
        def batch(x):
            return torch.as_tensor(np.array(x), dtype=torch.float32).unsqueeze(0)

        obs = {
            "observation.images." + key: batch(img).permute(0, 3, 1, 2) / 255
            for key, img in zip(("image", "image2"), images, strict=True)
        }
        obs["observation.robot_state"] = {
            "eef": {"pos": batch(record["pos"]), "quat": batch(record["quat"])},
            "gripper": {"qpos": batch(record["grip"])},
        }
        obs = self.env_pre.observation(obs)
        obs["task"] = str(record["task"])
        self.policy.reset()
        chunk = self.policy.predict_action_chunk(self.pre(obs), noise=noise.clone())
        out = self.post(chunk).float().numpy()
        assert out.shape == (1, 50, 7) and np.isfinite(out).all()
        return out[0]
