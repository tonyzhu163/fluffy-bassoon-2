"""Pin and strictly load the full SmolVLA checkpoint and its released processors."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import processor_smolvla  # register processor steps
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep

POLICY_REV = "a24f1f380461b665158d52acd140152c79e8b36d"
VLM_REV = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
WEIGHT_SHA = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"


class Motor:
    def __init__(self, cache):
        path, vlm = Path(cache) / "policy", Path(cache) / "vlm"
        with (path / "model.safetensors").open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHT_SHA
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = "cpu"
        cfg.vlm_model_name = str(vlm)
        # Initialize the architecture locally, then load the complete trained policy.
        cfg.load_vlm_weights = False
        self.policy = SmolVLAPolicy.from_pretrained(path, config=cfg, strict=True)
        self.policy.requires_grad_(False).eval()
        self.pre, self.post = make_pre_post_processors(cfg, str(path), preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": str(vlm)},
        })
        self.env = LiberoProcessorStep()
        assert not any(m.training for m in self.policy.modules())
        assert not any(p.requires_grad for p in self.policy.parameters())
        assert all(p.device.type == "cpu" for p in self.policy.parameters())

    @torch.inference_mode()
    def proposal(self, agent, wrist, obs, task, noise):
        self.policy.reset()
        batch = preprocess_observation({
            "pixels": {"image": np.ascontiguousarray(agent), "image2": np.ascontiguousarray(wrist)},
            "robot_state": {"eef": {"pos": obs["robot0_eef_pos"][None], "quat": obs["robot0_eef_quat"][None]},
                            "gripper": {"qpos": obs["robot0_gripper_qpos"][None]}},
        })
        batch = self.env.observation(batch)
        batch["task"] = task
        actions = self.policy.predict_action_chunk(self.pre(batch), noise=noise.clone())
        proposal = self.post(actions)[0].cpu().numpy()
        assert proposal.shape == (50, 7) and np.isfinite(proposal).all()
        return proposal
