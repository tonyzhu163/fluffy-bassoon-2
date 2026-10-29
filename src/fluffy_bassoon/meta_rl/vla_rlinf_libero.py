"""Frozen, strictly loaded RLinf pi0.5 LIBERO-130 SFT motor.

This is a distinct checkpoint/backend from released OpenPI pi05_libero. It
uses RLinf's native eval ODE, preprocessing, normalization and noise argument.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


MODEL_REPOSITORY = "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT"
MODEL_REVISION = "6222623f635769bfc73c9472e29fab9b7fd8e027"
MODEL_SHA256 = "4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683"
MODEL_BYTES = 7473091464
NORM_SHA256 = "dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1"
NORM_FILE = "physical-intelligence/libero/norm_stats.json"
RLINF_COMMIT = "e1a82ebf0c1af69da6ab9b1519f5f3692fe33566"
RLINF_TREE = "0bb0da449ff13778084839c22509fee2119e80cb"
RLINF_RECEIPT_SHA256 = "58d6d1667073b69d1c167604a3bc902ed2640d75a4ca3549f17b6ac79b3bab0e"
DEFAULT_SOURCE_RECEIPT = Path("/mowenjie/fluffy-bassoon/outputs/maniskill-pi05-baige-setup-20261028-r1/rlinf-source-verified.json")
DEFAULT_TOKENIZER_ROOT = Path("/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1")
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
FEATURE_KIND = "rlinf_pi05_libero130_final_prefix_visual_mean_language_mean_public_state_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(32 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(root):
    root = Path(root)
    receipt = json.loads((root / "READY.json").read_text())
    if (receipt.get("status") != "verified" or receipt.get("checkpoint") != MODEL_REPOSITORY
            or receipt.get("revision") != MODEL_REVISION):
        raise ValueError("Require the exact verified RLinf LIBERO-130 fullshot SFT checkpoint")
    entries = {row["name"]: row for row in receipt["files"]}
    for name, size, digest in (("model.safetensors", MODEL_BYTES, MODEL_SHA256),
                                (NORM_FILE, 4507, NORM_SHA256)):
        path = root / name
        if (name not in entries or entries[name]["bytes"] != size or entries[name]["sha256"] != digest
                or path.stat().st_size != size or sha256_file(path) != digest):
            raise ValueError(f"RLinf LIBERO-130 checkpoint data changed: {name}")
    return dict(receipt_sha256=sha256_file(root / "READY.json"), model_sha256=MODEL_SHA256,
                normalization_sha256=NORM_SHA256)


def validate_source(root, receipt_path=DEFAULT_SOURCE_RECEIPT):
    root, receipt_path = Path(root).resolve(), Path(receipt_path)
    if sha256_file(receipt_path) != RLINF_RECEIPT_SHA256:
        raise ValueError("RLinf source verification receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if receipt["source_commit"] != RLINF_COMMIT or receipt["git_tree"] != RLINF_TREE:
        raise ValueError("RLinf source commit/tree changed")
    selected = [entry for entry in receipt["files"] if entry["path"].startswith(
        ("rlinf/models/embodiment/openpi/", "rlinf/utils/ckpt_convertor/openpi/"))
        or entry["path"] in ("rlinf/config.py", "rlinf/envs/action_utils.py")]
    if len(selected) < 20:
        raise ValueError("Incomplete RLinf motor source receipt")
    for entry in selected:
        if sha256_file(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"RLinf motor source differs from its exact pin: {entry['path']}")
    return dict(commit=RLINF_COMMIT, tree=RLINF_TREE, verified_motor_files=len(selected),
                receipt_sha256=RLINF_RECEIPT_SHA256)


def model_configuration(checkpoint):
    """Native RLinf eval task; no learned exploration/value/DSRL head."""
    return dict(model_type="openpi", model_path=str(checkpoint), precision=None, pi05=True,
        is_lora=False, num_action_chunks=10, action_dim=7, use_proprio=True,
        num_steps=10, add_value_head=False,
        openpi_data=dict(norm_stats_path=str(Path(checkpoint) / NORM_FILE)),
        openpi=dict(task="eval", config_name="pi05_libero", num_images_in_input=2,
            action_horizon=10, action_chunk=10, num_steps=10, action_env_dim=7,
            model_action_dim=32, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
            max_token_len=200, discrete_state_input=False, torch_compile=False, use_rlt=False))


def environment_observation(obs, instruction):
    state = np.asarray(obs["state"], np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Require the eight public LIBERO robot-state coordinates")
    images = [np.asarray(obs[key]) for key in ("image", "wrist")]
    if any(image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 for image in images):
        raise ValueError("Require two public uint8 RGB camera images")
    return dict(main_images=images[0][None].copy(), wrist_images=images[1][None].copy(),
                states=state[None].copy(), task_descriptions=[str(instruction)])


class RLinfPrefixFeatures:
    def __init__(self, motor):
        self.motor, self.calls, self.seconds = motor, 0, 0.
        self.contract = dict(kind=FEATURE_KIND, model_config="rlinf_pi05_libero130",
            checkpoint_receipt_sha256=motor.metadata["receipt_sha256"],
            model_sha256=MODEL_SHA256, rlinf_commit=RLINF_COMMIT,
            native_config_name="pi05_libero", hidden_width=2048, feature_dim=4104,
            pooling="separate masked visual/language means, materialized float32, then raw public state8")

    def __call__(self, obs, instruction):
        from .vla_features import pool_prefix
        from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
        started = time.monotonic()
        with self.motor.torch.inference_mode():
            observation = self.motor.model.env_obs_to_observation(environment_observation(obs, instruction))
            prepared = preprocess_observation(observation, train=False)
            hidden, mask, _ = self.motor.model.build_prefix_cache(prepared)
            feature = pool_prefix(hidden.float().cpu().numpy(), mask.cpu().numpy(),
                prepared.tokenized_prompt.shape[-1], np.asarray(obs["state"], np.float32)[None])[0]
        self.calls += 1
        self.seconds += time.monotonic() - started
        if feature.shape != (4104,) or not np.isfinite(feature).all():
            raise ValueError("Unexpected RLinf native-prefix feature shape/value")
        return feature.astype(np.float32)

    def qualify(self, obs, instruction):
        first, second = self(obs, instruction), self(dict(obs), instruction)
        delta = float(np.max(np.abs(first - second)))
        if delta > 2e-4:
            raise RuntimeError("RLinf native-prefix features did not repeat within float32 tolerance")
        return dict(passed=True, maximum_delta=delta, contract=self.contract,
            native_path="preprocess_observation -> build_prefix_cache -> materialized masked float32 pooling",
            calls=self.calls, seconds=self.seconds)


class FrozenRLinfPi05Libero:
    """Current run_episode motor API, backed by the exact RLinf-130 release."""
    def __init__(self, args):
        if sys.platform != "linux":
            raise ValueError("Load robot motors only on allocated Linux")
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
        checkpoint, source = Path(args.checkpoint).resolve(), Path(args.rlinf_root).resolve()
        verified = validate_checkpoint(checkpoint)
        source_proof = validate_source(source, getattr(args, "rlinf_source_receipt", DEFAULT_SOURCE_RECEIPT))
        sys.path.insert(0, str(source))
        import openpi
        import torch
        from omegaconf import OmegaConf
        from rlinf.models.embodiment import openpi as factory
        from rlinf.models.embodiment.openpi import checkpoint as checkpoint_module
        actual_openpi = Path(openpi.__file__).resolve().parent
        if (actual_openpi != Path(args.openpi_root).resolve()
                or importlib.metadata.version("rlinf-openpi") != "0.1.1"
                or not Path(factory.__file__).resolve().is_relative_to(source)):
            raise ValueError("Require the pinned RLinf source and its installed rlinf-openpi 0.1.1 package")
        tokenizer_root = Path(os.environ.get("OPENPI_DATA_HOME", DEFAULT_TOKENIZER_ROOT))
        if sha256_file(tokenizer_root / "big_vision/paligemma_tokenizer.model") != TOKENIZER_SHA256:
            raise ValueError("Require the verified released RLinf PaliGemma tokenizer")
        os.environ["OPENPI_DATA_HOME"] = str(tokenizer_root)
        if torch.cuda.device_count() != 1:
            raise ValueError("Require exactly one allocation-visible Torch CUDA GPU")
        loading = {}
        def strict_loader(model, path):
            # Native alias/conversion equations, with complete strict coverage.
            # This replacement is process-local during construction; no source
            # file, checkpoint or shared environment is edited.
            values = checkpoint_module._convert_official_openpi_keys(
                checkpoint_module._load_safetensors_state_dict(path), path)
            incompatible = model.load_state_dict(values, strict=True)
            loading.update(strict=True, loaded_tensors=len(values),
                missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys))
        original_loader = factory.load_base_safetensors
        factory.load_base_safetensors = strict_loader
        try:
            self.model = factory.get_model(OmegaConf.create(model_configuration(checkpoint))).cuda().eval()
        finally:
            factory.load_base_safetensors = original_loader
        self.model.requires_grad_(False)
        self.torch, self.horizon, self.calls, self.output = torch, int(self.model.action_horizon), 0, Path(args.output)
        if self.horizon != 10 or int(self.model.action_dim) != 32 or not loading.get("strict"):
            raise ValueError("RLinf motor dimensions or strict loading changed")
        self.metadata = dict(backend="rlinf_pi05_libero130_torch", checkpoint=MODEL_REPOSITORY,
            revision=MODEL_REVISION, **verified, source_proof=source_proof, strict_loading=loading,
            openpi_package="rlinf-openpi 0.1.1", openpi_package_root=str(actual_openpi),
            flow_steps=10, predicted_horizon=10, action_dim=32, environment_action_dim=7,
            weights_frozen=True, trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            native_sampler="RLinf Pi0Eval deterministic Euler ODE; explicit noise before denoising",
            normalization_path=str(checkpoint / NORM_FILE), rlinf_commit=RLINF_COMMIT)
        self.feature_encoder = RLinfPrefixFeatures(self) if args.feature_interface == "vla-prefix" else None

    def infer(self, obs, instruction, noise):
        noise = np.asarray(noise, np.float32)
        if noise.shape != (self.horizon, 32) or not np.isfinite(noise).all():
            raise ValueError("Require the exact finite native initial-noise tensor [10,32]")
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(noise.copy())[None].to("cuda", dtype=self.torch.float32)
            actions, _ = self.model.predict_action_batch(environment_observation(obs, instruction),
                mode="eval", noise=tensor)
            result = actions.detach().float().cpu().numpy()[0].copy()
        self.calls += 1
        if self.calls == 1:
            from .vla_libero import attest_process_gpu
            attest_process_gpu(self.output / "motor-binding.json", graphics=False)
        if result.shape != (10, 7) or not np.isfinite(result).all():
            raise ValueError("Unexpected RLinf LIBERO decoded action shape/value")
        return result
