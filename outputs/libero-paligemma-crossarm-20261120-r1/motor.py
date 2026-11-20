"""Source-only PaliGemma action-chunk SFT for the LIBERO-Pro2 cross-arm run."""

import argparse
import fcntl
import json
import math
import os
import platform
import random
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration

from motor_data import (
    BODIES,
    HORIZON,
    instruction,
    read_episodes,
    select_batch,
    split_poses,
    valid_anchors,
)
from motor_weights import sha256, verified_snapshot

GLOBAL_BATCH = 8
RANK = 8
HEAD_LR = 3e-4
LORA_LR = 1e-4
CHECKPOINT_EVERY = 25
VALIDATE_EVERY = 50
SCHEMA = 1


class LoRALinear(nn.Module):
    """Keep the pretrained projection, adding a zero-initialized rank-eight update."""

    def __init__(self, linear):
        super().__init__()
        self.base = linear
        self.down = nn.Linear(linear.in_features, RANK, bias=False)
        self.up = nn.Linear(RANK, linear.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + 2.0 * self.up(self.down(x))


class ChunkHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.patch = nn.Linear(width, 48)
        self.spatial = nn.Conv2d(48, 24, kernel_size=3, stride=2, padding=1)
        self.language = nn.Linear(width, 96)
        self.proprio = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64))
        self.readout = nn.Sequential(
            nn.Linear(2 * 8 * 8 * 24 + 96 + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, HORIZON * 7),
        )

    def forward(self, context, mask, state):
        b = len(state)
        context = self.norm(context.float())
        patches = self.patch(context[:, :512]).reshape(b * 2, 16, 16, 48)
        grid = F.silu(self.spatial(F.silu(patches).permute(0, 3, 1, 2))).reshape(b, -1)
        last = mask.sum(1) - 1
        text = self.language(context[torch.arange(b, device=context.device), last])
        features = torch.cat((grid, F.silu(text), self.proprio(state)), dim=-1)
        return self.readout(features).reshape(b, HORIZON, 7)


def require_gpu():
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This motor requires one visible allocated BF16 CUDA GPU")


def camera_batch(observations):
    groups = defaultdict(list)
    for obs in observations:
        for camera in ("rgb", "wrist"):
            frame = np.asarray(obs[camera])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{camera} must be uint8 H x W x 3")
            groups[frame.shape].append(frame)
    resized = {}
    for shape, frames in groups.items():
        pixels = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float().div_(255)
        resized[shape] = iter(
            F.interpolate(pixels, (224, 224), mode="bicubic", align_corners=False, antialias=True)
            .clamp_(0, 1)
            .mul_(2)
            .sub_(1)
        )
    # Placeholder ordering is batch-major, then external/wrist, then raster patch.
    return torch.stack([next(resized[np.shape(obs[k])]) for obs in observations for k in ("rgb", "wrist")])


class Motor(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        require_gpu()
        with verified_snapshot(weights_path) as (snapshot, self.identity):
            self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
            full, info = PaliGemmaForConditionalGeneration.from_pretrained(
                snapshot,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                attn_implementation="sdpa",
                local_files_only=True,
                output_loading_info=True,
            )
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                if info.get(key):
                    raise ValueError(f"Incomplete original PaliGemma load ({key}): {info[key]}")
            self.vlm = full.model
        self.vlm.requires_grad_(False)
        cfg = self.vlm.config
        layers = self.vlm.language_model.layers
        if (
            len(layers),
            cfg.text_config.hidden_size,
            cfg.vision_config.image_size,
            cfg.vision_config.patch_size,
        ) != (18, 2048, 224, 14):
            raise ValueError("Unexpected PaliGemma vision/language architecture")
        for layer in layers[-2:]:
            for name in ("q_proj", "v_proj"):
                attn = layer.self_attn
                setattr(attn, name, LoRALinear(getattr(attn, name)))
        self.head = ChunkHead(cfg.text_config.hidden_size)
        self.register_buffer("state_mean", torch.zeros(8))
        self.register_buffer("state_scale", torch.ones(8))
        self.prompt_ids = {}
        self.cuda().eval()

    def prefix(self, observations):
        tokens = []
        for obs in observations:
            prompt = instruction(obs["prompt"])
            if prompt not in self.prompt_ids:
                pieces = self.tokenizer.encode(prompt.rstrip("\n") + "\n", add_special_tokens=False)
                if len(pieces) + 1 > 128 or self.vlm.config.image_token_id in pieces:
                    raise ValueError(
                        "Instruction exceeds the original prefix or contains an image placeholder"
                    )
                self.prompt_ids[prompt] = [self.tokenizer.bos_token_id, *pieces]
            tokens.append(self.prompt_ids[prompt])
        size = 512 + max(map(len, tokens))
        rows = [
            [self.vlm.config.image_token_id] * 512 + t + [self.tokenizer.pad_token_id] * (size - 512 - len(t))
            for t in tokens
        ]
        ids = torch.tensor(rows, device="cuda", dtype=torch.long)
        lengths = torch.tensor([512 + len(t) for t in tokens], device="cuda")
        mask = torch.arange(size, device="cuda")[None, :] < lengths[:, None]
        pixels = camera_batch(observations)
        # Autograd begins at the first LoRA projection. Neither camera features nor
        # earlier language activations require a backward graph.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = self.vlm(
                input_ids=ids,
                pixel_values=pixels,
                attention_mask=mask,
                token_type_ids=torch.zeros_like(ids),
                position_ids=mask.long().cumsum(-1),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        return context, mask

    def forward(self, observations):
        states = np.stack([obs["state"] for obs in observations]).astype(np.float32, copy=False)
        if states.shape != (len(observations), 8) or not np.isfinite(states).all():
            raise ValueError("Expected finite xyz/quaternion/jaw state")
        state = (torch.from_numpy(states).cuda() - self.state_mean) / self.state_scale
        context, mask = self.prefix(observations)
        return self.head(context, mask, state.clamp(-10, 10))

    def delta(self):
        tensors = {k: v for k, v in self.named_parameters() if v.requires_grad}
        tensors.update(state_mean=self.state_mean, state_scale=self.state_scale)
        return {k: v.detach().cpu().clone() for k, v in tensors.items()}

    @torch.no_grad()
    def restore_delta(self, saved):
        expected = {k: v for k, v in self.named_parameters() if v.requires_grad}
        expected.update(state_mean=self.state_mean, state_scale=self.state_scale)
        if saved.keys() != expected.keys():
            raise ValueError("Checkpoint action-head, adapter or normalization keys differ")
        for name, dest in expected.items():
            if dest.shape != saved[name].shape or dest.dtype != saved[name].dtype:
                raise ValueError(f"Invalid delta shape/dtype for {name}")
            dest.copy_(saved[name])

    @classmethod
    def load(cls, output_dir, weights_path):
        saved = torch.load(Path(output_dir) / "checkpoint.pt", weights_only=True, map_location="cpu")
        if saved["schema"] != SCHEMA or saved["best"] is None:
            raise ValueError("No validated motor in this checkpoint")
        motor = cls(weights_path)
        identity = saved["contract"]["weights"]
        if (motor.identity["repo"], motor.identity["revision"]) != (identity["repo"], identity["revision"]):
            raise ValueError("Base VLM differs from the trained motor")
        motor.restore_delta(saved["best"]["model"])
        return motor.requires_grad_(False).eval()

    @torch.inference_mode()
    def act(self, obs):
        output = self([obs])[0].float()
        if not torch.isfinite(output).all():
            raise FloatingPointError("Motor returned a nonfinite action")
        return output.clamp(-1, 1).cpu().numpy()


def collate(batch):
    observations, actions, masks = zip(*batch)
    return observations, torch.from_numpy(np.stack(actions)).cuda(), torch.from_numpy(np.stack(masks)).cuda()


@torch.inference_mode()
def validate(motor, anchors, micro, source_bodies=BODIES):
    by_body = defaultdict(list)
    for start in range(0, len(anchors), micro):
        group = anchors[start : start + micro]
        obs, target, mask = collate([ep.sample(t) for ep, t in group])
        err = (motor(obs).clamp(-1, 1) - target).abs()
        per_axis = (err * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        for (ep, _), row in zip(group, per_axis.cpu().numpy()):
            by_body[ep.body].append(row)
    axes = {b: np.mean(by_body[b], axis=0).tolist() for b in source_bodies}
    score = float(np.mean(list(axes.values())))
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite held-out action error")
    return {
        "mae": score,
        "axis_mae_by_body": axes,
        "anchors_by_body": {b: len(by_body[b]) for b in source_bodies},
    }


def rng_state(sampler):
    algo, keys, pos, gauss, cached = np.random.get_state()
    return {
        "sampler": sampler.bit_generator.state,
        "python": random.getstate(),
        "numpy": (algo, keys.tolist(), pos, gauss, cached),
        "torch_cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state, sampler):
    sampler.bit_generator.state = state["sampler"]
    random.setstate(state["python"])
    algo, keys, pos, gauss, cached = state["numpy"]
    np.random.set_state((algo, np.array(keys, dtype=np.uint32), pos, gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["cuda"])


def atomic_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def recipe():
    return {
        "horizon": HORIZON,
        "global_batch": GLOBAL_BATCH,
        "lora_rank": RANK,
        "lora_alpha": 2 * RANK,
        "lora_layers": [16, 17],
        "lora_projections": ["q_proj", "v_proj"],
        "head_lr": HEAD_LR,
        "lora_lr": LORA_LR,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "huber_beta": 0.1,
        "validate_every": VALIDATE_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
    }


def train(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_gpu()
    if args.steps < 1 or not 0 <= args.seed < 2**32 or GLOBAL_BATCH % args.microbatch:
        raise ValueError("Require positive total updates, uint32 seed and a microbatch dividing global batch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    sampler = np.random.default_rng(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((output / "motor.lock").open("a"))
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = output / "checkpoint.pt"
        if checkpoint.exists() != args.resume:
            raise ValueError("Use --resume for an existing checkpoint and a fresh output for a new fit")
        scratch = Path(stack.enter_context(TemporaryDirectory(dir=output, prefix="raw-frames-")))
        start_time = perf_counter()
        source_bodies = tuple(args.source_bodies.split(","))
        episodes, records = read_episodes(args.demos, scratch, source_bodies)
        train_eps, valid_eps, held = split_poses(episodes, args.seed, source_bodies)
        groups = {b: [ep for ep in train_eps if ep.body == b] for b in source_bodies}
        anchors = valid_anchors(valid_eps)
        for r in records:
            r["split"] = "validation" if r["pose"] in held else "train"
        motor = Motor(args.weights)
        states = np.concatenate([ep.state for ep in train_eps]).astype(np.float64)
        motor.state_mean.copy_(torch.tensor(states.mean(0)))
        motor.state_scale.copy_(torch.tensor(np.maximum(states.std(0), 0.01)))
        adapter = [p for p in motor.vlm.parameters() if p.requires_grad]
        parameters = adapter + list(motor.head.parameters())
        optim = torch.optim.AdamW(
            [{"params": adapter, "lr": LORA_LR}, {"params": motor.head.parameters(), "lr": HEAD_LR}],
            weight_decay=0.01,
        )
        code = Path(__file__).resolve().parent
        prop = torch.cuda.get_device_properties(0)
        contract = {
            "schema": SCHEMA,
            "seed": args.seed,
            "microbatch": args.microbatch,
            "recipe": recipe(),
            "source_bodies": list(source_bodies),
            "weights": motor.identity,
            "episodes": records,
            "holdout_poses": held,
            "train_poses": sorted({ep.pose for ep in train_eps}),
            "all_motor_poses": sorted({ep.pose for ep in episodes}),
            "collector_index_sha256": sha256(args.demos / "index.json")
            if (args.demos / "index.json").is_file()
            else None,
            "validation_anchors": [(ep.body, ep.pose, t) for ep, t in anchors],
            "sources": {
                name: sha256(code / name) for name in ("motor.py", "motor_data.py", "motor_weights.py")
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "gpu": prop.name,
                "capability": [prop.major, prop.minor],
            },
            "trainable": {
                "lora": sum(p.numel() for p in adapter),
                "head": sum(p.numel() for p in motor.head.parameters()),
            },
        }
        completed_steps, curve, best = 0, [], None
        if args.resume:
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            if saved["schema"] != SCHEMA or saved["contract"] != contract or saved["step"] > args.steps:
                raise ValueError("Resume data, weights, recipe, code, runtime or step count differs")
            motor.restore_delta(saved["model"])
            optim.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"], sampler)
            completed_steps, curve, best = saved["step"], saved["curve"], saved["best"]
            del saved
            if completed_steps == args.steps:
                write_curve(output, checkpoint, best, curve)
                return
        atomic_json(
            output / "manifest.json",
            {
                "contract": contract,
                "demos_path": str(args.demos.resolve()),
                "weights_path": str(args.weights.resolve()),
                "requested_steps": args.steps,
                "argv": sys.argv,
                "allocated_gpu_bytes": prop.total_memory,
                "source_revision": os.environ.get("MOTOR_CODE_REVISION"),
                "episode_count": len(records),
                "setup_seconds": perf_counter() - start_time,
            },
        )
        torch.cuda.reset_peak_memory_stats()
        for step in range(completed_steps + 1, args.steps + 1):
            torch.cuda.synchronize()
            tick = perf_counter()
            batch = select_batch(groups, sampler, GLOBAL_BATCH)
            optim.zero_grad(set_to_none=True)
            sum_loss = torch.zeros((), device="cuda")
            for i in range(0, GLOBAL_BATCH, args.microbatch):
                obs, target, mask = collate(batch[i : i + args.microbatch])
                pred = motor(obs)
                losses = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none").mean(-1)
                per_anchor = (losses * mask).sum(1) / mask.sum(1)
                loss = per_anchor.sum() / GLOBAL_BATCH
                loss.backward()
                sum_loss += loss.detach()
            # At init only LoRA's zero-initialized up projections have gradient.
            # Record the first measured signal through the pretrained Gemma tail.
            if step == 1:
                lora_norm = torch.stack([p.grad.norm() for p in adapter if p.grad is not None]).norm().item()
                if not math.isfinite(lora_norm) or lora_norm <= 0:
                    raise RuntimeError("The supervised actions produced no finite LoRA gradient")
            norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optim.step()
            torch.cuda.synchronize()
            seconds = perf_counter() - tick
            item = {
                "step": step,
                "train_huber": sum_loss.item(),
                "grad_norm": norm.item(),
                "update_seconds": seconds,
                "frames_per_second": GLOBAL_BATCH / seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if step == 1 or step % VALIDATE_EVERY == 0 or step == args.steps:
                if step == 1:
                    item["lora_grad_norm"] = lora_norm
                val_start = perf_counter()
                item["validation"] = validate(motor, anchors, args.microbatch, source_bodies)
                item["validation_seconds"] = perf_counter() - val_start
                if best is None or item["validation"]["mae"] < best["mae"]:
                    best = {"step": step, "mae": item["validation"]["mae"], "model": motor.delta()}
            curve.append(item)
            if step == 1 or step % CHECKPOINT_EVERY == 0 or step == args.steps:
                tmp = checkpoint.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "schema": SCHEMA,
                        "contract": contract,
                        "step": step,
                        "curve": curve,
                        "best": best,
                        "model": motor.delta(),
                        "optimizer": optim.state_dict(),
                        "rng": rng_state(sampler),
                    },
                    tmp,
                )
                tmp.replace(checkpoint)
                write_curve(output, checkpoint, best, curve)
                print(json.dumps(item, allow_nan=False), flush=True)


def write_curve(output, checkpoint, best, curve):
    atomic_json(
        output / "curve.json",
        {
            "best_step": best["step"],
            "best_mae": best["mae"],
            "checkpoint_sha256": sha256(checkpoint),
            "updates": curve,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    for flag in ("demos", "output", "weights"):
        fit.add_argument("--" + flag, type=Path, required=True)
    fit.add_argument(
        "--steps", type=int, default=300, help="Total optimizer updates, including resumed updates"
    )
    fit.add_argument("--source-bodies", choices=("Panda", ",".join(BODIES)), default=",".join(BODIES))
    fit.add_argument("--seed", type=int, default=20261120)
    fit.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=4)
    fit.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
