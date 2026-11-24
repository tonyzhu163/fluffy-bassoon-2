"""Frozen SmolVLA on physical LIBERO-Pro2.0 arms. CPU only, one native episode per call.

    bash run.sh SCRATCH TAG ARM LAYOUT [LAYOUT ...]          # see README.md
    python harness.py --scratch SCRATCH --arm Kinova3 --layout 14 --out evidence/raw

Adapters subclass `Hooks`. They receive public observations and issued commands only.
"""
import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

POLICY_REV = "a24f1f380461b665158d52acd140152c79e8b36d"
VLM_REV = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
WEIGHT_SHA = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
PARAMETERS = 450_046_176
SEED_BASE = 42710300
NOISE_SHAPE = (1, 50, 32)
PUBLIC = ("agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_eef_pos",
          "robot0_eef_quat_xyzw", "robot0_gripper_qpos")


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


class Hooks:
    """Adapter points. The defaults are the untreated policy.

    `context` is `dict(step, replan, commands)`: controls issued so far, the replan index and the
    list of issued 7D commands. Nothing in it comes from reward, contacts or simulator state.
    """

    name = "raw"
    proposals = 1  # K noise draws and forward passes per replan
    replan = 50    # controls consumed per chosen chunk (official n_action_steps)

    def observation(self, obs, context):
        """Public observation dict (both images and proprioception) the motor preprocesses."""
        return obs

    def state(self, state8, obs, context):
        """Unnormalized 8D policy state: grip-site xyz, hand axis-angle, two finger slides."""
        return state8

    def prompt(self, prompt, obs, context):
        return prompt

    def noise(self, draw, context):
        """`draw` is the next standard-normal array of NOISE_SHAPE from the episode's sequence."""
        return draw

    def choose(self, proposals, obs, context):
        """Index of the proposal (each 50x7, unnormalized LIBERO commands) to execute."""
        return 0


class Motor:
    """Strictly loaded, fully frozen `lerobot/smolvla_libero` with its released processors."""

    def __init__(self, scratch, threads=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla import processor_smolvla  # noqa: F401  registers processor steps
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.env_processor import LiberoProcessorStep

        if threads:
            torch.set_num_threads(threads)
        self.torch, self._preprocess = torch, preprocess_observation
        path, vlm = Path(scratch) / "policy", Path(scratch) / "vlm"
        with (path / "model.safetensors").open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != WEIGHT_SHA:
                raise RuntimeError("Unexpected SmolVLA weights")
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = "cpu"
        cfg.vlm_model_name = str(vlm)
        cfg.load_vlm_weights = False  # architecture only; strict loading supplies every trained weight
        self.cfg = cfg
        self.policy = SmolVLAPolicy.from_pretrained(path, config=cfg, strict=True)
        self.policy.requires_grad_(False).eval()
        self.pre, self.post = make_pre_post_processors(cfg, str(path), preprocessor_overrides={
            "device_processor": {"device": "cpu"}, "tokenizer_processor": {"tokenizer_name": str(vlm)}})
        self.env_step = LiberoProcessorStep()
        parameters = list(self.policy.parameters())
        if (sum(p.numel() for p in parameters) != PARAMETERS or any(p.requires_grad for p in parameters)
                or any(m.training for m in self.policy.modules())):
            raise RuntimeError("SmolVLA is not the complete frozen checkpoint")
        self.weights_before = self.fingerprint()

    def fingerprint(self):
        """Cheap readback over every parameter tensor; equal before and after a run."""
        h = hashlib.sha256()
        for name, p in self.policy.named_parameters():
            h.update(name.encode())
            h.update(p.detach().float().sum().numpy().tobytes())
        return h.hexdigest()[:16]

    def proposal(self, obs, prompt, noise, hooks=None, context=None):
        """One 50x7 unnormalized chunk and the digests of what the policy was given."""
        torch = self.torch
        with torch.inference_mode():
            self.policy.reset()
            batch = self._preprocess({
                "pixels": {"image": obs["agentview_image"], "image2": obs["robot0_eye_in_hand_image"]},
                "robot_state": {"eef": {"pos": obs["robot0_eef_pos"][None], "quat": obs["robot0_eef_quat_xyzw"][None]},
                                "gripper": {"qpos": obs["robot0_gripper_qpos"][None]}},
            })
            batch = self.env_step.observation(batch)
            state = batch["observation.state"]
            if hooks is not None:
                changed = np.asarray(hooks.state(state[0].numpy().copy(), obs, context), dtype=np.float32)
                if changed.shape != (8,) or not np.isfinite(changed).all():
                    raise ValueError("State hook must return a finite 8D state")
                batch["observation.state"] = torch.from_numpy(changed)[None].to(state.dtype)
            used_state = batch["observation.state"][0].numpy().copy()
            batch["task"] = prompt
            noise = torch.from_numpy(np.asarray(noise, dtype=np.float32).copy())
            actions = self.policy.predict_action_chunk(self.pre(batch), noise=noise)
            chunk = self.post(actions)[0].cpu().numpy().astype(np.float32)
        if chunk.shape != (50, 7) or not np.isfinite(chunk).all():
            raise RuntimeError("Invalid SmolVLA proposal")
        return chunk, dict(state=used_state.tolist(), state_digest=digest(used_state),
                           agent_digest=digest(obs["agentview_image"]),
                           wrist_digest=digest(obs["robot0_eye_in_hand_image"]))


def public(observation):
    return {key: observation[key] for key in PUBLIC}


def run_episode(env, motor, layout, seed, noise_seed, hooks=None, log=None):
    """One native episode. `env` follows libero-crossarm's ArmEnv reset/step contract."""
    hooks = hooks or Hooks()
    first = env.reset(init_state_id=layout, seed=seed)
    obs, instruction = public(first["observation"]), first["language_instruction"]
    rng = np.random.default_rng(noise_seed)
    commands, replans, reward, clipped = [], [], 0.0, 0
    success = terminated = truncated = False
    started = time.time()
    while not (terminated or truncated):
        context = dict(step=len(commands), replan=len(replans), commands=[c.copy() for c in commands])
        seen = hooks.observation(obs, context)
        prompt = hooks.prompt(instruction, seen, context)
        draws = [np.asarray(hooks.noise(rng.standard_normal(NOISE_SHAPE, dtype=np.float32), context), dtype=np.float32)
                 for _ in range(hooks.proposals)]
        if any(d.shape != NOISE_SHAPE for d in draws):
            raise ValueError("Noise hook must keep NOISE_SHAPE")
        results = [motor.proposal(seen, prompt, d, hooks, context) for d in draws]
        proposals = [r[0] for r in results]
        chosen = int(hooks.choose(proposals, seen, context))
        record = dict(results[chosen][1], step=len(commands), prompt=prompt, chosen=chosen,
                      noise_digests=[digest(d) for d in draws], proposal_digests=[digest(p) for p in proposals])
        replans.append(record)
        for raw in proposals[chosen][:hooks.replan]:
            command = np.clip(raw, -1, 1).astype(np.float32)
            clipped += int(not np.array_equal(command, raw))
            step = env.step(command)
            commands.append(command)
            obs = public(step["observation"])
            reward += step["reward"]
            success, terminated, truncated = step["success"], step["terminated"], step["truncated"]
            if terminated or truncated:
                break
        if log:
            log(f"layout {layout} step {len(commands)} success {success}")
    reproducibility = first.get("environment_reproducibility", {})
    return dict(layout=int(layout), seed=int(seed), noise_seed=int(noise_seed), adapter=hooks.name,
                proposals=hooks.proposals, replan=hooks.replan, instruction=instruction,
                success=bool(success), reward=float(reward), controls=len(commands), truncated=bool(truncated),
                clipped_controls=clipped, max_steps=first.get("max_steps"), replans=replans,
                commands=np.asarray(commands).round(6).tolist(), command_digest=digest(np.asarray(commands)),
                scene_fingerprint=reproducibility.get("scene_fingerprint"),
                initial_state_fingerprint=reproducibility.get("initial_state_fingerprint"),
                alignment=reproducibility.get("alignment"), seconds=round(time.time() - started, 1))


def load_hooks(spec):
    """`module:Class` importable from PYTHONPATH, or empty for the untreated policy."""
    if not spec:
        return Hooks()
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", required=True, type=Path, help="directory prepared by setup.sh")
    parser.add_argument("--arm", required=True, choices=("Panda", "Kinova3", "IIWA", "UR5e", "Sawyer"))
    parser.add_argument("--task", type=int, default=3)
    parser.add_argument("--layout", type=int, nargs="+", required=True)
    parser.add_argument("--noise-offset", type=int, default=0, help="added to the reset seed for policy noise")
    parser.add_argument("--hooks", default="", help="module:Class adapter; default untreated")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 2)))
    args = parser.parse_args()
    from arm_env import ArmEnv  # libero-crossarm-20261120-r1, on PYTHONPATH through run.sh

    args.out.mkdir(parents=True, exist_ok=True)
    hooks = load_hooks(args.hooks)
    motor = Motor(args.scratch, args.threads)
    for layout in args.layout:
        target = args.out / f"{args.arm}-goal{args.task}-layout{layout:02d}.json"
        if target.exists():
            continue
        seed = SEED_BASE + layout
        with ArmEnv(arm=args.arm, suite="libero_goal", task_id=args.task) as env:
            episode = run_episode(env, motor, layout, seed, seed + args.noise_offset, hooks,
                                  log=lambda text: print(args.arm, text, flush=True))
        episode.update(arm=args.arm, suite="libero_goal", task=args.task, policy_revision=POLICY_REV,
                       weights_sha256=WEIGHT_SHA, parameters=PARAMETERS,
                       weights_unchanged=motor.fingerprint() == motor.weights_before,
                       evidence="native-reward", device="cpu", threads=args.threads)
        target.write_text(json.dumps(episode) + "\n")
        print(args.arm, "layout", layout, "success", episode["success"], "controls", episode["controls"],
              "seconds", episode["seconds"], flush=True)


if __name__ == "__main__":
    main()
