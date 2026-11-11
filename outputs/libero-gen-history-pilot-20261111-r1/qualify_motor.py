"""Bounded frozen-motor qualification. Run only in an allocated Linux GPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time


TASKS = [
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_2",
]
SPLIT = "libero_spatial_selected_combinations_inverse_view"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 2**20):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-test", type=int, default=1, choices=[1, 5, 20])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import yaml
    import torch
    import dill
    import hydra
    from omegaconf import OmegaConf
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    random.seed(7301)
    np.random.seed(7301)
    torch.manual_seed(7301)
    base = args.source / "behavior_prompting/train_network"
    init_dir = args.output / "paired_init" / SPLIT
    init_dir.mkdir(parents=True)
    source_init = base / "env/libero/init_files" / SPLIT / (TASKS[0] + ".pruned_init")
    for task in TASKS:
        shutil.copyfile(source_init, init_dir / (task + ".pruned_init"))
    config_dir = args.output / "libero_config"
    config_dir.mkdir()
    config = {"benchmark_root": str(args.source / "deps/LIBERO/libero/libero"),
              "assets": str(args.source / "deps/LIBERO/libero/libero/assets"),
              "bddl_files": str(base / "env/libero/bddl_files"),
              "init_states": str(args.output / "paired_init"),
              "datasets": str(args.assets / "dataset/demonstration_data")}
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    from behavior_prompting.train_network.env_runner.libero_image_runner import LiberoImageRunner
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    discover_and_register_benchmarks()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    checkpoint = args.assets / "model/liberogen_spatial_combination_behavior_prompting.ckpt"
    assert digest(checkpoint) == "74e0f841f3b86f1383adda23dab59fddee8806a4bcb5685ed7f826afb320c097"
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    (args.output / "checkpoint-config.yaml").write_text(OmegaConf.to_yaml(cfg))
    # Strictly load all saved weights, avoiding an unnecessary pretrained-ViT download.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    def disable_pretrained(node):
        if OmegaConf.is_dict(node):
            for key in node:
                if key == "pretrained":
                    node[key] = False
                else:
                    disable_pretrained(node[key])
        elif OmegaConf.is_list(node):
            for value in node:
                disable_pretrained(value)
    disable_pretrained(model_cfg)
    policy = hydra.utils.instantiate(model_cfg)
    policy.load_state_dict(payload["state_dicts"]["model"], strict=True)
    del payload
    policy.requires_grad_(False).eval().cuda()
    versions = {name: p._version for name, p in policy.named_parameters()}
    (args.output / "instantiated-model-config.yaml").write_text(OmegaConf.to_yaml(model_cfg))

    class TraceProxy:
        def __init__(self, inner, folder):
            self.inner, self.folder = inner, folder
            self.initial = []
            self.events = []
            self.native_rewards = []
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def reset(self, *a, **kw):
            obs = self.inner.reset(*a, **kw)
            index = len(self.initial)
            np.savez_compressed(self.folder / f"reset-{index:03d}.npz", **obs)
            hashes = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                      for key, value in obs.items()}
            self.initial.append(hashes)
            self.events.append({"type": "reset", "index": index})
            return obs
        def step(self, action):
            obs, reward, done, info = self.inner.step(action)
            index = sum(event["type"] == "step" for event in self.events)
            np.savez_compressed(self.folder / f"step-{index:05d}.npz", action=action,
                                reward=reward, done=done)
            self.events.append({"type": "step", "index": index})
            return obs, reward, done, info
        def call(self, name, *a, **kw):
            value = self.inner.call(name, *a, **kw)
            if name == "get_attr" and a == ("reward",):
                self.native_rewards.append([np.asarray(v).tolist() for v in value])
            return value

    results = []
    initial_hashes = []
    for task in TASKS:
        folder = args.output / task
        folder.mkdir()
        runner_kwargs = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
        runner_kwargs.pop("_target_", None)
        runner_kwargs["shape_meta"] = cfg.task.shape_meta
        runner_kwargs.update(dataset_path=str(args.assets / "dataset/demonstration_data" / SPLIT / (task + "_demo.hdf5")),
                             output_dir=str(folder), cache_dir=str(args.output / "prompt-cache"),
                             n_envs=1, n_train=0, n_train_vis=0, n_test=args.n_test,
                             n_test_vis=args.n_test, is_seen=True, eval_seen=True, eval_unseen=False,
                             vis_prompt=False, vis_attention_map=False, vis_actions_rollout=False,
                             vis_proprio_rollout=False, vis_trajectory_video=False, render_gpu_device_id=0)
        runner = LiberoImageRunner(**runner_kwargs)
        proxy = TraceProxy(runner.env, folder)
        runner.env = proxy
        started = time.monotonic()
        try:
            with torch.inference_mode():
                scores = runner.run(policy, enable_expensive_vis=False)
            numeric_scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float, np.number))}
            result = {"task": task, "seconds": time.monotonic() - started, "scores": numeric_scores,
                      "native_reward_traces": proxy.native_rewards, "events": proxy.events,
                      "reset_hashes": proxy.initial,
                      "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
            (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            initial_hashes.append(proxy.initial[:args.n_test])
            print("MOTOR_RESULT", task, json.dumps(numeric_scores), flush=True)
        finally:
            runner.close()
    assert initial_hashes[0] == initial_hashes[1], "Candidate initial observations differ"
    assert versions == {name: p._version for name, p in policy.named_parameters()}, "Motor parameters changed"
    final = {"complete": True, "paired_initial_observations": True,
             "checkpoint_sha256": digest(checkpoint), "source_init_sha256": digest(source_init),
             "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
             "n_test_per_candidate": args.n_test, "results": results}
    (args.output / "result.json").write_text(json.dumps(final, indent=2) + "\n")
    (args.output / "complete").write_text("complete\n")


if __name__ == "__main__":
    main()
