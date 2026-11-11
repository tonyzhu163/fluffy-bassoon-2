"""CPU import and prompt-schema preflight on allocated Linux; no model execution."""
import argparse
import importlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    import yaml
    base = args.source / "behavior_prompting/train_network"
    config = {"benchmark_root": str(args.source / "deps/LIBERO/libero/libero"),
              "assets": str(args.source / "deps/LIBERO/libero/libero/assets"),
              "bddl_files": str(base / "env/libero/bddl_files"),
              "init_states": str(base / "env/libero/init_files"),
              "datasets": str(args.assets / "dataset/demonstration_data")}
    (args.output / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(args.output)
    modules = ["torch", "torchvision", "gym", "robosuite", "mujoco", "h5py", "open3d",
               "behavior_prompting.train_network.env_runner.libero_image_runner",
               "behavior_prompting.train_network.policy.diffusion_unet_policy",
               "behavior_prompting.train_network.model.prompt.prompt_obs_encoder"]
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "imported")
        print("IMPORT_OK", name, versions[name], flush=True)
    import h5py
    from qualify_motor import TASKS, SPLIT
    files = []
    for task in TASKS:
        path = args.assets / "dataset/demonstration_data" / SPLIT / (task + "_demo.hdf5")
        with h5py.File(path, "r") as data:
            names = sorted(data["data"], key=lambda x: int(x.removeprefix("demo_")))
            assert names, "No prompt demonstrations"
            demo = data["data"][names[0]]
            assert len(demo["actions"]) > 20 and demo["actions"].shape[-1] == 7
            assert "agentview_rgb" in demo["obs"] and "eye_in_hand_rgb" in demo["obs"]
            files.append({"task": task, "demonstrations": len(names), "sample": names[0],
                          "actions_shape": list(demo["actions"].shape),
                          "observation_shapes": {k: list(v.shape) for k, v in demo["obs"].items()}})
    result = {"complete": True, "stage": "CPU imports and HDF5 schema only",
              "versions": versions, "prompts": files, "gpu_execution_qualified": False}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
