"""CPU readback of the paired policy inputs and one native original-PRO reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def policy():
    import cv2
    import eval_stage2a_live_ur5e as stage
    import jax
    from openpi.policies import policy_config
    from openpi.training import config

    require(all(d.platform == "cpu" for d in jax.devices()), "Expected JAX CPU devices")
    openpi = Path(os.environ["OPENPI_SOURCE"]).resolve()
    require(Path(policy_config.__file__).resolve().is_relative_to(openpi), "Unexpected OpenPI")
    require(
        Path(stage.__file__).resolve().is_relative_to(Path(os.environ["SYMMETA_SOURCE"])),
        "Unexpected reader implementation",
    )
    config_name = config.get_config("pi05_libero").name
    ckpt = Path(os.environ["PI05_CHECKPOINT"])
    marker = ckpt / "READY.json"
    marker_hash = sha256(marker)
    require(marker_hash == "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170", "Policy manifest changed")
    manifest = json.loads(marker.read_text())
    require(manifest["checkpoint"] == "gs://openpi-assets/checkpoints/pi05_libero", "Wrong policy")
    for item in manifest["files"]:
        path = ckpt / item["name"]
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"Bad policy file {path}")
    reader = Path(os.environ["SYMMETA_CHECKPOINT"])
    digest = sha256(reader)
    require(digest == "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4", "Bad reader")
    params, cfg, meta = stage.load_v2_checkpoint(reader)
    require(len(jax.tree.leaves(params)) > 0 and cv2.DISOpticalFlow_create() is not None, "Reader unavailable")
    return {
        "policy_config": config_name,
        "policy_source": policy_config.__file__,
        "policy_manifest_sha256": marker_hash,
        "policy_files_hashed": len(manifest["files"]),
        "policy_bytes_hashed": sum(item["bytes"] for item in manifest["files"]),
        "reader_source": stage.__file__,
        "reader_sha256": digest,
        "reader_tag": meta["tag"],
        "reader_config": cfg.model_type,
        "reader_param_arrays": len(jax.tree.leaves(params)),
        "jax_devices": [str(d) for d in jax.devices()],
        "opencv": cv2.__version__,
    }


def sim():
    import numpy as np
    import torch
    import mujoco
    import robosuite
    from OpenGL import GL
    from liberopro.liberopro import benchmark
    from liberopro.liberopro.envs import OffScreenRenderEnv
    import inspect

    release = Path(os.environ["ORIGINAL_PRO_ROOT"]).resolve()
    source = Path(inspect.getfile(OffScreenRenderEnv)).resolve()
    require(source.is_relative_to(release) and Path(benchmark.__file__).is_relative_to(release), "Wrong simulator")
    assets = Path(os.environ["ORIGINAL_PRO_ASSETS"])
    counts = {}
    for base in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        for suffix in ("object", "swap", "lan", "task"):
            family = f"{base}_{suffix}"
            tasks = {p.stem for p in (assets / "bddl_files" / family).glob("*.bddl")}
            states = {p.stem for p in (assets / "init_files" / family).glob("*.pruned_init")}
            require(tasks == states and len(tasks) == 10, f"Incomplete {family} assets")
            counts[family] = len(tasks)

    family = "libero_goal_task"
    stem = "put_the_bowl_on_the_stove"
    bddl = assets / "bddl_files" / family / f"{stem}.bddl"
    inits = assets / "init_files" / family / f"{stem}.pruned_init"
    bddl_sha, init_sha = sha256(bddl), sha256(inits)
    require(bddl_sha == "917e7f6796897bd88ed248872b3b03183b27a794aed19566ef41116c8518c6eb", "Unexpected BDDL")
    require(init_sha == "e681dd9c20bc64127118c5743d3d9016e5754de17111e75254219f12b5d0be46", "Unexpected starts")
    states = torch.load(inits, map_location="cpu", weights_only=False)
    np.random.seed(7)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             render_gpu_device_id=0)
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(states[28])
        require(np.array_equal(env.get_sim_state(), states[28]), "State restoration differs")
        arm = env.robots[0]
        robot = type(arm.robot_model).__name__
        low, high = env.env.action_spec
        require(robot == "MountedPanda" and arm.controller.name == "OSC_POSE", "Unexpected native controller")
        require(low.shape == high.shape == (7,) and env.env.control_freq == 20, "Unexpected control interface")
        require(env.language_instruction.lower() == "put the plate on the stove", "Unexpected native instruction")
        cameras = {}
        for name in ("agentview_image", "robot0_eye_in_hand_image"):
            frame = np.ascontiguousarray(obs[name])
            require(frame.shape == (256, 256, 3) and frame.dtype == np.uint8 and np.ptp(frame) > 0, name)
            cameras[name] = {"shape": list(frame.shape), "sha256": hashlib.sha256(frame).hexdigest()}
        renderer = (GL.glGetString(GL.GL_RENDERER) or b"").decode()
        require("llvmpipe" in renderer.lower(), f"Expected Mesa llvmpipe, got {renderer!r}")
        _, reward, done, _ = env.step([0.0] * 6 + [-1.0])
        return {
            "simulator_source": str(source), "mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
            "renderer": renderer, "robot": robot, "controller": arm.controller.name,
            "action_dim": int(low.size), "control_hz": env.env.control_freq, "case_id": "libero_goal_task/01/28",
            "bddl_sha256": bddl_sha, "init_sha256": init_sha, "available_pairs": sum(counts.values()),
            "stored_states_shape": list(states.shape), "selected_state_shape": list(states[28].shape),
            "state_exact": True, "seed": 7, "native_instruction": env.language_instruction, "cameras": cameras,
            "one_settling_control": {"reward": float(reward), "done": bool(done), "success": bool(env.check_success())},
        }
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("policy", "sim"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(platform.system() == "Linux" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Requires CPU-only Linux")
    require(os.environ.get("JAX_PLATFORMS") == "cpu" and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
            "Requires the CPU qualification environment")
    result = (policy if args.mode == "policy" else sim)()
    result.update(host=platform.node(), python=sys.version.split()[0], mode=args.mode, executable=sys.executable)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result, sort_keys=True))
