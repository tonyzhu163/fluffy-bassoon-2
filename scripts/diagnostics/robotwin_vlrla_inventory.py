"""Verify CPU imports and static RoboTwin/VLRLA inputs without loading a model."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run CPU qualification on allocated Linux resources")
    project = Path(__file__).resolve().parents[2]
    protocol_path = project / "configs/robotwin_vlrla.json"
    protocol = json.loads(protocol_path.read_text())
    tasks = protocol["seen_tasks"] + protocol["unseen_tasks"]
    if (len(protocol["seen_tasks"]) != 6 or len(protocol["unseen_tasks"]) != 6
            or len(set(tasks)) != 12 or "adjust_bottle" not in protocol["unseen_tasks"]):
        raise ValueError("Incorrect paper task split")
    for package in ("sapien", "mplib"):
        if metadata.version(package) != protocol["runtime"][f"{package}_version"]:
            raise ValueError(f"Incorrect {package} version")
    importlib.import_module("mplib.sapien_utils")
    for task in tasks:
        if not (args.robotwin_root / "envs" / f"{task}.py").is_file():
            raise FileNotFoundError(task)
        importlib.import_module(f"envs.{task}")
    importlib.import_module("rlinf.envs.sim.robotwin.robotwin_env")
    importlib.import_module("rlinf.models")
    for asset in ("assets/embodiments", "assets/objects", "task_config/demo_randomized.yml"):
        if not (args.robotwin_root / asset).exists():
            raise FileNotFoundError(asset)
    index_path = args.model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in set(index["weight_map"].values()):
        if not (args.model_path / name).is_file():
            raise FileNotFoundError(name)
    norm = args.model_path / "physical-intelligence/robotwin/norm_stats.json"
    json.loads(norm.read_text())
    result = dict(cpu_imports_passed=True, tasks=tasks, executable=sys.executable,
        project_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        packages={p: metadata.version(p) for p in ["torch", "numpy", "sapien", "mplib", "open3d", "transformers", "rlinf-openpi"]},
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        model_path=str(args.model_path), gpu_work_performed=False, renderer_qualified=False,
        model_loaded=False, classification="CPU setup only; real allocated simulator episode still required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
