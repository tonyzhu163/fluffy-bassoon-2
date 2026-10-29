"""Diagnose frozen VLA prefix numerics inside an allocated simulator/motor runtime.

This command always writes measurements before enforcing the original feature
qualification threshold. It does not train or run robot-control episodes.
"""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", choices=("pi0_libero", "pi05_libero"), default="pi0_libero")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--initial-state", type=int, default=40)
    parser.add_argument("--scene-seed", type=int, default=100000)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Feature model computation requires allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    from .vla_libero import FrozenOpenPI, SimulatorClient, attest_process_gpu, write_json
    import os
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    root = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != args.source_commit or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Require exact clean reviewed source")
    args.output.mkdir(parents=True, exist_ok=False)
    args.feature_interface = "vla-prefix"
    sim = SimulatorClient(args.port)
    try:
        obs = sim.reset(args.initial_state, args.scene_seed)["observation"]
        motor = FrozenOpenPI(args)
        report = motor.feature_encoder.diagnose(obs, sim.metadata["instruction"])
        report.update(source_commit=actual, hostname=socket.gethostname(),
                      simulator=sim.metadata, motor=motor.metadata,
                      initial_state=args.initial_state, scene_seed=args.scene_seed)
        # Measurements survive a binding or original-threshold failure.
        write_json(args.output / "feature-diagnostic.json", report)
        attest_process_gpu(args.output / "feature-compute-binding.json", graphics=False)
        print(json.dumps(report), flush=True)
        if not report["identical_feature_function_deterministic"]:
            raise RuntimeError("Repeated identical feature function is not deterministic")
        if not report["original_qualification_passed"]:
            raise RuntimeError("Original feature threshold still fails; numerical diagnostics saved")
    except Exception as error:
        write_json(args.output / "failure.json", dict(error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    main()
