"""Allocated same-process OpenPI/JAX and history-student CUDA qualification.

No pretrained motor weights are loaded. Real JAX GPU convolutions bracket two
HistoryNoisePolicy CUDA optimization steps, using the production NLL path.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, value in policy.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-torch-cuda")
    parser.add_argument("--expected-torch-version")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Require a fresh diagnostic receipt")
    started = time.monotonic()
    report = dict(status="running", label=args.label, hostname=socket.gethostname(),
                  pid=os.getpid(), executable=sys.executable, python=sys.version,
                  source_commit=args.source_commit, purpose="same-process JAX and CUDA student qualification")
    def stage(name):
        report.update(stage=name, elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output, report)
    try:
        stage("execution_boundary")
        if sys.platform != "linux":
            raise ValueError("Run model computation only on allocated Linux")
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
        root = Path(__file__).resolve().parents[3]
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
            raise ValueError("Require the exact clean reviewed diagnostic source")
        stage("import_production_runtime")
        # Match the motor path's import order before either framework computes.
        import jax
        from openpi.policies import policy_config  # noqa: F401
        from openpi.training import config
        import numpy as np
        import torch
        model_config = config.get_config("pi05_libero").model
        if not model_config.pi05 or model_config.action_horizon != 10:
            raise ValueError("The diagnostic imported a different OpenPI backbone config")
        report["packages"] = {}
        for name, expected in (("numpy", "1.26.4"), ("jax", "0.5.3"),
                               ("flax", "0.10.2"), ("optax", "0.2.4"), ("torch", None)):
            package = importlib.import_module(name)
            filename = Path(package.__file__).resolve()
            version = package.__version__
            report["packages"][name] = dict(version=version, file=str(filename),
                source_file_sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
            if expected is not None and version != expected:
                raise ValueError(f"Original OpenPI dependency changed: {name} {version} != {expected}")
        report["torch_cuda_build"] = torch.version.cuda
        version_file = Path(torch.version.__file__).resolve()
        report["torch_version_file"] = dict(file=str(version_file),
            sha256=hashlib.sha256(version_file.read_bytes()).hexdigest())
        if (args.expected_torch_version and torch.__version__ != args.expected_torch_version
                or args.expected_torch_cuda and torch.version.cuda != args.expected_torch_cuda):
            raise ValueError("The isolated CUDA overlay did not select the declared Torch wheel")
        stage("jax_gpu_convolution_before_student")
        devices = jax.devices("gpu")
        report["jax_devices"] = list(map(str, devices))
        if len(devices) != 1:
            raise ValueError("Require exactly one allocation-visible JAX GPU")
        import jax.numpy as jnp
        @jax.jit
        def convolution(images, kernel):
            return jax.nn.relu(jax.lax.conv_general_dilated(images, kernel,
                window_strides=(2, 2), padding="VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")))
        images = jax.device_put(jnp.sin(jnp.arange(2 * 64 * 64 * 3, dtype=jnp.float32)).reshape(2, 64, 64, 3), devices[0])
        kernel = jax.device_put(jnp.cos(jnp.arange(3 * 3 * 3 * 32, dtype=jnp.float32)).reshape(3, 3, 3, 32) / 32., devices[0])
        before_array = convolution(images, kernel)
        if before_array.device.platform != "gpu":
            raise RuntimeError("JAX diagnostic convolution did not execute on the GPU")
        before = np.asarray(before_array)
        if not np.isfinite(before).all():
            raise RuntimeError("Non-finite JAX GPU convolution before student training")
        report["jax_convolution_before"] = dict(shape=list(before.shape),
            mean=float(before.mean()), sha256=hashlib.sha256(before.tobytes()).hexdigest(),
            device=str(before_array.device))
        stage("torch_cuda_device_and_kernels")
        report.update(torch_cuda_available=torch.cuda.is_available(),
            torch_device_count=torch.cuda.device_count(), torch_arch_list=torch.cuda.get_arch_list())
        if not report["torch_cuda_available"] or report["torch_device_count"] != 1:
            raise ValueError("Require exactly one allocation-visible Torch CUDA device")
        report.update(torch_device_name=torch.cuda.get_device_name(0),
                      torch_device_capability=list(torch.cuda.get_device_capability(0)))
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        from .vla_history import HistoryConfig, HistoryNoisePolicy
        policy = HistoryNoisePolicy(HistoryConfig(observation_dim=4104, noise_dim=32,
            width=128, layers=2, heads=4, context_length=300)).to("cuda")
        report["student_device"] = str(next(policy.parameters()).device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
        tokens = torch.randn(4, 2, policy.core.input_dim, device="cuda") * .1
        targets = torch.rand(4, 2, 32, device="cuda") * 2. - 1.
        valid = torch.ones(4, 2, dtype=torch.bool, device="cuda")
        episode_starts = torch.zeros_like(valid)
        episode_starts[0] = True
        episode_starts[2] = True
        torch.cuda.synchronize()
        initial = parameter_digest(policy)
        report["student_parameter_sha256_before"] = initial
        report["student_updates"] = []
        for update in range(1, 3):
            stage(f"student_cuda_update_{update}")
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = policy.sequence_nll(tokens, targets, valid, episode_starts, mode="full")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            if not math.isfinite(float(loss.detach())) or not 0 < float(gradient_norm) < math.inf:
                raise RuntimeError("Student NLL/gradient is non-finite or has no gradient")
            optimizer.step()
            torch.cuda.synchronize()
            report["student_updates"].append(dict(update=update, nll=float(loss.detach()),
                gradient_norm=float(gradient_norm), seconds=time.monotonic() - update_started))
        final = parameter_digest(policy)
        report["student_parameter_sha256_after"] = final
        if final == initial or any(not bool(torch.isfinite(value).all()) for value in policy.parameters()):
            raise RuntimeError("Student optimization did not make a finite parameter update")
        stage("jax_gpu_convolution_after_student")
        after = np.asarray(convolution(images, kernel))
        if not np.isfinite(after).all() or not np.allclose(before, after, rtol=1e-5, atol=1e-6):
            raise RuntimeError("JAX GPU convolution changed after Torch student CUDA optimization")
        report["jax_convolution_after"] = dict(shape=list(after.shape), mean=float(after.mean()),
            sha256=hashlib.sha256(after.tobytes()).hexdigest(),
            maximum_delta=float(np.max(np.abs(before - after))))
        stage("actual_process_gpu_binding")
        from .vla_libero import attest_process_gpu
        report["process_gpu_binding"] = attest_process_gpu(
            args.output.with_suffix(".binding.json"), graphics=False)
        recovery = args.output.with_suffix(".recovery.pt")
        torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), update=2,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            source_commit=args.source_commit), recovery)
        report.update(status="passed", recovery_checkpoint=str(recovery),
            recovery_sha256=hashlib.sha256(recovery.read_bytes()).hexdigest())
        stage("complete")
        print(json.dumps(report), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, report)
        print(json.dumps(report), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
