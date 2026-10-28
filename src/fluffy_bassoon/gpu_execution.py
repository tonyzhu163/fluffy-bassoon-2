"""Explicit GPU execution boundary for scheduler and authorized direct hosts."""
import re


def validate_execution(mode, environ):
    if mode == "slurm":
        if not environ.get("SLURM_JOB_ID") or environ.get("SLURM_STEP_ID") in (None, "batch", "extern"):
            raise ValueError("SLURM execution requires an active srun job step")
    elif mode == "baige":
        import socket
        from .baige_gpu import validate_container
        validate_container(environ, socket.gethostname())
        selected = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(selected) != 1 or re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]{36})", selected[0]) is None:
            raise ValueError("Baige evaluation requires one explicitly selected GPU")
    elif mode == "direct":
        if environ.get("SLURM_JOB_ID"):
            raise ValueError("Direct execution cannot bypass a SLURM allocation")
        selected = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(set(selected)) != len(selected) or any(
            re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]{36})", value) is None for value in selected
        ):
            raise ValueError("Direct execution requires explicit distinct authorized CUDA devices")
    else:
        raise ValueError("Unknown GPU execution environment")
