"""Imported once by each clean Linux forkserver, before any simulation, VLA or device context."""
import os
import sys

import torch
from lerobot.envs.libero import LiberoEnv
from reground.eval.rollout import _env_worker

if torch.cuda.is_initialized() or len(os.listdir("/proc/self/task")) != 1:
    raise RuntimeError("Simulator import server requires a single thread and uninitialized CUDA")
for name in ("mujoco.egl", "robosuite.renderers.context.egl_context"):
    if getattr(sys.modules.get(name), "EGL_DISPLAY", None) is not None:
        raise RuntimeError("Simulator import server must not acquire an EGL context")
print(f"Simulator import server {os.getpid()} ready, single thread, no CUDA/EGL context", flush=True)
