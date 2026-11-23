"""Isolated CPU process for the archived seed-41 JAX reader.

The pipe carries only per-episode lane numbers, public snapshots, and five commands
already stepped. It is deliberately independent of LeRobot, ReGround and torch.
"""
from hashlib import sha256
from multiprocessing.connection import Connection
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np


WEIGHTS = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
READER = "f83b1c0b9ada76c2126f366294785222e121703e02977c66bf530b1fb731b1e5"
ARCHIVE = "ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c"
READER_REV = "fd909227c2013cc58be1ced519b03ea1328199ea"
PUBLIC = {"agentview_image", "robot0_eef_pos", "robot0_gripper_qpos"}


def serve(fd):
    import cv2
    import jax
    import eval_reader

    if (jax.default_backend() != "cpu" or os.environ["CUDA_VISIBLE_DEVICES"] != ""
            or sha256(Path(eval_reader.__file__).read_bytes()).hexdigest() != READER):
        raise ValueError("Reader interpreter, backend or source differs")
    ckpt = Path(os.environ["SYMMETA_CHECKPOINT"])
    archive = Path(os.environ["SYMMETA_SOURCE"])
    rev = subprocess.check_output(["git", "-C", archive, "rev-parse", "HEAD"], text=True).strip()
    if rev != ARCHIVE or sha256(ckpt.read_bytes()).hexdigest() != WEIGHTS:
        raise ValueError("Reader archive or weights differ")
    cv2.setNumThreads(1)
    model = eval_reader.load_v2_checkpoint(ckpt)
    states = []
    with Connection(fd) as pipe:
        pipe.send(dict(archive=rev, reader_revision=READER_REV, reader_sha256=READER,
                       weights_sha256=WEIGHTS, backend=jax.default_backend(), jax=jax.__version__,
                       numpy=np.__version__, opencv=cv2.__version__, python=sys.version.split()[0],
                       update_limit=eval_reader.UPDATE_LIMIT))
        while True:
            cmd, arg = pipe.recv()
            if cmd == "close":
                return
            if cmd == "begin":
                states = [eval_reader.OnlineReader(model) for _ in range(arg)]
                pipe.send(None)
                continue
            i, before, after, issued = arg
            if cmd != "observe" or set(before) != PUBLIC or set(after) != PUBLIC or issued.shape != (5, 7):
                raise ValueError("Unexpected reader input")
            start = time.perf_counter()
            update = states[i].observe(before, after, issued)
            update["compute_s"] = time.perf_counter() - start
            for tag, snap in (("before", before), ("after", after)):
                upright = eval_reader.downsample_agentview(snap["agentview_image"])
                update[tag + "_64_sha256"] = sha256(upright.tobytes()).hexdigest()
                update[tag + "_eef"] = snap["robot0_eef_pos"].tolist()
                update[tag + "_jaw_qpos"] = snap["robot0_gripper_qpos"].tolist()
            pipe.send(update)


if __name__ == "__main__":
    serve(int(sys.argv[1]))
