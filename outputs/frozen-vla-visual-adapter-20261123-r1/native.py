"""Three matched Kinova3 evaluation conditions with the stock frozen OpenPI episode loop."""
import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pipe
import subprocess
import sys

import numpy as np
from ur5e_sim_worker import UR5eSimulator, SIM_PYTHON
from panel import TASKS, case

HERE = Path(__file__).resolve().parent
MODES = ("raw", "uniform", "linkwise")


class PaintedSimulator(UR5eSimulator):
    def __init__(self, mode, palette, *, allow_mesa=False):
        self._closed = False
        self.pipe, peer = Pipe()
        try:
            self.worker = subprocess.Popen([str(SIM_PYTHON), "-Bu", str(HERE / "native.py"),
                                            "serve", str(peer.fileno()), mode, str(palette)],
                                           pass_fds=(peer.fileno(),),
                                           env={**os.environ, "FLUFFY_ALLOW_MESA": str(int(allow_mesa))})
        finally:
            peer.close()


def check_pair(native, other):
    for key in ("physics", "scene_fingerprint", "model_class", "perturbation_plan", "noise_sha256",
                "noise_seed", "horizon", "prompt", "action_schema"):
        if native["reset"][key] != other["reset"][key]:
            raise ValueError(f"Unpaired {key}")
    for key, digest in native["reset"]["public"].items():
        if not key.endswith("_image") and digest != other["reset"]["public"][key]:
            raise ValueError(f"Unpaired state {key}")


def run(out, palette):
    from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
    from ur5e_motor import load_frozen_pi05, run_episode, write_json
    policy = load_frozen_pi05()
    results = []
    # Sequential worker lifetimes reduce simultaneous EGL contexts and host RAM.
    for mode in MODES:
        with PaintedSimulator(mode, palette) as sim:
            for suite, task in TASKS:
                for init in (14, 15):
                    c = case("Kinova3", suite, task, init)
                    row = run_episode(sim, policy, c, get_embodiment_spec("fam_nom"))
                    if not (row["terminated"] or row["truncated"]):
                        raise ValueError("Incomplete episode")
                    if mode != "raw":
                        raw = next(r for r in results if r["mode"]=="raw" and r["case"]==c)
                        check_pair(raw, row)
                    record = dict(row, mode=mode, case=c)
                    write_json(out / f"{mode}-{suite}-{task}-{init}.json", record)
                    results.append(record)
                    print(f"{mode} {suite}/{task} {init}: success={row['success']} controls={row['controls']}", flush=True)
    by_method = {m: np.array([int(r["success"]) for r in results if r["mode"]==m]).reshape(len(TASKS),2) for m in MODES}
    pairs = {}
    for a,b in (("linkwise","uniform"),("linkwise","raw"),("uniform","raw")):
        diff = by_method[a] - by_method[b]
        # Every ordered nonparametric task cluster bootstrap resample (3**3).
        from itertools import product
        reps = [diff[list(sample)].mean() for sample in product(range(len(TASKS)), repeat=len(TASKS))]
        pairs[f"{a}_minus_{b}"] = dict(gain=int((diff>0).sum()), loss=int((diff<0).sum()),
                                     delta=float(diff.mean()), task_bootstrap_95=np.quantile(reps,[.025,.975]).tolist())
    write_json(out/"result.json", dict(checkpoint="/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1",
               matrix={m:x.tolist() for m,x in by_method.items()}, tasks=TASKS, initial_states=[14,15],
               descriptive_task_cluster_intervals=pairs, task_selection="historical failures and one successful task"))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        from adapter import appearance
        from ur5e_sim_worker import serve
        with appearance(sys.argv[3], json.loads(Path(sys.argv[4]).read_text())):
            serve(int(sys.argv[2]))
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("output", type=Path)
        parser.add_argument("palette", type=Path)
        args = parser.parse_args()
        run(args.output, args.palette)
