"""CPU validation of the production lock and sampler on a historical Panda opening, with zero fitted updates."""
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

from panel import require, save
from ur5e_motor import NOISE_SEED_OFFSET, array_digest, public_digest, to_openpi_observation
from ur5e_sim_worker import UR5eSimulator


def main(out):
    start = time.monotonic()
    require(not os.environ["CUDA_VISIBLE_DEVICES"] and os.environ["JAX_PLATFORMS"] == "cpu"
            and len(os.sched_getaffinity(0)) <= 2, "CPU fixture allocation")
    case = dict(arm="Panda", suite="libero_goal", task_id=3, variant="BASE", init_state_id=14, seed=41810314)
    # Historical unscored opening. No action is sent to the simulator.
    with UR5eSimulator(allow_mesa=True) as sim:
        _, first = sim.call("create", dict(case, embodiment="nom"))
    item = to_openpi_observation(first["observation"], first["language_instruction"], canonicalize_quat=False)
    noise = np.random.default_rng(case["seed"]+NOISE_SEED_OFFSET).standard_normal((10,32)).astype(np.float32)

    from cotrain_policy import SCHEMA, LockedPolicy, base_identity, file_sha, kernel_receipt, load_stock
    from flow_head import CONDITIONS, initial_params
    stock = load_stock()
    projection = stock._model.action_out_proj
    print("Stock restored; building temporary zero-update serialization fixture", flush=True)
    with TemporaryDirectory(dir=out) as temp:
        bundle = Path(temp)
        params = {c: initial_params(projection.in_features, c) for c in CONDITIONS}
        np.savez(bundle/"heads.npz", **{f"{c}/{k}": np.asarray(v) for c,p in params.items() for k,v in p.items()})
        save(bundle/"fit.json", dict(schema=SCHEMA, base=base_identity(verify_files=False),
             heads_sha256=file_sha(bundle/"heads.npz"), parameters={c:kernel_receipt(projection,p) for c,p in params.items()}))
        motor = LockedPolicy(stock, bundle)
        before = motor.digests()
        parity = motor.parity(item, noise=noise)
        decoded = {}
        for c in CONDITIONS:
            actions = motor.infer(item, noise=noise, condition=c, history=np.zeros(52,np.float32))["actions"]
            decoded[c] = array_digest(actions)
        require(len(set(decoded.values())) == 1 and parity["max_abs"] == 0, "no-op fixture diverged")
        after = motor.digests()
        require(before == after, "CPU live stock/head/normalization fingerprint drift")
    save(out/"stock-readback.json", dict(kind="Historical opening, original stock and zero-initialized branches",
         pin=subprocess.check_output(["git","-C",str(Path(__file__).parent),"rev-parse","HEAD"],text=True).strip(),
         host=socket.gethostname(), case=case, issued_commands=0, gpu_hours=0,
         cpu_affinity=sorted(os.sched_getaffinity(0)), release_fingerprint=first["environment_reproducibility"]["released_row_fingerprint"],
         public=public_digest(first["observation"]), converted={k:array_digest(v) for k,v in item.items() if k!="prompt"},
         noise_sha256=array_digest(noise), parity=parity, decoded=decoded, frozen_before=before, frozen_after=after,
         wall_s=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
