"""Recover one missing held segment by verifying the entire saved native prefix."""
import json
import os
from pathlib import Path
import sys

import numpy as np

from collect import ArmEnv, Recorder, pack, rewind, save
from design import checked
from score import main as score, read


def finish(old, out):
    assert os.sched_getaffinity(0)=={39} and os.environ["CUDA_VISIBLE_DEVICES"]==""
    target = "IIWA-libero_spatial-0"
    for arm in ("UR5e", "IIWA"):
        name = f"runtime-{arm}.json"
        (out/name).symlink_to(Path("..")/old.name/name)
    for source in sorted(old.glob("*-libero_*")):
        dest = out/source.name
        dest.mkdir()
        for item in source.iterdir():
            if source.name == target and item.name in ("case.json", "general-held.npz"):
                continue
            (dest/item.name).symlink_to(Path("../..")/old.name/source.name/item.name)
    dest = out/target
    base = old/target
    meta = json.loads((base/"case.json").read_text())
    assert "complete" not in meta and all((base/f"{f}-held.npz").exists() for f in ("native", "family", "swap"))
    raw = json.loads(checked(meta["query"]["path"], meta["query"]["sha256"]))
    cmd = np.concatenate([c["cmd"] for c in raw["chunks"]]).astype(np.float32)
    np.testing.assert_array_equal(cmd, read(base/"archive-controls.npz")["query"])
    previous = read(base/"native-prefix.npz")
    n = meta["midpoint_start"]
    assert len(previous["issued"])==n
    with ArmEnv("IIWA",meta["suite"],meta["task"]) as env:
        # The original collector constructed its layout-14 simulation as the second reset
        # on this object. Sensor and dynamics identity are tested against the saved prefix.
        first = env.reset(14,raw["case"]["seed"])
        ref = meta["held_reset"]
        current = first["environment_reproducibility"]
        for k in ("released_row_fingerprint","bddl_sha256","scene_fingerprint","perturbation_plan_fingerprint"):
            assert current[k] == ref[k]
        rec = Recorder(env)
        orig = pack(env,first["observation"])
        obs, replay = rec.run(dest/"reconstructed-prefix.npz",rewind(env,orig),cmd[:n],np.eye(2))
        # Validate a full physical, sensor and control trajectory before issuing the missing branch.
        audit=dict(n=n, cpu02_reset=ref["initial_state_fingerprint"], cpu03_reset=current["initial_state_fingerprint"])
        for k in ("issued","physical","goal_m","q","dq","fk","external","wrist","reward","terminal","jaw","quat","time"):
            error=float(np.max(np.abs(replay[k].astype(float)-previous[k].astype(float))))
            audit[k]=error
        save(out/"prefix-readback.json",audit)
        for k in ("issued","physical","external","wrist","reward","terminal"):
            np.testing.assert_array_equal(replay[k],previous[k])
        for k in ("goal_m","q","dq","fk","jaw","quat","time"):
            np.testing.assert_allclose(replay[k],previous[k],atol=1e-7,rtol=0)
        mid=pack(env,obs)
        matrix=np.array(meta["maps"]["general"]["A"])
        _, held=rec.run(dest/"general-held.npz",rewind(env,mid),cmd[n:n+20],matrix)
        assert len(held["issued"])==20
        for f in ("native","family","swap"):
            prev=read(base/f"{f}-held.npz")
            for k in ("fk","q","dq","external","wrist","time"):
                np.testing.assert_array_equal(held[k][0],prev[k][0])
    meta["complete"]=True
    meta["recovered_held_family"]="general"
    save(dest/"case.json",meta)
    score(out)


if __name__=="__main__":
    finish(Path(sys.argv[1]),Path(sys.argv[2]))
