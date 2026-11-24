"""Execute the released arms using the original body's native evaluator and primitive wire capture."""
import argparse
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch


from release import HERE, KIN, UR, ORIGINAL, PLAN, RECEIPT, SEAL, checked, digest, read

TRACES = {
    "UR5e": ("5000e5ee526d0ee3b34162440327cfc3e4831482","ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c"),
    "Kinova3": ("d2b307bc4ca8c0cf723b3aaba31755d14a2f8303","465f33dc657ceab341dbceaf6b98355b94810059"),
}
TRANSPORT = "28310343bab088490f4e84110ce13861ccdeb3eda1bb4dd7c4bc26df84b8428c"
MOTOR_READY = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"


def git_pin(path):
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()


def run(body, out, release, sha, pin):
    import jax
    import ur5e_motor
    import ur5e_sim_worker
    packet = read(release,sha)
    assert packet["registration_sha256"] == digest((HERE/"PLAN.md").read_bytes()) == PLAN
    assert packet["body"]==body and packet["source_receipt_sha256"]==RECEIPT[body] and packet["seal_sha256"]==SEAL
    assert packet["old_panel_sha256"]==ORIGINAL[body][2] and len(packet["rows"])==64 and packet["seed"]==41
    original = Path(os.environ["REALNATIVE_ORIGINAL"])
    deps = Path(os.environ["PAIRED_DEPS"])
    assert git_pin(original)==TRACES[body][0] and git_pin(deps)==TRACES[body][1] and git_pin(HERE)==pin
    native_dir = original/"outputs"/UR
    sys.path[:0] = [str(native_dir),str(original/"outputs"/KIN)]
    from native import episode
    from protocol import atomic, encode
    from paired_inputs import native
    if body == "Kinova3":
        from native_panel import audited_episode

    ev = native()
    assert ur5e_motor.OPENPI_CKPT == Path("/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1")
    checked(ur5e_motor.OPENPI_CKPT/"transport-manifest.json",TRANSPORT)
    ready = read(ur5e_motor.OPENPI_CKPT/"READY.json",MOTOR_READY)
    assert ready["status"]=="verified" and ready["checkpoint_path"]==str(ur5e_motor.OPENPI_CKPT)
    gpu = json.loads((out/"gpu.json").read_bytes())
    device, = gpu["devices"]
    assert gpu["hostname"]==socket.gethostname() and gpu["purpose"]=="evaluation" and gpu["status"]=="passed"
    assert gpu["allocation_type"]=="baige_container" and "RTX 4090" in device["cuda_name"].upper()
    assert len(jax.devices())==1 and jax.devices()[0].platform=="gpu"
    runtime = dict(body=body,host=gpu["hostname"],pin=pin,original=TRACES[body][0],deps=TRACES[body][1],
                   motor_manifest_sha256=TRANSPORT,motor_ready_sha256=MOTOR_READY,release_sha256=sha,
                   egl_device=os.environ["MUJOCO_EGL_DEVICE_ID"],cuda_uuid=device["uuid"],
                   worker_sha256=digest((native_dir/"audited_worker.py").read_bytes()))
    atomic(out/"runtime.json",encode(runtime))
    motor = ev.load_frozen_policy("pi05")
    assert (motor._model.action_horizon,motor._model.action_dim)==(10,32)
    with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}), patch.object(ur5e_sim_worker,"__file__",str(native_dir/"audited_worker.py")):
        sim = ev.UR5eSimulator()
    expected = ["A_real","B_real"] + (["frozen_warp"] if body=="UR5e" else [])
    with sim:
        previous = None
        for row in packet["rows"]:
            assert set(row["corrections"])==set(expected)
            if row["shard"] != previous:
                sim.call("create",dict(row["query"],embodiment=row["embodiment"]))
                previous = row["shard"]
            order = expected.copy()
            if (row["shard"]+row["slot"])%2:
                order[0],order[1] = order[1],order[0]
            arms, resets = {}, {}
            for name in order:
                if body=="UR5e":
                    ep = episode(ev,sim,motor,row,name,resets,False)
                else:
                    ep = audited_episode(ev,sim,motor,row,name,resets)
                arms[name] = ep
                first = arms[order[0]]
                assert ep["native"]["reset"]==first["native"]["reset"] and ep["env"]==first["env"] and ep["body"]==first["body"]
                assert ep["chunks"][0]["raw"]==first["chunks"][0]["raw"] and ep["body"]["cameras"]==["agentview","robot0_eye_in_hand"]
                if row is packet["rows"][0] and name==order[0]:
                    atomic(out/"first-episode.json",encode(dict(body=body,index=row["index"],model=name,qualified=True,
                          opening=ep["native"]["opening_proposal_sha256"],runtime=runtime)))
            record = dict(body=body,release_sha256=sha,index=row["index"],execution_order=order,arms=arms)
            path = out/"pairs"/f"{row['index']:02d}.json.gz"
            atomic(path,gzip.compress(encode(record),compresslevel=4,mtime=0))
            print(time.strftime("%FT%T"),"qualified complete case",row["index"],flush=True)
    assert len(list((out/"pairs").glob("*.json.gz")))==64


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("body",choices=TRACES)
    p.add_argument("out",type=Path)
    p.add_argument("release",type=Path)
    p.add_argument("sha")
    p.add_argument("pin")
    run(**vars(p.parse_args()))
