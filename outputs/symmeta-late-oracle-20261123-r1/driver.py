import argparse
import os
from pathlib import Path
import socket

from paired_inputs import module
from paired_records import atomic, checked, encode, parse, require
from panel import ARMS, CHECKPOINT, CHECKPOINT_SHA, HELPERS, prepare
from records import panel, summary, validate

RETRY = HELPERS / "outputs/symmeta-perceiver-retry-20261123-r1"


def devices(out, count):
    from fluffy_bassoon.baige_gpu import cuda_pci, map_devices, pci, validate_container
    reference = module("reference", HELPERS/"outputs/reground-rotation-reference-20261123-r1/reference.py")
    probe = parse((out/"gpu.json").read_bytes())
    rows = probe["devices"]
    require(probe["status"] == "passed" and probe["allocation_type"] == "baige_container" and
            probe["purpose"] == "evaluation" and probe["hostname"] == socket.gethostname()
            and len(rows) == count, "GPU probe")
    map_devices(rows, [r["inventory"] for r in rows], validate_container(os.environ,socket.gethostname()))
    for i, r in enumerate(rows):
        name = r["cuda_name"].upper()
        require(r["allocation_ordinal"] == i and pci(r["pci_bus_id"]) == cuda_pci(i) and
                (name in {"NVIDIA RPBZZZ6","NVIDIA GEFORCE RTX 4090"} or "RTX PRO 6000" in name), "GPU mapping/type")
        r["egl_device"] = reference.egl_device(i)
    require(len({r["egl_device"] for r in rows}) == count and len({r["cuda_name"] for r in rows}) == 1,
            "Nonexclusive render devices or mixed accelerator types")
    atomic(out/"devices.json", encode(rows))


def lane(out, lane_id, count):
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    from rollout import ev, run_arm
    setup = module("paired_live", RETRY/"live.py")
    data = panel(out)
    device = parse((out/"devices.json").read_bytes())[lane_id]
    require(device["visibility_token"] == os.environ["CUDA_VISIBLE_DEVICES"] and
            pci(device["pci_bus_id"]) == cuda_pci(0) and str(device["egl_device"]) == os.environ["MUJOCO_EGL_DEVICE_ID"],
            "Wrong lane binding")
    checked(CHECKPOINT, CHECKPOINT_SHA)
    weights = ev.load_v2_checkpoint(CHECKPOINT)[:2]
    policy = ev.load_frozen_policy("pi05")
    for shard in range(lane_id,8,count):
        ref = data["conditions"][shard]
        group = data["rows"][shard*12:(shard+1)*12]
        with setup.simulator(ev) as sim:
            setup.warmup(sim,group[0],ref["reset"],None)
            resets = {}
            for r in group:
                arms = {name:run_arm(sim,policy,r,name,ref,resets,weights) for name in ARMS}
                validate(r, arms, ref)
                atomic(out/"cases"/f"{r['index']:02d}.json", encode(dict(arms=arms, row=r, lane=lane_id,
                       panel_sha256=(out/"panel.sha256").read_text().strip())))
                print(r["index"], {k:v["success"] for k,v in arms.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare","devices","lane","merge","cpu"))
    p.add_argument("out", type=Path)
    p.add_argument("--lane", type=int)
    p.add_argument("--cards", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.out)
    elif a.mode == "devices":
        devices(a.out,a.cards)
    elif a.mode == "lane":
        lane(a.out,a.lane,a.cards)
    elif a.mode == "cpu":
        from qualify import qualify
        qualify(a.out)
    else:
        summary(a.out)
