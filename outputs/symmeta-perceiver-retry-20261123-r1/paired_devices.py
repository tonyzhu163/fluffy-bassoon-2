"""Resolve inference and rendering against the single parent-container probe."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import socket
import subprocess

from paired_inputs import module
from paired_records import HERE, atomic, contract, digest, encode, parse, require


def hardware(receipt, lane):
    row = receipt["devices"][lane]
    return receipt["software"], row["cuda_name"], row["cuda_total_bytes"]


def capture(out, count, revision):
    from fluffy_bassoon.baige_gpu import cuda_pci, map_devices, pci, validate_container

    ref = module("reground_reference", HERE.parent/"reground-rotation-reference-20261123-r1/reference.py")
    raw = (out/"gpu.json").read_bytes()
    probe = parse(raw)
    host = socket.gethostname()
    devices = probe["devices"]
    require(probe["status"] == "passed" and probe["allocation_type"] == "baige_container"
            and probe["purpose"] == "evaluation" and probe["hostname"] == host and len(devices) == count, "GPU probe")
    map_devices(devices, [d["inventory"] for d in devices], validate_container(os.environ, host))
    for lane, d in enumerate(devices):
        name = d["cuda_name"].upper()
        require(d["allocation_ordinal"] == lane and pci(d["pci_bus_id"]) == cuda_pci(lane)
                and (name in {"NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090"} or "RTX PRO 6000" in name),
                f"Unsupported or incorrectly mapped lane {lane}")
        d["egl_device"] = ref.egl_device(lane)
    require(len({d["egl_device"] for d in devices}) == count and len({d["cuda_name"] for d in devices}) == 1,
            "GPU/EGL allocation mismatch")
    packages = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "flax",
                "orbax-checkpoint", "openpi-client")
    versions = {k: metadata.version(k) for k in packages}
    sim_python = Path(os.environ["LIBERO_PRO_ROOT"]).parent/"env/bin/python"
    sim_versions = subprocess.check_output([sim_python, "-c",
        "import json; from importlib.metadata import version; "
        "print(json.dumps({k:version(k) for k in ('mujoco','robosuite','numpy','PyOpenGL')}))"], text=True)
    versions["simulator"] = parse(sim_versions)
    versions["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    versions["jax_matmul_precision"] = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
    versions["driver"] = sorted(set(driver.splitlines()))
    sha, _ = contract(out)
    raw = encode(dict(schema=1, hostname=host, deployment=revision, attempt=out.name, contract_sha256=sha,
                      gpu_probe_sha256=digest(raw), software=versions, devices=devices))
    atomic(out/"receipts"/(digest(raw)+".json"), raw)
    atomic(out/"execution.sha256", (digest(raw)+"\n").encode())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("count", type=int)
    p.add_argument("revision")
    args = p.parse_args()
    capture(args.out, args.count, args.revision)
