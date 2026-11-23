"""Native tuned Perceiver / paper SymMeta evaluation for a single panel process."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import socket
from types import SimpleNamespace

from paired_devices import hardware
from paired_inputs import native, prepare
from paired_records import (PANELS, READERS, atomic, checked, contract, encode, pair_path, parse,
                            read_pair, receipt, require, write_pair)


@contextmanager
def simulator(ev, cpu=False):
    # Robosuite 1.4 compares the EGL index to the CUDA mask as a string. The
    # simulator only renders; its EGL index was resolved with the full allocation.
    # Its subprocess needs no CUDA compute devices. The motor retains its lane token.
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sim = ev.UR5eSimulator(allow_mesa=cpu)
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
    with sim:
        yield sim


def warmup(sim, row, ref, identity, cpu=False):
    case = dict(row["query"], embodiment=row["embodiment"])
    spec, first = sim.call("create", case)
    model = first["model_identity"]
    require(spec == dict(case, max_steps=ref["horizon"]) and first["language_instruction"] == ref["prompt"]
            and all(model[k] == ref[k] for k in ("arm", "joint_count", "model_class"))
            and model["gripper"] == "PandaGripper" and model["controller"] == "OSC_POSE"
            and (identity is None or model == identity) and first["worker_step_index"] == 0
            and first["action_schema"] == ref["action_schema"] and
            (cpu or first["renderer_vendor"] == "NVIDIA Corporation"), f"Wrong native setup {case}")
    return first


def episode(ev, sim, policy, row, name, ref, resets):
    c = row[name]
    # The original Goal6 runner predates the canonicalize_quat parameter.
    kwargs = {"canonicalize_quat": True} if row["panel"] == "gpu08pi0fix" else {}
    return ev.run_paired_query(
        ev.run_episode, sim, policy, row["query"], SimpleNamespace(name=row["embodiment"]), arm_name=name,
        resets=resets, key=(row["suite"], row["task_id"], row["query"]["init_state_id"]),
        support_physics=row["support_physics"], expected_prompt=ref["prompt"], create_attempts=3,
        yaw_hat=c["yaw_deg"], scale_hat=c["xy_scale"], grip_hat=c["gripper_sign"], **kwargs)


def reusable(sources, rows, ref, sha, execution, lane):
    candidates = {}
    disputed = set()
    for src in sources:
        old_sha, _ = contract(src)
        if old_sha != sha:
            print(f"Skipping incompatible contract {src}", flush=True)
            continue
        receipts = {}
        for r in rows:
            if not pair_path(src, r).is_file():
                continue
            try:
                raw, b = read_pair(src, r, ref, sha, receipts)
            except (ValueError, KeyError, IndexError, TypeError, FileNotFoundError) as error:
                print(f"Ignoring invalid prior record {pair_path(src, r)}: {error}", flush=True)
                continue
            origin = receipts[b["execution_sha256"]]
            if hardware(origin, b["lane"]) != hardware(execution, lane):
                continue
            k = r["index"], r["seed"]
            if k in disputed:
                continue
            if k in candidates and candidates[k][1] != raw:
                # Independent origins need a fresh pair; source order must not select an outcome.
                del candidates[k]
                disputed.add(k)
                print(f"Repeating independently recorded {r['panel']} {k}", flush=True)
                continue
            candidates[k] = (src, raw, b)
    return candidates


def adopt(out, rows, candidates):
    origins = []
    for r in rows:
        k = r["index"], r["seed"]
        if k not in candidates:
            continue
        src, raw, body = candidates[k]
        sha = body["execution_sha256"]
        receipt_path = Path("receipts")/(sha+".json")
        atomic(out/receipt_path, checked(src/receipt_path, sha))
        atomic(pair_path(out, r), raw)
        origins.append(dict(index=k[0], seed=k[1], source=str(src), execution_sha256=sha))
    return origins


def run_shard(out, panel, shard, lane, sources):
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    ev = native()
    sha, c = contract(out)
    data = parse(checked(out/panel/"panel.json", c["panels"][panel]))
    execution_sha = (out/"execution.sha256").read_text().strip()
    execution = receipt(out, execution_sha)
    device = execution["devices"][lane]
    require(execution["contract_sha256"] == sha and execution["hostname"] == socket.gethostname()
            and device["allocation_ordinal"] == lane and device["visibility_token"] == os.environ["CUDA_VISIBLE_DEVICES"]
            and cuda_pci(0) == pci(device["pci_bus_id"]) and str(device["egl_device"]) == os.environ["MUJOCO_EGL_DEVICE_ID"],
            "Lane CUDA/EGL binding")
    ref = data["refs"][shard]
    rows = [r for r in data["rows"] if r["shard"] == shard]
    require(len(rows) == 36 and all(not pair_path(out, r).exists() for r in rows), "Fresh task destination required")
    candidates = reusable(sources, rows, ref, sha, execution, lane)
    origins = adopt(out, rows, candidates)
    atomic(out/panel/f"shard-{shard}-reuse.json", encode(origins))
    missing = [r for r in rows if (r["index"], r["seed"]) not in candidates]
    print(f"{panel} task {shard}: adopted {len(origins)} validated pairs", flush=True)
    if not missing:
        return
    policy = ev.load_frozen_policy(data["runtime"]["policy"])
    with simulator(ev) as sim:
        warmup(sim, rows[0], ref, data["model_identities"][shard])
        for r in missing:
            # Scope the native pre-action reset reference to this experimental pair.
            resets = {}
            arms = {a: episode(ev, sim, policy, r, a, ref, resets) for a in READERS}
            write_pair(out, r, arms, ref, sha, execution_sha, lane)
            print(f"{panel} task {shard} case {r['index']} seed {r['seed']} "
                  f"tuned={arms['tuned']['success']} symmeta={arms['symmeta']['success']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "shard"))
    p.add_argument("panel", choices=PANELS)
    p.add_argument("out", type=Path)
    p.add_argument("--predictions-sha")
    p.add_argument("--cpu-fixture", action="store_true")
    p.add_argument("--shard", type=int, choices=range(8))
    p.add_argument("--lane", type=int)
    p.add_argument("--reuse", nargs="*", type=Path, default=[])
    args = p.parse_args()
    if args.mode == "prepare":
        require(not args.cpu_fixture or (os.environ.get("JAX_PLATFORMS") == "cpu" and
                os.environ.get("CUDA_VISIBLE_DEVICES") == ""), "CPU-only fixture")
        prepare(args.panel, args.out, args.predictions_sha, allow_fixture=args.cpu_fixture)
    else:
        run_shard(args.out, args.panel, args.shard, args.lane, args.reuse)
