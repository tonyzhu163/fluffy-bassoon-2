"""R2 layout admission and rendered subprocess, with a training-only teacher command."""

from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
NAME = "libero-paligemma-dualarm-20261120-r2"
PIN = "fc61bf2afb48d4639664db93dae64ae9cb04e2df"
CHECKOUT = Path(os.environ.get("R2_CHECKOUT", f"/mowenjie/fluffy-bassoon-2/setups/{NAME}/source-fc61bf2a"))
sys.path[1:1] = [str(CHECKOUT / "outputs" / NAME),
                 str(CHECKOUT / "outputs/libero-paligemma-crossarm-20261120-r1"), str(CHECKOUT / "src")]

from large_panel_protocol import (ARMS, GEOMETRY_FIELDS, bind, implementation,
                                 mobile_joints, publish, read, require, sha, training)
from large_panel_sim import Session, physical_reset

PANEL_SHA = "663077ec7aad3621173d725d2cf1550e3859a7ca46a0de50727e25aec8c6bfc0"
FIT_SHA = "307e81088eb5b67506e8b30b3348f564c1c142f7e293e6db34f3149776929e80"
CHECKPOINT = "77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f"
REPORT_SHA = "7e2122efbd976638680c8215c3830eeb974179c5944f5185b38771988a3d464a"
DEV = range(740000, 740050)
TRAIN = range(360000, 370000)
N_TRAIN = 64
FORK = 60


def dev_panel(args):
    path = args.panel / "dev/manifest.json"
    require(sha(path) == PANEL_SHA and sha(args.fit / "manifest.json") == FIT_SHA,
            "Expected original large-panel-v3 and sft02/fit")
    data = read(path)
    require(data["revision"] == PIN and implementation() == data["implementation"] and
            [r["layout_seed"] for r in data["layouts"]] == list(DEV), "Wrong evaluator or source layouts")
    return data


def prepare(args):
    import generated_layouts as g
    from expert import session
    from generated_split import validate_origin

    data = dev_panel(args)
    _, fit_seeds, fit_hashes = training(args.fit)
    base = args.out / "layouts"
    # Compare identity/geometry fields only. Future uninstantiated reservations are
    # excluded by the training seed range. Released geometries include all 50 rows.
    paths = sorted({args.panel / "dev/manifest.json", *args.panel.parent.glob("large-panel*/dev/manifest.json"),
                    *args.panel.parent.glob("large-panel*/test/manifest.json")})
    seen = {k: fit_hashes | {r[k] for r in data["released"] if k in r} for k in GEOMETRY_FIELDS}
    excluded_seeds = set(fit_seeds)
    sources = {}
    for path in paths:
        panel = read(path)
        sources[str(path)] = sha(path)
        for r in panel["layouts"]:
            excluded_seeds.add(r["layout_seed"])
            for k in GEOMETRY_FIELDS:
                value = mobile_joints(r["sampled_scene"]) if k == "mobile_joint_xy_hash" else r[k]
                seen[k].add(value)
    public = g.released_fingerprints()
    rows, rejections = [], []
    with closing(g.make_sampler()) as sampler:
        for seed in TRAIN:
            try:
                if seed in excluded_seeds:
                    raise g.LayoutRejected("Seed already used")
                state, origin = g.sample_layout(sampler, seed, public)
                validate_origin(origin)
                row = dict(origin, layout_id=origin["source_index"], state_file=f"rows/{seed}.npy",
                           mobile_joint_xy_hash=mobile_joints(origin["sampled_scene"]), preflight={})
                for k in GEOMETRY_FIELDS:
                    if row[k] in seen[k]:
                        raise g.LayoutRejected(f"Duplicate {k}: {row[k]}")
                for arm in ARMS:
                    with session(arm, False) as env:
                        _, info, poses = physical_reset(env, arm, False, state, row)
                    row["preflight"][arm] = info["admission"]
                    if arm == "Panda":
                        row["panda_settled_poses"] = poses
                    else:
                        row["source_pair_errors"] = info["paired_scene_errors"]
            except g.LayoutRejected as exc:
                rejections.append(dict(seed=seed, reason=str(exc)))
                continue
            except ValueError as exc:
                # The R2 ArmEnv distinguishes normal geometric admission failures
                # from calibration, runtime and provenance errors.
                if not str(exc).startswith(("Released grip-site pose unreachable", "Settling moved the paired task scene",
                                            "Task already successful", "Post-settle scene differs from paired Panda")):
                    raise
                rejections.append(dict(seed=seed, reason=str(exc)))
                continue
            path = base / row["state_file"]
            with publish(path) as tmp, tmp.open("wb") as stream:
                np.save(stream, state, allow_pickle=False)
            row["file_sha256"] = sha(path)
            for k in GEOMETRY_FIELDS:
                seen[k].add(row[k])
            rows.append(row)
            print(f"prepare {len(rows)}/{N_TRAIN} seed={seed}", flush=True)
            if len(rows) == N_TRAIN:
                break
    require(len(rows) == N_TRAIN, "Exhausted training reservation")
    bind(base / "manifest.json", dict(split="train", reservation=[TRAIN.start, TRAIN.stop - 1],
         motor_fit_sha256=FIT_SHA, dev_manifest_sha256=PANEL_SHA, denied_manifests=sources,
         motor_seeds=sorted(fit_seeds), geometry_fields=list(GEOMETRY_FIELDS),
         released_bank_sha256=data["released_bank_sha256"], rejected=rejections, layouts=rows))
    bind(base / "manifest.sha256.json", sha(base / "manifest.json"))


def train_rows(out):
    path = out / "layouts/manifest.json"
    require(sha(path) == read(path.with_suffix(".sha256.json")), "Changed training layout manifest")
    data = read(path)
    rows = data["layouts"]
    seeds = [r["layout_seed"] for r in rows]
    require(data["split"] == "train" and data["motor_fit_sha256"] == FIT_SHA and
            data["dev_manifest_sha256"] == PANEL_SHA and len(rows) == len(set(seeds)) == N_TRAIN and
            all(s in TRAIN for s in seeds) and all(r["split"] == "train" and
            r["layout_id"] == 1_000_000 + r["layout_seed"] and set(r["preflight"]) == set(ARMS) for r in rows),
            "Wrong training cohort")
    return rows


def serve(fd, base, arm, split):
    from multiprocessing.connection import Connection
    from expert import session
    from sim import _pack, _public

    with Connection(fd) as channel:
        try:
            require(arm in ARMS and split in ("train", "dev"), "Unexpected source worker")
            manifest = read(base / "manifest.json")
            rows = {r["layout_seed"]: r for r in manifest["layouts"]}
            obs = teacher = mode = None
            with session(arm, True) as env:
                while True:
                    op, value = channel.recv()
                    if op == "close":
                        return
                    if op == "reset":
                        seed, mode = value
                        allowed = ("off", "always", "gated", "teacher") if split == "train" else ("off", "always", "gated")
                        require(mode in allowed, "Branch outside split")
                        row = rows[seed]
                        path = base / row["state_file"]
                        require(row["split"] == split and sha(path) == row["file_sha256"], "Donor changed")
                        teacher = None
                        obs, info, _ = physical_reset(env, arm, True, np.load(path, allow_pickle=False), row)
                        result = _pack(obs), info
                    elif op == "advance":
                        native = env.native.arm
                        count = len(value)
                        require(obs is not None and 1 <= count <= 5, "Invalid native action chunk")
                        takeover = mode == "teacher" and native._step >= FORK
                        if takeover and teacher is None:
                            require(split == "train" and native._step == FORK, "Bad teacher fork")
                            from teacher import make_teacher
                            teacher = make_teacher(native, obs)
                        trace = []
                        for requested in value:
                            action = teacher(obs) if takeover else requested
                            p = native.step(action)
                            obs = _public(p)
                            issued = p["issued_action"]
                            require(np.array_equal(issued, np.asarray(action, np.float32)),
                                    "Native action differs from requested control")
                            trace.append((issued.tolist(), p["reward"], p["terminated"], p["truncated"], p["success"],
                                          p["worker_step_index"]))
                            if p["terminated"] or p["truncated"]:
                                break
                        result = _pack(obs), trace
                    else:
                        raise ValueError(f"Unknown native operation {op}")
                    channel.send((True, result))
        except (EOFError, BrokenPipeError):
            pass
        except Exception:
            channel.send((False, traceback.format_exc()))


class RenderedSession(Session):
    """Use R2's Pipe protocol and lifetime management with the optional training teacher."""

    def __init__(self, base, arm, split):
        from multiprocessing import Pipe
        self.closed = False
        self.pipe, peer = Pipe()
        runtime = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            self.worker = subprocess.Popen([os.environ["CROSSARM_SIM_PYTHON"], "-Bu", str(Path(__file__).resolve()),
                str(peer.fileno()), str(base), arm, split], pass_fds=(peer.fileno(),), env=runtime)
        except BaseException:
            self.pipe.close()
            raise
        finally:
            peer.close()

    def start(self, seed, mode):
        from sim import _unpack
        packet, info = self._call("reset", (seed, mode))
        return _unpack(packet), info


if __name__ == "__main__":
    serve(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
