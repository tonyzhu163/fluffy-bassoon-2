"""Execute immutable, paired, native Goal 5 triplets on one shard."""
import argparse
import os
import signal
from pathlib import Path
import time

import numpy as np

from large_panel_protocol import bind, read, require, sha
from native import PROMPT, Simulator
from source_data import PublicHistory
from ur5e_motor import array_digest, public_digest, to_openpi_observation

MODES = ("stock", "panda", "mixed")


def bank(row):
    return np.random.default_rng(row["noise_seed"]).standard_normal((60, 10, 32)).astype(np.float32)


def begin(sim, row, opening):
    for attempt in range(3):
        raw, info = sim.call("reset", row)
        key = dict(info=info, public=public_digest(raw))
        if opening is None or key == opening:
            return raw, key, attempt + 1
    raise ValueError(f"fixed case {row['index']} failed exact reset before any policy call")


def episode(sim, motor, row, noise, mode, dst, opening, fixture):
    raw, opening_key, tries = begin(sim, row, opening)
    history = PublicHistory()
    calls, diagnostic = [], None
    t = 0
    while t < 300:
        h = history.vector()
        item = to_openpi_observation(raw, PROMPT)
        require(np.array_equal(item["observation/state"][-2:], raw["robot0_gripper_qpos"]), "physical fingers changed")
        z = noise[t // 5]
        if fixture:
            proposal = np.zeros((10, 7), np.float32)
            route = dict(graph="fixture")
        else:
            result, route = motor.infer_with_receipt(item, noise=z, condition=mode, history=h)
            proposal = np.asarray(result["actions"])
            require(route["graph"] in (("stock_native",) if mode == "stock" else
                    ("plain", "stock_native") if t == 0 else ("history",)), "incorrect issued sampler")
            if t == 0:
                stock = motor.stock.infer(item, noise=z)["actions"]
                require(np.array_equal(stock, proposal), f"{mode} opening differs from original policy")
            if t == 5 and mode == "mixed":
                original = np.asarray(motor.stock.infer(item, noise=z)["actions"])
                forced, graph = motor.forced_zero(item, noise=z)
                forced = np.asarray(forced["actions"])
                require(graph["hlo_sha256"] == route["hlo_sha256"] and bool(graph["hlo_sha256"]), "diagnostic graph mismatch")
                diagnostic = dict(t=t, stock=original.tolist(), zero_add=forced.tolist(), learned=proposal.tolist(),
                                  zero_route=graph, stock_vs_zero_max=float(np.max(np.abs(original - forced))),
                                  learned_vs_zero_max=float(np.max(np.abs(proposal - forced))))
        require(proposal.shape == (10, 7) and np.isfinite(proposal).all(), "malformed proposal")
        if t == 0:
            require(not np.any(h), "nonempty opening history")
        call = dict(t=t, observation=public_digest({k:v for k,v in item.items() if k != "prompt"}),
                    noise=array_digest(z), history=h.tolist(), proposed=proposal.tolist(), route=route)
        issued = np.clip(proposal[:5], -1, 1).astype(np.float32)
        following, positions, fingers, done = sim.call("chunk", issued)
        for cmd, xyz, joints in zip(issued[:len(positions)], positions, fingers, strict=True):
            nxt = dict(robot0_eef_pos=xyz, robot0_gripper_qpos=joints)
            history.append(cmd, raw, nxt)
            raw = nxt
        raw = following
        t += len(positions)
        call["issued_count"] = len(positions)
        calls.append(call)
        if done:
            break
    require(done, "native endpoint missing")
    sim.call("save", str(dst / f"{mode}.h5"))
    body = dict(mode=mode, case_id=row["index"], reset=opening_key, reset_attempt=tries,
                noise_bank=array_digest(noise), calls=calls, controls=t, diagnostic=diagnostic,
                tape_sha256=sha(dst / f"{mode}.h5"))
    bind(dst / f"{mode}.json", body)
    return opening_key


def run(args):
    out = args.out
    fixture = args.fixture
    if fixture:
        require(len(os.sched_getaffinity(0)) <= 2, "CPU fixture affinity")
        motor = None
        frozen = dict(fixture=True)
    else:
        from plate_policy import PlatePolicy
        require(sha(args.fit / "fit.json") == args.fit_sha, "fit seal changed")
        require(sha(args.manifest) == args.manifest_sha, "physical manifest changed")
        rows = read(args.manifest)["cases"]
        require(0 <= args.start < args.stop <= len(rows), "shard bounds")
        fit_receipt = read(args.fit / "fit.json")
        data = args.fit.parent.parent / "data/dataset-v1/seal.json"
        require(fit_receipt["code_revision"] == args.fit_pin and sha(data) == fit_receipt["data_seal_sha256"],
                "fit code revision or source collection changed")
        sources = read(data)["layouts"]
        require(len(sources) == 20, "incomplete source layout seal")
        for k, source_key in (("scene_hash", "scene_hash"), ("initial_state_sha256", "generated_row_fingerprint"),
                              ("movable_xy_hash", "movable_xy_hash"), ("layout_seed", "layout_seed")):
            require({r[k] for r in rows}.isdisjoint(r[source_key] for r in sources), f"source overlap at {k}")
        motor = PlatePolicy.load(args.fit)
        frozen = motor.digests()
    bind(out / "before.json", frozen)
    def timed_out(signum, frame):
        raise TimeoutError(f"shard received signal {signum}")

    signal.signal(signal.SIGTERM, timed_out)
    try:
        with Simulator(fixture) as sim:
            owned = [sim.call("fixture")] if fixture else rows[args.start + args.shard:args.stop:4]
            require(bool(owned), "empty shard")
            # Force first-time model registration outside recorded resets, using no control or policy.
            sim.call("reset", owned[0])
            for row in owned:
                index = row["index"]
                require(fixture or index % 4 == args.shard, "manifest shard assignment changed")
                case = out / f"case-{index:05}"
                if args.resume is not None:
                    earlier = args.resume / case.name
                    if (earlier / "complete.json").is_file():
                        require(read(args.resume / "before.json") == read(args.resume / "after.json") == frozen,
                                "recovery weights differ")
                        require((out / "run.txt").read_text().splitlines()[:7] ==
                                (args.resume / "run.txt").read_text().splitlines()[:7], "recovery contract mismatch")
                        seal = read(earlier / "complete.json")
                        require(seal["case_id"] == index and seal["layout_seed"] == row["layout_seed"] and
                                seal["noise_seed"] == row["noise_seed"] and set(seal["receipts"]) == set(MODES),
                                "recovery case mismatch")
                        for m in MODES:
                            require(seal["receipts"][m] == sha(earlier / f"{m}.json") and
                                    read(earlier / f"{m}.json")["tape_sha256"] == sha(earlier / f"{m}.h5"),
                                    "recovery tape mismatch")
                        case.symlink_to(earlier.resolve(), target_is_directory=True)
                        continue
                case.mkdir()
                started = time.monotonic()
                noise = bank(row)
                opening = None
                for mode in (("stock",) if fixture else MODES):
                    opening = episode(sim, motor, row, noise, mode, case, opening, fixture)
                seconds = time.monotonic() - started
                bind(case / "complete.json", dict(case_id=index, layout_seed=row["layout_seed"], noise_seed=row["noise_seed"],
                     scene_hash=row["scene_hash"], movable_xy_hash=row["movable_xy_hash"], seconds=seconds,
                     receipts={m: sha(case / f"{m}.json") for m in (("stock",) if fixture else MODES)}))
                print("sealed case", index, "seconds", round(seconds, 2), flush=True)
    finally:
        after = frozen if fixture else motor.digests()
        bind(out / "after.json", after)
        require(frozen == after, "frozen weights changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha")
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fit-sha")
    parser.add_argument("--fit-pin")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--shard", type=int)
    run(parser.parse_args())
