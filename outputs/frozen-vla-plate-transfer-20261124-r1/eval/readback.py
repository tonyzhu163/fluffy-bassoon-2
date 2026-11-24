"""Read complete native tapes independently on two CPU cores before releasing any look."""
import argparse
import os
from pathlib import Path
import statistics
import runpy

import h5py
import numpy as np

from large_panel_protocol import bind, digest, mobile_joints, read, require, sha
from native import CAMERAS, PROPRIO, PROMPT
from panel import MODES, bank
from plan import seed
from source_data import PublicHistory
from ur5e_motor import array_digest, public_digest, to_openpi_observation



# Read the release's pure fingerprint function without initializing its remote HTTP client.
numerical_state_fingerprint = runpy.run_path(str(Path(os.environ["CROSSARM_RELEASE"]) /
    "libero_remote/reproducibility.py"))["numerical_state_fingerprint"]


def episode(case, mode, expected, fixture):
    packet = read(case / f"{mode}.json")
    h5_path = case / f"{mode}.h5"
    require(packet["mode"] == mode and packet["case_id"] == expected["index"] and sha(h5_path) == packet["tape_sha256"],
            "trajectory/receipt binding failed")
    noise = bank(expected)
    info = packet["reset"]["info"]
    phys = info["environment_reproducibility"]
    body = info["model_identity"]
    require(packet["noise_bank"] == array_digest(noise) and phys["scene_fingerprint"] == expected["scene_hash"]
            and phys["seed"] == expected["layout_seed"] and phys["case_id"] == expected["index"]
            and phys["perturbation_plan"]["case_id"] == "BASE" and phys["suite"] == "libero_goal" and phys["task_id"] == 5
            and (body["model_class"], body["control_freq"], body["controller"], info["max_steps"]) ==
            ("MountedKinova3", 20, "OSC_POSE", 300), "task, placement, noise or control identity mismatch")
    if not fixture:
        require(info["renderer"][0] == "NVIDIA Corporation" and
                phys["generated_row_fingerprint"] == numerical_state_fingerprint(expected["donor_state"]),
                "donor or renderer mismatch")
    calls = packet["calls"]
    with h5py.File(h5_path, "r") as f:
        n = len(f["reward"])
        require(0 < n <= 300 and n == packet["controls"] and len(calls) == (n + 4) // 5, "invalid native length")
        actions, rewards = f["issued_action"][:], f["reward"][:]
        success, terminated, truncated = (f[k][:] for k in ("success", "terminated", "truncated"))
        require(actions.shape == (n, 7) and np.isfinite(actions).all() and (abs(actions) <= 1).all() and
                np.array_equal(f["worker_step_index"][:], np.arange(1, n + 1)), "native control sequence")
        require(all(a.shape == (n,) and not np.any(a[:-1]) for a in (success, terminated, truncated, rewards)) and
                rewards[-1] in (0, 1) and success[-1] == (rewards[-1] == 1) and
                bool(terminated[-1]) != bool(truncated[-1]) and
                ((success[-1] and terminated[-1]) or (not success[-1] and n == 300)), "malformed native endpoint")
        proprio = {k: f[k][:] for k in PROPRIO}
        require(all(v.shape == (n+1, width) and np.isfinite(v).all()
                    for v, width in zip(proprio.values(), (3, 4, 2, 7, 7), strict=True)), "invalid proprioception")
        physics = f["physics"][:]
        require(physics.shape[0] == n+1 and np.isfinite(physics).all() and
                numerical_state_fingerprint(physics[0]) == phys["initial_state_fingerprint"], "physics trace mismatch")
        require(all(f[k].shape == (n+1, 256, 256, 3) and f[k].dtype == np.uint8 for k in CAMERAS),
                "camera coverage or encoded dtype")
        frames = {}
        h = PublicHistory()
        states, history_graphs = [], set()
        for t in range(n+1):
            for k in CAMERAS:
                delta = f[k][t]
                frames[k] = delta if t == 0 else np.bitwise_xor(frames[k], delta)
            raw = {**frames, **{k: v[t] for k, v in proprio.items()}}
            states.append((array_digest(physics[t]), *(array_digest(v) for v in raw.values())))
            if t == 0:
                require(all(packet["reset"]["public"][k] == array_digest(v) for k,v in raw.items()),
                        "opening not reproduced from stored trajectories")
            if t < n and t % 5 == 0:
                call = calls[t//5]
                length = min(n-t, 5)
                item = to_openpi_observation(raw, PROMPT)
                proposed = np.array(call["proposed"])
                require(proposed.shape == (10, 7) and np.isfinite(proposed).all() and
                        call["t"] == t and call["issued_count"] == length and call["noise"] == array_digest(noise[t//5])
                        and call["observation"] == public_digest({k:v for k,v in item.items() if k != "prompt"})
                        and np.array_equal(np.asarray(call["history"], np.float32), h.vector()) and
                        np.array_equal(np.clip(proposed[:length], -1, 1).astype(np.float32), actions[t:t+length]),
                        f"model inputs, history or commands changed at {t}")
                route = call["route"]
                if not fixture:
                    require(route["condition"] == mode and route["graph"] == (
                        "stock_native" if mode == "stock" else "plain" if t == 0 else "history"), "issued program")
                    if mode != "stock" and t:
                        require(isinstance(route["hlo_sha256"], str) and len(route["hlo_sha256"]) == 64, "missing HLO")
                        history_graphs.add(route["hlo_sha256"])
            if t < n:
                h.append(actions[t], raw, {k:v[t+1] for k,v in proprio.items()})
    if mode == "mixed" and n > 5 and not fixture:
        d = packet["diagnostic"]
        require(d["t"] == 5 and d["learned"] == calls[1]["proposed"] and
                d["zero_route"]["hlo_sha256"] == calls[1]["route"]["hlo_sha256"] and
                d["zero_route"]["condition"] == "stock", "unissued counterfactual mismatch")
    return dict(endpoint=int(rewards[-1]), n=n, states=states, actions=actions, opening=calls[0]["proposed"],
                reset=packet["reset"], graphs=history_graphs, sha256=packet["tape_sha256"])


def verify_case(path, row, fixture=False):
    complete = read(path / "complete.json")
    modes = ("stock",) if fixture else MODES
    require(complete["case_id"] == row["index"] and all(complete[k] == row[k] for k in
            ("layout_seed", "noise_seed", "movable_xy_hash", "scene_hash")) and set(complete["receipts"]) == set(modes),
            "case manifest membership")
    tapes = {}
    for m in modes:
        require(sha(path / f"{m}.json") == complete["receipts"][m], "case receipt changed")
        tapes[m] = episode(path, m, row, fixture)
        for prev in list(tapes)[:-1]:
            a, b = tapes[prev], tapes[m]
            require(a["reset"] == b["reset"] and a["opening"] == b["opening"], "reset/opening unpaired")
            prefix = 0
            while prefix < min(a["n"], b["n"]) and np.array_equal(a["actions"][prefix], b["actions"][prefix]):
                prefix += 1
            require(prefix >= min(5, a["n"], b["n"]) and a["states"][:prefix+1] == b["states"][:prefix+1],
                    f"observations/physics diverged under identical issued prefix at {row['index']}")
    graphs = set().union(*(v["graphs"] for v in tapes.values()))
    require(len(graphs) == int(not fixture and any(tapes[m]["n"] > 5 for m in ("panda", "mixed"))),
            "trained branches did not share one compiled executable")
    result = {k: complete[k] for k in ("case_id", "layout_seed", "noise_seed", "movable_xy_hash", "scene_hash")}
    result.update({m: p["endpoint"] for m,p in tapes.items()})
    audit = dict(case_id=row["index"], seconds=complete["seconds"], complete_sha256=sha(path / "complete.json"),
                 trajectory_sha256={m:p["sha256"] for m,p in tapes.items()}, controls={m:p["n"] for m,p in tapes.items()},
                 compiled_history=list(graphs), pfs=str(path.resolve()))
    return result, audit


def main(args):
    require(len(os.sched_getaffinity(0)) <= 2, "readback requires bounded CPU")
    if args.fixture:
        path, = args.attempt
        complete = read(path / "case--0001/complete.json")
        row = dict(complete, index=-1)
        endpoints, audit = verify_case(path / "case--0001", row, fixture=True)
        bind(args.out / "cpu-fixture.json", dict(fixture=True, native=endpoints, **audit,
             exit=(path / "exit.txt").read_text(), launch=(path / "launch.txt").read_text()))
        return
    doc = read(args.manifest)
    rows = doc["cases"][:args.n]
    require(len(rows) == args.n and args.n >= 64 and args.n & (args.n-1) == 0, "unscheduled look")
    priors = read(args.manifest.with_name("prior.json"))
    require(sha(args.manifest.with_name("prior.json")) == doc["prior_sha256"], "prior inventory changed")
    for k in ("initial_state_sha256", "movable_xy_hash", "scene_hash", "mobile_joint_xy_hash", "layout_seed", "noise_seed"):
        values = {r[k] for r in rows}
        require(len(values) == args.n and values.isdisjoint(priors["fingerprints"].get(k, [])), f"placement overlap: {k}")
    contexts, cases, frozen = {}, {}, None
    for attempt in args.attempt:
        exit_record = (attempt / "exit.txt").read_text().split()
        require(exit_record[0] == "0", "incomplete allocation")
        probe = read(attempt / "gpu.json")
        require(probe["status"] == "passed" and all("4090" in d["cuda_name"] for d in probe["devices"]), "GPU readback")
        for shard in sorted(attempt.glob("shard[0-3]")):
            info = (shard / "run.txt").read_text().splitlines()
            pin, fit_pin, fit_sha, manifest = info[:4]
            start, stop, rank = map(int, info[4:7])
            require(manifest == sha(args.manifest) or (args.prior and manifest in args.prior), "unknown physical manifest")
            before = read(shard / "before.json")
            require(before == read(shard / "after.json") and before["files"]["fit.json"] == fit_sha
                    and before["resident_stock"] == before["deployed_stock"] and before["deployed_kernel"] and before["resident_heads"],
                    "weight drift or missing fit identity")
            if frozen is None:
                frozen = before
            require(frozen == before, "weights vary across shards or cohorts")
            contexts[str(shard)] = dict(pin=pin, fit_pin=fit_pin, fit_sha=fit_sha, probe=sha(attempt/"gpu.json"),
                                        exit=exit_record, frozen_before_sha=sha(shard/"before.json"))
            for i in range(start+rank, stop, 4):
                require(i not in cases, "duplicate evaluated case")
                cases[i] = shard / f"case-{i:05}"
    require(set(cases) == set(range(args.n)), "scheduled cumulative case/shard set incomplete")
    all_rows, audits = [], []
    for i, row in enumerate(rows):
        require(row["index"] == i and row["layout_seed"] == seed("layout", i, row["draw"])
                and row["noise_seed"] == seed("noise", i) and row["shard"] == i % 4 and
                numerical_state_fingerprint(row["donor_state"]) == row["initial_state_sha256"] and
                digest(row["sampled_scene"]) == row["scene_hash"] and mobile_joints(row["sampled_scene"]) == row["mobile_joint_xy_hash"],
                "physical parameters or ordered draw mismatch")
        result, audit = verify_case(cases[i], row)
        all_rows.append(result)
        audits.append(audit)
    report = args.out / f"readback-{args.n}.json"
    bind(report, dict(n=args.n, frozen=frozen, allocations=contexts, audits=audits,
                      median_seconds_per_case=statistics.median(r["seconds"] for r in audits)))
    seal = args.out / f"look-{args.n}.json"
    bind(seal, dict(n=args.n, manifest_sha256=sha(args.manifest), native_readback=report.name,
                    native_readback_sha256=sha(report), rows=all_rows))
    print("SEALED", seal, sha(seal), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--attempt", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--n", type=int)
    p.add_argument("--prior", action="append", help="A previously verified sealed prefix manifest SHA for an extension")
    main(p.parse_args())
