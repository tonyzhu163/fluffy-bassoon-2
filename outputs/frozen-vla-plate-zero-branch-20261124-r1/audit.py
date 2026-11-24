"""Independently read native HDF5, verify the intervention, and score complete looks."""
import argparse
from collections import Counter
from itertools import combinations
import gzip
import hashlib
import os
from pathlib import Path
import statistics

import h5py
import numpy as np

from analyze import check_manifest, check_sources, cp, exact_p, read_endpoint, spend_at
from large_panel_protocol import digest, mobile_joints, read, require, sha
from readback import episode, numerical_state_fingerprint, publish_report

MODES = ("stock", "mixed_on", "mixed_zero")
ZERO_SHA = hashlib.sha256(str(("float32", (1, 10, 32))).encode() + b"\0" + bytes(1 * 10 * 32 * 4)).hexdigest()
DATA_SHA = "3e133b69d850318cb20d61549e2f96ce6a7da8548cf05d25af7401ec28047c78"
FIT_SHA = "7d845ce26bbb7f7f5da45b4885660023dd0e33b622cb9913547d50290ca16fe2"
HEAD_SHA = "86a26caf81832e02f8dc6f823904eb8214fc2b529ff9f652b983340cf6080d54"
STOCK_SHA = "8c163ce2d7842d52c2ef01070b51b75f0409a99627df70d6c0d19ae33fd4489e"
PRIOR_SHA = "cc5d8aae55c5346ab36adde51abcdb44a19a89954e2ec18201fd1732f149b824"
PREPARED_SHA = "b7996a9e110c9ceed9babb081a18476c1dc0b9a8d7e623703f5912031e44bc3f"
OLD_LOOK_SHA = "c37059501e6fe4eaffba5b9e8d4ebdb8ca5f07d6e5f85b0e481e562022aa134d"


def enrollment(manifest):
    require(sha(manifest) == PREPARED_SHA and sha(manifest.with_name("prior.json")) == PRIOR_SHA, "enrollment pins changed")
    doc = check_manifest(manifest)
    old = manifest.parents[3]
    source = old / "data/dataset-v1/seal.json"
    fit = old / "fit/gpu03"
    require(sha(fit / "fit.json") == FIT_SHA and sha(fit / "mixed/head.npz") == HEAD_SHA, "fit binding")
    check_sources(source, DATA_SHA, {manifest: doc})
    source_seal = read(source)
    selection = old / "data/gpu01/selection.json"
    require(sha(selection) == source_seal["selection_sha256"], "source roster binding")
    candidates = read(selection)["candidates"]
    require(len(candidates) == 253, "eligible source roster changed")
    old_look = old / "eval/gpu01s3/audit/look-64.json"
    require(sha(old_look) == OLD_LOOK_SHA, "old enrollment seal changed")
    # Extract membership only from the previously published readback.
    old_ids = [r["case_id"] for r in read(old_look)["rows"]]
    require(old_ids == list(range(64)), "old study coverage changed")
    rows = doc["cases"]
    old_rows = [rows[i] for i in old_ids]
    overlaps = {}
    for key in ("initial_state_sha256", "scene_hash", "movable_xy_hash", "mobile_joint_xy_hash", "layout_seed", "noise_seed"):
        fresh = {row[key] for row in rows[64:]}
        earlier = {row[key] for row in old_rows}
        require(len(fresh) == len(rows) - 64 and fresh.isdisjoint(earlier), f"old target overlap at {key}")
        source_key = "generated_row_fingerprint" if key == "initial_state_sha256" else key
        overlaps[key] = dict(old_target=len(fresh & earlier),
                             source_candidates=len(fresh & {r.get(source_key) for r in candidates}))
        require(not overlaps[key]["source_candidates"], f"source candidate overlap at {key}")
    for row in rows:
        require(numerical_state_fingerprint(row["donor_state"]) == row["initial_state_sha256"]
                and digest(row["sampled_scene"]) == row["scene_hash"]
                and mobile_joints(row["sampled_scene"]) == row["mobile_joint_xy_hash"], "raw enrollment geometry")
    return rows, dict(manifest_sha256=PREPARED_SHA, prior_sha256=PRIOR_SHA, old_membership_seal_sha256=OLD_LOOK_SHA,
                      preflight_used_target_outcomes=False, old_cases=old_ids, first_look_cases=list(range(64, 128)),
                      all_eligible_source_candidates=len(candidates), fitted_source_layouts=len(source_seal["layouts"]),
                      source_seal_sha256=DATA_SHA, fit_sha256=FIT_SHA, head_sha256=HEAD_SHA, overlaps=overlaps,
                      host=os.uname().nodename, affinity=sorted(os.sched_getaffinity(0)))


def lane_context(attempt, *, live=False):
    gpu = read(attempt / "gpu.json")
    lines = (attempt / "launch.txt").read_text().splitlines()
    pin, host, _, name, lane, _, _, _ = lines
    require(gpu["status"] == "passed" and gpu["purpose"] == "evaluation" and
            gpu["allocation_type"] == "baige_container" and host == gpu["hostname"] and
            len(gpu["devices"]) == 1 and gpu["devices"][0]["cuda_name"].upper() == "NVIDIA GEFORCE RTX 4090" and
            name == attempt.name and pin == lines[2], "scheduler or GPU provenance")
    shard = attempt / f"shard{lane}"
    context = (shard / "run.txt").read_text().splitlines()
    require(context[:8] == [pin, "796acc7fa4ff8fb536eaa705a4e9d83c86cda0c3",
            "7f749ae21b8d74ad58536c00ae4aafd5ffca45c5", FIT_SHA, PREPARED_SHA, lines[5], lines[6], lane] and
            context[8].isdecimal(), "shard metadata changed")
    initial = read(shard / "before.json")
    require(initial["resident_stock"] == initial["deployed_stock"] == STOCK_SHA and
            initial["files"]["fit.json"] == FIT_SHA and initial["files"]["mixed/head.npz"] == HEAD_SHA and
            initial["resident_heads"]["mixed"] == read(attempt.parent.parent /
                "frozen-vla-plate-transfer-20261124-r1/fit/gpu03/mixed/fit.json")["parameters"], "frozen stock or mixed head mismatch")
    seconds = None
    if not live:
        rc, start, finish = map(int, (attempt / "exit.txt").read_text().split())
        require(rc == 0 and initial == read(shard / "after.json"), "incomplete attempt or changed parameters")
        seconds = finish - start
    graphs = {}
    for graph, name in (("stock_native", "stock"), ("history", "history")):
        path = shard / f"{name}.hlo.gz"
        if path.is_file():
            with gzip.open(path, "rb") as h:
                text = h.read()
            require(text.startswith(b"HloModule ") and b"ENTRY " in text, "invalid HLO archive")
            graphs[graph] = hashlib.sha256(text).hexdigest()
    require("stock_native" in graphs and ("history" not in graphs or graphs["history"] != graphs["stock_native"]),
            "sampler HLO binding")
    return shard, dict(pin=pin, job=host.removesuffix("-master-0"), lane=int(lane), graphs=graphs, seconds=seconds,
                       first=int(lines[5]), stop=int(lines[6]), gpu_sha256=sha(attempt / "gpu.json"),
                       frozen_sha256=sha(shard / "before.json"))


def verify_case(path, row, graphs):
    complete = read(path / "complete.json")
    keys = ("layout_seed", "noise_seed", "movable_xy_hash", "scene_hash")
    require(complete["case_id"] == row["index"] and all(complete[k] == row[k] for k in keys) and
            set(complete["receipts"]) == set(MODES), "case membership and arm coverage")
    tapes, motion, packets, calls = {}, {}, {}, Counter()
    for mode in MODES:
        receipt = path / f"{mode}.json"
        require(sha(receipt) == complete["receipts"][mode], "modified rollout receipt")
        # Reuse qualified mechanics audit, then apply the new study's strict route and renderer checks.
        tape = tapes[mode] = episode(path, mode, row, fixture=True)
        packet = packets[mode] = read(receipt)
        info = tape["reset"]["info"]
        physical = info["environment_reproducibility"]
        require(info["renderer"][0] == "NVIDIA Corporation" and
                physical["generated_row_fingerprint"] == row["initial_state_sha256"] and
                all(physical[k] == row[k] for k in ("source_pin", "bddl_sha256", "manifest_sha256")) and
                info["model_identity"]["gripper"] == "PandaGripper", "renderer, donor, asset or release mismatch")
        require(packet["diagnostic"] is None, "unregistered unissued diagnostic")
        for call in packet["calls"]:
            r = call["route"]
            graph = "stock_native" if mode == "stock" or call["t"] == 0 else "history"
            treatment = {"stock": "none", "mixed_on": "learned", "mixed_zero": "zero"}[mode] if graph == "history" else "none"
            require(r["condition"] == mode and r["graph"] == graph and r["hlo_sha256"] == graphs[graph] and
                    r["intervention"] == treatment, "wrong graph or intervention at a replan")
            if graph == "stock_native":
                require(r["offset_sha256"] is None, "stock received a residual")
            else:
                require(r["offset_dtype"] == "float32" and r["offset_shape"] == [1, 10, 32] and
                        len(bytes.fromhex(r["offset_sha256"])) == 32 and
                        (mode != "mixed_zero" or r["offset_sha256"] == ZERO_SHA), "invalid dynamic offset")
                calls[f"{mode}_nonzero"] += int(r["offset_sha256"] != ZERO_SHA)
            calls[f"{mode}_{graph}"] += 1
        with h5py.File(path / f"{mode}.h5", "r") as h:
            require(read_endpoint(h) == tape["endpoint"], "raw endpoint mismatch")
            xyz = h["robot0_eef_pos"][:].astype(np.float64)
            motion[mode] = dict(controls=tape["n"], displacement_m=float(np.linalg.norm(xyz[-1] - xyz[0])),
                                path_length_m=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                                clipped_fraction=float((np.abs(tape["actions"]) == 1).mean()))
    prefixes = {}
    t5_differences = {}
    for left, right in combinations(MODES, 2):
        a, b = tapes[left], tapes[right]
        prefix = 0
        while prefix < min(a["n"], b["n"]) and np.array_equal(a["actions"][prefix], b["actions"][prefix]):
            prefix += 1
        require(a["reset"] == b["reset"] and a["opening"] == b["opening"] and
                prefix >= min(5, a["n"], b["n"]) and a["states"][:prefix+1] == b["states"][:prefix+1],
                f"reset, opening or exact physical input pairing failed at {row['index']}")
        pair = f"{left}_vs_{right}"
        prefixes[pair] = prefix
        if min(a["n"], b["n"]) > 5:
            x, y = packets[left]["calls"][1], packets[right]["calls"][1]
            require(x["observation"] == y["observation"] and x["noise"] == y["noise"] and
                    x["history"] == y["history"], "first post-opening call not paired")
            t5_differences[pair] = float(np.max(np.abs(np.asarray(x["proposed"]) - y["proposed"])))
    return dict(case_id=row["index"], endpoints={m: t["endpoint"] for m, t in tapes.items()},
                tape_sha256={m: t["sha256"] for m, t in tapes.items()}, complete_sha256=sha(path / "complete.json"),
                bytes={m: (path / f"{m}.h5").stat().st_size for m in MODES}, calls=dict(calls), motion=motion,
                paired_identical_controls=prefixes, t5_proposal_max_abs=t5_differences)


def comparison(cases, first, second, component_alpha):
    cells = Counter((r["endpoints"][first], r["endpoints"][second]) for r in cases)
    n, b, c = len(cases), cells[1, 0], cells[0, 1]
    q = cp(b + c, n, component_alpha)
    s = cp(b, b + c, component_alpha)
    bounds = [rate * (2*sign - 1) for rate in q for sign in s]
    return dict(both_succeed=cells[1, 1], both_fail=cells[0, 0], first_only=b, second_only=c,
                paired_risk_difference=(b-c)/n, interval=[min(bounds), max(bounds)])


def score(cases):
    n = len(cases)
    alpha = spend_at(n)
    primary = comparison(cases, "mixed_zero", "mixed_on", float(alpha)/2)
    b, c = primary["first_only"], primary["second_only"]
    p = exact_p(b, c)
    decision = ("benefit" if b > c else "harm") if p <= alpha else "continue"
    successes = {m: sum(r["endpoints"][m] for r in cases) for m in MODES}
    return dict(n=n, decision=decision, alpha_increment=float(alpha), exact_p_two_sided=float(p),
                exact_p_fraction=str(p), primary=primary, successes=successes,
                descriptive_marginal_95={m: cp(k, n, .05) for m, k in successes.items()},
                descriptive_difference_95={m: comparison(cases, m, "stock", .025) for m in MODES[1:]},
                motion_medians={m: {key: statistics.median(r["motion"][m][key] for r in cases)
                                    for key in cases[0]["motion"][m]} for m in MODES})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--attempt", type=Path, action="append", default=[])
    p.add_argument("--n", type=int)
    p.add_argument("--first-native", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    require(os.uname().sysname == "Linux" and len(os.sched_getaffinity(0)) <= 2, "Linux readback must use at most two cores")
    rows, inputs = enrollment(args.manifest)
    if not args.attempt:
        publish_report(args.out, inputs)
        return
    contexts, covered = {}, {}
    for attempt in args.attempt:
        shard, context = lane_context(attempt, live=args.first_native)
        contexts[attempt.name] = context
        for path in sorted(shard.glob("case-*")):
            if not (path / "complete.json").is_file():
                continue
            i = int(path.name.removeprefix("case-"))
            require(context["first"] <= i < context["stop"] and i % 4 == context["lane"] and i not in covered,
                    "unexpected or duplicate case")
            covered[i] = path, context["graphs"]
    if args.first_native:
        i = min(covered)
        case, graphs = covered[i]
        verified = verify_case(case, rows[i], graphs)
        del verified["endpoints"]  # Qualification only, no outcome report or interim look.
        publish_report(args.out, dict(case=verified, allocations=contexts, enrollment=inputs))
        return
    spend_at(args.n)
    require(set(covered) == set(range(64, 64 + args.n)), "incomplete registered prefix")
    cases = [verify_case(covered[i][0], rows[i], covered[i][1]) for i in sorted(covered)]
    report = args.out / f"readback-{64 + args.n}.json"
    publish_report(report, dict(enrollment=inputs, allocations=contexts, cases=cases))
    look = args.out / f"look-{64 + args.n}.json"
    publish_report(look, dict(readback_sha256=sha(report), **score(cases)))
    print("SEALED", look, sha(look), flush=True)


if __name__ == "__main__":
    main()
