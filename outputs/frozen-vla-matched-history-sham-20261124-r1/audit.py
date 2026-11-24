"""Offline readback from native reward, applied action, fitted-array and executable receipts."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import matched
import verify
from panel import key, require
from ur5e_motor import array_digest

MODES = ("real", "sham", "plain")
ZERO = array_digest(np.zeros((1, 10, 32), np.float32))
FIT = "4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route(r, mode, graphs, kernel):
    graph = "plain" if mode == "plain" else "history"
    require(r["kernel"] == kernel and r["graph"] == graph and r["executable"] == graphs[graph], "route identity")
    expected = {"real": r["computed_offset"], "sham": ZERO, "plain": None}[mode]
    require(r["passed_offset"] == expected, "runtime argument")


def probe(v, graphs, kernel):
    t = v["calls"][2]
    m = v["mechanism"]
    require(t["t"] == m["at_control"] == 10 and m["obs"] == t["obs"] and m["noise"] == t["noise"]
            and m["prompt"] == v["native"]["reset"]["prompt"] and m["history"] == t["used"]
            and m["rng"] == [0, 0], "common probe input")
    a = {}
    for name, p in m["probes"].items():
        mode = "sham" if name == "sham_replay" else name
        route(p["route"], mode, graphs, kernel)
        a[name] = np.asarray(p["actions"], np.float32)
        require(p["route"]["computed_offset"] == t["route"]["computed_offset"] and
                a[name].shape == (10, 7) and np.isfinite(a[name]).all() and
                array_digest(a[name]) == p["proposal"], "unissued sample")
    require(m["probes"]["real"]["proposal"] == t["proposal"] and
            m["probes"]["sham"]["proposal"] == m["probes"]["sham_replay"]["proposal"], "repeatability")
    effects = {}
    for label, left, right in (("content", "real", "sham"), ("arithmetic", "sham", "plain")):
        diff = np.abs(a[left] - a[right])
        effects[label] = dict(max_first_five=float(diff[:5].max()), max_chunk=float(diff.max()))
    return effects


def audit(out):
    reg, cases, tasks = matched.contract()
    meta = read(out/"run.json")
    fixture = meta["fixture"]
    require(meta["contract"] == digest(matched.HERE/"contract.json")
            == "063c9640096eeabdeb560b2e8dd4bddac71c3250ed4b5636922d398d562d2999", "enrollment manifest")
    before = read(out/"frozen-before.json")
    require(before == read(out/"frozen-after.json") and before["disk_fit"] == FIT, "frozen readback")
    fit = Path(meta["fit"])
    report = read(fit/"fit.json")
    require(digest(fit/"fit.json") == FIT and digest(fit/"heads.npz") == before["disk_heads"] ==
            report["heads_sha256"] and before["resident_heads"]["joint"] == report["parameters"]["joint"]["fitted"],
            "published joint head and archive")
    kernel = before["resident_effective_kernels"]["joint"]
    if not fixture:
        require(kernel == report["parameters"]["joint"]["action_out_proj/kernel"] and before["resident_stock"]
                and before["normalization"] == report["base"]["normalization"] and
                before["checkpoint_manifest"] == report["base"]["checkpoint_manifest"], "stock and projection")
    graphs = read(out/"graphs.json")
    require(len({v["process_object"] for v in graphs.values()}) == 2, "two compiled objects")
    rows = []
    scenes = {}
    expected = {f"{key(c)}-{m}.json" for c in cases for m in MODES}
    require(len(expected) == 24 and {p.name for p in (out/"episodes").glob("*.json")} == expected, "24 episodes")
    for case in cases:
        tapes, outcomes, opening, diagnostic = {}, {}, None, None
        for mode in MODES:
            p = out/"episodes"/f"{key(case)}-{mode}.json"
            v = read(p)
            require(v["case"] == case and v["mode"] == mode, "case join")
            with patch.object(verify, "TASKS", tasks):
                ep = verify.validate(v, fixture)
            calls = v["calls"]
            for call in calls:
                route(call["route"], mode if call["t"] else "plain", graphs, kernel)
            first = calls[0]
            require(first["route"]["computed_offset"] == ZERO and not np.any(first["history"]), "opening offset")
            env, reset = v["environment"], v["native"]["reset"]
            signature = [reset, env, v["body"], first]
            if opening is None:
                opening = signature
            require(opening == signature, "identical opening including commands and executed transitions")
            scene = case["suite"], case["init_state_id"]
            common = [env[k] for k in ("scene_fingerprint", "bddl_sha256", "released_row_fingerprint",
                                        "perturbation_plan_fingerprint")] + [
                reset[k] for k in ("prompt", "noise_sha256", "horizon", "action_schema")]
            require(scenes.setdefault(scene, common) == common, "shared scene and bank across robots")
            if len(calls) > 1:
                control5 = [calls[1][k] for k in ("obs", "noise", "history")] + [calls[1]["route"]["computed_offset"]]
                if mode == "real":
                    post_open = control5
                require(post_open == control5, "control 5 shared input and computed offset")
            if mode == "real" and len(calls) > 2:
                diagnostic = probe(v, graphs, kernel)
            outcomes[mode] = dict(**ep, terminated=v["native"]["terminated"], truncated=v["native"]["truncated"],
                                  file_sha256=digest(p), replans=len(calls))
            tapes[mode] = calls
        differences = {}
        for name, a, b in (("content", "real", "sham"), ("arithmetic", "sham", "plain")):
            differences[name] = dict(delta=int(outcomes[a]["success"])-int(outcomes[b]["success"]),
                                     **verify.first_change(tapes[a], tapes[b]))
        rows.append(dict(case=case, endpoints=outcomes, differences=differences, unissued=diagnostic))
    require(len(rows) == 8 and len(scenes) == 4, "complete enrollment")
    return dict(fixture=fixture, cases=8, episodes=24, complete_openings=8, shared_starts=4,
                control10_cases=sum(r["unissued"] is not None for r in rows), kernel=kernel,
                fit=FIT, graphs=graphs, rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    result = audit(a.out)
    require(result == read(a.out/"paired.json"), "independent readback differs from recorded pairs")
    print(f"Verified {result['episodes']} endpoints on {result['cases']} cases; control 10 on {result['control10_cases']}.")
